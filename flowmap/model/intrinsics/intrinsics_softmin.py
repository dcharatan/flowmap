from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from jaxtyping import Float
from torch import Tensor

from ...dataset.types import Batch
from ...flow.flow_predictor import Flows
from ..backbone.backbone import BackboneOutput
from ..projection import (
    align_surfaces,
    compute_backward_flow,
    sample_image_grid,
    unproject,
)
from .common import focal_lengths_to_intrinsics
from .intrinsics import Intrinsics
from .intrinsics_regressed import IntrinsicsRegressed, IntrinsicsRegressedCfg


@dataclass
class RegressionCfg:
    after_step: int
    window: int


@dataclass
class IntrinsicsSoftminCfg:
    name: Literal["softmin"]
    num_procrustes_points: int
    min_focal_length: float
    max_focal_length: float
    num_candidates: int
    regression: RegressionCfg | None


class IntrinsicsSoftmin(Intrinsics[IntrinsicsSoftminCfg]):
    focal_length_candidates: Float[Tensor, " candidate"]

    def __init__(self, cfg: IntrinsicsSoftminCfg) -> None:
        super().__init__(cfg)

        # Define the set of candidate focal lengths.
        focal_length_candidates = torch.linspace(
            cfg.min_focal_length,
            cfg.max_focal_length,
            cfg.num_candidates,
        )
        self.register_buffer(
            "focal_length_candidates",
            focal_length_candidates,
            persistent=False,
        )

        if cfg.regression is not None:
            intrinsics_regressed_cfg = IntrinsicsRegressedCfg("regressed", 0.0)
            self.intrinsics_regressed = IntrinsicsRegressed(intrinsics_regressed_cfg)
            self.window = []

    def forward(
        self,
        batch: Batch,
        flows: Flows,
        backbone_output: BackboneOutput,
        global_step: int,
    ) -> Float[Tensor, "batch frame 3 3"]:
        b, f, _, h, w = batch.videos.shape
        n = self.cfg.num_candidates
        device = batch.videos.device

        # Handle the second stage (in which the intrinsics are regressed).
        if (
            self.cfg.regression is not None
            and global_step >= self.cfg.regression.after_step
        ):
            if global_step == self.cfg.regression.after_step:
                initial_value = torch.stack(self.window).mean()
                self.intrinsics_regressed.focal_length.data = initial_value
            return self.intrinsics_regressed(batch, flows, backbone_output, global_step)

        # Convert the candidate focal lengths into 3x3 intrinsics matrices.
        candidate_intrinsics = focal_lengths_to_intrinsics(
            self.focal_length_candidates, (h, w)
        )

        # Align the first two frames with all possible intrinsics.
        indices = torch.randperm(h * w, device=device)[: self.cfg.num_procrustes_points]
        xy, _ = sample_image_grid((h, w), device=device)
        surfaces = unproject(
            xy,
            repeat(backbone_output.depths[:, :2], "b f h w -> (b n) f h w", n=n),
            repeat(candidate_intrinsics, "n i j -> (b n) f () () i j", b=b, f=2),
        )
        extrinsics = align_surfaces(
            surfaces,
            repeat(flows.backward[:, :1], "b f h w xy -> (b n) f h w xy", n=n),
            repeat(backbone_output.weights[:, :1], "b f h w -> (b n) f h w", n=n),
            indices,
        )

        # Compute pose-induced backward flow.
        xy_flowed_backward = compute_backward_flow(
            rearrange(surfaces, "bn f h w xyz -> bn f (h w) xyz")[:, :, indices],
            extrinsics,
            repeat(candidate_intrinsics, "n i j -> (b n) f i j", b=b, f=2),
        )
        xy_flowed_backward = rearrange(
            xy_flowed_backward, "(b n) () p xy -> b n p xy", b=b, n=n
        )
        xy, _ = sample_image_grid((h, w), device)
        xy = rearrange(xy, "h w xy -> (h w) xy")[indices]
        flow = xy_flowed_backward - xy

        # Sample from the ground-truth flow and backward correspondence weights.
        flow_gt = rearrange(flows.backward[:, :1], "b () h w xy -> b () (h w) xy")
        flow_gt = flow_gt[:, :, indices]
        weights = rearrange(backbone_output.weights[:, :1], "b () h w -> b () (h w) ()")
        weights = weights[:, :, indices]

        # Compute flow error for each of the candidate intrinsics.
        error = ((flow - flow_gt) * weights).abs()
        error = reduce(error, "b n p xy -> b n", "sum")

        # Compute a softmin-weighted sum of candidates.
        weights = (error - error.min(dim=1, keepdim=True).values) * 10
        weights = F.softmin(weights, dim=1)
        candidate_intrinsics = repeat(candidate_intrinsics, "n i j -> b n i j", b=b)
        intrinsics = (candidate_intrinsics * weights[:, :, None, None]).sum(dim=1)

        # Handle the window that's used to initialize the intrinsics.
        if self.cfg.regression is not None:
            start = self.cfg.regression.after_step - self.cfg.regression.window
            if global_step >= start and self.training:
                self.window.append(
                    (self.focal_length_candidates * weights).sum().detach()
                )

        return repeat(intrinsics, "b i j -> b f i j", f=f)

    def unnormalized_focal_lengths(
        self,
        image_shape: tuple[int, int],
    ) -> Float[Tensor, " candidate"]:
        focal_lengths = self.focal_length_candidates

        # Unnormalize the focal lengths based on the geometric mean of the image's side
        # lengths. This makes the candidate focal lengths invariant to the image's
        # aspect ratio.
        h, w = image_shape
        focal_lengths = focal_lengths * (h * w) ** 0.5

        return focal_lengths
