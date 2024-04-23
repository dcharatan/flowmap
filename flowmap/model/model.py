from dataclasses import dataclass

import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn

from ..dataset.types import Batch
from ..flow.flow_predictor import Flows
from .backbone import BackboneCfg, get_backbone
from .extrinsics import ExtrinsicsCfg, get_extrinsics
from .intrinsics import IntrinsicsCfg, get_intrinsics
from .projection import sample_image_grid, unproject


@dataclass
class ModelCfg:
    backbone: BackboneCfg
    intrinsics: IntrinsicsCfg
    extrinsics: ExtrinsicsCfg
    use_correspondence_weights: bool


@dataclass
class ModelOutput:
    depths: Float[Tensor, "batch frame height width"]
    surfaces: Float[Tensor, "batch frame height width xyz=3"]
    intrinsics: Float[Tensor, "batch frame 3 3"]
    extrinsics: Float[Tensor, "batch frame 4 4"]
    backward_correspondence_weights: Float[Tensor, "batch frame-1 height width"]


@dataclass
class ModelExports:
    extrinsics: Float[Tensor, "batch frame 4 4"]
    intrinsics: Float[Tensor, "batch frame 3 3"]
    colors: Float[Tensor, "batch frame 3 height width"]
    depths: Float[Tensor, "batch frame height width"]


class Model(nn.Module):
    def __init__(
        self,
        cfg: ModelCfg,
        num_frames: int | None = None,
        image_shape: tuple[int, int] | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.backbone = get_backbone(cfg.backbone, num_frames, image_shape)
        self.intrinsics = get_intrinsics(cfg.intrinsics)
        self.extrinsics = get_extrinsics(cfg.extrinsics, num_frames)

    def forward(
        self,
        batch: Batch,
        flows: Flows,
        global_step: int,
    ) -> ModelOutput:
        device = batch.videos.device
        _, _, _, h, w = batch.videos.shape

        # Run the backbone, which provides depths and correspondence weights.
        backbone_out = self.backbone.forward(batch, flows)

        # Allow the correspondence weights to be ignored as an ablation.
        if not self.cfg.use_correspondence_weights:
            backbone_out.weights = torch.ones_like(backbone_out.weights)

        # Compute the intrinsics.
        intrinsics = self.intrinsics.forward(batch, flows, backbone_out, global_step)

        # Use the intrinsics to calculate camera-space surfaces (point clouds).
        xy, _ = sample_image_grid((h, w), device=device)
        surfaces = unproject(
            xy,
            backbone_out.depths,
            rearrange(intrinsics, "b f i j -> b f () () i j"),
        )

        # Finally, compute the extrinsics.
        extrinsics = self.extrinsics.forward(batch, flows, backbone_out, surfaces)

        return ModelOutput(
            backbone_out.depths,
            surfaces,
            intrinsics,
            extrinsics,
            backbone_out.weights,
        )

    @torch.no_grad()
    def export(
        self,
        batch: Batch,
        flows: Flows,
        global_step: int,
    ) -> ModelExports:
        # For now, only implement exporting with a batch size of 1.
        b, _, _, _, _ = batch.videos.shape
        assert b == 1

        output = self.forward(batch, flows, global_step)

        return ModelExports(
            output.extrinsics,
            output.intrinsics,
            batch.videos,
            output.depths,
        )
