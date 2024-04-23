from dataclasses import dataclass
from typing import Literal

import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn

from ...dataset.types import Batch
from ...flow.flow_predictor import Flows
from ..backbone.backbone import BackboneOutput
from ..projection import get_extrinsics
from .extrinsics import Extrinsics


# https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py
def quaternion_to_matrix(
    quaternions: Float[Tensor, "*batch 4"],
    eps: float = 1e-8,
) -> Float[Tensor, "*batch 3 3"]:
    # Order changed to match scipy format!
    i, j, k, r = torch.unbind(quaternions, dim=-1)
    two_s = 2 / ((quaternions * quaternions).sum(dim=-1) + eps)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return rearrange(o, "... (i j) -> ... i j", i=3, j=3)


@dataclass
class ExtrinsicsRegressedCfg:
    name: Literal["regressed"]


class ExtrinsicsRegressed(Extrinsics[ExtrinsicsRegressedCfg]):
    def __init__(
        self,
        cfg: ExtrinsicsRegressedCfg,
        num_frames: int,
    ) -> None:
        super().__init__(cfg, num_frames)

        assert num_frames >= 2

        # Initialize identity translations and rotations.
        self.translations = nn.Parameter(
            torch.zeros((num_frames - 1, 3), dtype=torch.float32)
        )
        rotations = torch.zeros((num_frames - 1, 4), dtype=torch.float32)
        rotations[:, -1] = 1
        self.rotations = nn.Parameter(rotations)

    def forward(
        self,
        batch: Batch,
        flows: Flows,
        backbone_output: BackboneOutput,
        surfaces: Float[Tensor, "batch frame height width 3"],
    ) -> Float[Tensor, "batch frame 4 4"]:
        device = surfaces.device
        b, f, _, _, _ = surfaces.shape

        # Regressing the extrinsics only makes sense during overfitting.
        assert b == 1

        tf = torch.eye(4, dtype=torch.float32, device=device)
        tf = tf.broadcast_to((f - 1, 4, 4)).contiguous()
        tf[:, :3, :3] = quaternion_to_matrix(self.rotations)
        tf[:, :3, 3] = self.translations

        return get_extrinsics(tf)[None]
