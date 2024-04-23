from dataclasses import dataclass
from typing import Literal

import torch
from einops import repeat
from jaxtyping import Float
from torch import Tensor, nn

from ...dataset.types import Batch
from ...flow.flow_predictor import Flows
from ..backbone.backbone import BackboneOutput
from .common import focal_lengths_to_intrinsics
from .intrinsics import Intrinsics


@dataclass
class IntrinsicsRegressedCfg:
    name: Literal["regressed"]
    initial_focal_length: float


class IntrinsicsRegressed(Intrinsics[IntrinsicsRegressedCfg]):
    def __init__(self, cfg: IntrinsicsRegressedCfg) -> None:
        super().__init__(cfg)
        focal_length = torch.full(
            tuple(),
            cfg.initial_focal_length,
            dtype=torch.float32,
        )
        self.focal_length = nn.Parameter(focal_length)

    def forward(
        self,
        batch: Batch,
        flows: Flows,
        backbone_output: BackboneOutput,
        global_step: int,
    ) -> Float[Tensor, "batch frame 3 3"]:
        b, f, _, h, w = batch.videos.shape
        intrinsics = focal_lengths_to_intrinsics(self.focal_length, (h, w))
        return repeat(intrinsics, "i j -> b f i j", b=b, f=f)
