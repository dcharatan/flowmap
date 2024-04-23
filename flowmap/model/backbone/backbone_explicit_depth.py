from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from ...dataset.types import Batch
from ...flow.flow_predictor import Flows
from .backbone import Backbone, BackboneOutput


@dataclass
class BackboneExplicitDepthCfg:
    name: Literal["explicit_depth"]
    initial_depth: float
    weight_sensitivity: float


class BackboneExplicitDepth(Backbone[BackboneExplicitDepthCfg]):
    def __init__(
        self,
        cfg: BackboneExplicitDepthCfg,
        num_frames: int | None,
        image_shape: tuple[int, int] | None,
    ) -> None:
        super().__init__(cfg, num_frames=num_frames, image_shape=image_shape)
        depth = torch.full(
            (num_frames, *image_shape), cfg.initial_depth, dtype=torch.float32
        )
        self.depth = nn.Parameter(depth)
        weights = torch.full((num_frames - 1, *image_shape), 0, dtype=torch.float32)
        self.weights = nn.Parameter(weights)

    def forward(self, batch: Batch, flows: Flows) -> BackboneOutput:
        b, _, _, _, _ = batch.videos.shape
        assert b == 1

        return BackboneOutput(
            self.depth[None],
            (self.cfg.weight_sensitivity * self.weights).sigmoid()[None],
        )
