from dataclasses import dataclass
from typing import Literal

from jaxtyping import Float
from torch import Tensor

from ...dataset.types import Batch
from ...flow.flow_predictor import Flows
from ..backbone.backbone import BackboneOutput
from .intrinsics import Intrinsics


@dataclass
class IntrinsicsGroundTruthCfg:
    name: Literal["ground_truth"]


class IntrinsicsGroundTruth(Intrinsics[IntrinsicsGroundTruthCfg]):
    def forward(
        self,
        batch: Batch,
        flows: Flows,
        backbone_output: BackboneOutput,
        global_step: int,
    ) -> Float[Tensor, "batch frame 3 3"]:
        # Just return the ground-truth intrinsics.
        return batch.intrinsics
