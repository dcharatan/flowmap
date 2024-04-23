from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from jaxtyping import Float
from torch import Tensor, nn

from ...dataset.types import Batch
from ...flow.flow_predictor import Flows
from ..backbone.backbone import BackboneOutput

T = TypeVar("T")


class Extrinsics(nn.Module, ABC, Generic[T]):
    cfg: T
    num_frames: int | None

    def __init__(self, cfg: T, num_frames: int | None) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_frames = num_frames

    @abstractmethod
    def forward(
        self,
        batch: Batch,
        flows: Flows,
        backbone_output: BackboneOutput,
        surfaces: Float[Tensor, "batch frame height width 3"],
    ) -> Float[Tensor, "batch frame 4 4"]:
        pass
