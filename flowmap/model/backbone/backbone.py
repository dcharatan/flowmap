from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

from jaxtyping import Float
from torch import Tensor, nn

from ...dataset.types import Batch
from ...flow.flow_predictor import Flows

T = TypeVar("T")


@dataclass
class BackboneOutput:
    depths: Float[Tensor, "batch frame height width"]
    weights: Float[Tensor, "batch frame-1 height width"]


class Backbone(nn.Module, ABC, Generic[T]):
    cfg: T

    def __init__(
        self,
        cfg: T,
        num_frames: int | None,
        image_shape: tuple[int, int] | None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_frames = num_frames
        self.image_shape = image_shape

    @abstractmethod
    def forward(self, batch: Batch, flows: Flows) -> BackboneOutput:
        pass
