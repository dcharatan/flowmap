from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

from jaxtyping import Bool, Float
from torch import Tensor, nn

from ..misc.manipulable import Manipulable

T = TypeVar("T")


@dataclass
class Tracks(Manipulable):
    xy: Float[Tensor, "batch frame point 2"]
    visibility: Bool[Tensor, "batch frame point"]

    # This is the first frame in the track sequence, not the query frame used to
    # generate the sequence, which is often different.
    start_frame: int


class TrackPredictor(nn.Module, ABC, Generic[T]):
    def __init__(self, cfg: T) -> None:
        super().__init__()
        self.cfg = cfg

    @abstractmethod
    def forward(
        self,
        videos: Float[Tensor, "batch frame 3 height width"],
        query_frame: int,
    ) -> Tracks:
        pass
