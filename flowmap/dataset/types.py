from dataclasses import dataclass
from typing import Literal

from jaxtyping import Float, Int64
from torch import Tensor

from ..misc.manipulable import Manipulable

Stage = Literal["train", "test", "val"]


@dataclass
class Batch(Manipulable):
    videos: Float[Tensor, "batch frame 3 height=_ width=_"]
    indices: Int64[Tensor, "batch frame"]
    scenes: list[str]
    datasets: list[str]
    extrinsics: Float[Tensor, "batch frame 4 4"] | None = None
    intrinsics: Float[Tensor, "batch frame 3 3"] | None = None
