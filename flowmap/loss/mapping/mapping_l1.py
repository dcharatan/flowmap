from dataclasses import dataclass
from typing import Literal

from jaxtyping import Float
from torch import Tensor

from .mapping import Mapping


@dataclass
class MappingL1Cfg:
    name: Literal["l1"]


class MappingL1(Mapping[MappingL1Cfg]):
    def forward_undistorted(
        self,
        delta: Float[Tensor, "*batch 2"],
    ) -> Float[Tensor, " *batch"]:
        return delta.norm(dim=-1)
