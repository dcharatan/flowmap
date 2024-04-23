from dataclasses import dataclass
from typing import Literal

from jaxtyping import Float
from torch import Tensor

from .mapping import Mapping


@dataclass
class MappingL2Cfg:
    name: Literal["l2"]


class MappingL2(Mapping[MappingL2Cfg]):
    def forward_undistorted(
        self,
        delta: Float[Tensor, "*batch 2"],
    ) -> Float[Tensor, " *batch"]:
        # Multiply by 0.5 to match torch.nn.functional.huber_loss.
        return 0.5 * (delta * delta).sum(dim=-1)
