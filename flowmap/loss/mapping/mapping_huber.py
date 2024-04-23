from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from .mapping import Mapping


@dataclass
class MappingHuberCfg:
    name: Literal["huber"]
    delta: float


class MappingHuber(Mapping[MappingHuberCfg]):
    def forward_undistorted(
        self,
        delta: Float[Tensor, "*batch 2"],
    ) -> Float[Tensor, " *batch"]:
        norm = delta.norm(dim=-1)

        mapped = F.huber_loss(
            norm,
            torch.zeros_like(norm),
            reduction="none",
            delta=self.cfg.delta,
        )

        # Divide by the delta so that the gradient magnitude in the linear region
        # matches that of a regular L1 loss.
        return mapped / self.cfg.delta
