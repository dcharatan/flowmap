from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

import torch
from jaxtyping import Float
from torch import Tensor, nn

from ..dataset.types import Batch
from ..flow import Flows
from ..model.model import ModelOutput
from ..tracking import Tracks


@dataclass
class LossCfgCommon:
    enable_after: int
    weight: float


T = TypeVar("T", bound=LossCfgCommon)


class Loss(nn.Module, ABC, Generic[T]):
    cfg: T

    def __init__(self, cfg: T) -> None:
        super().__init__()
        self.cfg = cfg

    def forward(
        self,
        batch: Batch,
        flows: Flows,
        tracks: list[Tracks] | None,
        model_output: ModelOutput,
        global_step: int,
    ) -> Float[Tensor, ""]:
        # Before the loss is enabled, don't compute the loss.
        if global_step < self.cfg.enable_after:
            return torch.tensor(0, dtype=torch.float32, device=batch.videos.device)

        # Multiply the computed loss value by the weight.
        loss = self.compute_unweighted_loss(
            batch, flows, tracks, model_output, global_step
        )
        return self.cfg.weight * loss

    @abstractmethod
    def compute_unweighted_loss(
        self,
        batch: Batch,
        flows: Flows,
        tracks: list[Tracks] | None,
        model_output: ModelOutput,
        global_step: int,
    ) -> Float[Tensor, ""]:
        pass
