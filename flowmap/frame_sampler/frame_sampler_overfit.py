from dataclasses import dataclass
from typing import Literal

import torch
from jaxtyping import Int64
from torch import Tensor

from .frame_sampler import FrameSampler


@dataclass
class FrameSamplerOverfitCfg:
    name: Literal["overfit"]
    start: int | None
    num_frames: int | None
    step: int | None


class FrameSamplerOverfit(FrameSampler[FrameSamplerOverfitCfg]):
    def sample(
        self,
        num_frames_in_video: int,
        device: torch.device,
    ) -> Int64[Tensor, " frame"]:
        start = self.cfg.start or 0
        num_frames = self.cfg.num_frames or num_frames_in_video
        step = self.cfg.step or 1
        return torch.arange(
            start,
            start + num_frames * step,
            step,
            device=device,
        )
