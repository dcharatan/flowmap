from dataclasses import dataclass
from typing import Literal

import torch
from jaxtyping import Int64
from torch import Tensor

from .frame_sampler import FrameSampler


@dataclass
class FrameSamplerPretrainCfg:
    name: Literal["pretrain"]
    num_frames: int


class FrameSamplerPretrain(FrameSampler[FrameSamplerPretrainCfg]):
    def sample(
        self,
        num_frames_in_video: int,
        device: torch.device,
    ) -> Int64[Tensor, " frame"]:
        # If the video doesn't have enough frames, just repeat the last frame.
        if num_frames_in_video < self.cfg.num_frames:
            indices = torch.arange(self.cfg.num_frames, device=device)
            indices[indices >= num_frames_in_video] = num_frames_in_video - 1
            return indices

        # If the video has enough frames, pick a random starting point.
        start = torch.randint(0, num_frames_in_video - self.cfg.num_frames + 1, tuple())
        return torch.arange(start, start + self.cfg.num_frames, device=device)
