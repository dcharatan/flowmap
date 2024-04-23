from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import torch
from jaxtyping import Int64
from torch import Tensor

T = TypeVar("T")


class FrameSampler(ABC, Generic[T]):
    """A frame sampler picks the frames that should be sampled from a dataset's video.
    It makes sense to break the logic for frame sampling into an interface because
    pre-training and fine-tuning require different frame sampling strategies (generally,
    whole video vs. batch of video segments of same length).
    """

    cfg: T

    def __init__(self, cfg: T) -> None:
        self.cfg = cfg

    @abstractmethod
    def sample(
        self,
        num_frames_in_video: int,
        device: torch.device,
    ) -> Int64[Tensor, " frame"]:  # frame indices
        pass
