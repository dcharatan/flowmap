from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import torch
from jaxtyping import Float
from torch import Tensor, nn


def fix_aspect_ratio(
    points: Float[Tensor, "*batch 2"],
    image_shape: tuple[int, int],
) -> Float[Tensor, "*batch 2"]:
    """When computing losses on normalized image coordinates (width in range [0, 1] and
    height in range [0, 1]), distances are skewed based on the aspect ratio. This
    function scales space based on the aspect ratio to correct for this skew.
    """
    h, w = image_shape
    scale = (h * w) ** 0.5
    correction = torch.tensor(
        (w / scale, h / scale),
        dtype=points.dtype,
        device=points.device,
    )
    return points * correction


T = TypeVar("T")


class Mapping(nn.Module, ABC, Generic[T]):
    def __init__(self, cfg: T) -> None:
        super().__init__()
        self.cfg = cfg

    def forward(
        self,
        a: Float[Tensor, "*#batch 2"],
        b: Float[Tensor, "*#batch 2"],
        image_shape: tuple[int, int],
    ) -> Float[Tensor, " *batch"]:
        a = fix_aspect_ratio(a, image_shape)
        b = fix_aspect_ratio(b, image_shape)
        return self.forward_undistorted(a - b)

    @abstractmethod
    def forward_undistorted(
        self,
        delta: Float[Tensor, "*batch 2"],
    ) -> Float[Tensor, " *batch"]:
        pass
