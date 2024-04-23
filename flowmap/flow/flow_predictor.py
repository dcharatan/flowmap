from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn

from ..dataset.types import Batch
from ..misc.manipulable import Manipulable
from ..model.projection import sample_image_grid
from .common import split_videos


@dataclass
class Flows(Manipulable):
    forward: Float[Tensor, "batch pair flow_height flow_width 2"]
    backward: Float[Tensor, "batch pair flow_height flow_width 2"]
    forward_mask: Float[Tensor, "batch pair flow_height flow_width"]
    backward_mask: Float[Tensor, "batch pair flow_height flow_width"]


T = TypeVar("T")


class FlowPredictor(nn.Module, ABC, Generic[T]):
    def __init__(self, cfg: T) -> None:
        super().__init__()
        self.cfg = cfg

    @abstractmethod
    def forward(
        self,
        videos: Float[Tensor, "batch frame 3 height width"],
    ) -> Float[Tensor, "batch frame-1 height width 2"]:
        pass

    @staticmethod
    def rescale_flow(
        flow: Float[Tensor, "batch frame height width 2"],
        shape: tuple[int, int],
    ) -> Float[Tensor, "batch frame height_scaled width_scaled 2"]:
        b, f, _, _, _ = flow.shape
        flow = rearrange(flow, "b f h w xy -> (b f) xy h w")
        flow = F.interpolate(flow, shape, mode="bilinear", align_corners=False)
        return rearrange(flow, "(b f) xy h w -> b f h w xy", b=b, f=f)

    @staticmethod
    def rescale_mask(
        mask: Float[Tensor, "batch frame height width"],
        shape: tuple[int, int],
    ) -> Float[Tensor, "batch frame height_scaled width_scaled"]:
        b, f, _, _ = mask.shape
        flow = rearrange(mask, "b f h w -> (b f) () h w")
        flow = F.interpolate(flow, shape, mode="bilinear", align_corners=False)
        return rearrange(flow, "(b f) () h w -> b f h w", b=b, f=f)

    @staticmethod
    def compute_consistency_mask(
        videos: Float[Tensor, "batch frame 3 height width"],
        flow: Float[Tensor, "batch frame-1 height width 2"],
    ) -> Float[Tensor, "batch frame-1 height width"]:
        source, target, b, f = split_videos(videos)

        # Sample a target pixel for each source pixel.
        _, _, h, w = source.shape
        source_xy, _ = sample_image_grid((h, w), source.device)
        target_xy = source_xy + rearrange(flow, "b f h w xy -> (b f) h w xy")
        target_pixels = F.grid_sample(
            target,
            target_xy * 2 - 1,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )

        # Map pixel color differences to mask weights.
        deltas = (source - target_pixels).abs().max(dim=1).values
        return rearrange((1 - deltas) ** 8, "(b f) h w -> b f h w", b=b, f=f - 1)

    def compute_bidirectional_flow(
        self,
        batch: Batch,
        flow_shape: tuple[int, int],
    ) -> Flows:
        forward = self.forward(batch.videos)
        forward_mask = self.compute_consistency_mask(batch.videos, forward)
        forward = self.rescale_flow(forward, flow_shape)
        forward_mask = self.rescale_mask(forward_mask, flow_shape)

        backward_videos = batch.videos.flip(dims=(1,))

        backward = self.forward(backward_videos)
        backward_mask = self.compute_consistency_mask(backward_videos, backward)
        backward = self.rescale_flow(backward, flow_shape)
        backward_mask = self.rescale_mask(backward_mask, flow_shape)

        backward = backward.flip(dims=(1,))
        backward_mask = backward_mask.flip(dims=(1,))

        return Flows(forward, backward, forward_mask, backward_mask)
