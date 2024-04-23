from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float
from torch import Tensor

from .track_predictor import TrackPredictor, Tracks


@dataclass
class TrackPredictorCoTrackerCfg:
    name: Literal["cotracker"]
    grid_size: int
    similarity_threshold: float


class TrackPredictorCoTracker(TrackPredictor[TrackPredictorCoTrackerCfg]):
    def __init__(self, cfg: TrackPredictorCoTrackerCfg) -> None:
        super().__init__(cfg)
        self.tracker = torch.hub.load(
            "facebookresearch/co-tracker:v1.0", "cotracker_w8"
        )

    def forward(
        self,
        videos: Float[Tensor, "batch frame 3 height width"],
        query_frame: int,
    ) -> Tracks:
        xy, visibility = self.tracker(
            videos * 255,
            grid_size=self.cfg.grid_size,
            grid_query_frame=query_frame,
            backward_tracking=True,
        )

        # Normalize the coordinates.
        b, f, _, h, w = videos.shape
        wh = torch.tensor((w, h), dtype=torch.float32, device=videos.device)
        xy = xy / wh

        # Filter visibility based on RGB values.
        rgb = F.grid_sample(
            rearrange(videos, "b f c h w -> (b f) c h w"),
            rearrange(xy, "b f p xy -> (b f) p () xy"),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        rgb = rearrange(rgb, "(b f) c p () -> b f p c", b=b, f=f)
        rgb_delta = (rgb[:, [query_frame]] - rgb).abs().norm(dim=-1)
        visibility = visibility & (rgb_delta < self.cfg.similarity_threshold)

        return Tracks(xy, visibility, 0)
