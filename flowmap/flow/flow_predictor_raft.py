from dataclasses import dataclass
from functools import partial
from typing import Literal

import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
from tqdm import tqdm

from .common import split_videos
from .flow_predictor import FlowPredictor


@dataclass
class FlowPredictorRaftCfg:
    name: Literal["raft"]
    num_flow_updates: int
    max_batch_size: int
    show_progress_bar: bool


class FlowPredictorRaft(FlowPredictor[FlowPredictorRaftCfg]):
    def __init__(self, cfg: FlowPredictorRaftCfg) -> None:
        super().__init__(cfg)
        self.raft = raft_large(weights=Raft_Large_Weights.DEFAULT)

    def forward(
        self,
        videos: Float[Tensor, "batch frame 3 height width"],
    ) -> Float[Tensor, "batch frame-1 height width 2"]:
        source, target, b, f = split_videos(videos)

        # RAFT seems to be unhappy with large batch sizes.
        bar = (
            partial(tqdm, desc="Computing RAFT flow")
            if self.cfg.show_progress_bar
            else lambda x: x
        )
        flow = [
            self.raft(
                source_chunk * 2 - 1,
                target_chunk * 2 - 1,
                num_flow_updates=self.cfg.num_flow_updates,
            )[-1]
            for source_chunk, target_chunk in zip(
                bar(source.split(self.cfg.max_batch_size)),
                target.split(self.cfg.max_batch_size),
            )
        ]
        flow = torch.cat(flow)

        # Normalize the optical flow.
        _, _, h, w = source.shape
        wh = torch.tensor((w, h), dtype=torch.float32, device=flow.device)
        return rearrange(flow, "(b f) xy h w -> b f h w xy", b=b, f=f - 1) / wh
