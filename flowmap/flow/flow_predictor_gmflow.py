import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor

try:
    from ..third_party.gmflow.gmflow.gmflow import GMFlow
except ImportError:
    GMFlow = None

from .common import split_videos
from .flow_predictor import FlowPredictor


@dataclass
class FlowPredictorGMFlowCfg:
    name: Literal["gmflow"]
    cache_path: Path


class FlowPredictorGMFlow(FlowPredictor[FlowPredictorGMFlowCfg]):
    def __init__(self, cfg: FlowPredictorGMFlowCfg) -> None:
        super().__init__(cfg)

        # Warn that GMFlow isn't installed.
        if GMFlow is None:
            print(
                "Warning: GMFlow could not be imported. Did you forget to initialize "
                "the git submodules?"
            )
            sys.exit(1)

        # Ensure that the checkpoint exists.
        checkpoint = "gmflow-scale1-mixdata-train320x576-4c3a6e9a.pth"
        checkpoint_path = cfg.cache_path / checkpoint
        if not checkpoint_path.exists():
            checkpoint_path.parent.mkdir(exist_ok=True, parents=True)
            print("Downloading GMFlow checkpoint.")
            urllib.request.urlretrieve(
                f"https://s3.eu-central-1.amazonaws.com/avg-projects/unimatch/pretrained/{checkpoint}",
                checkpoint_path,
            )

        # Set up the model.
        self.model = GMFlow(
            feature_channels=128,
            num_scales=1,
            upsample_factor=8,
            num_head=1,
            attention_type="swin",
            ffn_dim_expansion=4,
            num_transformer_layers=6,
        )

        # Load the pre-trained checkpoint.
        checkpoint = torch.load(checkpoint_path)
        weights = checkpoint["model"] if "model" in checkpoint else checkpoint
        self.model.load_state_dict(weights, strict=False)

    def forward(
        self,
        videos: Float[Tensor, "batch frame 3 height width"],
    ) -> Float[Tensor, "batch frame-1 height width 2"]:
        source, target, b, f = split_videos(videos)

        result = self.model(
            source * 255,
            target * 255,
            attn_splits_list=[2],
            corr_radius_list=[-1],
            prop_radius_list=[-1],
            pred_bidir_flow=False,
        )
        flow = result["flow_preds"][-1]

        # Normalize the optical flow.
        _, _, h, w = source.shape
        wh = torch.tensor((w, h), dtype=torch.float32, device=flow.device)
        return rearrange(flow, "(b f) xy h w -> b f h w xy", b=b, f=f - 1) / wh
