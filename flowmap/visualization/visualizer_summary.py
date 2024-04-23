from dataclasses import dataclass
from typing import Literal

import torch
from einops import rearrange
from flow_vis_torch import flow_to_color
from jaxtyping import Float
from torch import Tensor

from ..dataset.types import Batch
from ..flow import Flows
from ..model.model import Model, ModelOutput
from ..model.projection import compute_backward_flow, sample_image_grid
from ..tracking import Tracks
from .color import apply_color_map_to_image
from .depth import color_map_depth
from .layout import add_border, add_label, hcat, vcat
from .visualizer import Visualizer


def flow_with_key(
    flow: Float[Tensor, "frame height width 2"],
) -> Float[Tensor, "3 height vis_width"]:
    _, h, w, _ = flow.shape
    length = min(h, w)
    x = torch.linspace(-1, 1, length, device=flow.device)
    y = torch.linspace(-1, 1, length, device=flow.device)
    key = torch.stack(torch.meshgrid((x, y), indexing="xy"), dim=0)
    flow = rearrange(flow, "f h w xy -> f xy h w")
    return hcat(
        *(flow_to_color(flow) / 255),
        flow_to_color(key) / 255,
    )


@dataclass
class VisualizerSummaryCfg:
    name: Literal["summary"]
    num_vis_frames: int


class VisualizerSummary(Visualizer[VisualizerSummaryCfg]):
    def visualize(
        self,
        batch: Batch,
        flows: Flows,
        tracks: list[Tracks] | None,
        model_output: ModelOutput,
        model: Model,
        global_step: int,
    ) -> dict[str, Float[Tensor, "3 _ _"] | Float[Tensor, ""]]:
        # For now, only support batch size 1 for visualization.
        b, f, _, h, w = batch.videos.shape
        assert b == 1

        # Pick a random interval to visualize.
        frames = torch.ones(f, dtype=torch.bool, device=batch.videos.device)
        pairs = torch.ones(f - 1, dtype=torch.bool, device=batch.videos.device)
        if self.cfg.num_vis_frames < f:
            start = torch.randint(f - self.cfg.num_vis_frames, (1,)).item()
            frames[:] = False
            frames[start : start + self.cfg.num_vis_frames] = True
            pairs[:] = False
            pairs[start : start + self.cfg.num_vis_frames - 1] = True

        # Color-map the ground-truth optical flow.
        # fwd_gt = flow_with_key(flows.forward[0, pairs])
        bwd_gt = flow_with_key(flows.backward[0, pairs])

        # Color-map the pose-induced optical flow.
        xy_flowed_backward = compute_backward_flow(
            model_output.surfaces,
            model_output.extrinsics,
            model_output.intrinsics,
        )
        xy, _ = sample_image_grid((h, w), batch.videos.device)
        bwd_hat = flow_with_key((xy_flowed_backward - xy)[0, pairs])

        # Color-map the depth.
        depth = color_map_depth(model_output.depths[0, frames])

        # Color-map the correspondence weights.
        bwd_weights = apply_color_map_to_image(
            model_output.backward_correspondence_weights[0, pairs], "gray"
        )

        visualization = vcat(
            add_label(hcat(*batch.videos[0, frames]), "Video (Ground Truth)"),
            add_label(hcat(*depth), "Depth (Predicted)"),
            add_label(bwd_gt, "Backward Flow (Ground Truth)"),
            add_label(bwd_hat, "Backward Flow (Predicted)"),
            add_label(hcat(*(bwd_weights)), "Backward Correspondence Weights"),
        )

        return {"summary": add_border(visualization)}
