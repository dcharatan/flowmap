from dataclasses import dataclass
from typing import Literal

from jaxtyping import Float
from torch import Tensor

from ..dataset.types import Batch
from ..flow import Flows
from ..model.model import ModelOutput
from ..model.projection import (
    compute_backward_flow,
    compute_forward_flow,
    sample_image_grid,
)
from ..tracking import Tracks
from .loss import Loss, LossCfgCommon
from .mapping import MappingCfg, get_mapping


@dataclass
class LossFlowCfg(LossCfgCommon):
    name: Literal["flow"]
    mapping: MappingCfg


class LossFlow(Loss[LossFlowCfg]):
    def __init__(self, cfg: LossFlowCfg) -> None:
        super().__init__(cfg)
        self.mapping = get_mapping(cfg.mapping)

    def compute_unweighted_loss(
        self,
        batch: Batch,
        flows: Flows,
        tracks: list[Tracks] | None,
        model_output: ModelOutput,
        global_step: int,
    ) -> Float[Tensor, ""]:
        _, _, _, h, w = batch.videos.shape
        device = batch.videos.device
        xy, _ = sample_image_grid((h, w), device)

        loss_sum = 0
        valid_sum = 0

        # Compute a loss based on forward flow.
        xy_flowed_forward = compute_forward_flow(
            model_output.surfaces,
            model_output.extrinsics,
            model_output.intrinsics,
        )
        forward_loss = self.mapping.forward(
            xy_flowed_forward - xy, flows.forward, (h, w)
        )
        loss_sum = loss_sum + (forward_loss * flows.forward_mask).sum()
        valid_sum = valid_sum + flows.forward_mask.sum()

        # Compute a loss based on backward flow.
        xy_flowed_backward = compute_backward_flow(
            model_output.surfaces,
            model_output.extrinsics,
            model_output.intrinsics,
        )
        backward_loss = self.mapping.forward(
            xy_flowed_backward - xy, flows.backward, (h, w)
        )
        loss_sum = loss_sum + (backward_loss * flows.backward_mask).sum()
        valid_sum = valid_sum + flows.backward_mask.sum()

        return loss_sum / (valid_sum or 1)
