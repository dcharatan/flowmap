from dataclasses import dataclass
from typing import Literal

from einops import rearrange
from jaxtyping import Float
from torch import Tensor

from ..dataset.types import Batch
from ..flow import Flows
from ..model.model import ModelOutput
from ..model.projection import compute_track_flow
from ..tracking import Tracks
from .loss import Loss, LossCfgCommon
from .mapping import MappingCfg, get_mapping


@dataclass
class LossTrackingCfg(LossCfgCommon):
    name: Literal["tracking"]
    mapping: MappingCfg


class LossTracking(Loss[LossTrackingCfg]):
    def __init__(self, cfg: LossTrackingCfg) -> None:
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
        # Tracks must be available for the tracking loss.
        assert tracks is not None

        _, _, _, h, w = batch.videos.shape

        loss_sum = 0
        valid_sum = 0

        for segment_tracks in tracks:
            _, f, _, _ = segment_tracks.xy.shape
            s = segment_tracks.start_frame

            xy_target, visibility = compute_track_flow(
                model_output.surfaces[:, s : s + f],
                model_output.extrinsics[:, s : s + f],
                model_output.intrinsics[:, s : s + f],
                segment_tracks,
            )
            xy_target_gt = rearrange(segment_tracks.xy, "b ft p xy -> b () ft p xy")

            loss = self.mapping.forward(xy_target, xy_target_gt, (h, w)) * visibility

            loss_sum = loss_sum + loss.sum()
            valid_sum = valid_sum + visibility.sum()

        return loss_sum / (valid_sum or 1)
