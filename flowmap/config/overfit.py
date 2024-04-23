from dataclasses import dataclass
from pathlib import Path

from ..model.model_wrapper_overfit import ModelWrapperOverfitCfg
from ..tracking import TrackPrecomputationCfg, TrackPredictorCfg
from .common import CommonCfg


@dataclass
class OverfitCfg(CommonCfg):
    tracking: TrackPredictorCfg
    track_precomputation: TrackPrecomputationCfg
    model_wrapper: ModelWrapperOverfitCfg
    local_save_root: Path | None
