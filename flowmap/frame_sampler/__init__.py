from typing import Any

from .frame_sampler import FrameSampler
from .frame_sampler_overfit import FrameSamplerOverfit, FrameSamplerOverfitCfg
from .frame_sampler_pretrain import FrameSamplerPretrain, FrameSamplerPretrainCfg

FRAME_SAMPLER = {
    "overfit": FrameSamplerOverfit,
    "pretrain": FrameSamplerPretrain,
}

FrameSamplerCfg = FrameSamplerPretrainCfg | FrameSamplerOverfitCfg


def get_frame_sampler(cfg: FrameSamplerCfg) -> FrameSampler[Any]:
    return FRAME_SAMPLER[cfg.name](cfg)
