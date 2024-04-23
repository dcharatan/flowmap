from .extrinsics import Extrinsics
from .extrinsics_procrustes import ExtrinsicsProcrustes, ExtrinsicsProcrustesCfg
from .extrinsics_regressed import ExtrinsicsRegressed, ExtrinsicsRegressedCfg

EXTRINSICS = {
    "procrustes": ExtrinsicsProcrustes,
    "regressed": ExtrinsicsRegressed,
}

ExtrinsicsCfg = ExtrinsicsProcrustesCfg | ExtrinsicsRegressedCfg


def get_extrinsics(
    cfg: ExtrinsicsCfg,
    num_frames: int | None,
) -> Extrinsics:
    return EXTRINSICS[cfg.name](cfg, num_frames)
