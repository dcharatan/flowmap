from .backbone import Backbone
from .backbone_explicit_depth import BackboneExplicitDepth, BackboneExplicitDepthCfg
from .backbone_midas import BackboneMidas, BackboneMidasCfg

BACKBONES = {
    "explicit_depth": BackboneExplicitDepth,
    "midas": BackboneMidas,
}

BackboneCfg = BackboneExplicitDepthCfg | BackboneMidasCfg


def get_backbone(
    cfg: BackboneCfg,
    num_frames: int | None,
    image_shape: tuple[int, int] | None,
) -> Backbone:
    return BACKBONES[cfg.name](cfg, num_frames, image_shape)
