from .mapping import Mapping
from .mapping_huber import MappingHuber, MappingHuberCfg
from .mapping_l1 import MappingL1, MappingL1Cfg
from .mapping_l2 import MappingL2, MappingL2Cfg

MAPPINGS = {
    "huber": MappingHuber,
    "l1": MappingL1,
    "l2": MappingL2,
}

MappingCfg = MappingHuberCfg | MappingL1Cfg | MappingL2Cfg


def get_mapping(cfg: MappingCfg) -> Mapping:
    return MAPPINGS[cfg.name](cfg)
