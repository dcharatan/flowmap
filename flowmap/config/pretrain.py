from dataclasses import dataclass

from ..dataset.data_module_pretrain import DataModulePretrainCfg
from ..model.model_wrapper_pretrain import ModelWrapperPretrainCfg
from .common import CommonCfg


@dataclass
class StageCfg:
    batch_size: int
    num_workers: int


@dataclass
class PretrainCfg(CommonCfg):
    model_wrapper: ModelWrapperPretrainCfg
    data_module: DataModulePretrainCfg
