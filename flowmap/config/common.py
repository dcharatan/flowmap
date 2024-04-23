from dataclasses import dataclass
from typing import Literal, Type, TypeVar

from omegaconf import DictConfig

from ..dataset import DatasetCfg
from ..flow import FlowPredictorCfg
from ..frame_sampler import FrameSamplerCfg
from ..loss import LossCfg
from ..misc.cropping import CroppingCfg
from ..model.model import ModelCfg
from ..visualization import VisualizerCfg
from .tools import get_typed_config, separate_multiple_defaults


@dataclass
class WandbCfg:
    project: str
    mode: Literal["online", "offline", "disabled"]
    name: str | None
    group: str | None
    tags: list[str] | None


@dataclass
class CheckpointCfg:
    every_n_train_steps: int
    load: str | None  # str instead of Path, since it could be wandb://...


@dataclass
class TrainerCfg:
    val_check_interval: int
    max_steps: int


@dataclass
class CommonCfg:
    wandb: WandbCfg
    checkpoint: CheckpointCfg
    trainer: TrainerCfg
    flow: FlowPredictorCfg
    dataset: list[DatasetCfg]
    frame_sampler: FrameSamplerCfg
    model: ModelCfg
    loss: list[LossCfg]
    visualizer: list[VisualizerCfg]
    cropping: CroppingCfg


T = TypeVar("T")


def get_typed_root_config(cfg_dict: DictConfig, cfg_type: Type[T]) -> T:
    return get_typed_config(
        cfg_type,
        cfg_dict,
        {
            list[DatasetCfg]: separate_multiple_defaults(DatasetCfg),
            list[LossCfg]: separate_multiple_defaults(LossCfg),
            list[VisualizerCfg]: separate_multiple_defaults(VisualizerCfg),
        },
    )
