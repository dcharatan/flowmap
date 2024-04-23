import random
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from lightning.pytorch import LightningDataModule
from torch import Generator
from torch.utils.data import DataLoader, Dataset, IterableDataset

from ..frame_sampler import FrameSamplerCfg
from . import DatasetCfg, get_dataset
from .types import Stage
from .validation_wrapper import ValidationWrapper


@dataclass
class DataLoaderStageCfg:
    batch_size: int
    num_workers: int
    persistent_workers: bool
    seed: int | None


@dataclass
class DataModulePretrainCfg:
    train: DataLoaderStageCfg
    val: DataLoaderStageCfg


DatasetShim = Callable[[Dataset, Stage], Dataset]


def worker_init_fn(worker_id: int) -> None:
    random.seed(int(torch.utils.data.get_worker_info().seed) % (2**32 - 1))
    np.random.seed(int(torch.utils.data.get_worker_info().seed) % (2**32 - 1))


class DataModulePretrain(LightningDataModule):
    def __init__(
        self,
        dataset_cfgs: list[DatasetCfg],
        data_module_cfg: DataModulePretrainCfg,
        frame_sampler_cfg: FrameSamplerCfg,
        global_rank: int,
    ) -> None:
        super().__init__()
        self.dataset_cfgs = dataset_cfgs
        self.data_module_cfg = data_module_cfg
        self.frame_sampler_cfg = frame_sampler_cfg
        self.global_rank = global_rank

    def get_persistent(self, loader_cfg: DataLoaderStageCfg) -> bool | None:
        return None if loader_cfg.num_workers == 0 else loader_cfg.persistent_workers

    def get_generator(self, loader_cfg: DataLoaderStageCfg) -> torch.Generator | None:
        if loader_cfg.seed is None:
            return None
        generator = Generator()
        generator.manual_seed(loader_cfg.seed + self.global_rank)
        return generator

    def train_dataloader(self):
        dataset = get_dataset(self.dataset_cfgs, "train", self.frame_sampler_cfg)
        return DataLoader(
            dataset,
            self.data_module_cfg.train.batch_size,
            shuffle=not isinstance(dataset, IterableDataset),
            num_workers=self.data_module_cfg.train.num_workers,
            generator=self.get_generator(self.data_module_cfg.train),
            worker_init_fn=worker_init_fn,
            persistent_workers=self.get_persistent(self.data_module_cfg.train),
        )

    def val_dataloader(self):
        dataset = get_dataset(self.dataset_cfgs, "val", self.frame_sampler_cfg)
        return DataLoader(
            ValidationWrapper(dataset, 1),
            self.data_module_cfg.val.batch_size,
            num_workers=self.data_module_cfg.val.num_workers,
            generator=self.get_generator(self.data_module_cfg.val),
            worker_init_fn=worker_init_fn,
            persistent_workers=self.get_persistent(self.data_module_cfg.val),
        )
