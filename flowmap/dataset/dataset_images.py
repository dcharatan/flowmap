from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import torchvision.transforms as tf
from PIL import Image
from torch.utils.data import Dataset

from ..frame_sampler.frame_sampler import FrameSampler
from .dataset import DatasetCfgCommon
from .types import Stage


@dataclass
class DatasetImagesCfg(DatasetCfgCommon):
    name: Literal["images"]
    root: Path


FAKE_REPETITIONS = 1000


class DatasetImages(Dataset):
    def __init__(
        self,
        cfg: DatasetImagesCfg,
        stage: Stage,
        frame_sampler: FrameSampler,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.frame_sampler = frame_sampler

        # Fixed image shapes are intended for pretraining, but this dataset is intended
        # for overfitting.
        assert cfg.image_shape is None

        # Load the images.
        self.frame_paths = tuple(sorted(cfg.root.iterdir()))
        self.images = [tf.ToTensor()(Image.open(path))[:3] for path in self.frame_paths]

    def __getitem__(self, index: int):
        # Run the frame sampler.
        num_frames = len(self.images)
        indices = self.frame_sampler.sample(num_frames, torch.device("cpu"))

        return {
            "videos": torch.stack([self.images[i] for i in indices]),
            "indices": indices,
            "scenes": self.cfg.root.stem,
            "datasets": "images",
            "frame_paths": [self.frame_paths[i] for i in indices],
        }

    def __len__(self) -> int:
        # Return a much larger length for compatibility with PyTorch Lightning.
        return FAKE_REPETITIONS
