from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import torchvision.transforms as tf
from PIL import Image
from torch.utils.data import Dataset

from ..export.colmap import read_colmap_model
from ..frame_sampler.frame_sampler import FrameSampler
from .dataset import DatasetCfgCommon
from .dataset_images import DatasetImages, DatasetImagesCfg
from .types import Stage


@dataclass
class DatasetCOLMAPCfg(DatasetCfgCommon):
    name: Literal["colmap"]
    root: Path
    reorder: bool
    use_image_folder_fallback: bool


FAKE_REPETITIONS = 1000


class DatasetCOLMAP(Dataset):
    def __init__(
        self,
        cfg: DatasetCOLMAPCfg,
        stage: Stage,
        frame_sampler: FrameSampler,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.frame_sampler = frame_sampler

        # Use the image dataset as a fallback.
        if cfg.use_image_folder_fallback and not (cfg.root / "sparse").exists():
            self.fallback = DatasetImages(
                DatasetImagesCfg(
                    self.cfg.image_shape, self.cfg.scene, "images", self.cfg.root
                ),
                stage,
                frame_sampler,
            )
            return
        else:
            self.fallback = None

        # Read the COLMAP model.
        self.extrinsics, self.intrinsics, image_names = read_colmap_model(
            cfg.root / "sparse/0", reorder=cfg.reorder
        )
        # Fixed image shapes are intended for pretraining, but this dataset is intended
        # for overfitting.
        assert cfg.image_shape is None

        # Load the images.
        self.frame_paths = [cfg.root / "images" / name for name in image_names]
        self.images = [tf.ToTensor()(Image.open(path))[:3] for path in self.frame_paths]

    def __getitem__(self, index: int):
        if self.fallback is not None:
            return self.fallback.__getitem__(index)

        # Run the frame sampler.
        num_frames = len(self.images)
        indices = self.frame_sampler.sample(num_frames, torch.device("cpu"))

        return {
            "videos": torch.stack([self.images[i] for i in indices]),
            "extrinsics": self.extrinsics[indices],
            "intrinsics": self.intrinsics[indices],
            "indices": indices,
            "scenes": self.cfg.root.stem,
            "datasets": "images",
            "frame_paths": [self.frame_paths[i] for i in indices],
        }

    def __len__(self) -> int:
        # Return a much larger length for compatibility with PyTorch Lightning.
        return FAKE_REPETITIONS
