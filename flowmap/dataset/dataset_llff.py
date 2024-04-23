from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torchvision.transforms as tf
from einops import rearrange, repeat
from jaxtyping import Float
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from ..frame_sampler.frame_sampler import FrameSampler
from ..misc.cropping import center_crop_intrinsics, resize_to_cover
from .dataset import DatasetCfgCommon
from .types import Stage


@dataclass
class Metadata:
    name: str
    extrinsics: Float[Tensor, "frame 4 4"]
    intrinsics: Float[Tensor, "frame 3 3"]


def load_image(
    path: Path,
    shape: tuple[int, int] | None,
) -> tuple[
    Float[Tensor, "3 height width"],  # image
    tuple[int, int],  # image shape after scaling, before cropping
]:
    image = Image.open(path)
    if shape is None:
        pre_crop_shape = (image.height, image.width)
    else:
        image, pre_crop_shape = resize_to_cover(image, shape)
    return tf.ToTensor()(image)[:3], pre_crop_shape


@dataclass
class DatasetLLFFCfg(DatasetCfgCommon):
    name: Literal["llff"]
    root: Path


FAKE_REPETITIONS = 1000


class DatasetLLFF(Dataset):
    def __init__(
        self,
        cfg: DatasetLLFFCfg,
        stage: Stage,
        frame_sampler: FrameSampler,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.frame_sampler = frame_sampler

        if cfg.scene is None:
            self.scenes = [path.name for path in cfg.root.iterdir() if path.is_dir()]
        else:
            self.scenes = [cfg.scene]

    def __getitem__(self, index: int):
        path = self.cfg.root / self.scenes[index % len(self.scenes)]

        # Load the metadata and use it to run the frame sampler.
        metadata = self.load_metadata(path)
        num_frames = len(metadata.extrinsics)
        indices = self.frame_sampler.sample(num_frames, torch.device("cpu"))

        # Collect all image paths, then subsample them using the indices.
        image_paths = sorted((path / "images").iterdir())
        image_paths = [image_paths[index] for index in indices]

        # Since loading full-resolution LLFF images is noticeably slow, we cache resized
        # LLFF images on disk.
        images = []
        for image_path in image_paths:
            image, pre_crop_shape = load_image(image_path, self.cfg.image_shape)
            images.append(image)
        images = torch.stack(images)
        _, _, h, w = images.shape

        return {
            "extrinsics": metadata.extrinsics[indices],
            "intrinsics": center_crop_intrinsics(
                metadata.intrinsics[indices], pre_crop_shape, (h, w)
            ),
            "videos": images,
            "indices": indices,
            "scenes": metadata.name,
            "datasets": "llff",
            "frame_paths": [str(path) for path in image_paths],
        }

    def __len__(self) -> int:
        # Return a much larger length for compatibility with PyTorch Lightning.
        return len(self.scenes) * FAKE_REPETITIONS

    @staticmethod
    def load_metadata(path: Path) -> Metadata:
        # Load the metadata.
        metadata = np.load(path / "poses_bounds.npy")
        metadata = torch.tensor(metadata)

        # Extract extrinsics (rotation and translation) and intrinsics (image size and
        # focal length).
        b, _ = metadata.shape
        cameras = rearrange(metadata[:, :-2], "b (i j) -> b i j", i=3, j=5)
        rotation = cameras[:, :3, :3]
        translation = cameras[:, :3, 3]
        h, w, f = cameras[:, :3, 4].unbind(dim=-1)

        # Load the extrinsics.
        extrinsics = repeat(torch.eye(4), "i j -> b i j", b=b).clone()
        extrinsics[:, :3, :3] = rotation
        extrinsics[:, :3, 3] = translation

        # Convert the extrinsics to OpenCV-style camera-to-world format.
        conversion = torch.zeros((4, 4), dtype=torch.float32)
        conversion[0, 1] = 1
        conversion[1, 0] = 1
        conversion[2, 2] = -1
        conversion[3, 3] = 1
        extrinsics = extrinsics @ conversion

        # Load the intrinsics and normalize them.
        intrinsics = repeat(torch.eye(3), "i j -> b i j", b=b).clone()
        intrinsics[:, :2, 2] = 0.5
        intrinsics[:, 0, 0] = f / w
        intrinsics[:, 1, 1] = f / h

        return Metadata(path.stem, extrinsics, intrinsics)
