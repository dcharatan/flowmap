import json
from dataclasses import dataclass
from functools import cached_property
from io import BytesIO
from pathlib import Path
from typing import Literal

import torch
import torchvision.transforms as tf
from einops import rearrange, repeat
from jaxtyping import Float, UInt8
from PIL import Image
from torch import Tensor
from torch.utils.data import IterableDataset

from ..frame_sampler.frame_sampler import FrameSampler
from ..misc.cropping import resize_to_cover_with_intrinsics
from .dataset import DatasetCfgCommon
from .types import Stage


@dataclass
class DatasetRE10kCfg(DatasetCfgCommon):
    name: Literal["re10k"]
    roots: list[Path]
    load_cameras: bool
    frame_skip: int


class DatasetRE10k(IterableDataset):
    def __init__(
        self,
        cfg: DatasetRE10kCfg,
        stage: Stage,
        frame_sampler: FrameSampler,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.frame_sampler = frame_sampler
        self.to_tensor = tf.ToTensor()
        self.stage = stage

        # Collect chunks.
        self.chunks = []
        for root in cfg.roots:
            root = root / self.data_stage
            root_chunks = sorted(
                [path for path in root.iterdir() if path.suffix == ".torch"]
            )
            self.chunks.extend(root_chunks)

        # If a specific scene is specified, only load its particular chunk.
        if self.cfg.scene is not None:
            chunk_path = self.index[self.cfg.scene]
            self.chunks = [chunk_path]

    def shuffle(self, lst: list) -> list:
        indices = torch.randperm(len(lst))
        return [lst[x] for x in indices]

    def __iter__(self):
        device = torch.device("cpu")

        # Chunks must be shuffled here (not inside __init__) for validation to show
        # random chunks.
        if self.stage in ("train", "val"):
            self.chunks = self.shuffle(self.chunks)

        for chunk_path in self.chunks:
            # Load the chunk.
            chunk = torch.load(chunk_path)

            if self.cfg.scene is not None:
                item = [x for x in chunk if x["key"] == self.cfg.scene]
                assert len(item) == 1
                chunk = item * len(chunk)

            if self.stage in ("train", "val"):
                chunk = self.shuffle(chunk)

            for example in chunk:
                extrinsics, intrinsics = self.convert_poses(example["cameras"])
                scene = example["key"]

                # Ideally, we would want different frame samplers for each dataset.
                # Since that would be more complicated, we do this instead.
                indices = torch.arange(len(extrinsics))
                indices = indices[:: self.cfg.frame_skip]

                # Pick the frames from the example.
                indices_in_indices = self.frame_sampler.sample(len(indices), device)
                indices = indices[indices_in_indices]

                # Load the images.
                frames = [example["images"][index.item()] for index in indices]
                frames = self.convert_images(frames)

                # If desired, resize and crop the images and adjust the intrinsics
                # accordingly.
                if self.cfg.image_shape is not None:
                    frames, intrinsics = resize_to_cover_with_intrinsics(
                        frames, self.cfg.image_shape, intrinsics
                    )

                result = {
                    "videos": torch.stack([self.to_tensor(frame) for frame in frames]),
                    "indices": indices,
                    "scenes": scene,
                    "datasets": "re10k",
                }

                if self.cfg.load_cameras:
                    result["extrinsics"] = extrinsics[indices]
                    result["intrinsics"] = intrinsics[indices]

                yield result

    def convert_poses(
        self,
        poses: Float[Tensor, "batch 18"],
    ) -> tuple[
        Float[Tensor, "batch 4 4"],  # extrinsics
        Float[Tensor, "batch 3 3"],  # intrinsics
    ]:
        b, _ = poses.shape

        # Convert the intrinsics to a 3x3 normalized K matrix.
        intrinsics = torch.eye(3, dtype=torch.float32)
        intrinsics = repeat(intrinsics, "h w -> b h w", b=b).clone()
        fx, fy, cx, cy = poses[:, :4].T
        intrinsics[:, 0, 0] = fx
        intrinsics[:, 1, 1] = fy
        intrinsics[:, 0, 2] = cx
        intrinsics[:, 1, 2] = cy

        # Convert the extrinsics to a 4x4 OpenCV-style W2C matrix.
        w2c = repeat(torch.eye(4, dtype=torch.float32), "h w -> b h w", b=b).clone()
        w2c[:, :3] = rearrange(poses[:, 6:], "b (h w) -> b h w", h=3, w=4)
        return w2c.inverse(), intrinsics

    def convert_images(
        self,
        images: list[UInt8[Tensor, "..."]],
    ) -> list[Image.Image]:
        return [Image.open(BytesIO(image.numpy().tobytes())) for image in images]

    @property
    def data_stage(self):
        if self.cfg.scene is not None:
            return "test"
        if self.stage == "val":
            return "test"
        return self.stage

    @cached_property
    def index(self) -> dict[str, Path]:
        merged_index = {}
        data_stages = [self.data_stage]
        if self.cfg.scene is not None:
            data_stages = ("test", "train")
        for data_stage in data_stages:
            for root in self.cfg.roots:
                # Load the root's index.
                with (root / data_stage / "index.json").open("r") as f:
                    index = json.load(f)
                index = {k: Path(root / data_stage / v) for k, v in index.items()}

                # The constituent datasets should have unique keys.
                assert not (set(merged_index.keys()) & set(index.keys()))

                # Merge the root's index into the main index.
                merged_index = {**merged_index, **index}
        return merged_index

    def __len__(self) -> int:
        return len(self.index.keys())
