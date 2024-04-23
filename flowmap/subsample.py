import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Float, install_import_hook
from omegaconf import DictConfig
from torch import Tensor
from tqdm import tqdm

# Configure beartype and jaxtyping.
with install_import_hook(
    ("flowmap",),
    ("beartype", "beartype"),
):
    from .config.common import get_typed_root_config
    from .flow import FlowPredictor, FlowPredictorCfg, get_flow_predictor
    from .misc.cropping import center_crop_images, compute_patch_cropped_shape
    from .misc.image_io import load_image


@dataclass
class SubsampleCfg:
    flow: FlowPredictorCfg
    in_path: Path
    out_path: Path
    target_num_frames: int
    flow_resolution: int
    limit_num_seconds: float | None


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="subsample",
)
def subsample(cfg_dict: DictConfig):
    cfg = get_typed_root_config(cfg_dict, SubsampleCfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up the flow predictor.
    flow_predictor = get_flow_predictor(cfg.flow)
    flow_predictor.to(device)

    with tempfile.TemporaryDirectory() as work_dir:
        work_dir = Path(work_dir)

        # If the input isn't a directory, assume it's a video.
        if cfg.in_path.is_dir():
            frame_dir = cfg.in_path
        else:
            video_to_frames(cfg.in_path, work_dir, cfg.limit_num_seconds)
            frame_dir = work_dir

        subsample_frames(
            flow_predictor,
            frame_dir,
            cfg.out_path,
            cfg.target_num_frames,
            cfg.flow_resolution,
            device,
        )


def video_to_frames(
    in_path: Path,
    out_path: Path,
    limit_num_seconds: float | None,
) -> None:
    """Convert a full video to frames using ffmpeg."""
    out_path.mkdir(exist_ok=True, parents=True)
    limit = None if limit_num_seconds is None else f"-t {limit_num_seconds}"
    command = f"ffmpeg -i {in_path} {limit} {out_path}/frame_%06d.jpg"
    if subprocess.run(command.split(" ")).returncode != 0:
        raise ValueError("ffmpeg conversion failed")


def resize_to_resolution(
    image: Float[Tensor, "3 height width"],
    resolution: int,
) -> Float[Tensor, "3 resized_height resized_width"]:
    _, h, w = image.shape

    scale = (resolution / (h * w)) ** 0.5
    return F.interpolate(
        image[None],
        scale_factor=scale,
        mode="bilinear",
        align_corners=False,
    )[0]


def subsample_frames(
    flow_predictor: FlowPredictor,
    full_video_path: Path,
    subsampled_path: Path,
    target_num_frames: int,
    flow_resolution: int,
    device: torch.device,
) -> None:
    # Just symlink the frames if they don't need to be subsampled.
    if len(list(full_video_path.iterdir())) <= target_num_frames:
        subsampled_path.parent.mkdir(exist_ok=True, parents=True)
        shutil.copytree(full_video_path, subsampled_path)
        return

    last = None
    mean_flows = []
    for image in tqdm(
        list(sorted(full_video_path.iterdir())), desc="Computing mean flows"
    ):
        # Get the current two-frame video segment.
        if last is None:
            last = resize_to_resolution(load_image(image), flow_resolution)
            continue
        current = resize_to_resolution(load_image(image), flow_resolution)
        videos = torch.stack((last, current))[None].to(device)

        # Crop the video segment.
        _, _, _, h, w = videos.shape
        new_shape = compute_patch_cropped_shape((h, w), 8)
        videos = center_crop_images(videos, new_shape)

        # Compute the mean flows.
        mean_flows.append(flow_predictor.forward(videos).norm(dim=-1).mean().item())

    flow_step = sum(mean_flows) / target_num_frames
    remaining = 0
    subsampled_path.mkdir(exist_ok=True, parents=True)
    num_saved = 0
    for mean_flow, frame in zip(mean_flows, sorted(full_video_path.iterdir())):
        if remaining <= 0:
            shutil.copy(frame, subsampled_path / frame.name)
            remaining += flow_step
            num_saved += 1

        remaining -= mean_flow

    # Randomly fill in the remaining frames.
    generator = np.random.default_rng(seed=0)
    paths = list(full_video_path.iterdir())
    while num_saved < target_num_frames:
        # Pick a random frame.
        frame = paths[generator.choice(len(paths))]
        if (subsampled_path / frame.name).exists():
            continue
        shutil.copy(frame, subsampled_path / frame.name)
        num_saved += 1

    assert num_saved == target_num_frames


if __name__ == "__main__":
    subsample()
