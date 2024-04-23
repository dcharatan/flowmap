from dataclasses import dataclass
from pathlib import Path

import torch
from jaxtyping import Float, Int64
from torch import Tensor
from tqdm import trange

from ..dataset.types import Batch
from ..misc.disk_cache import make_cache
from ..misc.nn_module_tools import convert_to_buffer
from .track_predictor import TrackPredictor, Tracks
from .track_predictor_cotracker import (
    TrackPredictorCoTracker,
    TrackPredictorCoTrackerCfg,
)

TRACKERS = {
    "cotracker": TrackPredictorCoTracker,
}

TrackPredictorCfg = TrackPredictorCoTrackerCfg


def get_track_predictor(cfg: TrackPredictorCfg) -> TrackPredictor:
    tracker = TRACKERS[cfg.name](cfg)
    convert_to_buffer(tracker, persistent=False)
    return tracker


def get_cache_key(
    dataset: str,
    scene: str,
    indices: Int64[Tensor, " frame"],
    min_num_tracks: int,
    max_track_interval: int,
) -> tuple[str, str, int, int, int, int]:
    first, *_, last = indices
    return (
        dataset,
        scene,
        first.item(),
        last.item(),
        min_num_tracks,
        max_track_interval,
    )


def generate_video_tracks(
    tracker: TrackPredictor,
    videos: Float[Tensor, "batch frame 3 height width"],
    interval: int,
    radius: int,
) -> list[Tracks]:
    segment_tracks = []

    _, f, _, _, _ = videos.shape
    for middle_frame in trange(0, f, interval, desc="Computing tracks"):
        # Retrieve the video segment we want to compute tracks for.
        start_frame = max(0, middle_frame - radius)
        end_frame = min(f, middle_frame + radius + 1)
        segment = videos[:, start_frame:end_frame]

        # Compute tracks on the segment, then mark the tracks with the segment's
        # starting frame so that they can be matched to the segment.
        tracks = tracker.forward(segment, middle_frame - start_frame)
        tracks.start_frame = start_frame
        segment_tracks.append(tracks)

    return segment_tracks


@dataclass
class TrackPrecomputationCfg:
    cache_path: Path | None
    interval: int
    radius: int


def compute_tracks(
    batch: Batch,
    device: torch.device,
    tracking_cfg: TrackPredictorCfg,
    precomputation_cfg: TrackPrecomputationCfg,
) -> list[Tracks]:
    # Set up the tracker.
    tracker = get_track_predictor(tracking_cfg)
    tracker.to(device)

    # Since we only use tracks for overfitting, assert that the batch size is 1.
    b, _, _, _, _ = batch.videos.shape
    assert b == 1

    cache_key = get_cache_key(
        batch.datasets[0],
        batch.scenes[0],
        batch.indices[0],
        precomputation_cfg.interval,
        precomputation_cfg.radius,
    )
    disk_cache = make_cache(precomputation_cfg.cache_path)
    return disk_cache(
        cache_key,
        lambda: generate_video_tracks(
            tracker,
            batch.videos[:1].to(device),
            precomputation_cfg.interval,
            precomputation_cfg.radius,
        ),
    )
