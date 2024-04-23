from dataclasses import dataclass

import torch
from einops import einsum, rearrange, reduce
from jaxtyping import Bool, Float
from torch import Tensor

from flowmap.flow import Flows
from flowmap.model.model import ModelExports
from flowmap.model.projection import homogenize_points, sample_image_grid, unproject
from flowmap.tracking import Tracks

# In the end, we only wanted one correspondence, but these are some other good ones.
HIGHLIGHT_POINTS = [
    # (x, y) in percent of image width/height
    # [44, 25],
    # [63, 42],
    # [77, 26],
    [44, 58],
    # [26, 75],
    # [67, 81],
]


def get_highlight_mask(
    image_shape: tuple[int, int],
    device: torch.device,
) -> Bool[Tensor, "height width"]:
    h, w = image_shape
    mask = torch.zeros((h, w), dtype=torch.bool, device=device)
    for x, y in HIGHLIGHT_POINTS:
        mask[int(y * h / 100), int(x * w / 100)] = True
    return mask


@dataclass
class LoadedScene:
    exports: ModelExports
    xyz_camera_space: Float[Tensor, "frame height width 3"]
    xyz_world_space: Float[Tensor, "frame height width 3"]
    mask: Bool[Tensor, "frame height width"]
    highlight_mask: Bool[Tensor, "height width"]
    midpoint: Float[Tensor, "xyz=3"]  # approx. center of mass
    cutoff: float  # approx. frustum length
    flows: Flows
    tracks: Tracks


def load_birds(device: torch.device) -> LoadedScene:
    exports: ModelExports = torch.load("paper/assets/exports_birds_compact.pt")
    flows, tracks = torch.load("paper/assets/flows_tracks_birds_compact.pt")
    _, _, h, w = exports.depths.shape

    # Generate camera-space 3D points.
    xy, _ = sample_image_grid((h, w), device=device)
    xyz_camera_space = unproject(
        xy,
        exports.depths,
        rearrange(exports.intrinsics, "b f i j -> b f () () i j"),
    )
    xyz_world_space = einsum(
        exports.extrinsics,
        homogenize_points(xyz_camera_space),
        "b f i j, b f h w j -> b f h w i",
    )[..., :3]

    # Figure out what the depth cutoff is (so that far-away points aren't rendered).
    minima = reduce(xyz_world_space[0, 0], "h w xyz -> xyz", "min")
    maxima = reduce(xyz_world_space[0, 0], "h w xyz -> xyz", "max")
    cutoff = minima[..., -1] + (maxima[..., -1] - minima[..., -1]) * 0.35
    cutoff = cutoff.round(decimals=1)
    maxima[-1] = cutoff

    # Define the masks (of included points).
    mask = xyz_camera_space[0, ..., -1] < cutoff
    highlight_mask = get_highlight_mask((h, w), device)
    mask[0] = mask[0] | highlight_mask

    return LoadedScene(
        exports,
        xyz_camera_space[0],
        xyz_world_space[0],
        mask,
        highlight_mask,
        0.5 * (minima + maxima),
        cutoff.item(),
        flows,
        tracks,
    )
