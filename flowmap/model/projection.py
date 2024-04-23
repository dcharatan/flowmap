import torch
import torch.nn.functional as F
from einops import einsum, rearrange
from jaxtyping import Bool, Float, Int64
from torch import Tensor

from ..tracking import Tracks
from .procrustes import align_rigid


def homogenize_points(
    points: Float[Tensor, "*batch dim"],
) -> Float[Tensor, "*batch dim+1"]:
    """Convert batched points (xyz) to (xyz1)."""
    return torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)


def homogenize_vectors(
    vectors: Float[Tensor, "*batch dim"],
) -> Float[Tensor, "*batch dim+1"]:
    """Convert batched vectors (xyz) to (xyz0)."""
    return torch.cat([vectors, torch.zeros_like(vectors[..., :1])], dim=-1)


def transform_rigid(
    homogeneous_coordinates: Float[Tensor, "*#batch dim"],
    transformation: Float[Tensor, "*#batch dim dim"],
) -> Float[Tensor, "*batch dim"]:
    """Apply a rigid-body transformation to points or vectors."""
    return einsum(transformation, homogeneous_coordinates, "... i j, ... j -> ... i")


def transform_cam2world(
    homogeneous_coordinates: Float[Tensor, "*#batch dim"],
    extrinsics: Float[Tensor, "*#batch dim dim"],
) -> Float[Tensor, "*batch dim"]:
    """Transform points from 3D camera coordinates to 3D world coordinates."""
    return transform_rigid(homogeneous_coordinates, extrinsics)


def transform_world2cam(
    homogeneous_coordinates: Float[Tensor, "*#batch dim"],
    extrinsics: Float[Tensor, "*#batch dim dim"],
) -> Float[Tensor, "*batch dim"]:
    """Transform points from 3D world coordinates to 3D camera coordinates."""
    return transform_rigid(homogeneous_coordinates, extrinsics.inverse())


def project_camera_space(
    points: Float[Tensor, "*#batch dim"],
    intrinsics: Float[Tensor, "*#batch dim dim"],
    epsilon: float = 1e-5,
    infinity: float = 1e8,
) -> Float[Tensor, "*batch dim-1"]:
    points = points / (points[..., -1:] + epsilon)
    points = points.nan_to_num(posinf=infinity, neginf=-infinity)
    points = einsum(intrinsics, points, "... i j, ... j -> ... i")
    return points[..., :-1]


def project(
    points: Float[Tensor, "*#batch dim"],
    extrinsics: Float[Tensor, "*#batch dim+1 dim+1"],
    intrinsics: Float[Tensor, "*#batch dim dim"],
    epsilon: float = 1e-5,
) -> tuple[
    Float[Tensor, "*batch dim-1"],  # xy coordinates
    Bool[Tensor, " *batch"],  # whether points are in front of the camera
]:
    points = homogenize_points(points)
    points = transform_world2cam(points, extrinsics)[..., :-1]
    in_front_of_camera = points[..., -1] >= 0
    return project_camera_space(points, intrinsics, epsilon=epsilon), in_front_of_camera


def unproject(
    coordinates: Float[Tensor, "*#batch dim"],
    z: Float[Tensor, "*#batch"],
    intrinsics: Float[Tensor, "*#batch dim+1 dim+1"],
) -> Float[Tensor, "*batch dim+1"]:
    """Unproject 2D camera coordinates with the given Z values."""

    # Apply the inverse intrinsics to the coordinates.
    coordinates = homogenize_points(coordinates)
    ray_directions = einsum(
        intrinsics.inverse(), coordinates, "... i j, ... j -> ... i"
    )

    # Apply the supplied depth values.
    return ray_directions * z[..., None]


def sample_image_grid(
    shape: tuple[int, ...],
    device: torch.device = torch.device("cpu"),
) -> tuple[
    Float[Tensor, "*shape dim"],  # float coordinates (xy indexing)
    Int64[Tensor, "*shape dim"],  # integer indices (ij indexing)
]:
    """Get normalized (range 0 to 1) coordinates and integer indices for an image."""

    # Each entry is a pixel-wise integer coordinate. In the 2D case, each entry is a
    # (row, col) coordinate.
    indices = [torch.arange(length, device=device) for length in shape]
    stacked_indices = torch.stack(torch.meshgrid(*indices, indexing="ij"), dim=-1)

    # Each entry is a floating-point coordinate in the range (0, 1). In the 2D case,
    # each entry is an (x, y) coordinate.
    coordinates = [(idx + 0.5) / length for idx, length in zip(indices, shape)]
    coordinates = reversed(coordinates)
    coordinates = torch.stack(torch.meshgrid(*coordinates, indexing="xy"), dim=-1)

    return coordinates, stacked_indices


def reproject_points(
    xyz: Float[Tensor, "*#batch 3"],
    relative_transformations: Float[Tensor, "*#batch 4 4"],
    intrinsics: Float[Tensor, "*#batch 3 3"],
) -> Float[Tensor, "*#batch 2"]:
    """Transform the input points using the provided relative transformations, then
    project them using the provided intrinsics. After transformation, the points are
    assumed to be in camera space.
    """

    # Transform the 3D locations into the target view's camera space.
    xyz = einsum(
        relative_transformations,
        homogenize_points(xyz),
        "... i j, ... j -> ... i",
    )[..., :3]

    # Project the 3D locations in the target view's camera space.
    return project_camera_space(xyz, intrinsics)


# Given data with leading (batch, frame) dimensions, these helper functions select the
# earlier and later set of frames, respectively.
earlier = lambda x: x[:, :-1]  # noqa
later = lambda x: x[:, 1:]  # noqa


def compute_forward_flow(
    surfaces: Float[Tensor, "batch frame *grid xyz=3"],
    extrinsics: Float[Tensor, "batch frame 4 4"],
    intrinsics: Float[Tensor, "batch frame 3 3"],
) -> Float[Tensor, "batch frame-1 *grid xy=2"]:
    """Return the positions of all surface points with forward optical flow applied."""

    # Since the poses are camera-to-world and transformations are applied right to
    # left, this can be understood as follows: First, transform from the earlier
    # frame's camera space to world space. Then, transform from world space into the
    # later frame's camera space.
    forward_transformation = later(extrinsics).inverse() @ earlier(extrinsics)

    singletons = " ".join(["()"] * (surfaces.ndim - 3))
    pattern = f"b f i j -> b f {singletons} i j"
    return reproject_points(
        earlier(surfaces),
        rearrange(forward_transformation, pattern),
        rearrange(later(intrinsics), pattern),
    )


def compute_backward_flow(
    surfaces: Float[Tensor, "batch frame *grid xyz=3"],
    extrinsics: Float[Tensor, "batch frame 4 4"],
    intrinsics: Float[Tensor, "batch frame 3 3"],
) -> Float[Tensor, "batch frame-1 *grid xy=2"]:
    """Return the positions of all surface points with backward optical flow applied."""

    # Since the poses are camera-to-world and transformations are applied right to
    # left, this can be understood as follows: First, transform from the later
    # frame's camera space to world space. Then, transform from world space into the
    # earlier frame's camera space.
    backward_transformation = earlier(extrinsics).inverse() @ later(extrinsics)

    singletons = " ".join(["()"] * (surfaces.ndim - 3))
    pattern = f"b f i j -> b f {singletons} i j"
    return reproject_points(
        later(surfaces),
        rearrange(backward_transformation, pattern),
        rearrange(earlier(intrinsics), pattern),
    )


def get_extrinsics(
    inverse_relative_transformations: Float[Tensor, "*batch pair 4 4"],
) -> Float[Tensor, "*batch pair+1 4 4"]:
    """Convert the inverse relative transformations from ModelOutput to extrinsics.
    Each inverse relative transformation transforms points from frame {i + 1}'s
    camera space to frame i's camera space. Since our extrinsics are in
    camera-to-world format, this means that expressed in terms of camera poses, each
    inverse relative transformation is (P_i^-1 @ P_{i + 1}). If we assume that P_0
    is I (the identity pose), we can thus extract camera poses as follows:

    P_n = (I @ P_1) @ (P_1^-1 @ P_2) @ ... @ (P_{n - 1}^-1 @ P_n)

    This is slightly counterintuitive, since transformations are generally composed
    by right-to-left multiplication.
    """
    *batch, step, _, _ = inverse_relative_transformations.shape
    device = inverse_relative_transformations.device
    pose = torch.eye(4, dtype=torch.float32, device=device)
    pose = pose.expand((*batch, 4, 4)).contiguous()
    result = [pose]
    for i in range(step):
        pose = pose @ inverse_relative_transformations[..., i, :, :]
        result.append(pose)
    return torch.stack(result, dim=-3)


def align_surfaces(
    surfaces: Float[Tensor, "batch frame height width 3"],
    backward_flows: Float[Tensor, "batch frame-1 height width xy=2"],
    backward_weights: Float[Tensor, "batch frame-1 height width"],
    indices: Int64[Tensor, " pixel_index"],
) -> Float[Tensor, "batch frame 4 4"]:
    b, f, h, w, _ = surfaces.shape

    # Convert the depth maps into camera-space 3D surfaces (b, f, h, w, xyz).
    xy, _ = sample_image_grid((h, w), device=surfaces.device)

    # Subsample the surfaces to select points for Procrustes alignment. Select the later
    # points from the surfaces using the provided indices.
    xyz_later = rearrange(later(surfaces), "b f h w xyz -> b f (h w) xyz")
    xyz_later = xyz_later[:, :, indices]

    # Flow the grid of XY locations backwards, then select from the flowed XY locations
    # using the provided indices.
    xy_earlier = rearrange(xy + backward_flows, "b f h w xy -> b f (h w) xy")
    xy_earlier = xy_earlier[:, :, indices]

    # Use the earlier XY locations to select from the earlier 3D surfaces.
    xyz_earlier = F.grid_sample(
        rearrange(earlier(surfaces), "b f h w xyz -> (b f) xyz h w"),
        rearrange(xy_earlier * 2 - 1, "b f p xy -> (b f) p () xy"),
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )
    xyz_earlier = rearrange(xyz_earlier, "(b f) xyz p () -> b f p xyz", b=b, f=f - 1)

    # Estimate poses via Procrustes alignment.
    inverse_relative_transformations = align_rigid(
        xyz_later,
        xyz_earlier,
        rearrange(backward_weights, "b f h w -> b f (h w)")[..., indices],
    )
    extrinsics = get_extrinsics(inverse_relative_transformations)

    return extrinsics


def compute_track_flow(
    surfaces: Float[Tensor, "batch frame height width xyz=3"],
    extrinsics: Float[Tensor, "batch frame 4 4"],
    intrinsics: Float[Tensor, "batch frame 3 3"],
    tracks: Tracks,
) -> tuple[
    Float[Tensor, "batch frame_source frame_target point 2"],  # flow
    Bool[Tensor, "batch frame_source frame_target point"],  # visibility
]:
    # Sample the surfaces at the track locations.
    b, f, _, _, _ = surfaces.shape
    xyz = F.grid_sample(
        rearrange(surfaces, "b f h w xyz -> (b f) xyz h w"),
        rearrange(tracks.xy * 2 - 1, "b f p xy -> (b f) () p xy"),
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )
    xyz = rearrange(xyz, "(b f) xy () p -> b f p xy", b=b, f=f)

    # Add singleton dimensions so that everything broadcasts to the following shape:
    # (b = batch, fs = source frame, ft = target frame, p = point)
    xy_source = rearrange(tracks.xy, "b fs p xy -> b fs () p xy")
    xyz_source = rearrange(xyz, "b fs p xyz -> b fs () p xyz")
    extrinsics_source = rearrange(extrinsics, "b fs i j -> b fs () () i j")
    extrinsics_target = rearrange(extrinsics, "b ft i j -> b () ft () i j")
    intrinsics_target = rearrange(intrinsics, "b ft i j -> b () ft () i j")
    visibility_source = rearrange(tracks.visibility, "b fs p -> b fs () p")
    visibility_target = rearrange(tracks.visibility, "b ft p -> b () ft p")

    # Compute flow and visibility.
    xy_target = reproject_points(
        xyz_source,
        extrinsics_target.inverse() @ extrinsics_source,
        intrinsics_target,
    )
    visibility = visibility_source & visibility_target

    # Filter out points that are not in the frame for either the source or target.
    source_in_frame = (xy_source >= 0).all(dim=-1) & (xy_source < 1).all(dim=-1)
    target_in_frame = (xy_target >= 0).all(dim=-1) & (xy_target < 1).all(dim=-1)
    visibility = visibility & source_in_frame & target_in_frame

    return xy_target, visibility
