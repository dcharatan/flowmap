import shutil
from pathlib import Path

import numpy as np
import torch
from einops import einsum, rearrange
from jaxtyping import Float
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation as R
from torch import Tensor

from ..misc.cropping import center_crop_intrinsics
from ..model.model import ModelExports
from ..model.projection import homogenize_points, sample_image_grid, unproject
from ..third_party.colmap.read_write_model import Camera, Image, read_model, write_model


def read_ply(path: Path) -> tuple[
    Float[np.ndarray, "point 3"],  # xyz
    Float[np.ndarray, "point 3"],  # rgb
]:
    # Adapted from https://github.com/graphdeco-inria/gaussian-splatting/blob/2eee0e26d2d5fd00ec462df47752223952f6bf4e/scene/dataset_readers.py#L107
    plydata = PlyData.read(path)
    vertices = plydata["vertex"]
    xyz = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T
    rgb = np.vstack([vertices["red"], vertices["green"], vertices["blue"]]).T / 255.0
    return xyz, rgb


def write_ply(
    path: Path,
    xyz: Float[np.ndarray, "point 3"],
    rgb: Float[np.ndarray, "point 3"],
) -> None:
    # Adapted from https://github.com/graphdeco-inria/gaussian-splatting/blob/2eee0e26d2d5fd00ec462df47752223952f6bf4e/scene/dataset_readers.py#L115
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]
    normals = np.zeros_like(xyz)
    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb * 255), axis=1)
    elements[:] = list(map(tuple, attributes))
    vertex_element = PlyElement.describe(elements, "vertex")
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def export_to_colmap(
    exports: ModelExports,
    frame_paths: list[Path],
    uncropped_exports_shape: tuple[int, int],
    uncropped_videos: Float[Tensor, "batch frame 3 uncropped_height uncropped_width"],
    path: Path,
) -> None:
    # Account for the cropping that FlowMap does during optimization.
    _, _, h_cropped, w_cropped = exports.depths.shape
    h_uncropped, w_uncropped = uncropped_exports_shape
    intrinsics = center_crop_intrinsics(
        exports.intrinsics,
        (h_cropped, w_cropped),
        (h_uncropped, w_uncropped),
    )

    # Write out the camera parameters.
    sparse_path = path / "sparse/0"
    _, _, _, h_full, w_full = uncropped_videos.shape
    write_colmap_model(
        sparse_path,
        exports.extrinsics[0],
        intrinsics[0],
        [path.name for path in frame_paths],
        (h_full, w_full),
    )

    # Define the point cloud. For compatibility with 3D Gaussian Splatting, this is
    # stored as a .ply instead of Points3D, which seems to be intended for a much
    # smaller number of points.
    _, _, dh, dw = exports.depths.shape
    xy, _ = sample_image_grid((dh, dw), exports.extrinsics.device)
    bundle = zip(
        exports.extrinsics[0],
        exports.intrinsics[0],
        exports.depths[0],
        exports.colors[0],
    )
    points = []
    colors = []
    for extrinsics, intrinsics, depths, rgb in bundle:
        xyz = unproject(xy, depths, intrinsics)
        xyz = homogenize_points(xyz)
        xyz = einsum(extrinsics, xyz, "i j, ... j -> ... i")[..., :3]
        points.append(rearrange(xyz, "h w xyz -> (h w) xyz").detach().cpu().numpy())
        colors.append(rearrange(rgb, "c h w -> (h w) c").detach().cpu().numpy())
    points = np.concatenate(points)
    colors = np.concatenate(colors)

    sparse_path.mkdir(parents=True, exist_ok=True)
    write_ply(sparse_path / "points3D.ply", points, colors)

    # Write out the images.
    (path / "images").mkdir(exist_ok=True, parents=True)
    for frame_path in frame_paths:
        shutil.copy(frame_path, path / "images" / frame_path.name)


def read_colmap_model(
    path: Path,
    device: torch.device = torch.device("cpu"),
    reorder: bool = True,
) -> tuple[
    Float[Tensor, "frame 4 4"],  # extrinsics
    Float[Tensor, "frame 3 3"],  # intrinsics
    list[str],  # image names
]:
    model = read_model(path)
    if model is None:
        raise FileNotFoundError()
    cameras, images, _ = model

    all_extrinsics = []
    all_intrinsics = []
    all_image_names = []

    for image in images.values():
        camera: Camera = cameras[image.camera_id]

        # Read the camera intrinsics.
        intrinsics = torch.eye(3, dtype=torch.float32, device=device)
        if camera.model == "SIMPLE_PINHOLE":
            fx, cx, cy = camera.params
            fy = fx
        elif camera.model == "PINHOLE":
            fx, fy, cx, cy = camera.params
        intrinsics[0, 0] = fx
        intrinsics[1, 1] = fy
        intrinsics[0, 2] = cx
        intrinsics[1, 2] = cy
        intrinsics[0] /= camera.width
        intrinsics[1] /= camera.height
        all_intrinsics.append(intrinsics)

        # Read the camera extrinsics.
        qw, qx, qy, qz = image.qvec
        w2c = torch.eye(4, dtype=torch.float32, device=device)
        rotation = R.from_quat([qx, qy, qz, qw]).as_matrix()
        w2c[:3, :3] = torch.tensor(rotation, dtype=torch.float32, device=device)
        w2c[:3, 3] = torch.tensor(image.tvec, dtype=torch.float32, device=device)
        extrinsics = w2c.inverse()
        all_extrinsics.append(extrinsics)

        # Read the image name.
        all_image_names.append(image.name)

    # Since COLMAP shuffles the images, we generally want to re-order them according
    # to their file names so that they form a video again.
    if reorder:
        ordered = sorted([(name, index) for index, name in enumerate(all_image_names)])
        indices = torch.tensor([index for _, index in ordered])
        all_extrinsics = [all_extrinsics[index] for index in indices]
        all_intrinsics = [all_intrinsics[index] for index in indices]
        all_image_names = [all_image_names[index] for index in indices]

    return torch.stack(all_extrinsics), torch.stack(all_intrinsics), all_image_names


def write_colmap_model(
    path: Path,
    extrinsics: Float[Tensor, "frame 4 4"],
    intrinsics: Float[Tensor, "frame 3 3"],
    image_names: list[str],
    image_shape: tuple[int, int],
) -> None:
    h, w = image_shape

    # Define the cameras (intrinsics).
    cameras = {}
    for index, k in enumerate(intrinsics):
        id = index + 1

        # Undo the normalization we apply to the intrinsics.
        k = k.detach().clone()
        k[0] *= w
        k[1] *= h

        # Extract the intrinsics' parameters.
        fx = k[0, 0]
        fy = k[1, 1]
        cx = k[0, 2]
        cy = k[1, 2]

        cameras[id] = Camera(id, "PINHOLE", w, h, (fx, fy, cx, cy))

    # Define the images (extrinsics and names).
    images = {}
    for index, (c2w, name) in enumerate(zip(extrinsics, image_names)):
        id = index + 1

        # Convert the extrinsics to COLMAP's format.
        w2c = c2w.inverse().detach().cpu().numpy()
        qx, qy, qz, qw = R.from_matrix(w2c[:3, :3]).as_quat()
        qvec = np.array((qw, qx, qy, qz))
        tvec = w2c[:3, 3]
        images[id] = Image(id, qvec, tvec, id, name, [], [])

    path.mkdir(exist_ok=True, parents=True)
    write_model(cameras, images, None, path)
