import shutil
from pathlib import Path

import click
import yaml
from omegaconf import DictConfig
from PIL import Image
from tqdm import tqdm

from flowmap.config.tools import get_typed_config
from flowmap.export.colmap import read_colmap_model, write_colmap_model
from flowmap.misc.cropping import CroppingCfg, get_image_shape

from .run_dense import run_dense
from .run_sparse import run_sparse


@click.command()
@click.argument("input_path", type=click.Path(path_type=Path))
@click.argument("output_sparse_path", type=click.Path(path_type=Path))
@click.argument("output_dense_path", type=click.Path(path_type=Path))
@click.argument("workspace_path", type=click.Path(path_type=Path))
def main(
    input_path: Path,
    output_sparse_path: Path,
    output_dense_path: Path,
    workspace_path: Path,
) -> None:
    # Load the FlowMap configuration.
    with Path("config/overfit.yaml").open("r") as f:
        cfg = yaml.safe_load(f)
    cfg = get_typed_config(CroppingCfg, DictConfig(cfg["cropping"]))

    # Then, we crop the input images to the appropriate shape.
    resized_images_dir = workspace_path / "resized_images"
    resized_images_dir.mkdir(exist_ok=True, parents=True)
    for image_path in tqdm(list(input_path.iterdir()), desc="Resizing images"):
        image = Image.open(image_path)

        h_original = image.height
        w_original = image.width

        h, w = (
            dim * cfg.flow_scale_multiplier
            for dim in get_image_shape((image.height, image.width), cfg)
        )
        image = image.resize((w, h), Image.LANCZOS)
        image.save(resized_images_dir / image_path.name)

    # Run sparse COLMAP.
    resized_sparse_dir = workspace_path / "resized_sparse"
    run_sparse(resized_images_dir, resized_sparse_dir, 0, "extreme", "video")

    # Run dense COLMAP.
    resized_dense_dir = workspace_path / "resized_dense"
    run_dense(resized_sparse_dir, resized_dense_dir)

    def resize_metadata(path: Path) -> None:
        extrinsics, intrinsics, image_names = read_colmap_model(path)
        write_colmap_model(
            path,
            extrinsics,
            intrinsics,
            image_names,
            (h_original, w_original),
        )

    # Copy and rescale the sparse outputs.
    print("Resizing sparse outputs.")
    output_sparse_path.parent.mkdir(exist_ok=True, parents=True)
    shutil.copytree(resized_sparse_dir, output_sparse_path)
    shutil.rmtree(output_sparse_path / "images")
    shutil.copytree(input_path, output_sparse_path / "images")
    resize_metadata(output_sparse_path / "sparse/0")
    (output_sparse_path / "sparse/0/points3D.bin").unlink()
    shutil.copy(
        resized_sparse_dir / "sparse/0/points3D.bin",
        output_sparse_path / "sparse/0/points3D.bin",
    )

    # Copy and rescale the dense outputs.
    print("Resizing dense outputs.")
    output_dense_path.parent.mkdir(exist_ok=True, parents=True)
    shutil.copytree(resized_dense_dir, output_dense_path)
    shutil.rmtree(output_dense_path / "images")
    shutil.copytree(input_path, output_dense_path / "images")
    resize_metadata(output_dense_path / "sparse/0")
    shutil.rmtree(output_dense_path / "dense/images")
    shutil.copytree(input_path, output_dense_path / "dense/images")
    resize_metadata(output_dense_path / "dense/sparse")


if __name__ == "__main__":
    main()
