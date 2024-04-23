import json
import shutil
import subprocess
from pathlib import Path
from time import time

import click


def run_dense(input_path: Path, output_path: Path) -> None:
    if (input_path / "sparse/1").exists():
        raise FileExistsError(
            "Make sure only a single sparse model exists! If there's more than one "
            "model, sparse reconstruction failed."
        )

    # Copy the input path to the output path.
    shutil.copytree(input_path, output_path)

    start_time = time()

    # Run the dense reconstruction commands.
    command = (
        "colmap image_undistorter",
        f"--image_path {output_path}/images",
        f"--input_path {output_path}/sparse/0",
        f"--output_path {output_path}/dense",
        "--output_type COLMAP",
    )
    if subprocess.run(" ".join(command).split(" ")).returncode != 0:
        raise Exception("COLMAP dense reconstruction failed.")

    command = (
        "colmap patch_match_stereo",
        f"--workspace_path {output_path}/dense",
        "--workspace_format COLMAP",
        "--PatchMatchStereo.geom_consistency true",
    )
    if subprocess.run(" ".join(command).split(" ")).returncode != 0:
        raise Exception("COLMAP dense reconstruction failed.")

    command = (
        "colmap stereo_fusion",
        f"--workspace_path {output_path}/dense",
        "--workspace_format COLMAP",
        "--input_type geometric",
        f"--output_path {output_path}/dense/fused.ply",
    )
    if subprocess.run(" ".join(command).split(" ")).returncode != 0:
        raise Exception("COLMAP dense reconstruction failed.")

    # Copy the dense point cloud to the right location for 3D Gaussian Splatting.
    shutil.copy(output_path / "dense/fused.ply", output_path / "sparse/0/points3D.ply")

    elapsed = time() - start_time
    with (output_path / "runtime.json").open("w") as f:
        json.dump({"runtime": elapsed}, f)


@click.command()
@click.argument("input_path", type=click.Path(path_type=Path))
@click.argument("output_path", type=click.Path(path_type=Path))
def main(input_path: Path, output_path: Path) -> None:
    run_dense(input_path, output_path)


if __name__ == "__main__":
    main()
