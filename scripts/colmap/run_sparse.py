import json
import shutil
import subprocess
from pathlib import Path
from time import time
from typing import Literal

import click

Quality = Literal["low", "medium", "high", "extreme"]
DataType = Literal["individual", "video", "internet"]


def run_sparse(
    input_path: Path,
    output_path: Path,
    seed: int | None,
    quality: Quality,
    data_type: DataType,
) -> None:
    output_path.mkdir(exist_ok=True, parents=True)
    start_time = time()

    # Run COLMAP sparse reconstruction.
    seed = seed or 0
    command = (
        "colmap automatic_reconstructor",
        f"--image_path {input_path}",
        f"--workspace_path {output_path}",
        "--sparse 1",
        "--dense 0",
        f"--quality {quality}",
        f"--data_type {data_type}",
        "--camera_model SIMPLE_PINHOLE",
        "--single_camera 1",
        "--use_gpu 1",
        f"--random_seed {seed}",
    )

    if subprocess.run(" ".join(command).split(" ")).returncode != 0:
        raise Exception("COLMAP sparse reconstruction failed.")

    elapsed = time() - start_time
    with (output_path / "runtime.json").open("w") as f:
        json.dump({"runtime": elapsed}, f)

    shutil.copytree(input_path, output_path / "images")


@click.command()
@click.argument("input_path", type=click.Path(path_type=Path))
@click.argument("output_path", type=click.Path(path_type=Path))
@click.option("--seed", type=int, default=None)
def main(
    input_path: Path,
    output_path: Path,
    seed: int | None,
) -> None:
    run_sparse(input_path, output_path, seed, "extreme", "video")


if __name__ == "__main__":
    main()
