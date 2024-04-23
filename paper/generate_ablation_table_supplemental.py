# Disable Ruff's rules on star imports for common.
# ruff: noqa: F405, F403

from collections import defaultdict
from pathlib import Path

from jaxtyping import install_import_hook

with install_import_hook(
    ("flowmap", "paper"),
    ("beartype", "beartype"),
):
    from .common import *
    from .table import make_latex_table

OUT_PATH = Path("tables")

METHODS = (
    METHOD_FLOWMAP,
    METHOD_ABLATION_SINGLE_STAGE,
    METHOD_ABLATION_EXPLICIT_FOCAL_LENGTH,
    METHOD_ABLATION_EXPLICIT_DEPTH,
    METHOD_ABLATION_EXPLICIT_EXPLICIT_POSE,
    METHOD_ABLATION_NO_TRACKS,
    METHOD_ABLATION_RANDOM_INITIALIZATION,
    METHOD_ABLATION_RANDOM_INITIALIZATION_LONG,
    METHOD_ABLATION_NO_CORRESPONDENCE_WEIGHTS,
)

DATASETS = (
    DATASET_LLFF,
    DATASET_MIPNERF360,
    DATASET_TANDT,
    DATASET_CO3D,
)

METRICS = (
    METRIC_PSNR,
    METRIC_SSIM,
    METRIC_LPIPS,
    METRIC_RUNTIME,
    METRIC_COLMAP_ATE,
)

SCENES = (*SCENES_LLFF, *SCENES_MIPNERF360, *SCENES_TANDT, *SCENES_CO3D)
SCENES_PER_ROW = 3

if __name__ == "__main__":
    chunks = []
    df = load_metrics(SCENES, METHODS, METRICS)
    grouped = df.groupby(["scene_tag", "metric_tag", "method_tag"])[
        "metric_value"
    ].mean()

    for i in range(0, len(SCENES), SCENES_PER_ROW):
        row_scenes = SCENES[i : i + SCENES_PER_ROW]
        rows = defaultdict(list)
        multi_headers = []
        for scene in row_scenes:
            multi_headers.append(
                (f"{scene.full_name} ({scene.dataset.full_name})", len(METRICS))
            )
            for method in METHODS:
                for metric in METRICS:
                    star = "*" if method.requires_intrinsics else ""
                    rows[f"{method.full_name}{star}"].append(
                        grouped[scene.tag][metric.tag][method.tag]
                    )

        table = make_latex_table(
            rows,
            [metric.full_name for metric in METRICS] * len(row_scenes),
            [2, 3, 3, 1, 5] * len(row_scenes),
            [metric.order for metric in METRICS] * len(row_scenes),
            multi_headers=multi_headers,
        )
        chunks.append(table)

    out_path = OUT_PATH / "ablation_supplemental.tex"
    out_path.parent.mkdir(exist_ok=True, parents=True)
    with Path(out_path).open("w") as f:
        f.write("\n".join(chunks))
