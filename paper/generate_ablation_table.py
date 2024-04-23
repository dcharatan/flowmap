# Disable Ruff's rules on star imports for common.
# ruff: noqa: F405, F403

from collections import defaultdict
from pathlib import Path

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
    # Supplemental!
    # METHOD_ABLATION_RANDOM_INITIALIZATION,
    # METHOD_ABLATION_RANDOM_INITIALIZATION_LONG,
    # METHOD_ABLATION_NO_CORRESPONDENCE_WEIGHTS,
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
)

SCENES = (*SCENES_LLFF, *SCENES_MIPNERF360, *SCENES_TANDT, *SCENES_CO3D)

if __name__ == "__main__":
    df = load_metrics(SCENES, METHODS, METRICS)

    # Generate the table grouped by dataset.
    grouped = df.groupby(["dataset_tag", "metric_tag", "method_tag"])[
        "metric_value"
    ].mean()

    rows = defaultdict(list)
    multi_headers = []
    for dataset in DATASETS:
        multi_headers.append((dataset.full_name, len(METRICS)))
        for method in METHODS:
            for metric in METRICS:
                star = "*" if method.requires_intrinsics else ""
                rows[f"{method.full_name}{star}"].append(
                    grouped[dataset.tag][metric.tag][method.tag]
                )

    table = make_latex_table(
        rows,
        [metric.full_name for metric in METRICS] * len(DATASETS),
        [2, 3, 3] * len(DATASETS),
        [metric.order for metric in METRICS] * len(DATASETS),
        multi_headers=multi_headers,
    )

    out_path = OUT_PATH / "ablation_per_dataset.tex"
    out_path.parent.mkdir(exist_ok=True, parents=True)
    with Path(out_path).open("w") as f:
        f.write(table)

    # Generate the table averaged across everything.
    grouped = df.groupby(["metric_tag", "method_tag"])["metric_value"].mean()

    rows = defaultdict(list)
    for method in METHODS:
        for metric in METRICS:
            star = "*" if method.requires_intrinsics else ""
            rows[f"{method.full_name}{star}"].append(grouped[metric.tag][method.tag])

    table = make_latex_table(
        rows,
        [metric.full_name for metric in METRICS],
        [2, 3, 3],
        [metric.order for metric in METRICS],
    )

    out_path = OUT_PATH / "ablation.tex"
    out_path.parent.mkdir(exist_ok=True, parents=True)
    with Path(out_path).open("w") as f:
        f.write(table)
