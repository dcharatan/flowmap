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
    METHOD_COLMAP,
    METHOD_MVSCOLMAP,
    METHOD_DROIDSLAM,
    METHOD_NOPENERF,
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

SCENES = (
    (*SCENES_LLFF, *SCENES_MIPNERF360),
    (*SCENES_TANDT, *SCENES_CO3D),
)


if __name__ == "__main__":
    chunks = []
    for row_scenes in SCENES:
        df = load_metrics(row_scenes, METHODS, METRICS)
        row_datasets = tuple(set(scene.dataset for scene in row_scenes))

        grouped = df.groupby(["dataset_tag", "metric_tag", "method_tag"])[
            "metric_value"
        ].mean()

        rows = defaultdict(list)
        multi_headers = []
        for dataset in row_datasets:
            num_scenes = sum([scene.dataset.tag == dataset.tag for scene in row_scenes])
            multi_headers.append(
                (f"{dataset.full_name} ({num_scenes} scenes)", len(METRICS))
            )
            for method in METHODS:
                for metric in METRICS:
                    star = "*" if method.requires_intrinsics else ""
                    rows[f"{method.full_name}{star}"].append(
                        grouped[dataset.tag][metric.tag][method.tag]
                    )

        table = make_latex_table(
            rows,
            [metric.full_name for metric in METRICS] * len(row_datasets),
            [2, 3, 3, 1, 5] * len(row_datasets),
            [metric.order for metric in METRICS] * len(row_datasets),
            multi_headers=multi_headers,
        )
        chunks.append(table)

    out_path = OUT_PATH / "main_comparison.tex"
    out_path.parent.mkdir(exist_ok=True, parents=True)
    with Path(out_path).open("w") as f:
        f.write("\n".join(chunks))
