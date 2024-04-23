import json
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import pandas as pd
from jaxtyping import Float
from torch import Tensor

from flowmap.export.colmap import read_colmap_model
from flowmap.misc.ate import compute_ate

METRICS_PATH = Path("/mnt/sn850x/flowmap/metrics")
RESULTS_PATH = Path("/mnt/sn850x/flowmap/results")
METRICS_PREFIX = "paper_v17_"


@dataclass(frozen=True)
class Method:
    tag: str
    full_name: str
    requires_intrinsics: bool
    color: str | None = None


METHOD_FLOWMAP = Method("flowmap_ablation_none", "FlowMap", False, "#F033E6")
METHOD_COLMAP = Method("colmap", "COLMAP", False)
METHOD_MVSCOLMAP = Method("mvscolmap", "COLMAP (MVS)", False)
METHOD_DROIDSLAM = Method("droid", "DROID-SLAM", True)
METHOD_NOPENERF = Method("nope", "NoPE-NeRF", True)

METHOD_ABLATION_EXPLICIT_DEPTH = Method(
    "flowmap_ablation_explicit_depth", "Expl. Depth", False, "#E6194B"
)
METHOD_ABLATION_EXPLICIT_FOCAL_LENGTH = Method(
    "flowmap_ablation_explicit_focal_length", "Expl. Focal Length", False, "#4665D8"
)
METHOD_ABLATION_EXPLICIT_EXPLICIT_POSE = Method(
    "flowmap_ablation_explicit_pose", "Expl. Pose", False, "#F58435"
)
METHOD_ABLATION_NO_CORRESPONDENCE_WEIGHTS = Method(
    "flowmap_ablation_no_correspondence_weights", "No Corresp. Weights", False
)
METHOD_ABLATION_NO_TRACKS = Method("flowmap_ablation_no_tracks", "No Tracks", False)
METHOD_ABLATION_RANDOM_INITIALIZATION = Method(
    "flowmap_ablation_random_initialization", "Random Init.", False
)
METHOD_ABLATION_RANDOM_INITIALIZATION_LONG = Method(
    "flowmap_ablation_random_initialization_long", "Random Init. (20k)", False
)
METHOD_ABLATION_SINGLE_STAGE = Method(
    "flowmap_ablation_single_stage", "Single Stage", False, "#3CB44B"
)


@dataclass(frozen=True)
class Metric:
    tag: str
    full_name: str

    # The order determines what the metric means:
    # 1: higher is better
    # 0: there's no concept of ranking
    # -1: lower is better
    order: int


METRIC_PSNR = Metric("psnr", "PSNR", 1)
METRIC_SSIM = Metric("ssim", "SSIM", 1)
METRIC_LPIPS = Metric("lpips", "LPIPS", -1)
METRIC_RUNTIME = Metric("runtime", "Time (min.)", -1)
METRIC_COLMAP_ATE = Metric("ate", "ATE", 0)


@dataclass(frozen=True)
class Dataset:
    tag: str
    full_name: str
    short_name: str


DATASET_TANDT = Dataset("tandt", "Tanks \& Temples", "T\&T")
DATASET_MIPNERF360 = Dataset("mipnerf360", "MipNeRF 360", "MipNeRF 360")
DATASET_LLFF = Dataset("llff", "LLFF", "LLFF")
DATASET_CO3D = Dataset("co3d", "CO3D", "CO3D")


@dataclass(frozen=True)
class Scene:
    tag: str
    full_name: str
    dataset: Dataset


SCENES_LLFF = [
    Scene(f"llff_{name}", name.capitalize(), DATASET_LLFF)
    for name in (
        "fern",
        "flower",
        "fortress",
        "horns",
        "orchids",
        "room",
        "trex",
        # We exclude this FlowMap failure case where FlowMap falls into a hollow-face
        # minimum (inverted geometry).
        # "leaves",
    )
]

SCENES_MIPNERF360 = [
    Scene(f"mipnerf360_{name}", name.capitalize(), DATASET_MIPNERF360)
    for name in (
        "bonsai",
        "kitchen",
        "counter",
        # We exclude the garden scene because even though it's somewhat video-like in
        # that it has a consistent trajectory, the visual distance between frames is
        # very large, making tracking unreliable.
        # "garden",
    )
]

SCENES_TANDT = [
    Scene(f"tandt_{name}", name.capitalize(), DATASET_TANDT)
    for name in (
        "barn",
        "caterpillar",
        "church",
        "courthouse",
        "family",
        "francis",
        "horse",
        "ignatius",
        "m60",
        "museum",
        "panther",
        "playground",
        "train",
        "truck",
        # We exclude these scenes where COLMAP fails. We define failure as being unable
        # to create a single sparse model/reconstruction.
        # "meetingroom",
        # "courtroom",
        # "ballroom",
        # We exclude these scenes where FlowMap fails. We define failure as having bad
        # optical flow/tracking input or falling into a hollow-face minimum.
        # "lighthouse",
        # "temple",
        # We exclude this scene where both FlowMap and COLMAP fail.
        # "auditorium",
    )
]

SCENES_CO3D = [
    Scene(f"co3d_{name}", name.capitalize(), DATASET_CO3D)
    for name in ("bench", "hydrant")  # demo scenes
]


@cache
def load_trajectory(method: Method, scene: Scene) -> Float[Tensor, "point 3"]:
    path = RESULTS_PATH / method.tag / scene.tag / "sparse/0"
    extrinsics, _, _ = read_colmap_model(path)
    return extrinsics[:, :3, 3]


def load_metrics(
    scenes: Iterable[Scene],
    methods: Iterable[Method],
    metrics: Iterable[Metric],
) -> pd.DataFrame:
    metrics = tuple(metrics)

    data = defaultdict(list)

    for method in methods:
        for scene in scenes:
            # Load the scene's metrics.
            try:
                with (
                    METRICS_PATH
                    / f"{METRICS_PREFIX}{method.tag}_{scene.tag}/metrics.json"
                ).open("r") as f:
                    scene_metrics = json.load(f)
            except FileNotFoundError:
                scene_metrics = {}

            # Load the runtime separately.
            try:
                # NoPE-NeRF runtimes are different since it combines view synthesis with
                # camera pose estimation.
                if method == METHOD_NOPENERF:
                    with (
                        METRICS_PATH
                        / f"{METRICS_PREFIX}{method.tag}_{scene.tag}/runtime.json"
                    ).open("r") as f:
                        scene_metrics["runtime"] = json.load(f)["runtime"]
                else:
                    with (RESULTS_PATH / method.tag / scene.tag / "runtime.json").open(
                        "r"
                    ) as f:
                        scene_metrics["runtime"] = json.load(f)["runtime"]

                # Convert seconds to minutes.
                scene_metrics["runtime"] = scene_metrics["runtime"] / 60
            except FileNotFoundError:
                pass

            if METRIC_COLMAP_ATE in metrics:
                if method in (METHOD_COLMAP, METHOD_MVSCOLMAP):
                    scene_metrics["ate"] = None
                else:
                    try:
                        gt = load_trajectory(METHOD_COLMAP, scene)
                        hat = load_trajectory(method, scene)
                        scene_metrics["ate"] = compute_ate(gt, hat)[0]
                    except FileNotFoundError:
                        scene_metrics["ate"] = None

            # Select the metrics we care about.
            for metric in metrics:
                data["scene_tag"].append(scene.tag)
                data["scene_full_name"].append(scene.full_name)

                data["dataset_tag"].append(scene.dataset.tag)
                data["dataset_full_name"].append(scene.dataset.full_name)

                data["method_tag"].append(method.tag)
                data["method_full_name"].append(method.full_name)

                data["metric_tag"].append(metric.tag)
                data["metric_full_name"].append(metric.full_name)
                data["metric_value"].append(scene_metrics.get(metric.tag, None))

    return pd.DataFrame(data)
