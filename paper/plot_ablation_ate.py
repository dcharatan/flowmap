# Disable Ruff's rules on star imports for common.
# ruff: noqa: F405, F403

from pathlib import Path

import matplotlib.pyplot as plt

from .common import *

METHODS = (
    METHOD_FLOWMAP,
    METHOD_ABLATION_EXPLICIT_DEPTH,
    METHOD_ABLATION_EXPLICIT_EXPLICIT_POSE,
    METHOD_ABLATION_EXPLICIT_FOCAL_LENGTH,
    METHOD_ABLATION_SINGLE_STAGE,
)

if __name__ == "__main__":
    fig, ax = plt.subplots(figsize=(5, 2))

    for method in METHODS:
        tag = method.tag.replace("flowmap_", "")  # whoops
        with Path(f"ates/{tag}.json").open("r") as f:
            ate = json.load(f)

        assert method.color is not None
        ax.plot(ate, color=method.color)

    ax.set_yscale("log")
    ax.grid(axis="y", which="major", color="#eee")
    ax.grid(axis="y", which="minor", color="#eee")

    ax.set_xlim(0, 2000)

    ax.set_xticks([0, 400, 800, 1200, 1600, 2000])

    fig.savefig("figures/ablation_ate_plot.svg")
