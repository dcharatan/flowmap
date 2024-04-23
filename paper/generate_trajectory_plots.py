from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from einops import einsum
from jaxtyping import Float
from scipy.spatial import procrustes
from sklearn.decomposition import PCA

from flowmap.export.colmap import read_colmap_model

SOURCES = (
    # (name, path, color)
    ("COLMAP", "/nobackup/nvme1/datasets/flowmap/colmap", "#000000"),
    ("FlowMap", "/nobackup/nvme1/charatan/flowmap_paper_outputs/v6", "#E6194B"),
)

SCENES = (
    # (name, whether to manually adjust orientation)
    ("co3d_bench", True),
    ("co3d_hydrant", False),
    ("llff_flower", False),
    ("llff_horns", False),
    ("mipnerf360_bonsai", True),
    ("mipnerf360_garden", False),
    ("tandt_caterpillar", False),
    ("tandt_horse", False),
)

MARGIN = 0
SQUASH = 0.6


def get_rotation(
    points: Float[np.ndarray, "point 3"],
    flip: bool,  # For when PCA doesn't give the desired orientation.
) -> Float[np.ndarray, "3 3"]:
    pca = PCA(n_components=3).fit(points)

    x, y, _ = pca.components_.T
    z = np.cross(x, y)
    y = np.cross(z, x)

    rotation = np.linalg.inv(np.stack([x, y, z]))
    return rotation[[0, 2, 1]] if flip else rotation


if __name__ == "__main__":
    # https://stackoverflow.com/questions/16488182/removing-axes-margins-in-3d-plot
    from mpl_toolkits.mplot3d.axis3d import Axis

    if not hasattr(Axis, "_get_coord_info_old"):

        def _get_coord_info_new(self, renderer):
            mins, maxs, centers, deltas, tc, highs = self._get_coord_info_old(renderer)
            mins += deltas / 4
            maxs -= deltas / 4
            return mins, maxs, centers, deltas, tc, highs

        Axis._get_coord_info_old = Axis._get_coord_info
        Axis._get_coord_info = _get_coord_info_new

    for scene, flip in SCENES:
        # Load the trajectories.
        trajectories = [
            read_colmap_model(Path(path) / scene / "sparse/0")[0][:, :3, 3]
            .detach()
            .cpu()
            .numpy()
            for _, path, _ in SOURCES
        ]

        # Align the trajectories to the first one.
        trajectories[1:] = [procrustes(trajectories[0], t)[1] for t in trajectories[1:]]

        # Scale the first trajectory so it's consistent with the others.
        trajectories[0] = procrustes(trajectories[0], trajectories[1])[0]

        # Figure out the transformation based on the first trajectory.
        rotation = get_rotation(trajectories[0], flip)
        trajectories = [einsum(rotation, t, "i j, p j -> p i") for t in trajectories]

        # Create the plot.
        fig = plt.figure(figsize=(1.18, 1.18), dpi=100)
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        ax.set_proj_type("ortho")
        ax.view_init(elev=30, azim=45)

        # Plot the trajectories.
        for i, trajectory in enumerate(trajectories):
            _, _, color = SOURCES[i]
            ax.plot3D(
                *trajectory.T,
                color=color,
                linewidth=0.5,
                linestyle="--" if i == 0 else "-",
            )

        # Set the axis limits.
        points = np.concatenate(trajectories)
        minima = points.min(axis=0)
        maxima = points.max(axis=0)
        span = (maxima - minima).max() * (1 + MARGIN) * np.array([1, 1, SQUASH])
        means = 0.5 * (maxima + minima)
        starts = means - 0.5 * span
        ends = means + 0.5 * span
        ax.set_xlim(starts[0], ends[0])
        ax.set_ylim(starts[1], ends[1])
        ax.set_zlim(starts[2], ends[2])
        ax.set_aspect("equal")

        # Style the plot.
        ax.set_xticks(np.linspace(starts[0], ends[0], 6))
        ax.set_yticks(np.linspace(starts[1], ends[1], 6))
        ax.set_zticks(np.linspace(starts[2], ends[2], 4))

        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis.set_ticklabels([])
            axis._axinfo["axisline"]["linewidth"] = 0.75
            axis._axinfo["axisline"]["color"] = (0, 0, 0)
            axis._axinfo["grid"]["linewidth"] = 0.25
            axis._axinfo["grid"]["linestyle"] = "-"
            axis._axinfo["grid"]["color"] = (0.8, 0.8, 0.8)
            axis._axinfo["tick"]["inward_factor"] = 0.0
            axis._axinfo["tick"]["outward_factor"] = 0.0
            axis.set_pane_color((1, 1, 1))

        Path("figures/trajectories").mkdir(exist_ok=True, parents=True)
        fig.savefig(f"figures/trajectories/{scene}.svg")
        plt.close(fig)
