from functools import partial
from pathlib import Path
from typing import Callable

import svg
import torch
from einops import einsum, rearrange
from flow_vis_torch import flow_to_color
from jaxtyping import Bool, Float, Int64, install_import_hook
from torch import Tensor
from tqdm import tqdm

with install_import_hook(
    ("flowmap", "paper"),
    ("beartype", "beartype"),
):
    from flowmap.flow.flow_predictor_raft import FlowPredictorRaft, FlowPredictorRaftCfg
    from flowmap.misc.image_io import save_image
    from flowmap.model.model import ModelExports
    from flowmap.model.projection import (
        homogenize_points,
        project_camera_space,
        sample_image_grid,
        unproject,
    )
    from flowmap.visualization.depth import color_map_depth
    from flowmap.visualization.drawing import draw_lines, draw_points

    from .colors import DISTINCT_COLORS, DISTINCT_COLORS_TORCH, to_hex
    from .point_clouds.birds import LoadedScene, load_birds
    from .svg_tools import save_svg

CANVAS_SIZE = 2800
SCALE = 400.0
BATCH_SIZE = 1024
BACKGROUND_POINT_PROBABILITY = 0.0075
SECOND_FRAME = 4
POINT_RADIUS = 6


def get_frustums(
    exports: ModelExports,
    z_value: float,
) -> Float[Tensor, "frame endpoint=2 line=8 xyz=3"]:
    # Generate xy points at the corners.
    xy, _ = sample_image_grid((2, 2), device=device)
    xy = xy * 2 - 0.5

    # Un-project the corners to the specified Z value.
    rays = unproject(
        xy,
        torch.ones_like(xy[..., 0]),
        rearrange(exports.intrinsics, "b f i j -> b f () () i j"),
    )
    rays = rays / rays[..., -1:] * z_value

    # Add the camera origin.
    rays = rearrange(rays[0, 0], "h w xyz -> (h w) xyz")
    rays = torch.cat((torch.zeros_like(rays[:1]), rays), dim=0)

    # Convert the points into world space.
    rays = einsum(
        exports.extrinsics[0],
        homogenize_points(rays),
        "f i j, p j -> f p i",
    )[..., :3]

    # Aggregate the lines needed to make frustums.
    o, a, b, c, d = rays.unbind(dim=-2)
    lines = [
        [a, b],
        [b, d],
        [d, c],
        [c, a],
        [o, c],
        [o, a],
        [o, b],
        [o, d],
    ]
    return torch.stack([torch.stack(line, dim=1) for line in lines], dim=-2)


def project(
    points: Float[Tensor, "*batch xyz=3"],
    midpoint: Float[Tensor, "xyz=3"],
    scale: float,
    canvas_size: float | int,
) -> tuple[
    Float[Tensor, "*batch xy=2"],  # projected (x, y)
    Int64[Tensor, " *batch"],  # index (in depth ordering)
]:
    # Project the points (isometric).
    projection = [
        [2, 1],
        [0, 2.25],
        [2, -1],
    ]
    projection = torch.tensor(projection, dtype=torch.float32, device=device)
    xy = einsum(projection, points - midpoint, "i j, ... i -> ... j")
    xy = xy * scale + 0.5 * canvas_size

    # Figure out the correct depth ordering.
    look = torch.tensor((1, -1, -1), dtype=torch.float32, device=device)
    depth = einsum(points, look, "... xyz, xyz -> ...")
    ordering = depth.view(-1).argsort().view(depth.shape)

    return xy, ordering


def render_frustums(
    birds: LoadedScene,
    proj: Callable,
) -> None:
    # Save the frustums to .SVG for use in Figma.
    for z_value, tag in zip((birds.cutoff, 0.25 * birds.cutoff), ("large", "small")):
        frustums = get_frustums(birds.exports, z_value)
        xy_frustums, _ = proj(frustums)
        for index, frustum in enumerate(xy_frustums):
            fig = svg.SVG(
                width=CANVAS_SIZE,
                height=CANVAS_SIZE,
                elements=[],
                viewBox=svg.ViewBoxSpec(0, 0, CANVAS_SIZE, CANVAS_SIZE),
            )
            for start, end in rearrange(frustum, "e l xy -> l e xy").tolist():
                line = svg.Line(
                    x1=start[0],
                    y1=start[1],
                    x2=end[0],
                    y2=end[1],
                    stroke="#000000",
                    stroke_width=4,
                    stroke_linecap="round",
                )
                fig.elements.append(line)

            save_svg(fig, Path(f"figures/frustums/{index:0>2}_{tag}.svg"))


def render_point_clouds(
    device: torch.device,
    birds: LoadedScene,
    proj: Callable,
) -> None:
    for index, (xyz, mask, color) in enumerate(
        zip(birds.xyz_camera_space, birds.mask, birds.exports.colors[0])
    ):
        canvas = torch.zeros(
            (4, CANVAS_SIZE, CANVAS_SIZE),
            dtype=torch.float32,
            device=device,
        )
        xy_pts, ordering_pts = proj(xyz[mask])
        color_pts = rearrange(color, "c h w -> h w c")[mask]
        xy_pts = xy_pts[ordering_pts]
        color_pts = color_pts[ordering_pts]
        for xy_batch, color_batch in zip(
            xy_pts.split(BATCH_SIZE),
            tqdm(color_pts.split(BATCH_SIZE)),
        ):
            canvas = draw_points(
                canvas,
                xy_batch,
                color_batch,
                radius=POINT_RADIUS,
            )
        save_image(canvas, f"figures/point_cloud_frame_{index:0>2}.png")


def render_background_scene_flow(
    device: torch.device,
    birds: LoadedScene,
    proj: Callable,
    background_mask: Bool[Tensor, "height width"],
) -> None:
    # Render the background points and lines.
    mask = background_mask & (~birds.highlight_mask)
    xyz_background_original = birds.xyz_camera_space[0][mask]
    num_points = xyz_background_original.shape[0]
    xyz_background_other = einsum(
        birds.exports.extrinsics[0, SECOND_FRAME].inverse(),
        homogenize_points(xyz_background_original),
        "i j, p j -> p i",
    )[..., :3]
    xyz_background = torch.cat((xyz_background_original, xyz_background_other), dim=0)
    is_other_background = (
        torch.zeros((num_points,), dtype=torch.bool, device=device),
        torch.ones((num_points,), dtype=torch.bool, device=device),
    )
    is_other_background = torch.cat(is_other_background, dim=0)
    xy_background, ordering_background = proj(xyz_background)
    lines_background = rearrange(xy_background, "(e l) xy -> l e xy", e=2)
    xy_background = xy_background[ordering_background]
    is_other_background = is_other_background[ordering_background]

    # Export to SVG.
    fig = svg.SVG(
        width=CANVAS_SIZE,
        height=CANVAS_SIZE,
        elements=[],
        viewBox=svg.ViewBoxSpec(0, 0, CANVAS_SIZE, CANVAS_SIZE),
    )

    # Draw the connecting lines.
    for start, end in lines_background.tolist():
        line = svg.Line(
            x1=start[0],
            y1=start[1],
            x2=end[0],
            y2=end[1],
            stroke="#cccccc",
            stroke_width=4,
            stroke_linecap="round",
        )
        fig.elements.append(line)

    # Draw the endpoints.
    for (x, y), is_background in zip(xy_background.tolist(), is_other_background):
        line = svg.Line(
            x1=x,
            y1=y,
            x2=x,
            y2=y,
            stroke=DISTINCT_COLORS[1 if is_background else 0],
            stroke_width=12,
            stroke_linecap="round",
        )
        fig.elements.append(line)

    save_svg(fig, Path("figures/scene_flow/background_lines.svg"))


def render_highlighted_scene_flow(
    birds: LoadedScene,
    proj: Callable,
) -> None:
    # Render the background points and lines.
    xyz_original = birds.xyz_camera_space[0][birds.highlight_mask]
    xyz_other = einsum(
        birds.exports.extrinsics[0, SECOND_FRAME].inverse(),
        homogenize_points(xyz_original),
        "i j, p j -> p i",
    )[..., :3]
    xy_original, _ = proj(xyz_original)
    xy_other, _ = proj(xyz_other)

    # Export to SVG.
    fig = svg.SVG(
        width=CANVAS_SIZE,
        height=CANVAS_SIZE,
        elements=[],
        viewBox=svg.ViewBoxSpec(0, 0, CANVAS_SIZE, CANVAS_SIZE),
    )

    # Draw the outlines.
    for start, end in zip(xy_original.tolist(), xy_other.tolist()):
        line = svg.Line(
            x1=start[0],
            y1=start[1],
            x2=end[0],
            y2=end[1],
            stroke="#ffffff",
            stroke_width=48,
            stroke_linecap="round",
        )
        fig.elements.append(line)
    for x, y in xy_other.tolist():
        line = svg.Line(
            x1=x,
            y1=y,
            x2=x,
            y2=y,
            stroke="#ffffff",
            stroke_width=80,
            stroke_linecap="round",
        )
        fig.elements.append(line)
    for x, y in xy_original.tolist():
        line = svg.Line(
            x1=x,
            y1=y,
            x2=x,
            y2=y,
            stroke="#ffffff",
            stroke_width=80,
            stroke_linecap="round",
        )
        fig.elements.append(line)

    # Draw the connecting lines.
    for start, end in zip(xy_original.tolist(), xy_other.tolist()):
        line = svg.Line(
            x1=start[0],
            y1=start[1],
            x2=end[0],
            y2=end[1],
            stroke="#cccccc",
            stroke_width=32,
            stroke_linecap="round",
        )
        fig.elements.append(line)

    # Draw the endpoints.
    for x, y in xy_other.tolist():
        line = svg.Line(
            x1=x,
            y1=y,
            x2=x,
            y2=y,
            stroke=DISTINCT_COLORS[1],
            stroke_width=64,
            stroke_linecap="round",
        )
        fig.elements.append(line)
    for x, y in xy_original.tolist():
        line = svg.Line(
            x1=x,
            y1=y,
            x2=x,
            y2=y,
            stroke=DISTINCT_COLORS[0],
            stroke_width=64,
            stroke_linecap="round",
        )
        fig.elements.append(line)

    save_svg(fig, Path("figures/scene_flow/highlight_lines.svg"))


def render_point_cloud_silhouettes(
    device: torch.device,
    birds: LoadedScene,
    proj: Callable,
) -> None:
    xyz_original = birds.xyz_camera_space[0][birds.mask[0]]
    xyz_other = einsum(
        birds.exports.extrinsics[0, SECOND_FRAME].inverse(),
        homogenize_points(xyz_original),
        "i j, p j -> p i",
    )[..., :3]

    for xyz, tag, color in zip(
        (xyz_original, xyz_other),
        ("original", "other"),
        (DISTINCT_COLORS_TORCH[0], DISTINCT_COLORS_TORCH[1]),
    ):
        canvas = torch.zeros(
            (4, CANVAS_SIZE, CANVAS_SIZE),
            dtype=torch.float32,
            device=device,
        )
        xy, ordering = proj(xyz)
        xy = xy[ordering]
        for xy_batch in xy.split(BATCH_SIZE):
            canvas = draw_points(
                canvas,
                xy_batch,
                color,
                radius=POINT_RADIUS,
            )
        save_image(canvas, f"figures/silhouettes/{tag}.png")


def render_scene_flow_projection(
    device: torch.device,
    birds: LoadedScene,
    background_mask: Bool[Tensor, "height width"],
    canvas_size: int = 1024,
) -> None:
    canvas = torch.ones(
        (3, canvas_size, canvas_size), dtype=torch.float32, device=device
    )

    for mask, line_color, line_width, point_width, is_outlined in zip(
        (background_mask, birds.highlight_mask),
        ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        (2, 24),
        (12, 48),
        (False, True),
    ):
        # Render the background points and lines.
        xyz_original = birds.xyz_camera_space[0][mask]
        xyz_other = einsum(
            birds.exports.extrinsics[0, SECOND_FRAME].inverse(),
            homogenize_points(xyz_original),
            "i j, p j -> p i",
        )[..., :3]

        xy_original = project_camera_space(xyz_original, birds.exports.intrinsics[0, 0])
        xy_other = project_camera_space(xyz_other, birds.exports.intrinsics[0, 0])

        if is_outlined:
            canvas = draw_lines(
                canvas,
                xy_original,
                xy_other,
                (1, 1, 1),
                line_width + 8,
                x_range=(0, 1),
                y_range=(0, 1),
            )
        canvas = draw_lines(
            canvas,
            xy_original,
            xy_other,
            line_color,
            line_width,
            x_range=(0, 1),
            y_range=(0, 1),
        )
        if is_outlined:
            canvas = draw_points(
                canvas,
                xy_original,
                (1, 1, 1),
                (point_width + 8) // 2,
                x_range=(0, 1),
                y_range=(0, 1),
            )
            canvas = draw_points(
                canvas,
                xy_other,
                (1, 1, 1),
                (point_width + 8) // 2,
                x_range=(0, 1),
                y_range=(0, 1),
            )
        canvas = draw_points(
            canvas,
            xy_original,
            DISTINCT_COLORS_TORCH[0],
            point_width // 2,
            x_range=(0, 1),
            y_range=(0, 1),
        )
        canvas = draw_points(
            canvas,
            xy_other,
            DISTINCT_COLORS_TORCH[1],
            point_width // 2,
            x_range=(0, 1),
            y_range=(0, 1),
        )

    save_image(canvas, "figures/projected_flow.png")


def render_depths(birds: LoadedScene) -> None:
    images = color_map_depth(birds.exports.depths[0])
    for index, image in enumerate(images):
        save_image(image, f"figures/depths/depth_{index:0>2}.png")


def render_flows(device: torch.device, birds: LoadedScene) -> None:
    # We just want the flow between images 0 and 4.
    raft = FlowPredictorRaft(FlowPredictorRaftCfg("raft", 32, 8)).to(device)
    flow = raft.forward(
        birds.exports.colors[0, :1],
        birds.exports.colors[0, SECOND_FRAME : SECOND_FRAME + 1],
    )
    images = flow_to_color(rearrange(flow, "b h w xy -> b xy h w")) / 255
    save_image(images[0], "figures/flow_0_to_4.png")

    # Render a little legend thing
    x = torch.linspace(-1, 1, 256, device=flow.device)
    y = torch.linspace(-1, 1, 256, device=flow.device)
    key = torch.stack(torch.meshgrid((x, y), indexing="xy"), dim=0)
    save_image(flow_to_color(key) / 255, "figures/flow_key.png")


def render_joint_point_cloud(
    device: torch.device,
    birds: LoadedScene,
    proj: Callable,
) -> None:
    # Combine the points from frames 0 and 4.
    xyz = (
        birds.xyz_world_space[0][birds.mask[0]],
        birds.xyz_world_space[SECOND_FRAME][birds.mask[SECOND_FRAME]],
    )
    xyz = torch.cat(xyz)
    color = (
        rearrange(birds.exports.colors[0, 0], "c h w -> h w c")[birds.mask[0]],
        rearrange(birds.exports.colors[0, SECOND_FRAME], "c h w -> h w c")[
            birds.mask[SECOND_FRAME]
        ],
    )
    color = torch.cat(color)

    canvas = torch.zeros(
        (4, CANVAS_SIZE, CANVAS_SIZE),
        dtype=torch.float32,
        device=device,
    )
    xy, ordering = proj(xyz)
    xy = xy[ordering]
    color = color[ordering]
    for xy_batch, color_batch in zip(
        xy.split(BATCH_SIZE),
        tqdm(color.split(BATCH_SIZE)),
    ):
        canvas = draw_points(
            canvas,
            xy_batch,
            color_batch,
            radius=POINT_RADIUS,
        )
    save_image(canvas, "figures/point_cloud_joint.png")


if __name__ == "__main__":
    device = torch.device("cuda")
    birds = load_birds(device)
    proj = partial(
        project, midpoint=birds.midpoint, scale=SCALE, canvas_size=CANVAS_SIZE
    )

    # Determine which points to use for background flow visualization.
    g = torch.Generator(device)
    g.manual_seed(123)
    r = torch.rand(birds.mask[0].shape, device=device, dtype=torch.float32, generator=g)
    background_mask = (r < BACKGROUND_POINT_PROBABILITY) & birds.mask[0]

    # render_joint_point_cloud(device, birds, proj)
    # render_flows(device, birds)
    # render_depths(birds)
    # render_frustums(birds, proj)
    # render_point_clouds(device, birds, proj)
    # render_joint_point_cloud(device, birds, proj)
    # render_background_scene_flow(device, birds, proj, background_mask)
    # render_highlighted_scene_flow(birds, proj)
    # render_point_cloud_silhouettes(device, birds, proj)
    render_scene_flow_projection(device, birds, background_mask)
