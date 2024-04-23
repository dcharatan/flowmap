from jaxtyping import Float
from torch import Tensor

from .color import apply_color_map_to_image


def color_map_depth(
    depth: Float[Tensor, "batch height width"],
    cmap: str = "inferno",
    invert: bool = True,
) -> Float[Tensor, "batch 3 height width"]:
    depth = depth.log()
    # Normalize the depth.
    near = depth.min()
    far = depth.max()
    depth = (depth - near) / (far - near)
    depth = depth.clip(min=0, max=1)
    if invert:
        depth = 1 - depth
    return apply_color_map_to_image(depth, cmap)
