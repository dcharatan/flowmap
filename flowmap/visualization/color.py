import torch
from einops import rearrange
from jaxtyping import Float
from matplotlib import cm
from torch import Tensor


def apply_color_map(
    x: Float[Tensor, " *batch"],
    color_map: str = "inferno",
) -> Float[Tensor, "*batch 3"]:
    cmap = cm.get_cmap(color_map)

    # Convert to NumPy so that Matplotlib color maps can be used.
    mapped = cmap(x.detach().clip(min=0, max=1).cpu().numpy())[..., :3]

    # Convert back to the original format.
    return torch.tensor(mapped, device=x.device, dtype=torch.float32)


def apply_color_map_to_image(
    image: Float[Tensor, "*batch height width"],
    color_map: str = "inferno",
) -> Float[Tensor, "*batch 3 height with"]:
    image = apply_color_map(image, color_map)
    return rearrange(image, "... h w c -> ... c h w")
