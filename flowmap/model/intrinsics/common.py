import torch
from jaxtyping import Float
from torch import Tensor


def focal_lengths_to_intrinsics(
    focal_lengths: Float[Tensor, " *batch"],
    image_shape: tuple[int, int],
) -> Float[Tensor, "*batch 3 3"]:
    device = focal_lengths.device
    h, w = image_shape
    focal_lengths = focal_lengths * (h * w) ** 0.5

    intrinsics = torch.eye(3, dtype=torch.float32, device=device)
    intrinsics[:2, 2] = 0.5
    intrinsics = intrinsics.broadcast_to((*focal_lengths.shape, 3, 3)).contiguous()
    intrinsics[..., 0, 0] = focal_lengths / w  # fx
    intrinsics[..., 1, 1] = focal_lengths / h  # fy

    return intrinsics
