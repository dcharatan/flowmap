from dataclasses import dataclass, replace

import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float
from PIL import Image
from torch import Tensor

from ..dataset.types import Batch


@dataclass
class CroppingCfg:
    image_shape: tuple[int, int] | int
    flow_scale_multiplier: int
    patch_size: int


def resize_batch(batch: Batch, shape: tuple[int, int]) -> Batch:
    b, f, _, _, _ = batch.videos.shape
    h, w = shape

    videos = rearrange(batch.videos, "b f c h w -> (b f) c h w")
    videos = F.interpolate(videos, (h, w), mode="bilinear", align_corners=False)
    videos = rearrange(videos, "(b f) c h w -> b f c h w", b=b, f=f)

    return replace(batch, videos=videos)


def compute_patch_cropped_shape(
    shape: tuple[int, int],
    patch_size: int,
) -> tuple[int, int]:
    h, w = shape

    h_new = (h // patch_size) * patch_size
    w_new = (w // patch_size) * patch_size

    return h_new, w_new


def center_crop_images(
    images: Float[Tensor, "*batch channel height width"],
    new_shape: tuple[int, int],
) -> Float[Tensor, "*batch channel cropped_height cropped_width"]:
    *_, h, w = images.shape
    h_new, w_new = new_shape
    row = (h - h_new) // 2
    col = (w - w_new) // 2
    return images[..., row : row + h_new, col : col + w_new]


def center_crop_intrinsics(
    intrinsics: Float[Tensor, "*#batch 3 3"] | None,
    old_shape: tuple[int, int],
    new_shape: tuple[int, int],
) -> Float[Tensor, "*batch 3 3"] | None:
    """Modify the given intrinsics to account for center cropping."""

    if intrinsics is None:
        return None

    h_old, w_old = old_shape
    h_new, w_new = new_shape
    intrinsics = intrinsics.clone()
    intrinsics[..., 0, 0] *= w_old / w_new  # fx
    intrinsics[..., 1, 1] *= h_old / h_new  # fy
    return intrinsics


def patch_crop_batch(batch: Batch, patch_size: int) -> Batch:
    _, _, _, h, w = batch.videos.shape
    old_shape = (h, w)
    new_shape = compute_patch_cropped_shape((h, w), patch_size)
    return replace(
        batch,
        intrinsics=center_crop_intrinsics(batch.intrinsics, old_shape, new_shape),
        videos=center_crop_images(batch.videos, new_shape),
    )


def get_image_shape(
    original_shape: tuple[int, int],
    cfg: CroppingCfg,
) -> tuple[int, int]:
    # If the image shape is exact, return it.
    if isinstance(cfg.image_shape, tuple):
        return cfg.image_shape

    # Otherwise, the image shape is assumed to be an approximate number of pixels.
    h, w = original_shape
    scale = (cfg.image_shape / (h * w)) ** 0.5
    return (round(h * scale), round(w * scale))


def crop_and_resize_batch_for_model(
    batch: Batch,
    cfg: CroppingCfg,
) -> tuple[Batch, tuple[int, int]]:
    # Resize the batch to the desired model input size.
    image_shape = get_image_shape(tuple(batch.videos.shape[-2:]), cfg)
    batch = resize_batch(batch, image_shape)

    # Record the pre-cropping shape.
    _, _, _, h, w = batch.videos.shape

    # Center-crop the batch so it's cleanly divisible by the patch size.
    return patch_crop_batch(batch, cfg.patch_size), (h, w)


def crop_and_resize_batch_for_flow(batch: Batch, cfg: CroppingCfg) -> Batch:
    # Figure out the image size that's used for flow.
    image_shape = get_image_shape(tuple(batch.videos.shape[-2:]), cfg)
    flow_shape = tuple(dim * cfg.flow_scale_multiplier for dim in image_shape)

    # Resize the batch to match the desired flow shape.
    batch = resize_batch(batch, flow_shape)

    # Center-crop the batch so it's cleanly divisible by the patch size times the flow
    # multiplier. This ensures that the aspect ratio matches the model input's aspect
    # ratio.
    return patch_crop_batch(batch, cfg.patch_size * cfg.flow_scale_multiplier)


def resize_to_cover(
    image: Image.Image,
    shape: tuple[int, int],
) -> tuple[
    Image.Image,  # the image itself
    tuple[int, int],  # image shape after scaling, before cropping
]:
    w_old, h_old = image.size
    h_new, w_new = shape

    # Figure out the scale factor needed to cover the desired shape with a uniformly
    # scaled version of the input image. Then, resize the input image.
    scale_factor = max(h_new / h_old, w_new / w_old)
    h_scaled = round(h_old * scale_factor)
    w_scaled = round(w_old * scale_factor)
    image_scaled = image.resize((w_scaled, h_scaled), Image.LANCZOS)

    # Center-crop the image.
    x = (w_scaled - w_new) // 2
    y = (h_scaled - h_new) // 2
    image_cropped = image_scaled.crop((x, y, x + w_new, y + h_new))
    return image_cropped, (h_scaled, w_scaled)


def resize_to_cover_with_intrinsics(
    images: list[Image.Image],
    shape: tuple[int, int],
    intrinsics: Float[Tensor, "*batch 3 3"] | None,
) -> tuple[
    list[Image.Image],  # cropped images
    Float[Tensor, "*batch 3 3"] | None,  # intrinsics, adjusted for cropping
]:
    scaled_images = []
    for image in images:
        image, old_shape = resize_to_cover(image, shape)
        scaled_images.append(image)

    if intrinsics is not None:
        intrinsics = center_crop_intrinsics(intrinsics, old_shape, shape)

    return scaled_images, intrinsics
