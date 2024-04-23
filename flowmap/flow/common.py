from einops import rearrange
from jaxtyping import Float
from torch import Tensor


def split_videos(
    videos: Float[Tensor, "batch frame 3 height width"],
) -> tuple[
    Float[Tensor, "batch*(frame-1) 3 height width"],  # source (flattened batch dims)
    Float[Tensor, "batch*(frame-1) 3 height width"],  # target (flattened batch dims)
    int,  # batch
    int,  # frame
]:
    b, f, _, _, _ = videos.shape
    return (
        rearrange(videos[:, :-1], "b f c h w -> (b f) c h w"),
        rearrange(videos[:, 1:], "b f c h w -> (b f) c h w"),
        b,
        f,
    )
