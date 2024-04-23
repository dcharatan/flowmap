import torch
from einops import einsum
from jaxtyping import Float
from torch import Tensor


def align_rigid(
    p: Float[Tensor, "*batch point 3"],
    q: Float[Tensor, "*batch point 3"],
    weights: Float[Tensor, "*batch point"],
) -> Float[Tensor, "*batch 4 4"]:
    """Compute a rigid transformation that, when applied to p, minimizes the weighted
    squared distance between transformed points in p and points in q. See "Least-Squares
    Rigid Motion Using SVD" by Olga Sorkine-Hornung and Michael Rabinovich for more
    details (https://igl.ethz.ch/projects/ARAP/svd_rot.pdf).
    """

    device = p.device
    dtype = p.dtype
    *batch, _, _ = p.shape

    # 1. Compute the centroids of both point sets.
    weights_normalized = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)
    p_centroid = (weights_normalized[..., None] * p).sum(dim=-2)
    q_centroid = (weights_normalized[..., None] * q).sum(dim=-2)

    # 2. Compute the centered vectors.
    p_centered = p - p_centroid[..., None, :]
    q_centered = q - q_centroid[..., None, :]

    # 3. Compute the 3x3 covariance matrix.
    covariance = (q_centered * weights[..., None]).transpose(-1, -2) @ p_centered

    # 4. Compute the singular value decomposition and then the rotation.
    u, _, vt = torch.linalg.svd(covariance)
    s = torch.eye(3, dtype=dtype, device=device)
    s = s.expand((*batch, 3, 3)).contiguous()
    s[..., 2, 2] = (u.det() * vt.det()).sign()
    rotation = u @ s @ vt

    # 5. Compute the optimal translation.
    translation = q_centroid - einsum(rotation, p_centroid, "... i j, ... j -> ... i")

    # Compose the results into a single transformation matrix.
    shape = (*rotation.shape[:-2], 4, 4)
    r = torch.eye(4, dtype=torch.float32, device=device).expand(shape).contiguous()
    r[..., :3, :3] = rotation
    t = torch.eye(4, dtype=torch.float32, device=device).expand(shape).contiguous()
    t[..., :3, 3] = translation

    return t @ r
