import torch
from jaxtyping import Float
from scipy import spatial
from torch import Tensor


def compute_ate(
    gt: Float[Tensor, "point 3"],
    predicted: Float[Tensor, "point 3"],
) -> tuple[
    Float[Tensor, ""],  # ate
    Float[Tensor, "point 3"],  # aligned gt
    Float[Tensor, "point 3"],  # aligned predicted
]:
    aligned_gt, aligned_predicted, _ = spatial.procrustes(
        gt.detach().cpu().numpy(),
        predicted.cpu().numpy(),
    )
    aligned_gt = torch.tensor(aligned_gt, dtype=torch.float32, device=gt.device)
    aligned_predicted = torch.tensor(
        aligned_predicted, dtype=torch.float32, device=predicted.device
    )

    ate = ((aligned_gt - aligned_predicted) ** 2).mean() ** 0.5
    return ate, aligned_gt, aligned_predicted
