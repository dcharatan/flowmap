from dataclasses import dataclass

import torch
from einops import reduce
from lightning import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import optim

from ..dataset.types import Batch
from ..flow import FlowPredictorCfg, Flows, get_flow_predictor
from ..loss import Loss
from ..misc.cropping import (
    CroppingCfg,
    crop_and_resize_batch_for_flow,
    crop_and_resize_batch_for_model,
)
from ..misc.image_io import prep_image
from ..visualization import Visualizer
from .model import Model


@dataclass
class ModelWrapperPretrainCfg:
    lr: float
    patch_size: int


class ModelWrapperPretrain(LightningModule):
    def __init__(
        self,
        cfg: ModelWrapperPretrainCfg,
        cfg_cropping: CroppingCfg,
        cfg_flow: FlowPredictorCfg,
        model: Model,
        losses: list[Loss],
        visualizers: list[Visualizer],
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.cfg_cropping = cfg_cropping
        self.flow_predictor = get_flow_predictor(cfg_flow)
        self.model = model
        self.losses = losses
        self.visualizers = visualizers

    @torch.no_grad()
    def preprocess_batch(self, batch_dict: dict) -> tuple[Batch, Flows]:
        # Convert the batch from an untyped dict to a typed dataclass.
        batch_dict.pop("frame_paths", None)
        batch = Batch(**batch_dict)

        # Compute optical flow and tracks.
        batch_for_model, _ = crop_and_resize_batch_for_model(batch, self.cfg_cropping)
        batch_for_flow = crop_and_resize_batch_for_flow(batch, self.cfg_cropping)
        _, _, _, h, w = batch_for_model.videos.shape
        flows = self.flow_predictor.compute_bidirectional_flow(batch_for_flow, (h, w))

        return batch_for_model, flows

    def training_step(self, batch):
        batch, flows = self.preprocess_batch(batch)

        # Compute depths, poses, and intrinsics using the model.
        model_output = self.model(batch, flows, self.global_step)

        # Compute and log the loss.
        total_loss = 0
        for loss_fn in self.losses:
            loss = loss_fn.forward(batch, flows, None, model_output, self.global_step)
            self.log(f"train/loss/{loss_fn.cfg.name}", loss)
            total_loss = total_loss + loss

        # Log intrinsics error.
        if batch.intrinsics is not None:
            fx_hat = reduce(model_output.intrinsics[..., 0, 0], "b f ->", "mean")
            fy_hat = reduce(model_output.intrinsics[..., 1, 1], "b f ->", "mean")
            fx_gt = reduce(batch.intrinsics[..., 0, 0], "b f ->", "mean")
            fy_gt = reduce(batch.intrinsics[..., 1, 1], "b f ->", "mean")
            self.log("train/intrinsics/fx_error", (fx_gt - fx_hat).abs())
            self.log("train/intrinsics/fy_error", (fy_gt - fy_hat).abs())

        return total_loss

    def validation_step(self, batch):
        batch, flows = self.preprocess_batch(batch)

        # Compute depths, poses, and intrinsics using the model.
        model_output = self.model(batch, flows, self.global_step)

        # Generate visualizations.
        for visualizer in self.visualizers:
            visualizations = visualizer.visualize(
                batch, flows, None, model_output, self.model, self.global_step
            )
            for key, visualization_or_metric in visualizations.items():
                if visualization_or_metric.ndim == 0:
                    # If it has 0 dimensions, it's a metric.
                    self.logger.log_metrics(
                        {key: visualization_or_metric},
                        step=self.global_step,
                    )
                else:
                    # If it has 3 dimensions, it's an image.
                    self.logger.log_image(
                        key,
                        [prep_image(visualization_or_metric)],
                        step=self.global_step,
                    )

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return optim.Adam(self.parameters(), lr=self.cfg.lr)
