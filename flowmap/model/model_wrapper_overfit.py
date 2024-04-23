from dataclasses import dataclass

import torch
from einops import reduce
from lightning import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import optim

from ..dataset.types import Batch
from ..flow import Flows
from ..loss import Loss
from ..misc.image_io import prep_image
from ..tracking import Tracks
from ..visualization import Visualizer
from .model import Model, ModelExports


@dataclass
class ModelWrapperOverfitCfg:
    lr: float
    patch_size: int


class ModelWrapperOverfit(LightningModule):
    def __init__(
        self,
        cfg: ModelWrapperOverfitCfg,
        model: Model,
        batch: Batch,
        flows: Flows,
        tracks: list[Tracks] | None,
        losses: list[Loss],
        visualizers: list[Visualizer],
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.batch = batch
        self.flows = flows
        self.tracks = tracks
        self.model = model
        self.losses = losses
        self.visualizers = visualizers

    def to(self, device: torch.device) -> None:
        self.batch = self.batch.to(device)
        self.flows = self.flows.to(device)
        if self.tracks is not None:
            self.tracks = [tracks.to(device) for tracks in self.tracks]
        super().to(device)

    def training_step(self, dummy):
        # Compute depths, poses, and intrinsics using the model.
        model_output = self.model(self.batch, self.flows, self.global_step)

        # Compute and log the loss.
        total_loss = 0
        for loss_fn in self.losses:
            loss = loss_fn.forward(
                self.batch, self.flows, self.tracks, model_output, self.global_step
            )
            self.log(f"train/loss/{loss_fn.cfg.name}", loss)
            total_loss = total_loss + loss

        # Log intrinsics error.
        if self.batch.intrinsics is not None:
            fx_hat = reduce(model_output.intrinsics[..., 0, 0], "b f ->", "mean")
            fy_hat = reduce(model_output.intrinsics[..., 1, 1], "b f ->", "mean")
            fx_gt = reduce(self.batch.intrinsics[..., 0, 0], "b f ->", "mean")
            fy_gt = reduce(self.batch.intrinsics[..., 1, 1], "b f ->", "mean")
            self.log("train/intrinsics/fx_error", (fx_gt - fx_hat).abs())
            self.log("train/intrinsics/fy_error", (fy_gt - fy_hat).abs())

        return total_loss

    def validation_step(self, dummy):
        # Compute depths, poses, and intrinsics using the model.
        model_output = self.model(self.batch, self.flows, self.global_step)

        # Generate visualizations.
        for visualizer in self.visualizers:
            visualizations = visualizer.visualize(
                self.batch,
                self.flows,
                self.tracks,
                model_output,
                self.model,
                self.global_step,
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

    def export(self, device: torch.device) -> ModelExports:
        return self.model.export(
            self.batch.to(device),
            self.flows.to(device),
            self.global_step,
        )
