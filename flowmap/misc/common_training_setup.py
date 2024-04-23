from pathlib import Path

import hydra
import torch
import wandb
from lightning.pytorch.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import Logger
from lightning.pytorch.loggers.wandb import WandbLogger
from omegaconf import DictConfig, OmegaConf

from ..config.common import CommonCfg
from .local_logger import LOG_PATH, LocalLogger
from .wandb_tools import update_checkpoint_path


def run_common_training_setup(
    cfg: CommonCfg,
    cfg_dict: DictConfig,
) -> tuple[list[Callback], Logger, Path | None, Path]:
    torch.set_float32_matmul_precision("highest")

    # Set up callbacks.
    callbacks = [
        LearningRateMonitor("step", True),
        ModelCheckpoint(
            (LOG_PATH / "checkpoints") if cfg.wandb.mode == "disabled" else None,
            every_n_train_steps=cfg.checkpoint.every_n_train_steps,
            save_top_k=-1,
        ),
    ]

    # Set up logging.
    if cfg.wandb.mode == "disabled":
        logger = LocalLogger()
        output_dir = LOG_PATH
    else:
        output_dir = Path(
            hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"]
        )
        output_dir = output_dir / cfg.wandb.name
        logger = WandbLogger(
            project=cfg.wandb.project,
            name=cfg.wandb.name,
            mode=cfg.wandb.mode,
            tags=cfg.wandb.tags,
            group=cfg.wandb.group,
            config=OmegaConf.to_container(cfg_dict),
            log_model="all",
            save_dir=output_dir,
        )

        # Log code to wandb if rank is 0. On rank != 0, wandb.run is None.
        if wandb.run is not None:
            wandb.run.log_code("flowmap")

    # Prepare the checkpoint for loading.
    checkpoint_path = update_checkpoint_path(cfg.checkpoint.load, cfg.wandb)

    return callbacks, logger, checkpoint_path, output_dir
