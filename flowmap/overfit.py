import json
import shutil
from pathlib import Path
from time import time

import hydra
import torch
import wandb
from jaxtyping import install_import_hook
from lightning import Trainer
from lightning.pytorch.plugins.environments import SLURMEnvironment
from omegaconf import DictConfig
from torch.utils.data import default_collate

# Configure beartype and jaxtyping.
with install_import_hook(
    ("flowmap",),
    ("beartype", "beartype"),
):
    from .config.common import get_typed_root_config
    from .config.overfit import OverfitCfg
    from .dataset import get_dataset
    from .dataset.data_module_overfit import DataModuleOverfit
    from .dataset.types import Batch
    from .export.colmap import export_to_colmap
    from .flow import compute_flows
    from .loss import get_losses
    from .misc.common_training_setup import run_common_training_setup
    from .misc.cropping import (
        crop_and_resize_batch_for_flow,
        crop_and_resize_batch_for_model,
    )
    from .model.model import Model
    from .model.model_wrapper_overfit import ModelWrapperOverfit
    from .tracking import compute_tracks
    from .visualization import get_visualizers


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="overfit",
)
def overfit(cfg_dict: DictConfig) -> None:
    start_time = time()
    cfg = get_typed_root_config(cfg_dict, OverfitCfg)
    callbacks, logger, checkpoint_path, output_dir = run_common_training_setup(
        cfg, cfg_dict
    )
    device = torch.device("cuda:0")

    # Load the full-resolution batch.
    dataset = get_dataset(cfg.dataset, "train", cfg.frame_sampler)
    batch = next(iter(dataset))
    frame_paths = batch.pop("frame_paths", None)
    if frame_paths is not None:
        frame_paths = [Path(path) for path in frame_paths]
    batch = Batch(**default_collate([batch]))

    # Compute optical flow and tracks.
    batch_for_model, pre_crop = crop_and_resize_batch_for_model(batch, cfg.cropping)
    batch_for_flow = crop_and_resize_batch_for_flow(batch, cfg.cropping)
    _, f, _, h, w = batch_for_model.videos.shape
    flows = compute_flows(batch_for_flow, (h, w), device, cfg.flow)

    # Only compute tracks if the tracking loss is enabled.
    if any([loss.name == "tracking" for loss in cfg.loss]):
        tracks = compute_tracks(
            batch_for_flow, device, cfg.tracking, cfg.track_precomputation
        )
    else:
        tracks = None

    # Set up the model.
    optimization_start_time = time()
    model = Model(cfg.model, num_frames=f, image_shape=(h, w))
    losses = get_losses(cfg.loss)
    visualizers = get_visualizers(cfg.visualizer)
    model_wrapper = ModelWrapperOverfit(
        cfg.model_wrapper,
        model,
        batch_for_model,
        flows,
        tracks,
        losses,
        visualizers,
    )

    # Only load the model's saved state (so that optimization restarts).
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path)
        model_wrapper.load_state_dict(checkpoint["state_dict"], strict=False)

    trainer = Trainer(
        max_epochs=-1,
        accelerator="gpu",
        logger=logger,
        devices="auto",
        strategy=(
            "ddp_find_unused_parameters_true"
            if torch.cuda.device_count() > 1
            else "auto"
        ),
        callbacks=callbacks,
        val_check_interval=cfg.trainer.val_check_interval,
        max_steps=cfg.trainer.max_steps,
        plugins=[SLURMEnvironment(auto_requeue=False)],
        log_every_n_steps=1,
    )
    trainer.fit(
        model_wrapper,
        datamodule=DataModuleOverfit(),
    )

    # Export the result.
    print("Exporting results.")
    model_wrapper.to(device)
    exports = model_wrapper.export(device)

    colmap_path = output_dir / "colmap"
    export_to_colmap(
        exports,
        frame_paths,
        pre_crop,
        batch.videos,
        colmap_path,
    )
    shutil.make_archive(colmap_path, "zip", output_dir, "colmap")

    if cfg.local_save_root is not None:
        # Save the COLMAP-style output.
        cfg.local_save_root.mkdir(exist_ok=True, parents=True)
        shutil.copytree(colmap_path, cfg.local_save_root, dirs_exist_ok=True)

        # Save the runtime. For a fair comparison with COLMAP, we record how long it
        # takes until the COLMAP-style output has been saved.
        times = {
            "runtime": time() - start_time,
            "optimization_runtime": time() - optimization_start_time,
        }
        with (cfg.local_save_root / "runtime.json").open("w") as f:
            json.dump(times, f)

        # Save the exports (poses, intrinsics, depth maps + corresponding colors).
        torch.save(exports, cfg.local_save_root / "exports.pt")

        # Save a checkpoint.
        trainer.save_checkpoint(cfg.local_save_root / "final.ckpt")

    if cfg.wandb.mode != "disabled":
        artifact = wandb.Artifact(f"colmap_{wandb.run.id}", type="colmap")
        artifact.add_file(f"{colmap_path}.zip", name="colmap.zip")
        wandb.log_artifact(artifact)
        artifact.wait()


if __name__ == "__main__":
    overfit()
