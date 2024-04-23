import hydra
import torch
from jaxtyping import install_import_hook
from lightning import Trainer
from lightning.pytorch.plugins.environments import SLURMEnvironment
from omegaconf import DictConfig

# Configure beartype and jaxtyping.
with install_import_hook(
    ("flowmap",),
    ("beartype", "beartype"),
):
    from .config.common import get_typed_root_config
    from .config.pretrain import PretrainCfg
    from .dataset.data_module_pretrain import DataModulePretrain
    from .loss import get_losses
    from .misc.common_training_setup import run_common_training_setup
    from .model.model import Model
    from .model.model_wrapper_pretrain import ModelWrapperPretrain
    from .visualization import get_visualizers


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="pretrain",
)
def pretrain(cfg_dict: DictConfig) -> None:
    cfg = get_typed_root_config(cfg_dict, PretrainCfg)
    callbacks, logger, checkpoint_path, _ = run_common_training_setup(cfg, cfg_dict)

    # Configure the datasets to load the model's desired image shape.
    multiplier = cfg.cropping.flow_scale_multiplier
    flow_shape = tuple(x * multiplier for x in cfg.cropping.image_shape)
    for dataset_cfg in cfg.dataset:
        dataset_cfg.image_shape = flow_shape

    # Set up the model.
    model = Model(cfg.model)
    losses = get_losses(cfg.loss)
    visualizers = get_visualizers(cfg.visualizer)
    model_wrapper = ModelWrapperPretrain(
        cfg.model_wrapper,
        cfg.cropping,
        cfg.flow,
        model,
        losses,
        visualizers,
    )
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
        datamodule=DataModulePretrain(
            cfg.dataset,
            cfg.data_module,
            cfg.frame_sampler,
            trainer.global_rank,
        ),
        ckpt_path=checkpoint_path,
    )


if __name__ == "__main__":
    pretrain()
