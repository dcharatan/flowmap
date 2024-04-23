from ..frame_sampler import FrameSamplerCfg, get_frame_sampler

try:
    from .dataset_co3d import DatasetCO3D, DatasetCO3DCfg
    from .dataset_colmap import DatasetCOLMAP, DatasetCOLMAPCfg
    from .dataset_images import DatasetImages, DatasetImagesCfg
    from .dataset_llff import DatasetLLFF, DatasetLLFFCfg
    from .dataset_merged import DatasetMerged
    from .dataset_re10k import DatasetRE10k, DatasetRE10kCfg
    from .types import Stage

    DATASETS = {
        "co3d": DatasetCO3D,
        "colmap": DatasetCOLMAP,
        "images": DatasetImages,
        "llff": DatasetLLFF,
        "re10k": DatasetRE10k,
    }

    DatasetCfg = (
        DatasetCO3DCfg
        | DatasetCOLMAPCfg
        | DatasetImagesCfg
        | DatasetLLFFCfg
        | DatasetRE10kCfg
    )

    def get_dataset(
        dataset_cfgs: list[DatasetCfg],
        stage: Stage,
        frame_sampler_cfg: FrameSamplerCfg,
    ) -> DatasetMerged:
        frame_sampler = get_frame_sampler(frame_sampler_cfg)
        datasets = [
            DATASETS[cfg.name](cfg, stage, frame_sampler) for cfg in dataset_cfgs
        ]
        return DatasetMerged(datasets)

except ImportError:
    get_dataset = None
