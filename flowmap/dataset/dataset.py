from dataclasses import dataclass


@dataclass
class DatasetCfgCommon:
    image_shape: tuple[int, int] | None
    scene: str | None
