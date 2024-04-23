from dataclasses import dataclass
from pathlib import Path
from typing import Type, TypeVar

from dacite import Config, from_dict
from omegaconf import DictConfig, OmegaConf

TYPE_HOOKS = {
    Path: Path,
}


T = TypeVar("T")


def get_typed_config(
    data_class: Type[T],
    cfg: DictConfig,
    extra_type_hooks: dict = {},
) -> T:
    return from_dict(
        data_class,
        OmegaConf.to_container(cfg),
        config=Config(type_hooks={**TYPE_HOOKS, **extra_type_hooks}, cast=[tuple]),
    )


def separate_multiple_defaults(data_class_union):
    """Return a function that will pull individual configurations out of a merged dict.
    For example, the merged dict might look like this:

    {
        a: ...
        b: ...
    }

    The returned function will generate this:

    [{ name: a, ... }, { name: b, ... }]

    In other words, this function makes the types for default lists with single and
    multiple items be typed identically.
    """

    def separate_fn(joined: dict) -> list:
        # The dummy allows the union to be converted.
        @dataclass
        class Dummy:
            dummy: data_class_union

        return [
            get_typed_config(Dummy, DictConfig({"dummy": {"name": name, **cfg}})).dummy
            for name, cfg in joined.items()
        ]

    return separate_fn
