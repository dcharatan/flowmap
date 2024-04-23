from dataclasses import fields, replace
from typing import Any, TypeVar

import numpy as np
import torch
from jaxtyping import Bool, Int64
from torch import Tensor

T = TypeVar("T")

SliceLike = slice | int | Int64[Tensor, "..."] | Bool[Tensor, "..."]
Sliceable = Tensor | np.ndarray | list | tuple


def to_tuple(lst):
    return tuple(to_tuple(element) for element in lst) if isinstance(lst, list) else lst


class Manipulable:
    """Give a dataclass that bundles tensors, arrays, and lists functionality that
    mimics regular tensor manipulation. For example, this allows you to slice every
    element of a dataclass at once.
    """

    def to(self: T, device: torch.device) -> T:
        """Return a shallow copy of this instance in which all tensors have been moved
        to the specified device.
        """

        replacements = {}

        for field in fields(self):
            # Move fields that are torch.Tensor to the specified device.
            value = getattr(self, field.name)
            if isinstance(value, Tensor):
                replacements[field.name] = value.to(device)

        return replace(self, **replacements)

    def __getitem__(self: T, slices: slice | tuple[SliceLike, ...]) -> T:
        """Return a shallow copy of this instance in which all tensors, arrays, and
        lists have been sliced according to the specified slices. If the number of
        slices exceeds the number of dimensions of a particular dataclass element,
        ignore the trailing slices. For now, ellipses are not supported.
        """

        # To simplify the implementation below, ensure that slices is a tuple of slices.
        if not isinstance(slices, tuple):
            slices = (slices,)

        replacements = {}

        # Recursively slice any sliceable fields.
        for field in fields(self):
            value = getattr(self, field.name)
            value = Manipulable.recursive_slice(value, slices)
            replacements[field.name] = value

        return replace(self, **replacements)

    @staticmethod
    def recursive_slice(value: Any, slices: tuple[SliceLike, ...]) -> Any:
        if not isinstance(value, Sliceable):
            return value

        # Torch tensors and NumPy arrays don't need to be recursively sliced, since they
        # already support tuples of slices.
        if isinstance(value, Tensor) or isinstance(value, np.ndarray):
            # If the number of slices exceeds the number of dimensions, drop the
            # trailing slices.
            slices = slices[: value.ndim]

            # Apply the remaining slices.
            return value[slices]

        # Lists and tuples need to be recursively sliced. To check if we should recurse
        # further, we peek at the value's first element.
        can_recurse = isinstance(value[0], Sliceable)

        # Slice the value.
        leading_slice, *remaining_slices = slices
        value = value[leading_slice]

        # If the first element was sliceable, recursively apply slicing, making sure
        # to keep the value's type the same (e.g., tuples remain tuples, and lists
        # remain lists).
        if len(remaining_slices) > 0 and can_recurse:
            value = type(value)(
                [Manipulable.recursive_slice(x, remaining_slices) for x in value]
            )

        return value

    @staticmethod
    def cat(manipulables: list[T], dim: int) -> T:
        assert len(manipulables) != 0
        model = manipulables[0]

        replacements = {}

        # Record list and tuple data types.
        for field in fields(model):
            values = [getattr(manipulable, field.name) for manipulable in manipulables]
            replacements[field.name] = Manipulable.cat_sliceable(values, dim)

        return replace(model, **replacements)

    @staticmethod
    def cat_sliceable(values: list[Sliceable], dim: int) -> Sliceable:
        model = values[0]
        if isinstance(model, Tensor):
            return torch.cat(values, dim=dim)
        if isinstance(model, np.ndarray):
            return np.concatenate(values, axis=dim)

        # Handle lists and tuples.
        was_tuple = isinstance(model, tuple)
        values = [np.array(value) for value in values]
        values = np.concatenate(values, axis=dim)
        values = values.tolist()
        if was_tuple:
            values = to_tuple(values)
        return values
