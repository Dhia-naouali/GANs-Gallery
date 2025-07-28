import os
import cv2
import random
import numpy as np
import torch
from torch import nn

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import torch

from kornia.core import Device, Tensor  # noqa: TID252
from kornia.augmentation import (  # noqa: TID252
    AugmentationSequential,
    ColorJitter,
    ImageSequential,
    RandomAffine,
    RandomErasing,
    RandomGaussianNoise,
    RandomHorizontalFlip,
    RandomRotation90,
)
from kornia.augmentation.base import _AugmentationBase  # noqa: TID252
from kornia.augmentation.container.params import ParamItem  # noqa: TID252

_data_keys_type = List[str]
_inputs_type = Union[Tensor, Dict[str, Tensor]]



from nvidia.dali import pipeline_def, fn, types, Pipeline
from nvidia.dali.plugin.pytorch import DALIGenericIterator


@pipeline_def(batch_size=8, enable_conditionals=False)
def data_pipeline(root_dir, image_size, target_subdir=None):
    image_files, _ = fn.readers.file(
        file_root=root_dir,
        dir_filters=target_subdir,
        random_shuffle=True, 
        name="Reader"
    )
    images = fn.decoders.image(image_files, device="mixed", output_type=types.RGB)
    images = fn.resize(
        images,
        resize_x=image_size,
        resize_y=image_size,
        device="gpu",
        interp_type=types.INTERP_TRIANGULAR,
    )

    images = fn.normalize(
        images,
        mean=127.5,
        stddev=127.5
    )
    return fn.transpose(images)


def setup_dataloader(config):
    *root_dir, target_subdir = config.data.get("root_dir", "data/afhq/cat").split(os.sep)
    root_dir = os.sep.join(root_dir)
    pipe = data_pipeline(
        root_dir=root_dir,
        target_subdir=target_subdir,
        seed=config.get("seed", 12),
        batch_size=config.training.get("batch_size", 32),
        image_size=config.training.get("image_size", 256),
        device_id=0,
        num_threads=os.cpu_count(),
    )
    pipe.build()
    return DALIGenericIterator(
        [pipe],
        ["images"],
        auto_reset=False,
        reader_name="Reader"
    )


class AdaptiveDiscriminatorAugmentation(AugmentationSequential):
    def __init__(
        self,
        *args: Union[_AugmentationBase, ImageSequential],
        initial_p: float = 1e-5,
        adjustment_speed: float = 1e-2,
        max_p: float = 0.8,
        target_real_acc: float = 0.85,
        ema_lambda: float = 0.99,
        update_every: int = 5,
        erasing_scale: Union[Tensor, Tuple[float, float]] = (0.02, 0.33),
        erasing_ratio: Union[Tensor, Tuple[float, float]] = (0.3, 3.3),
        erasing_fill_value: float = 0.0,
        data_keys: Optional[_data_keys_type] = None,
        same_on_batch: Optional[bool] = False,
        **kwargs: Any,
    ) -> None:
        if not args:
            args = self.default_ada_transfroms(erasing_scale, erasing_ratio, erasing_fill_value)

        super().__init__(
            *args,
            data_keys=data_keys
            if data_keys is not None
            else [
                "input",
            ],
            same_on_batch=same_on_batch,
            **kwargs,
        )

        if adjustment_speed <= 0:
            raise ValueError(f"Invalid `adjustment_speed` ({adjustment_speed}) — must be greater than 0")

        if not 0 <= target_real_acc <= 1:
            raise ValueError(f"Invalid `target_real_acc` ({target_real_acc}) — must be in [0, 1]")

        if not 0 <= ema_lambda <= 1:
            raise ValueError(f"Invalid `ema_lambda` ({ema_lambda}) — must be in [0, 1]")

        if update_every < 1:
            raise ValueError(f"Invalid `update_every` ({update_every}) — must be at least 1")

        if not 0 <= max_p <= 1:
            raise ValueError(f"Invalid `max_p` ({max_p}) — must be in [0, 1]")

        if not 0 <= initial_p <= 1:
            raise ValueError(f"Invalid `initial_p` ({initial_p}) — must be in [0, 1]")

        if initial_p > max_p:
            warnings.warn(
                f"`initial_p` ({initial_p}) is greater than `max_p` ({max_p}), resetting `initial_p` to `max_p`",
                stacklevel=2,
            )
            initial_p = max_p

        self.p = initial_p
        self.adjustment_speed = adjustment_speed
        self.max_p = max_p
        self.target_real_acc = target_real_acc
        self.ema_lambda = ema_lambda
        self.update_every = update_every
        self.real_acc_ema: float = 0.5
        self._num_calls = 0  # -update_every  # to avoid updating in the first `update_every` steps

    def default_ada_transfroms(
        self, scale: Union[Tensor, Tuple[float, float]], ratio: Union[Tensor, Tuple[float, float]], value: float
    ) -> Tuple[Union[_AugmentationBase, ImageSequential], ...]:
        # if changed in the future, please change the expected transforms list in test_presets.py
        return (
            RandomHorizontalFlip(p=1),
            RandomRotation90(times=(0, 3), p=1.0),
            RandomErasing(scale=scale, ratio=ratio, value=value, p=0.9),
            RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), p=1.0),
            ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1.0),
            RandomGaussianNoise(std=0.1, p=1.0),
        )

    def update(self, real_acc: float) -> None:
        self._num_calls += 1

        if self._num_calls < self.update_every:
            return
        self._num_calls = 0

        self.real_acc_ema = self.ema_lambda * self.real_acc_ema + (1 - self.ema_lambda) * real_acc

        if self.real_acc_ema < self.target_real_acc:
            self.p = max(0, self.p - self.adjustment_speed)
        else:
            self.p = min(self.p + self.adjustment_speed, self.max_p)

    def _get_inputs_metadata(self, inputs: _inputs_type, data_keys: _data_keys_type) -> int:
        if isinstance(inputs, dict):
            key = data_keys[0]
            batch_size = inputs[key].size(0)
        else:
            batch_size = inputs.size(0)

        return batch_size

    def _sample_inputs(self, inputs: _inputs_type, data_keys: _data_keys_type, p_tensor: Tensor) -> _inputs_type:
        if isinstance(inputs, dict):
            return {key: inputs[key][p_tensor] for key in data_keys}
        else:
            return inputs[p_tensor]

    def _merge_inputs(
        self,
        original: _inputs_type,
        augmented: _inputs_type,
        p_tensor: Tensor,
    ) -> _inputs_type:
        merged: _inputs_type
        if isinstance(original, dict) and isinstance(augmented, dict):
            merged = {}
            for key in original.keys():
                merged_tensor = original[key].clone()
                merged_tensor[p_tensor] = augmented[key]
                merged[key] = merged_tensor
        elif isinstance(original, Tensor) and isinstance(augmented, Tensor):
            merged = original.clone()
            merged[p_tensor] = augmented
        else:
            raise TypeError(
                f"original inputs and augmented inputs aren't of the same type "
                f"(type({type(original)}), type({type(augmented)}))"
            )
        return merged

    def forward(  # type: ignore[override]
        self,
        inputs: _inputs_type,
        params: Optional[List[ParamItem]] = None,
        data_keys: Optional[_data_keys_type] = None,
        real_acc: Optional[float] = None,
    ) -> _inputs_type:
        if real_acc is not None:
            self.update(real_acc)

        if self.p == 0:
            return inputs

        if data_keys is None:
            data_keys = (
                [k.name for k in self.data_keys]
                if self.data_keys is not None
                else [
                    "input",
                ]
            )

        batch_size = self._get_inputs_metadata(inputs, data_keys=data_keys)

        p_tensor = torch.bernoulli(torch.full((batch_size,), self.p, dtype=torch.float32)).bool()

        if not p_tensor.any():
            return inputs

        selected_inputs: _inputs_type = self._sample_inputs(inputs, data_keys=data_keys, p_tensor=p_tensor)
        augmented_inputs = cast(
            _inputs_type,
            super().forward(
                selected_inputs,  # type: ignore[arg-type]
                params=params,
                data_keys=data_keys,
            ),
        )

        return self._merge_inputs(inputs, augmented_inputs, p_tensor)
