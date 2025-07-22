import os
import cv2
import random
import numpy as np
import torch
from torch import nn

# pip install nvidia-dali-cuda110  # wheel matching CUDA 11 on P100
# pip install kornia
from nvidia.dali import pipeline_def, fn, types, Pipeline
from nvidia.dali.plugin.pytorch import DALIGenericIterator
import kornia.augmentations as K


@pipeline_def(batch_size=32, image_size=256, enable_conditionals=False)
def data_pipeline(root_dir, horizontal_flip=.5):
    image_size=Pipeline.current().image_size

    image_files = fn.readers.file(file_root=root_dir, random_shuffle=True, name="Reader")
    images = fn.decoders.image(image_files, device="mixed", output_type=types.RGB)
    images = fn.resize(
        resize_x=image_size,
        resize_y=image_size,
        device="gpu",
        interp_type=types.INTERP_TRIANGULAR,
    )
    coin_flip = fn.coin_flip(probability=horizontal_flip, dtype=types.BOOL)
    images = fn.flip(images, horizontal=coin_flip)
    images = fn.cast(images / 255.0, dtype=types.FLOAT16)

    images = fn.normalize(
        images,
        mean=[.5]*3,
        std=[.5]*3
    )

    return images


def setup_dataloader(config):
    pipe = data_pipeline(
        root_dir = config.data.get("root_dir", "data/afhq/cat"),
        seed=config.get("seed", 12),
        batch_size=config.training.get("batch_size", 32),
        image_size=config.training.get("image_size", 256)        
    )
    pipe.build()
    return DALIGenericIterator(
        [pipe],
        ["images"],
        size=pipe.epoch_size("Reader"),
        auto_reset=True
    )

class AdaptiveDiscriminatorAugmentation:
    def __init__(self, target_acc=.6, adjustment_speed=1e-2, max_prob=.8):
        self.target_acc = target_acc
        self.p_step = adjustment_speed
        self.max_prob = max_prob
        self.p = 0
        self.real_acc_ema = 0.5

        self.transform = K.AugmentationSequential(
            K.RandomHorizontalFlip(p=self.p),
            K.RandomVerticalFlip(p=self.p/4),
            K.RandomRotation(10, p=self.p),
            K.ColorJitter(brightness=.2, contrast=.2, saturation=.2, hue=.1, p=self.p),
            K.affine(degrees=10, translate=(.1, .1), p=self.p),
            data_keys=["input"],
            same_on_batch=False
        ).half()

    def update(self, real_acc):
        self.real_acc_ema = .99 * self.real_acc_ema + .01 * real_acc

        if self.real_acc_ema > self.target_acc:
            self.p = min(self.p + self.p_step, self.max_prob)
        else:
            self.p = max(self.p - self.p_step, 0)

    def __call__(self, images):
        if self.p == 0:
            return images
        
        P = torch.bernoulli(
            torch.full((images.size(0),), p)
        ).bool()
        
        images[P] = self.transform(images[P])
        return images
