import os
import cv2
import random
import numpy as np
import torch
from torch import nn

from nvidia.dali import pipeline_def, fn, types, Pipeline
from nvidia.dali.plugin.pytorch import DALIGenericIterator
import kornia.augmentation as K


@pipeline_def(batch_size=8, enable_conditionals=False)
def data_pipeline(root_dir, image_size):
    image_files, _ = fn.readers.file(file_root=root_dir, random_shuffle=True, name="Reader")
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
        stddev=127.5*3
    )
    images = fn.transpose(
        images,
        output_layout="CHW"
    )

    return fn.cast(images, dtype=types.FLOAT16)
    


def setup_dataloader(config):
    # *root_dir, target_class = config.data.get("root_dir", "data/afhq/cat").split(os.sep)
    # print("\n"*4)
    # print("ROOT DIR", os.sep.join(root_dir))
    # print("TARGET CLASS", target_class)
    # print("\n"*4)
    pipe = data_pipeline(
        root_dir=config.data.get("root_dir", "data/afhq/cat"),
        seed=config.get("seed", 12),
        batch_size=config.training.get("batch_size", 32),
        image_size=config.training.get("image_size", 256),
        # target_class=target_class,
        device_id=0,
        num_threads=os.cpu_count(),
    )
    pipe.build()
    return DALIGenericIterator(
        [pipe],
        ["images"],
        auto_reset=True,
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
            K.RandomVerticalFlip(p=self.p),
            K.RandomRotation(10, p=self.p),
            K.ColorJitter(brightness=.2, contrast=.2, saturation=.2, hue=.1, p=self.p),
            K.RandomAffine(degrees=10, translate=(.1, .1), p=self.p),
            data_keys=["input"],
            same_on_batch=False
        )

    def update(self, real_acc):
        self.real_acc_ema = .99 * self.real_acc_ema + .01 * real_acc

        if self.real_acc_ema > self.target_acc:
            self.p = min(self.p + self.p_step, self.max_prob)
        else:
            self.p = max(self.p - self.p_step, 0)
        for aug in self.transform:
            aug.p = self.p

    def __call__(self, images):
        print("ADA", images.size())
        if self.p == 0:
            return images
        
        P = torch.bernoulli(
            torch.full((images.size(0),), self.p)
        ).bool()
        
        images[P] = self.transform(images[P])
        return images
