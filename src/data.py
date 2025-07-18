import os
import cv2
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
import albumentations as A 
from albumentations.pytorch import ToTensorV2 as ToTensor

class CarsDataset(Dataset):
    def __init__(self, root_dir, image_size=256, augmentations=None):
        self.root_dir = root_dir
        self.image_size = image_size
        
        self.image_paths = [
            os.path.join(self.root_dir, f)                 
            for f in os.listdir(self.root_dir)
        ]

        self.transform = self._init_A(augmentations or {})

    def _init_A(self, augs):
        transform = [A.Resize(self.image_size, self.image_size)]


        if augs.get("horizontal_flip", 0):
            transform.append(A.HorizontalFlip(.5))
        
        if augs.get("rotation", 0):
            transform.append(A.Rotate(augs.get("rotation", 0)))

        if p := augs.get("color_jitter", 0):
            transform.append(
                A.ColorJitter(
                    brightness=p,
                    saturation=p,
                    hue=p//2
                )
            )


        transform.extend([
            A.Normalize(mean=(.5, .5, .5), std=(.5, .5, .5)),
            ToTensor(),
        ])

        return A.Compose(transform)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        return self.transform(image=image)["image"]



class AdaptiveDiscriminatorAugmentation:
    def __init__(self, target_acc=.6, adjustment_speed=1e-3, max_prob=.8):
        self.target_acc = target_acc
        self.p_step = adjustment_speed
        self.max_prob = max_prob
        self.p = 0
        self.real_acc_ema = 0.5

        self.transform = A.Compose([
            A.HorizontalFlip(.5),
            A.VerticalFlip(.1),
            A.Rotate(10),
            A.ColorJitter(brightness=.2, contrast=.2, saturation=.2, hue=.1),
            A.Affine(degrees=0, translate_percent=(.1, .1), scale=(.9, 1.1)),
            A.Normalize(mean=(.5, .5, .5), std=(.5, .5, .5)),
            ToTensor(),
        ])

    def update(self, real_acc):
        self.real_acc_ema = .99 * self.real_acc_ema + .01 * real_acc

        if self.real_acc_ema > self.target_acc:
            self.p = min(self.p + self.p_step, self.max_prob)
        else:
            self.p = max(self.p - self.p_step, 0)

    def __call__(self, images):
        device = images.device
        if self.p == 0:
            return images
        
        aug_images = []
        for image in images:
            if random.random() < self.p:
                image = image.permute(1, 2, 0).cpu().numpy()
                image = (
                    ((image * .5) + .5) * 255
                ).astype(np.uint8)
                image = self.transform(image=image)["image"].to(device)
            aug_images.append(image)

        return torch.stack(aug_images)


def setup_dataloader(config):

    dataset = CarsDataset(
        root_dir = config.data.get("root_dir", "data/afhq/cat"),
        image_size = config.data.get("image_size", 256),
        augmentations = config.data.get("augmentations", {})
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config.training.get("batch_size", 32),
        shuffle=True,
        num_workers=config.training.get("num_workers", os.cpu_count()),
        pin_memory=config.training.get("pin_memory", True),
        drop_last=True,
    )

    return dataloader