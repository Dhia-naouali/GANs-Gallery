import os
import cv2
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image

class CarsDataset(Dataset):
    def __init__(self, root_dir, image_size=256, augmentations=None):
        self.root_dir = root_dir
        self.image_size = image_size
        
        self.image_paths = [
            os.path.join(self.root_dir, f)                 
            for f in os.listdir(self.root_dir)
            ]

        self.transforms = self._init_transforms(augmentations or {})

    def _init_transforms(self, augs):
        transforms = [transforms.Resize((self.image_size, self.image_size))]

        
        if augs.get("horizontal_flip", 0):
            transforms.append(transforms.RandomHorizontalFlip(.5))
        
        if augs.get("rotation", 0):
            transforms.append(transforms.RandomRotation(augs["rotation"]))

        if augs.get("color_jitter", 0):
            p = augs["color_jitter"]
            transforms.ColorJitter(
                brightness=p,
                saturation=p,
                hue=p//2
            )

        transforms.extend([
            transforms.ToTensor(),
            transforms.Normalize(.5, .5)
        ])

        return transforms
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        return self.transforms(image)



class AdaptiveDiscriminatorAugmentation:
    def __init__(self, target_acc=.6, adjustment_speed=1e-3, max_prob=.8):
        self.target_acc = target_acc
        self.p_step = adjustment_speed
        self.max_prob = max_prob
        self.p = 0
        self.real_acc_ema = 0.5

        self.transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(.5),
            transforms.RandomVerticalFlip(.1),
            transforms.RandomRotatoin(10),
            transforms.ColorJitter(brightness=.2, contrast=.2, saturation=.2, hue=.1),
            transforms.RandomAffine(degrees=0, translate=(.1, .1), scale=(.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(.5, .5)

        ])

        def update(self, real_acc):
            self.real_acc_ema = .99 * self._real_acc_ema + .01 * real_acc

            if self.real_acc_ema > self.target_acc:
                self.p = min(self.current_prob + self.p_step, self.max_prob)
            else:
                self.p = max(self.p - self.p_step, 0)

        def __call__(self, images):
            if self.p == 0:
                return images
            
            aug_images = []
            for image in images:
                if random.random() < self.p:
                    image = transforms.ToPILImage(image.cpu())
                    image = self.transforms(image)
                aug_images.append(image)

            return torch.stack(aug_images)
        



def create_dataloader(root_dir, batch_size=32, image_size=256, num_workers=os.cpu_count(), pin_memory=True, augs=None):
    dataset = CarsDataset(
        root_dir=root_dir,
        image_size=image_size,
        augmentations=augs
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True, # no test data ...
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )

    return dataloader

# images to track ADA
def save_smaples(images, path, rows=4):
    images = (images + 1) / 2
    save_image(images, path, rows)