import os
import cv2
from torch.utils.data import Dataset, DataLoader


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