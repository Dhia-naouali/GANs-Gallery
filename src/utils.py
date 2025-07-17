import os
import random
import numpy as np
import torch
import wandb # I should get a wandb sticker
from torchvision.utils import save_image

def seed_all(seed=12):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False # hoping to land on good enough kernels T-T


def count_params(model, trainable_only=False):
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters()) 


def log_to_wandb():
    ...


@torch.no_grad
def generate_sample_images(
        generator, # EMA primarily
        num_samples=16,
        lat_dim=None, # yet to be selected
        path=".",
        rows=4,
    ):
    generator.eval()
    device = next(generator.parameters()).device
    
    noise = torch.randn(num_samples, lat_dim, device=device)
    fake_images = .5 * (generator(noise) + 1)
    save_image(fake_images, path, nrow=rows)




def setup_directories(config):
    for dir in [
        config.get("log_dir", "logs"),
        config.get("checkpoint_dir", "checkpoints"),
        config.get("samples_dir", "samples"),
    ]:
        os.makedirs(dir, exist_ok=True)


class Scheduler:
    def __init__(self, *args):
        ...

class CheckpointManager:
    def __init__(self, *args):
        ...