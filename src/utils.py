import os
import random
import numpy as np

import wandb # I should get a wandb sticker
import inspect

import torch
from torch import optim
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



class Scheduler(optim.lr_scheduler._LRScheduler):
    PER_STEP = object()
    PER_EPOCH = object()
    FREQ = {
        "warm_up_cosine": PER_STEP,
        "warmp_up_linear": PER_STEP,
        "step": PER_EPOCH,
        "constant": PER_EPOCH, # will be ignored :'(
    }

    def __init__(self, optimizer, total_steps, config):
        super().__init__(optimizer, last_epoch=-1)

    def take_step(self):
        if (
            self.FREQ[self.name] == self.PER_STEP and 
            inspect.stack()[1].function == "train_step"
        ) or (
            self.FREQ[self.name] == self.PER_EPOCH and 
            inspect.stack()[1].function == "train_epoch"
        ):
            return False
        return True



class CheckpointManager:
    def __init__(self, *args):
        ...