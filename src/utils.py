import os
import random
import numpy as np

import wandb # I should get a wandb sticker
import math

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
        "warm_up_linear": PER_STEP,
        "step": PER_EPOCH,
        "constant": PER_EPOCH, # will be ignored :'(
    }

    def __init__(self, optimizer):
        super().__init__(optimizer, last_epoch=-1)

    def take_step(self, epoch_call=False):
        if (
            self.FREQ[self.NAME] == self.PER_STEP and 
            epoch_call
        ) or (
            self.FREQ[self.NAME] == self.PER_EPOCH and 
            not epoch_call
        ):
            return False
        return True
    
    def step(self, epoch_call=False):
        if self.take_step(epoch_call):
            super().step()



class WarmUpLinearDecay(Scheduler):
    NAME = "warm_up_linear"
    def __init__(self, optimizer, total_steps, config):
        self.total_steps = total_steps
        self.init_lr = optimizer.param_groups[0]['lr']
        self.min_lr = config.get("min_lr", 5e-8)
        super().__init__(optimizer)


    def get_lr(self):
        progress = self.last_epoch / self.total_steps
        progress = min(max(progress, 0), 1)
        
        lrs = [
            base_lr * (1 - progress) + self.min_lr * progress
            for base_lr in self.base_lrs
        ]
        return lrs


class WarmUpCosine(Scheduler):
    def __init__(self, optimizer, total_steps, config):
        self.total_steps = total_steps

        self.eta_min_ratio = config.get("eta_min_ratio", 1e-2)
        warmup_phase = config.get("warm_up_phase", 0.05)
        if warmup_phase > 1:
            warmup_phase /= 100

        self.warmup_steps = int(total_steps * warmup_phase)
        self.peak_lr = optimizer.param_groups[0]['lr']
        super().__init__(optimizer)


    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            scale = self.last_epoch / max(1, self.warmup_steps)
        else:
            progress = (self.last_epoch - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            scale = self.eta_min_ratio + (1 - self.eta_min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

        lrs = [
            base_lr * scale
            for base_lr in self.base_lrs
        ]
        return lrs


class Step(Scheduler):
    NAME = "step"
    def __init__(self, optimizer, total_steps, config):
        self.update_freq = config.get("update_freq", 1)
        self.gamma = config.get("gamma", .98)
        self.calls = 0
        super().__init__(optimizer)


    def get_lr(self):
        self.calls += 1
        scale = self.gamma if not self.calls % self.update_freq else 1
 
        lrs = [
            base_lr * scale
            for base_lr in self.base_lrs
        ]
        return lrs



SCHEDULERS = {
        "warm_up_cosine": None,
        "warm_up_linear": WarmUpLinearDecay,
        "step": None,
        "constant": None,
    }


def setup_scheduler(optimizer, total_steps, config):
    return SCHEDULERS[config.scheduler.name](optimizer, total_steps, config.scheduler)


class CheckpointManager:
    def __init__(self, *args):
        ...