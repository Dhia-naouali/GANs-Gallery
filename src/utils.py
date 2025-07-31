import os
import random
import numpy as np

import wandb # I should get a wandb sticker
import math

import torch
from torch import optim, nn
from torchvision.utils import save_image

def seed_all(seed=12):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # aaaah the confidence
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_params(model, trainable_only=False):
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


@torch.no_grad
def generate_sample_images(
        generator,
        noise,
    ):
    generator.eval()
    fake_images = .5 * (generator(noise) + 1)
    return fake_images


def save_sample_images(images, path, rows=4):
    images = (images + 1) / 2
    save_image(images, path, nrow=rows)


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
    NEVER = object()
    FREQ = {
        "warm_up_cosine": PER_STEP,
        "warm_up_linear": PER_STEP,
        "step_decay": PER_EPOCH,
        "constant": NEVER, # will be ignored :'(
    }

    def __init__(self, optimizer):
        super().__init__(optimizer, last_epoch=-1)

    def take_step(self, epoch_call=False):
        if self.FREQ[self.NAME] == self.NEVER:
            return False

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

    def warm_up(self):
        if self.last_epoch < self.warm_up_steps:
            return self.last_epoch / max(1, self.warm_up_steps)
        return None


class WarmUpCosineScheduler(Scheduler):
    NAME = "warm_up_cosine"
    def __init__(self, optimizer, total_steps, config):
        self.total_steps = total_steps

        self.eta_min_ratio = config.get("eta_min_ratio", 1e-2)
        warm_up_phase = config.get("warm_up_phase", 0.05)
        warm_up_phase = warm_up_phase / 100 if warm_up_phase > 1 else warm_up_phase

        self.warm_up_steps = int(total_steps * warm_up_phase)
        self.peak_lr = optimizer.param_groups[0]['lr']
        super().__init__(optimizer)


    def get_lr(self):
        scale = super().warm_up()
        if scale is None:
            progress = (self.last_epoch - self.warm_up_steps) / max(1, self.total_steps - self.warm_up_steps)
            scale = self.eta_min_ratio + (1 - self.eta_min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

        lrs = [
            base_lr * scale
            for base_lr in self.base_lrs
        ]
        return lrs


class WarmUpLinearDecayScheduler(Scheduler):
    NAME = "warm_up_linear"
    def __init__(self, optimizer, total_steps, config):
        self.total_steps = total_steps
        self.init_lr = optimizer.param_groups[0]['lr']
        self.eta_min_ratio = config.get("min_lr", 1)

        warm_up_phase = config.get("warm_up_phase", 0.05)
        warm_up_phase = warm_up_phase / 100 if warm_up_phase > 1 else warm_up_phase
        self.warm_up_steps = int(total_steps * warm_up_phase)
        super().__init__(optimizer)


    def get_lr(self):
        scale = super().warm_up()
        if scale is None:
            progress = (self.last_epoch - self.warm_up_steps) / max(1, self.total_steps - self.warm_up_steps)
            progress = min(max(progress, 0), 1)
            scale = 1 - progress * (1 - self.eta_min_ratio)

        lrs = [
            base_lr * scale
            for base_lr in self.base_lrs
        ]
        return lrs


class StepDecayScheduler(Scheduler):
    NAME = "step_decay"
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


class ConstantScheduler(Scheduler):
    NAME = "constant"
    def __init__(self, optimizer, total_steps, config):
        super().__init__(optimizer)


SCHEDULERS = {
        "warm_up_cosine": WarmUpCosineScheduler,
        "warm_up_linear": WarmUpLinearDecayScheduler,
        "step_decay": StepDecayScheduler,
        "constant": ConstantScheduler,
    }


def setup_scheduler(optimizer, total_steps, config):
    return SCHEDULERS[config.scheduler.name](optimizer, total_steps, config.scheduler)



class MetricsTracker:
    class Metric:
        def __init__(self, name):
            self.name = name

        def reset(self):
            self.val = 0
            self.avg = 0
            self.sum = 0
            self.count = 0

        def update(self, new_val):
            self.val = new_val
            self.count += 1
            self.sum += new_val
            self.avg = self.sum / self.count


    names = ["G_loss", "D_loss", "fake_acc", "real_acc"] #, "epoch_time"]
    def __init__(self, log_freq):
        self.log_freq = log_freq
        self.tracked_metrics = {name: self.Metric(name) for name in self.names}


    def log(self, metrics, batch_idx, pbar=None):
        for name in self.names:
            self.tracked_metrics[name].update(metrics[name])

        if pbar:
            pbar.set_postfix({
                k: f"{metrics[k]:.4f}" 
                for k in self.tracked_metrics
            })

        if not batch_idx % self.log_freq and wandb.run is not None:
            wandb.log({
                f"train/{k}": metrics[k] for k in self.tracked_metrics
            })

    def averages(self):
        return {
            metric: metric.avg for metric in self.tracked_metrics.values()
        }

    def reset(self):
        for metric in self.tracked_metrics.values():
            metric.reset()


class CheckpointManager:
    def __init__(self, checkpoint_dir, G, D, G_optimizer, D_optimizer):
        self.checkpoint_dir = checkpoint_dir
        self.G = G
        self.D = D
        self.G_optimizer = G_optimizer
        self.D_optimizer = D_optimizer
        os.makedirs(checkpoint_dir, exist_ok=True)


    def save(self, epoch, epoch_metrics):
        checkpoint = {
            "epoch": epoch,
            "metrics": epoch_metrics,
            "generator_state_dict": self.G.state_dict(),
            "discriminator_state_dict": self.D.state_dict(),
            "generator_optimizer_state_dict": self.G_optimizer.state_dict(),
            "discriminator_optimizer_state_dict": self.D_optimizer.state_dict(),
        }

        torch.save(
            checkpoint, 
            os.path.join(
                self.checkpoint_dir,
                f"checkpoint_{epoch:04d}.pth"
            )
        )

    def load(self, checkpoint):
        checkpoint = torch.load(checkpoint, map_location=self.G.device)
        
        self.G.load_state_dict(checkpoint["generator_state_dict"])
        self.D.load_state_dict(checkpoint["discriminator_state_dict"])
        self.G_optimizer.load_state_dict(checkpoint["generator_optimizer_state_dict"])
        self.D_optimizer.load_state_dict(checkpoint["discriminator_optimizer_state_dict"])


    def cherry_pick(self):
        # to be called at the end to pick the best checkpoint
        ...



def init_weights(model, init_scheme="normal", gain=0.02):
    def init_func(module):
        modulename = module.__class__.__name__
        if hasattr(module, "weight") and ("Conv" in modulename or "Linear" in modulename):
            match init_scheme:
                case "normal":
                    nn.init.normal_(module.weight.data, 0., gain)
                case "xavier":
                    nn.init.xavier_normal_(module.weight.data, a=0, gain=gain)
                case "kaiming":
                    nn.init.kaiming_normal_(module.weight.data, gain=gain, mode="fan_in")
                case "orthogonal":
                    nn.init.orthogonal_(module.weight.data, gain=gain)
                case _:
                    raise Exception(f"{init_scheme} not implemented")
                    
            if hasattr(module, "bias") and module.bias is not None:
                nn.init.zeros_(module.bias.data)
        
        elif hasattr(module, "weight") and "Norm" in modulename and module.weight is not None: # to dodge pixelnorm & instancenorm layers
            nn.init.normal_(module.weight.data, 1.0, gain)
            nn.init.zeros_(module.bias.data)
    
    model.apply(init_func)



class EMA:
    def __init__(self, G, decay=.992):
        self.G = G
        self.decay = decay
        self.moving = {}
        self.backup = {}


    def register(self):
        for name, param in self.G.named_parameters():
            if param.requires_grad:
                self.moving[name] = param.data.clone()
    
    def update(self):
        for name, param in self.G.named_parameters():
            if param.requires_grad:
                self.moving[name] = self.decay * self.moving[name] +\
                    (1 - self.decay) * self.para.data


    def apply_moving(self):
        for name, param in self.G.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.moving[name]

    def restore(self):
        for name, param in self.G.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]