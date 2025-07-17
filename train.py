import torch
from torch import optim
from torch.cuda.amp import GradScaler, autocast

import time
import hydra
import wandb
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf

from src.utils import (
    seed_all,
    init_directories,
    count_params,
    Scheduler,
    CheckpointManager,
)
from src.models import setup_models
from src.data import setup_dataloader, AdaptiveDiscriminatorAugmentation



class Trainer:
    def __init__(self, config):
        # init directories: samples, chekcpoints, ...
        # scheduler
        # regulizers
        # scalers
        # checkpoint manager
        # compile ?
        # metrics


        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu") # someone's CPU goin down 💀
        seed_all()

        init_directories()

        self.G, self.D = setup_models(config.model)
        self.G.to(self.device); self.D.to(self.device)

        # proper weights_init (muP ?)

        print(f"Generator: {count_params(self.G) * 1e-6:.2f} \n"
              f"Discriminator: {count_params(self.D) * 1e-6:.2f}")
        
        self.setup_optimizers()
        self.setup_loss()

        self.dataloader = setup_dataloader(
            config
        )

        if config.data.use_ADA:
            self.ada = AdaptiveDiscriminatorAugmentation(
                target_acc=config.data.ada_target_acc
            )
        else:
            self.ada = None


        self.G_scaler = GradScaler()
        self.D_scaler = GradScaler()

        self.checkpoint_manager = CheckPointManager(
            ...
        )


        self.NOISE = torch.randn(32, self.lat_dim, device=self.device)

        

    def setup_optimizers(self):
        config = self.config.optimizer

        G_lr = config.G_lr
        D_lr = config.D_lr
        # D_lr = config.D_lr if D_lr in config else G_lr / config.

        self.G_optimizer = optim.AdamW(
            self.G.parameters(),
            lr=G_lr,
            betas=(),
            weight_decay=config.weight_decay,
        )

        self.D_optimizer = optim.AdamW(
            self.D.parameters(),
            lr=D_lr,
            betas=(),
            weight_decay=config.weight_decay,
        )

        self.G_scheduler = Scheduler(self.G_optimizer, config)
        self.D_scheduler = Scheduler(self.D_optimzier, config)


    def setup_loss(self):
        self.criterion = init_criterion(
            self.config.loss,
            device=self.device,
        )

        self.regs = {}



    def step(self, real_images):
        # called from train_epoch: G & D in train mode 
        bs = real_images.size(0)

        if self.ada:
            real_images = self.ada(real_images)
        
        noise = torch.randn(bs, self.config.model.lat_dim, device=self.device)

        # D step
        D_loss, real_acc, fake_acc = self.D_step(real_images, noise)

        # G step
        G_loss = self.G_step(noise)

        if self.ada:
            self.ada.update(real_acc)

        return {
            "G_loss": G_loss,
            "D_loss": D_loss,
            "real_acc": real_acc,
            "fake_acc": fake_acc,
        }


    def D_step(self, real_images, noise):
        self.D.zero_grad()
        with autocast(device_type=self.device):
            real_images.requires_grad_(True)
            real_logits = self.D(real_images)

            with torch.no_grad():
                fake_images = self.G(noise).detach()
            fake_logits = self.D(fake_images)

            D_loss = self.criterion.discriminator_loss(real_logits, fake_logits)
        
        self.D_scaler.scale(D_loss).backward()
        self.D_scaler.step(self.D_optimizer)
        self.D_scaler.update()

        with torch.no_grad():
            fake_acc = (torch.tanh(fake_logits) < 0).float().mean().item()
            real_acc = (torch.tanh(real_logits) > 0).float().mean().item()

        return D_loss, fake_acc, real_acc




    def G_step(self, noise):
        self.G.zero_grad()

        with autocast(device_type=self.device):
            fake_images = self.G(noise)
            fake_logits = self.D(fake_images)

            G_loss = self.criterion.generator_loss(fake_logits)
        self.G_scaler.scale(G_loss).backward()
        self.G_scaler.step(self.G_optimizer)
        self.G_scaler.update()
        
        return G_loss



    def train_epoch(self, epoch, epochs):
        self.G.train()
        self.D.train()

        pbar = tqdm(self.dataloader, desc=f"[Epoch {epoch}/{epochs}]: ")

        for batch_idx, real_imgeaes in enumerate(pbar):
            real_images = real_images.to(self.device)
            step_metrics = self.step(real_images)

            pbar.set_postfix({
                "G_loss": f"{self.metrics['']:.4f}",
                "D_loss": f"{self.metrics['']:.4f}",
                "real_acc": f"{self.metrics['']:.4f}",
                "fake_acc": f"{self.metrics['']:.4f}",
            })

            if wandb.run is not None and not batch_idx % self.config.wandb.log_freq:
                wandb.log({
                    "train/G_loss": step_metrics["G_loss"],
                    "train/D_loss": step_metrics["D_loss"],
                    "train/real_acc": step_metrics["real_acc"],
                    "train/fake_acc": step_metrics["fake_acc"],
                })

        return {

        }
    

    def train(self):
        # main training loop script
        for epoch in range(1, self.config.training.epochs + 1):
            start_time = time.time()
            epoch_metrics = self.train_epoch(epoch, self.config.training.epochs)

            # if scheduler per epoch
            self.G_scheduler.step()
            self.D_scheduler.step()

            if not epoch % self.config.training.sample_freq:
                self.generate_samples(epoch)

            if not epoch % self.config.training.save_every:
                self.checkpoint_manager.save_checkpoint()

            if wandb.run:
                wandb.log({
                    "epoch/G_loss": epoch_metrics["G_loss"],
                    "epoch/D_loss": epoch_metrics["D_loss"],
                    "epoch/real_acc": epoch_metrics["real_acc"],
                    "epoch/fake_acc": epoch_metrics["fake_acc"],
                    "epoch/time": time.time() - start_time,
                    "dpoch": epoch,
                })

            



    def generate_samples(self, epoch):
        ...