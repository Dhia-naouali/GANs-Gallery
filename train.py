import torch
from torch import optim
from torch.amp import GradScaler, autocast

torch.backends.cudnn.benchmark = True 
torch.backends.cudnn.deterministic = False

from torchvision.utils import make_grid

import os
import time
import hydra
import wandb
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf

from src.utils import (
    seed_all,
    setup_directories,
    count_params,
    setup_scheduler,
    MetricsTracker,
    CheckpointManager,
    generate_sample_images,
    save_sample_images,
)
from src.models import setup_models
from src.data import setup_dataloader, AdaptiveDiscriminatorAugmentation
from src.losses import setup_loss


class Trainer:
    def __init__(self, config):
        # regularizers
        # compile ?


        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu") # someone's CPU goin down 💀
        seed_all()

        setup_directories(self.config)

        self.G, self.D = setup_models(config.model)
        print(self.G, self.D)
        self.G.to(self.device, memory_format=torch.channels_last); self.D.to(self.device, memory_format=torch.channels_last)

        # proper weights_init (muP ?)

        print(f"Generator: {count_params(self.G) * 1e-6:.2f} \n"
              f"Discriminator: {count_params(self.D) * 1e-6:.2f}")
        

        self.dataloader = setup_dataloader(
            config
        )

        self.n_critic = self.config.training.get("n_critic", 1)
        self.setup_optimizers()
        self.setup_loss_and_regs()


        self.ada = AdaptiveDiscriminatorAugmentation(
            target_acc=config.ADA.ada_target_acc
        ) if config.ADA.use_ADA else None


        self.G_scaler = GradScaler(device=self.device)
        self.D_scaler = GradScaler(device=self.device)

        self.checkpoint_manager = CheckpointManager(
            self.config.checkpoint_dir,
            self.G,
            self.D,
            self.G_optimizer,
            self.D_optimizer
        )

        self.tracker = MetricsTracker(log_freq=self.config.wandb.log_freq)
        self.NOISE = torch.randn(16, self.config.model.lat_dim, device=self.device)

        

    def setup_optimizers(self):
        config = self.config.optimizer

        G_lr = self.config.optimizer.G_lr
        D_lr = self.config.optimizer.D_lr
        # D_lr = config.D_lr if D_lr in config else G_lr / config.

        self.G_optimizer = optim.Adam(
            self.G.parameters(),
            lr=G_lr,
            betas=(
                self.config.training.beta1,
                self.config.training.beta2
                ),
        )

        self.D_optimizer = optim.Adam(
            self.D.parameters(),
            lr=D_lr,
            betas=(
                self.config.training.beta1,
                self.config.training.beta2
                ),
        )

        total_steps = len(self.dataloader) * len(self.dataloader.dataset)
        self.G_scheduler = setup_scheduler(self.G_optimizer, total_steps, self.config)
        self.D_scheduler = setup_scheduler(self.D_optimizer, total_steps * self.n_critic, self.config)


    def setup_loss_and_regs(self):
        self.criterion = setup_loss(
            self.config.loss,
            D=self.D if self.config.loss.criterion == "wgan_gp" else None
        )


    def train_step(self, real_images):
        # safe to have out of the loop as long as we're using drop_last in the loader
        bs = real_images.size(0)
        
        for _ in range(self.n_critic):
            noise = torch.randn(bs, self.config.model.lat_dim, device=self.device)
            # using the same samples for multiple D steps, at least let's introduce some augs
            if self.ada:
                real_images = self.ada(real_images)

            D_loss, real_acc, fake_acc, real_logits = self.D_train_step(noise, real_images)

            if self.ada:
                self.ada.update(real_acc)

            self.D_scheduler.step()
        
        noise = torch.randn(bs, self.config.model.lat_dim, device=self.device)
        G_loss = self.G_train_step(noise, real_logits.detach())
        self.G_scheduler.step()

        return {
            "G_loss": G_loss,
            "D_loss": D_loss,
            "real_acc": real_acc,
            "fake_acc": fake_acc,
        }


    def D_train_step(self, noise, real_images):
        self.D.zero_grad()
        with autocast(device_type=self.device.type):
            real_images.detach().requires_grad_(True)
            real_logits = self.D(real_images)

            with torch.no_grad():
                fake_images = self.G(noise).detach()
            fake_logits = self.D(fake_images)
            
            D_loss = self.criterion.discriminator_loss(fake_logits, real_logits)
            if self.config.loss.criterion == "wgan_gp":
                D_loss += self.criterion.gradient_penalty(fake_images, real_images)
        
        self.D_scaler.scale(D_loss).backward()
        self.D_scaler.step(self.D_optimizer)
        self.D_scaler.update()

        with torch.no_grad():
            fake_acc = (torch.tanh(fake_logits) < 0).float().mean().item()
            real_acc = (torch.tanh(real_logits) > 0).float().mean().item()

        return D_loss.item(), fake_acc, real_acc, real_logits.detach()

    def G_train_step(self, noise, real_logits):
        self.G.zero_grad()

        with autocast(device_type=self.device.type):
            fake_images = self.G(noise)
            fake_logits = self.D(fake_images)

            G_loss = self.criterion.generator_loss(fake_logits, real_logits)
        self.G_scaler.scale(G_loss).backward()
        self.G_scaler.step(self.G_optimizer)
        self.G_scaler.update()
        
        return G_loss.item()


    def train_epoch(self, epoch, epochs):
        start_time = time.time()
        self.G.train()
        self.D.train()
        self.tracker.reset()

        pbar = tqdm(self.dataloader, desc=f"Epoch {epoch}/{epochs}: ")

        for batch_idx, real_images in enumerate(pbar):
            real_images = real_images.to(self.device, memory_format=torch.channels_last)
            step_metrics = self.train_step(real_images)
            self.tracker.log(step_metrics, batch_idx, pbar=pbar)
            
        return {**self.tracker.averages(), "epoch_time": time.time() - start_time}


    def train(self):
        # main training loop script
        for epoch in range(1, self.config.training.epochs + 1):
            epoch_metrics = self.train_epoch(epoch, self.config.training.epochs)

            self.G_scheduler.step(epoch_call=True)
            self.D_scheduler.step(epoch_call=True)

            if not epoch % self.config.training.sample_every:
                self.generate_samples(epoch)

            if not epoch % self.config.training.save_every:
                self.checkpoint_manager.save(
                    epoch,
                    epoch_metrics
                )
                

    @torch.no_grad()
    def generate_samples(self, epoch):
        self.G.eval()
        
        sample_grid = generate_sample_images(
            self.G,
            self.NOISE
        )

        
        sample_path = os.path.join(self.config.sample_dir, f"epoch_{epoch:04d}.png")
        save_sample_images(sample_grid, sample_path, rows=4)
        wandb.log({f"sample_{epoch:03f}": wandb.Image(make_grid(sample_grid, nrow=4, normalize=True, value_range=(0, 1)))})
        self.G.train()


@hydra.main(config_path="config", config_name="defaults.yaml", version_base=None)
def main(config):
    print(OmegaConf.to_yaml(config))
    print("\n"*4)
    wandb.init(
        project="GANs",
        name=f"GAN_run_{int(time.time())}",
        config=OmegaConf.to_container(config, resolve=True),
        reinit=True
    )
    Trainer(config).train()
    if wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
