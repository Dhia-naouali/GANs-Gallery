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
from omegaconf import OmegaConf

from src.utils import (
    seed_all,
    setup_directories,
    count_params,
    setup_scheduler,
    MetricsTracker,
    CheckpointManager,
    generate_sample_images,
    save_sample_images,
    init_model_params,
)
from src.models import setup_models
from src.data import setup_dataloader, AdaptiveDiscriminatorAugmentation
from src.losses import setup_loss, R1Regularizer, PathLengthREgularizer


torch.set_default_device("cuda:0")
device = torch.device("cuda:0")


class Trainer:
    def __init__(self, config):
        # compile ?

        self.config = config
        seed_all() # seed dali
        setup_directories(self.config)


        # (muP ?), hold on buddy brb
        self.G, self.D = setup_models(config.model)
        init_model_params(self.G)
        init_model_params(self.D)
        if self.config.training.compile:
            self.G = torch.compile(self.G, mode="max-autotune-no-cudagraphs")
            self.D = torch.compile(self.D, mode="max-autotune-no-cudagraphs")
        
        print(f"Generator: {count_params(self.G) * 1e-6:.2f} \n"
              f"Discriminator: {count_params(self.D) * 1e-6:.2f}")



        # compiling ada didn't go well
        self.ada = AdaptiveDiscriminatorAugmentation(
            # target__real_acc=config.ADA.ada_target_acc
        ) if config.ADA.use_ADA else None
        self.real_acc = None

        self.dataloader = setup_dataloader(config)
        self.batch_size = self.config.training.batch_size

        self.n_critic = self.config.training.get("n_critic", 1)
        self.setup_optimizers()
        self.setup_loss_and_regs()

        self.G_scaler = GradScaler()
        self.D_scaler = GradScaler()

        self.checkpoint_manager = CheckpointManager(
            self.config.checkpoint_dir,
            self.G,
            self.D,
            self.G_optimizer,
            self.D_optimizer
        )

        self.tracker = MetricsTracker(log_freq=self.config.wandb.log_freq)
        self.NOISE = torch.randn(16, self.config.model.lat_dim)
        
        self.stream1 = torch.cuda.Stream()
        self.stream2 = torch.cuda.Stream()

        self.g_loss_computed = torch.cuda.Event()
        self.fake_images_generated = torch.cuda.Event()
        self.real_logits_r1_computed = torch.cuda.Event()


        

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

        total_steps = self.dataloader._size
        self.G_scheduler = setup_scheduler(self.G_optimizer, total_steps, self.config)
        self.D_scheduler = setup_scheduler(self.D_optimizer, total_steps * self.n_critic, self.config)


    def setup_loss_and_regs(self):
        self.criterion = setup_loss(
            self.config,
            D=self.D if self.config.loss.criterion == "wgan_gp" else None
        )

        self.r1_regularizer = None
        self.path_length_regularizer = None
        self.gradient_penalty_ = None

        if r1_penalty := self.config.loss.get("r1_penalty", 0):
            self.r1_regularizer = R1Regularizer(r1_penalty)

        if path_length_penalty := self.config.loss.get("path_length_penalty", 0):
            self.path_length_regularizer = PathLengthREgularizer(path_length_penalty)

        if gradient_penalty_ := self.config.loss.get("gradient_penalty", False):
            self.gradient_penalty_ = True
        
        
    def train_step(self, real_images):
        self.G.zero_grad()
        self.D.zero_grad()

        noise = torch.randn(self.batch_size, self.G.lat_dim)
        if self.ada:
            real_images = self.ada(real_images, self.real_acc)
        
        real_logits = None
        with torch.cuda.stream(self.stream1), autocast(device_type="cuda"):
                fake_images = self.G(noise)
                if self.ada:
                    fake_images = self.ada(fake_images, real_acc=self.real_acc)
                self.fake_images_generated.record(self.stream1)

                fake_logits = self.D(fake_images)

                self.stream1.wait_event(self.real_logits_r1_computed)
                self.G_loss = self.criterion.generator_loss(fake_logits, real_logits)                
                self.g_loss_computed.record(self.stream1)

                D_loss = self.criterion.discriminator_loss(fake_logits, real_logits)


        with torch.cuda.stream(self.stream2), autocast(device_type="cuda"):
                real_logits = self.D(real_images)
                if self.r1_regularizer:
                    r1_penalty = self.r1_regularizer(real_logits, real_images)
                self.real_logits_r1_computed.record(self.stream2)
                
                if self.gradient_penalty_:
                    self.stream2.wait_event(self.fake_images_generated)
                    gradient_penalty = self.criterion.gradient_penalty(fake_images, real_images)
                
                if self.path_length_regularizer:
                    self.stream2.wait_event(self.g_loss_computed)
                    path_length_penalty = self.path_length_regularizer(fake_images, self.G._w)

        main_stream = torch.cuda.current_stream()

        main_stream.wait_stream(self.stream1)
        main_stream.wait_stream(self.stream2)

        if self.gradient_penalty_:
            D_loss += gradient_penalty
        if self.r1_regularizer:
            D_loss += r1_penalty
        
        if self.path_length_regularizer:
            G_loss += path_length_penalty
            
            
        self.D_scaler.scale(D_loss).backward()
        self.D_scaler.step(self.D_optimizer)
        self.D_scaler.update()

        with torch.no_grad():
            fake_acc = (fake_logits < 0).float().mean().item()
            real_acc = (real_logits > 0).float().mean().item()

        self.real_acc = real_acc
        self.G_scaler.scale(G_loss).backward()
        self.G_scaler.step(self.G_optimizer)
        self.G_scaler.update()
        
        
        return {
            "G_loss": G_loss.item(),
            "D_loss": D_loss.item(),
            "real_acc": real_acc,
            "fake_acc": fake_acc,
        }


    def _train_step(self, real_images):
        # safe to have out of the loop as long as we're using drop_last in the loader
                    
        noise = torch.randn(self.batch_size, self.config.model.lat_dim, dtype=torch.float16)
        D_loss, fake_acc, real_acc, real_logits = self.D_train_step(noise, real_images)
        self.D_scheduler.step()
        
        noise = torch.randn(self.batch_size, self.config.model.lat_dim)
        G_loss = self.G_train_step(noise, real_logits)
        self.G_scheduler.step()

        return {
            "G_loss": G_loss,
            "D_loss": D_loss,
            "real_acc": real_acc,
            "fake_acc": fake_acc,
        }


    def D_train_step(self, noise, real_images):
        self.D.zero_grad()

        with autocast(device_type="cuda"):
            if self.ada:
                real_images = self.ada(real_images, real_acc=self.real_acc)

            real_images = real_images.detach().requires_grad_(True)
            real_logits = self.D(real_images)

            with torch.no_grad():
                fake_images = self.G(noise)
                fake_images = self.ada(fake_images, real_acc=self.real_acc).detach()
            fake_logits = self.D(fake_images)
            
            D_loss = self.criterion.discriminator_loss(fake_logits, real_logits)
            if self.config.loss.criterion == "wgan_gp":
                D_loss += self.criterion.gradient_penalty(fake_images, real_images)
        
            if self.r1_regularizer:
                D_loss += self.r1_regularizer(real_logits, real_images)
        
        self.D_scaler.scale(D_loss).backward()
        self.D_scaler.step(self.D_optimizer)
        self.D_scaler.update()

        with torch.no_grad():
            fake_acc = (fake_logits < 0).float().mean().item()
            real_acc = (real_logits > 0).float().mean().item()

        self.real_acc = real_acc
        return D_loss.item(), fake_acc, real_acc, real_logits

    def G_train_step(self, noise, real_logits):
        self.G.zero_grad()

        with autocast(device_type="cuda"):
            fake_images = self.G(noise)
            fake_logits = self.D(fake_images)

            G_loss = self.criterion.generator_loss(fake_logits, real_logits)
            if self.path_length_regularizer:
                G_loss += self.path_length_regularizer(fake_images, self.G._w)

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
            real_images = real_images[0]["images"]
            real_images = real_images
            step_metrics = self.train_step(real_images)
            self.tracker.log(step_metrics, batch_idx, pbar=pbar)
            
        return {**self.tracker.averages(), "epoch_time": time.time() - start_time}


    def train(self):
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
