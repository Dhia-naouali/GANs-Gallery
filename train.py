import torch
from torch import optim
from torch.amp import GradScaler, autocast

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
    EMA
)
from src.models import setup_models
from src.data import setup_dataloader, AdaptiveDiscriminatorAugmentation
from src.losses import setup_loss, R1Regularizer, PathLengthREgularizer
from evaluate import Evaluator

torch.set_default_device("cuda:0")
device = torch.device("cuda:0")
torch._functorch.config.donated_buffer = False


def _assert_finite(t, name):
    if torch.isnan(t).any() or torch.isinf(t).any():
        raise RuntimeError(f"{name} contains NaN/Inf")
    
    
    
    
def _nan_hook(self):
    def _hook(module, inp, out):
        # works for both single tensor and tuple outputs
        tensors = out if isinstance(out, (tuple, list)) else (out,)
        for i, t in enumerate(tensors):
            if isinstance(t, torch.Tensor):
                _assert_finite(t, f"{module.__class__.__name__}_out[{i}]")
    return _hook

class Trainer:
    def __init__(self, config):
        self.config = config
        seed_all()
        setup_directories(self.config)


        # weigth init within
        self.G, self.D = setup_models(config.model)
        self.G_ema = EMA(self.G)

        if self.config.training.compile:
            self.G = torch.compile(self.G, mode="max-autotune-no-cudagraphs")
            self.D = torch.compile(self.D, mode="max-autotune-no-cudagraphs")
            print("#"*40)
            print("models compiled")
            print("#"*40)
        
        print(self.G)
        print(self.D)
        for net, name in [(self.G, "G"), (self.D, "D")]:
            for m in net.modules():
                m.register_forward_hook(_nan_hook(self))
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
        self.evaluator = Evaluator(self.G, self.dataloader, config, self.batch_size, device)
        self.tracker = MetricsTracker(log_freq=self.config.wandb.log_freq)
        self.NOISE = torch.randn(16, self.config.model.lat_dim)
        
        self.penalties_stream = torch.cuda.Stream()
        self.losses_stream = torch.cuda.Stream()
        self.G_loss_computed = torch.cuda.Event()
        self.plp_computed = torch.cuda.Event()


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

        self.gradient_penalty_ = self.config.loss.get("gradient_penalty", False)


    def train_step(self, real_images):
        try:
            for _ in range(self.n_critic):
                D_loss, fake_acc, real_acc, real_logits = self.D_train_step(real_images)
            G_loss = self.G_train_step(real_logits.detach())
            self.G_ema.update()
            return {"G_loss": G_loss, "D_loss": D_loss,
                    "real_acc": real_acc, "fake_acc": fake_acc}

        except RuntimeError as e:
            if "NaN/Inf" in str(e):
                print(f"NaN detected: {e}")
                # save checkpoint for inspection
                torch.save({
                    "epoch": getattr(self, "epoch", 0),
                    "batch": getattr(self, "batch_idx", 0),
                    "G_state": self.G.state_dict(),
                    "D_state": self.D.state_dict(),
                    "real_images": real_images,
                }, f"nan_dump_{int(time.time())}.pt")
                raise Exception(real_images.min(), real_images.max())
            else:
                raise Exception(real_images.min(), real_images.max())

    def D_train_step(self, real_images):
        self.D.zero_grad(set_to_none=True)
        r1_penalty = torch.tensor(0)
        gradient_penalty = torch.tensor(0)

        with autocast(device_type="cuda", enabled=False):
            noise = torch.randn(self.batch_size, self.G.lat_dim)
            real_images = self.ada(real_images) if self.ada else real_images
            real_logits = self.D(real_images)
            
            with torch.no_grad():
                fake_images = self.ada(self.G(noise), real_acc=self.real_acc) \
                    if self.ada else self.G(noise)

            fake_logits = self.D(fake_images)
            D_loss = self.criterion.discriminator_loss(fake_logits, real_logits)
            _assert_finite(D_loss, "D_loss")

            if self.r1_regularizer:
                r1_penalty = self.r1_regularizer(real_logits, real_images)
            if self.gradient_penalty_:
                gradient_penalty = self.criterion.gradient_penalty(fake_images, real_images)


            D_loss += r1_penalty + gradient_penalty        
            self.D_scaler.scale(D_loss).backward()
            self.D_scaler.unscale_(self.D_optimizer)
            torch.nn.utils.clip_grad_norm_(self.D.parameters(), max_norm=1.0)
            self.D_scaler.step(self.D_optimizer)
            self.D_scaler.update()
            self.D_scheduler.step()
            
        with torch.no_grad():
            fake_acc = (fake_logits < 0).float().mean().item()
            real_acc = (real_logits > 0).float().mean().item()

        self.real_acc = real_acc
        return D_loss.item(), fake_acc, real_acc, real_logits


    def G_train_step(self, real_logits):
        self.G.zero_grad(set_to_none=True)
        path_length_penalty = torch.tensor(0)
        
        with autocast(device_type="cuda", enabled=False):
            noise = torch.randn(self.batch_size, self.G.lat_dim)
            fake_images = self.G(noise)
            fake_logits = self.D(fake_images)
            G_loss = self.criterion.generator_loss(fake_logits, real_logits)
            _assert_finite(G_loss, "G_loss")
            
            if self.path_length_regularizer:
                w = self.G.mapper(noise)
                fake_images_ = self.G.synthesis(w)
                path_length_penalty = self.path_length_regularizer(fake_images_, w)

        G_loss += path_length_penalty
        self.G_scaler.scale(G_loss).backward()
        self.G_scaler.unscale_(self.G_optimizer)
        torch.nn.utils.clip_grad_norm_(self.G.parameters(), max_norm=1.0)
        self.G_scaler.step(self.G_optimizer)
        self.G_scaler.update()
        self.G_scheduler.step()

        return G_loss.item()

    def _train_step(self, real_images):
        # sus list: set_to_none=True
        self.G.zero_grad(set_to_none=True)
        self.D.zero_grad(set_to_none=True)
        path_length_penalty = torch.tensor(0)
        r1_penalty = torch.tensor(0)
        gradient_penalty = torch.tensor(0)

        noise = torch.randn(self.batch_size, self.G.lat_dim)

        with torch.cuda.stream(self.losses_stream):
            with autocast(device_type="cuda", enabled=False):
                if self.ada:
                    real_images = self.ada(real_images, real_acc=self.real_acc)

                #################################################################
                # G loss
                #################################################################
                real_logits = self.D(real_images)
                real_logits.record_stream(self.penalties_stream)
                real_images.record_stream(self.penalties_stream)

                with torch.no_grad():   
                    fake_images = self.ada(self.G(noise)) if self.ada else self.G(noise)
                fake_images.record_stream(self.penalties_stream)

                fake_logits = self.D(fake_images)
                D_loss = self.criterion.discriminator_loss(fake_logits, real_logits)

                self.G_loss_computed.record(self.losses_stream)
                self.penalties_stream.wait_event(self.plp_computed)
                #################################################################
                # G loss
                #################################################################
                fake_images_G = self.G(noise)
                fake_logits_G = self.D(fake_images_G)
                G_loss = self.criterion.generator_loss(fake_logits_G, real_logits)


        if self.path_length_regularizer or self.r1_regularizer or self.gradient_penalty_:
            with torch.cuda.stream(self.penalties_stream):
                #################################################################
                # path length penalty, penalties stream
                #################################################################
                if self.path_length_regularizer:
                    w = self.G.mapping(noise)
                    fake_images_ = self.G.synthesis(w)
                    path_length_penalty = self.path_length_regularizer(fake_images_, w)

                self.plp_computed.record(self.penalties_stream)
                self.penalties_stream.wait_event(self.G_loss_computed)
                #################################################################
                # R1 & Gradient penalty, penalties stream
                #################################################################
                if self.r1_regularizer:
                    r1_penalty = self.r1_regularizer(real_logits, real_images)
                if self.gradient_penalty_:
                    gradient_penalty = self.criterion.gradient_penalty(fake_images.detach(), real_images)


        main_stream = torch.cuda.current_stream()
        main_stream.wait_stream(self.losses_stream)
        main_stream.wait_stream(self.penalties_stream)

        D_loss += r1_penalty + gradient_penalty
        G_loss += path_length_penalty
        
        
        self.D_scaler.scale(D_loss).backward()
        torch.nn.utils.clip_grad_norm_(self.D.parameters(), max_norm=1.0)
        self.D_scaler.step(self.D_optimizer)
        self.D_scaler.update()
        self.D_scheduler.step()

        with torch.no_grad():
            fake_acc = (fake_logits < 0).float().mean().item()
            real_acc = (real_logits > 0).float().mean().item()

        self.real_acc = real_acc
        self.G_scaler.scale(G_loss).backward()
        torch.nn.utils.clip_grad_norm_(self.G.parameters(), max_norm=1.0)
        self.G_scaler.step(self.G_optimizer)
        self.G_scaler.update()
        self.G_scheduler.step()


        return {
            "G_loss": G_loss.item(),
            "D_loss": D_loss.item(),
            "real_acc": real_acc,
            "fake_acc": fake_acc,
        }


    def train_epoch(self, epoch, epochs):
        start_time = time.time()
        self.G.train()
        self.D.train()
        self.tracker.reset()

        pbar = tqdm(self.dataloader, desc=f"Epoch {epoch}/{epochs}: ")

        for batch_idx, real_images in enumerate(pbar):
            real_images = real_images[0]["images"]
            real_images = real_images.float()
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
            
            if not epoch % self.config.training.evaluate_every:
                self.G_ema.apply_moving()
                self.G.eval()
                self.D.eval()
                evals = self.evaluator.evaluate(len(self.dataloader))
                wandb.log(evals)
                self.G_ema.restore()
                self.G.train()
                self.D.train()

    @torch.no_grad()
    def generate_samples(self, epoch):
        
        self.G_ema.apply_moving()
        sample_grid = generate_sample_images(
            self.G,
            self.NOISE
        )

        
        sample_path = os.path.join(self.config.sample_dir, f"epoch_{epoch:04d}.png")
        save_sample_images(sample_grid, sample_path, rows=4)
        wandb.log({f"sample_{epoch:03f}": wandb.Image(make_grid(sample_grid, nrow=4, normalize=True, value_range=(0, 1)))})
        self.G_ema.restore()


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