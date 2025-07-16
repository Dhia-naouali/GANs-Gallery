import torch
from torch import optim
from torch.cuda.amp import GradScaler, autocast

import hydra
from omegaconf import DictConfig, OmegaConf

from src.utils import (
    seed_all,
    init_directories,
    count_params,
    Scheduler,

)
from src.models import init_models
from src.data import create_dataloader



class Trainer:
    def __init__(self, config):
        # seed
        # init directories: samples, chekcpoints, ...
        # init models
        # print param counts
        # setup optimizers
        # setup loss
        # setup loader
        # ADA ?
        # scalers
        # checkpoint manager
        # compile ?
        # metrics
        # fixed noise sample



        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available else "cpu") # someone's CPU goin down 💀
        seed_all()

        init_directories()

        self.G, self.D = init_models(config.model)
        self.G.to(self.device); self.D.to(self.device)

        # proper weights_init (muP ?)

        print(f"Generator: {count_params(self.G) * 1e-6:.2f} \n"
              f"Discriminator: {count_params(self.D) * 1e-6:.2f}")
        
        self.setup_optimizers()
        self.setup_loss()

        self.dataloader = create_dataloader(
            root_dir=config.data.root_dir
        )

        self.G_scaler = GradScaler()
        self.D_scaler = GradScaler()

        

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
        ...