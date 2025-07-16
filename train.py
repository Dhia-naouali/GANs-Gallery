import torch
from src.utils import *
from src.models import init_models
from src.data import create_dataloader
from torch.cuda.amp import GradScaler, autocast
import hydra
from omegaconf import DictConfig, OmegaConf

class Trainer:
    def __init__(self, config):
        # seed
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
        self.device = torch.device(config.device if torch.cuda.is_available else "cpu") # someone's CPU gonna melt 💀
        seed_all()

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

        