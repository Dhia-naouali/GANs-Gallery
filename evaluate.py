import torch
from torchmetrics import (...)
from tqdm import tqdm


class Evaluator:
    def __init__(self, G, config, device): # may be used post training on cpu (ig ?)
        self.G = G.eval()
        self.config = config
        self.device = device
        
        self.FID = ...
        self.IS = ...
        self.KID = ...
        self.LPIPS = ...
        
        
    @torch.no_grad
    def generate_samples(self, batch_size, num_batches):
        for _ in num_batches:            
            noise = torch.randn(
                batch_size,
                self.config.model.lat_dim,
                device=self.device
            )
            
            samples = self.G(noise).byte()
            yield samples
    
    def load_samples(self, dataloader, batch_size, num_batches):

        for _ in tqdm(num_batches):
            # to make sure that the batch size is the same used in the loader
            samples = next(dataloader)[0]["image"] # to check if compatible with dali
            yield samples
            
            
    def compute_fid(self, fake_images, real_images):
        ...
        
        
    def compute_inception_score(self, fake_images, rela_images):
        ...
        
    
    def compute_kid(self, fake_images, real_images):
        ...
        
        
    def compute_lpips_diversity(self, fake_images, ...):
        ...
    
    def evaluta(self, dataloader, batch_size, num_batches):
        fake_images = self.generate_samples()
        real_images = self.load_samples()
        fid = self.compute_fid()
        inception_score = self.compute_inception_score()
        lpips = self.compute_lpips_diversity()
        kid = self.compute_kit()
        
        return {
            "FID": ,
            "IS": ,
            "KID": ,
            "LPIPS": ,
        }
        
        
def main():
    # to use post training, using cli args ?, checkpoint path, ...
    ...