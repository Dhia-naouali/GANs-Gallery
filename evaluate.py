# pip install torchmetrics[image]
import torch
from torchmetrics.image import (
    FrechetInceptionDistance,
    InceptionScore,
    LearnedPerceptualImagePatchSimilarity,
    KernelInceptionDistance
)
from tqdm import tqdm


class Evaluator:
    def __init__(self, G, dataloader, config, batch_size, device): # may be used post training on cpu (ig ?)
        self.G = G.eval()
        self.dataloader = dataloader
        self.config = config
        self.batch_size = batch_size
        self.device = device
        
        self.FID = FrechetInceptionDistance(normalize=True).to(device)
        self.IS = InceptionScore(normalize=True).to(device)
        self.KID = KernelInceptionDistance(normalize=True).to(device)
        self.LPIPS = LearnedPerceptualImagePatchSimilarity(normalize=True).to(device)


    @torch.no_grad
    def generate_samples(self, num_batches):
        for _ in num_batches:            
            noise = torch.randn(
                self.batch_size,
                self.config.model.lat_dim,
                device=self.device
            )
            
            samples = self.G(noise)
            yield (samples * .5 + .5).byte()
    

    def load_samples(self, dataloader, num_batches):
        for _ in tqdm(num_batches):
            samples = next(dataloader)[0]["image"]
            yield (samples * .5 + .5).byte()
    
    
    @torch.no_grad()
    def evalute(self, num_batches):
        self.FID.reset()
        self.IS.reset()
        self.KID.reset()
        lpips_score = 0

        fake_gen = self.generate_samples(num_batches)
        real_gen = self.load_samples(num_batches)


        for _ in tqdm(range(num_batches)):
            fake_images = next(fake_gen)
            real_images = next(real_gen)


            self.FID.update(fake_images, real=False)
            self.FID.update(real_images, real=True)
            self.IS.update(fake_images)
            self.KID.update(fake_images, real=False)
            self.KID.update(real_images, real=True)

            lpips_score += self.LPIPS(fake_images, real_images).mean().item()


        fid_score = self.FID.compute().item()
        mean_is, std_is = self.IS.compute()
        mean_kid, std_kid = self.KID.compute()
        lpips_score /= num_batches

        
        return {
            "FID": fid_score,
            "IS_mean": mean_is,
            "IS_std": std_is,
            "KID_mean": mean_kid,
            "KID_mean": std_kid,
            "LPIPS": lpips_score,
        }
        

if __name__ == "__main__":
    from src.models import *
    def main(checkpoint_path):
        ...