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
    def __init__(self, G, config, device): # may be used post training on cpu (ig ?)
        self.G = G.eval()
        self.config = config
        self.device = device
        
        self.FID = FrechetInceptionDistance(normalize=True).to(device)
        self.IS = InceptionScore(normalize=True).to(device)
        self.KID = KernelInceptionDistance(normalize=True).to(device)
        self.LPIPS = LearnedPerceptualImagePatchSimilarity(normalize=True).to(device)


    @torch.no_grad
    def generate_samples(self, batch_size, num_batches):
        for _ in num_batches:            
            noise = torch.randn(
                batch_size,
                self.config.model.lat_dim,
                device=self.device
            )
            
            samples = self.G(noise)            
            yield (samples * .5 + .5).byte()
    
    def load_samples(self, dataloader, batch_size, num_batches):

        for _ in tqdm(num_batches):
            # to make sure that the batch size is the same used in the loader
            samples = next(dataloader)[0]["image"] # to check if compatible with dali
            yield (samples * .5 + .5).byte()
            
            
    def compute_fid(self, fake_images, real_images):
        self.FID.reset()
        self.FID.update(fake_images, real=False)
        self.FID.update(real_images, real=True)
        fid_mean, fid_std = self.FID.compute().item()
        return fid_mean.item(), fid_std.item()

        
    def compute_inception_score(self, fake_images):
        self.IS.reset()
        self.IS.update(fake_images)
        return self.IS.compute()
        
    
    def compute_kid(self, fake_images, real_images):
        self.KID.reset()
        self.KID.update(fake_images, real=False)
        self.KID.update(real_images, real=True)
        kid_mean, kid_std =  self.KID.compute()
        return kid_mean.item(), kid_std.item()
        
        
    @torch.no_grad        
    def compute_lpips_diversity(self, fake_images, real_images):
        return self.LPIPS(fake_images, real_images)
    
    def evaluta(self, dataloader, batch_size, num_batches):
        fake_images = self.generate_samples()
        real_images = self.load_samples()


        for humm in num_batches:
            fid = self.compute_fid(fake_images, real_images)
            inception_score = self.compute_inception_score(fake_images)
            lpips = self.compute_lpips_diversity(fake_images, real_images)
            kid = self.compute_kit(fake_images, real_images)
        
        return {
            "FID": ,
            "IS": ,
            "KID": ,
            "LPIPS": ,
        }
        
        
def main():
    # to use post training, using cli args ?, checkpoint path, ...
    ...