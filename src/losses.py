# starting with the basics

import torch
from torch import nn, autograd
import torch.nn.functional as F

class BCELoss:
    def __init__(self, label_smoothing=.1):
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss()
        self.label_smoothing = label_smoothing

    def generator_loss(self, fake_logits):
        labels = torch.ones_like(fake_logits)
        return self.criterion(fake_logits, labels)

    def discriminator_loss(self, real_logits, fake_logits):

        real_labels = torch.full_like(real_logits, 1.0 - self.label_smoothing)
        fake_labels = torch.zeros_like(fake_logits)

        real_loss = self.criterion(real_logits, real_labels)
        fake_loss = self.criterion(fake_logits, fake_labels)

        return (real_loss + fake_loss) * .5
    

class WGANGPLoss:
    def __init__(self, lambda_gp=10, device="cpu"):
        self.lambda_ = lambda_gp
        self.D # a reference to the Discriminator

    def generator_loss(self, fake_logits):
        return -fake_logits.mean()

    def discriminator_loss(self, fake_logits, real_logits):
        return - fake_logits.mean() + real_logits.mean()
    
    def gradient_penalty(self, D, fake_samples, real_samples):
        bs = real_samples.size(0)

        alpha = torch.randn(bs, 1, 1, 1, device=self.device)
        interpolated_x = alpha * real_samples + (1 - alpha) * fake_samples
        interpolated_x.requires_grad_(True)

        interpolated_logits = D(interpolated_x)

        grads = autograd.grad(
            outputs=interpolated_logits,
            inputs=interpolated_x,
            grad_outputs=torch.ones_like(interpolated_logits),
            create_graph=True,
            only_inputs=True,
            retain_graph=True, # in case using some of the other regs
        )[0]

        grads = grads.view(bs, -1)
        grad_norm = grads.norm(2, dim=1)
        return self.lambda_ * ((grad_norm - 1) ** 2).mean()
    



class RelavisticAverageGANLoss:
    def __init__(self, device="cpu"):
        self.criterion = nn.BCEWithLigtsLoss()

    def generator_loss(self, fake_logits, real_logits):
        real_loss = self.criterion(
            real_logits - fake_logits.mean(),
            torch.ones_like(real_logits)
        )

        fake_loss = self.criterion(
            fake_logits - real_logits.mean(),
            torch.zeros_like(real_logits)
        )

        return real_loss + fake_loss
    

    def discriminator_loss(self, fake_logits, real_logits):
        real_loss = self.self.criterion(
            real_logits - fake_logits.mean(),
            torch.zeros_like(real_logits)
        )

        fake_loss = self.criterion(
            fake_logits - real_logits.mean(),
            torch.ones_like(real_logits)
        )

        return real_loss + fake_loss
    

