# starting with the basics

import torch
from torch import nn, autograd
import torch.nn.functional as F



class Loss:
    def generator_loss(self, fake_logits):
        raise NotImplementedError
    
    def discriminator_loss(self, fake_logits, real_logits):
        raise NotImplementedError


class BCELoss(Loss):
    def __init__(self, label_smoothing=0.):
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
    

class WGANGPLoss(Loss):
    def __init__(self, lambda_gp=10, D=None):
        self.lambda_ = lambda_gp
        self.D # a reference to the Discriminator

    def generator_loss(self, fake_logits):
        return -fake_logits.mean()

    def discriminator_loss(self, fake_logits, real_logits):
        return - fake_logits.mean() + real_logits.mean()
    
    def gradient_penalty(self, D, fake_samples, real_samples):
        bs = real_samples.size(0)
        device = real_samples.device

        alpha = torch.randn(bs, 1, 1, 1, device=device)
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
    def __init__(self):
        self.criterion = nn.BCEWithLogitsLoss()

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
    


LOSSES = {
    "bce": BCELoss,
    "wgan-gp": WGANGPLoss,
    "ragan": RelavisticAverageGANLoss
}

def setup_criterion(config):
    # temp
    return LOSSES[config.get("criterion", "bce")]()
    

class R1R2Regularizer:
    def __init__(self, lambda_r1=10):
        self.lambda_ = lambda_r1

    def __call__(self, fake_logits, real_logits, fake_samples, real_samples):
        fake_samples.require_grad_(True)
        real_samples.require_grad_(True)

        return self.lambda_ * (
            .5 * self.r_gp(fake_logits, fake_samples) +
            .5 * self.r_gp(real_logits, real_samples)
        )


    def r_gp(self, logits, samples):
        grads = autograd.grad(
            outputs=logits,
            inputs=samples,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        return grads.view(
            grads.size(0), -1
            ).norm(2, dim=1).pow(2).mean()
    


# class PathLengthREgularizer:
#     def __init__(self, lambda_path_len=2, path_len_decay=1e-2):
#         self.lambda_ = lambda_path_len
#         self.decay_ = path_len_decay
#         self.mean_ = torch.ones(1)

#     def __call__(self, fake_images, lat_):
