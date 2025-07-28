import torch
from torch import nn, autograd
import torch.nn.functional as F



class Loss:
    def generator_loss(self, fake_logits):
        raise NotImplementedError
    
    def discriminator_loss(self, fake_logits, real_logits):
        raise NotImplementedError


class BCELoss(Loss):
    def __init__(self, config, label_smoothing=0.):
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss()
        self.label_smoothing = label_smoothing

    def generator_loss(self, fake_logits, real_logits):
        labels = torch.ones_like(fake_logits)
        return self.criterion(fake_logits, labels)

    def discriminator_loss(self, fake_logits, real_logits):
        real_labels = torch.full_like(real_logits, 1.0 - self.label_smoothing)
        fake_labels = torch.zeros_like(fake_logits)

        real_loss = self.criterion(real_logits, real_labels)
        fake_loss = self.criterion(fake_logits, fake_labels)

        return (real_loss + fake_loss) * .5
    

class WGANGPLoss(Loss):
    def __init__(self, config, lambda_gp=10, D=None):
        self.lambda_ = config.get("grad_penalty", 10)
        self.D = D

    def generator_loss(self, fake_logits, real_logits):
        return -fake_logits.mean()

    def discriminator_loss(self, fake_logits, real_logits):
        return fake_logits.mean() - real_logits.mean()
    
    def gradient_penalty(self, fake_samples, real_samples):
        bs = real_samples.size(0)
        device = real_samples.device

        alpha = torch.rand(bs, 1, 1, 1, device=device)
        interpolated_x = alpha * real_samples + (1 - alpha) * fake_samples
        interpolated_x.requires_grad_(True)

        interpolated_logits = self.D(interpolated_x)

        grads = autograd.grad(
            outputs=interpolated_logits,
            inputs=interpolated_x,
            grad_outputs=torch.ones_like(interpolated_logits),
            create_graph=True,
            only_inputs=True,
            retain_graph=True,
        )[0]

        grads = grads.reshape(bs, -1)
        grad_norm = grads.norm(2, dim=1)
        return self.lambda_ * ((grad_norm - 1) ** 2).mean()


class RelavisticAverageGANLoss(Loss):
    def __init__(self, config):
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
        real_loss = self.criterion(
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
    "wgan_gp": WGANGPLoss,
    "ragan": RelavisticAverageGANLoss
}


def setup_loss(config, D=None):
    if config.criterion == "wgan_gp":
        return LOSSES[config.get("criterion", "bce")](config, D=D)
    return LOSSES[config.get("criterion", "bce")](config)


class R1Regularizer:
    def __init__(self, lambda_r1=10):
        self.lambda_ = lambda_r1

    def __call__(self, real_logits, real_samples):
        real_samples.require_grad_(True)

        grads = autograd.grad(
            outputs=real_logits.sum(),
            inputs=real_samples,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        grads = grads.view(grads.size(0), -1).norm(2, dim=1).pow(2).mean()
        return self.lambda_ * grads    


class PathLengthREgularizer:
    def __init__(self, lambda_path_len=2, path_len_decay=1e-2):
        self.lambda_ = lambda_path_len
        self.decay_ = path_len_decay
        self.mean_ = torch.ones(1)

    def __call__(self, fake_images, w):
        y_hat = (
            (fake_images.size(2) * fake_images.size(3)) ** .5
        ) * torch.randn_like(fake_images)
        
        s = (fake_images * y_hat).sum()
        grads = autograd.grad(
            outputs=s,
            inputs=w,
            create_graph=True,
            only_inputs=True
        )[0].norm(2, dim=1)
        
        if self.plp_ema is not None:
            self.plp_ema = self.lambda_ema * self.plp_ema + (1 - self.lambda_ema) * grads.mean().detach()
        else:
            self.plp_ema = grads.mean().detach()
            
        return self.lambda_pl * ((grads - self.plp_ema) ** 2).mean()