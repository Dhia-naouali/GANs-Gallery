import torch
from torch import nn, autograd
import torch.nn.functional as F



class Loss(nn.Module):
    def generator_loss(self, fake_logits):
        raise NotImplementedError
    
    def discriminator_loss(self, fake_logits, real_logits):
        raise NotImplementedError


class BCELoss(Loss):
    def __init__(self, batch_size=32, label_smoothing=0.):
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss()
        self.register_buffer(
            "real_labels", 
            torch.full(
                size=(batch_size, 1), 
                fill_value=1-label_smoothing
            )
        )

        self.register_buffer(
            "fake_labels", 
            torch.full(
                size=(batch_size, 1), 
                fill_value=label_smoothing
            )
        )

    def generator_loss(self, fake_logits, real_logits):
        return self.criterion(fake_logits, self._real_labels)

    def discriminator_loss(self, fake_logits, real_logits):
        real_loss = self.criterion(real_logits, self.real_labels)
        fake_loss = self.criterion(fake_logits, self.fake_labels)

        return (real_loss + fake_loss) * .5


class WGANGPLoss(Loss):
    def __init__(self, config, batch_size, label_smoothing, lambda_gp=10, D=None):
        super().__init__()
        self.batch_size = batch_size
        self.lambda_ = config.get("grad_penalty", 10)
        self.D = D
        self.register_buffer(
            "real_labels", 
            torch.full(
                size=(batch_size, 1), 
                fill_value=1-label_smoothing
            )
        )


    def generator_loss(self, fake_logits, real_logits):
        return -fake_logits.mean()

    def discriminator_loss(self, fake_logits, real_logits):
        return fake_logits.mean() - real_logits.mean()


    def gradient_penalty(self, fake_samples, real_samples):
        alpha = torch.rand(self.batch_size, 1, 1, 1)
        interpolated_x = alpha * real_samples + (1 - alpha) * fake_samples
        interpolated_x.requires_grad_(True)

        interpolated_logits = self.D(interpolated_x)

        grads = autograd.grad(
            outputs=interpolated_logits,
            inputs=interpolated_x,
            grad_outputs=self.real_labels,
            create_graph=True,
            only_inputs=True,
            retain_graph=True,
        )[0]

        grads = grads.reshape(self.batch_size, -1)
        grad_norm = grads.norm(2, dim=1)
        return self.lambda_ * ((grad_norm - 1) ** 2).mean()


class RelavisticAverageGANLoss(Loss):
    def __init__(self, config, batch_size, label_smoothing):
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss()
        self.register_buffer(
            "real_labels", 
            torch.full(
                size=(batch_size, 1), 
                fill_value=1-label_smoothing
            )
        )

        self.register_buffer(
            "fake_labels", 
            torch.full(
                size=(batch_size, 1),
                fill_value=label_smoothing
            )
        )
        

    def generator_loss(self, fake_logits, real_logits):
        real_loss = self.criterion(
            real_logits - fake_logits.mean(),
            self.real_labels
        )

        fake_loss = self.criterion(
            fake_logits - real_logits.mean(),
            self.fake_labels
        )

        return real_loss + fake_loss
    

    def discriminator_loss(self, fake_logits, real_logits):
        real_loss = self.criterion(
            real_logits - fake_logits.mean(),
            self.fake_labels
        )

        fake_loss = self.criterion(
            fake_logits - real_logits.mean(),
            self.real_labels
        )

        return real_loss + fake_loss


LOSSES = {
    "bce": BCELoss,
    "wgan_gp": WGANGPLoss,
    "ragan": RelavisticAverageGANLoss
}


def setup_loss(config, D=None):
    batch_size = config.training.get("batch_size", 32)
    label_smoothing = config.loss.get("label_smoothing", 0)
    lambda_gp = config.loss.get("grad_penalty", 0)
    
    loss_config = {
        "batch_size": batch_size,
        "label_smoothing": label_smoothing,
        "lambda_gp": lambda_gp
    }
    if D is not None:
        loss_config["D"] = D
    
    return LOSSES[config.get("criterion", "bce")](**loss_config)


class R1Regularizer:
    def __init__(self, lambda_r1=10):
        self.lambda_ = lambda_r1

    def __call__(self, real_logits, real_samples):
        real_samples.requires_grad_(True)

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
    def __init__(self, plp_lambda=2, plp_ema=.99):
        self.lambda_pl = plp_lambda
        self.lambda_ema = plp_ema
        self.plp_ema = None

    def __call__(self, fake_images, w):
        w.requires_grad_(True)

        y_hat = torch.randn_like(fake_images) * (
            (fake_images.size(2) * fake_images.size(3)) ** .5
        )
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