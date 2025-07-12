# starting with the basics

import torch
from torch import nn
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