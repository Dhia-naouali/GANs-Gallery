import torch
from torch import nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.q = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.k = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.v = nn.Conv2d(in_channels, in_channels // 8, 1)

        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        bs, c, h, w = x.size()

        query = self.q(x).view(bs, -1, h*w).permute(0, 2, 1) # .T eq
        key = self.k(x).view(bs, -1, h*w)
        value = self.v(x).view(bs, -1, h*w)

        attention = F.softmax(torch.bmm(query, key), dim=-1)
        out = torch.bmm(value, attention.permute(0, 2, 1)).view(bs, c, h, w)
        return self.alpha * out + x