import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

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


class Conv_(nn.Module):
    NORMS = {
        "batch": nn.BatchNorm2d,
        "layer": nn.LayerNorm,
        "instance": nn.InstanceNorm2d
    }
    
    def __init__(self, out_channels, norm, use_SN):
        super().__init__()
        if use_SN:
            self.conv = spectral_norm(self.conv)

        self.norm = self.NORMS[norm](out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return self.act(x)        


class ConvBlock(Conv_):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
            norm="batch",
            use_SN=True
        ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias=False
        )
        super().__init__(out_channels, norm, use_SN)
    

    
class DeConvBlock(Conv_):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
            norm="batch",
            use_SN=True
        ):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias=False
        )

        super().__init__(out_channels, norm, use_SN)



# renaming it to gang cuz it's funnier
class GANG(nn.Module):
    def __init__(self, lat_dim, hidden_dim, depth, attention_layers=None):
        super().__init__()
        self.lat_dim = lat_dim
        self.attention_layers = attention_layers or []

        self.init_size = 4
        init_channels  = hidden_dim * (2**(depth-1))

        self.projector = nn.Linear(lat_dim, init_channels * (self.init_size**2))

        block_kwargs = {
            "kernel_size":4,
            "stride":2,
            "padding":1,
        }

        self.layers = []
        in_channels = init_channels
        for i in range(depth-1):
            out_channels = hidden_dim * (2** (depth - i - 1))
            self.layers.append(
                DeConvBlock(
                    in_channels,
                    out_channels,
                    **block_kwargs
                )
            )
            in_channels = out_channels
        
        # last layer
        self.layers.append(
            nn.ConvTranspose2d(
                in_channels,
                3,
                **block_kwargs,
                bias=False
            )
        )
        self.layers = nn.Sequential(*self.layers)

    def forward(self, z):
        x = self.projector(z)
        x = x.view(x.size(0), -1, self.init_size, self.init_size)
        x = self.layers(x)
        return torch.tanh(x)



class GAND(nn.Module):
    def __init__(self, hidden_dim, depth, attention_layers=None):
        super().__init__()
        self.attention_layers = attention_layers or []


        block_kwargs = {
                    "kernel_size":4,
                    "stride":2,
                    "padding":1
                }


        in_channels = 3
        self.layers = []
        for i in range(depth):
            out_channels = hidden_dim * (2**i)
            self.layers.append(
                ConvBlock(
                    in_channels, out_channels, **block_kwargs
                )
            )

            in_channels = out_channels
        self.layers = nn.Sequential(*self.layers)

        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, 1)
        )
        

    def forward(self, x):
        x = self.layers(x)
        x = self.cls_head(x)
        return x