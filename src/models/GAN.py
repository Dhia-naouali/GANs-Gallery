import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from ..utils import init_weights

class PixelNorm(nn.Module):
    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdims=True) + 1e-8)



class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.q = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.k = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.v = nn.Conv2d(in_channels, in_channels, 1)

        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        b, c, h, w = x.size()

        query = self.q(x).view(b, -1, h*w).permute(0, 2, 1)
        key = self.k(x).view(b, -1, h*w)
        value = self.v(x).view(b, -1, h*w)

        attention = F.softmax(torch.bmm(query, key), dim=-1)
        out = torch.bmm(value, attention.permute(0, 2, 1)).view(b, c, h, w)
        return self.alpha * out + x


class _Conv(nn.Module):
    NORMS = {
        "batch": nn.BatchNorm2d,
        "layer": nn.LayerNorm,
        "instance": nn.InstanceNorm2d,
        "pixel": PixelNorm,
        "none": nn.Identity
    }
    
    def __init__(self, conv, out_channels, norm, activation, leak, use_SN):
        super().__init__()
        if use_SN:
            conv = spectral_norm(conv)
        self.conv = conv

        self.norm = self.NORMS[norm](out_channels)
        match activation:
            case "relu":
                self.activation = nn.ReLU(inplace=True)
            case "leaky_relu":
                self.activation = nn.LeakyReLU(leak, inlace=True)
            case "elu":
                self.activation = nn.ELU(inplace=True)
            case "swich":
                self.activation = nn.SiLU(inplace=True)
            case _:
                raise Exception(f"invalid activation function config [{activation}]")
            

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return self.activation(x)


class ConvBlock(_Conv):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
            norm="batch",
            activation="elu",
            leak=.1,
            use_SN=True
        ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias=False
        )
        super().__init__(conv, out_channels, norm, activation, leak, use_SN)
    

    
class DeConvBlock(_Conv):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
            norm="batch",
            activation="elu",
            leak=.1,
            use_SN=True
        ):
        conv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias=False
        )

        super().__init__(conv, out_channels, norm, activation, leak, use_SN)
        


# renaming it to gang cuz it's funnier
class GANG(nn.Module):
    def __init__(
            self,
            lat_dim,
            channels,
            attention_layers=None,
            norm="batch",
            activation="elu",
            leak=.1,
            use_SA=False,
            use_SN=True,
        ):
        super().__init__()
        self.lat_dim = lat_dim
        self.attention_layers = attention_layers or []

        self.init_size = 4
        init_channels  = lat_dim // (self.init_size**2)

        self.projector = nn.Sequential(
            nn.Linear(lat_dim, init_channels * (self.init_size**2)),
            nn.BatchNorm1d(init_channels * (self.init_size**2)),
            nn.ReLU(inplace=True)
        )

        block_kwargs = {
            "kernel_size":4,
            "stride":2,
            "padding":1,
            "norm": norm,
            "activation": activation,
            "leak": leak,
            "use_SN": use_SN
        }


        self.layers = []
        in_channels = init_channels
        for i in range(len(channels)):
            out_channels = out_channels[i]
            self.layers.append(
                DeConvBlock(
                    in_channels,
                    out_channels,
                    **block_kwargs
                )
            )
            if use_SA and i in self.attention_layers:
                self.layers.append(SelfAttention(out_channels))
            in_channels = out_channels


        self.layers.append(
            nn.ConvTranspose2d(
                in_channels,
                3,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            )
        )
        self.layers = nn.Sequential(*self.layers)

        init_weights(self)


    def mapper(self, z):
        return self.projector(z).view(z.size(0), -1, self.init_size, self.init_size)

    def synthesis(self, w):
        return torch.tanh(self.layers(w))

    def forward(self, z):
        w = self.mapper(z)
        return self.synthesis(w)


class GAND(nn.Module):
    def __init__(
            self,
            channels,
            hidden_dim,
            depth,
            attention_layers=None,
            norm="batch",
            activation="elu",
            leak=.1,
            use_SN=True,
            use_SA=False
        ):
        super().__init__()
        self.attention_layers = attention_layers or []


        block_kwargs = {
                    "kernel_size":4,
                    "stride":2,
                    "padding":1,
                    "norm": norm,
                    "activation": activation,
                    "leak": leak,
                    "use_SN": use_SN
                }


        in_channels = 3
        self.layers = []
        for i in range(depth):
            out_channels = channels[i]
            self.layers.append(
                ConvBlock(
                    in_channels, out_channels, **block_kwargs
                )
            )
            if use_SA and i in self.attention_layers:
                self.layers.append(SelfAttention(out_channels))

            in_channels = out_channels
        self.layers = nn.Sequential(*self.layers)

        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, 1)
        )
        init_weights(self)


    def forward(self, x):
        x = self.layers(x)
        x = self.cls_head(x)
        return x
