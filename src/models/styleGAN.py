import torch
from torch import nn
import torch.nn.functional as F

from ..utils import init_weights


class EqualizedLR:
    # "weight" scaling at run time !!! sweeet
    def __init__(self, module, gain=1):
        self.module = module
        self.gain = gain
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.module.weight)
        if self.module.bias is not None:
            nn.init.zeros_(self.module.bias)
        fan_in = self.module.weight[0].numel()
        self.scale = self.gain / (fan_in ** .5)

    def forward(self, x):
        return self.module(x * self.scale)



class Mapper(nn.Module):
    def __init__(self, z_dim, w_dim, depth=4):
        super().__init__()
        self.eps = 1e-8
        self.layers = []
        for _ in range(depth):
            self.layers += [
                EqualizedLR(nn.Linear(z_dim, w_dim)),
                nn.LeakyReLU(.2),
            ]
            z_dim = w_dim
            
        self.layers = nn.Sequential(*self.layers)
    
    def forward(self, z):
        return self.layers(
            z / torch.norm(z, dim=1, keepdim=True) + self.eps
        )



class AdaIN(nn.Module):
    ...



class ModConv(nn.Module): 
    def __init__(self, in_channels, out_channels, kernel_size, style_dim, demodulate=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.demodulate = demodulate
        self.style_projector = EqualizedLR(nn.Linear(style_dim, in_channels), gain=1)
        self.weight = nn.Parameter(torch.randn(1, out_channels, in_channels, kernel_size, kernel_size))
        self.eps = 1e-8
        
    def forward(self, x, style_vector):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        b, c, h, w = x.shape
        
        style = self.style_projector(style_vector).view(b, 1, self.in_channels, 1, 1)
        weight = self.weight * style

        if self.demodulate:
            demod_coef = torch.rsqrt((weight ** 2).sum([2, 3, 4]) + self.eps)
            weight = weight * demod_coef.view(b, self.out_channels, 1, 1, 1)

        weight = weight.reshape(
            b * self.out_channels,
            self.in_channels,
            self.kernel_size,
            self.kernel_size,
        )
        x = x.reshape(1, b*c, h, w)

        out = F.conv2d(x, weight, padding=self.kernel_size//2, groups=b)
        return out.view(b, self.out_channels, h, w)


class NoiseInjector(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, channels, 1, 1))
        
    def forward(self, x):
        b, _, h, w = x.shape
        noise = torch.randn(b, 1, h, w, device=x.device)
        return x + self.weight * noise


class StyleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, style_dim, upsample=True):
        super().__init__()
        self.upsample = upsample
        self.conv = ModConv(in_channels, out_channels, 3, style_dim)
        self.noise_injector = NoiseInjector(out_channels)
        self.act = nn.LeakyReLU(.2)
        
    def forward(self, x, style):
        x = self.conv(x, style)
        x = self.noise_injector(x)
        return self.act(x)
    
    
class StyleGANG(nn.Module):
    def __init__(self, channels, lat_dim=256, w_dim=256):
        super().__init__()
        init_channels = channels[0]
        self.lat_dim = lat_dim
        self.mapper = Mapper(lat_dim, w_dim)
        self.init_canvas = nn.Parameter(torch.randn(1, init_channels, 4, 4))
        self.blocks = nn.ModuleList()
        self.rgbs = nn.ModuleList()

        for i in range(len(channels[1:])):
            out_channels = channels[i]
            self.blocks.append(
                nn.Sequential(
                    StyleBlock(init_channels, out_channels, w_dim),
                    StyleBlock(out_channels, out_channels, w_dim, upsample=i),
                )
            )

            self.rgbs.append(ModConv(out_channels, 3, 1, w_dim, demodulate=False))
            init_channels = out_channels
        
        init_weights(self)
        for layer in self.mapper.layers:
            if isinstance(layer, EqualizedLR):
                layer._init_weights()
        
        for block in self.blocks:
            for b in block:
                if isinstance(b, EqualizedLR):
                    b.conv.style_projector._init_weights()            

    
    def synthesis(self, w):
        B = w.size(0)
        x = self.init_canvas.expand(B, *([-1]*3))
        rgb = None

        for block, to_rgb in enumerate(zip(self.blocks, self.rgbs)):
            for b in block:
                x = b(x, w)
            rgb = F.interpolate(rgb, scale_factor=2, mode="bilinear", align_corners=True) \
                if rgb is not None else 0
            rgb += to_rgb(x, w)

        return torch.tanh(rgb)


    def forward(self, z):
        w = self.mapper(z)
        return self.synthesis(w)



class BatchSTD(nn.Module):
    def forward(self, x):
        b, _, h, w = x.shape
        std = x.std(dim=0, keepdim=True).mean()
        return torch.cat([
            x,
            std.expand(b, 1, h, w)
        ], dim=1)


class StyleGAND(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.bstd = BatchSTD()
        self.blocks = []

        in_channels = 3
        for out_channels in channels:
            self.blocks.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 3, padding=1),
                    nn.LeakyReLU(.2),
                    nn.Conv2d(out_channels, out_channels, 3, padding=1),
                    nn.LeakyReLU(.2),
                    nn.AvgPool2d(2)
                )
            )
            in_channels = out_channels
        self.blocks = nn.Sequential(*self.blocks)

        self.head = nn.Sequential(
            nn.Conv2d(in_channels + 1, in_channels, 3, padding=1),
            nn.LeakyReLU(.2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, 1)
        )
        init_weights(self)


    def forward(self, x):
        x = self.blocks(x)
        x = self.bstd(x)
        return self.head(x)