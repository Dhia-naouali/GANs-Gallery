import torch
from torch import nn
import torch.nn.functional as F



class Mapper(nn.Module):
    def __init__(self, z_dim, w_dim, depth=8):
        super().__init__()
        self.layers = []
        for _ in range(depth):
            self.layers += [
                nn.Linear(z_dim, w_dim),
                nn.LeakyReLU(.2),
            ]
            z_dim = w_dim
            
        self.layers = nn.Sequential(*self.layers)
    
    def forward(self, z):
        return self.layers(
            z / torch.norm(z, dim=1, keepdim=True)
        )



class AdaIN(nn.Module):
    def __init__(self, ):
        super().__init__()
        ...        
        
        
class ModConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, style_dim, demodulate=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.demodulate = demodulate
        self.style_projector = nn.Linear(style_dim, in_channels)
        self.weight = nn.Parameter(torch.randn(1, out_channels, in_channels, kernel_size, kernel_size))
        self.eps = 1e-8
        
    def forward(self, x, style_vector):
        b, c, h, w = x.shape
        style = self.style_projector(style_vector).view(b, 1, self.in_channels, 1, 1)
        W = self.weight * style

        if self.demodulate:
            demod_coef = torch.rsqrt((W ** 2).sum([2, 3, 4]) + self.eps)
            W = W * demod_coef.view(b, self.out_channels, 1, 1)
        W = W.view(
            b * self.out_channels,
            self.in_channels,
            self.kernel_size,
            self.kernel_size,
        )
        x = x.view(1, b*self.out_channels, c, h, w)

        out = F.conv2d(x, W, padding=self.kernel_size//2, groups=b)
        return out.view(b, self.out_channels, h, w)


class NoiseInjector(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, channels, 1, 1))
        
    def forward(self, x):
        b, _, h, w = x.shape
        noise = torch.randn(b, 1, h, w)
        return x + self.weight * noise

    
class StyleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, style_dim):
        super().__init__()
        self.conv = ModConv(in_channels, out_channels, 3, style_dim)
        self.noise_injector = NoiseInjector(out_channels)
        self.act = nn.LeakyReLU(.2)
        
    def forward(self, x, style):
        x = self.conv(x, style)
        x = self.noise_injector(x)
        return self.act(x)


class ToRGB(nn.Module):
    def __init__(self, in_channels, style_dim):
        super().__init__()
        self.conv = ModConv(in_channels, 3, 1, style_dim, demodulate=False)
    
    def forward(self, x, style):
        return self.conv(x, style)

    
    
class StyleGANG(nn.Module):
    def __init__(self, z_dim=256, w_dim=256, depth=5, init_channels=256):
        super().__init__()
        self.mapper = Mapper(z_dim, w_dim)
        self.init_canvas = nn.Parameter(torch.randn(1, init_channels, 4, 4))
        self.blocks = nn.ModuleList()
        self.rgbs = nn.ModuleList()

        for _ in range(depth):
            out_channels = init_channels // (2**i)
            self.blocks.append(
                nn.Sequential(
                    StyleBlock(init_channels, out_channels, w_dim),
                    StyleBlock(out_channels, out_channels, w_dim),
                )
            )
            self.rgbs.append(ToRGB(out_channels, w_dim))
            init_channels = out_channels
        
            
    def forward(self, z):
        w = self.mapper(z)
        B = z.size(0)
        x = self.init_canvas.expand(B, *([-1]*3))
        rgb = None

        for block, to_rgb in zip(self.blocks, self.rgbs):
            x = block(x, w)
            rgb = F.interpolate(rgb, scale_factor=2, mode="bilinear", align_corners=True) \
                if rgb is not None else 0
            rgb += to_rgb(x, w)


        return torch.tanh(rgb)