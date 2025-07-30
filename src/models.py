import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class PixelNorm(nn.Module):
    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdims=True) + 1e-8)



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
            hidden_dim, 
            depth, 
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
        init_channels  = hidden_dim * (2**(depth-1))

        self.projector = nn.Sequential(
            nn.Linear(lat_dim, init_channels * (self.init_size**2)),
            nn.BatchNorm1d(init_channels * (self.init_size**2)),
            nn.ReLU()
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
        for i in range(depth-1):
            out_channels = hidden_dim * (2** (depth - i - 1))
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
        
        # last layer
        self.layers.append(
            nn.ConvTranspose2d(
                in_channels,
                3,
                **{
                    "kernel_size":4,
                    "stride":2,
                    "padding":1,
                },
                bias=False
            )
        )
        self.layers = nn.Sequential(*self.layers)


    def mapping(self, z):
        return self.projector(z).view(z.size(0), -1, self.init_size, self.init_size)

    def synthesis(self, w):
        return torch.tanh(self.layers(w))

    def forward(self, z, return_w=False):
        w = self.mapping(z)
        return self.synthesis(w)


class GAND(nn.Module):
    def __init__(
            self,
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
            out_channels = hidden_dim * (2**i)
            self.layers.append(
                ConvBlock(
                    in_channels, out_channels, **block_kwargs
                )
            )
            if use_SA and i in self.attention_layers:
                self.layers.append(SelfAttention(out_channels))

            in_channels = out_channels
        self.layers = nn.Sequential(*self.layers)


        # self.cls_head = nn.Conv2d(
        #     in_channels,
        #     out_channels,
        #     kernel_size=4,
        #     stdie=1,
        #     padding=0,
        # )


        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, 1)
        )
        

    def forward(self, x):
        x = self.layers(x)
        x = self.cls_head(x)
        return x

def setup_models(config):
    def retrieve(param, default):
        return specific.get(param, config.get(param, default))
    

    specific=config.generator
    generator_config = {
        "lat_dim": retrieve("lat_dim", 128),
        "hidden_dim": retrieve("hidden_dim", 32),
        "depth": retrieve("depth", 6),
        "norm": retrieve("norm", "batch"),
        "activation": retrieve("activation", "elu"),
        "leak": retrieve("leak", 0.1),
        "use_SA": retrieve("use_SA", False),
        "use_SN": retrieve("use_SN", False),
        "attention_layers": None
    }

    generator = GANG(
        **generator_config
    )


    specific=config.discriminator
    discriminator_config = {
        "hidden_dim": retrieve("hidden_dim", 32),
        "depth": retrieve("depth", 6),
        "norm": retrieve("norm", None),
        "activation": retrieve("activation", "elu"),
        "leak": retrieve("leak", 0.1),
        "use_SA": retrieve("use_SA", False),
        "use_SN": retrieve("use_SN", False),
        "attention_layers": None
    }


    descriminator = GAND(
        **discriminator_config
    )

    return generator, descriminator