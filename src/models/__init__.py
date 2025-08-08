from .GAN import GANG, GAND
from .styleGAN import StyleGANG, StyleGAND

def setup_models(config):
    def retrieve(param, default):
        return specific.get(param, config.get(param, default))
    if config.get("name", "GAN").upper() == "GAN":
        
        specific=config.generator
        generator_config = {
            "lat_dim": retrieve("lat_dim", 128),
            "channels": retrieve("channels", [512, 256, 256, 128, 128, 64]),
            "init_size": retrieve("init_size", 4),
            "norm": retrieve("norm", "batch"),
            "activation": retrieve("activation", "leaky_relu"),
            "leak": retrieve("leak", 0.1),
            "use_SA": retrieve("use_SA", False),
            "use_SN": retrieve("use_SN", False),
            "attention_layers": retrieve("attention_layers", None),
            "upsample": retrieve("upsample", "deconv")
        }

        generator = GANG(
            **generator_config
        )


        specific=config.discriminator
        discriminator_config = {
            "channels": retrieve("channels", [64, 128, 128, 256]),
            "norm": retrieve("norm", "none"),
            "activation": retrieve("activation", "elu"),
            "leak": retrieve("leak", 0.1),
            "use_SA": retrieve("use_SA", False),
            "use_SN": retrieve("use_SN", False),
            "attention_layers": retrieve("attention_layers", None)
        }


        discriminator = GAND(
            **discriminator_config
        )

    elif config.get("name", "other").upper() == "STYLEGAN":
        generator = StyleGANG(
            lat_dim=config.get("lat_dim", 128),
            init_channels=config.generator.get("init_channels", 128),
            channels=config.generator.get("channels", [256, 256, 128, 128, 64]),
            w_dim=config.generator.get("w_dim", 128),
        )

        discriminator = StyleGAND(
            channels = config.discriminator.get("channels", [64, 128, 128, 256]),
        )

    else:
        raise Exception(f"invalid or missing model name: {config.get('name')}")

    return generator, discriminator