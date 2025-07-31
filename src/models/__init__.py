from GAN import GANG, GAND
from styleGAN import StyleGANG, StyleGAND

def setup_models(config):
    if config.get("name", "GAN") == "GAN":
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


        discriminator = GAND(
            **discriminator_config
        )

    elif config.get("name", "StyleGAN"):
        generator = StyleGANG(
            z_dim=config.generator.get("lat_dim", 256),
            w_sim=config.generator.get("w_dim", 256),
            depth=config.generator.get("depth", 5),
            init_channels=config.generator.get("hidden_dim", 256)
        )

        discriminator = StyleGAND(
            init_channels=config.discriminator.get("hidden_dim", 64),
            depth=config.discriminator.get("depth", 5)
        )

    return generator, discriminator