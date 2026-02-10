from src.gan.architectures.dcgan           import Generator as DC_G,  Discriminator as DC_D
from src.gan.architectures.dcgan_v2        import Generator as DC_G2, Discriminator as DC_D2
from src.gan.architectures.resnet          import Generator as RN_G,  Discriminator as RN_D
from src.gan.architectures.chest_xray      import Generator as CXR_G, Discriminator as CXR_D
from src.gan.architectures.stl10_sagan     import Generator as SA_G,  Discriminator as SA_D
from src.gan.architectures.cifar10_sagan   import Generator as CF_G,  Discriminator as CF_D   # ← NEW
from src.gan.architectures.imagenet        import Generator as Imagenet_G, Discriminator as Imagenet_D

from src.gan.loss import (
    NS_GeneratorLoss,
    NS_DiscriminatorLoss,
    W_GeneratorLoss,
    WGP_DiscriminatorLoss,
    HingeR1_DiscriminatorLoss,
)

# ─────────────────────────────────── GAN factory ───────────────────────────────────
def construct_gan(config, img_size, device):
    loss_name = config["loss"]["name"].lower()

    # critic-like outputs (no sigmoid) for WGAN-GP and hinge losses
    is_critic = loss_name in {"wgan-gp", "hinge-r1", "ns"}

    
    use_batch_norm = loss_name != "wgan-gp"

    arch = config["architecture"]


    if arch["name"] == "dcgan":
        G = DC_G(img_size, z_dim=config["z_dim"],
                 filter_dim=arch["g_filter_dim"],
                 n_blocks=arch["g_num_blocks"]).to(device)
        D = DC_D(img_size,
                 filter_dim=arch["d_filter_dim"],
                 n_blocks=arch["d_num_blocks"],
                 use_batch_norm=use_batch_norm,
                 is_critic=is_critic).to(device)

    elif arch["name"] == "dcgan-v2":
        G = DC_G2(img_size, z_dim=config["z_dim"],
                  filter_dim=arch["g_filter_dim"],
                  n_blocks=arch["g_num_blocks"]).to(device)
        D = DC_D2(img_size,
                  filter_dim=arch["d_filter_dim"],
                  n_blocks=arch["d_num_blocks"],
                  use_batch_norm=use_batch_norm,
                  is_critic=is_critic).to(device)

    elif arch["name"] == "resnet":
        G = RN_G(img_size, z_dim=config["z_dim"],
                 gf_dim=arch["g_filter_dim"]).to(device)
        D = RN_D(img_size,
                 df_dim=arch["d_filter_dim"],
                 use_batch_norm=use_batch_norm,
                 is_critic=is_critic).to(device)

    elif arch["name"] == "chest-xray":
        # chest_xray architecture expects `fmap` (not `filter_dim`)
        G = CXR_G(
            img_size=img_size,
            z_dim=config["z_dim"],
            fmap=arch["g_filter_dim"],
        ).to(device)
        D = CXR_D(
            img_size=img_size,
            fmap=arch["d_filter_dim"],
            is_critic=is_critic,
        ).to(device)

    elif arch["name"] == "stl10_sagan":
        # stl10_sagan expects `fmap` and supports configurable `n_blocks`
        G = SA_G(
            img_size=img_size,
            z_dim=config["z_dim"],
            fmap=arch["g_filter_dim"],
            n_blocks=arch["g_num_blocks"],
        ).to(device)
        D = SA_D(
            img_size=img_size,
            fmap=arch["d_filter_dim"],
            n_blocks=arch["d_num_blocks"],
            is_critic=is_critic,
        ).to(device)

    elif arch["name"] == "cifar10_sagan":
        # cifar10_sagan currently has fixed depth (3 generator/discriminator blocks)
        if arch["g_num_blocks"] != 3 or arch["d_num_blocks"] != 3:
            raise ValueError(
                "cifar10_sagan uses a fixed-depth architecture. "
                "Set g_num_blocks=3 and d_num_blocks=3."
            )
        G = CF_G(
            img_size=img_size,
            z_dim=config["z_dim"],
            fmap=arch["g_filter_dim"],
        ).to(device)
        D = CF_D(
            img_size=img_size,
            fmap=arch["d_filter_dim"],
            is_critic=is_critic,
        ).to(device)

    elif arch["name"] == "imagenet":
        raise NotImplementedError(
            "imagenet architecture is class-conditional (requires labels in G/D forward), "
            "but the current training loop is unconditional. "
            "Use a non-imagenet architecture or implement conditional GAN training first."
        )

    else:
        raise ValueError(f"Unsupported architecture: {arch['name']}")

    return G, D

# ────────────────────────────────── loss factory ──────────────────────────────────
def construct_loss(config, D):
    name = config["name"].lower()

    if name == "ns":
        return NS_GeneratorLoss(D), NS_DiscriminatorLoss(D)

    elif name == "wgan-gp":
        λ = config["args"]["lambda"]
        return W_GeneratorLoss(), WGP_DiscriminatorLoss(D, λ)

    elif name == "hinge-r1":
        λ = config["args"]["lambda"]
        return W_GeneratorLoss(), HingeR1_DiscriminatorLoss(D, λ)

    else:
        raise ValueError(f"Unsupported loss: {config['name']}")
