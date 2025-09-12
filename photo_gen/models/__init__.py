"""
Photo-Gen Models Package

This package contains all the model architectures including:
- UNet diffusion models
- VAE (Variational Autoencoders)
- WGAN (Wasserstein GANs)
- Standard GANs
- Set Transformers
"""

from .unet import UNET, ResBlock, Attention, UnetLayer, SinusoidalEmbeddings, train as unet_train, inference as unet_inference
from .vae import VAE
from .wgan import WGAN
from .standard_gan import StandardGAN
from .simple_unet import SimpleUNet
from .utils import DDPM_Scheduler, set_seed

__all__ = [
    "UNET",
    "ResBlock", 
    "Attention",
    "UnetLayer",
    "SinusoidalEmbeddings",
    "unet_train",
    "unet_inference",
    "VAE",
    "WGAN", 
    "StandardGAN",
    "SimpleUNet",
    "DDPM_Scheduler",
    "set_seed",
]
