"""
latent_dim: 128
img_channels: 1
img_size: [101,91]  # Will be overridden by actual data size
hidden_dim: 200
"""
import torch
from wgan import Generator, Critic, WGAN

latent_dim = 128
img_channels = 1
img_size = (101, 91)
hidden_dim = 200

generator = Generator(latent_dim=latent_dim, img_channels=img_channels, img_size=img_size, hidden_dim=hidden_dim)
critic = Critic(img_channels=img_channels, img_size=img_size, hidden_dim=hidden_dim)
gan = WGAN(img_channels=img_channels, img_size=img_size, hidden_dim=hidden_dim,
        device='cpu')


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Generator parameters: {count_parameters(generator)}")
print(f"Critic parameters: {count_parameters(critic)}")
print(f"Total parameters: {count_parameters(generator) + count_parameters(critic)}")
print(count_parameters(gan.generator), count_parameters(gan.critic))
