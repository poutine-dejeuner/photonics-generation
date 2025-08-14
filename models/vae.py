"""
Variational Autoencoder (VAE) implementation for nanophotonics design generation.
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from models.evaluation import get_evaluation_function
import random
from typing import Tuple, Optional
from models.utils import set_seed


class Encoder(nn.Module):
    """VAE Encoder network."""
    
    def __init__(self, img_channels: int = 1, img_size = 64, 
                 latent_dim: int = 128, hidden_dim: int = 64):
        super(Encoder, self).__init__()
        
        # Convert img_size from ListConfig to tuple if needed
        if hasattr(img_size, '_content'):  # ListConfig check
            img_size = tuple(img_size)
        elif isinstance(img_size, (list, tuple)) and len(img_size) == 2:
            img_size = tuple(img_size)
        elif isinstance(img_size, int):
            img_size = (img_size, img_size)
            
        self.img_size = img_size
        self.img_height, self.img_width = img_size
            
        self.latent_dim = latent_dim
        
        # Convolutional layers
        self.conv_layers = nn.Sequential(
            # Input: (img_channels, img_size, img_size)
            nn.Conv2d(img_channels, hidden_dim, 4, 2, 1),  # (hidden_dim, img_size/2, img_size/2)
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden_dim, hidden_dim * 2, 4, 2, 1),  # (hidden_dim*2, img_size/4, img_size/4)
            nn.BatchNorm2d(hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, 4, 2, 1),  # (hidden_dim*4, img_size/8, img_size/8)
            nn.BatchNorm2d(hidden_dim * 4),
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden_dim * 4, hidden_dim * 8, 4, 2, 1),  # (hidden_dim*8, img_size/16, img_size/16)
            nn.BatchNorm2d(hidden_dim * 8),
            nn.LeakyReLU(0.2),
        )
        
        # Use adaptive pooling to handle different input sizes
        self.global_pool = nn.AdaptiveAvgPool2d(4)  # Pool to 4x4
        pooled_size = hidden_dim * 8 * 4 * 4  # 4x4 after adaptive pooling
        
        # Fully connected layers for mean and log variance
        self.fc_mu = nn.Linear(pooled_size, latent_dim)
        self.fc_logvar = nn.Linear(pooled_size, latent_dim)
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.global_pool(x)  # Adaptive pooling
        x = x.view(x.size(0), -1)  # Flatten
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


class Decoder(nn.Module):
    """VAE Decoder network."""
    
    def __init__(self, latent_dim: int = 128, img_channels: int = 1, 
                 img_size = 64, hidden_dim: int = 64):
        super(Decoder, self).__init__()
        
        # Convert img_size from ListConfig to tuple if needed
        if hasattr(img_size, '_content'):  # ListConfig check
            img_size = tuple(img_size)
        elif isinstance(img_size, (list, tuple)) and len(img_size) == 2:
            img_size = tuple(img_size)
        elif isinstance(img_size, int):
            img_size = (img_size, img_size)
            
        self.img_size = img_size
        self.img_height, self.img_width = img_size
        
        self.hidden_dim = hidden_dim
        
        # Use fixed size that matches encoder output
        self.init_size = 4  # Fixed to match adaptive pooling
        fc_output_size = hidden_dim * 8 * self.init_size ** 2
        
        # Fully connected layer
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, fc_output_size),
            nn.ReLU()
        )
        
        # Deconvolutional layers
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim * 8, hidden_dim * 4, 4, 2, 1),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim * 4, hidden_dim * 2, 4, 2, 1),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, 4, 2, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim, img_channels, 4, 2, 1),
            nn.Sigmoid()  # Assuming normalized images [0,1]
        )
        
        # Final resize to ensure correct output size
        self.final_resize = nn.AdaptiveAvgPool2d((self.img_height, self.img_width))
        
    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), self.hidden_dim * 8, self.init_size, self.init_size)
        x = self.deconv_layers(x)
        # Ensure correct output size
        x = self.final_resize(x)
        return x


class VAE(nn.Module):
    """Variational Autoencoder."""
    
    def __init__(self, img_channels: int = 1, img_size = 64, 
                 latent_dim: int = 128, hidden_dim: int = 64):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        
        self.encoder = Encoder(img_channels, img_size, latent_dim, hidden_dim)
        self.decoder = Decoder(latent_dim, img_channels, img_size, hidden_dim)
        
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick to sample from N(mu, var) from N(0,1)."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)
        return recon_x, mu, logvar
    
    def sample(self, num_samples: int, device: str = 'cuda'):
        """Generate samples from the learned distribution."""
        z = torch.randn(num_samples, self.latent_dim).to(device)
        with torch.no_grad():
            samples = self.decoder(z)
        return samples


def vae_loss_function(recon_x, x, mu, logvar, beta: float = 1.0):
    """VAE loss function with KL divergence and reconstruction loss."""
    # Reconstruction loss (Binary Cross Entropy)
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
    
    # KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + beta * kl_loss, recon_loss, kl_loss


def train(data: np.ndarray, cfg, checkpoint_path: str, savedir: str, run=None):
    """Training function for VAE."""
    
    # Training parameters
    n_epochs = cfg.n_epochs
    batch_size = cfg.batch_size
    lr = cfg.lr
    latent_dim = cfg.latent_dim
    beta = cfg.beta  # KL weight for β-VAE
    seed = cfg.seed
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training VAE on {device}")
    print(f"{n_epochs} epochs total")
    
    # Set seed
    set_seed(random.randint(0, 2**32-1)) if seed == -1 else set_seed(seed)
    
    # Prepare data
    data = torch.tensor(data, dtype=torch.float32)
    if len(data.shape) == 3:  # Add channel dimension if missing
        data = data.unsqueeze(1)
    
    # Normalize data to [0, 1] if needed
    if data.max() > 1.0:
        data = (data - data.min()) / (data.max() - data.min())
    
    img_size = (data.shape[-2], data.shape[-1])  # (height, width)
    img_channels = data.shape[1]
    
    # Initialize model
    model = VAE(img_channels=img_channels, img_size=img_size, 
                latent_dim=latent_dim, hidden_dim=cfg.hidden_dim).to(device)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
    
    # Data loader
    dataset = TensorDataset(data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                           drop_last=True, num_workers=4)
    
    # Load checkpoint if exists
    start_epoch = 0
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from epoch {start_epoch}")
    
    # Training loop
    model.train()
    sample_batch = None  # Store a sample batch for visualization
    
    for epoch in range(start_epoch, n_epochs):
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{n_epochs}")
        
        for batch_idx, (data_batch,) in enumerate(pbar):
            data_batch = data_batch.to(device)
            
            # Store first batch for visualization
            if sample_batch is None:
                sample_batch = data_batch[:16].cpu()
            
            optimizer.zero_grad()
            
            # Forward pass
            recon_batch, mu, logvar = model(data_batch)
            
            # Compute loss
            loss, recon_loss, kl_loss = vae_loss_function(
                recon_batch, data_batch, mu, logvar, beta
            )
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Recon': f"{recon_loss.item():.4f}",
                'KL': f"{kl_loss.item():.4f}"
            })
            
            # Break early in debug mode (after first batch)
            if cfg.get('debug', False):
                break
        
        # Calculate average losses
        avg_loss = total_loss / len(dataloader.dataset)
        avg_recon_loss = total_recon_loss / len(dataloader.dataset)
        avg_kl_loss = total_kl_loss / len(dataloader.dataset)
        
        # Step scheduler
        scheduler.step(avg_loss)
        
        # Log to wandb if available
        if run is not None:
            run.log({
                'epoch': epoch,
                'total_loss': avg_loss,
                'reconstruction_loss': avg_recon_loss,
                'kl_loss': avg_kl_loss,
                'learning_rate': optimizer.param_groups[0]['lr']
            })
        
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Recon={avg_recon_loss:.4f}, KL={avg_kl_loss:.4f}")
        
        # Break early in debug mode (after first epoch)
        if cfg.get('debug', False):
            print("Debug mode: stopping after 1 epoch")
            break
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0 or epoch == n_epochs - 1:
            checkpoint = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'config': cfg
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch+1}")
        
        # Generate sample images
        if (epoch + 1) % 20 == 0:
            model.eval()
            with torch.no_grad():
                # Generate from random latent vectors
                samples = model.sample(16, device)
                
                # Also show reconstructions
                test_data = sample_batch[:16].to(device)
                recon_data, _, _ = model(test_data)
                
                # Plot samples and reconstructions
                fig, axes = plt.subplots(4, 8, figsize=(16, 8))
                
                # Original images (top 2 rows)
                for i in range(16):
                    row = i // 8
                    col = i % 8
                    axes[row, col].imshow(test_data[i].cpu().squeeze().numpy(), cmap='gray')
                    axes[row, col].set_title('Original' if row == 0 else '')
                    axes[row, col].axis('off')
                
                # Reconstructed images (2nd 2 rows)
                for i in range(16):
                    row = (i // 8) + 2
                    col = i % 8
                    axes[row, col].imshow(recon_data[i].cpu().squeeze().numpy(), cmap='gray')
                    axes[row, col].set_title('Reconstructed' if row == 2 else '')
                    axes[row, col].axis('off')
                
                plt.tight_layout()
                plt.savefig(os.path.join(savedir, f'reconstructions_epoch_{epoch+1}.png'))
                plt.close()
                
                # Plot generated samples
                fig, axes = plt.subplots(4, 4, figsize=(8, 8))
                for idx, ax in enumerate(axes.flat):
                    img = samples[idx].cpu().squeeze().numpy()
                    ax.imshow(img, cmap='gray')
                    ax.axis('off')
                
                plt.suptitle(f'Generated Samples - Epoch {epoch+1}')
                plt.tight_layout()
                plt.savefig(os.path.join(savedir, f'samples_epoch_{epoch+1}.png'))
                plt.close()
            
            model.train()
    
    print("Training completed!")


def inference(checkpoint_path: str, savepath: str, cfg):
    """Generate samples using trained VAE."""
    
    # Get n_to_generate from config, with default
    n_samples = cfg.get('n_to_generate', 1000)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load checkpoint
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_cfg = checkpoint['config']
    
    # Initialize model
    model = VAE(img_channels=model_cfg.get('img_channels', 1),
                img_size=model_cfg.get('img_size', 64),
                latent_dim=model_cfg.latent_dim,
                hidden_dim=model_cfg.hidden_dim).to(device)
    
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    print(f"Generating {n_samples} samples...")
    
    generated_images = []
    batch_size = 64
    
    with torch.no_grad():
        for i in tqdm(range(0, n_samples, batch_size)):
            current_batch_size = min(batch_size, n_samples - i)
            batch_imgs = model.sample(current_batch_size, device)
            generated_images.append(batch_imgs.cpu().numpy())
    
    generated_images = np.concatenate(generated_images, axis=0)
    
    # Save images
    np.save(os.path.join(savepath, "images.npy"), generated_images)
    
    # Compute FOM using configurable evaluation function
    eval_fn = get_evaluation_function(cfg)
    fom = eval_fn(generated_images)
    
    # Save FOM
    np.save(os.path.join(savepath, "fom.npy"), fom)
    
    return generated_images, fom
