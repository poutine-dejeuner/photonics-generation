"""
Simplified UNet implementation that handles arbitrary image sizes.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import random
from models.utils import set_seed
from models.evaluation import get_evaluation_function


class SimpleUNet(nn.Module):
    """Simplified UNet that handles arbitrary image sizes."""
    
    def __init__(self, input_channels=1, output_channels=1, base_channels=64):
        super(SimpleUNet, self).__init__()
        
        # Encoder
        self.enc1 = self.conv_block(input_channels, base_channels)
        self.enc2 = self.conv_block(base_channels, base_channels * 2)
        self.enc3 = self.conv_block(base_channels * 2, base_channels * 4)
        
        # Bottleneck
        self.bottleneck = self.conv_block(base_channels * 4, base_channels * 8)
        
        # Decoder
        self.dec3 = self.conv_block(base_channels * 8 + base_channels * 4, base_channels * 4)
        self.dec2 = self.conv_block(base_channels * 4 + base_channels * 2, base_channels * 2)
        self.dec1 = self.conv_block(base_channels * 2 + base_channels, base_channels)
        
        # Output
        self.final = nn.Conv2d(base_channels, output_channels, kernel_size=1)
        
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Store original size
        orig_size = x.shape[2:]
        
        # Encoder path
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        
        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool2d(e3, 2))
        
        # Decoder path with adaptive upsampling
        d3 = F.interpolate(bottleneck, size=e3.shape[2:], mode='bilinear', align_corners=False)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = F.interpolate(d3, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = F.interpolate(d2, size=e1.shape[2:], mode='bilinear', align_corners=False)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        # Final output - ensure exact original size
        output = self.final(d1)
        output = F.interpolate(output, size=orig_size, mode='bilinear', align_corners=False)
        
        return output


def train(data: np.ndarray, cfg, checkpoint_path: str, savedir: str, run=None):
    """Training function for Simple UNet."""
    
    n_epochs = cfg.n_epochs
    batch_size = cfg.batch_size
    lr = cfg.lr
    seed = cfg.seed
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Simple UNet on {device}")
    print(f"{n_epochs} epochs total")
    
    set_seed(seed)
    
    # Prepare data
    data = torch.tensor(data, dtype=torch.float32)
    if len(data.shape) == 3:  # Add channel dimension if missing
        data = data.unsqueeze(1)
    if data.max() > 1.0:
        data = (data - data.min()) / (data.max() - data.min())
    
    img_channels = data.shape[1]
    
    model = SimpleUNet(input_channels=img_channels, output_channels=img_channels).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    dataset = TensorDataset(data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                           drop_last=True, num_workers=1)
    
    start_epoch = 0
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device,
                weights_only=False)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from epoch {start_epoch}")
    
    # Training loop
    model.train()
    
    for epoch in range(start_epoch, n_epochs):
        total_loss = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{n_epochs}")
        
        for batch_idx, (data_batch,) in enumerate(pbar):
            data_batch = data_batch.to(device)
            
            optimizer.zero_grad()
            
            # Add noise for denoising task
            noise = torch.randn_like(data_batch) * 0.1
            noisy_data = data_batch + noise
            
            # Forward pass
            output = model(noisy_data)
            
            # Loss (denoising)
            loss = criterion(output, data_batch)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({'Loss': f"{loss.item():.4f}"})
            
            # Break early in debug mode
            if cfg.get('debug', False):
                break
        
        avg_loss = total_loss / max(1, len(dataloader))
        
        # Log to wandb if available
        if run is not None:
            run.log({
                'epoch': epoch,
                'loss': avg_loss,
            })
        
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}")
        
        # Save checkpoint (always save in debug mode or at intervals)
        if cfg.get('debug', False) or (epoch + 1) % 10 == 0 or epoch == n_epochs - 1:
            checkpoint = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'config': cfg
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch+1}")
        
        # Break early in debug mode (after first epoch)
        if cfg.get('debug', False):
            print("Debug mode: stopping after 1 epoch")
            break
    
    print("Training completed!")


def inference(checkpoint_path: str, savepath: str, cfg):
    """Generate samples using trained Simple UNet."""
    
    # Get n_to_generate from config
    n_samples = cfg.get('n_to_generate', 1000)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load checkpoint
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device,
            weights_only=False)
    model_cfg = checkpoint['config']
    
    # Initialize model (we'll need to infer the architecture from checkpoint)
    model = SimpleUNet().to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    print(f"Generating {n_samples} samples...")
    
    # Generate samples by denoising random noise
    generated_images = []
    batch_size = 64
    
    # We need to know the image size - let's use a default or get from config
    img_size = model_cfg.get('img_size', (64, 64))
    img_channels = model_cfg.get('img_channels', 1)
    
    with torch.no_grad():
        for i in tqdm(range(0, n_samples, batch_size)):
            current_batch_size = min(batch_size, n_samples - i)
            
            # Start with pure noise
            noise = torch.randn(current_batch_size, img_channels, *img_size).to(device)
            
            # Multiple denoising steps
            for step in range(10):  # Simple iterative denoising
                noise = model(noise)
                noise = torch.clamp(noise, 0, 1)  # Keep in valid range
            
            # Ensure proper normalization
            noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)
            
            generated_images.append(noise.cpu().numpy())
    
    generated_images = np.concatenate(generated_images, axis=0)
    
    # Save images
    np.save(os.path.join(savepath, "images.npy"), generated_images)
    
    # Compute FOM using configurable evaluation function
    eval_fn = get_evaluation_function(cfg)
    fom = eval_fn(generated_images)
    
    # Save FOM
    np.save(os.path.join(savepath, "fom.npy"), fom)
    
    return generated_images, fom
