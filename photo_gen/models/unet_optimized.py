import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import hydra
from timm.utils.model_ema import ModelEmaV3
from torch.cuda.amp import GradScaler, autocast

from photo_gen.models.utils import DDPM_Scheduler, set_seed
from photo_gen.utils.unet_utils import UNetPad


def train_optimized(data: np.ndarray, cfg, checkpoint_path: os.PathLike, savedir: os.PathLike, run=None):
    """
    Optimized training function with mixed precision, larger batch sizes, and memory optimizations.
    """
    # Training parameters
    n_epochs = cfg.n_epochs
    lr = cfg.lr
    batch_size = getattr(cfg, 'batch_size', 32)  # Default to larger batch size
    num_time_steps = cfg.num_time_steps
    ema_decay = cfg.ema_decay
    use_mixed_precision = getattr(cfg, 'use_mixed_precision', True)
    gradient_accumulation_steps = getattr(cfg, 'gradient_accumulation_steps', 1)
    compile_model = getattr(cfg, 'compile_model', True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Memory optimizations
    torch.backends.cudnn.benchmark = True  # Optimize cuDNN for consistent input sizes
    
    print("OPTIMIZED TRAINING")
    print(f"{n_epochs} epochs total")
    print(f"Batch size: {batch_size}")
    print(f"Mixed precision: {use_mixed_precision}")
    print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    
    # Data preprocessing optimizations
    if data.dtype != np.float32:
        print("Converting data to float32 for better performance")
        data = data.astype(np.float32)
    
    data = torch.tensor(data, dtype=torch.float32)
    if data.ndim == 3:
        data = data.unsqueeze(1)
    
    # Model setup
    scheduler = DDPM_Scheduler(num_time_steps=num_time_steps)
    model = hydra.utils.instantiate(cfg.model)
    model = model.to(device)
    
    # Model compilation for additional speedup (PyTorch 2.0+)
    if compile_model and hasattr(torch, 'compile'):
        print("Compiling model with torch.compile")
        model = torch.compile(model)
    
    depth = model.num_layers // 2
    pad_fn = UNetPad(data, depth=depth)
    
    # Optimized data loader
    train_dataset = TensorDataset(data)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        drop_last=True,
        num_workers=8,  # Increased from 4
        pin_memory=True,  # Speed up CPU->GPU transfer
        persistent_workers=True  # Keep workers alive between epochs
    )
    
    # Optimizer with better defaults
    optimizer = optim.AdamW(  # AdamW often performs better than Adam
        model.parameters(), 
        lr=lr,
        weight_decay=0.01,  # Add weight decay for regularization
        betas=(0.9, 0.999)
    )
    
    # Learning rate scheduler
    scheduler_lr = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr * 2,
        total_steps=n_epochs * len(train_loader),
        pct_start=0.1
    )
    
    # Mixed precision scaler
    scaler = GradScaler() if use_mixed_precision else None
    
    ema = ModelEmaV3(model, decay=ema_decay)
    
    # Load checkpoint if exists
    start_epoch = 0
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        model.load_state_dict(checkpoint['weights'])
        ema.load_state_dict(checkpoint['ema'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch']
        if scaler and 'scaler' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler'])
        print(f"Resumed from epoch {start_epoch}")
    
    criterion = nn.MSELoss(reduction='mean')
    
    # Training loop
    for epoch in range(start_epoch, n_epochs):
        model.train()
        total_loss = 0
        accumulated_loss = 0
        
        # Reset gradients at the start of accumulation
        optimizer.zero_grad()
        
        for step, [x] in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}")):
            x = x.to(device, non_blocking=True)
            x = pad_fn(x)
            
            # Generate random timesteps and noise
            t = torch.randint(0, num_time_steps, (x.size(0),), device=device)
            e = torch.randn_like(x, device=device)
            a = scheduler.alpha[t].view(-1, 1, 1, 1).to(device)
            
            # Forward diffusion
            x_noisy = (torch.sqrt(a) * x) + (torch.sqrt(1 - a) * e)
            
            # Forward pass with mixed precision
            if use_mixed_precision:
                with autocast():
                    output = model(x_noisy, t)
                    loss = criterion(output, e) / gradient_accumulation_steps
                
                # Backward pass with scaling
                scaler.scale(loss).backward()
                accumulated_loss += loss.item()
                
                # Update weights every gradient_accumulation_steps
                if (step + 1) % gradient_accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    ema.update(model)
                    scheduler_lr.step()
            else:
                output = model(x_noisy, t)
                loss = criterion(output, e) / gradient_accumulation_steps
                loss.backward()
                accumulated_loss += loss.item()
                
                if (step + 1) % gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    ema.update(model)
                    scheduler_lr.step()
            
            total_loss += accumulated_loss
            
            if (step + 1) % gradient_accumulation_steps == 0:
                accumulated_loss = 0
            
            if cfg.debug:
                break  # Only run one batch in debug mode
        
        avg_loss = total_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch+1} | Loss {avg_loss:.5f} | LR {current_lr:.6f}')
        
        if run is not None:
            run.log({
                "loss": avg_loss,
                "learning_rate": current_lr,
                "epoch": epoch + 1
            })
        
        # Save checkpoint more frequently for long training
        if (epoch + 1) % 50 == 0 or epoch == n_epochs - 1:
            checkpoint = {
                'weights': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'ema': ema.state_dict(),
                'epoch': epoch + 1
            }
            if scaler:
                checkpoint['scaler'] = scaler.state_dict()
            torch.save(checkpoint, checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch + 1}")
        
        if cfg.debug:
            break  # Only run one epoch in debug mode
    
    return total_loss


def calculate_model_size(model):
    """Calculate model parameters for the optimized architecture."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params