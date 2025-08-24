"""
Training script for comparing different generative models (wGAN, VAE, etc.) with diffusion models.
This script follows the same hydra configuration structure as train4.py for consistency.
"""
import os
import json
import datetime

import numpy as np
from einops import rearrange
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import hydra
from omegaconf import OmegaConf

from nanophoto.utils import make_wandb_run
from nanophoto.evaluation.evalgen import eval_metrics

from icecream import ic, install

ic.configureOutput(includeContext=True)
install()
OmegaConf.register_new_resolver("eval", eval)


def visualize_generated_samples(images: np.ndarray, savepath: str, model_name: str, n_samples: int = 16):
    """
    Create a grid visualization of generated samples and save it.
    
    Args:
        images: Generated images array of shape (N, C, H, W) or (N, H, W)
        savepath: Directory to save the visualization
        model_name: Name of the model for the title
        n_samples: Number of samples to display in grid (default 16)
    """
    # Ensure we don't exceed available samples
    n_samples = min(n_samples, images.shape[0])
    
    # Calculate grid dimensions (prefer square grids)
    grid_size = int(np.ceil(np.sqrt(n_samples)))
    
    # Create figure
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    fig.suptitle(f'{model_name} - Generated Samples', fontsize=16, fontweight='bold')
    
    # Handle case where we have only one subplot
    if grid_size == 1:
        axes = [[axes]]
    elif grid_size == 2:
        axes = axes.reshape(2, 2)
    
    sample_idx = 0
    for i in range(grid_size):
        for j in range(grid_size):
            ax = axes[i][j] if grid_size > 1 else axes[i]
            
            if sample_idx < n_samples:
                # Get the image
                img = images[sample_idx]
                
                # Handle different image formats
                if len(img.shape) == 3:  # (C, H, W)
                    if img.shape[0] == 1:  # Single channel
                        img = img.squeeze(0)
                    else:  # Multi-channel, transpose to (H, W, C)
                        img = img.transpose(1, 2, 0)
                elif len(img.shape) == 2:  # (H, W)
                    pass  # Already in correct format
                
                # Normalize to [0, 1] for display
                img_display = (img - img.min()) / (img.max() - img.min() + 1e-8)
                
                # Display image
                if len(img_display.shape) == 2:  # Grayscale
                    ax.imshow(img_display, cmap='gray', vmin=0, vmax=1)
                else:  # Color
                    ax.imshow(np.clip(img_display, 0, 1))
                
                ax.set_title(f'Sample {sample_idx + 1}', fontsize=10)
                sample_idx += 1
            else:
                # Hide empty subplots
                ax.set_visible(False)
            
            ax.axis('off')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)  # Make room for suptitle
    
    # Save the visualization
    save_file = os.path.join(savepath, f"{model_name.lower()}_samples_grid.png")
    plt.savefig(save_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Sample grid visualization saved: {save_file}")


@hydra.main(config_path="config", config_name="comparison_config")
def main(cfg):
    OmegaConf.set_struct(cfg, False)
    
    savedir = 'nanophoto/comparison/experiments/'
    savedir = os.path.join(os.environ.get("SCRATCH", "./"), savedir)
    
    if cfg.debug:
        savedir = os.path.join(savedir, 'debug')
    else:
        jobid = os.environ.get("SLURM_JOB_ID", "local_run")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        savedir = os.path.join(savedir, f"{jobid}_{timestamp}")
    
    model_name = cfg.model.get('_target_', 'unknown').split('.')[-1]
    savedir = os.path.join(savedir, model_name)
    
    if cfg.inference_only:
        checkpoint_path = os.path.expanduser(cfg.checkpoint_load_path)
    else:
        checkpoint_path = os.path.join(savedir, "checkpoint.pt")

    train_fn = hydra.utils.instantiate(cfg.train)
    inference_fn = hydra.utils.instantiate(cfg.model.inference)

    os.makedirs(savedir, exist_ok=True)
    datapath = os.path.expanduser(cfg.data_path)
    images_savepath = os.path.join(savedir, "images")
    os.makedirs(images_savepath, exist_ok=True)
    
    data = np.load(datapath)
    print(f"Loaded data shape: {data.shape}")
    print(f"Data dtype: {data.dtype}, min: {data.min():.3f}, max: {data.max():.3f}")
    
    modcfg = cfg.model
    
    # Apply parameter count constraint if specified AND not loading from checkpoint
    target_params = cfg.get('target_params', None)
    auto_adjust = cfg.get('auto_adjust_architecture', True)
    
    if target_params is not None and auto_adjust and not cfg.inference_only:
        # Check if we're loading from an existing checkpoint
        checkpoint_exists = cfg.checkpoint_load_path and os.path.exists(os.path.expanduser(cfg.checkpoint_load_path))
        
        if not checkpoint_exists:
            print(f"Adjusting model configuration for target parameter count: {target_params:,}")
            print(f"Adjusted hidden_dim: {modcfg.get('hidden_dim', 'N/A')}")
            print(f"Adjusted latent_dim: {modcfg.get('latent_dim', 'N/A')}")
        else:
            print(f"Existing checkpoint found at {cfg.checkpoint_load_path}")
            print("Using current configuration from config file...")
            # Note: Using config file parameters instead of extracting from checkpoint
            # Ensure model parameters match what was used during training
    elif cfg.inference_only:
        # For inference only, use the configuration from config files
        checkpoint_path = os.path.expanduser(cfg.checkpoint_load_path)
        if os.path.exists(checkpoint_path):
            print("Inference mode: using model configuration from config file...")
            print("Note: Ensure config parameters match the checkpoint being loaded")
        else:
            print(f"Warning: Checkpoint path {checkpoint_path} does not exist")
    
    if cfg.debug:
        modcfg.n_images = 1
        modcfg.train_set_size = 16
        modcfg.n_epochs = 1
    
    n_samples = data.shape[0] if modcfg.train_set_size == -1 else modcfg.train_set_size 
    data = data[:n_samples]
    
    if len(data.shape) == 3:
        data = data[:, None, :, :]
    
    actual_img_height, actual_img_width = data.shape[-2], data.shape[-1]
    print(f"Actual image size: {actual_img_height}x{actual_img_width}")
    
    modcfg.img_size = (actual_img_height, actual_img_width)
    modcfg.img_channels = data.shape[1]
    
    modcfg.debug = cfg.debug
    
    if data.max() > 1.0:
        data = (data - data.min()) / (data.max() - data.min())
        print(f"Normalized data to range [0, 1]")
    
    # Ensure n_epochs is properly computed
    if hasattr(modcfg, 'n_compute_steps') and modcfg.n_compute_steps:
        modcfg.n_epochs = int(modcfg.n_compute_steps / n_samples)
        print(f"Computed n_epochs: {modcfg.n_epochs} (from {modcfg.n_compute_steps} compute steps / {n_samples} samples)")
    elif not hasattr(modcfg, 'n_epochs') or not modcfg.n_epochs:
        # Fallback to a reasonable default
        modcfg.n_epochs = 100
        print(f"Using fallback n_epochs: {modcfg.n_epochs}")

    # Log model configuration if training (parameter estimation removed due to missing dependencies)
    if cfg.inference_only is False:
        target_params = cfg.get('target_params', None)
        if target_params:
            print(f"Target parameter count: {target_params:,}")
        print(f"Model configuration:")
        print(f"  - Model type: {modcfg.get('_target_', 'unknown')}")
        print(f"  - Latent dim: {modcfg.get('latent_dim', 'N/A')}")
        print(f"  - Hidden dim: {modcfg.get('hidden_dim', 'N/A')}")
        print(f"  - Image channels: {modcfg.get('img_channels', 'N/A')}")
        print(f"  - Image size: {modcfg.get('img_size', 'N/A')}")

    if cfg.inference_only is False:
        # Copy training parameters from model config to top level for UNet compatibility
        training_params = ['n_epochs', 'lr', 'batch_size', 'num_time_steps', 'ema_decay', 'seed', 'img_size', 'img_channels']
        for param in training_params:
            if hasattr(modcfg, param):
                setattr(cfg, param, getattr(modcfg, param))
                print(f"Copied {param}: {getattr(modcfg, param)} to top-level config")
        
        run = None
        if cfg.logger:
            group_name = f"comparison_{model_name}"
            run_name = f"{model_name}_{os.environ.get('SLURM_JOB_ID', 'local')}"
            run = make_wandb_run(config=dict(cfg), savepath=savedir,
                                 group=group_name, run_name=run_name)
        
        train_fn(data=data, checkpoint_path=checkpoint_path,
                 savedir=savedir, run=run, cfg=cfg)

    images, fom = inference_fn(checkpoint_path=checkpoint_path, 
                              savepath=images_savepath, cfg=cfg)
    
    visualize_generated_samples(images, images_savepath, model_name, n_samples=16)
    
    plt.figure(figsize=(10, 6))
    plt.hist(fom, bins=100, alpha=0.7, edgecolor='black')
    plt.title(f"FOM Histogram - {model_name}")
    plt.xlabel("Figure of Merit")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(savedir, "fom_histogram.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    dataset_cfg = OmegaConf.create([{"name": f"{model_name}_{os.environ.get('SLURM_JOB_ID', 'local')}",
                                   "path": images_savepath}])
    eval_metrics(dataset_cfg, os.path.dirname(datapath))
    
    results = {
        'model_type': cfg.model.get('_target_', 'unknown'),
        'train_set_size': cfg.train_set_size,
        'debug': cfg.debug,
        'experiment_path': savedir,
        'config': OmegaConf.to_container(cfg, resolve=True)
    }
    
    with open(os.path.join(savedir, 'experiment_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    return fom.mean()

if __name__ == '__main__':
    main()
