"""
Training script for comparing different generative models (wGAN, VAE, etc.) with diffusion models.
This script follows the same hydra configuration structure as train4.py for consistency.
"""
import os
import random
import json
from tqdm import tqdm
from typing import List, Dict, Any
import datetime

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import torch.optim as optim
import hydra
from omegaconf import OmegaConf

from models.utils import set_seed
from utils import make_wandb_run
from nanophoto.meep_compute_fom import compute_FOM_parallele
from nanophoto.evaluation.evalgen import eval_metrics

from icecream import ic, install

ic.configureOutput(includeContext=True)
install()
OmegaConf.register_new_resolver("eval", eval)


@hydra.main(config_path="config", config_name="comparison_config")
def main(cfg):
    OmegaConf.set_struct(cfg, False)
    
    # Setup save directory
    savedir = 'nanophoto/comparison/experiments/'
    savedir = os.path.join(os.environ.get("SCRATCH", "./"), savedir)
    
    if cfg.debug:
        savedir = os.path.join(savedir, 'debug')
    else:
        jobid = os.environ.get("SLURM_JOB_ID", "local_run")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        savedir = os.path.join(savedir, f"{jobid}_{timestamp}")
    
    # Model-specific directory
    model_name = cfg.model.get('_target_', 'unknown').split('.')[-1]
    savedir = os.path.join(savedir, model_name)
    
    if cfg.inference_only:
        checkpoint_path = os.path.expanduser(cfg.checkpoint_load_path)
    else:
        checkpoint_path = os.path.join(savedir, "checkpoint.pt")

    # Instantiate training and inference functions
    train_fn = hydra.utils.instantiate(cfg.train)
    inference_fn = hydra.utils.instantiate(cfg.inference)

    # Setup directories
    os.makedirs(savedir, exist_ok=True)
    datapath = os.path.expanduser(cfg.data_path)
    images_savepath = os.path.join(savedir, "images")
    os.makedirs(images_savepath, exist_ok=True)
    
    # Load data
    data = np.load(datapath)
    print(f"Loaded data shape: {data.shape}")
    print(f"Data dtype: {data.dtype}, min: {data.min():.3f}, max: {data.max():.3f}")
    
    # Model configuration
    modcfg = cfg.model
    
    # Debug mode adjustments (do this early to affect data loading)
    if cfg.debug:
        modcfg.n_images = 1
        modcfg.train_set_size = 16
        modcfg.n_epochs = 1
    
    n_samples = data.shape[0] if modcfg.train_set_size == -1 else modcfg.train_set_size 
    data = data[:n_samples]
    
    # Ensure data has correct dimensions and is properly normalized
    if len(data.shape) == 3:  # Add channel dimension if missing
        data = data[:, None, :, :]
    
    # Get the actual image size from data
    actual_img_height, actual_img_width = data.shape[-2], data.shape[-1]
    print(f"Actual image size: {actual_img_height}x{actual_img_width}")
    
    # Update model config with actual image size
    modcfg.img_size = (actual_img_height, actual_img_width)
    modcfg.img_channels = data.shape[1]
    
    # Pass debug flag to model config
    modcfg.debug = cfg.debug
    
    # Normalize data to [0, 1] if needed
    if data.max() > 1.0:
        data = (data - data.min()) / (data.max() - data.min())
        print(f"Normalized data to range [0, 1]")
    
    # Calculate epochs if needed
    if hasattr(modcfg, 'n_compute_steps'):
        modcfg.n_epochs = int(modcfg.n_compute_steps / n_samples)

    # Training phase
    if cfg.inference_only is False:
        run = None
        if cfg.logger:
            group_name = f"comparison_{model_name}"
            run_name = f"{model_name}_{os.environ.get('SLURM_JOB_ID', 'local')}"
            run = make_wandb_run(config=dict(cfg), data_path=savedir,
                                 group_name=group_name, run_name=run_name)
        
        # Train the model
        train_fn(data=data, checkpoint_path=checkpoint_path,
                 savedir=savedir, run=run, cfg=modcfg)

    # Inference phase
    images, fom = inference_fn(checkpoint_path=checkpoint_path, 
                              savepath=images_savepath, cfg=cfg)
    
    # Plot FOM histogram
    plt.figure(figsize=(10, 6))
    plt.hist(fom, bins=100, alpha=0.7, edgecolor='black')
    plt.title(f"FOM Histogram - {model_name}")
    plt.xlabel("Figure of Merit")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(savedir, "fom_histogram.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Evaluate metrics
    dataset_cfg = OmegaConf.create([{"name": f"{model_name}_{os.environ.get('SLURM_JOB_ID', 'local')}",
                                   "path": images_savepath}])
    eval_metrics(dataset_cfg, os.path.dirname(datapath))
    
    # Save experiment results
    results = {
        'model_type': cfg.model.get('_target_', 'unknown'),
        'train_set_size': cfg.train_set_size,
        'debug': cfg.debug,
        'experiment_path': savedir,
        'config': OmegaConf.to_container(cfg, resolve=True)  # Convert to regular dict
    }
    
    with open(os.path.join(savedir, 'experiment_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    return fom.mean()


if __name__ == '__main__':
    main()
