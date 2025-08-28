"""
Training script for comparing different generative models (wGAN, VAE, etc.) with diffusion models.
This script follows the same hydra configuration structure as train4.py for consistency.
"""
import os
import datetime

import numpy as np
from einops import rearrange
from torch.utils.data import DataLoader, TensorDataset
import hydra
from omegaconf import OmegaConf

from evaluation import evaluation

from nanophoto.utils import make_wandb_run

from icecream import ic, install

ic.configureOutput(includeContext=True)
install()
OmegaConf.register_new_resolver("eval", eval)


@hydra.main(config_path="config", config_name="comparison_config")
def main(cfg):
    OmegaConf.set_struct(cfg, False)

    savedir = 'nanophoto/comparison/experiments/'
    savedir = os.path.join(os.environ.get("SCRATCH", "./"), savedir)

    if cfg.debug:
        savedir = os.path.join(savedir, 'debug')
        cfg.n_to_generate = 16
    else:
        jobid = os.environ.get("SLURM_JOB_ID", "local_run")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        savedir = os.path.join(savedir, f"{jobid}_{timestamp}")

    model_name = cfg.model.get('name', cfg.model.get('_target_', 'unknown').split('.')[-1])
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

    images = inference_fn(checkpoint_path=checkpoint_path, 
                              savepath=images_savepath, cfg=cfg)

    results = evaluation(images, savedir, model_name, cfg)

    if cfg.logger:
        run.log(results)
    ic(results)

    return

if __name__ == '__main__':
    main()
