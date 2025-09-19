"""
Training script for comparing different generative models (wGAN, VAE, etc.) with diffusion models.
This script follows the same hydra configuration structure as train4.py for consistency.
"""
import os
import datetime
from pathlib import Path

import numpy as np
import hydra
from omegaconf import OmegaConf

from photo_gen.evaluation.evaluation import evaluate_model
from photo_gen.utils.utils import make_wandb_run

from icecream import ic, install

ic.configureOutput(includeContext=True)
install()
OmegaConf.register_new_resolver("eval", eval)


@hydra.main(version_base="1.1", config_path="config", config_name="comparison_config")
def main(cfg):
    OmegaConf.set_struct(cfg, False)

    savedir = 'nanophoto/comparison/experiments/'
    savedir = Path(os.environ.get("SCRATCH")) / savedir

    if cfg.debug:
        savedir = savedir / 'debug'
        cfg.n_to_generate = 16
    else:
        jobid = os.environ.get("SLURM_JOB_ID", "local_run")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        savedir = savedir / f"{jobid}_{timestamp}"

    model_name = cfg.model.name
    savedir = savedir / model_name

    if cfg.inference_only:
        checkpoint_path = os.path.expanduser(cfg.checkpoint_load_path)
    else:
        checkpoint_path = savedir / "checkpoint.pt"

    train_fn = hydra.utils.instantiate(cfg.train)
    inference_fn = hydra.utils.instantiate(cfg.model.inference)

    os.makedirs(savedir, exist_ok=True)
    datapath = os.path.expanduser(cfg.data_path)
    images_savepath = savedir / "images"
    os.makedirs(images_savepath, exist_ok=True)

    data = np.load(datapath)
    print(f"Loaded data shape: {data.shape}")
    print(f"Data dtype: {data.dtype}, min: {data.min():.3f}, max: {data.max():.3f}")

    modcfg = cfg.model

    if not cfg.inference_only:
        checkpoint_exists = cfg.checkpoint_load_path and os.path.exists(os.path.expanduser(cfg.checkpoint_load_path))
    else:
        checkpoint_path = os.path.expanduser(cfg.checkpoint_load_path)

    if cfg.debug:
        modcfg.n_to_generate = 1
        modcfg.train_set_size = 32
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
    
    n_epochs = int(modcfg.n_compute_steps / n_samples)
    assert n_epochs > 0, ic(n_epochs, modcfg.n_compute_steps, n_samples)
    modcfg.n_epochs = n_epochs

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
            run = make_wandb_run(config=dict(cfg), data_path=savedir,
                                 group_name=group_name, run_name=run_name)

        train_fn(data=data, checkpoint_path=checkpoint_path,
                 savedir=savedir, run=run, cfg=cfg)

    images = inference_fn(checkpoint_path=checkpoint_path, 
                              savepath=images_savepath, cfg=cfg)

    results = evaluate_model(images, savedir, cfg)

    if cfg.logger:
        run.log(results)

    # if debuggin, ask the user if they want to delete the savedir
    if cfg.debug:
        delete = input(f"Debug mode: Delete savedir {savedir}? (y/n): ")
        if delete.lower() == 'y':
            import shutil
            shutil.rmtree(savedir)
            print(f"Deleted {savedir}")
        else:
            print(f"Kept {savedir}")

    return

if __name__ == '__main__':
    main()
