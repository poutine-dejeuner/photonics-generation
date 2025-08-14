import os
import random
from tqdm import tqdm
from typing import List

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import torch.optim as optim
from timm.utils import ModelEmaV3
import hydra
from omegaconf import OmegaConf

from models.utils import DDPM_Scheduler, set_seed

from utils import UNetPad, make_wandb_run, memory_monitor, select_diverse_images_kmeans
from nanophoto.meep_compute_fom import compute_FOM_parallele
from nanophoto.evaluation.evalgen import eval_metrics

# from orion.client import report_objective
from icecream import ic, install

bp = breakpoint

"""
modele de diffusion, apprendre à générer les design
"""

# Memory monitoring configuration
ENABLE_MEMORY_MONITORING = True  # Set to False to disable memory monitoring
SAVE_MEMORY_STATS = True        # Set to False to disable saving memory stats to file

ic.configureOutput(includeContext=True)
install()
OmegaConf.register_new_resolver("eval", eval)
# OmegaConf.register_new_resolver("eval", lambda expr: eval(expr))


@memory_monitor(enabled=ENABLE_MEMORY_MONITORING, save_stats=SAVE_MEMORY_STATS)
def train(data: np.ndarray, cfg, checkpoint_path: os.path, savedir: os.path,
          run=None):
    seed = -1
    n_epochs = cfg.training.n_epochs
    lr = cfg.training.lr
    batch_size = cfg.training.batch_size
    n_time_steps = cfg.model.n_time_steps
    ema_decay = cfg.model.ema_decay
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("TRAINING")
    print(f"{n_epochs} epochs total")
    
    set_seed(random.randint(0, 2**32-1)) if seed == -1 else set_seed(seed)
    dtype = torch.float32

    data = torch.tensor(data, dtype=dtype)
    data = data.unsqueeze(1)

    scheduler = DDPM_Scheduler(num_time_steps=n_time_steps)

    model = hydra.utils.instantiate(cfg.model)
    model = model.to(device)
    depth = model.num_layers//2

    transform = UNetPad(data, depth=depth)

    train_dataset = TensorDataset(data)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True,
        num_workers=1, pin_memory=False)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    ema = ModelEmaV3(model, decay=ema_decay)
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        model.load_state_dict(checkpoint['weights'])
        ema.load_state_dict(checkpoint['ema'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    criterion = nn.MSELoss(reduction='mean')

    for i in range(n_epochs):
        total_loss = 0
        for bidx, x in enumerate(train_loader):
            x = x[0]
            x = x.cuda()
            t = torch.randint(0, n_time_steps, (batch_size,))
            e = torch.randn_like(x, requires_grad=False)
            a = scheduler.alpha[t].view(batch_size, 1, 1, 1).cuda()
            x = (torch.sqrt(a)*x) + (torch.sqrt(1-a)*e)
            x = transform(x)
            output = model(x, t.to(device))
            optimizer.zero_grad()
            output = transform.inverse(output)
            loss = criterion(output, e)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            ema.update(model)
        
        print(f'Epoch {i+1} | Loss {total_loss / len(train_loader):.5f}')
        if run is not None:
            run.log({"loss": total_loss})
        
        # Clear CUDA cache to prevent memory fragmentation
        torch.cuda.empty_cache()
        
        if i % 100 == 0:
            checkpoint = {
                'weights': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'ema': ema.state_dict()
            }
            torch.save(checkpoint, checkpoint_path)
    # report_objective(loss.item(), 'loss')
    return total_loss


@memory_monitor(enabled=ENABLE_MEMORY_MONITORING, save_stats=SAVE_MEMORY_STATS)
def inference_parallele(cfg,
                        checkpoint_path: str = None,
                        ):
    n_time_steps = cfg.model.n_time_steps if cfg.debug is False else 100
    ema_decay = cfg.model.ema_decay
    batch_size = cfg.inference.batch_size
    n_images = cfg.inference.n_images if cfg.debug is False else batch_size
    n_batches = int(n_images/batch_size)
    image_shape = tuple(cfg.image_shape)

    print("INFERENCE")
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    device = torch.device("cuda")

    model = hydra.utils.instantiate(cfg.model)
    model = model.to(device)

    model.load_state_dict(checkpoint['weights'])
    ema = ModelEmaV3(model, decay=ema_decay)
    ema.load_state_dict(checkpoint['ema'])

    # Enable optimizations
    model = ema.module.eval()

    # Try to compile the model for better performance (PyTorch 2.0+)
    try:
        # Use a more conservative compilation mode to avoid backend errors
        model = torch.compile(model, mode="default")
        print("Model compiled successfully with default mode")
    except Exception as e:
        print(f"Model compilation failed: {e}")
        print("Continuing without compilation...")

    scheduler = DDPM_Scheduler(num_time_steps=n_time_steps)

    z = torch.randn((1, 1,)+image_shape)
    padding_fn = UNetPad(z, depth=model.num_layers//2)
    padded_image_shape = padding_fn(z).shape[2:]

    # Move scheduler tensors to GPU and precompute coefficients
    beta_gpu = scheduler.beta.to(device)
    alpha_gpu = scheduler.alpha.to(device)

    # Precompute all coefficients to avoid repeated computation in loop
    sqrt_one_minus_alpha = torch.sqrt(1 - alpha_gpu)
    sqrt_one_minus_beta = torch.sqrt(1 - beta_gpu)
    temp_coeffs = beta_gpu / (sqrt_one_minus_alpha * sqrt_one_minus_beta)
    main_coeffs = 1 / sqrt_one_minus_beta
    sqrt_beta = torch.sqrt(beta_gpu)

    # Precompute all time tensors to avoid repeated tensor creation
    time_tensors = {}
    for t_idx in range(n_time_steps):
        time_tensors[t_idx] = torch.full(
            (batch_size,), t_idx, device=device, dtype=torch.long)

    all_samples = []

    with torch.no_grad():
        # Enable cuDNN benchmarking for better performance
        torch.backends.cudnn.benchmark = True

        # Warm up the model with a few forward passes to optimize CUDA kernels
        print("Warming up model...")
        z_warmup = torch.randn((batch_size, 1,)+padded_image_shape, device=device)
        if cfg.debug is not True:
            for _ in range(3):
                # Use a mid-range timestep for warmup
                _ = model(z_warmup, time_tensors[500])
            torch.cuda.synchronize()
            del z_warmup

        print(f"Starting inference loop with {n_batches} batches of {batch_size} images each...")

        # Main loop over batches
        for batch_idx in range(n_batches):
            print(f"Processing batch {batch_idx + 1}/{n_batches}")

            # Pre-generate all random noise for this batch
            all_noise = torch.randn(
                (n_time_steps - 1, batch_size, 1) + padded_image_shape, device=device)
            noise_tensors = [all_noise[i] for i in range(n_time_steps - 1)]

            z = torch.randn((batch_size, 1,)+padded_image_shape, device=device)

            # Use a simpler progress tracking to avoid tqdm overhead
            noise_idx = 0
            for t_idx in reversed(range(1, n_time_steps)):
                if t_idx % 200 == 0:  # Reduce printing frequency
                    print(f"  Batch {batch_idx + 1} - Step {n_time_steps - t_idx}/{n_time_steps}")

                # Use precomputed time tensor
                t = time_tensors[t_idx]

                # Use precomputed coefficients (scalars - no GPU operations needed)
                temp = temp_coeffs[t_idx]
                coeff = main_coeffs[t_idx]
                sqrt_beta_t = sqrt_beta[t_idx]

                # Efficient denoising step - minimize intermediate tensors
                with torch.autocast(device_type="cuda"):  # Use automatic mixed precision for faster inference
                    model_output = model(z, t)
                z.mul_(coeff).sub_(model_output, alpha=temp)

                # Use pre-generated noise instead of creating new tensors
                e = noise_tensors[noise_idx]
                z.add_(e, alpha=sqrt_beta_t)
                noise_idx += 1

            # Final denoising step
            t_final = time_tensors[0]
            temp_final = temp_coeffs[0]
            coeff_final = main_coeffs[0]

            with torch.autocast(device_type="cuda"):  # Use AMP for final step too
                model_output = model(z, t_final)
            x = coeff_final * z - temp_final * model_output

            # Process batch samples
            x = rearrange(x, 'b c h w -> b h w c').detach()
            batch_samples = x.cpu().numpy().squeeze()
            batch_samples = padding_fn.inverse(batch_samples).squeeze()

            # Efficient per-sample normalization
            if batch_samples.ndim == 3:  # Multiple samples
                for i in range(batch_samples.shape[0]):
                    sample = batch_samples[i]
                    batch_samples[i] = (sample - sample.min()) / \
                        (sample.max() - sample.min())
            else:  # Single sample
                batch_samples = (batch_samples - batch_samples.min()) / (batch_samples.max() - batch_samples.min())

            all_samples.append(batch_samples)

            # Clean up batch-specific GPU memory
            del all_noise, noise_tensors, z, x
            torch.cuda.empty_cache()

    # Concatenate all batch samples
    if len(all_samples) > 1:
        samples = np.concatenate(all_samples, axis=0)
    else:
        samples = all_samples[0]

    return samples


def inference(cfg,
              checkpoint_path: str = None,
              savepath: str = "images",
              meep_eval: bool = True,
              ):
    n_time_steps = cfg.model.n_time_steps
    ema_decay = cfg.model.ema_decay
    n_images = cfg.n_images if cfg.debug is False else 1
    image_shape = tuple(cfg.image_shape)

    print("INFERENCE")
    checkpoint = torch.load(checkpoint_path, weights_only=True)

    model = hydra.utils.instantiate(cfg.model)
    model = model.to("cuda")

    model.load_state_dict(checkpoint['weights'])
    ema = ModelEmaV3(model, decay=ema_decay)
    ema.load_state_dict(checkpoint['ema'])
    scheduler = DDPM_Scheduler(num_time_steps=n_time_steps)
    times = [0, 15, 50, 100, 200, 300, 400, 550, 700, 999]
    images = []
    z = torch.randn((1, 1,)+image_shape)
    padding_fn = UNetPad(z, depth=model.num_layers//2)

    with torch.no_grad():
        samples = []
        model = ema.module.eval()
        for i in tqdm(range(n_images)):
            z = torch.randn((1, 1,)+image_shape)
            z = padding_fn(z)

            for t in reversed(range(1, n_time_steps)):
                t = [t]
                temp = (scheduler.beta[t]/((torch.sqrt(1-scheduler.alpha[t]))
                                           * (torch.sqrt(1-scheduler.beta[t]))))
                z = (
                    1/(torch.sqrt(1-scheduler.beta[t])))*z - (temp*model(z.cuda(), t).cpu())
                if t[0] in times:
                    images.append(z)
                e = torch.randn((1, 1,) + image_shape)
                e = padding_fn(e)
                z = z + (e*torch.sqrt(scheduler.beta[t]))
            temp = scheduler.beta[0]/((torch.sqrt(1-scheduler.alpha[0]))
                                      * (torch.sqrt(1-scheduler.beta[0])))
            x = (1/(torch.sqrt(1-scheduler.beta[0]))
                 * z - temp*model(z.cuda(), [0]).cpu())

            samples.append(x)
            images.append(x)
            x = rearrange(x.squeeze(0), 'c h w -> h w c').detach()
            x = x.numpy()
            display_reverse(images, savepath, i)
            images = []
    samples = torch.concat(samples, dim=0)
    samples = padding_fn.inverse(samples).squeeze()
    samples = samples.cpu().numpy()
    samples = (samples - samples.min()) / (samples.max() - samples.min())
    np.save(os.path.join(savepath, "images.npy"), samples)

    return samples


def display_reverse(images: List, savepath: str, idx: int):
    fig, axes = plt.subplots(1, 10, figsize=(10, 1))
    for i, ax in enumerate(axes.flat):
        x = images[i].squeeze(0)
        x = rearrange(x, 'c h w -> h w c')
        x = x.numpy()
        ax.imshow(x)
        ax.axis('off')
    plt.savefig(os.path.join(savepath, f"im{idx}.png"))
    plt.close()


@hydra.main(config_path="config", config_name="config")
def main(cfg):
    OmegaConf.set_struct(cfg, False)
    savedir = 'nanophoto/diffusion/train3/'
    savedir = os.path.join(os.environ["SCRATCH"], savedir)
    if cfg.debug:
        savedir = os.path.join(savedir, 'debug')
    else:
        jobid = os.environ["SLURM_JOB_ID"]
        savedir = os.path.join(savedir, jobid)
    if cfg.inference_only:
        checkpoint_path = os.path.expanduser(cfg.checkpoint_load_path)
    else:
        checkpoint_path = os.path.join(savedir, "checkpoint.pt")

    os.makedirs(savedir, exist_ok=True)
    datapath = os.path.expanduser(cfg.data_path)
    data = np.load(datapath)

    n_samples = cfg.training.n_samples if cfg.training.n_samples > 0 else \
                data.shape[0]
    data = data[:n_samples]
    cfg.training.n_epochs = int(cfg.training.n_compute_steps / n_samples)

    if cfg.debug:
        cfg.training.n_images = 1
        cfg.training.n_samples = 16
        cfg.training.n_epochs = 1

    if cfg.inference_only is False:
        run = None
        if cfg.logger:
            run = make_wandb_run(config=dict(cfg), data_path=savedir,
                                 group_name="diffusion data scaling",
                                 run_name=os.environ["SLURM_JOB_ID"])
        train(data=data, checkpoint_path=checkpoint_path,
              savedir=savedir, run=run, cfg=cfg)
    images_savepath = os.path.join(savedir, "images")
    os.makedirs(images_savepath, exist_ok=True)
    images = inference_parallele(checkpoint_path=checkpoint_path,
                                 cfg=cfg)
    np.save(os.path.join(savedir, "images.npy"), images)
    if cfg.inference.fom_eval:
        if cfg.debug is False:
            fom_fn = compute_FOM_parallele
        else:
            def fom_fn(x): return np.random.rand(x.shape[0])
        fom = fom_fn(images)
        np.save(os.path.join(savedir, "fom.npy"), fom)

        plt.hist(fom, bins=100)
        plt.title("fom histogram")
        plt.savefig(os.path.join(savedir, "hist.png"))
        plt.close()
        dataset_cfg = OmegaConf.create([{"name": os.environ["SLURM_JOB_ID"],
                                         "path": images_savepath}])
        eval_metrics(dataset_cfg, os.path.dirname(datapath))


if __name__ == '__main__':
    main()
