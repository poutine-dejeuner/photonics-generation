"""
modele de diffusion, apprendre à générer les design
"""
import os
import random
from tqdm import tqdm
from typing import List
import datetime

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

from utils import UNetPad, make_wandb_run
from nanophoto.meep_compute_fom import compute_FOM_parallele
from nanophoto.evaluation.evalgen import eval_metrics

# from orion.client import report_objective
from icecream import ic, install

ic.configureOutput(includeContext=True)
install()
OmegaConf.register_new_resolver("eval", eval)
# OmegaConf.register_new_resolver("eval", lambda expr: eval(expr))


def train(data: np.ndarray, cfg, checkpoint_path: os.path, savedir: os.path,
          run=None):
    seed = -1
    n_epochs = cfg.n_epochs
    lr = cfg.lr
    batch_size = cfg.batch_size
    num_time_steps = cfg.num_time_steps
    ema_decay = cfg.ema_decay
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("TRAINING")
    print(f"{n_epochs} epochs total")
    set_seed(random.randint(0, 2**32-1)) if seed == -1 else set_seed(seed)
    dtype = torch.float32

    data = torch.tensor(data, dtype=dtype)
    data = data.unsqueeze(1)

    scheduler = DDPM_Scheduler(num_time_steps=num_time_steps)

    model = hydra.utils.instantiate(cfg)
    model = model.to(device)
    depth = model.num_layers//2

    transform = UNetPad(data, depth=depth)

    train_dataset = TensorDataset(data)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True,
        num_workers=4)

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
            t = torch.randint(0, num_time_steps, (batch_size,))
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
        if i % 100 == 0:
            checkpoint = {
                'weights': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'ema': ema.state_dict()
            }
            torch.save(checkpoint, checkpoint_path)
    # report_objective(loss.item(), 'loss')
    return total_loss


def inference(cfg,
              checkpoint_path: str = None,
              savepath: str = "images",
              meep_eval: bool = True,
              ):
    num_time_steps = cfg.model.num_time_steps
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
    scheduler = DDPM_Scheduler(num_time_steps=num_time_steps)
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

            for t in reversed(range(1, num_time_steps)):
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
            x = (1/(torch.sqrt(1-scheduler.beta[0]))) * \
                z - (temp*model(z.cuda(), [0]).cpu())

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

    if meep_eval:
        if cfg.debug is False:
            fom_fn = compute_FOM_parallele 
        else:
            fom_fn = lambda x: np.random.rand(x.shape[0])
        fom = fom_fn(samples)
        np.save(os.path.join(savepath, "fom.npy"), fom)
    else:
        fom = np.empty((0,))

    return samples, fom


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

    modcfg = cfg.model
    n_samples = cfg.hydra.sweeper.params.n_samples
    data = data[:n_samples]
    modcfg.n_epochs = int(modcfg.n_compute_steps / n_samples)

    if cfg.debug:
        modcfg.n_images = 1
        modcfg.n_samples = 16
        modcfg.n_epochs = 1

    if cfg.inference_only is False:
        run = None
        if cfg.logger:
            run = make_wandb_run(config=dict(cfg), data_path=savedir,
                                 group_name="diffusion data scaling",
                                 run_name=os.environ["SLURM_JOB_ID"])
        train(data=data, checkpoint_path=checkpoint_path,
              savedir=savedir, run=run, cfg=modcfg)
    images_savepath = os.path.join(savedir, "images")
    os.makedirs(images_savepath, exist_ok=True)
    images, fom = inference(checkpoint_path=checkpoint_path, savepath=images_savepath,
              cfg=cfg)
    plt.hist(fom, bins=100)
    plt.title("fom histogram")
    plt.savefig(os.path.join(savedir, "hist.png"))
    plt.close()
    dataset_cfg = OmegaConf.create([{"name":os.environ["SLURM_JOB_ID"],
                   "path": images_savepath}])
    eval_metrics(dataset_cfg, os.path.dirname(datapath))
    return fom.mean()

if __name__ == '__main__':
    main()
