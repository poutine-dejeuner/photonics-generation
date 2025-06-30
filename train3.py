"""
modele de diffusion, apprendre à générer les design
"""
import os
import random
from tqdm import tqdm
from typing import List
import argparse
import time
import datetime

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
# from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import torch.optim as optim
from timm.utils import ModelEmaV3

# from models.ddpm_basic import ddpm_simple
from models.unet import UNET
from models.utils import DDPM_Scheduler, set_seed

from utils import UNetPad, make_wandb_run

from orion.client import report_objective
from icecream import ic, install
ic.configureOutput(includeContext=True)
install()


def train(data: np.ndarray,
          checkpoint_dir: str = None,
          batch_size: int = 16,
          num_time_steps: int = 1000,
          n_epochs: int = 15,
          seed: int = -1,
          ema_decay: float = 0.9999,
          lr=2e-5,
          run = None,
          **kwargs):
    print("TRAINING")
    print(f"{n_epochs} epochs total")
    set_seed(random.randint(0, 2**32-1)) if seed == -1 else set_seed(seed)
    dtype = torch.float32

    data = torch.tensor(data, dtype=dtype)
    data = data.unsqueeze(1)

    scheduler = DDPM_Scheduler(num_time_steps=num_time_steps)
    model = UNET().cuda()
    depth = model.num_layers//2

    transform = UNetPad(data, depth=depth)

    train_dataset = TensorDataset(data)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True,
        num_workers=4)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    ema = ModelEmaV3(model, decay=ema_decay)
    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pt")
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
            output = model(x, t)
            optimizer.zero_grad()
            output = transform.inverse(output)
            loss = criterion(output, e)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            ema.update(model)
        print(f'Epoch {i+1} | Loss {total_loss / len(train_loader):.5f}')
        if run is not None:
            run.log({"loss":total_loss})
        if i % 100 == 0:
            checkpoint = {
                'weights': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'ema': ema.state_dict()
                        }
            torch.save(checkpoint, checkpoint_path)
    #report_objective(loss.item(), 'loss')

def inference(checkpoint_path: str = None,
              n_images: int = 10,
              num_time_steps: int = 1000,
              ema_decay: float = 0.9999,
              savepath: str = "images",
              **kwargs,):
    print("INFERENCE")
    checkpoint_path = os.path.join(checkpoint_path, "checkpoint.pt")
    checkpoint = torch.load(checkpoint_path, weights_only=True)

    image_shape = (101, 91)

    model = UNET().cuda()
    N = 0
    params = list(model.parameters())
    for p in params:
        N += p.numel()
    ic(N)

    model.load_state_dict(checkpoint['weights'])
    ema = ModelEmaV3(model, decay=ema_decay)
    ema.load_state_dict(checkpoint['ema'])
    scheduler = DDPM_Scheduler(num_time_steps=num_time_steps)
    times = [0, 15, 50, 100, 200, 300, 400, 550, 700, 999]
    images = []
    z = torch.randn((1,1,)+image_shape)
    padding_fn = UNetPad(z, depth=model.num_layers//2)

    with torch.no_grad():
        samples = []
        model = ema.module.eval()
        for i in tqdm(range(n_images)):
            z = torch.randn((1,1,)+image_shape)
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


def display_reverse(images: List, savepath: str, idx: int):
    fig, axes = plt.subplots(1, 10, figsize=(10, 1))
    for i, ax in enumerate(axes.flat):
        x = images[i].squeeze(0)
        x = rearrange(x, 'c h w -> h w c')
        x = x.numpy()
        ax.imshow(x)
        ax.axis('off')
    plt.savefig(os.path.join(savepath, f"im{idx}.png"))
    plt.clf()


def main(checkpoint_path, configs):
    datapath = "~/scratch/nanophoto/topoptim/fulloptim/images.npy"
    datapath = os.path.expanduser(datapath)
    data = np.load(datapath)
    n_samples = data.shape[0]
    configs["n_epochs"] = int(5e6 / n_samples)

    if debug is True:
        configs["n_images"] = 1
        configs["n_samples"] = 16
        configs["n_epochs"] = 1
        configs["n_epochs"] = 1

    data = data[:n_samples]
    if inference_only is False:
        # run = make_wandb_run()
        train(data, checkpoint_path, **configs)
    images_savepath = os.path.join(checkpoint_path, "images")
    os.makedirs(images_savepath, exist_ok=True)
    inference(checkpoint_path=checkpoint_path, savepath=images_savepath,
            **configs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_steps', type=int, default=int(5e6))
    parser.add_argument('--n_time_steps', type=int, default=1000)
    parser.add_argument('--n_images', type=int, default=10)
    parser.add_argument('--ema_decay', type=float, default=0.9999)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--chkptdir', type=str, default=None)
    parser.add_argument('-d', action='store_true', default=False)
    parser.add_argument('-i', action='store_true', default=False,
                        help='inference only')
    args = parser.parse_args()
    global debug, inference_only
    debug = args.d
    inference_only = args.i

    if args.chkptdir is not None:
        chkptdir = args.chkptdir
    else:
        chkptdir = 'train3/'
        if debug:
            chkptdir = os.path.join(chkptdir, 'debug')
        else:
            date = datetime.datetime.now().strftime("%m-%d_%Hh%M") 
            chkptdir = os.path.join(chkptdir, date)
        if not os.path.exists(chkptdir):
            os.makedirs(chkptdir)

    configs = vars(args)
    main(checkpoint_path=chkptdir, configs=configs)
