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
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import torch.optim as optim
from timm.utils import ModelEmaV3

from models.unet import UNET
from models.utils import DDPM_Scheduler, set_seed

from utils import UNetPad, AttrDict

from icecream import ic, install

ic.configureOutput(includeContext=True)
install()
np.set_printoptions(precision=2)


def guided_inference(init_image, cfg):
    print("INFERENCE")
    assert init_image.ndim == 4
    image_shape = init_image.shape[-2:]
    n_images = init_image.shape[0]
    num_time_steps = cfg.num_time_steps if cfg.debug is False else 1

    checkpoint_path = os.path.expanduser(cfg.checkpoint_path)
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    model = UNET().cuda()
    model.load_state_dict(checkpoint['weights'])
    ema = ModelEmaV3(model, decay=cfg.ema_decay)
    ema.load_state_dict(checkpoint['ema'])
    scheduler = DDPM_Scheduler(num_time_steps=cfg.num_time_steps)
    times = [0, 15, 50, 100, 200, 300, 400, 550, 700, 999]
    images = []
    z = torch.randn((1,1,)+image_shape)
    padding_fn = UNetPad(z, depth=model.num_layers//2)

    with torch.no_grad():
        samples = []
        model = ema.module.eval()
        for i in tqdm(range(n_images)):
            # z = torch.randn((1,1,)+image_shape)
            z = init_image
            # ic(z.shape, z.min(), z.max())
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
            # display_reverse(images, cfg.savepath, i)
            images = []
    samples = torch.concat(samples, dim=0)
    samples = padding_fn.inverse(samples).squeeze()
    samples = samples.cpu().numpy()
    samples = (samples - samples.min()) / (samples.max() - samples.min())
    # np.save(os.path.join(cfg.savepath, "images.npy"), samples)
    return samples


def display_reverse(images: List, savepath: str, idx: int):
    ic(savepath)
    fig, axes = plt.subplots(1, 10, figsize=(10, 1))
    for i, ax in enumerate(axes.flat):
        x = images[i].squeeze(0)
        x = rearrange(x, 'c h w -> h w c')
        x = x.numpy()
        ax.imshow(x)
        ax.axis('off')
    plt.savefig(os.path.join(savepath, f"im{idx}.png"))
    plt.clf()

def comparison(x, y):
    sim = np.dot(x.flatten(), y.flatten())
    dist = np.linalg.norm(x.flatten() - y.flatten())
    return sim, dist

def main(config):
    dtype = torch.float32
    os.makedirs(config.savepath, exist_ok=True)
    guide_file = os.path.expanduser(config.guide_file)
    guide = torch.tensor(np.load(guide_file)).unsqueeze(0).unsqueeze(0)
    image_shape = guide.shape[-2:]
    noise = torch.randn((1, 1,) + image_shape, dtype=dtype)

    rand_gen = guided_inference(noise, config).squeeze()
    print("rand gen")
    comparison(guide.numpy(), rand_gen)

    out_images = [guide.squeeze()]
    in_images = [guide.squeeze()]
    comparison_list = [(1, 0)]
    guide_norm = torch.linalg.vector_norm(guide) ** 2
    N = 3
    titles = [
            "guide",
            "random gen",
            ]

    # add random gen
    in_images.append(noise.numpy().squeeze())
    rand_gen = guided_inference(noise, config).squeeze()
    out_images.append(rand_gen.squeeze())

    print("rand gen:")
    sim, dist = comparison(guide.numpy(), rand_gen)
    comparison_list.append((sim / guide_norm, dist))

    # add guided gen
    indices = list(range(N))
    scales = [10 ** (-i) for i in indices]
    scales.insert(0, 2)
    for scale in scales:
        noisy_guide = (scale * guide + noise).to(torch.float32)
        in_images.append(noisy_guide.numpy().squeeze())
        guided_gen = guided_inference(noisy_guide, config).squeeze()
        out_images.append(guided_gen.squeeze())
        titles.append(f"guided {scale}")

        print("scale:", scale)
        sim, dist = comparison(guide.numpy(), guided_gen)
        comparison_list.append((sim / guide_norm, dist))


    _, axes = plt.subplots(2, len(in_images))
    for i in range(len(in_images)):
        axes[0,i].imshow(in_images[i])
        axes[0,i].axis("off")
        axes[0,i].set_title(titles[i])
        axes[1,i].imshow(out_images[i])
        axes[1,i].axis("off")
        # axes[1,i].set_title(titles[i])
        sim, dist = comparison_list[i]
        axes[1,i].text(0.5, -10, f"{sim:.2f}, {dist:.2f}")

    plt.tight_layout()
    plt.savefig(os.path.join(config.savepath, "guided_gen_tests.png"))


if __name__ == '__main__':
    import yaml
    with open("config.yaml") as file:
        config = yaml.safe_load(file)
    config = AttrDict(config)
    main(config)
