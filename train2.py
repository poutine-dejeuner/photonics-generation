"""
modele de diffusion, recréer les champs électriques en ajoutant les indices
"""
import os
import random
from tqdm import tqdm
from typing import List
import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import torch.optim as optim
# from models.ddpm_basic import ddpm_simple
from models.unet import UNET
from models.utils import DDPM_Scheduler, set_seed
from timm.utils import ModelEmaV3
import numpy as np
from icecream import ic

from nanophoto.utils import make_wandb_run

# from icecream import ic


class unet_pad_fun():
    def __init__(self, num_layers, data_sample):
        self.N = 2**num_layers

        def difference_with_next_multiple(self, x):
            reste = x % self.N
            if reste == 0:
                return 0
            else:
                return self.N - reste

        padding = []
        for dim in data_sample.shape[-2:]:
            padding_needed = difference_with_next_multiple(self, dim)
            padding_left = padding_needed // 2
            padding_right = padding_needed - padding_left
            padding.extend([padding_left, padding_right])
        self.padding = padding

    def pad(self, x):
        n = x.ndim
        n = n - len(self.padding) // 2
        padding = [0, 0]*n + self.padding
        padding.reverse()
        padded_tensor = F.pad(x, padding, mode='constant', value=0)
        return padded_tensor

    def crop(self, x):
        a, b, c, d = self.padding[:4]
        cropped_tensor = x[..., a:-b, c:-d]
        return cropped_tensor


def get_indices_fields(num_samples=-1, dtype=torch.float):
    datapath = '~/scratch/nanophoto/lowfom/nodata/fields/'
    datapath = os.path.expanduser(datapath)
    indices = np.load(os.path.join(datapath, 'indices.npy'))[:num_samples]
    fields = np.load(os.path.join(datapath, 'fields.npy'))[:num_samples]
    fields = np.stack([np.real(fields[:, :, :, 0]),
                       np.real(fields[:, :, :, 1])], axis=1)
    # separer les indices et champs en 2 pour reduire le nombre de calculs
    indices = np.expand_dims(indices, axis=1)
    indices = indices[:, :, :, 101:]
    fields = fields[:, :, :, 101:]
    plt.imshow(fields[0, 0])
    plt.savefig('test.png')
    # if debug is True:
    #     indices = indices[:10]
    #     fields = fields[:10]
    indices = torch.tensor(indices, dtype=dtype)
    fields = torch.tensor(fields, dtype=dtype)
    return indices, fields


def train(batch_size: int = 2,
          num_time_steps: int = 1000,
          num_epochs: int = 15,
          seed: int = -1,
          ema_decay: float = 0.9999,
          lr=2e-5,
          checkpoint_path: str = None):
    configs = {
                'batch size':batch_size,
                'num time steps':num_time_steps,
                'num epochs':num_epochs,
                'ema_decay':ema_decay
            }
    savepath = os.path.dirname(checkpoint_path)
    run = make_wandb_run(configs, savepath, 'diffusion', 'fieldpred')
    set_seed(random.randint(0, 2**32-1)) if seed == -1 else set_seed(seed)

    indices, fields = get_indices_fields(num_samples=-1 if debug is False else
                                         2*batch_size)
    train_dataset = TensorDataset(indices, fields)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, drop_last=True, num_workers=4)

    scheduler = DDPM_Scheduler(num_time_steps=num_time_steps)

    model = UNET(input_channels=3, output_channels=2).cuda()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    ema = ModelEmaV3(model, decay=ema_decay)
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['weights'])
        ema.load_state_dict(checkpoint['ema'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    criterion = nn.MSELoss(reduction='mean')
    transfo = unet_pad_fun(num_layers=6, data_sample=fields[0])

    for i in range(num_epochs):
        t0 = time.process_time()
        total_loss = 0
        # for bidx, (struct, field) in enumerate(tqdm(train_loader, desc=f"Epoch {i+1}/{num_epochs}")):
        for bidx, (struct, field) in enumerate(train_loader):
            struct = struct.cuda()
            field = field.cuda()
            # struct = F.pad(struct, (2, 2, 2, 2))
            # field = F.pad(field, (2, 2, 2, 2))
            struct = transfo.pad(struct)
            field = transfo.pad(field)
            t = torch.randint(0, num_time_steps, (batch_size,))
            e = torch.randn_like(field, requires_grad=False)
            a = scheduler.alpha[t].view(batch_size, 1, 1, 1).cuda()
            x = (torch.sqrt(a)*field) + (torch.sqrt(1-a)*e)
            x = torch.concat([struct, x], dim=1)
            output = model(x, t)
            optimizer.zero_grad()
            loss = criterion(output, e)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            ema.update(model)
        run.log({'loss':total_loss/(batch_size*len(train_loader))})
        print(f'Epoch {i+1} | Loss {total_loss / (60000/batch_size):.5f}')
        t1 = time.process_time()
        ic(t1-t0)

    checkpoint = {
        'weights': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'ema': ema.state_dict()
    }
    torch.save(checkpoint, checkpoint_path)


def inference(checkpoint_path: str = None,
              savepath: str = 'inferimages',
              num_time_steps: int = 1000,
              ema_decay: float = 0.9999,):
    if debug is True:
        num_time_steps = 16
    checkpoint = torch.load(checkpoint_path)
    model = UNET(input_channels=3, output_channels=2).cuda()
    model.load_state_dict(checkpoint['weights'])
    ema = ModelEmaV3(model, decay=ema_decay)
    ema.load_state_dict(checkpoint['ema'])
    scheduler = DDPM_Scheduler(num_time_steps=num_time_steps)
    times = [0, 15, 50, 100, 200, 300, 400, 550, 700, 999]
    images = []

    indices, _ = get_indices_fields()
    transfo = unet_pad_fun(num_layers=6, data_sample=indices[0])

    with torch.no_grad():
        model = ema.module.eval()
        for i in range(10):
            indexi = indices[i].unsqueeze(0)
            indexi = transfo.pad(indexi).cuda()
            shape = (1, 2) + indexi.shape[2:]
            z = torch.randn(shape)

            for t in reversed(range(1, num_time_steps)):
                t = [t]
                temp = ( scheduler.beta[t]/((torch.sqrt(1-scheduler.alpha[t]))*(torch.sqrt(1-scheduler.beta[t]))))
                y = model(torch.concat([indexi, z.cuda()], dim=1), t).cpu()
                z = (1/(torch.sqrt(1-scheduler.beta[t])))*z - (temp*y)
                if t[0] in times:
                    sample = z[0,0,:,:]
                    sample = transfo.crop(sample)
                    images.append(sample)
                e = torch.randn(shape)
                z = z + (e*torch.sqrt(scheduler.beta[t]))
            temp = scheduler.beta[0]/((torch.sqrt(1-scheduler.alpha[0]))
                                      * (torch.sqrt(1-scheduler.beta[0])))
            y = model(torch.concat([indexi, z.cuda()], dim=1), t).cpu()
            x = (1/(torch.sqrt(1-scheduler.beta[0]))) * z - (temp*y)

            image = transfo.crop(x[:,0,:,:].squeeze())
            images.append(image)
            # x = rearrange(x.squeeze(0), 'c h w -> h w c').detach()
            # x = x.numpy()
            # plt.imshow(x)
            # plt.show()
            os.makedirs(savepath, exist_ok=True)
            impath = os.path.join(savepath, f'im{i}.png')
            display_reverse(images, savepath=impath)
            images = []


def display_reverse(images: List, savepath: str):
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(n, 1))
    for i, ax in enumerate(axes.flat):
        x = images[i].squeeze(0)
        ic(x.shape)
        # x = rearrange(x, 'c h w -> h w c')
        x = x.numpy()
        ax.imshow(x)
        ax.axis('off')
    plt.savefig(os.path.join(savepath))
    plt.clf()


def main(checkpoint_path, inference_only=False):
    if inference_only is False:
        train(checkpoint_path=checkpoint_path, lr=2e-6, num_epochs=5)
    inference(checkpoint_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('-s', type=int, default=2000)
    parser.add_argument('-d', action='store_true', default=False)
    parser.add_argument('-i', action='store_true', default=False,
        help='inference only')
    args = parser.parse_args()
    global debug
    debug = args.d

    # scratch = os.environ['SCRATCH']
    scratch = "~/scratch"
    chkptdir = os.path.join(scratch, 'nanophoto/diffusion')
    chkptdir = os.path.expanduser(chkptdir)
    if debug:
        chkptdir = os.path.join(chkptdir, 'debug')
    if not os.path.exists(chkptdir):
        os.makedirs(chkptdir)
    checkpoint_path = os.path.join(chkptdir, 'checkpoint.pt')
    
    main(checkpoint_path=checkpoint_path, inference_only=args.i)
