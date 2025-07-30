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

from utils import UNetPad, make_wandb_run, display_reverse
from nanophoto.meep_compute_fom import compute_FOM_parallele
from nanophoto.evaluation.evalgen import eval_metrics

 # from orion.client import report_objective
from icecream import ic, install

ic.configureOutput(includeContext=True)
install()
OmegaConf.register_new_resolver("eval", eval)
# OmegaConf.register_new_resolver("eval", lambda expr: eval(expr))


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

    train = hydra.utils.instantiate(cfg.train)
    inference = hydra.utils.instantiate(cfg.inference)

    os.makedirs(savedir, exist_ok=True)
    datapath = os.path.expanduser(cfg.data_path)
    images_savepath = os.path.join(savedir, "images")
    os.makedirs(images_savepath, exist_ok=True)
    data = np.load(datapath)

    modcfg = cfg.model
    n_samples = data.shape[0] if modcfg.n_samples == -1 else modcfg.n_samples 
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
