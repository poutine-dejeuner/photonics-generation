import os
import time
from icecream import ic

import torch
import numpy as np

import matplotlib.pyplot as plt

import hydra
from omegaconf import OmegaConf


from train3 import inference_parallele
from nanophoto.evaluation.evalgen import eval_metrics

@hydra.main(config_path="config", config_name="config")
def test__inference(cfg):
    for n in range(2, 8):
        n_samples = 2 ** n
        ic(n_samples)
        OmegaConf.set_struct(cfg, False)
        checkpoint_path = os.path.expanduser(cfg.checkpoint_load_path)
        images_savepath = "test"
        os.makedirs(images_savepath, exist_ok=True)
        t0 = time.time()
        images = inference_parallele(cfg=cfg,
                                checkpoint_path=checkpoint_path,
                                savepath=images_savepath,
                                meep_eval=True)
        t1 = time.time()
        ic(t1-t0, n_samples)
        fig, axes = plt.subplots(4,4)
        for i, ax in enumerate(axes.flatten()):
            ax.axis('off')
            ax.imshow(images[i].cpu().numpy().squeeze())
        plt.tight_layout()
        plt.savefig(os.path.join("test", f"test_{n_samples}.png"))

@hydra.main(config_path="config", config_name="inference")
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

    for model_cfg in cfg.models:
        modconfig = cfg.model
        modconfig.n_epochs = int(modconfig.n_compute_steps / n_samples)

        images_savepath = os.path.join(savedir, "images")
        os.makedirs(images_savepath, exist_ok=True)
        t0 = time.time()
        images, fom = inference(checkpoint_path=checkpoint_path, savepath=images_savepath,
                                cfg=cfg)
        t1 = time.time()
        ic(t1-t0, "inference time")
        plt.hist(fom, bins=100)
        plt.title("fom histogram")
        plt.savefig(os.path.join(savedir, "hist.png"))
        plt.close()
        dataset_cfg = OmegaConf.create([{"name": os.environ["SLURM_JOB_ID"],
                                         "path": images_savepath}])
        eval_metrics(dataset_cfg, os.path.dirname(datapath))


if __name__ == "__main__":
    test__inference()
    # main()
