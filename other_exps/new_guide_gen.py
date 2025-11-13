import os
from tqdm import tqdm
from typing import List
from pathlib import Path

import numpy as np
import torch
from einops import rearrange
import matplotlib.pyplot as plt
from timm.utils import ModelEmaV3
from omegaconf import OmegaConf

from photo_gen.models.unet import UNET
from photo_gen.utils.unet_utils import DDPM_Scheduler, UNetPad
from photo_gen.evaluation.evaluation import CompareToTrainClosestImage, compute_FOM_parallele


from icecream import ic, install

ic.configureOutput(includeContext=True)
install()
np.set_printoptions(precision=2)


def guided_inference(init_image, cfg):
    print("INFERENCE")
    assert init_image.ndim == 4
    image_shape = init_image.shape[-2:]
    n_images = init_image.shape[0]
    num_time_steps = cfg.model.num_time_steps if cfg.debug is False else 1

    checkpoint = torch.load(cfg.checkpoint_path, weights_only=True)

    model = UNET(
        first_channels=cfg.model.first_channels,
        num_layers=cfg.model.num_layers,
        input_channels=cfg.model.input_channels,
        output_channels=cfg.model.output_channels,
        Attentions=cfg.model.Attentions,
        Upscales=cfg.model.Upscales,
        num_groups=cfg.model.num_groups,
        num_heads=cfg.model.num_heads,
        dropout_prob=cfg.model.dropout_prob
    ).cuda()

    model.load_state_dict(checkpoint['weights'])
    ema = ModelEmaV3(model, decay=cfg.model.ema_decay)
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
            z = init_image
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
            images = []
    samples = torch.concat(samples, dim=0)
    samples = padding_fn.inverse(samples).squeeze()
    samples = samples.cpu().numpy()
    samples = (samples - samples.min()) / (samples.max() - samples.min())
    return samples


def comparison(x, y):
    sim = np.dot(x.flatten(), y.flatten())
    dist = np.linalg.norm(x.flatten() - y.flatten())
    return sim, dist


def main():
    basepath = Path("checkpoints/unet_fast").resolve()
    configpath = basepath / "config.yaml"
    # with open(configpath, 'r') as f:
        # cfg = initialize(f)

    cfg = OmegaConf.load(configpath)
    cfg.checkpoint_path = basepath / "checkpoint.pt"
    dtype = torch.float32
    savepath = basepath / "gen"
    os.makedirs(savepath, exist_ok=True)

    # cfg.debug = True

    if cfg.debug:
        cfg.model.num_time_steps = 10

    all_guides = []
    num_samples = 32 if not cfg.debug else 4
    guide_size = cfg.image_shape

    for _ in range(num_samples):
        guide = np.random.rand(*guide_size).astype(np.float32)
        from scipy.ndimage import gaussian_filter
        guide = gaussian_filter(guide, sigma=3)
        guide = (guide - guide.min()) / (guide.max() - guide.min())
        all_guides.append(guide)
    all_guides = np.stack(all_guides, axis=0)

    # Process each sample
    for sample_idx, guide_data in enumerate(all_guides):
        print(f"\nProcessing sample {sample_idx + 1}/{all_guides.shape[0]}")

        # Prepare guide tensor
        guide = torch.tensor(guide_data).unsqueeze(0).unsqueeze(0).to(dtype)
        image_shape = guide.shape[-2:]
        noise = torch.randn((1, 1,) + image_shape, dtype=dtype)

        # Generate random sample for comparison
        rand_gen = guided_inference(noise, cfg).squeeze()
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
        rand_gen = guided_inference(noise, cfg).squeeze()
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
            guided_gen = guided_inference(noisy_guide, cfg).squeeze()
            out_images.append(guided_gen.squeeze())
            titles.append(f"guided {scale}")

            print("scale:", scale)
            sim, dist = comparison(guide.numpy(), guided_gen)
            comparison_list.append((sim / guide_norm, dist))

        # Create visualization for this sample
        _, axes = plt.subplots(2, len(in_images))
        for i in range(len(in_images)):
            axes[0, i].imshow(in_images[i])
            axes[0, i].axis("off")
            axes[0, i].set_title(titles[i])
            axes[1, i].imshow(out_images[i])
            axes[1, i].axis("off")
            # axes[1,i].set_title(titles[i])
            sim, dist = comparison_list[i]
            axes[1, i].text(0.5, -10, f"{sim:.2f}, {dist:.2f}")

        plt.tight_layout()
        # Save with sample index in filename
        plt.savefig( savepath / f"guided_gen_tests_sample_{sample_idx:03d}.png")
        plt.close()  # Close the figure to free memory

        # Save the generated samples for this guide
        sample_results = {
            'guide': guide.squeeze().numpy(),
            'random_gen': rand_gen,
            'guided_gens': [out_images[i] for i in range(2, len(out_images))],
            'scales': scales,
            'comparisons': comparison_list
        }
        out_images = np.stack(out_images)
        np.save(savepath / "guided_gen_images.npy", out_images)
        compare = CompareToTrainClosestImage(train_set_path=cfg.data.path)
        compare(images=out_images, savepath=savepath, model_name="diffusion",
                cfg=cfg)
        fom = compute_FOM_parallele(out_images)
        np.save(savepath / "fom.npy", fom)


if __name__ == '__main__':
    main()
