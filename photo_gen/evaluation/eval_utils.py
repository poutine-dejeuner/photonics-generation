import os
from pathlib import Path
import yaml

import torch


def normalise(images):
    """rescales the array to have only values in [0,1]"""
    if images.max() - images.min() == 0:
        return images
    images = (images - images.min())/(images.max() - images.min())

    return images


def tonumpy(array):
    if type(array) is torch.Tensor:
        return array.detach().cpu().numpy()
    else:
        return array



def update_stats_yaml(stats_path: Path, new_stats: dict) -> None:
    """
    Update or add a key-value pair in a YAML file.
    If the key exists, overwrite it. If not, add it.
    """

    # Load existing data or create empty dict
    if os.path.exists(stats_path):
        with open(stats_path, 'r', encoding='utf-8') as f:
            try:
                stats = yaml.safe_load(f) or {}
            except yaml.YAMLError:
                stats = {}
    else:
        stats = {}

    stats = new_stats | stats

    # Write back to file
    with open(stats_path, 'w', encoding='utf-8') as f:
        yaml.dump(stats, f, default_flow_style=False)

def make_config(path):
    """
    search subdirectories of path. If dir/images contains an images.npy and
    some subdirectory of wandb contains a files/config.yaml, then search
    config.yaml for a "n_samples" key and record the following entries in a
    list
    "name": dir
    "path": dir/images
    "n_samples": config["n_samples"]
    Finally, save the discovered entries in a config.yaml file
    """
    datasets = []
    for subdir in Path(path).iterdir():
        if subdir.is_dir() and (subdir / 'images.npy').exists():
            images_path = subdir / 'images.npy'
            config_path = subdir / 'wandb' / 'files' / 'config.yaml'
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                n_samples = config.get('n_samples', None)
                datasets.append({
                    "name": subdir.name,
                    "path": str(images_path),
                    "n_samples": n_samples
                })
    with open(os.path.join(path, 'datasets.yaml'), 'w', encoding='utf-8') as f:
        yaml.dump(datasets, f)