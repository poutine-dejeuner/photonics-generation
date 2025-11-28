"""
Contains functions to be called at the end of a training run to evaluate the
image generation of the model.
"""
import os
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union
import yaml

import numpy as np
import torch
import matplotlib.pyplot as plt
from omegaconf import OmegaConf, DictConfig
import hydra
import infomeasure as im

from photo_gen.utils.utils import load_wandb_config
from photo_gen.evaluation.eval_utils import (
    update_stats_yaml, tonumpy, find_files)
from photo_gen.evaluation.meep_compute_fom import compute_FOM_parallele


class EvaluationFunction(ABC):
    """Abstract base class for all evaluation functions."""

    def __init__(self, **kwargs):
        """Initialize the evaluation function with any configuration parameters."""
        self.config = kwargs

    @abstractmethod
    def __call__(self, images: np.ndarray, savepath: str, model_name: str,
                 fom: Optional[np.ndarray] = None, cfg: Optional[OmegaConf] = None) -> Any:
        """
        Execute the evaluation function.

        Args:
            images: Generated images array
            fom: Figure of merit values (optional)
            savepath: Directory to save results (optional)
            model_name: Name of the model (optional)
            cfg: Configuration object (optional)

        Returns:
            Evaluation results (type depends on specific function)
        """
        pass

    @property
    def name(self) -> str:
        """Return the name of this evaluation function."""
        return self.__class__.__name__


class CompareToTrainClosestImage(EvaluationFunction):
    """
    Plot the generated image alongside its closest match from the training set.
    """

    def __init__(self, train_set_path: os.PathLike, **kwargs):
        train_set_path = os.path.expanduser(train_set_path)
        self.train_set = np.load(train_set_path)

    def __call__(self, images: np.ndarray, savepath: str, model_name: str,
                 fom: Optional[np.ndarray] = None, cfg: Optional[OmegaConf] = None):
        plot_closest_images2(images=images, fom=fom, train_set=self.train_set, savepath=savepath)



def closest_image(img: np.ndarray, train_set: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Find the closest image in the training set to the given image using L2 norm.

    Args:
        img: Generated image of shape (H, W) or (C, H, W)
        train_set: Training set images of shape (N, C, H, W) or (N, H, W)
    """
    img_flat = img.flatten()
    min_dist = float('inf')
    closest_img = None
    for train_img in train_set:
        train_img_flat = train_img.flatten()
        dist = np.linalg.norm(img_flat - train_img_flat)
        if dist < min_dist:
            min_dist = dist
            closest_img = train_img
    return closest_img, min_dist
        
        
def plot_closest_images(images: np.ndarray, fom: np.ndarray, train_set: np.ndarray, 
                        savepath: str = "."):
    """
    Plot some generated image alongside its closest match from the training set.
    The distances are computed in L2 norm. Then, the interval [min_dist, max_dist] is separated in 3 equal parts and N images are uniformly sampled from each part according to their distance to the closest training image.
    A figure is created, it has 3 rows (one per distance range) and N columns (one per image). Each subplot contains the generated image on the left and its closest training image on the right.

    Args:
        images: Generated images array of shape (N, C, H, W) or (N, H, W)
        savepath: Directory to save the comparison plots
        model_name: Name of the model for file naming
        cfg: Configuration object (unused)
    """
    images = images.squeeze()
    train_set = train_set.squeeze()
    distances = []
    closest_images = []
    for img in images:
        ts_image, min_dist = closest_image(img, train_set)
        distances.append(min_dist)
        closest_images.append(ts_image)
    distances = np.array(distances)
    closest_images = np.array(closest_images)
    min_dist = distances.min()
    max_dist = distances.max()
    n_samples_per_range = 4
    distance_ranges = np.linspace(min_dist, max_dist, 4)
    fig, axes = plt.subplots(3, n_samples_per_range, figsize=(15, 9))
    fig.suptitle('Generated vs Closest Training Images',
                 fontsize=16, fontweight='bold')
    for i in range(3):
        range_min = distance_ranges[i]
        range_max = distance_ranges[i + 1]
        indices_in_range = np.where(
            (distances >= range_min) & (distances < range_max))[0]
        if len(indices_in_range) == 0:
            continue
        sampled_indices = np.random.choice(
            indices_in_range, size=min(n_samples_per_range, len(indices_in_range)), replace=False)
        for j, idx in enumerate(sampled_indices):
            ax = axes[i, j]
            gen_img = images[idx]
            train_img = closest_images[idx]
            gen_img = (gen_img - gen_img.min()) / (gen_img.max() - gen_img.min() + 1e-8)
            train_img = (train_img - train_img.min()) / (train_img.max() - train_img.min() + 1e-8)
            combined_img = np.concatenate((gen_img, train_img), axis=1)
            ax.imshow(combined_img, vmin=0, vmax=1)
            ax.set_title(f'FOM: {fom[idx]:.3f}, Dist: {distances[idx]:.4f}')
            ax.axis('off')
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    save_file = os.path.join(savepath,
                             "generated_vs_closest_train.png")
    plt.savefig(save_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_closest_images2(images: np.ndarray, fom: np.ndarray, train_set: np.ndarray,  savepath: str = "."):
    """
    Plot some generated image alongside its closest match from the training set.
    The distances are computed in L2 norm. Then, the interval [min_dist, max_dist] is separated in 3 equal parts and N images are uniformly sampled from each part according to their distance to the closest training image.
    A figure is created, it has 3 rows (one per distance range) and N columns (one per image). Each subplot contains the generated image on the left and its closest training image on the right.

    Args:
        images: Generated images array of shape (N, C, H, W) or (N, H, W)
        fom: 
        savepath: Directory to save the comparison plots
        model_name: Name of the model for file naming
        cfg: Configuration object (unused)
    """
    images = images.squeeze()
    train_set = train_set.squeeze()
    distances = []
    closest_images = []
    for img in images:
        ts_image, min_dist = closest_image(img, train_set)
        distances.append(min_dist)
        closest_images.append(ts_image)
    distances = np.array(distances)
    closest_images = np.array(closest_images)
    min_dist = distances.min()

    n_samples = 4
    indices = np.flip(np.argsort(distances))[:4]
    
    fig, axes = plt.subplots(2, n_samples, figsize=(15, 9))

    for j, idx in enumerate(indices):
        
        gen_img = images[idx]
        train_img = closest_images[idx]
        gen_img = (gen_img - gen_img.min()) / (gen_img.max() - gen_img.min() + 1e-8)
        train_img = (train_img - train_img.min()) / (train_img.max() - train_img.min() + 1e-8)

        axes[0, j].imshow(gen_img, vmin=0, vmax=1)
        axes[0, j].set_title(f'FOM: {fom[idx]:.3f}, Dist: {distances[idx]:.4f}')
        axes[0, j].axis('off')
        axes[1, j].imshow(train_img, vmin=0, vmax=1)
        axes[1, j].axis('off')
    plt.subplots_adjust(top=0.92)
    save_file = os.path.join(savepath,
                             "generated_vs_closest_train.png")
    plt.savefig(save_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_closest_images3(images: np.ndarray, fom: np.ndarray, train_set: np.ndarray, 
                        savepath: str = "."):
    images = (images - images.min()) / (images.max() - images.min())
    train_set = (train_set - train_set.min()) / (train_set.max()
            - train_set.min())
    images = images.squeeze()
    train_set = train_set.squeeze()
    distances = []
    closest_images = []
    for img in images:
        ts_image, min_dist = closest_image(img, train_set)
        distances.append(min_dist)
        closest_images.append(ts_image)
    distances = np.array(distances)
    closest_images = np.array(closest_images)
    n_samples_per_range = 4
    fig, axes = plt.subplots(2, n_samples_per_range, figsize=(15, 5))
    indices = np.zeros((2, n_samples_per_range))
    indices[0, :] = np.argsort(distances)[:n_samples_per_range]
    indices[1, :] = np.flip(np.argsort(distances))[:n_samples_per_range]
    indices = indices.astype(int)
    for i in range(2):
        for j in range(n_samples_per_range):
            ax = axes[i, j]
            idx = indices[i, j]
            gen_img = images[idx]
            train_img = closest_images[idx]
            gen_img = (gen_img - gen_img.min()) / (gen_img.max() - gen_img.min() + 1e-8)
            train_img = (train_img - train_img.min()) / (train_img.max() - train_img.min() + 1e-8)
            combined_img = np.concatenate((gen_img, train_img), axis=1)
            ax.imshow(combined_img, vmin=0, vmax=1)
            ax.set_title(f'FOM: {fom[idx]:.3f}, Dist: {distances[idx]:.4f}')
            ax.axis('off')
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    save_file = os.path.join(savepath,
                             "generated_vs_closest_train.png")
    plt.savefig(save_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_closest_images_test():
    path = Path("../../data/diffusion/unet_fast")
    train_set = Path("../../data/topoptim/images.npy")
    images = (path / "images.npy").resolve()
    fom = (path / "fom.npy").resolve()
    train_set = np.load(train_set)
    images = np.load(images)
    fom = np.load(fom)
    plot_closest_images(images, fom, train_set)


def plot_closest_images2_test():
    path = Path("../../data/diffusion/unet_fast")
    train_set = Path("../../data/topoptim/images.npy")
    images = (path / "images.npy").resolve()
    fom = (path / "fom.npy").resolve()
    train_set = np.load(train_set)
    images = np.load(images)
    fom = np.load(fom)
    plot_closest_images2(images, fom, train_set)


def plot_closest_images3_test():
    path = Path("../../data/diffusion/unet_fast")
    train_set = Path("../../data/topoptim/images.npy")
    images = (path / "images.npy").resolve()
    fom = (path / "fom.npy").resolve()
    train_set = np.load(train_set)
    images = np.load(images)
    fom = np.load(fom)
    plot_closest_images3(images, fom, train_set)
    

def ts_dist_hist(images: np.ndarray, train_set: np.ndarray):
    distances = []
    for img in images:
        ts_image, min_dist = closest_image(img, train_set)
        distances.append(min_dist)
    plt.hist(distances, bins=100)
    plt.savefig("ts_dist_hist.png")


def ts_dist_hist_test():
    images = Path("../../data/diffusion/unet_fast/images.npy")
    train_set = Path("../../data/topoptim/images.npy")
    train_set = np.load(train_set)
    images = np.load(images)
    ts_dist_hist(images, train_set)


class BinarizationLoss(EvaluationFunction):
    """
    Measure the binarization of the generated images (average distance to 0 or 1).
    """

    def __call__(self, images: np.ndarray, savepath: str, model_name: str,
                 fom: Optional[np.ndarray] = None, cfg: Optional[OmegaConf] = None) -> float:
        """
        Measure the binarization of the generated images.

        Args:
            images: Generated images array of shape (N, C, H, W) or (N, H, W)
            savepath: Directory to save results (unused)
            model_name: Name of the model (unused)
            fom: Figure of merit values (unused)
            cfg: Configuration object (unused)

        Returns:
            Average binarization metric across all images
        """
        images = images.reshape(images.shape[0], -1)
        binarization_metric = np.minimum(
            np.abs(images - 0), np.abs(images - 1))
        avg_binarization = float(binarization_metric.mean())
        return avg_binarization


class VisualizeGeneratedSamples(EvaluationFunction):
    """Create a grid visualization of generated samples and save it."""

    def __init__(self, n_samples: int = 16, **kwargs):
        super().__init__(**kwargs)
        self.n_samples = n_samples

    def __call__(self, images: np.ndarray, savepath: str, model_name: str,
                 fom: Optional[np.ndarray] = None, cfg: Optional[OmegaConf] = None) -> str:
        """
        Create a grid visualization of generated samples and save it.

        Args:
            images: Generated images array of shape (N, C, H, W) or (N, H, W)
            savepath: Directory to save the visualization
            model_name: Name of the model for the title
            cfg: Configuration object (unused)

        Returns:
            Path to the saved visualization file
        """

        n_samples = min(self.n_samples, images.shape[0])
        assert n_samples >= 4
        images = images.squeeze()
        grid_size = int(np.ceil(np.sqrt(n_samples)))
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))

        # Ensure axes is always a 2D array for consistent indexing
        if grid_size == 1:
            axes = np.array([[axes]])
        elif len(axes.shape) == 1:
            axes = axes.reshape(-1, 1)

        fig.suptitle(f'{model_name} - Generated Samples',
                     fontsize=16, fontweight='bold')

        sample_idx = 0
        for i in range(grid_size):
            for j in range(grid_size):
                ax = axes[i, j]
                if sample_idx >= n_samples:
                    # Hide empty subplots
                    ax.axis('off')
                    continue

                img = images[sample_idx]
                img = (img - img.min()) / \
                    (img.max() - img.min() + 1e-8)
                ax.imshow(img, vmin=0, vmax=1)
                ax.set_title(f'samples from {model_name}', fontsize=10)
                ax.axis('off')
                sample_idx += 1

        plt.tight_layout()
        plt.subplots_adjust(top=0.93)

        save_file = os.path.join(savepath,
                                 f"{model_name.lower()}_samples_grid.png")
        plt.savefig(save_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        print(f"Sample grid visualization saved: {save_file}")
        return save_file


class PlotFomHistogram(EvaluationFunction):
    """Plot histogram of Figure of Merit values."""

    def __call__(self, images: np.ndarray, savepath: str, model_name: str,
                 fom: Optional[np.ndarray] = None, cfg: Optional[OmegaConf] = None) -> str:
        """Plot FOM histogram and save it."""
        if fom is None:
            return "No FOM values provided for histogram."

        plt.figure(figsize=(10, 6))
        plt.hist(fom, bins=100, alpha=0.7, range=[0, 0.5])
        plt.title(f"FOM Histogram - {model_name}")
        plt.xlabel("Figure of Merit")
        plt.ylabel("Frequency")
        plt.grid(True, alpha=0.3)

        save_file = os.path.join(savepath, "fom_histogram.png")
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
        plt.close()

        return save_file


class FOM(EvaluationFunction):
    """Compute Figure of Merit for generated images."""

    def __call__(self, images: np.ndarray, savepath: str, model_name: str,
                 fom: Optional[np.ndarray] = None, cfg: Optional[OmegaConf] = None) -> tuple[float, float]:
        if cfg and hasattr(cfg, 'debug') and cfg.debug and hasattr(cfg, 'meep') and not cfg.meep:
            computed_fom = np.random.rand(images.shape[0])
        else:
            computed_fom = compute_FOM_parallele(images)

        np.save(os.path.join(savepath, "fom.npy"), computed_fom)
        return computed_fom


def pca_dim_reduction_entropy(images, dim, n_neighbors):
    """
        returns the per-dimension entropy (entropy divided by dimension) of
        the data after projection to the first dim PCA components.
    """
    from sklearn.decomposition import PCA
    import infomeasure as im

    x = images.reshape(images.shape[0], -1)
    pca = PCA(n_components=dim)
    x_pca = pca.fit_transform(x)
    h = im.entropy(x_pca, approach="metric", k=n_neighbors)
    return float(h)


class PCAProjPerDimEntropy(EvaluationFunction):
    def __init__(self, n_neighbors: int = 4, dim: int = 50, **kwargs):
        """
        n_neighbors: the number of nearest neighbors to use for the estimation of the entropy
        dim: the number of PCA components to use for projection
        """
        self.n_neighbors = n_neighbors
        self.dim = dim

    def __call__(self, images: np.ndarray, **kwargs):
        return pca_dim_reduction_entropy(images, self.dim, self.n_neighbors)


class Entropy(EvaluationFunction):
    def __init__(self, n_neighbors: int = 4, **kwargs):

        self.n_neighbors = n_neighbors

    def __call__(self, images: np.ndarray, savepath: str, model_name: str,
                 fom: Optional[np.ndarray] = None, cfg: Optional[OmegaConf] = None) -> float:

        from infomeasure import entropy

        images = images.reshape(images.shape[0], -1)
        h = entropy(images, approach="metric", k=self.n_neighbors)
        return float(h)


class NNDistanceTrainSet(EvaluationFunction):
    def __init__(self, train_set_path: os.PathLike, **kwargs):
        train_set_path = os.path.expanduser(train_set_path)
        self.train_set = np.load(train_set_path)

    def __call__(self, images: np.ndarray, savepath: str, model_name: str,
                 fom: Optional[np.ndarray] = None, cfg: Optional[OmegaConf] = None) -> dict[str, float]:
        distances_dict = nn_distance_to_train_ds(
            model_name, images, self.train_set, savepath)

        return distances_dict


def nn_distance_to_train_ds(ds_name: str,
                            gen_ds: torch.Tensor | np.ndarray,
                            train_ds: torch.Tensor | np.ndarray,
                            savepath: str) -> dict[str, float]:
    """
        For each element in ds1, compute the distance to the nearest element in
        ds2.
    """
    if isinstance(gen_ds, np.ndarray):
        gen_ds = torch.tensor(gen_ds)
    if isinstance(train_ds, np.ndarray):
        train_ds = torch.tensor(train_ds)

    distances = []
    for x in gen_ds:
        min_dist = float('inf')
        for y in train_ds:
            dist = torch.norm(x - y)
            if dist < min_dist:
                min_dist = dist

        distances.append(min_dist)
    # make nn distance histogram
    distances = tonumpy(torch.stack(distances))
    plt.hist(distances, bins=100, density=True, label=ds_name, alpha=0.5)
    plt.title('Nearest training set neighbor distances')
    plt.savefig(os.path.join(savepath, 'nn_distance_histogram.png'))
    plt.legend()
    plt.close()

    results = {"mean": distances.mean().item(), "std": distances.std().item()}
    return results


class PairwiseDistanceEntropy(EvaluationFunction):
    """Compute entropy of pairwise distances between generated images."""

    def __init__(self, n_neighbors: int = 4, **kwargs):
        """
        Args:
            n_neighbors: Number of nearest neighbors for entropy estimation
        """
        super().__init__(**kwargs)
        self.n_neighbors = n_neighbors

    def __call__(self, images: np.ndarray, savepath: str, model_name: str,
                 fom: Optional[np.ndarray] = None, cfg: Optional[OmegaConf] = None) -> float:
        """
        Compute entropy of pairwise distances between images.

        Returns:
            Entropy value of the pairwise distance distribution
        """
        from scipy.spatial.distance import pdist

        images_flat = images.reshape(images.shape[0], -1)
        pairwise_distances = pdist(images_flat, metric='euclidean')
        distances_array = pairwise_distances.reshape(-1, 1)

        entropy_value = im.entropy(
            distances_array, approach="metric", k=self.n_neighbors)

        return float(entropy_value)


class ImageAverageEntropy(EvaluationFunction):
    """Compute entropy of the average of generated images."""

    def __init__(self, n_neighbors: int = 4, **kwargs):
        """
        Args:
            n_neighbors: Number of nearest neighbors for entropy estimation
        """
        super().__init__(**kwargs)
        self.n_neighbors = n_neighbors

    def __call__(self, images: np.ndarray, savepath: str, model_name: str,
                 fom: Optional[np.ndarray] = None, cfg: Optional[OmegaConf] = None) -> float:
        """
        Compute entropy of the average image across all generated samples.

        Returns:
            Entropy value of the average image
        """
        # Compute the average image across all samples
        avg_image = np.mean(images, axis=0)

        # Flatten the average image for entropy computation
        avg_image_flat = avg_image.flatten()

        # Compute entropy using k-nearest neighbors approach
        entropy_value = im.entropy(
            avg_image_flat, approach="metric", k=self.n_neighbors)

        return float(entropy_value)


def evaluate_model(images: np.ndarray, savepath: Path, cfg: DictConfig, fom: np.ndarray | None = None, **kwargs) -> Dict[str, Any]:
    """
    Main evaluation function that runs all configured evaluation functions.

    Args:
        images: Generated images array
        savepath: Path to save results (string or Path object)
        cfg: Configuration object

    Returns:
        Dictionary of evaluation results
    """
    model_name = getattr(cfg, "name", None) or getattr(cfg.model, "name", None)
    print(model_name)
    results = dict()
    force_recompute = getattr(cfg, 'force_recompute', True)

    # Load existing stats to check if functions have already been run
    stats_file_path = savepath / 'stats.yaml'
    existing_stats = {}
    if stats_file_path.exists() and not force_recompute:
        try:
            with open(stats_file_path, 'r', encoding='utf-8') as f:
                existing_stats = yaml.safe_load(f) or {}
        except yaml.YAMLError:
            existing_stats = {}
    elif force_recompute:
        print("Force recompute enabled - will re-run all evaluation functions")

    if fom is None:
        # calcul FOM
        fompath = find_files(savepath, ["fom.npy"])[0]
        if fompath.exists():
            fom = np.load(fompath)
            if fom.shape[0] != images.shape[0]:
                print(
                    f"FOM file found but wrong shape for images.npy file, recomputing FOM, fom shape {fom.shape}, image shape {images.shape}")
                eval_cfg = get_evaluation_config(cfg)
                eval_fom = hydra.utils.instantiate(eval_cfg.fom)
                fom = eval_fom(images, savepath, model_name, cfg)
        else:
            eval_cfg = get_evaluation_config(cfg)
            eval_fom = hydra.utils.instantiate(eval_cfg.fom)
            fom = eval_fom(images, savepath, model_name, cfg)
    results["fom mean"] = fom.mean().item()
    results["fom std"] = fom.std().item()

    eval_cfg = get_evaluation_config(cfg)
    for eval_fn_cfg in eval_cfg.functions:
        eval_fn = hydra.utils.instantiate(eval_fn_cfg)

        if hasattr(eval_fn, '__name__'):
            fn_name = eval_fn.__name__
        else:
            fn_name = str(eval_fn_cfg.get(
                '_target_', 'unknown')).split('.')[-1]

        # Check if this function has already been computed
        if fn_name in existing_stats and not force_recompute:
            print(
                f"Skipping evaluation function: {fn_name} (already computed)")
            if "metric" in eval_fn_cfg:
                results[fn_name] = existing_stats[fn_name]
            continue
        elif fn_name in existing_stats and force_recompute:
            print(
                f"Re-running evaluation function: {fn_name} (force recompute enabled)")

        print(f"Running evaluation function: {fn_name}")

        out = eval_fn(images=images, fom=fom, savepath=savepath,
                      model_name=model_name, cfg=cfg)
        if "metric" in eval_fn_cfg:
            results[fn_name] = out

    update_stats_yaml(stats_file_path, results)

    return results


def get_evaluation_config(cfg):
    """Helper to get evaluation config whether it's nested or at root level."""
    if hasattr(cfg, 'evaluation'):
        return cfg.evaluation
    else:
        return cfg


@hydra.main(version_base=None, config_path='../config/', config_name='config')
def main(cfg):
    for ds in cfg.evaluation.datasets:
        path = ds.path
        n_samples = ds.n_samples
        path = Path(path)
        if path.name == "images.npy":
            path = path.parent.parent

        filenames_list = ['config.yaml', 'images.npy']
        files = find_files(path, filenames_list)
        if files is None:
            print(f"Skipping dataset {path} as required files not found")
            continue

        config_path = files[filenames_list.index('config.yaml')]
        images_path = files[filenames_list.index('images.npy')]

        with open(config_path, encoding="utf-8") as model_cfg:
            model_cfg = yaml.safe_load(model_cfg)
        model_cfg = load_wandb_config(model_cfg)
        images = np.load(images_path)

        evaluate_model(images, path, cfg)


if __name__ == "__main__":
    # main()
    # plot_closest_images_test()
    # plot_closest_images2_test()
    # plot_closest_images3_test()
    ts_dist_hist_test()

