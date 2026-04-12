#!/usr/bin/env python
"""
Standalone version of eval_single_file.py — no photo_gen dependency.
All referenced functions are copied inline.
"""

import argparse
import os
import warnings
import multiprocessing
from abc import ABC, abstractmethod
from pathlib import Path, PosixPath
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
import yaml

try:
    import meep as mp
    import meep.adjoint as mpa
    import autograd.numpy as npa
    MEEP_AVAILABLE = True
except ImportError:
    MEEP_AVAILABLE = False

try:
    import infomeasure as im
    INFOMEASURE_AVAILABLE = True
except ImportError:
    INFOMEASURE_AVAILABLE = False

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x


# ============================================================
# Utilities (from eval_utils.py)
# ============================================================

def tonumpy(array):
    if type(array) is torch.Tensor:
        return array.detach().cpu().numpy()
    else:
        return array


def update_stats_yaml(stats_path: Path, new_stats: dict) -> None:
    if os.path.exists(stats_path):
        with open(stats_path, 'r', encoding='utf-8') as f:
            try:
                stats = yaml.safe_load(f) or {}
            except yaml.YAMLError:
                stats = {}
    else:
        stats = {}
    stats = new_stats | stats
    with open(stats_path, 'w', encoding='utf-8') as f:
        yaml.dump(stats, f, default_flow_style=False)


class NullPath(PosixPath):
    """A Path subclass that always returns False for exists()."""
    def exists(self):
        return False


def find_files(rootdir: Path, filenames: List[str]) -> List[Optional[Path]]:
    found_files = dict()
    for dirpath, dirnames, files in os.walk(Path(rootdir)):
        for filename in filenames:
            if filename in files and filename not in found_files:
                filepath = (Path(dirpath) / filename)
                found_files[filename] = filepath
        if set(file.name for file in found_files.values()) == set(filenames):
            if len(filenames) == 1:
                found_files = [filepath]
            else:
                found_files = [found_files[key] for key in filenames]
            return found_files
    warnings.warn(f"Some files were not found {found_files}, {filenames}")
    return [NullPath()]


# ============================================================
# Meep FOM computation (from meep_compute_fom.py)
# ============================================================

def _require_meep():
    if not MEEP_AVAILABLE:
        raise ImportError(
            "meep is required for FOM computation. "
            "Install it or provide pre-computed fom via --fomfile."
        )


def double_with_mirror(image):
    mirrored_image = np.fliplr(image)
    doubled_image = np.concatenate((mirrored_image[:, :-1], image), axis=1)
    return doubled_image


def mapping(x, eta, beta, Nx, Ny, filter_radius, design_region_width,
            design_region_height, design_region_resolution, **kwargs):
    x = (npa.fliplr(x.reshape(Nx, Ny)) + x.reshape(Nx, Ny)) / 2
    filtered_field = mpa.conic_filter(
        x, filter_radius, design_region_width,
        design_region_height, design_region_resolution)
    projected_field = mpa.tanh_projection(filtered_field, beta, eta)
    return projected_field.flatten()


def get_sim(symmetry_enable=True):
    _require_meep()
    pml_size = 1.0
    dx = 0.02
    opt_size_x = 101 * dx
    opt_size_y = 181 * dx
    size_x = 2.6 + pml_size
    size_y = 4.5 + pml_size
    out_wg_dist = 1.25
    wg_width = 0.5
    mode_width = 3 * wg_width
    wg_index = 2.8
    bg_index = 1.44

    source_x = -size_x / 2 - 0.1
    source_y = 0
    source_yspan = mode_width
    source_z = 0
    center_wavelength = 1.550

    seed = 240
    np.random.seed(seed)
    mp.verbosity(0)
    Si = mp.Medium(index=wg_index)
    SiO2 = mp.Medium(index=bg_index)
    delta = dx
    resolution = 1 / delta
    waveguide_width = wg_width
    design_region_width = opt_size_x
    design_region_height = opt_size_y
    arm_separation = out_wg_dist

    minimum_length = 0.09
    eta_e = 0.75
    filter_radius = mpa.get_conic_radius_from_eta_e(minimum_length, eta_e)
    eta_i = 0.5
    design_region_resolution = int(resolution)
    frequencies = 1 / np.linspace(1.5, 1.6, 5)

    Nx = int(design_region_resolution * design_region_width)
    Ny = int(design_region_resolution * design_region_height)

    design_variables = mp.MaterialGrid(mp.Vector3(Nx, Ny), SiO2, Si)
    size = mp.Vector3(design_region_width, design_region_height)
    volume = mp.Volume(center=mp.Vector3(), size=size)
    design_region = mpa.DesignRegion(design_variables, volume=volume)

    Sx = 2 * pml_size + size_x
    Sy = 2 * pml_size + size_y
    cell_size = mp.Vector3(Sx, Sy)
    pml_layers = [mp.PML(pml_size)]

    fcen = 1 / center_wavelength
    width = 0.2
    fwidth = width * fcen
    source_center = [source_x, source_y, source_z]
    source_size = mp.Vector3(0, source_yspan, 0)
    kpoint = mp.Vector3(1, 0, 0)

    src = mp.GaussianSource(frequency=fcen, fwidth=fwidth)
    source = [mp.EigenModeSource(
        src, eig_band=1, direction=mp.NO_DIRECTION,
        eig_kpoint=kpoint, size=source_size, center=source_center,
        eig_parity=mp.EVEN_Z + mp.ODD_Y)]
    mon_pt = mp.Vector3(*source_center)

    sim_args = {
        'Nx': Nx, 'Ny': Ny, 'Sx': Sx, 'Sy': Sy,
        'eta_i': eta_i, 'filter_radius': filter_radius,
        'design_region_width': design_region_width,
        'design_region_height': design_region_height,
        'design_region_resolution': design_region_resolution,
        'fcen': fcen, 'frequencies': frequencies,
        'source_x': source_x, 'source_center': source_center,
        'size_x': size_x, 'arm_separation': arm_separation,
        'mon_pt': mon_pt, 'design_region_size': size,
        'waveguide_width': waveguide_width,
    }

    geometry = [
        mp.Block(center=mp.Vector3(x=-Sx / 4), material=Si,
                 size=mp.Vector3(Sx / 2 + 1, waveguide_width, 0)),
        mp.Block(center=mp.Vector3(x=Sx / 4, y=arm_separation), material=Si,
                 size=mp.Vector3(Sx / 2 + 1, waveguide_width, 0)),
        mp.Block(center=mp.Vector3(x=Sx / 4, y=-arm_separation), material=Si,
                 size=mp.Vector3(Sx / 2 + 1, waveguide_width, 0)),
        mp.Block(center=design_region.center, size=design_region.size,
                 material=design_variables)
    ]

    symmetries = ([mp.Mirror(direction=mp.Y, phase=-1)]
                  if symmetry_enable else None)
    sim = mp.Simulation(
        cell_size=cell_size, boundary_layers=pml_layers,
        geometry=geometry, sources=source, symmetries=symmetries,
        default_material=SiO2, resolution=resolution)
    return sim, design_region, sim_args


def compute_FOM(image, symmetry_enable=True, debug=False):
    _require_meep()
    assert image.shape == (101, 91), f"{image.shape} != (101,91)"
    assert not ((image > 1.) + (image < 0.)).any()

    sim, design_region, sim_args = get_sim()

    Sx = sim_args['Sx']
    Sy = sim_args['Sy']
    fcen = sim_args['fcen']
    waveguide_width = sim_args['waveguide_width']
    source_x = sim_args['source_x']
    source_center = sim_args['source_center']
    size_x = sim_args['size_x']
    arm_separation = sim_args['arm_separation']
    mon_pt = sim_args['mon_pt']

    idx_map = double_with_mirror(image)
    index_map = mapping(idx_map, 0.5, 256, **sim_args)
    design_region.update_design_parameters(index_map)

    monsize = mp.Vector3(y=3 * waveguide_width)

    sim.run(until_after_sources=100)

    def get_eigenmode_coeffs(sim, fluxregion, mon_pt):
        sim.reset_meep()
        flux = sim.add_flux(fcen, 0, 1, fluxregion)
        mon_pt = mp.Vector3(*source_center)
        sim.run(until_after_sources=mp.stop_when_fields_decayed(
            50, mp.Ez, mon_pt, 1e-9))
        res = sim.get_eigenmode_coefficients(
            flux, [1], eig_parity=mp.EVEN_Z + mp.ODD_Y)
        coeffs = res.alpha
        return coeffs

    abs_src_coeff = 57.97435797757672

    topmoncenter = mp.Vector3(size_x / 2, arm_separation, 0)
    topfluxregion = mp.FluxRegion(topmoncenter, monsize)
    top_coeffs = get_eigenmode_coeffs(sim, topfluxregion, mon_pt)

    fom1 = np.abs(top_coeffs[0, 0, 0]) ** 2 / abs_src_coeff
    if symmetry_enable:
        fom1 = fom1 / 2
    return fom1


def compute_FOM_parallele(images, debug=False):
    """
    Compute FOM for a batch of images using meep in parallel.

    Input images are (101, 91) and are doubled to shape (101, 181) and
    normalized to have values in the range [0, 1] then meep simulates the FOM.

    inputs:
        images: np.array (batch_size, 101, 91)
    outputs:
        foms: np.array (batch_size)
    """
    _require_meep()
    if images.ndim == 2:
        return compute_FOM(images)
    if images.ndim > 3:
        images = images.squeeze()
        assert images.ndim == 3, f"too many dimensions {images.ndim} > 3"

    images = [images[i] for i in range(images.shape[0])]
    with multiprocessing.Pool() as pool:
        results = list(tqdm(pool.imap(compute_FOM, images)))
    foms = np.array(results)
    return foms


# ============================================================
# Evaluation functions (from evaluation.py)
# ============================================================

class EvaluationFunction(ABC):
    """Abstract base class for all evaluation functions."""

    def __init__(self, **kwargs):
        self.config = kwargs

    @abstractmethod
    def __call__(self, images: np.ndarray, savepath: str, model_name: str,
                 fom: Optional[np.ndarray] = None, **kwargs) -> Any:
        pass

    @property
    def name(self) -> str:
        return self.__class__.__name__


def closest_image(img: np.ndarray, train_set: np.ndarray) -> tuple:
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


class CompareToTrainClosestImage(EvaluationFunction):
    """Plot generated images alongside their closest training set matches."""

    def __init__(self, train_set_path: os.PathLike, **kwargs):
        train_set_path = os.path.expanduser(train_set_path)
        self.train_set = np.load(train_set_path)

    def __call__(self, images: np.ndarray, savepath: str, model_name: str,
                 fom: Optional[np.ndarray] = None, **kwargs):
        images = images.squeeze()
        train_set = self.train_set.squeeze()
        distances = []
        closest_images = []
        for img in images:
            ts_image, min_dist = closest_image(img, train_set)
            distances.append(min_dist)
            closest_images.append(ts_image)
        distances = np.array(distances)
        closest_images_arr = np.array(closest_images)

        n_samples = 4
        indices = np.flip(np.argsort(distances))[:n_samples]
        fig, axes = plt.subplots(2, n_samples, figsize=(15, 9))
        for j, idx in enumerate(indices):
            gen_img = images[idx]
            train_img = closest_images_arr[idx]
            gen_img = (gen_img - gen_img.min()) / (gen_img.max() - gen_img.min() + 1e-8)
            train_img = (train_img - train_img.min()) / (train_img.max() - train_img.min() + 1e-8)
            axes[0, j].imshow(gen_img, vmin=0, vmax=1)
            axes[0, j].set_title(f'FOM: {fom[idx]:.3f}, Dist: {distances[idx]:.4f}')
            axes[0, j].axis('off')
            axes[1, j].imshow(train_img, vmin=0, vmax=1)
            axes[1, j].axis('off')
        plt.subplots_adjust(top=0.92)
        save_file = os.path.join(savepath, "generated_vs_closest_train.png")
        plt.savefig(save_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()


class BinarizationLoss(EvaluationFunction):
    """Measure the binarization of the generated images."""

    def __call__(self, images: np.ndarray, savepath: str, model_name: str,
                 fom: Optional[np.ndarray] = None, **kwargs) -> float:
        images = images.reshape(images.shape[0], -1)
        binarization_metric = np.minimum(np.abs(images - 0), np.abs(images - 1))
        return float(binarization_metric.mean())


class VisualizeGeneratedSamples(EvaluationFunction):
    """Create a grid visualization of generated samples."""

    def __init__(self, n_samples: int = 16, **kwargs):
        super().__init__(**kwargs)
        self.n_samples = n_samples

    def __call__(self, images: np.ndarray, savepath: str, model_name: str,
                 fom: Optional[np.ndarray] = None, **kwargs) -> str:
        n_samples = min(self.n_samples, images.shape[0])
        assert n_samples >= 4
        images = images.squeeze()
        grid_size = int(np.ceil(np.sqrt(n_samples)))
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
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
                    ax.axis('off')
                    continue
                img = images[sample_idx]
                img = (img - img.min()) / (img.max() - img.min() + 1e-8)
                ax.imshow(img, vmin=0, vmax=1)
                ax.set_title(f'samples from {model_name}', fontsize=10)
                ax.axis('off')
                sample_idx += 1
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        save_file = os.path.join(savepath, f"{model_name.lower()}_samples_grid.png")
        plt.savefig(save_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Sample grid visualization saved: {save_file}")
        return save_file


class PlotFomHistogram(EvaluationFunction):
    """Plot histogram of Figure of Merit values."""

    def __call__(self, images: np.ndarray, savepath: str, model_name: str,
                 fom: Optional[np.ndarray] = None, **kwargs) -> str:
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
    """Compute Figure of Merit for generated images using meep."""

    def __call__(self, images: np.ndarray, savepath: str, model_name: str,
                 fom: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        computed_fom = compute_FOM_parallele(images)
        np.save(os.path.join(savepath, "fom.npy"), computed_fom)
        return computed_fom


def pca_dim_reduction_entropy(images, dim, n_neighbors):
    from sklearn.decomposition import PCA
    x = images.reshape(images.shape[0], -1)
    pca = PCA(n_components=dim)
    x_pca = pca.fit_transform(x)
    h = im.entropy(x_pca, approach="metric", k=n_neighbors)
    return float(h)


class PCAProjPerDimEntropy(EvaluationFunction):
    def __init__(self, n_neighbors: int = 4, dim: int = 50, **kwargs):
        self.n_neighbors = n_neighbors
        self.dim = dim

    def __call__(self, images: np.ndarray, **kwargs):
        return pca_dim_reduction_entropy(images, self.dim, self.n_neighbors)


class Entropy(EvaluationFunction):
    def __init__(self, n_neighbors: int = 4, **kwargs):
        self.n_neighbors = n_neighbors

    def __call__(self, images: np.ndarray, savepath: str, model_name: str,
                 fom: Optional[np.ndarray] = None, **kwargs) -> float:
        from infomeasure import entropy
        images = images.reshape(images.shape[0], -1)
        h = entropy(images, approach="metric", k=self.n_neighbors)
        return float(h)


class NNDistanceTrainSet(EvaluationFunction):
    def __init__(self, train_set_path: os.PathLike, **kwargs):
        train_set_path = os.path.expanduser(train_set_path)
        self.train_set = np.load(train_set_path)

    def __call__(self, images: np.ndarray, savepath: str, model_name: str,
                 fom: Optional[np.ndarray] = None, **kwargs) -> dict:
        gen_ds = torch.tensor(images) if isinstance(images, np.ndarray) else images
        train_ds = torch.tensor(self.train_set) if isinstance(self.train_set, np.ndarray) else self.train_set

        distances = []
        for x in gen_ds:
            min_dist = float('inf')
            for y in train_ds:
                dist = torch.norm(x - y)
                if dist < min_dist:
                    min_dist = dist
            distances.append(min_dist)

        distances = tonumpy(torch.stack(distances))
        plt.hist(distances, bins=100, density=True, label=model_name, alpha=0.5)
        plt.title('Nearest training set neighbor distances')
        plt.savefig(os.path.join(savepath, 'nn_distance_histogram.png'))
        plt.legend()
        plt.close()
        return {"mean": distances.mean().item(), "std": distances.std().item()}


class PairwiseDistanceEntropy(EvaluationFunction):
    """Compute entropy of pairwise distances between generated images."""

    def __init__(self, n_neighbors: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.n_neighbors = n_neighbors

    def __call__(self, images: np.ndarray, savepath: str, model_name: str,
                 fom: Optional[np.ndarray] = None, **kwargs) -> float:
        from scipy.spatial.distance import pdist
        images_flat = images.reshape(images.shape[0], -1)
        pairwise_distances = pdist(images_flat, metric='euclidean')
        distances_array = pairwise_distances.reshape(-1, 1)
        entropy_value = im.entropy(distances_array, approach="metric", k=self.n_neighbors)
        return float(entropy_value)


class ImageAverageEntropy(EvaluationFunction):
    """Compute entropy of the average of generated images."""

    def __init__(self, n_neighbors: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.n_neighbors = n_neighbors

    def __call__(self, images: np.ndarray, savepath: str, model_name: str,
                 fom: Optional[np.ndarray] = None, **kwargs) -> float:
        avg_image = np.mean(images, axis=0)
        avg_image_flat = avg_image.flatten()
        entropy_value = im.entropy(avg_image_flat, approach="metric", k=self.n_neighbors)
        return float(entropy_value)


# ============================================================
# Main evaluation logic (replaces hydra-based evaluate_model)
# ============================================================

DEFAULT_TRAIN_SET_PATH = "~/scratch/nanophoto/topoptim/fulloptim/images.npy"

# Evaluation functions to run, with (class, kwargs, is_metric) tuples.
# Mirrors the "complete" config from photo_gen/config/evaluation/functions/complete.yaml
def get_default_eval_functions(train_set_path: str = DEFAULT_TRAIN_SET_PATH):
    return [
        (VisualizeGeneratedSamples, {"n_samples": 16}, False),
        (PlotFomHistogram, {}, False),
        (Entropy, {"n_neighbors": 4}, True),
        (PCAProjPerDimEntropy, {"dim": 50, "n_neighbors": 4}, True),
        (NNDistanceTrainSet, {"train_set_path": train_set_path}, True),
        (ImageAverageEntropy, {"n_neighbors": 4}, True),
        (PairwiseDistanceEntropy, {"n_neighbors": 4}, True),
        (BinarizationLoss, {}, True),
        (CompareToTrainClosestImage, {"train_set_path": train_set_path}, False),
    ]


def evaluate_model(images: np.ndarray, savepath: Path,
                   fom: np.ndarray | None = None,
                   model_name: str = "model",
                   force_recompute: bool = False,
                   train_set_path: str = DEFAULT_TRAIN_SET_PATH,
                   ) -> Dict[str, Any]:
    """
    Main evaluation function that runs all evaluation functions.

    Args:
        images: Generated images array
        savepath: Path to save results
        fom: Pre-computed figure of merit (optional)
        model_name: Name of the model
        force_recompute: If True, re-run all evaluation functions
        train_set_path: Path to training set images for distance-based metrics
    """
    savepath = Path(savepath)
    results = dict()

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

    # Compute or load FOM
    if fom is None:
        fompath = find_files(savepath, ["fom.npy"])[0]
        if fompath.exists():
            fom = np.load(fompath)
            if fom.shape[0] != images.shape[0]:
                print(f"FOM shape mismatch (fom: {fom.shape}, images: {images.shape}), recomputing...")
                fom_fn = FOM()
                fom = fom_fn(images, str(savepath), model_name)
        else:
            fom_fn = FOM()
            fom = fom_fn(images, str(savepath), model_name)

    results["fom mean"] = fom.mean().item()
    results["fom std"] = fom.std().item()

    eval_functions = get_default_eval_functions(train_set_path)

    for eval_cls, kwargs, is_metric in eval_functions:
        fn_name = eval_cls.__name__

        if fn_name in existing_stats and not force_recompute:
            print(f"Skipping evaluation function: {fn_name} (already computed)")
            if is_metric:
                results[fn_name] = existing_stats[fn_name]
            continue

        print(f"Running evaluation function: {fn_name}")
        try:
            eval_fn = eval_cls(**kwargs)
            out = eval_fn(images=images, fom=fom, savepath=str(savepath),
                          model_name=model_name)
            if is_metric:
                results[fn_name] = out
        except Exception as e:
            print(f"  Error in {fn_name}: {e}")

    update_stats_yaml(stats_file_path, results)
    return results


def eval_single_file(images: np.ndarray, savepath: Path = Path("."),
                     fom: np.ndarray | None = None, **kwargs) -> Dict[str, Any]:
    """
    Evaluate a single set of generated images.

    Args:
        images: Generated images array
        savepath: Directory to save results
        fom: Pre-computed figure of merit (optional)
        **kwargs: Extra arguments forwarded to evaluate_model
    """
    return evaluate_model(images, savepath, fom=fom, **kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone evaluation of generated images")
    parser.add_argument("--dir", type=str, default=".")
    parser.add_argument("-i", "--imagefile", type=str, default=None)
    parser.add_argument("-f", "--fomfile", type=str, default=None)
    parser.add_argument("--train-set", type=str, default=DEFAULT_TRAIN_SET_PATH,
                        help="Path to training set images.npy")
    parser.add_argument("--force-recompute", action="store_true")
    args = parser.parse_args()

    path = Path(args.dir)

    if args.imagefile is not None:
        images = np.load(args.imagefile)
    else:
        images = np.load(path / "images.npy")

    fom = None
    if args.fomfile is not None:
        fompath = Path(args.fomfile)
    else:
        fompath = path / "fom.npy"
    if fompath.exists():
        fom = np.load(fompath)

    eval_single_file(
        images=images,
        savepath=path,
        fom=fom,
        train_set_path=args.train_set,
        force_recompute=args.force_recompute,
    )
