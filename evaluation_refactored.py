import os
import json
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union

import numpy as np
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
import hydra

from icecream import ic


class EvaluationFunction(ABC):
    """Abstract base class for all evaluation functions."""
    
    def __init__(self, **kwargs):
        """Initialize the evaluation function with any configuration parameters."""
        self.config = kwargs
    
    @abstractmethod
    def __call__(self, images: np.ndarray, fom: Optional[np.ndarray] = None, 
                 savepath: Optional[str] = None, model_name: Optional[str] = None, 
                 cfg: Optional[OmegaConf] = None) -> Any:
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


class VisualizeGeneratedSamples(EvaluationFunction):
    """Create a grid visualization of generated samples and save it."""
    
    def __init__(self, n_samples: int = 16, **kwargs):
        super().__init__(**kwargs)
        self.n_samples = n_samples
    
    def __call__(self, images: np.ndarray, fom: Optional[np.ndarray] = None, 
                 savepath: Optional[str] = None, model_name: Optional[str] = None, 
                 cfg: Optional[OmegaConf] = None) -> str:
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
        if savepath is None or model_name is None:
            raise ValueError("savepath and model_name are required")
            
        # Ensure we don't exceed available samples
        n_samples = min(self.n_samples, images.shape[0])

        # Calculate grid dimensions (prefer square grids)
        grid_size = int(np.ceil(np.sqrt(n_samples)))

        # Create figure
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
        fig.suptitle(f'{model_name} - Generated Samples',
                     fontsize=16, fontweight='bold')

        # Handle case where we have only one subplot
        if grid_size == 1:
            axes = [[axes]]
        elif grid_size == 2 and len(axes.shape) == 1:
            axes = axes.reshape(2, 2)

        sample_idx = 0
        for i in range(grid_size):
            for j in range(grid_size):
                if grid_size == 1:
                    ax = axes[0][0]
                elif grid_size > 1:
                    ax = axes[i][j]
                else:
                    ax = axes[i]

                if sample_idx < n_samples:
                    # Get the image
                    img = images[sample_idx]

                    # Handle different image formats
                    if len(img.shape) == 3:  # (C, H, W)
                        if img.shape[0] == 1:  # Single channel
                            img = img.squeeze(0)
                        else:  # Multi-channel, transpose to (H, W, C)
                            img = img.transpose(1, 2, 0)
                    elif len(img.shape) == 2:  # (H, W)
                        pass  # Already in correct format

                    # Normalize to [0, 1] for display
                    img_display = (img - img.min()) / (img.max() - img.min() + 1e-8)

                    # Display image
                    if len(img_display.shape) == 2:  # Grayscale
                        ax.imshow(img_display, cmap='gray', vmin=0, vmax=1)
                    else:  # Color
                        ax.imshow(np.clip(img_display, 0, 1))

                    ax.set_title(f'Sample {sample_idx + 1}', fontsize=10)
                    sample_idx += 1
                else:
                    # Hide empty subplots
                    ax.set_visible(False)

                ax.axis('off')

        plt.tight_layout()
        plt.subplots_adjust(top=0.93)  # Make room for suptitle

        # Save the visualization
        save_file = os.path.join(savepath, f"{model_name.lower()}_samples_grid.png")
        plt.savefig(save_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        print(f"Sample grid visualization saved: {save_file}")
        return save_file


class PlotFomHistogram(EvaluationFunction):
    """Plot histogram of Figure of Merit values."""

    def __call__(self, images: np.ndarray, fom: Optional[np.ndarray] = None, 
                 savepath: Optional[str] = None, model_name: Optional[str] = None, 
                 cfg: Optional[OmegaConf] = None) -> str:
        """Plot FOM histogram and save it."""
        if fom is None or savepath is None or model_name is None:
            raise ValueError("fom, savepath, and model_name are required")

        plt.figure(figsize=(10, 6))
        plt.hist(fom, bins=100, alpha=0.7, edgecolor='black')
        plt.title(f"FOM Histogram - {model_name}")
        plt.xlabel("Figure of Merit")
        plt.ylabel("Frequency")
        plt.grid(True, alpha=0.3)

        save_file = os.path.join(savepath, "fom_histogram.png")
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
        plt.close()

        return save_file


class ComputeFom(EvaluationFunction):
    """Compute Figure of Merit for generated images."""
    
    def __call__(self, images: np.ndarray, fom: Optional[np.ndarray] = None, 
                 savepath: Optional[str] = None, model_name: Optional[str] = None, 
                 cfg: Optional[OmegaConf] = None) -> tuple[float, float]:

        from nanophoto.meep_compute_fom import compute_FOM_parallele

        if cfg.debug:
            computed_fom = np.random.rand(images.shape[0])
        else:
            computed_fom = compute_FOM_parallele(images)

        np.save(os.path.join(savepath, "fom.npy"), computed_fom)
        return computed_fom.mean(), computed_fom.std()
    
class ComputeEntropy(EvaluationFunction):
    def __init__(self, n_neighbors: int=4, **kwargs):

        self.n_neighbors = n_neighbors

    def __call__(self, images: np.ndarray, fom: Optional[np.ndarray] = None, 
                 savepath: Optional[str] = None, model_name: Optional[str] = None, 
                 cfg: Optional[OmegaConf] = None) -> tuple[float, float]:

        from infomeasure import entropy

        images = np.reshape(images.shape[0], -1)
        h = entropy(images, approach="metric", k=self.n_neighbors)
        return h
    
class NNDistanceTrainSet(EvaluationFunction):
    def __init__(self, train_set_path:os.PathLike, **kwargs):
        self.train_set = np.load(train_set_path)

    def __call__(self, gen_set):

        from nanophoto.evaluation.gen_models_comparison import nn_distance_to_train_ds

        distances = nn_distance_to_train_ds(gen_set, self.train_set)
        
        return (distances.mean(), distances.std())
        

# Legacy function wrappers for backward compatibility
def visualize_generated_samples(images: np.ndarray, savepath: str, model_name: str,     n_samples: int = 16) -> str:
    """Legacy wrapper for VisualizeGeneratedSamples."""
    evaluator = VisualizeGeneratedSamples(n_samples=n_samples)
    return evaluator(images, savepath=savepath, model_name=model_name)


def plot_fom_hist(fom: np.ndarray, model_name: str, savedir: str) -> str:
    """Legacy wrapper for PlotFomHistogram."""
    evaluator = PlotFomHistogram()
    return evaluator(None, fom=fom, savepath=savedir, model_name=model_name)


def compute_fom(images: np.ndarray, savepath: str, model_name: str, cfg: OmegaConf) -> tuple[float, float]:
    """Legacy wrapper for ComputeFom."""
    evaluator = ComputeFom()
    return evaluator(images, savepath=savepath, model_name=model_name, cfg=cfg)


def evaluation(images: np.ndarray, fom: np.ndarray, model_name: str, cfg: OmegaConf) -> Dict[str, Any]:
    """
    Main evaluation function that runs all configured evaluation functions.
    
    Args:
        images: Generated images array
        fom: Figure of merit values
        model_name: Name of the model
        cfg: Configuration object
        
    Returns:
        Dictionary of evaluation results
    """
    savepath = cfg.savepath
    results = dict()

    for eval_fn_cfg in cfg.evaluation:
        ic(eval_fn_cfg)
        eval_fn = hydra.utils.instantiate(eval_fn_cfg)

        if hasattr(eval_fn, '__name__'):
            fn_name = eval_fn.__name__
        else:
            fn_name = str(eval_fn_cfg.get('_target_', 'unknown'))
        print(f"Running evaluation function: {fn_name}")
        out = eval_fn(images, fom, savepath, model_name)
        results[fn_name] = out

    return results


def test_evaluation():
    """
    Test function to debug evaluation with properly initiated variables.
    """
    from omegaconf import OmegaConf
    import tempfile

    # Create mock data
    images = np.random.rand(8, 1, 64, 64)  # (N, C, H, W) format
    fom = np.random.rand(8)
    model_name = "test_model"

    # Create temporary directory for saving
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a test configuration with evaluation functions
        cfg = OmegaConf.create({
            'savepath': temp_dir,
            'model': {'_target_': 'test.model'},
            'train_set_size': 1000,
            'debug': True,
            'data_path': '/tmp/test_data',
            'evaluation': [
                {'_target_': 'evaluation_refactored.VisualizeGeneratedSamples', 'n_samples': 16},
                {'_target_': 'evaluation_refactored.PlotFomHistogram'},
                {'_target_': 'evaluation_refactored.ComputeFom'},
                {'_target_': 'evaluation_refactored.ComputeFom'},
            ]
        })

        print("Config evaluation functions:")
        print(cfg.evaluation)
        print(f"Number of evaluation functions: {len(cfg.evaluation)}")

        # Test the evaluation function
        try:
            results = evaluation(images, fom, model_name, cfg)
            print(f"Evaluation completed successfully. Results keys: {list(results.keys())}")
            return results
        except Exception as e:
            print(f"Error during evaluation: {e}")
            import traceback
            traceback.print_exc()
            return None


if __name__ == "__main__":
    test_evaluation()
