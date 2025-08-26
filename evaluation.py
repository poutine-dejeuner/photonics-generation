import os
import json

import numpy as np
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
import hydra

from icecream import ic


def visualize_generated_samples(images: np.ndarray, savepath: str, model_name: str, n_samples: int = 16):
    """
    Create a grid visualization of generated samples and save it.

    Args:
        images: Generated images array of shape (N, C, H, W) or (N, H, W)
        savepath: Directory to save the visualization
        model_name: Name of the model for the title
        n_samples: Number of samples to display in grid (default 16)
    """
    # Ensure we don't exceed available samples
    n_samples = min(n_samples, images.shape[0])

    # Calculate grid dimensions (prefer square grids)
    grid_size = int(np.ceil(np.sqrt(n_samples)))

    # Create figure
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    fig.suptitle(f'{model_name} - Generated Samples',
                 fontsize=16, fontweight='bold')

    # Handle case where we have only one subplot
    if grid_size == 1:
        axes = [[axes]]
    elif grid_size == 2:
        axes = axes.reshape(2, 2)

    sample_idx = 0
    for i in range(grid_size):
        for j in range(grid_size):
            ax = axes[i][j] if grid_size > 1 else axes[i]

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
                img_display = (img - img.min()) / \
                    (img.max() - img.min() + 1e-8)

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
    save_file = os.path.join(
        savepath, f"{model_name.lower()}_samples_grid.png")
    plt.savefig(save_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Sample grid visualization saved: {save_file}")




def plot_fom_hist(fom, model_name, savedir):
    plt.figure(figsize=(10, 6))
    plt.hist(fom, bins=100, alpha=0.7, edgecolor='black')
    plt.title(f"FOM Histogram - {model_name}")
    plt.xlabel("Figure of Merit")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(savedir, "fom_histogram.png"),
                dpi=300, bbox_inches='tight')
    plt.close()


def eval_static(images, fom, savedir, model_name, cfg):
    from nanophoto.evaluation.evalgen import eval_metrics

    visualize_generated_samples(images, savedir, model_name, n_samples=16)

    plot_fom_hist(fom, model_name, savedir)

    dataset_cfg = OmegaConf.create([{"name": f"{model_name}_{os.environ.get('SLURM_JOB_ID', 'local')}",
                                   "path": savedir}])
    eval_metrics(dataset_cfg, os.path.dirname(cfg.data_path))

    results = {
        'model_type': cfg.model.get('_target_', 'unknown'),
        'train_set_size': cfg.train_set_size,
        'debug': cfg.debug,
        'experiment_path': savedir,
        'config': OmegaConf.to_container(cfg, resolve=True)
    }

    with open(os.path.join(savedir, 'experiment_results.json'), 'w') as f:
        json.dump(results, f, indent=2)


def evaluation(images, fom, model_name, cfg):
    savepath = cfg.savepath
    
    results = dict()
    ic(cfg.evaluation.functions)
    for eval_fn_cfg in cfg.evaluation.functions:
        eval_fn = hydra.utils.instantiate(eval_fn_cfg)
        try:
            if hasattr(eval_fn, '__name__'):
                fn_name = eval_fn.__name__
            else:
                fn_name = str(eval_fn_cfg.get('_target_', 'unknown'))
            
            print(f"Running evaluation function: {fn_name}")
            
            # Handle different function signatures
            if 'eval_static' in fn_name:
                out = eval_fn(images, fom, savepath, model_name, cfg)
            elif any(x in fn_name for x in ['compute_entropy', 'nn_distance']):
                # These functions from nanophoto.evaluation might have different signatures
                out = eval_fn(images)
            else:
                # Default signature for visualization functions
                out = eval_fn(images, fom, savepath, model_name)
            
            results[fn_name]=out
            
        except Exception as e:
            print(f"Error running evaluation function {eval_fn_cfg.get('_target_', 'unknown')}: {e}")
    
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
            'evaluation': {
                'functions': [
                    {'_target_': 'evaluation.visualize_generated_samples',
                    n_samples: 16},
                    {'_target_': 'nanophoto.evaluation.gen_models_comparison.compute_entropy'},
                    {'_target_': 'evaluation.plot_fom_hist'},
                    {'_target_': 'nanophoto.evaluation.gen_models_comparison.nn_distance_to_train_ds'}
                ]
            }
        })
        
        print("Config evaluation functions:")
        print(cfg.evaluation.functions)
        print(f"Number of evaluation functions: {len(cfg.evaluation.functions)}")
        
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


# def inception_score(images, batch_size=32, splits=10):
#     """
#     Compute Inception Score (IS) for generated images.
    
#     Args:
#         images: Generated images array of shape (N, C, H, W) or (N, H, W)
#         batch_size: Batch size for processing (default 32)
#         splits: Number of splits for computing IS (default 10)
    
#     Returns:
#         tuple: (mean_IS, std_IS)
#     """

#     import torch
#     import torch.nn.functional as F
#     from torchvision.models import inception_v3
#     from torch.utils.data import DataLoader, TensorDataset
    
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
#     # Load pre-trained Inception v3 model
#     inception_model = inception_v3(pretrained=True, transform_input=False).to(device)
#     inception_model.eval()
    
#     # Convert numpy array to torch tensor
#     if isinstance(images, np.ndarray):
#         images_tensor = torch.from_numpy(images).float()
#     else:
#         images_tensor = images
    
#     # Handle different image formats
#     if len(images_tensor.shape) == 3:  # (N, H, W) -> (N, 1, H, W)
#         images_tensor = images_tensor.unsqueeze(1)
    
#     # Ensure we have 3 channels for Inception (RGB)
#     if images_tensor.shape[1] == 1:  # Grayscale -> RGB
#         images_tensor = images_tensor.repeat(1, 3, 1, 1)
    
#     # Resize to 299x299 (Inception input size)
#     if images_tensor.shape[-1] != 299 or images_tensor.shape[-2] != 299:
#         images_tensor = F.interpolate(images_tensor, size=(299, 299), mode='bilinear', align_corners=False)
    
#     # Normalize to [-1, 1] range expected by Inception
#     images_tensor = (images_tensor - images_tensor.min()) / (images_tensor.max() - images_tensor.min())
#     images_tensor = images_tensor * 2.0 - 1.0
    
#     # Create DataLoader
#     dataset = TensorDataset(images_tensor)
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
#     # Get predictions
#     all_preds = []
#     with torch.no_grad():
#         for batch in dataloader:
#             batch_images = batch[0].to(device)
#             preds = inception_model(batch_images)
#             preds = F.softmax(preds, dim=1)
#             all_preds.append(preds.cpu())
    
#     all_preds = torch.cat(all_preds, dim=0)
    
#     # Compute IS
#     scores = []
#     n_samples = all_preds.shape[0]
#     split_size = n_samples // splits
    
#     for i in range(splits):
#         start_idx = i * split_size
#         end_idx = (i + 1) * split_size if i < splits - 1 else n_samples
        
#         split_preds = all_preds[start_idx:end_idx]
        
#         # Compute KL divergence
#         marginal = torch.mean(split_preds, dim=0, keepdim=True)
#         kl_div = split_preds * (torch.log(split_preds + 1e-8) - torch.log(marginal + 1e-8))
#         kl_div = torch.sum(kl_div, dim=1)
#         is_score = torch.exp(torch.mean(kl_div))
#         scores.append(is_score.item())
    
#     mean_is = np.mean(scores)
#     std_is = np.std(scores)
    
#     return mean_is, std_is


