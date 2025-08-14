# Generative Model Comparison Framework

This framework allows you to train and compare different generative models (wGAN, VAE, Standard GAN) for nanophotonics design generation.

## Files Overview

### Main Training Scripts
- `train_comparison.py` - Main training script for model comparison
- `train4.py` - Original diffusion model training script

### Model Implementations
- `models/wgan.py` - Wasserstein GAN with Gradient Penalty
- `models/vae.py` - Variational Autoencoder (β-VAE)
- `models/standard_gan.py` - Standard GAN with adversarial loss
- `models/unet.py` - Original U-Net for diffusion models

### Configuration Files
- `config/comparison_config.yaml` - Main config for comparison experiments
- `config/model/` - Model-specific configurations
  - `wgan.yaml` - WGAN configuration
  - `vae.yaml` - VAE configuration  
  - `standard_gan.yaml` - Standard GAN configuration
- `config/train/` - Training configurations for each model
- `config/inference/` - Inference configurations for each model

### Analysis Tools
- `analyze_comparison_results.py` - Results analysis and visualization
- `run_comparison_experiments.sh` - Batch experiment runner

## Usage

### 1. Single Model Training

Train a specific model (e.g., WGAN):
```bash
python train_comparison.py model=wgan train=wgan inference=wgan
```

Train VAE:
```bash
python train_comparison.py model=vae train=vae inference=vae
```

Train Standard GAN:
```bash
python train_comparison.py model=standard_gan train=standard_gan inference=standard_gan
```

### 2. Batch Experiments

Run all models in sequence:
```bash
./run_comparison_experiments.sh
```

For debug mode (faster, smaller datasets):
```bash
./run_comparison_experiments.sh debug
```

### 3. Results Analysis

After running experiments, analyze and compare results:
```bash
python analyze_comparison_results.py --results_dir ~/scratch/nanophoto/comparison/experiments --output_dir ./analysis_results
```

## Configuration

### Model Parameters

Each model has its own configuration file in `config/model/`:

**WGAN (`wgan.yaml`)**:
- `latent_dim`: Latent space dimension (default: 100)
- `n_critic`: Critic updates per generator update (default: 5)
- `lambda_gp`: Gradient penalty coefficient (default: 10)

**VAE (`vae.yaml`)**:
- `latent_dim`: Latent space dimension (default: 128)
- `beta`: KL divergence weight for β-VAE (default: 1.0)

**Standard GAN (`standard_gan.yaml`)**:
- `latent_dim`: Latent space dimension (default: 100)
- `lr_g`, `lr_d`: Learning rates for generator and discriminator
- `label_smoothing`: Label smoothing for discriminator (default: 0.1)

### Training Parameters

Common training parameters can be adjusted in the model configs:
- `n_epochs`: Number of training epochs
- `batch_size`: Training batch size
- `lr`: Learning rate
- `seed`: Random seed (-1 for random)

### Environment Variables

The scripts expect these environment variables:
- `SCRATCH`: Scratch directory for saving results
- `SLURM_JOB_ID`: Job ID for cluster runs (optional)

## Model Architectures

### WGAN-GP
- Generator: Transposed convolutions with batch normalization
- Critic: Convolutions with instance normalization
- Gradient penalty for training stability

### VAE
- Encoder: Convolutional layers → fully connected → μ, σ
- Decoder: Fully connected → transposed convolutions
- β-VAE variant with adjustable KL weight

### Standard GAN
- Generator: Similar to WGAN but with batch normalization
- Discriminator: Convolutional layers with batch normalization
- Binary cross-entropy loss with label smoothing

## Output Structure

Results are saved in the following structure:
```
~/scratch/nanophoto/comparison/experiments/
├── <job_id>_<timestamp>/
│   ├── wgan/
│   │   ├── checkpoint.pt
│   │   ├── images/
│   │   ├── results_summary.json
│   │   └── fom_histogram.png
│   ├── vae/
│   │   └── ...
│   └── standard_gan/
│       └── ...
```

## Results Analysis

The analysis script generates:
- **Comparison plots**: Bar charts, box plots, range comparisons
- **Performance report**: Ranking, detailed statistics, analysis
- **Summary statistics table**: Mean, std, min, max FOM values

## Extending the Framework

To add a new model:

1. Create model implementation in `models/your_model.py` with `train()` and `inference()` functions
2. Add configuration files:
   - `config/model/your_model.yaml`
   - `config/train/your_model.yaml`
   - `config/inference/your_model.yaml`
3. Update `run_comparison_experiments.sh` to include your model

## Notes

- All models use the same data preprocessing and FOM computation for fair comparison
- Checkpoints are saved regularly and can be resumed
- WandB logging is supported (set `logger: True` in config)
- Debug mode uses smaller datasets and fewer epochs for quick testing
