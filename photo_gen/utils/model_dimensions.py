import torch
import torch.nn as nn
from typing import List, Tuple, Dict
import sys
import os

# Add parent directory to path to import models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from photo_gen.models.vae import VAE
from photo_gen.models.wgan import WGAN
from photo_gen.models.standard_gan import StandardGAN


N_PARAMS = 35_000_000


def count_parameters(model: nn.Module) -> int:
    """Count the total number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def calculate_conv2d_params(in_channels: int, out_channels: int, kernel_size: int, bias: bool = True) -> int:
    """Calculate parameters for a Conv2d layer."""
    return in_channels * out_channels * kernel_size * kernel_size + (out_channels if bias else 0)


def calculate_linear_params(in_features: int, out_features: int, bias: bool = True) -> int:
    """Calculate parameters for a Linear layer."""
    return in_features * out_features + (out_features if bias else 0)


def calculate_batchnorm_params(num_features: int) -> int:
    """Calculate parameters for BatchNorm2d layer."""
    return num_features * 2  # weight and bias


def calculate_instancenorm_params(num_features: int) -> int:
    """Calculate parameters for InstanceNorm2d layer."""
    # InstanceNorm2d has no learnable parameters by default (affine=False)
    return 0


# VAE Parameter Calculation Functions
def calculate_vae_encoder_params(img_channels: int, hidden_dim: int, latent_dim: int) -> int:
    """Calculate parameters for VAE Encoder."""
    params = 0
    
    # Conv layers
    params += calculate_conv2d_params(img_channels, hidden_dim, 4, True)  # First conv
    params += calculate_conv2d_params(hidden_dim, hidden_dim * 2, 4, True)  # Second conv
    params += calculate_batchnorm_params(hidden_dim * 2)
    params += calculate_conv2d_params(hidden_dim * 2, hidden_dim * 4, 4, True)  # Third conv
    params += calculate_batchnorm_params(hidden_dim * 4)
    params += calculate_conv2d_params(hidden_dim * 4, hidden_dim * 8, 4, True)  # Fourth conv
    params += calculate_batchnorm_params(hidden_dim * 8)
    
    # FC layers (pooled_size = hidden_dim * 8 * 4 * 4)
    pooled_size = hidden_dim * 8 * 4 * 4
    params += calculate_linear_params(pooled_size, latent_dim, True)  # fc_mu
    params += calculate_linear_params(pooled_size, latent_dim, True)  # fc_logvar
    
    return params


def calculate_vae_decoder_params(latent_dim: int, img_channels: int, hidden_dim: int) -> int:
    """Calculate parameters for VAE Decoder."""
    params = 0
    
    # FC layer
    fc_output_size = hidden_dim * 8 * 4 * 4
    params += calculate_linear_params(latent_dim, fc_output_size, True)
    
    # Deconv layers
    params += calculate_conv2d_params(hidden_dim * 8, hidden_dim * 4, 4, True)  # First deconv
    params += calculate_batchnorm_params(hidden_dim * 4)
    params += calculate_conv2d_params(hidden_dim * 4, hidden_dim * 2, 4, True)  # Second deconv
    params += calculate_batchnorm_params(hidden_dim * 2)
    params += calculate_conv2d_params(hidden_dim * 2, hidden_dim, 4, True)  # Third deconv
    params += calculate_batchnorm_params(hidden_dim)
    params += calculate_conv2d_params(hidden_dim, img_channels, 4, True)  # Final deconv
    
    return params


def calculate_vae_total_params(img_channels: int, latent_dim: int, hidden_dim: int) -> int:
    """Calculate total parameters for VAE."""
    encoder_params = calculate_vae_encoder_params(img_channels, hidden_dim, latent_dim)
    decoder_params = calculate_vae_decoder_params(latent_dim, img_channels, hidden_dim)
    return encoder_params + decoder_params


# WGAN Parameter Calculation Functions
def calculate_wgan_generator_params(latent_dim: int, img_channels: int, hidden_dim: int, img_size: Tuple[int, int] = (101, 91)) -> int:
    """Calculate parameters for WGAN Generator."""
    params = 0
    
    # Initial linear layer
    init_size = max(4, min(img_size[0], img_size[1]) // 16)
    params += calculate_linear_params(latent_dim, hidden_dim * 4 * init_size * init_size, True)
    
    # Conv blocks
    params += calculate_batchnorm_params(hidden_dim * 4)
    params += calculate_conv2d_params(hidden_dim * 4, hidden_dim * 2, 3, True)
    params += calculate_batchnorm_params(hidden_dim * 2)
    params += calculate_conv2d_params(hidden_dim * 2, hidden_dim, 3, True)
    params += calculate_batchnorm_params(hidden_dim)
    params += calculate_conv2d_params(hidden_dim, hidden_dim, 3, True)
    params += calculate_batchnorm_params(hidden_dim)
    params += calculate_conv2d_params(hidden_dim, img_channels, 3, True)
    
    return params


def calculate_wgan_critic_params(img_channels: int, hidden_dim: int) -> int:
    """Calculate parameters for WGAN Critic."""
    params = 0
    
    # Conv blocks
    params += calculate_conv2d_params(img_channels, hidden_dim, 4, True)  # First block
    params += calculate_conv2d_params(hidden_dim, hidden_dim * 2, 4, True)  # Second block
    params += calculate_instancenorm_params(hidden_dim * 2)
    params += calculate_conv2d_params(hidden_dim * 2, hidden_dim * 4, 4, True)  # Third block
    params += calculate_instancenorm_params(hidden_dim * 4)
    params += calculate_conv2d_params(hidden_dim * 4, hidden_dim * 8, 4, True)  # Fourth block
    params += calculate_instancenorm_params(hidden_dim * 8)
    
    # Final linear layer
    params += calculate_linear_params(hidden_dim * 8, 1, True)
    
    return params


def calculate_wgan_total_params(latent_dim: int, img_channels: int, hidden_dim: int, img_size: Tuple[int, int] = (101, 91)) -> int:
    """Calculate total parameters for WGAN."""
    generator_params = calculate_wgan_generator_params(latent_dim, img_channels, hidden_dim, img_size)
    critic_params = calculate_wgan_critic_params(img_channels, hidden_dim)
    return generator_params + critic_params


# Standard GAN Parameter Calculation Functions
def calculate_standard_gan_generator_params(latent_dim: int, img_channels: int, hidden_dim: int, img_size: Tuple[int, int] = (101, 91)) -> int:
    """Calculate parameters for Standard GAN Generator."""
    params = 0
    
    # Initial linear layer (same as WGAN)
    init_size = max(4, min(img_size[0], img_size[1]) // 16)
    params += calculate_linear_params(latent_dim, hidden_dim * 4 * init_size * init_size, True)
    
    # Conv blocks (same structure as WGAN)
    params += calculate_batchnorm_params(hidden_dim * 4)
    params += calculate_conv2d_params(hidden_dim * 4, hidden_dim * 2, 3, True)
    params += calculate_batchnorm_params(hidden_dim * 2)
    params += calculate_conv2d_params(hidden_dim * 2, hidden_dim, 3, True)
    params += calculate_batchnorm_params(hidden_dim)
    params += calculate_conv2d_params(hidden_dim, hidden_dim, 3, True)
    params += calculate_batchnorm_params(hidden_dim)
    params += calculate_conv2d_params(hidden_dim, img_channels, 3, True)
    
    return params


def calculate_standard_gan_discriminator_params(img_channels: int, hidden_dim: int) -> int:
    """Calculate parameters for Standard GAN Discriminator."""
    params = 0
    
    # Conv blocks
    params += calculate_conv2d_params(img_channels, hidden_dim, 4, True)  # First block
    params += calculate_conv2d_params(hidden_dim, hidden_dim * 2, 4, True)  # Second block
    params += calculate_batchnorm_params(hidden_dim * 2)
    params += calculate_conv2d_params(hidden_dim * 2, hidden_dim * 4, 4, True)  # Third block
    params += calculate_batchnorm_params(hidden_dim * 4)
    params += calculate_conv2d_params(hidden_dim * 4, hidden_dim * 8, 4, True)  # Fourth block
    params += calculate_batchnorm_params(hidden_dim * 8)
    
    # Final linear layer
    params += calculate_linear_params(hidden_dim * 8, 1, True)
    
    return params


def calculate_standard_gan_total_params(latent_dim: int, img_channels: int, hidden_dim: int, img_size: Tuple[int, int] = (101, 91)) -> int:
    """Calculate total parameters for Standard GAN."""
    generator_params = calculate_standard_gan_generator_params(latent_dim, img_channels, hidden_dim, img_size)
    discriminator_params = calculate_standard_gan_discriminator_params(img_channels, hidden_dim)
    return generator_params + discriminator_params


def find_optimal_hidden_dims(target_params: int = N_PARAMS, 
                            img_channels: int = 1,
                            latent_dim: int = 128,
                            img_size: Tuple[int, int] = (101, 91),
                            tolerance: float = 0.05) -> Dict[str, List[Tuple[int, int]]]:
    """
    Find optimal hidden dimensions for each model type to achieve target parameter count.
    
    Args:
        target_params: Target number of parameters
        img_channels: Number of image channels
        latent_dim: Latent dimension
        img_size: Image size (height, width)
        tolerance: Tolerance for parameter count (±5% by default)
    
    Returns:
        Dictionary with model types as keys and list of (hidden_dim, actual_params) tuples
    """
    results = {
        'VAE': [],
        'WGAN': [],
        'Standard_GAN': []
    }
    
    min_params = target_params * (1 - tolerance)
    max_params = target_params * (1 + tolerance)
    
    # Search range for hidden dimensions
    hidden_dims = range(8, 512, 8)  # Search from 8 to 512 in steps of 8
    
    for hidden_dim in hidden_dims:
        # VAE
        vae_params = calculate_vae_total_params(img_channels, latent_dim, hidden_dim)
        if min_params <= vae_params <= max_params:
            results['VAE'].append((hidden_dim, vae_params))
        
        # WGAN
        wgan_params = calculate_wgan_total_params(latent_dim, img_channels, hidden_dim, img_size)
        if min_params <= wgan_params <= max_params:
            results['WGAN'].append((hidden_dim, wgan_params))
        
        # Standard GAN
        standard_gan_params = calculate_standard_gan_total_params(latent_dim, img_channels, hidden_dim, img_size)
        if min_params <= standard_gan_params <= max_params:
            results['Standard_GAN'].append((hidden_dim, standard_gan_params))
    
    return results


def print_parameter_analysis(target_params: int = N_PARAMS,
                           img_channels: int = 1,
                           latent_dim: int = 128,
                           img_size: Tuple[int, int] = (101, 91),
                           tolerance: float = 0.05):
    """
    Print detailed parameter analysis and optimal configurations.
    """
    print(f"Target Parameters: {target_params:,}")
    print(f"Tolerance: ±{tolerance*100:.1f}%")
    print(f"Acceptable range: {target_params*(1-tolerance):,.0f} - {target_params*(1+tolerance):,.0f}")
    print(f"Image channels: {img_channels}")
    print(f"Latent dimension: {latent_dim}")
    print(f"Image size: {img_size}")
    print("=" * 80)
    
    results = find_optimal_hidden_dims(target_params, img_channels, latent_dim, img_size, tolerance)
    
    for model_type, configs in results.items():
        print(f"\n{model_type} Optimal Configurations:")
        print("-" * 50)
        
        if configs:
            for hidden_dim, actual_params in configs:
                error_pct = abs(actual_params - target_params) / target_params * 100
                print(f"  Hidden dim: {hidden_dim:3d} | Parameters: {actual_params:,} | Error: {error_pct:.2f}%")
        else:
            print("  No configurations found within tolerance range.")
    
    # Additional analysis: show parameter breakdown for one example from each model
    print("\n" + "=" * 80)
    print("PARAMETER BREAKDOWN EXAMPLES:")
    print("=" * 80)
    
    # Use a common hidden dimension for comparison
    example_hidden_dim = 64
    
    # VAE breakdown
    print(f"\nVAE (hidden_dim={example_hidden_dim}):")
    encoder_params = calculate_vae_encoder_params(img_channels, example_hidden_dim, latent_dim)
    decoder_params = calculate_vae_decoder_params(latent_dim, img_channels, example_hidden_dim)
    print(f"  Encoder: {encoder_params:,} parameters")
    print(f"  Decoder: {decoder_params:,} parameters")
    print(f"  Total:   {encoder_params + decoder_params:,} parameters")
    
    # WGAN breakdown
    print(f"\nWGAN (hidden_dim={example_hidden_dim}):")
    gen_params = calculate_wgan_generator_params(latent_dim, img_channels, example_hidden_dim, img_size)
    critic_params = calculate_wgan_critic_params(img_channels, example_hidden_dim)
    print(f"  Generator: {gen_params:,} parameters")
    print(f"  Critic:    {critic_params:,} parameters")
    print(f"  Total:     {gen_params + critic_params:,} parameters")
    
    # Standard GAN breakdown
    print(f"\nStandard GAN (hidden_dim={example_hidden_dim}):")
    gen_params = calculate_standard_gan_generator_params(latent_dim, img_channels, example_hidden_dim, img_size)
    disc_params = calculate_standard_gan_discriminator_params(img_channels, example_hidden_dim)
    print(f"  Generator:     {gen_params:,} parameters")
    print(f"  Discriminator: {disc_params:,} parameters")
    print(f"  Total:         {gen_params + disc_params:,} parameters")


def create_config_table(target_params: int = N_PARAMS,
                       img_channels: int = 1,
                       latent_dim: int = 128,
                       img_size: Tuple[int, int] = (101, 91),
                       tolerance: float = 0.05) -> str:
    """
    Create a formatted table of optimal configurations.
    """
    results = find_optimal_hidden_dims(target_params, img_channels, latent_dim, img_size, tolerance)
    
    table = []
    table.append("| Model Type    | Hidden Dim | Parameters  | Error (%) |")
    table.append("|---------------|------------|-------------|-----------|")
    
    for model_type, configs in results.items():
        if configs:
            for hidden_dim, actual_params in configs:
                error_pct = abs(actual_params - target_params) / target_params * 100
                table.append(f"| {model_type:<13} | {hidden_dim:>10} | {actual_params:>11,} | {error_pct:>8.2f}% |")
        else:
            table.append(f"| {model_type:<13} | {'N/A':>10} | {'N/A':>11} | {'N/A':>8} |")
    
    return "\n".join(table)


def verify_actual_parameter_counts(img_channels: int = 1,
                                  latent_dim: int = 128,
                                  hidden_dim: int = 64,
                                  img_size: Tuple[int, int] = (101, 91)) -> Dict[str, Tuple[int, int]]:
    """
    Initialize actual models and compute their parameter counts using PyTorch's method.
    
    Args:
        img_channels: Number of image channels
        latent_dim: Latent dimension
        hidden_dim: Hidden dimension
        img_size: Image size (height, width)
    
    Returns:
        Dictionary with model types as keys and tuples of (calculated_params, actual_params)
    """
    device = torch.device('cpu')  # Use CPU to avoid CUDA requirements
    results = {}
    
    try:
        # VAE
        print(f"Initializing VAE with hidden_dim={hidden_dim}, latent_dim={latent_dim}...")
        vae_model = VAE(img_channels=img_channels, img_size=img_size, 
                       latent_dim=latent_dim, hidden_dim=hidden_dim).to(device)
        actual_vae_params = sum(p.numel() for p in vae_model.parameters() if p.requires_grad)
        calculated_vae_params = calculate_vae_total_params(img_channels, latent_dim, hidden_dim)
        results['VAE'] = (calculated_vae_params, actual_vae_params)
        print(f"VAE - Calculated: {calculated_vae_params:,}, Actual: {actual_vae_params:,}")
        
    except Exception as e:
        print(f"Error initializing VAE: {e}")
        results['VAE'] = (0, 0)
    
    try:
        # WGAN
        print(f"Initializing WGAN with hidden_dim={hidden_dim}, latent_dim={latent_dim}...")
        wgan_model = WGAN(latent_dim=latent_dim, img_channels=img_channels, 
                         img_size=img_size, hidden_dim=hidden_dim, device=device)
        actual_wgan_params = (sum(p.numel() for p in wgan_model.generator.parameters() if p.requires_grad) +
                             sum(p.numel() for p in wgan_model.critic.parameters() if p.requires_grad))
        calculated_wgan_params = calculate_wgan_total_params(latent_dim, img_channels, hidden_dim, img_size)
        results['WGAN'] = (calculated_wgan_params, actual_wgan_params)
        print(f"WGAN - Calculated: {calculated_wgan_params:,}, Actual: {actual_wgan_params:,}")
        
    except Exception as e:
        print(f"Error initializing WGAN: {e}")
        results['WGAN'] = (0, 0)
    
    try:
        # Standard GAN
        print(f"Initializing Standard GAN with hidden_dim={hidden_dim}, latent_dim={latent_dim}...")
        standard_gan_model = StandardGAN(latent_dim=latent_dim, img_channels=img_channels, 
                                       img_size=img_size, hidden_dim=hidden_dim, device=device)
        actual_standard_gan_params = (sum(p.numel() for p in standard_gan_model.generator.parameters() if p.requires_grad) +
                                    sum(p.numel() for p in standard_gan_model.discriminator.parameters() if p.requires_grad))
        calculated_standard_gan_params = calculate_standard_gan_total_params(latent_dim, img_channels, hidden_dim, img_size)
        results['Standard_GAN'] = (calculated_standard_gan_params, actual_standard_gan_params)
        print(f"Standard GAN - Calculated: {calculated_standard_gan_params:,}, Actual: {actual_standard_gan_params:,}")
        
    except Exception as e:
        print(f"Error initializing Standard GAN: {e}")
        results['Standard_GAN'] = (0, 0)
    
    return results


def compare_calculated_vs_actual(target_params: int = N_PARAMS,
                                img_channels: int = 1,
                                latent_dim: int = 128,
                                img_size: Tuple[int, int] = (101, 91),
                                tolerance: float = 0.05):
    """
    Compare calculated parameter counts with actual PyTorch model parameter counts.
    """
    print("=" * 80)
    print("CALCULATED vs ACTUAL PARAMETER COUNT VERIFICATION")
    print("=" * 80)
    
    # Get optimal configurations
    results = find_optimal_hidden_dims(target_params, img_channels, latent_dim, img_size, tolerance)
    
    # Test with a few different hidden dimensions
    test_hidden_dims = [32, 64, 128]
    
    # Also test with optimal configurations if found
    for model_type, configs in results.items():
        if configs:
            # Add the first optimal configuration to test
            optimal_hidden_dim = configs[0][0]
            if optimal_hidden_dim not in test_hidden_dims:
                test_hidden_dims.append(optimal_hidden_dim)
    
    for hidden_dim in test_hidden_dims:
        print(f"\nTesting with hidden_dim = {hidden_dim}")
        print("-" * 50)
        
        verification_results = verify_actual_parameter_counts(
            img_channels=img_channels,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            img_size=img_size
        )
        
        print("\nComparison Results:")
        print("| Model Type    | Calculated   | Actual       | Difference   | Match? |")
        print("|---------------|--------------|--------------|--------------|--------|")
        
        for model_type, (calculated, actual) in verification_results.items():
            if actual > 0:  # Only show if model was successfully initialized
                difference = actual - calculated
                match = "✓" if difference == 0 else "✗"
                print(f"| {model_type:<13} | {calculated:>12,} | {actual:>12,} | {difference:>12,} | {match:>6} |")
            else:
                print(f"| {model_type:<13} | {'ERROR':>12} | {'ERROR':>12} | {'ERROR':>12} | {'✗':>6} |")


if __name__ == "__main__":
    # Run the analysis
    print_parameter_analysis()
    
    print("\n" + "=" * 80)
    print("CONFIGURATION TABLE:")
    print("=" * 80)
    print(create_config_table())
    
    # Run verification
    compare_calculated_vs_actual()