#!/usr/bin/env python3
"""
Script to generate 1024 samples from each checkpoint.pt file in train3/job/ directories
and append the results to the existing images.npy files.
"""
import os
import sys
import numpy as np
import torch
import hydra
from omegaconf import OmegaConf, DictConfig
from pathlib import Path
from hydra import compose, initialize_config_dir

# Add the current directory to Python path to import from train3
sys.path.append('/home/mila/l/letournv/repos/diffusion-model')
from train3 import inference_parallele


def load_config():
    """Load the inference configuration using Hydra"""
    config_dir = '/home/mila/l/letournv/repos/diffusion-model/config'
    
    # Initialize Hydra with the config directory
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        # Compose the configuration
        cfg = compose(config_name="inference.yaml")
    
    return cfg


def get_job_directories():
    """Get all job directories containing checkpoint.pt files"""
    train3_path = Path('/home/mila/l/letournv/repos/diffusion-model/train3')
    job_dirs = []
    
    # List of known job directories based on your description
    job_names = ['7121883', '7121885', '7121886', '7121888', '7121889', '7121890', 
                 '7121891', '7212292', '7212298', '7212299', '7212301', '7212302']
    
    for job_name in job_names:
        job_dir = train3_path / job_name
        checkpoint_path = job_dir / 'checkpoint.pt'
        images_path = job_dir / 'images.npy'
        
        if checkpoint_path.exists():
            job_dirs.append({
                'name': job_name,
                'dir': job_dir,
                'checkpoint': checkpoint_path,
                'images_npy': images_path
            })
            print(f"Found checkpoint in {job_name}")
        else:
            print(f"Warning: No checkpoint.pt found in {job_name}")
    
    return job_dirs


def generate_samples_for_job(job_info, cfg, batch_size=64, total_samples=1024):
    """Generate samples for a single job checkpoint"""
    print(f"\nProcessing job {job_info['name']}...")
    
    checkpoint_path = str(job_info['checkpoint'])
    images_npy_path = job_info['images_npy']
    
    # Calculate number of batches needed
    num_batches = total_samples // batch_size
    remaining_samples = total_samples % batch_size
    
    all_samples = []
    
    # Generate samples in batches
    for batch_idx in range(num_batches):
        print(f"  Generating batch {batch_idx + 1}/{num_batches} ({batch_size} samples)...")
        
        # Create a copy of config for this batch
        cfg_batch = OmegaConf.create(OmegaConf.to_yaml(cfg))
        cfg_batch.n_images = batch_size
        
        try:
            # Generate samples using inference_parallele
            samples = inference_parallele(
                cfg=cfg_batch,
                checkpoint_path=checkpoint_path,
            )
            
            all_samples.append(samples)
            print(f"    Generated {samples.shape[0]} samples")
            
        except Exception as e:
            print(f"    Error generating batch {batch_idx + 1}: {e}")
            continue
    
    # Generate remaining samples if any
    if remaining_samples > 0:
        print(f"  Generating final batch ({remaining_samples} samples)...")
        cfg_remaining = OmegaConf.create(OmegaConf.to_yaml(cfg))
        cfg_remaining.n_images = remaining_samples
        
        try:
            samples = inference_parallele(
                cfg=cfg_remaining,
                checkpoint_path=checkpoint_path,
                savepath="temp",
                meep_eval=False
            )
            all_samples.append(samples)
            print(f"    Generated {samples.shape[0]} samples")
        except Exception as e:
            print(f"    Error generating final batch: {e}")
    
    # Concatenate all samples
    if all_samples:
        new_samples = np.concatenate(all_samples, axis=0)
        print(f"  Total generated samples: {new_samples.shape[0]}")
        
        # Load existing images.npy and append new samples
        if images_npy_path.exists():
            print(f"  Loading existing images from {images_npy_path}")
            existing_samples = np.load(images_npy_path)
            print(f"  Existing samples shape: {existing_samples.shape}")
            
            # Append new samples to existing ones
            combined_samples = np.concatenate([existing_samples, new_samples], axis=0)
        else:
            print(f"  No existing images.npy found, creating new file")
            combined_samples = new_samples
        
        # Save the combined samples
        print(f"  Saving {combined_samples.shape[0]} total samples to {images_npy_path}")
        np.save(images_npy_path, combined_samples)
        
        return len(new_samples)
    else:
        print(f"  No samples generated for job {job_info['name']}")
        return 0


def main():
    """Main function to process all checkpoints"""
    print("Starting sample generation script...")
    
    # Load configuration
    print("Loading configuration...")
    cfg = load_config()
    
    # Get all job directories
    print("Finding job directories...")
    job_dirs = get_job_directories()
    
    if not job_dirs:
        print("No job directories with checkpoint.pt files found!")
        return
    
    print(f"Found {len(job_dirs)} job directories to process")
    
    # Process each job
    total_generated = 0
    successful_jobs = 0
    
    for job_info in job_dirs:
        try:
            generated_count = generate_samples_for_job(job_info, cfg)
            total_generated += generated_count
            if generated_count > 0:
                successful_jobs += 1
        except Exception as e:
            print(f"Error processing job {job_info['name']}: {e}")
            continue
    
    print(f"\n=== Summary ===")
    print(f"Jobs processed successfully: {successful_jobs}/{len(job_dirs)}")
    print(f"Total samples generated: {total_generated}")
    print(f"Average samples per successful job: {total_generated/successful_jobs if successful_jobs > 0 else 0:.1f}")


if __name__ == "__main__":
    main()
