#!/usr/bin/env python3
"""
Benchmark script to compare training performance optimizations.
"""
import time
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torch.amp import GradScaler, autocast
import torch.nn as nn
from photo_gen.models.unet import UNET
from photo_gen.utils.unet_utils import UNetPad


def benchmark_training_speed():
    """Compare training speed with different optimizations."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("CUDA not available, skipping GPU benchmarks")
        return
    
    # Synthetic data similar to your dataset
    batch_size = 32
    data_shape = (100, 1, 101, 91)  # Similar to your 101x91 images
    data = torch.randn(data_shape, dtype=torch.float32)
    dataset = TensorDataset(data)
    
    print("Benchmarking UNet training performance...")
    print(f"Data shape: {data_shape}")
    print(f"Batch size: {batch_size}")
    print(f"Device: {device}")
    print("-" * 50)
    
    # Test configurations
    configs = [
        {
            "name": "Original (Large Model)",
            "model_params": {},  # Use default parameters
            "batch_size": 8,
            "mixed_precision": False
        },
        {
            "name": "Optimized (Smaller Model)",
            "model_params": {
                "Channels": [32, 64, 128, 256, 256, 192],
                "Attentions": [False, False, True, False, True, False],
                "Upscales": [False, False, False, True, True, True],
                "num_groups": 8,
                "dropout_prob": 0.1,
                "num_heads": 4
            },
            "batch_size": 8,
            "mixed_precision": False
        },
        {
            "name": "Larger Batch Size",
            "model_params": {},  # Use default but smaller for memory
            "batch_size": 16,
            "mixed_precision": False
        },
        {
            "name": "Mixed Precision Training",
            "model_params": {},
            "batch_size": 16,
            "mixed_precision": True
        }
    ]
    
    results = []
    
    for config in configs:
        print(f"Testing: {config['name']}")
        
        try:
            # Create model
            model = UNET(
                input_channels=1,
                output_channels=1,
                **config["model_params"],
                device=device
            ).to(device)
            
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            print(f"  Parameters: {total_params:,}")
            
            # Setup data loader
            loader = DataLoader(
                dataset, 
                batch_size=config["batch_size"], 
                shuffle=True,
                num_workers=4,
                pin_memory=True
            )
            
            # Setup training components
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
            criterion = nn.MSELoss()
            scaler = GradScaler(device=device) if config["mixed_precision"] else None
            
            # Create padding function
            sample_input = torch.randn(1, 1, 101, 91)
            depth = model.num_layers // 2
            pad_fn = UNetPad(sample_input, depth=depth)
            
            # Warm up
            model.train()
            for i, [x] in enumerate(loader):
                if i >= 2:  # Just a few warmup steps
                    break
                x = x.to(device)
                x = pad_fn(x)  # Apply padding
                t = torch.randint(0, 1000, (x.size(0),), device=device)
                
                if config["mixed_precision"]:
                    with autocast(device_type=device.type):
                        output = model(x, t)
                        target = torch.randn_like(output)
                        loss = criterion(output, target)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    output = model(x, t)
                    target = torch.randn_like(output)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                optimizer.zero_grad()
            
            # Benchmark
            torch.cuda.synchronize()
            start_time = time.time()
            
            steps = 0
            for i, [x] in enumerate(loader):
                if i >= 10:  # Test 10 steps
                    break
                
                x = x.to(device)
                x = pad_fn(x)  # Apply padding
                t = torch.randint(0, 1000, (x.size(0),), device=device)
                
                if config["mixed_precision"]:
                    with autocast():
                        output = model(x, t)
                        target = torch.randn_like(output)
                        loss = criterion(output, target)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    output = model(x, t)
                    target = torch.randn_like(output)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                
                optimizer.zero_grad()
                steps += 1
            
            torch.cuda.synchronize()
            end_time = time.time()
            
            time_per_step = (end_time - start_time) / steps
            samples_per_second = config["batch_size"] / time_per_step
            
            print(f"  Time per step: {time_per_step:.3f}s")
            print(f"  Samples/second: {samples_per_second:.1f}")
            print(f"  Memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            
            results.append({
                "name": config["name"],
                "params": total_params,
                "time_per_step": time_per_step,
                "samples_per_second": samples_per_second,
                "memory_gb": torch.cuda.memory_allocated() / 1024**3
            })
            
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                "name": config["name"],
                "error": str(e)
            })
        
        # Clear memory
        if 'model' in locals():
            del model
        torch.cuda.empty_cache()
        print()
    
    # Summary
    print("=" * 60)
    print("PERFORMANCE COMPARISON SUMMARY")
    print("=" * 60)
    
    baseline = None
    for result in results:
        if "error" in result:
            print(f"{result['name']}: ERROR - {result['error']}")
            continue
            
        if baseline is None:
            baseline = result
            speedup = 1.0
        else:
            speedup = baseline["time_per_step"] / result["time_per_step"]
        
        print(f"{result['name']}:")
        print(f"  Parameters: {result['params']:,}")
        print(f"  Time/step: {result['time_per_step']:.3f}s")
        print(f"  Speedup: {speedup:.1f}x")
        print(f"  Samples/sec: {result['samples_per_second']:.1f}")
        print(f"  Memory: {result['memory_gb']:.2f} GB")
        print()


if __name__ == "__main__":
    benchmark_training_speed()