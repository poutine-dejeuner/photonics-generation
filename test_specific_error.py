#!/usr/bin/env python3
"""
Specific test to reproduce the "Expected size 48 but got size 47" error.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
sys.path.append('/home/mila/l/letournv/repos/diffusion-model')

from unet_utils import UNetPad


class SimpleUNet(nn.Module):
    """Simplified UNet to reproduce the concatenation error."""
    
    def __init__(self, in_channels=1, out_channels=1, num_layers=4):
        super().__init__()
        self.num_layers = num_layers
        
        # Encoder
        self.encoder = nn.ModuleList()
        channels = [in_channels, 64, 128, 256, 512]
        
        for i in range(num_layers):
            self.encoder.append(
                nn.Sequential(
                    nn.Conv2d(channels[i], channels[i+1], 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channels[i+1], channels[i+1], 3, padding=1),
                    nn.ReLU(inplace=True)
                )
            )
        
        # Decoder
        self.decoder = nn.ModuleList()
        for i in range(num_layers-1, 0, -1):
            self.decoder.append(
                nn.Sequential(
                    nn.ConvTranspose2d(channels[i+1], channels[i], 2, stride=2),
                    nn.Conv2d(channels[i]*2, channels[i], 3, padding=1),  # *2 for skip connection
                    nn.ReLU(inplace=True)
                )
            )
        
        self.final_conv = nn.Conv2d(channels[1], out_channels, 1)
    
    def forward(self, x):
        # Encoder
        encoder_features = []
        current = x
        
        for i, layer in enumerate(self.encoder):
            current = layer(current)
            encoder_features.append(current)
            print(f"Encoder {i}: {current.shape}")
            
            if i < len(self.encoder) - 1:  # Don't pool at the deepest level
                current = F.max_pool2d(current, 2)
                print(f"After pooling {i}: {current.shape}")
        
        # Decoder
        current = encoder_features[-1]  # Start from deepest features
        
        for i, layer in enumerate(self.decoder):
            # Upsample
            upsampled = layer[0](current)  # ConvTranspose2d
            print(f"Decoder {i} upsampled: {upsampled.shape}")
            
            # Get skip connection
            skip_idx = len(encoder_features) - 2 - i
            skip = encoder_features[skip_idx]
            print(f"Skip connection {i}: {skip.shape}")
            
            # Try concatenation - this is where the error might occur
            try:
                concatenated = torch.cat([upsampled, skip], dim=1)
                print(f"Concatenated {i}: {concatenated.shape}")
            except RuntimeError as e:
                print(f"❌ CONCATENATION ERROR: {e}")
                print(f"Upsampled shape: {upsampled.shape}")
                print(f"Skip shape: {skip.shape}")
                return None
            
            # Apply conv
            current = layer[1:](concatenated)  # Remaining layers
            print(f"After conv {i}: {current.shape}")
        
        return self.final_conv(current)


def test_specific_error_reproduction():
    """Try to reproduce the exact error with different scenarios."""
    print("="*60)
    print("REPRODUCING SPECIFIC ERROR: Expected size 48 but got size 47")
    print("="*60)
    
    # Test cases that might trigger the error
    test_cases = [
        (101, 91, 4),  # Your original case
        (95, 95, 4),   # Square odd
        (47, 48, 4),   # One even, one odd
        (94, 95, 4),   # Both even/odd close to powers of 2
    ]
    
    for h, w, num_layers in test_cases:
        print(f"\n--- Testing H={h}, W={w}, num_layers={num_layers} ---")
        
        # Create input
        x = torch.randn(1, 1, h, w)
        print(f"Original input: {x.shape}")
        
        # Apply padding
        depth = num_layers // 2
        pad_fn = UNetPad(x, depth=depth)
        x_padded = pad_fn(x)
        print(f"Padded input: {x_padded.shape}")
        
        # Create model
        model = SimpleUNet(num_layers=num_layers)
        
        # Try forward pass
        try:
            with torch.no_grad():
                output = model(x_padded)
                if output is not None:
                    print(f"✅ SUCCESS: Output shape {output.shape}")
                else:
                    print("❌ FAILED: Forward pass returned None")
        except Exception as e:
            print(f"❌ ERROR during forward pass: {e}")


def test_manual_dimension_tracking():
    """Manually track dimensions through a typical UNet to find the mismatch."""
    print("\n" + "="*60)
    print("MANUAL DIMENSION TRACKING")
    print("="*60)
    
    # Use your specific case
    h, w = 101, 91
    num_layers = 4
    depth = num_layers // 2  # This should be 2
    
    print(f"Original dimensions: {h} x {w}")
    print(f"Number of layers: {num_layers}")
    print(f"Depth (num_layers//2): {depth}")
    
    # Apply UNetPad
    x = torch.randn(1, 1, h, w)
    pad_fn = UNetPad(x, depth=depth)
    x_padded = pad_fn(x)
    
    h_pad, w_pad = x_padded.shape[-2:]
    print(f"Padded dimensions: {h_pad} x {w_pad}")
    print(f"Required divisibility: 2^{depth} = {2**depth}")
    print(f"H divisible: {h_pad % (2**depth) == 0}")
    print(f"W divisible: {w_pad % (2**depth) == 0}")
    
    # Manually track encoder path
    current_h, current_w = h_pad, w_pad
    encoder_dims = [(current_h, current_w)]
    
    print(f"\nEncoder path:")
    for level in range(depth):
        print(f"Level {level}: {current_h} x {current_w}")
        if level < depth - 1:  # Don't downsample at deepest level
            current_h = current_h // 2
            current_w = current_w // 2
            encoder_dims.append((current_h, current_w))
    
    # Manually track decoder path
    print(f"\nDecoder path:")
    for level in range(depth-1):
        current_h = current_h * 2
        current_w = current_w * 2
        
        skip_level = depth - 2 - level
        skip_h, skip_w = encoder_dims[skip_level]
        
        print(f"Decoder level {level}:")
        print(f"  Upsampled: {current_h} x {current_w}")
        print(f"  Skip conn: {skip_h} x {skip_w}")
        print(f"  Match: {current_h == skip_h and current_w == skip_w}")
        
        if current_h != skip_h or current_w != skip_w:
            print(f"  ❌ DIMENSION MISMATCH!")
            print(f"     Expected: {current_h} x {current_w}")
            print(f"     Got:      {skip_h} x {skip_w}")
            return False
    
    print("✅ All dimensions match correctly!")
    return True


def test_actual_unet_forward():
    """Test with actual UNet model if available."""
    print("\n" + "="*60)
    print("TESTING WITH ACTUAL UNET (if available)")
    print("="*60)
    
    try:
        # Try to import the actual UNet model
        from models.unet import UNet
        
        # Create a small test
        x = torch.randn(1, 1, 101, 91)
        pad_fn = UNetPad(x, depth=2)  # Using depth=2 as in your code (num_layers//2)
        x_padded = pad_fn(x)
        
        print(f"Input shape: {x.shape}")
        print(f"Padded shape: {x_padded.shape}")
        
        # This would require knowing the exact UNet configuration
        # For now, just verify padding works
        print("✅ Padding successful")
        
    except ImportError:
        print("UNet model not available for testing")
    except Exception as e:
        print(f"Error testing with actual UNet: {e}")


def main():
    """Run specific error reproduction tests."""
    print("Starting specific error reproduction tests...")
    
    test_specific_error_reproduction()
    test_manual_dimension_tracking()
    test_actual_unet_forward()
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print("\n🔍 If the error persists, check:")
    print("1. The exact depth parameter being used in your UNet")
    print("2. Whether you're using model.num_layers or model.num_layers//2")
    print("3. Any custom upsampling/downsampling operations")
    print("4. Check if there are any custom Conv2d layers with different strides")


if __name__ == "__main__":
    main()
