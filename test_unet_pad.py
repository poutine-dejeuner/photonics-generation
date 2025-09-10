#!/usr/bin/env python3
"""
Test script to debug UNetPad function issues.

This test checks if the UNetPad function correctly ensures that
the last 2 dimensions are divisible by 2^num_layers, and verifies
that the padding/unpadding operations work correctly.
"""

import torch
import torch.nn.functional as F
import sys
import os
sys.path.append('/home/mila/l/letournv/repos/diffusion-model')

from unet_utils import UNetPad, unet_pad_fun


def test_unet_pad_divisibility():
    """Test if UNetPad ensures divisibility by 2^depth."""
    print("="*60)
    print("TESTING UNetPad DIVISIBILITY")
    print("="*60)
    
    # Test various input shapes and depths
    test_cases = [
        ((1, 1, 101, 91), 4),   # Your specific case
        ((1, 1, 101, 91), 3),   # Different depth
        ((1, 1, 101, 91), 5),   # Deeper
        ((2, 3, 64, 64), 4),    # Square, already divisible
        ((1, 1, 47, 48), 4),    # One divisible, one not
        ((1, 1, 100, 100), 4),  # Even numbers
        ((1, 1, 99, 97), 4),    # Odd numbers
    ]
    
    all_passed = True
    
    for i, (shape, depth) in enumerate(test_cases):
        print(f"\nTest Case {i+1}: shape={shape}, depth={depth}")
        print(f"Required divisibility: 2^{depth} = {2**depth}")
        
        # Create test tensor
        x = torch.randn(shape)
        print(f"Original shape: {x.shape}")
        
        # Apply UNetPad
        pad_fn = UNetPad(x, depth=depth)
        x_padded = pad_fn(x)
        
        print(f"Padded shape: {x_padded.shape}")
        print(f"Padding applied: {pad_fn.pad}")
        
        # Check divisibility
        h_padded, w_padded = x_padded.shape[-2:]
        h_divisible = h_padded % (2**depth) == 0
        w_divisible = w_padded % (2**depth) == 0
        
        print(f"H ({h_padded}) divisible by {2**depth}: {h_divisible}")
        print(f"W ({w_padded}) divisible by {2**depth}: {w_divisible}")
        
        # Test inverse operation
        x_unpadded = pad_fn.inverse(x_padded)
        print(f"Unpadded shape: {x_unpadded.shape}")
        
        # Check if original shape is recovered
        shapes_match = x.shape == x_unpadded.shape
        print(f"Original shape recovered: {shapes_match}")
        
        # Check if values are preserved (for the original region)
        values_match = torch.allclose(x, x_unpadded)
        print(f"Original values preserved: {values_match}")
        
        test_passed = h_divisible and w_divisible and shapes_match and values_match
        print(f"Test {i+1} PASSED: {test_passed}")
        
        if not test_passed:
            all_passed = False
            print(f"❌ FAILED!")
        else:
            print(f"✅ PASSED!")
    
    print(f"\n{'='*60}")
    print(f"OVERALL RESULT: {'PASSED' if all_passed else 'FAILED'}")
    print(f"{'='*60}")
    
    return all_passed


def test_unet_pad_fun_class():
    """Test the unet_pad_fun class for comparison."""
    print("\n" + "="*60)
    print("TESTING unet_pad_fun CLASS")
    print("="*60)
    
    # Test the problematic case
    shape = (1, 1, 101, 91)
    num_layers = 4
    
    x = torch.randn(shape)
    print(f"Original shape: {x.shape}")
    
    # Create unet_pad_fun instance
    pad_fn = unet_pad_fun(num_layers, x)
    print(f"Padding to apply: {pad_fn.padding}")
    
    # Apply padding
    x_padded = pad_fn.pad(x)
    print(f"Padded shape: {x_padded.shape}")
    
    # Check divisibility
    h_padded, w_padded = x_padded.shape[-2:]
    required_divisor = 2**num_layers
    h_divisible = h_padded % required_divisor == 0
    w_divisible = w_padded % required_divisor == 0
    
    print(f"H ({h_padded}) divisible by {required_divisor}: {h_divisible}")
    print(f"W ({w_padded}) divisible by {required_divisor}: {w_divisible}")
    
    # Test cropping
    try:
        x_cropped = pad_fn.crop(x_padded)
        print(f"Cropped shape: {x_cropped.shape}")
        
        shapes_match = x.shape == x_cropped.shape
        values_match = torch.allclose(x, x_cropped)
        
        print(f"Original shape recovered: {shapes_match}")
        print(f"Original values preserved: {values_match}")
        
        return h_divisible and w_divisible and shapes_match and values_match
        
    except Exception as e:
        print(f"❌ Error in cropping: {e}")
        return False


def test_concat_compatibility():
    """Test if padded tensors can be concatenated correctly (simulating UNet skip connections)."""
    print("\n" + "="*60)
    print("TESTING CONCATENATION COMPATIBILITY")
    print("="*60)
    
    # Simulate UNet encoder-decoder with skip connections
    original_shape = (1, 1, 101, 91)
    depth = 4
    
    x = torch.randn(original_shape)
    pad_fn = UNetPad(x, depth=depth)
    x_padded = pad_fn(x)
    
    print(f"Original shape: {x.shape}")
    print(f"Padded shape: {x_padded.shape}")
    
    # Simulate encoder path (downsampling)
    encoder_features = []
    current = x_padded
    
    for level in range(depth):
        print(f"Encoder level {level}: {current.shape}")
        encoder_features.append(current)
        
        # Simulate downsampling (divide by 2)
        if level < depth - 1:  # Don't downsample at the deepest level
            current = F.avg_pool2d(current, kernel_size=2, stride=2)
    
    # Simulate decoder path (upsampling + skip connections)
    current = encoder_features[-1]  # Start from deepest features
    
    for level in range(depth-2, -1, -1):  # Go back up
        # Upsample
        current = F.interpolate(current, scale_factor=2, mode='nearest')
        
        skip_connection = encoder_features[level]
        
        print(f"Decoder level {level}:")
        print(f"  Upsampled shape: {current.shape}")
        print(f"  Skip connection shape: {skip_connection.shape}")
        
        # Try concatenation
        try:
            concatenated = torch.cat([current, skip_connection], dim=1)
            print(f"  Concatenated shape: {concatenated.shape}")
            current = concatenated
        except RuntimeError as e:
            print(f"  ❌ Concatenation failed: {e}")
            return False
    
    print("✅ All concatenations successful!")
    return True


def test_edge_cases():
    """Test edge cases that might cause issues."""
    print("\n" + "="*60)
    print("TESTING EDGE CASES")
    print("="*60)
    
    edge_cases = [
        # (shape, depth, description)
        ((1, 1, 16, 16), 4, "Already perfectly divisible"),
        ((1, 1, 17, 17), 4, "Minimal padding needed"),
        ((1, 1, 1, 1), 4, "Very small input"),
        ((1, 1, 15, 16), 4, "One dimension needs padding"),
        ((1, 1, 32, 31), 4, "Large input, minimal padding"),
    ]
    
    all_passed = True
    
    for shape, depth, description in edge_cases:
        print(f"\nTesting: {description}")
        print(f"Shape: {shape}, Depth: {depth}")
        
        x = torch.randn(shape)
        pad_fn = UNetPad(x, depth=depth)
        x_padded = pad_fn(x)
        
        h, w = x_padded.shape[-2:]
        divisor = 2**depth
        
        h_ok = h % divisor == 0
        w_ok = w % divisor == 0
        
        print(f"Padded to {x_padded.shape}: H divisible={h_ok}, W divisible={w_ok}")
        
        if not (h_ok and w_ok):
            print(f"❌ FAILED: Not properly divisible")
            all_passed = False
        else:
            print(f"✅ PASSED")
    
    return all_passed


def debug_specific_error():
    """Debug the specific error case mentioned in the issue."""
    print("\n" + "="*60)
    print("DEBUGGING SPECIFIC ERROR CASE")
    print("="*60)
    
    print("Error message suggests tensor mismatch: Expected size 48 but got size 47")
    print("This typically happens during concatenation in skip connections.")
    
    # Try to reproduce the issue
    shape = (1, 1, 101, 91)
    depth = 4
    
    x = torch.randn(shape)
    pad_fn = UNetPad(x, depth=depth)
    x_padded = pad_fn(x)
    
    print(f"Original: {x.shape}")
    print(f"Padded: {x_padded.shape}")
    
    # Simulate the issue: downsampling and upsampling
    # This might create mismatched sizes
    
    h_padded, w_padded = x_padded.shape[-2:]
    print(f"Padded dimensions: H={h_padded}, W={w_padded}")
    
    # Check what happens after multiple downsampling operations
    current = x_padded
    for i in range(depth):
        print(f"Level {i}: {current.shape}")
        if i < depth - 1:
            current = F.avg_pool2d(current, kernel_size=2, stride=2)
    
    # Check if dimensions at each level are even (required for upsampling)
    current = x_padded
    for i in range(depth):
        h, w = current.shape[-2:]
        h_even = h % 2 == 0
        w_even = w % 2 == 0
        print(f"Level {i}: {current.shape} - H even: {h_even}, W even: {w_even}")
        
        if not (h_even and w_even) and i < depth - 1:
            print(f"❌ Problem at level {i}: Dimensions not even for downsampling!")
            return False
        
        if i < depth - 1:
            current = F.avg_pool2d(current, kernel_size=2, stride=2)
    
    return True


def main():
    """Run all tests."""
    print("Starting UNetPad debugging tests...")
    
    results = []
    
    # Run all tests
    results.append(("UNetPad Divisibility", test_unet_pad_divisibility()))
    results.append(("unet_pad_fun Class", test_unet_pad_fun_class()))
    results.append(("Concatenation Compatibility", test_concat_compatibility()))
    results.append(("Edge Cases", test_edge_cases()))
    results.append(("Specific Error Debug", debug_specific_error()))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*60)
    print(f"OVERALL: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    print("="*60)
    
    if not all_passed:
        print("\n🔍 DEBUGGING SUGGESTIONS:")
        print("1. Check if padding calculation is correct")
        print("2. Verify that all dimensions remain even after padding")
        print("3. Ensure upsampling/downsampling preserves proper dimensions")
        print("4. Check for off-by-one errors in padding calculations")


if __name__ == "__main__":
    main()
