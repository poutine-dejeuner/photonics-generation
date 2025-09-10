#!/usr/bin/env python3
"""
Test to demonstrate and fix the unet_pad_fun cropping bug.
"""

import torch
import torch.nn.functional as F


def test_cropping_bug():
    """Demonstrate the cropping bug in unet_pad_fun."""
    print("="*60)
    print("TESTING CROPPING BUG IN unet_pad_fun")
    print("="*60)
    
    # Create test tensor
    x = torch.randn(1, 1, 101, 91)
    print(f"Original shape: {x.shape}")
    
    # Test case that causes the bug
    padding = [5, 6, 2, 3]  # [left_w, right_w, left_h, right_h]
    print(f"Padding: {padding}")
    
    # Apply padding (simulating the pad method)
    n = x.ndim
    n = n - len(padding) // 2
    full_padding = [0, 0]*n + padding
    full_padding.reverse()
    print(f"Full padding for F.pad: {full_padding}")
    
    x_padded = F.pad(x, full_padding, mode='constant', value=0)
    print(f"Padded shape: {x_padded.shape}")
    
    # Try the buggy cropping
    print("\n--- BUGGY CROPPING ---")
    a, b, c, d = padding[:4]
    print(f"Crop parameters: a={a}, b={b}, c={c}, d={d}")
    print(f"Trying: x[..., {a}:-{b}, {c}:-{d}]")
    
    try:
        if b == 0 or d == 0:
            print(f"❌ PROBLEM: Using :-0 indexing!")
            print(f":-{b} = :-0" if b == 0 else f":-{b} works")
            print(f":-{d} = :-0" if d == 0 else f":-{d} works")
        
        cropped_buggy = x_padded[..., a:-b if b != 0 else None, c:-d if d != 0 else None]
        print(f"Cropped shape (manual fix): {cropped_buggy.shape}")
        
        # This is what the original code does (and it's wrong when b or d is 0)
        if b != 0 and d != 0:
            cropped_original = x_padded[..., a:-b, c:-d]
            print(f"Original cropping would work: {cropped_original.shape}")
        else:
            print("Original cropping would fail due to :-0 indexing")
            
    except Exception as e:
        print(f"❌ Cropping failed: {e}")
    
    # Test if values are preserved
    if 'cropped_buggy' in locals():
        values_match = torch.allclose(x, cropped_buggy)
        print(f"Values preserved: {values_match}")
        if not values_match:
            print(f"Expected shape: {x.shape}, got: {cropped_buggy.shape}")


def test_fixed_cropping():
    """Test the fixed cropping logic."""
    print("\n" + "="*60)
    print("TESTING FIXED CROPPING LOGIC")
    print("="*60)
    
    test_cases = [
        ([5, 6, 2, 3], "Normal case"),
        ([0, 5, 2, 3], "Zero left padding"),
        ([5, 0, 2, 3], "Zero right padding"),
        ([5, 6, 0, 3], "Zero top padding"),
        ([5, 6, 2, 0], "Zero bottom padding"),
        ([0, 0, 0, 0], "No padding"),
    ]
    
    x = torch.randn(1, 1, 101, 91)
    
    for padding, description in test_cases:
        print(f"\n--- {description}: {padding} ---")
        
        # Apply padding
        n = x.ndim
        n = n - len(padding) // 2
        full_padding = [0, 0]*n + padding
        full_padding.reverse()
        
        x_padded = F.pad(x, full_padding, mode='constant', value=0)
        print(f"Padded: {x.shape} -> {x_padded.shape}")
        
        # Fixed cropping logic
        a, b, c, d = padding[:4]
        
        # Handle zero padding correctly
        h_slice = slice(a, -b if b != 0 else None)
        w_slice = slice(c, -d if d != 0 else None)
        
        cropped = x_padded[..., h_slice, w_slice]
        print(f"Cropped: {x_padded.shape} -> {cropped.shape}")
        
        # Check if original is recovered
        shape_match = x.shape == cropped.shape
        value_match = torch.allclose(x, cropped)
        
        print(f"Shape recovered: {shape_match}")
        print(f"Values preserved: {value_match}")
        print(f"✅ PASSED" if shape_match and value_match else "❌ FAILED")


def main():
    """Run the cropping bug tests."""
    test_cropping_bug()
    test_fixed_cropping()
    
    print("\n" + "="*60)
    print("CONCLUSION")
    print("="*60)
    print("The bug is in the crop method of unet_pad_fun:")
    print("Using x[..., a:-b, c:-d] fails when b=0 or d=0")
    print("because Python interprets :-0 as 'everything except nothing'")
    print("which is different from 'everything'.")
    print("\nFix: Use slice objects with proper None handling.")


if __name__ == "__main__":
    main()
