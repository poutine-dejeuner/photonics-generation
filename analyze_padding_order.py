#!/usr/bin/env python3
"""
Deep dive into the unet_pad_fun padding/cropping logic.
"""

import torch
import torch.nn.functional as F


def analyze_padding_order():
    """Analyze the padding order issue in detail."""
    print("="*60)
    print("ANALYZING PADDING ORDER ISSUE")
    print("="*60)
    
    # Create a simple test tensor we can track
    x = torch.arange(12).reshape(1, 1, 3, 4).float()
    print("Original tensor:")
    print(x.squeeze())
    print(f"Shape: {x.shape}")
    
    # Let's understand what padding=[5, 6, 2, 3] should mean
    # According to the code: [left_w, right_w, left_h, right_h]
    padding = [5, 6, 2, 3]
    print(f"\nPadding values: {padding}")
    print("According to code: [left_w, right_w, left_h, right_h]")
    
    # Simulate the padding logic from unet_pad_fun
    n = x.ndim  # 4
    n = n - len(padding) // 2  # 4 - 2 = 2
    full_padding = [0, 0] * n + padding  # [0, 0, 0, 0, 5, 6, 2, 3]
    full_padding.reverse()  # [3, 2, 6, 5, 0, 0, 0, 0]
    
    print(f"Full padding for F.pad: {full_padding}")
    print("F.pad format: [left, right, top, bottom, front, back, ...]")
    print("So this means: left=3, right=2, top=6, bottom=5")
    
    x_padded = F.pad(x, full_padding, mode='constant', value=-1)
    print(f"\nPadded tensor:")
    print(x_padded.squeeze())
    print(f"Padded shape: {x_padded.shape}")
    
    # Now let's try the cropping
    a, b, c, d = padding[:4]  # 5, 6, 2, 3
    print(f"\nCropping with a={a}, b={b}, c={c}, d={d}")
    print(f"This means: x[..., {a}:-{b}, {c}:-{d}]")
    print("But wait - this seems wrong!")
    
    # The issue: padding order vs cropping order mismatch!
    print("\n❌ PROBLEM IDENTIFIED:")
    print("Padding order: [left_w, right_w, left_h, right_h]")
    print("But F.pad expects: [left, right, top, bottom]")
    print("And cropping uses: [h_start, h_end, w_start, w_end]")
    print("There's a dimension order mismatch!")


def test_corrected_logic():
    """Test the corrected padding/cropping logic."""
    print("\n" + "="*60)
    print("TESTING CORRECTED LOGIC")
    print("="*60)
    
    x = torch.arange(12).reshape(1, 1, 3, 4).float()
    print("Original tensor:")
    print(x.squeeze())
    
    # Let's fix the logic step by step
    h, w = x.shape[-2:]  # 3, 4
    N = 16  # 2^4
    
    # Calculate padding needed for each dimension
    h_pad_needed = (N - h % N) % N  # Padding needed for height
    w_pad_needed = (N - w % N) % N  # Padding needed for width
    
    print(f"\nDimensions: H={h}, W={w}")
    print(f"Target divisor: {N}")
    print(f"H padding needed: {h_pad_needed}")
    print(f"W padding needed: {w_pad_needed}")
    
    # Split padding symmetrically
    h_pad_left = h_pad_needed // 2
    h_pad_right = h_pad_needed - h_pad_left
    w_pad_left = w_pad_needed // 2
    w_pad_right = w_pad_needed - w_pad_left
    
    print(f"H padding: left={h_pad_left}, right={h_pad_right}")
    print(f"W padding: left={w_pad_left}, right={w_pad_right}")
    
    # Apply padding in correct F.pad format: [w_left, w_right, h_left, h_right]
    padding_for_f_pad = [w_pad_left, w_pad_right, h_pad_left, h_pad_right]
    x_padded = F.pad(x, padding_for_f_pad, mode='constant', value=-1)
    
    print(f"\nPadded tensor:")
    print(x_padded.squeeze())
    print(f"Padded shape: {x_padded.shape}")
    
    # Crop correctly
    h_slice = slice(h_pad_left, -h_pad_right if h_pad_right > 0 else None)
    w_slice = slice(w_pad_left, -w_pad_right if w_pad_right > 0 else None)
    
    x_cropped = x_padded[..., h_slice, w_slice]
    
    print(f"\nCropped tensor:")
    print(x_cropped.squeeze())
    print(f"Cropped shape: {x_cropped.shape}")
    
    # Check if values match
    values_match = torch.allclose(x, x_cropped)
    shapes_match = x.shape == x_cropped.shape
    
    print(f"\nShape recovered: {shapes_match}")
    print(f"Values preserved: {values_match}")
    
    return shapes_match and values_match


def main():
    """Run the analysis."""
    analyze_padding_order()
    success = test_corrected_logic()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if success:
        print("✅ Issue identified and fixed!")
        print("\nThe problem in unet_pad_fun:")
        print("1. Dimension order confusion between H/W and padding format")
        print("2. Incorrect cropping logic")
        print("\nCorrect approach:")
        print("1. Calculate padding for H and W separately")
        print("2. Use F.pad format: [w_left, w_right, h_left, h_right]")
        print("3. Crop using proper slice objects with None for zero padding")
    else:
        print("❌ Issue not fully resolved")


if __name__ == "__main__":
    main()
