"""
Analysis and fix for UNet model bugs.

This script identifies and documents the bugs in the UNet model:
1. Embeddings not recalculated in decoder loop
2. Dimension mismatch after concatenation in decoder
3. Incorrect concatenation logic
"""

import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.unet import UNET
from unet_utils import UNetPad


def analyze_unet_bugs():
    """Analyze and demonstrate the UNet bugs."""
    print("="*60)
    print("ANALYZING UNET MODEL BUGS")
    print("="*60)
    
    # Create a UNet model
    model = UNET(
        Channels=[64, 128, 256, 128],  # Simplified architecture
        Attentions=[False, False, False, False],
        Upscales=[False, False, True, True],
        num_groups=8,
        dropout_prob=0.0,
        num_heads=4,
        input_channels=1,
        output_channels=1,
        device='cpu',
        time_steps=1000
    )
    
    print(f"Model architecture:")
    print(f"Channels: {[64, 128, 256, 128]}")
    print(f"Upscales: {[False, False, True, True]}")
    print(f"num_layers: {model.num_layers}")
    print(f"num_layers//2: {model.num_layers//2}")
    
    # Test input
    x = torch.randn(1, 1, 101, 91)
    t = torch.randint(0, 1000, (1,))
    
    # Apply padding
    pad_fn = UNetPad(x, depth=model.num_layers//2)
    x_padded = pad_fn(x)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Padded shape: {x_padded.shape}")
    
    # Manually trace through the forward pass to identify bugs
    print(f"\n--- TRACING FORWARD PASS ---")
    
    # Shallow conv
    x_current = model.shallow_conv(x_padded)
    print(f"After shallow_conv: {x_current.shape}")
    
    residuals = []
    
    # Encoder (compression) - first num_layers//2 layers
    print(f"\n--- ENCODER (layers 1 to {model.num_layers//2}) ---")
    for i in range(model.num_layers//2):
        layer = getattr(model, f'Layer{i+1}')
        embeddings = model.embeddings(t)
        print(f"Layer {i+1} embeddings shape: {embeddings.shape}")
        print(f"Layer {i+1} input shape: {x_current.shape}")
        
        # Check layer configuration
        print(f"Layer {i+1} upscale: {layer.conv}")
        
        x_current, r = layer(x_current, embeddings)
        residuals.append(r)
        
        print(f"Layer {i+1} output shape: {x_current.shape}")
        print(f"Layer {i+1} residual shape: {r.shape}")
        print()
    
    # Decoder (decompression) - last num_layers//2 layers
    print(f"--- DECODER (layers {model.num_layers//2 + 1} to {model.num_layers}) ---")
    
    for i in range(model.num_layers//2, model.num_layers):
        layer = getattr(model, f'Layer{i+1}')
        
        print(f"Layer {i+1} input shape: {x_current.shape}")
        
        # BUG 1: embeddings not recalculated - using old embeddings from encoder!
        # This should be: embeddings = model.embeddings(t)
        
        layer_output, _ = layer(x_current, embeddings)  # Using old embeddings!
        print(f"Layer {i+1} output shape: {layer_output.shape}")
        
        # Get residual connection
        residual_idx = model.num_layers - i - 1
        residual = residuals[residual_idx]
        print(f"Connecting to residual {residual_idx} with shape: {residual.shape}")
        
        # BUG 2: This concatenation may have dimension mismatch
        try:
            x_current = torch.concat((layer_output, residual), dim=1)
            print(f"After concatenation: {x_current.shape}")
        except RuntimeError as e:
            print(f"❌ CONCATENATION ERROR: {e}")
            print(f"   layer_output: {layer_output.shape}")
            print(f"   residual: {residual.shape}")
            return False
        
        print()
    
    print("✅ Forward pass completed successfully")
    return True


def demonstrate_bugs_with_original_code():
    """Show how the original code fails."""
    print("\n" + "="*60)
    print("DEMONSTRATING BUGS WITH ORIGINAL CODE")
    print("="*60)
    
    model = UNET(
        Channels=[64, 128, 256, 128],
        Attentions=[False, False, False, False],
        Upscales=[False, False, True, True],
        num_groups=8,
        dropout_prob=0.0,
        num_heads=4,
        input_channels=1,
        output_channels=1,
        device='cpu',
        time_steps=1000
    )
    
    x = torch.randn(1, 1, 101, 91)
    t = torch.randint(0, 1000, (1,))
    
    pad_fn = UNetPad(x, depth=model.num_layers//2)
    x_padded = pad_fn(x)
    
    try:
        with torch.no_grad():
            output = model(x_padded, t)
        print("✅ Model forward pass successful")
        return True
    except Exception as e:
        print(f"❌ Model forward pass failed: {e}")
        return False


def show_fixed_forward_logic():
    """Show the corrected forward pass logic."""
    print("\n" + "="*60)
    print("CORRECTED FORWARD PASS LOGIC")
    print("="*60)
    
    print("The bugs in the original forward method:")
    print("1. Line 143: embeddings not recalculated in decoder loop")
    print("2. Line 143: concatenation after layer() call without checking dimensions")
    print("3. Line 146: wrong indexing for residual connections")
    
    print("\nCorrected logic should be:")
    print("""
# Encoder
for i in range(self.num_layers//2):
    layer = getattr(self, f'Layer{i+1}')
    embeddings = self.embeddings(t)  # Recalculate embeddings
    x, r = layer(x, embeddings)
    residuals.append(r)

# Decoder  
for i in range(self.num_layers//2, self.num_layers):
    layer = getattr(self, f'Layer{i+1}')
    embeddings = self.embeddings(t)  # ← FIX: Recalculate embeddings!
    upsampled, _ = layer(x, embeddings)
    
    # Get correct residual
    residual_idx = self.num_layers - i - 1
    residual = residuals[residual_idx]
    
    # ← FIX: Check dimensions before concatenation
    if upsampled.shape[2:] != residual.shape[2:]:
        # Handle dimension mismatch
        residual = F.interpolate(residual, size=upsampled.shape[2:])
    
    x = torch.cat([upsampled, residual], dim=1)
""")


def main():
    """Run the analysis."""
    print("UNet Model Bug Analysis")
    print("This script analyzes the bugs causing the tensor dimension mismatch.")
    
    # Run analysis
    success = analyze_unet_bugs()
    
    # Try with original code
    original_success = demonstrate_bugs_with_original_code()
    
    # Show fixes
    show_fixed_forward_logic()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Manual analysis success: {success}")
    print(f"Original model success: {original_success}")
    
    if not original_success:
        print("\n🔧 FIXES NEEDED:")
        print("1. Recalculate embeddings in decoder loop")
        print("2. Add dimension checking before concatenation")
        print("3. Handle spatial dimension mismatches with interpolation")
        print("4. Fix residual indexing logic")


if __name__ == "__main__":
    main()
