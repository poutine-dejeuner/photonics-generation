"""
Fixed UNet implementation that properly handles channel dimensions in decoder.

The main issues in the original UNet were:
1. Embeddings not recalculated in decoder loop
2. Channel mismatch after concatenation - ResBlocks expect original channel count but get doubled channels
3. No mechanism to reduce concatenated channels before passing to ResBlocks

This fixed version adds channel reduction convolutions after concatenation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List
from torch import device


class ResBlock(nn.Module):
    def __init__(self, C: int, num_groups: int, dropout_prob: float):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.gnorm1 = nn.GroupNorm(num_groups=num_groups, num_channels=C)
        self.gnorm2 = nn.GroupNorm(num_groups=num_groups, num_channels=C)
        self.conv1 = nn.Conv2d(C, C, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(C, C, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(p=dropout_prob, inplace=True)

    def forward(self, x, embeddings):
        x = x + embeddings[:, :x.shape[1], :, :]
        r = self.conv1(self.relu(self.gnorm1(x)))
        r = self.dropout(r)
        r = self.conv2(self.relu(self.gnorm2(r)))
        return r + x


class Attention(nn.Module):
    def __init__(self, C: int, num_heads: int, dropout_prob: float):
        super().__init__()
        self.proj1 = nn.Linear(C, C*3)
        self.proj2 = nn.Linear(C, C)
        self.num_heads = num_heads
        self.dropout_prob = dropout_prob

    def forward(self, x):
        h, w = x.shape[2:]
        x = x.reshape(x.shape[0], x.shape[1], h*w).transpose(1, 2)
        x_norm = F.layer_norm(x, (x.shape[-1],))
        q, k, v = self.proj1(x_norm).chunk(3, dim=-1)
        
        q = q.view(x.shape[0], h*w, self.num_heads, -1).transpose(1, 2)
        k = k.view(x.shape[0], h*w, self.num_heads, -1).transpose(1, 2)
        v = v.view(x.shape[0], h*w, self.num_heads, -1).transpose(1, 2)
        
        att = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
        att = F.softmax(att, dim=-1)
        att = F.dropout(att, p=self.dropout_prob, training=self.training)
        
        x = (att @ v).transpose(1, 2).reshape(x.shape[0], h*w, -1)
        x = self.proj2(x)
        return (x + x_norm).transpose(1, 2).reshape(x.shape[0], -1, h, w)


class UnetLayer(nn.Module):
    def __init__(self, 
            upscale: bool, 
            attention: bool, 
            num_groups: int, 
            dropout_prob: float,
            num_heads: int,
            C: int):
        super().__init__()
        self.ResBlock1 = ResBlock(C=C, num_groups=num_groups, dropout_prob=dropout_prob)
        self.ResBlock2 = ResBlock(C=C, num_groups=num_groups, dropout_prob=dropout_prob)
        if upscale:
            self.conv = nn.ConvTranspose2d(C, C//2, kernel_size=4, stride=2, padding=1)
        else:
            self.conv = nn.Conv2d(C, C*2, kernel_size=3, stride=2, padding=1)
        if attention:
            self.attention_layer = Attention(C, num_heads=num_heads, dropout_prob=dropout_prob)

    def forward(self, x, embeddings):
        x = self.ResBlock1(x, embeddings)
        if hasattr(self, 'attention_layer'):
            x = self.attention_layer(x)
        x = self.ResBlock2(x, embeddings)
        return self.conv(x), x


class SinusoidalEmbeddings(nn.Module):
    def __init__(self, time_steps: int, embed_dim: int, device: device):
        super().__init__()
        position = torch.arange(time_steps).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim))
        embeddings = torch.zeros(time_steps, embed_dim, requires_grad=False)
        embeddings[:, 0::2] = torch.sin(position * div)
        embeddings[:, 1::2] = torch.cos(position * div)
        self.embeddings = embeddings.to(device)

    def forward(self, t):
        embeds = self.embeddings[t]
        return embeds[:, :, None, None]


class UNET_Fixed(nn.Module):
    """Fixed UNet implementation that properly handles channel dimensions."""
    
    def __init__(self,
            Channels: List = [64, 128, 256, 512, 512, 384],
            Attentions: List = [False, True, False, False, False, True],
            Upscales: List = [False, False, False, True, True, True],
            num_groups: int = 32,
            dropout_prob: float = 0.0,
            num_heads: int = 8,
            input_channels: int = 1,
            output_channels: int = 1,
            device: device = 'cuda',
            time_steps: int = 1000,
            **kwargs):
        super().__init__()
        self.num_layers = len(Channels)
        self.shallow_conv = nn.Conv2d(input_channels, Channels[0], kernel_size=3, padding=1)
        
        # Create layers
        for i in range(self.num_layers):
            layer = UnetLayer(
                upscale=Upscales[i],
                attention=Attentions[i],
                num_groups=num_groups,
                dropout_prob=dropout_prob,
                C=Channels[i],
                num_heads=num_heads
            )
            setattr(self, f'Layer{i+1}', layer)
        
        # FIX: Add channel reduction convolutions for decoder concatenations
        # These reduce the doubled channels after concatenation back to expected size
        for i in range(self.num_layers//2, self.num_layers):
            decoder_idx = i + 1
            if i < self.num_layers - 1:  # Not the last layer
                # After concatenation: upsampled_channels + residual_channels -> target_channels
                encoder_layer_idx = self.num_layers - i - 1
                upsampled_channels = Channels[i] // 2 if Upscales[i] else Channels[i]
                residual_channels = Channels[encoder_layer_idx]
                concatenated_channels = upsampled_channels + residual_channels
                target_channels = Channels[i]
                
                channel_reducer = nn.Conv2d(concatenated_channels, target_channels, kernel_size=1)
                setattr(self, f'channel_reducer_{decoder_idx}', channel_reducer)
        
        # Output layers
        out_channels = (Channels[-1]//2) + Channels[0]
        self.late_conv = nn.Conv2d(out_channels, out_channels//2, kernel_size=3, padding=1)
        self.output_conv = nn.Conv2d(out_channels//2, output_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        
        # Embeddings
        self.embeddings = SinusoidalEmbeddings(time_steps=time_steps, embed_dim=2*max(Channels), device=device)

    def forward(self, x, t):
        x = self.shallow_conv(x)
        residuals = []
        
        # Encoder (compression) - first num_layers//2 layers
        for i in range(self.num_layers//2):
            layer = getattr(self, f'Layer{i+1}')
            embeddings = self.embeddings(t)  # Recalculate embeddings
            x, r = layer(x, embeddings)
            residuals.append(r)
        
        # Decoder (decompression) - last num_layers//2 layers
        for i in range(self.num_layers//2, self.num_layers):
            layer = getattr(self, f'Layer{i+1}')
            embeddings = self.embeddings(t)  # FIX: Recalculate embeddings for decoder
            
            upsampled, _ = layer(x, embeddings)
            
            # Get residual connection with correct indexing
            residual_idx = self.num_layers - i - 1
            residual = residuals[residual_idx]
            
            # Handle spatial dimension mismatch if needed
            if upsampled.shape[2:] != residual.shape[2:]:
                residual = F.interpolate(residual, size=upsampled.shape[2:], mode='bilinear', align_corners=False)
            
            # Concatenate
            x = torch.cat([upsampled, residual], dim=1)
            
            # FIX: Reduce channels after concatenation (except for last layer)
            if i < self.num_layers - 1:
                channel_reducer = getattr(self, f'channel_reducer_{i+1}')
                x = channel_reducer(x)
        
        # Final output layers
        return self.output_conv(self.relu(self.late_conv(x)))


def test_fixed_unet():
    """Test the fixed UNet implementation."""
    print("Testing Fixed UNet Implementation")
    print("="*50)
    
    # Create fixed model
    model = UNET_Fixed(
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
    
    # Test input
    x = torch.randn(1, 1, 101, 91)
    t = torch.randint(0, 1000, (1,))
    
    # Apply padding  
    from unet_utils import UNetPad
    pad_fn = UNetPad(x, depth=model.num_layers//2)
    x_padded = pad_fn(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Padded shape: {x_padded.shape}")
    
    try:
        with torch.no_grad():
            output = model(x_padded, t)
        print(f"✅ Forward pass successful!")
        print(f"Output shape: {output.shape}")
        
        # Verify output dimensions
        expected_output_shape = x.shape  # Should match input
        if output.shape[2:] == expected_output_shape[2:]:
            print(f"✅ Output spatial dimensions correct: {output.shape[2:]}")
        else:
            print(f"❌ Output spatial dimensions incorrect: got {output.shape[2:]}, expected {expected_output_shape[2:]}")
            
        return True
        
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_fixed_unet()
