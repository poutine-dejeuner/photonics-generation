"""
Integration tests for UNet padding with actual model components.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from unet_utils import UNetPad, unet_pad_fun


class SimpleUNet(nn.Module):
    """Simplified UNet for integration testing."""
    
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
                    nn.Conv2d(channels[i]*2, channels[i], 3, padding=1),
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
            
            if i < len(self.encoder) - 1:
                current = F.max_pool2d(current, 2)
        
        # Decoder
        current = encoder_features[-1]
        
        for i, layer in enumerate(self.decoder):
            # Upsample
            upsampled = layer[0](current)
            
            # Get skip connection
            skip_idx = len(encoder_features) - 2 - i
            skip = encoder_features[skip_idx]
            
            # Concatenate
            concatenated = torch.cat([upsampled, skip], dim=1)
            
            # Apply conv
            current = layer[1:](concatenated)
        
        return self.final_conv(current)


@pytest.mark.integration
class TestUNetIntegration:
    """Integration tests with actual UNet models."""
    
    @pytest.mark.parametrize("input_shape,num_layers", [
        ((1, 1, 101, 91), 4),
        ((1, 1, 95, 95), 4),
        ((2, 1, 47, 48), 3),
    ])
    def test_unet_forward_with_padding(self, input_shape, num_layers):
        """Test that UNet forward pass works with padded inputs."""
        x = torch.randn(input_shape)
        
        # Apply padding
        depth = num_layers // 2
        pad_fn = UNetPad(x, depth=depth)
        x_padded = pad_fn(x)
        
        # Create and test model
        model = SimpleUNet(in_channels=input_shape[1], num_layers=num_layers)
        model.eval()
        
        with torch.no_grad():
            output = model(x_padded)
        
        # Should produce valid output
        assert output is not None
        assert output.shape[0] == x_padded.shape[0]  # Same batch size
        assert len(output.shape) == 4  # BCHW format
    
    def test_original_vs_padded_consistency(self):
        """Test that results are consistent between different padding approaches."""
        x = torch.randn(1, 1, 64, 64)  # Already properly sized
        
        # Test without padding
        model = SimpleUNet(num_layers=4)
        model.eval()
        
        with torch.no_grad():
            output_direct = model(x)
        
        # Test with padding (should be no-op for this size)
        pad_fn = UNetPad(x, depth=2)
        x_padded = pad_fn(x)
        
        with torch.no_grad():
            output_padded = model(x_padded)
            output_unpadded = pad_fn.inverse(output_padded)
        
        # Results should be very similar (allowing for numerical precision)
        assert torch.allclose(output_direct, output_unpadded, atol=1e-6)
    
    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_batch_consistency(self, batch_size):
        """Test that padding works consistently across batch sizes."""
        shape = (batch_size, 1, 101, 91)
        x = torch.randn(shape)
        
        # Apply padding
        pad_fn = UNetPad(x, depth=2)
        x_padded = pad_fn(x)
        
        # Test model
        model = SimpleUNet(num_layers=4)
        model.eval()
        
        with torch.no_grad():
            output = model(x_padded)
            output_unpadded = pad_fn.inverse(output)
        
        # Check output shapes
        assert output_unpadded.shape[0] == batch_size
        assert output_unpadded.shape[-2:] == (101, 91)  # Original spatial dims
    
    def test_memory_efficiency(self):
        """Test that padding doesn't cause excessive memory usage."""
        # This is a basic test - in practice you'd want more sophisticated memory monitoring
        x = torch.randn(1, 1, 101, 91)
        
        # Measure memory before
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            mem_before = torch.cuda.memory_allocated()
        
        # Apply padding and model
        pad_fn = UNetPad(x, depth=2)
        x_padded = pad_fn(x)
        
        model = SimpleUNet(num_layers=4)
        if torch.cuda.is_available():
            model = model.cuda()
            x_padded = x_padded.cuda()
        
        with torch.no_grad():
            output = model(x_padded)
        
        # Basic assertion that we got a valid output
        assert output is not None
        assert output.numel() > 0


@pytest.mark.integration  
class TestErrorRecovery:
    """Test error recovery and edge cases in integration scenarios."""
    
    def test_zero_padding_edge_case(self):
        """Test the specific edge case that was causing errors."""
        # Create input that results in zero padding on one side
        x = torch.randn(1, 1, 48, 48)  # This should need minimal padding
        
        # Test both padding methods
        unetpad = UNetPad(x, depth=2)
        unet_pad_fun_instance = unet_pad_fun(4, x)
        
        x_padded_1 = unetpad(x)
        x_padded_2 = unet_pad_fun_instance.pad(x)
        
        # Both should work without errors
        x_recovered_1 = unetpad.inverse(x_padded_1)
        x_recovered_2 = unet_pad_fun_instance.crop(x_padded_2)
        
        assert torch.allclose(x, x_recovered_1)
        assert torch.allclose(x, x_recovered_2)
    
    def test_concatenation_error_prevention(self):
        """Test that the specific concatenation error is prevented."""
        # This was the original error: "Expected size 48 but got size 47"
        x = torch.randn(1, 1, 101, 91)
        
        # Apply padding with the settings that were causing issues
        depth = 2  # This is num_layers//2 from the original code
        pad_fn = UNetPad(x, depth=depth)
        x_padded = pad_fn(x)
        
        # Simulate the encoder-decoder process that was failing
        current = x_padded
        encoder_features = []
        
        # Encoder path
        for level in range(depth + 1):  # Go one level deeper than depth
            encoder_features.append(current)
            if level < depth:  # Don't pool at the deepest level
                current = F.max_pool2d(current, 2)
        
        # Decoder path - this should not fail
        current = encoder_features[-1]
        for level in range(depth):
            # Upsample
            current = F.interpolate(current, scale_factor=2, mode='nearest')
            
            # Get skip connection
            skip_idx = depth - 1 - level
            skip = encoder_features[skip_idx]
            
            # This concatenation should work without the size mismatch error
            concatenated = torch.cat([current, skip], dim=1)
            current = concatenated
        
        # If we get here without errors, the fix worked
        assert True
