#!/usr/bin/env python3
"""
Test script to verify debug configuration loading works correctly.
"""
import os
import sys
import tempfile
from pathlib import Path

# Add the photo_gen directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Import hydra and omegaconf
import hydra
from omegaconf import OmegaConf

def test_debug_config_loading():
    """Test that debug configurations are loaded correctly."""
    
    # Get absolute path to config directory
    config_dir = Path(__file__).parent / "photo_gen" / "config"
    
    # Initialize hydra with absolute config directory path
    with hydra.initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
        # Test normal mode
        cfg_normal = hydra.compose(config_name="comparison_config")
        print(f"Normal mode - n_to_generate: {cfg_normal.n_to_generate}")
        
        # Test debug mode
        cfg_debug = hydra.compose(config_name="comparison_config", overrides=["debug=True"])
        print(f"Debug mode initial - n_to_generate: {cfg_debug.n_to_generate}")
        
        # Apply debug configurations manually (as would happen in main.py)
        if cfg_debug.debug and hasattr(cfg_debug, 'debug_config'):
            for key, value in cfg_debug.debug_config.items():
                if hasattr(cfg_debug, key):
                    setattr(cfg_debug, key, value)
                    print(f"Debug mode: Setting {key} = {value}")
        
        print(f"Debug mode final - n_to_generate: {cfg_debug.n_to_generate}")
        
        # Verify the change happened
        assert cfg_debug.n_to_generate == 16, f"Expected 16, got {cfg_debug.n_to_generate}"
        print("✓ Debug configuration loading test passed!")

if __name__ == "__main__":
    test_debug_config_loading()