#!/usr/bin/env python3
"""
Training script entry point for photo-gen package.
"""

import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import hydra
from omegaconf import DictConfig

from photo_gen.train import main as train_main


@hydra.main(version_base="1.1", config_path="../../config", config_name="comparison_config")
def main(cfg: DictConfig) -> None:
    """Main training entry point."""
    return train_main(cfg)


if __name__ == "__main__":
    main()
