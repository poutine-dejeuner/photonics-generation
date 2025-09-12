#!/usr/bin/env python3
"""
Inference script entry point for photo-gen package.
"""

import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import hydra
from omegaconf import DictConfig

from photo_gen.inference import main as inference_main


@hydra.main(config_path="../../config", config_name="inference")
def main(cfg: DictConfig) -> None:
    """Main inference entry point."""
    return inference_main(cfg)


if __name__ == "__main__":
    main()