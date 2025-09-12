"""
Photo-Gen Evaluation Package

This package contains evaluation functions and metrics for model performance:
- FOM computation
- Model evaluation utilities
- Metric calculations
"""

from .meep_compute_fom import compute_FOM_parallele
from .evaluation import evaluation

__all__ = [
    "compute_FOM_parallele",
    "evaluation",
]