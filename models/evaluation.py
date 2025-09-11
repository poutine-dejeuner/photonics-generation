"""
Configurable evaluation functions for the model comparison framework.
"""
import numpy as np
from typing import Callable

from evaluation.meep_compute_fom import compute_FOM_parallele


def debug_fom_evaluation(images: np.ndarray) -> np.ndarray:
    return np.random.random(len(images))


def meep_fom_evaluation(images: np.ndarray, debug: bool= False) -> np.ndarray:
    """
    Physics-based FOM evaluation using MEEP.
    
    Args:
        images: Generated images array of shape (n_samples, height, width)
        
    Returns:
        MEEP-computed FOM values of shape (n_samples,)
    """
    return compute_FOM_parallele(images)


def get_evaluation_function(cfg) -> Callable[[np.ndarray], np.ndarray]:
    """
    Get the evaluation function based on config.
    
    Args:
        cfg: Hydra config with evaluation settings
        
    Returns:
        Evaluation function
    """
    if hasattr(cfg, 'evaluation') and cfg.evaluation is not None:
        # Use Hydra instantiation if evaluation config is provided
        try:
            import hydra
            # The target should be a function, not a function call
            target = cfg.evaluation._target_
            if target == 'models.evaluation.debug_fom_evaluation':
                return debug_fom_evaluation
            elif target == 'models.evaluation.meep_fom_evaluation':
                return meep_fom_evaluation
            else:
                # Try to instantiate as a callable
                return hydra.utils.instantiate(cfg.evaluation)
        except Exception as e:
            print(f"Warning: Failed to instantiate evaluation function: {e}")
            print("Falling back to debug evaluation")
            return debug_fom_evaluation
    elif hasattr(cfg, 'debug') and cfg.debug:
        # Fallback to debug evaluation
        return debug_fom_evaluation
    else:
        # Fallback to MEEP evaluation
        return meep_fom_evaluation
