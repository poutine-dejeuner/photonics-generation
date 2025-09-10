"""
Pytest configuration and fixtures for UNet padding tests.
"""

import pytest
import torch
import sys
import os

# Add the project directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


@pytest.fixture
def sample_tensor_small():
    """Small test tensor for quick tests."""
    return torch.randn(1, 1, 101, 91)


@pytest.fixture
def sample_tensor_large():
    """Larger test tensor for stress tests."""
    return torch.randn(4, 3, 256, 256)


@pytest.fixture
def sample_tensor_square():
    """Square tensor that's already properly divisible."""
    return torch.randn(1, 1, 64, 64)


@pytest.fixture(params=[1, 3, 4, 5])
def depth_values(request):
    """Parametrized fixture for different depth values."""
    return request.param


@pytest.fixture(params=[
    (1, 1, 101, 91),
    (1, 1, 47, 48),
    (2, 3, 64, 64),
    (1, 1, 99, 97),
])
def various_shapes(request):
    """Parametrized fixture for various tensor shapes."""
    return torch.randn(request.param)


@pytest.fixture
def problematic_case():
    """The specific case that was causing issues."""
    return torch.randn(1, 1, 101, 91)


# Configure torch for deterministic testing
@pytest.fixture(autouse=True)
def set_random_seed():
    """Set random seed for reproducible tests."""
    torch.manual_seed(42)
    yield
    # Cleanup if needed
