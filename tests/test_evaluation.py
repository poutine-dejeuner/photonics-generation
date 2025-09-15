import numpy as np
import pytest

from photo_gen.evaluation.evaluation import VisualizeGeneratedSamples


@pytest.mark.parametrize("shape", [
    (4, 1, 100, 100),
    (4, 100, 100),
    (1, 3, 64, 64),
    (8, 28, 28),
])
def test_visualize_generated_samples_shapes(shape):
    samples = np.random.rand(*shape)
    vis = VisualizeGeneratedSamples(shape[0])
    result = vis(samples)
    assert result is None or result is True  # Accepts None or True as valid
