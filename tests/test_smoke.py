"""M0 smoke tests: the C++ core builds, imports, and honours masks + noise weights."""

import delta
import numpy as np


def test_version():
    assert isinstance(delta.__version__, str)
    assert delta.__version__


def test_weighted_mean_excludes_masked_pixels():
    data = np.array([[1.0, 100.0], [3.0, 5.0]], dtype=np.float32)
    variance = np.ones((2, 2), dtype=np.float32)
    mask = np.array([[0, 1], [0, 0]], dtype=np.uint8)  # flag the outlier 100
    # Equal weights, masked pixel excluded -> mean of {1, 3, 5} == 3.
    assert abs(delta.weighted_mean(data, variance, mask) - 3.0) < 1e-6


def test_weighted_mean_uses_inverse_variance():
    data = np.array([[1.0, 3.0]], dtype=np.float32)
    variance = np.array([[1.0, 3.0]], dtype=np.float32)  # down-weight 2nd pixel
    mask = np.zeros((1, 2), dtype=np.uint8)
    # (1*1 + (1/3)*3) / (1 + 1/3) == 2 / (4/3) == 1.5
    assert abs(delta.weighted_mean(data, variance, mask) - 1.5) < 1e-6
