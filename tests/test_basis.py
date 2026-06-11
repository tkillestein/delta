"""M2 tests: Gauss-Hermite basis generation and separable convolution.

The convolution is validated against an independent, direct (non-separable)
NumPy reference using the full 2-D kernels.
"""

import delta
import numpy as np


def ref_convolve2d(img, kernel):
    """Direct 'same'-size, zero-padded true convolution (kernel mirrored)."""
    ks = kernel.shape[0]
    h = ks // 2
    height, width = img.shape
    padded = np.pad(img.astype(np.float64), h)
    out = np.zeros((height, width), dtype=np.float64)
    for a in range(ks):  # a = n + h
        n = a - h
        for b in range(ks):  # b = m + h
            m = b - h
            out += kernel[a, b] * padded[h - n : h - n + height, h - m : h - m + width]
    return out


def n_components(n_max):
    return (n_max + 1) * (n_max + 2) // 2


def test_shapes_and_counts():
    beta, n_max, radius = 2.0, 3, 6
    ks = 2 * radius + 1

    b1d = delta.gauss_hermite_basis1d(beta, n_max, radius)
    assert b1d.shape == (n_max + 1, ks)

    orders, kernels = delta.gauss_hermite_kernels(beta, n_max, radius)
    assert orders.shape == (n_components(n_max), 2)
    assert kernels.shape == (n_components(n_max), ks, ks)
    # Total order never exceeds n_max.
    assert orders.sum(axis=1).max() <= n_max


def test_kernels_are_separable_outer_products():
    beta, n_max, radius = 2.5, 4, 8
    b1d = delta.gauss_hermite_basis1d(beta, n_max, radius)
    orders, kernels = delta.gauss_hermite_kernels(beta, n_max, radius)

    for (nx, ny), kernel in zip(orders, kernels, strict=True):
        expected = np.outer(b1d[ny], b1d[nx])  # row = y (ny), col = x (nx)
        np.testing.assert_allclose(kernel, expected, rtol=1e-5, atol=1e-6)


def test_1d_basis_near_orthonormal():
    # On a unit grid with a wide enough footprint the discrete inner product
    # approximates the continuum orthonormality integral.
    beta, n_max, radius = 4.0, 4, 24
    b1d = delta.gauss_hermite_basis1d(beta, n_max, radius)
    gram = b1d @ b1d.T  # (n_max+1, n_max+1)
    np.testing.assert_allclose(gram, np.eye(n_max + 1), atol=1e-2)


def test_basis_convolve_matches_direct_reference():
    beta, n_max, radius = 2.0, 2, 5
    rng = np.random.default_rng(42)
    image = rng.standard_normal((13, 17)).astype(np.float32)

    stack = delta.basis_convolve(image, beta, n_max, radius)
    _, kernels = delta.gauss_hermite_kernels(beta, n_max, radius)
    assert stack.shape == (n_components(n_max), *image.shape)

    for component, kernel in zip(stack, kernels, strict=True):
        expected = ref_convolve2d(image, kernel.astype(np.float64))
        np.testing.assert_allclose(component, expected, rtol=1e-4, atol=1e-4)
