"""Convolution-engine tests (SPEC §3.2): the separable/SIMD B_n = phi_n ⊗ R pass.

These exercise ``basis_convolve`` directly. Elsewhere the engine is only covered
transitively (via basis/subtract/fit); here we pin its impulse response against the
exact Gauss-Hermite component kernels and check linearity.
"""

import delta._core as core
import numpy as np

BETA = 2.0
N_MAX = 3
RADIUS = 6


def test_basis_convolve_shape_and_finite():
    rng = np.random.default_rng(0)
    img = rng.standard_normal((24, 30)).astype(np.float32)
    bn = core.basis_convolve(img, BETA, N_MAX, RADIUS)

    nc = (N_MAX + 1) * (N_MAX + 2) // 2
    assert bn.shape == (nc, 24, 30)
    assert np.isfinite(bn).all()


def test_basis_convolve_impulse_matches_kernels():
    # Convolving a centred unit impulse with each basis component must reproduce
    # that component's kernel (placed at the impulse, up to the 180-degree flip that
    # distinguishes convolution from correlation). This validates the engine's
    # numerics, normalisation, and component ordering in one shot, convention-free.
    _orders, kernels = core.gauss_hermite_kernels(BETA, N_MAX, RADIUS)
    ks = kernels.shape[-1]
    r = ks // 2

    n = 2 * ks + 1
    img = np.zeros((n, n), dtype=np.float32)
    cy = cx = n // 2
    img[cy, cx] = 1.0

    bn = core.basis_convolve(img, BETA, N_MAX, RADIUS)
    assert bn.shape[0] == kernels.shape[0]

    for i in range(kernels.shape[0]):
        patch = bn[i, cy - r : cy + r + 1, cx - r : cx + r + 1]
        k = kernels[i].astype(np.float64)
        direct = np.allclose(patch, k, atol=1e-5)
        flipped = np.allclose(patch, np.rot90(k, 2), atol=1e-5)
        assert direct or flipped, f"component {i} impulse response != kernel"
        # The energy outside the kernel footprint must be ~zero.
        outside = bn[i].copy()
        outside[cy - r : cy + r + 1, cx - r : cx + r + 1] = 0.0
        assert np.abs(outside).max() < 1e-5


def test_basis_convolve_linearity():
    # The pass is a linear operator: B_n(a*R1 + b*R2) == a*B_n(R1) + b*B_n(R2).
    rng = np.random.default_rng(1)
    r1 = rng.standard_normal((20, 22)).astype(np.float32)
    r2 = rng.standard_normal((20, 22)).astype(np.float32)
    a, b = 1.7, -0.6

    lhs = core.basis_convolve((a * r1 + b * r2).astype(np.float32), BETA, N_MAX, RADIUS)
    rhs = a * core.basis_convolve(r1, BETA, N_MAX, RADIUS) + b * core.basis_convolve(
        r2, BETA, N_MAX, RADIUS
    )
    np.testing.assert_allclose(lhs, rhs, atol=1e-4)
