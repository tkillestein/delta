"""M7 tests: noise decorrelation (whitening) and the match-filtered score image.

The headline checks follow SPEC §3.4/§10: the matching convolution correlates the
difference noise, decorrelation flattens its autocorrelation while preserving the
per-pixel noise variance, and the matched filter yields a unit-variance S/N map
that peaks at injected point sources.
"""

import delta
import numpy as np


def _autocorr_lag1(img):
    """Mean nearest-neighbour autocorrelation (x and y lag-1), normalised."""
    a = img - img.mean()
    var = a.var()
    cx = (a[:, :-1] * a[:, 1:]).mean()
    cy = (a[:-1, :] * a[1:, :]).mean()
    return 0.5 * (cx + cy) / var


def _correlated_difference(n, kernel, vs, vr, rng):
    """D = white_S + (kernel (x) white_R): science + convolved reference noise."""
    ns = np.sqrt(vs) * rng.standard_normal((n, n))
    nr = np.sqrt(vr) * rng.standard_normal((n, n))
    ks = kernel.shape[0]
    r = ks // 2
    conv = np.zeros((n, n))
    kf = kernel[::-1, ::-1]
    padded = np.pad(nr, r, mode="wrap")  # circular to match periodic FFT filter
    for i in range(ks):
        for j in range(ks):
            conv += kf[i, j] * padded[i : i + n, j : j + n]
    return (ns + conv).astype(np.float32)


def test_decorrelation_flattens_autocorrelation():
    n, beta = 128, 2.0
    rng = np.random.default_rng(0)
    _, kernels = delta.gauss_hermite_kernels(beta, 0)
    kernel = kernels[0]  # broadening Gaussian -> correlated reference noise
    vs, vr = 1.0, 1.0

    diff = _correlated_difference(n, kernel, vs, vr, rng)
    white = delta.decorrelate_block(diff, kernel, vs, vr)

    # The input has strong positive autocorrelation; the output is ~flat.
    assert _autocorr_lag1(diff) > 0.15
    assert abs(_autocorr_lag1(white)) < 0.03


def test_decorrelation_preserves_noise_variance():
    n, beta = 128, 2.0
    rng = np.random.default_rng(1)
    _, kernels = delta.gauss_hermite_kernels(beta, 0)
    kernel = kernels[0]
    vs, vr = 1.5, 0.8
    sumk2 = float((kernel.astype(np.float64) ** 2).sum())

    diff = _correlated_difference(n, kernel, vs, vr, rng)
    white = delta.decorrelate_block(diff, kernel, vs, vr)

    # Normalisation keeps the per-pixel noise variance ~ vs + vr*sum(K^2).
    expected = vs + vr * sumk2
    np.testing.assert_allclose(diff.var(), expected, rtol=0.1)
    np.testing.assert_allclose(white.var(), expected, rtol=0.1)


def test_decorrelation_kernel_is_symmetric():
    beta = 2.0
    _, kernels = delta.gauss_hermite_kernels(beta, 0)
    kern = delta.decorrelation_kernel(kernels[0], 1.0, 1.0, 64)
    assert kern.shape == (64, 64)
    # Zero-phase real-even spectrum -> kern[i,j] == kern[(n-i)%n, (n-j)%n]
    # (reflection about the centre, with the even-n fft-shift offset).
    reflected = np.roll(kern[::-1, ::-1], shift=(1, 1), axis=(0, 1))
    np.testing.assert_allclose(kern, reflected, atol=1e-5)


def test_decorrelate_spatially_varying_matches_constant_interior():
    # With a constant field and constant noise, the block path should match the
    # single-block whitening in the interior (away from frame edges).
    n, beta, n_max = 192, 2.0, 1
    rng = np.random.default_rng(2)
    knots = delta.grid_knots(0.0, 0.0, n - 1.0, n - 1.0, 3, 3)
    k = knots.shape[0]
    _, kernels = delta.gauss_hermite_kernels(beta, n_max)
    ncomp = kernels.shape[0]

    # Constant kernel = the 0th component only (a_0 = 1, others 0, bg 0).
    theta = np.zeros((ncomp + 1) * k)
    theta[k - 3] = 1.0  # constant a_0 via affine intercept
    kernel0 = kernels[0]
    vs, vr = 1.0, 1.0

    diff = _correlated_difference(n, kernel0, vs, vr, rng)
    var_s = np.full((n, n), vs, np.float32)
    var_r = np.full((n, n), vr, np.float32)

    block_out = delta.decorrelate(diff, knots, theta, beta, n_max, var_s, var_r, block=64)
    single = delta.decorrelate_block(diff, kernel0, vs, vr)

    # Compare a central region (circular-edge artefacts excluded).
    c = slice(48, n - 48)
    assert abs(_autocorr_lag1(block_out[c, c])) < 0.05
    np.testing.assert_allclose(block_out[c, c], single[c, c], atol=0.15)


def test_matched_filter_unit_variance_noise():
    n = 200
    rng = np.random.default_rng(3)
    noise_var = 2.3
    img = (np.sqrt(noise_var) * rng.standard_normal((n, n))).astype(np.float32)
    var = np.full((n, n), noise_var, dtype=np.float32)

    _, kernels = delta.gauss_hermite_kernels(2.0, 0)
    psf = kernels[0]
    score = delta.matched_filter(img, psf, var)

    # Source-free score is ~unit-variance Gaussian.
    np.testing.assert_allclose(score.std(), 1.0, rtol=0.05)
    np.testing.assert_allclose(score.mean(), 0.0, atol=0.05)


def test_matched_filter_recovers_source_snr():
    n = 128
    _, kernels = delta.gauss_hermite_kernels(2.0, 0)
    psf = kernels[0].astype(np.float64)
    sumpsf2 = (psf**2).sum()
    noise_var = 1.0
    amp = 50.0

    # Noiseless point source with the PSF profile at the centre.
    img = np.zeros((n, n), np.float64)
    r = psf.shape[0] // 2
    cy = cx = n // 2
    img[cy - r : cy + r + 1, cx - r : cx + r + 1] += amp * psf
    var = np.full((n, n), noise_var, dtype=np.float32)
    score = delta.matched_filter(img.astype(np.float32), psf.astype(np.float32), var)

    expected_peak = amp * np.sqrt(sumpsf2 / noise_var)
    assert abs(score[cy, cx] - expected_peak) / expected_peak < 1e-3
    assert score.argmax() == cy * n + cx


def test_matched_filter_spatially_varying_noise():
    """Score is ~unit-Gaussian under spatially-varying noise."""
    n = 256
    rng = np.random.default_rng(7)

    # Two-region variance map: left half 1.0, right half 4.0.
    var = np.ones((n, n), dtype=np.float32)
    var[:, n // 2 :] = 4.0
    img = (np.sqrt(var) * rng.standard_normal((n, n))).astype(np.float32)

    _, kernels = delta.gauss_hermite_kernels(2.0, 0)
    psf = kernels[0]
    score = delta.matched_filter(img, psf, var)

    # Both halves should be ~unit-variance when normalised correctly.
    np.testing.assert_allclose(score[:, : n // 2].std(), 1.0, rtol=0.07)
    np.testing.assert_allclose(score[:, n // 2 :].std(), 1.0, rtol=0.07)
