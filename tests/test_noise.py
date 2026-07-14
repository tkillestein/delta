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
    assert abs(_autocorr_lag1(block_out["difference"][c, c])) < 0.05
    np.testing.assert_allclose(block_out["difference"][c, c], single[c, c], atol=0.15)
    # Post-whitening variance is the white-noise level vs + vr*ΣK².
    sumk2 = float((kernel0.astype(np.float64) ** 2).sum())
    expected_var = vs + vr * sumk2
    np.testing.assert_allclose(block_out["variance"][c, c], expected_var, rtol=0.05)


def test_decorrelate_kernel_cache_matches_exact():
    # The default decorrelate caches |Khat|^2 on a coarse cell lattice (one kernel
    # FFT per cell) instead of recomputing it per block. Under a deliberately
    # strong kernel gradient the cached (auto) path must still track the exact
    # per-block path (kernel_cell_blocks=1) to a small fraction of the signal.
    n, beta, n_max = 384, 2.0, 1
    rng = np.random.default_rng(7)
    knots = delta.grid_knots(0.0, 0.0, n - 1.0, n - 1.0, 3, 3)
    k = knots.shape[0]
    _, kernels = delta.gauss_hermite_kernels(beta, n_max)
    ncomp = kernels.shape[0]

    # Constant a_0 = 1 plus a strong linear ramp on a_1 across the frame, so the
    # matching kernel varies appreciably between coarse cells.
    theta = np.zeros((ncomp + 1) * k)
    theta[k - 3] = 1.0  # a_0 intercept
    theta[2 * k - 2] = 0.6  # a_1 x-slope (affine x term)

    diff = _correlated_difference(n, kernels[0], 1.0, 1.0, rng).astype(np.float32)
    var_s = np.full((n, n), 1.0, np.float32)
    var_r = np.full((n, n), 1.0, np.float32)

    auto = delta.decorrelate(diff, knots, theta, beta, n_max, var_s, var_r, block=64)
    exact = delta.decorrelate(
        diff, knots, theta, beta, n_max, var_s, var_r, block=64, kernel_cell_blocks=1
    )

    c = slice(48, n - 48)
    diff = auto["difference"][c, c] - exact["difference"][c, c]
    rel_rms = np.sqrt(np.mean(diff**2)) / np.std(exact["difference"][c, c])
    assert rel_rms < 0.02
    np.testing.assert_allclose(auto["variance"][c, c], exact["variance"][c, c], rtol=0.05)


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


def test_whiten_psf_preserves_unit_sum_and_changes_profile():
    beta, n_max, n = 2.0, 1, 128
    knots = delta.grid_knots(0.0, 0.0, n - 1.0, n - 1.0, 3, 3)
    k = knots.shape[0]
    _, kernels = delta.gauss_hermite_kernels(beta, n_max)
    ncomp = kernels.shape[0]
    theta = np.zeros((ncomp + 1) * k)
    theta[k - 3] = 1.0

    matching = kernels[0]
    psf = np.clip(matching, 0.0, None).astype(np.float32)
    psf /= psf.sum()

    white = delta.whiten_score_psf(
        psf,
        knots,
        theta,
        beta,
        n_max,
        np.full((n, n), 1.0, np.float32),
        np.full((n, n), 1.0, np.float32),
        block=64,
    )
    assert white.shape == psf.shape
    np.testing.assert_allclose(white.sum(), 1.0, atol=1e-5)
    # Phi is not identically 1 for a broadening kernel, so the profile moves.
    assert not np.allclose(white, psf, atol=1e-4)


def test_pipeline_decorrelated_variance_matches_pull():
    """After decorrelate, returned variance matches the whitened difference."""
    from delta import validation

    h = w = 200
    rng = np.random.default_rng(11)
    sig_ref, sig_sci = 1.6, 2.4
    gain, read = 1.5, 4.0
    background = 150.0
    shape = (h, w)
    positions, fluxes = validation.sample_starfield(
        shape, 20, rng, flux_range=(2000.0, 40000.0), border=30, min_separation=20.0
    )
    ref_s = background + validation.render_stars(shape, positions, fluxes, sig_ref)
    sci_s = background + validation.render_stars(shape, positions, fluxes, sig_sci)
    ref_var = (np.maximum(ref_s, 0.0) / gain + read**2).astype(np.float32)
    sci_var = (np.maximum(sci_s, 0.0) / gain + read**2).astype(np.float32)
    ref = (ref_s + rng.normal(0.0, np.sqrt(ref_var))).astype(np.float32)
    sci = (sci_s + rng.normal(0.0, np.sqrt(sci_var))).astype(np.float32)

    res = delta.subtract(
        sci,
        ref,
        science_var=sci_var,
        reference_var=ref_var,
        n_knots=4,
        stamp_radius=12,
        decorrelate=True,
        score=True,
        block=64,
        source_catalog=False,
    )
    assert res.variance is not None and res.score is not None
    mask = res.mask == 0 if res.mask is not None else np.ones(shape, bool)
    # Source-free corner: pull and score both ~unit Gaussian under Poisson noise.
    corner = np.s_[10:70, 10:70]
    good = mask[corner]
    pull = res.difference[corner][good] / np.sqrt(res.variance[corner][good])
    assert 0.7 < np.std(pull) < 1.4
    score_corner = res.score[corner][good]
    assert 0.6 < np.std(score_corner) < 1.5
    # Whitened variance must not inherit the raw Poisson peaks at star centres
    # (those belong to pre-whitening Var(D), not σ_D² after Phi).
    raw = delta.subtract(
        sci,
        ref,
        science_var=sci_var,
        reference_var=ref_var,
        n_knots=4,
        stamp_radius=12,
        decorrelate=False,
        score=False,
    )
    assert raw.variance is not None
    # At the brightest star, raw Var(D) is elevated; whitened variance stays near sky.
    bi = int(np.argmax(fluxes))
    x, y = int(positions[bi][0]), int(positions[bi][1])
    if mask[y, x]:
        sky_white = float(np.median(res.variance[corner]))
        assert res.variance[y, x] < 2.0 * sky_white
        assert raw.variance[y, x] > 1.5 * float(np.median(raw.variance[corner]))
