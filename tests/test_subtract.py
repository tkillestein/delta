"""M5 tests: full-frame spatially-varying subtraction + variance/mask propagation.

The difference is validated against an independent NumPy evaluation of the
factorized model (SPEC §3.2), the propagated variance against a direct squared-
kernel convolution (SPEC §3.4), and the mask growth against the kernel-footprint
dilation rule (SPEC §3.6).
"""

import delta
import numpy as np


def _fields(knots, theta, n_components, h, w):
    """a_n(x,y) and b(x,y) over the full grid, shape (h, w, n_components+1)."""
    ys, xs = np.mgrid[0:h, 0:w]
    points = np.column_stack([xs.ravel().astype(float), ys.ravel().astype(float)])
    design = delta.tps_design(knots, points)  # (h*w, k)
    k = knots.shape[0]
    c = theta.reshape(n_components + 1, k).T  # (k, n_components+1)
    return (design @ c).reshape(h, w, n_components + 1)


def _conv2d_same(img, kernel):
    """True 2-D convolution, zero-padded, 'same' size (matches the C++ engine)."""
    kh, kw = kernel.shape
    ry, rx = kh // 2, kw // 2
    h, w = img.shape
    padded = np.zeros((h + 2 * ry, w + 2 * rx))
    padded[ry : ry + h, rx : rx + w] = img
    kf = kernel[::-1, ::-1]  # mirror -> convolution
    out = np.zeros((h, w))
    for i in range(kh):
        for j in range(kw):
            out += kf[i, j] * padded[i : i + h, j : j + w]
    return out


def _exact_scale(knots, theta, sums, h, w):
    """Exact per-pixel photometric scale via the at-points evaluator, shape (h, w)."""
    ys, xs = np.mgrid[0:h, 0:w]
    points = np.column_stack([xs.ravel().astype(float), ys.ravel().astype(float)])
    return delta._core.photometric_scale_at(knots, theta, sums, points).reshape(h, w)


def test_coarse_field_matches_exact_large_frame():
    # The full-frame spatial-field evaluation uses a coarse lattice + bilinear
    # interpolation (issue #16). On a frame large enough to take that path it must
    # stay near-exact vs the per-point evaluator, since the thin-plate fields vary
    # on the knot length-scale (>> a pixel).
    # Wide knot spacing (3x3 over 400 px ~= 200 px apart) so the stride, set to a
    # fraction of the knot spacing, lands well above the exact-fallback threshold.
    h, w = 400, 400
    rng = np.random.default_rng(7)
    knots = delta.grid_knots(0.0, 0.0, w - 1.0, h - 1.0, 3, 3)
    k = knots.shape[0]
    nc = 6
    theta = 0.1 * rng.standard_normal((nc + 1) * k)
    sums = rng.standard_normal(nc)

    got = delta._core.photometric_scale(knots, theta, sums, h, w)
    exact = _exact_scale(knots, theta, sums, h, w)

    # Worst-case interpolation error is a small fraction of the field's dynamic
    # range. This frame's knot spacing forces the largest stride/spacing ratio the
    # scheme allows (~1/16); on the reference frame the stride caps out far below the
    # knot spacing, so the real-world error is ~30x smaller still. The coarse path is
    # genuinely active (not the exact fallback), so the error is non-zero.
    err = np.max(np.abs(got - exact))
    assert err < 5e-3 * np.ptp(exact)
    assert err > 0.0


def test_exact_fallback_small_frame_bit_for_bit():
    # Small frames bypass the coarse grid and evaluate exactly per pixel.
    h, w = 40, 48  # <= 2*stride -> exact path
    rng = np.random.default_rng(8)
    knots = delta.grid_knots(0.0, 0.0, w - 1.0, h - 1.0, 3, 3)
    k = knots.shape[0]
    nc = 4
    theta = 0.1 * rng.standard_normal((nc + 1) * k)
    sums = rng.standard_normal(nc)

    got = delta._core.photometric_scale(knots, theta, sums, h, w)
    exact = _exact_scale(knots, theta, sums, h, w)
    np.testing.assert_allclose(got, exact, rtol=1e-5, atol=1e-5)


def test_difference_matches_numpy_model():
    h, w, beta, n_max = 48, 56, 2.0, 2
    rng = np.random.default_rng(0)
    ref = rng.standard_normal((h, w)).astype(np.float32)
    sci = rng.standard_normal((h, w)).astype(np.float32)

    knots = delta.grid_knots(0.0, 0.0, w - 1.0, h - 1.0, 4, 4)
    bn = delta.basis_convolve(ref, beta, n_max)  # (ncomp, h, w)
    ncomp = bn.shape[0]
    theta = 0.1 * rng.standard_normal((ncomp + 1) * knots.shape[0])

    res = delta.subtract_model(sci, ref, knots, theta, beta, n_max)
    assert res["variance"] is None and res["mask"] is None

    fields = _fields(knots, theta, ncomp, h, w)
    model = fields[:, :, ncomp].copy()
    for n in range(ncomp):
        model += fields[:, :, n] * bn[n]
    np.testing.assert_allclose(res["difference"], sci - model, rtol=1e-4, atol=1e-4)


def test_self_subtraction_is_zero():
    # Science built exactly as the model -> difference is ~0 everywhere.
    h, w, beta, n_max = 40, 40, 1.8, 2
    rng = np.random.default_rng(1)
    ref = (5.0 + rng.standard_normal((h, w))).astype(np.float32)
    knots = delta.grid_knots(0.0, 0.0, w - 1.0, h - 1.0, 3, 3)
    bn = delta.basis_convolve(ref, beta, n_max)
    ncomp = bn.shape[0]
    theta = 0.05 * rng.standard_normal((ncomp + 1) * knots.shape[0])

    fields = _fields(knots, theta, ncomp, h, w)
    model = fields[:, :, ncomp].copy()
    for n in range(ncomp):
        model += fields[:, :, n] * bn[n]

    res = delta.subtract_model(model.astype(np.float32), ref, knots, theta, beta, n_max)
    np.testing.assert_allclose(res["difference"], 0.0, atol=1e-3)


def test_variance_propagation_constant_kernel():
    # n_max=0 -> single Gaussian component; a constant coefficient A makes the
    # variance term A^2 (phi_0^2 (x) Var(R)).
    h, w, beta = 50, 50, 2.0
    rng = np.random.default_rng(2)
    ref = rng.standard_normal((h, w)).astype(np.float32)
    sci = rng.standard_normal((h, w)).astype(np.float32)
    var_r = rng.uniform(0.5, 2.0, (h, w)).astype(np.float32)
    var_s = rng.uniform(0.5, 2.0, (h, w)).astype(np.float32)

    knots = delta.grid_knots(0.0, 0.0, w - 1.0, h - 1.0, 3, 3)
    k = knots.shape[0]
    amp = 1.3
    # theta = [c_0 (k) | b (k)]; constant field via the affine intercept column
    # (design = [radial (k-3) | 1, u, v]); background left at zero.
    theta = np.zeros(2 * k)
    theta[k - 3] = amp  # intercept of the c_0 field

    res = delta.subtract_model(
        sci, ref, knots, theta, beta, 0, science_var=var_s, reference_var=var_r
    )
    assert res["variance"] is not None

    orders, kernels = delta.gauss_hermite_kernels(beta, 0)
    phi0 = kernels[0].astype(np.float64)
    expected = var_s + amp**2 * _conv2d_same(var_r.astype(np.float64), phi0**2)
    np.testing.assert_allclose(res["variance"], expected, rtol=1e-3, atol=1e-3)


def test_variance_science_only_passthrough():
    # No reference variance -> Var(D) = Var(S) exactly (no convolution term).
    h, w, beta, n_max = 30, 30, 1.5, 1
    rng = np.random.default_rng(3)
    ref = rng.standard_normal((h, w)).astype(np.float32)
    sci = rng.standard_normal((h, w)).astype(np.float32)
    var_s = rng.uniform(0.5, 2.0, (h, w)).astype(np.float32)
    knots = delta.grid_knots(0.0, 0.0, w - 1.0, h - 1.0, 3, 3)
    ncomp = delta.basis_convolve(ref, beta, n_max).shape[0]
    theta = np.zeros((ncomp + 1) * knots.shape[0])

    res = delta.subtract_model(sci, ref, knots, theta, beta, n_max, science_var=var_s)
    assert res["variance"] is not None
    np.testing.assert_allclose(res["variance"], var_s, rtol=1e-6, atol=1e-6)


def test_mask_growth_dilation_and_edge():
    h, w, beta, n_max = 41, 41, 2.0, 1
    rng = np.random.default_rng(4)
    ref = rng.standard_normal((h, w)).astype(np.float32)
    sci = rng.standard_normal((h, w)).astype(np.float32)
    knots = delta.grid_knots(0.0, 0.0, w - 1.0, h - 1.0, 3, 3)
    ncomp = delta.basis_convolve(ref, beta, n_max).shape[0]
    theta = np.zeros((ncomp + 1) * knots.shape[0])

    # One bad reference pixel at the centre; one bad science pixel off-centre.
    ref_mask = np.zeros((h, w), np.uint8)
    ref_mask[20, 20] = 1  # kMaskBad
    sci_mask = np.zeros((h, w), np.uint8)
    sci_mask[10, 30] = 1

    res = delta.subtract_model(
        sci,
        ref,
        knots,
        theta,
        beta,
        n_max,
        science_mask=sci_mask,
        reference_mask=ref_mask,
    )
    mask = res["mask"]
    assert mask is not None

    # Reference radius for n_max=1: the kernel half-width used by subtract.
    orders, kernels = delta.gauss_hermite_kernels(beta, n_max)
    r = kernels.shape[1] // 2

    # The bad reference pixel is dilated to a (2r+1)^2 box (bit 1 set).
    assert (mask[20 - r : 20 + r + 1, 20 - r : 20 + r + 1] & 1).all()
    # Just outside the box (and outside the edge border) it is not bit-1 set.
    if 20 + r + 1 < w - r:
        assert (mask[20, 20 + r + 1] & 1) == 0
    # Science bad pixel propagates one-to-one (bit 1 set there).
    assert mask[10, 30] & 1
    # Edge border of width r is flagged kMaskEdge (bit 4 = 16).
    assert (mask[0, :] & 16).all()
    assert (mask[:, 0] & 16).all()
    assert (mask[r:-r, r:-r] & 16 == 0).all()
