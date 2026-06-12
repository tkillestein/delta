"""M6 tests: kernel fit from stamps, jointly-fit background, photometric scale.

The fit orchestration is validated by replicating its stamp-pixel assembly in
NumPy and feeding the identical design to the M4 solver engine (so any mismatch
is in the C++ assembly, not the solve). The photometric scale is checked against
its definition sum_n a_n(x,y) S_n with S_n the basis footprint sums.
"""

import delta
import numpy as np


def _component_sums(beta, n_max):
    # S_c = (sum g_nx)(sum g_ny) in float64, matching the C++ component_sums.
    orders, _ = delta.gauss_hermite_kernels(beta, n_max)
    s1d = delta.gauss_hermite_basis1d(beta, n_max).sum(axis=1)
    return np.array([s1d[nx] * s1d[ny] for nx, ny in orders], dtype=np.float64)


def _gather(science, reference, bn, sx, sy, sr, var_s=None, var_r=None):
    """Replicate fit_kernel's pixel gathering (no masks, non-overlapping stamps)."""
    h, w = science.shape
    px, tgt, wts, rows = [], [], [], []
    for cx, cy in zip(sx, sy, strict=True):
        for y in range(max(0, cy - sr), min(h - 1, cy + sr) + 1):
            for x in range(max(0, cx - sr), min(w - 1, cx + sr) + 1):
                v = 0.0
                have = False
                if var_s is not None:
                    v += var_s[y, x]
                    have = True
                if var_r is not None:
                    v += var_r[y, x]
                    have = True
                weight = 1.0 / v if have else 1.0
                px.append((float(x), float(y)))
                tgt.append(float(science[y, x]))
                wts.append(weight)
                rows.append(bn[:, y, x].astype(np.float64))
    return np.array(px), np.array(tgt), np.array(wts), np.array(rows)


def test_fit_kernel_matches_python_assembly():
    h, w, beta, n_max, sr = 64, 64, 2.0, 2, 4
    rng = np.random.default_rng(0)
    ref = rng.standard_normal((h, w)).astype(np.float32)
    sci = rng.standard_normal((h, w)).astype(np.float32)
    var_s = rng.uniform(0.5, 2.0, (h, w)).astype(np.float32)
    var_r = rng.uniform(0.5, 2.0, (h, w)).astype(np.float32)

    knots = delta.grid_knots(0.0, 0.0, w - 1.0, h - 1.0, 4, 4)
    # Non-overlapping stamp grid (spacing > 2*sr).
    sx, sy = np.meshgrid(np.arange(12, w - 12, 16), np.arange(12, h - 12, 16))
    sx = sx.ravel().astype(np.int32)
    sy = sy.ravel().astype(np.int32)
    grid = np.logspace(-6, 4, 12)

    fit = delta.fit_kernel(
        sci,
        ref,
        knots,
        sx,
        sy,
        sr,
        beta,
        n_max,
        grid,
        science_var=var_s,
        reference_var=var_r,
    )

    bn = delta.basis_convolve(ref, beta, n_max)
    pts, tgt, wts, rows = _gather(sci, ref, bn, sx, sy, sr, var_s, var_r)
    ref_fit = delta.solve_gls_gcv(knots, pts, tgt, wts, rows, grid)

    assert fit["n_pixels"] == len(tgt)
    assert fit["n_stamps_used"] == len(sx)
    np.testing.assert_allclose(fit["theta"], ref_fit["theta"], rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(fit["lambda"], ref_fit["lambda"])
    np.testing.assert_allclose(
        fit["component_sums"], _component_sums(beta, n_max), rtol=1e-6, atol=1e-9
    )


def test_fit_recovers_noiseless_model_at_stamps():
    # Science built exactly as the model from a known theta -> the fit reproduces
    # it at the stamp pixels (small RSS) at small lambda.
    h, w, beta, n_max, sr = 64, 64, 1.8, 2, 5
    rng = np.random.default_rng(1)
    ref = (10.0 + rng.standard_normal((h, w))).astype(np.float32)
    knots = delta.grid_knots(0.0, 0.0, w - 1.0, h - 1.0, 4, 4)
    bn = delta.basis_convolve(ref, beta, n_max)
    ncomp = bn.shape[0]
    k = knots.shape[0]

    ys, xs = np.mgrid[0:h, 0:w]
    points = np.column_stack([xs.ravel().astype(float), ys.ravel().astype(float)])
    design = delta.tps_design(knots, points)
    theta_true = 0.05 * rng.standard_normal((ncomp + 1) * k)
    c = theta_true.reshape(ncomp + 1, k).T
    fields = (design @ c).reshape(h, w, ncomp + 1)
    model = fields[:, :, ncomp].copy()
    for n in range(ncomp):
        model += fields[:, :, n] * bn[n]
    sci = model.astype(np.float32)

    sx, sy = np.meshgrid(np.arange(12, w - 12, 16), np.arange(12, h - 12, 16))
    sx = sx.ravel().astype(np.int32)
    sy = sy.ravel().astype(np.int32)

    fit = delta.fit_kernel(sci, ref, knots, sx, sy, sr, beta, n_max, np.array([1e-8]))
    # Predicted model at stamp pixels reproduces the science (low residual).
    assert fit["rss"] < 1e-3


def test_photometric_scale_constant_kernel():
    h, w, beta = 32, 40, 2.0
    knots = delta.grid_knots(0.0, 0.0, w - 1.0, h - 1.0, 3, 3)
    k = knots.shape[0]
    sums = _component_sums(beta, 0)  # single component
    amp = 0.7
    theta = np.zeros(2 * k)
    theta[k - 3] = amp  # constant c_0 via affine intercept

    scale = delta.photometric_scale(knots, theta, sums, h, w)
    assert scale.shape == (h, w)
    np.testing.assert_allclose(scale, amp * sums[0], rtol=1e-5, atol=1e-5)


def test_photometric_scale_at_matches_definition():
    h, w, beta, n_max = 50, 50, 2.0, 2
    rng = np.random.default_rng(2)
    knots = delta.grid_knots(0.0, 0.0, w - 1.0, h - 1.0, 4, 4)
    k = knots.shape[0]
    sums = _component_sums(beta, n_max)
    ncomp = len(sums)
    theta = 0.2 * rng.standard_normal((ncomp + 1) * k)

    pts = rng.uniform([0, 0], [w - 1, h - 1], size=(40, 2))
    got = delta.photometric_scale_at(knots, theta, sums, pts)

    # Reference: a_n at points dotted with S_n.
    design = delta.tps_design(knots, pts)
    c = theta.reshape(ncomp + 1, k).T
    a = design @ c  # (m, ncomp+1)
    expected = a[:, :ncomp] @ sums
    np.testing.assert_allclose(got, expected, rtol=1e-8, atol=1e-8)


def test_photometric_scale_image_consistent_with_at():
    h, w, beta, n_max = 24, 28, 1.6, 1
    rng = np.random.default_rng(3)
    knots = delta.grid_knots(0.0, 0.0, w - 1.0, h - 1.0, 3, 3)
    k = knots.shape[0]
    sums = _component_sums(beta, n_max)
    ncomp = len(sums)
    theta = 0.3 * rng.standard_normal((ncomp + 1) * k)

    img = delta.photometric_scale(knots, theta, sums, h, w)
    ys, xs = np.mgrid[0:h, 0:w]
    pts = np.column_stack([xs.ravel().astype(float), ys.ravel().astype(float)])
    at = delta.photometric_scale_at(knots, theta, sums, pts).reshape(h, w)
    np.testing.assert_allclose(img, at, rtol=1e-5, atol=1e-5)
