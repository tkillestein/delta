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

    # clip_iterations=0 -> raw single solve under 1/(Var_s+Var_r) weights, which
    # is what the NumPy reference replicates (the default fit additionally does
    # IRLS reweighting + per-stamp clipping, validated separately).
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
        clip_iterations=0,
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


def _matched_pair(seed, corrupt_n=0, gain=2.0, rn=5.0, size=512, n=50):
    """A matched reference/science pair (science PSF broader) with Poisson + read
    noise and per-pixel variance maps. Optionally drop cosmic-ray spikes into the
    science (not recorded in its variance) on the `corrupt_n` brightest stars,
    making unambiguously bad stamps the kernel cannot fit."""
    rng = np.random.default_rng(seed)
    pos, flux = delta.validation.sample_starfield(
        (size, size), n, rng, flux_range=(2000.0, 30000.0), min_separation=30.0
    )
    ref0 = 120.0 + delta.validation.render_stars((size, size), pos, flux, 1.6)
    sci0 = 120.0 + delta.validation.render_stars((size, size), pos, flux, 2.4)

    def noisify(clean):
        var = clean / gain + rn**2
        obs = clean + rng.normal(0.0, np.sqrt(var))
        return obs.astype(np.float32), var.astype(np.float32)

    sci, svar = noisify(sci0)
    ref, rvar = noisify(ref0)

    bad = np.argsort(flux)[::-1][:corrupt_n]
    for bx, by in pos[bad].astype(int):
        sci[by, bx] += 4000.0
        sci[by, bx + 1] += 3000.0
    return sci, ref, svar, rvar, pos[bad]


def test_reduced_chi2_near_unity_on_clean_data():
    # With a correct per-pixel noise model the fit's reduced chi^2 must be ~1:
    # the residual variance is Var_target + (K² ⊗ Var_conv).
    sci, ref, svar, rvar, _ = _matched_pair(7)
    sol = delta.subtract(
        sci,
        ref,
        science_var=svar,
        reference_var=rvar,
        n_knots=4,
        stamp_radius=12,
        threshold_sigma=8,
    ).solution
    assert abs(sol.reduced_chi2 - 1.0) < 0.1
    assert sol.n_stamps_rejected == 0


def test_masked_halo_does_not_ring_bn():
    """A bad reference pixel just outside the stamp must not poison B_n inside it.

    Detect clears only the stamp footprint; the fit convolves a kernel-radius
    halo. Without median-fill sanitisation the defect rings into stamp pixels.
    """
    h, w, beta, n_max, sr = 80, 80, 2.0, 2, 6
    rng = np.random.default_rng(21)
    ref = (50.0 + 0.1 * rng.standard_normal((h, w))).astype(np.float32)
    sci = ref.copy()
    cx = cy = 40
    _, kernels = delta.gauss_hermite_kernels(beta, n_max)
    r_basis = kernels.shape[1] // 2
    dx = sr + 1  # just outside the stamp, well inside the fit halo
    assert dx <= sr + r_basis
    ref[cy, cx + dx] = 1.0e6

    knots = delta.grid_knots(0.0, 0.0, w - 1.0, h - 1.0, 3, 3)
    sx = np.array([cx], dtype=np.int32)
    sy = np.array([cy], dtype=np.int32)

    # Spiked frame contaminates a raw full-frame B_n at the stamp edge.
    dirty = delta.basis_convolve(ref, beta, n_max)
    ref_clean = ref.copy()
    ref_clean[cy, cx + dx] = float(np.median(ref[ref < 1.0e5]))
    clean = delta.basis_convolve(ref_clean, beta, n_max)
    edge = cx + sr
    assert abs(float(dirty[0, cy, edge]) - float(clean[0, cy, edge])) > 10.0

    # Masked spike: fit_kernel median-fills the halo, so the solve stays healthy.
    mask = np.zeros((h, w), np.uint8)
    mask[cy, cx + dx] = 1
    fit = delta.fit_kernel(
        sci,
        ref,
        knots,
        sx,
        sy,
        sr,
        beta,
        n_max,
        np.array([1e-2]),
        clip_iterations=0,
        reference_mask=mask,
    )
    assert fit["n_pixels"] > 0
    assert np.all(np.isfinite(fit["theta"]))
    assert fit["rss"] < 1e3  # no 1e6-spike ringing into the design


def test_irls_weights_stable_under_poisson_cores():
    """IRLS with (K² ⊗ Var) keeps reduced chi² near 1 on Poisson star stamps."""
    sci, ref, svar, rvar, _ = _matched_pair(19, size=256, n=40)
    sol = delta.subtract(
        sci,
        ref,
        science_var=svar,
        reference_var=rvar,
        n_knots=4,
        stamp_radius=12,
        threshold_sigma=6,
    ).solution
    assert abs(sol.reduced_chi2 - 1.0) < 0.15
    assert sol.n_stamps_used >= 10


def test_sigma_clipping_rejects_bad_stamps():
    # Cosmic-ray-hit stamps badly inflate the unclipped reduced chi^2; clipping
    # should flag them and pull chi^2 back toward 1.
    sci, ref, svar, rvar, badpos = _matched_pair(3, corrupt_n=6)
    kw = dict(science_var=svar, reference_var=rvar, n_knots=4, stamp_radius=12, threshold_sigma=8)

    noclip = delta.subtract(sci, ref, clip_sigma=0.0, **kw).solution
    clipped = delta.subtract(sci, ref, clip_sigma=4.0, **kw).solution

    assert noclip.reduced_chi2 > 2.0  # cosmics wreck the unclipped fit
    assert clipped.n_stamps_rejected > 0
    assert abs(clipped.reduced_chi2 - 1.0) < abs(noclip.reduced_chi2 - 1.0)
    assert abs(clipped.reduced_chi2 - 1.0) < 0.3

    # The rejected stamps coincide with the corrupted stars.
    assert clipped.stamp_accepted is not None
    assert clipped.stamp_x is not None and clipped.stamp_y is not None
    rej = np.where(clipped.stamp_accepted == 0)[0]
    rx, ry = clipped.stamp_x[rej], clipped.stamp_y[rej]
    hits = sum(np.min((rx - bx) ** 2 + (ry - by) ** 2) < 9 for bx, by in badpos)
    assert hits >= 5  # of the 6 corrupted stars


def test_stamped_solve_matches_exact(monkeypatch):
    # The per-stamp factorised M-build (DELTA_STAMP_APPROX=1) freezes the spatial
    # design across each stamp; when stamps are small vs the knot spacing this is a
    # tiny approximation. On a frame where the gate engages (size 1024, n_knots=3,
    # stamp_radius=12 -> knot spacing ~20x the stamp), the difference image must
    # track the exact per-row solve to well under the image noise floor (~1e-3),
    # which is the quantity that matters downstream (theta itself can differ more in
    # the heavily-smoothed near-null directions without changing the model).
    sci, ref, svar, rvar, _ = _matched_pair(11, size=1024, n=120)
    kw = dict(science_var=svar, reference_var=rvar, n_knots=3, stamp_radius=12, cv_folds=5)

    monkeypatch.setenv("DELTA_STAMP_APPROX", "0")
    exact = delta.subtract(sci, ref, **kw)
    monkeypatch.setenv("DELTA_STAMP_APPROX", "1")
    stamped = delta.subtract(sci, ref, **kw)

    diff_rel = np.linalg.norm(stamped.difference - exact.difference) / np.linalg.norm(
        exact.difference
    )
    assert diff_rel < 1e-3
    assert abs(stamped.solution.reduced_chi2 - exact.solution.reduced_chi2) < 0.02


def test_cv_exact_design_bytes(monkeypatch):
    # Same fine-knots/large-stamp geometry as test_stamped_solve_matches_exact: the
    # 16x knot-spacing gate rejects the stamped fast path, so solve_gls_cv (the
    # exact per-row path) runs and should report a nonzero whitened-design size.
    sci, ref, svar, rvar, _ = _matched_pair(11, size=1024, n=120)
    sel = delta.select_stamps(sci, ref)
    knots = delta.grid_knots(0.0, 0.0, sci.shape[1] - 1.0, sci.shape[0] - 1.0, 3, 3)
    grid = np.logspace(-6, 4, 8)

    monkeypatch.setenv("DELTA_STAMP_APPROX", "0")
    exact = delta.fit_kernel(
        sci,
        ref,
        knots,
        sel["x"],
        sel["y"],
        12,
        2.0,
        2,
        grid,
        cv_folds=5,
        science_var=svar,
        reference_var=rvar,
    )
    assert exact["cv_exact_design_bytes"] > 0

    monkeypatch.setenv("DELTA_STAMP_APPROX", "1")
    stamped = delta.fit_kernel(
        sci,
        ref,
        knots,
        sel["x"],
        sel["y"],
        12,
        2.0,
        2,
        grid,
        cv_folds=5,
        science_var=svar,
        reference_var=rvar,
    )
    assert stamped["cv_exact_design_bytes"] == 0

    # cv_folds <= 1 never takes the CV path at all.
    gcv = delta.fit_kernel(
        sci,
        ref,
        knots,
        sel["x"],
        sel["y"],
        12,
        2.0,
        2,
        grid,
        cv_folds=0,
        science_var=svar,
        reference_var=rvar,
    )
    assert gcv["cv_exact_design_bytes"] == 0
