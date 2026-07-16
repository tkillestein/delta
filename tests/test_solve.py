"""M4b tests: penalised generalized least-squares solver + GCV lambda search.

The solver fits the factorized Alard & Lupton model (SPEC §3.2-3.3): for each
kernel component n and spatial term m the design column is psi_m(x,y)*B_n(x,y),
plus a background block psi_m(x,y). We validate against a design matrix built
independently in NumPy from ``tps_design`` and the limiting behaviour of the
effective degrees of freedom.
"""

import delta
import numpy as np
import pytest


def _build_design(knots, points, bn):
    """Full design X (N, (nc+1)*k) matching the C++ assembly."""
    d = delta.tps_design(knots, points)  # (N, k)
    nc = bn.shape[1]
    blocks = [d * bn[:, c : c + 1] for c in range(nc)]
    blocks.append(d)  # background block
    return np.hstack(blocks)


def test_exact_recovery_noiseless_small_lambda():
    knots = delta.grid_knots(0.0, 0.0, 100.0, 80.0, 4, 4)  # k = 16
    rng = np.random.default_rng(0)
    n = 400
    points = rng.uniform([0, 0], [100, 80], size=(n, 2))
    bn = rng.standard_normal((n, 3))  # 3 kernel components

    x = _build_design(knots, points, bn)
    theta_true = rng.standard_normal(x.shape[1])
    target = x @ theta_true
    weights = np.ones(n)

    res = delta.solve_gls(knots, points, target, weights, bn, lam=1e-10)
    assert res["n_components"] == 3
    assert res["n_spatial"] == 16
    np.testing.assert_allclose(res["theta"], theta_true, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(x @ res["theta"], target, rtol=1e-6, atol=1e-6)
    assert res["rss"] < 1e-8


def test_weights_irrelevant_for_noiseless_consistent_data():
    knots = delta.grid_knots(0.0, 0.0, 50.0, 50.0, 4, 4)
    rng = np.random.default_rng(1)
    n = 300
    points = rng.uniform(0, 50, size=(n, 2))
    bn = rng.standard_normal((n, 2))
    x = _build_design(knots, points, bn)
    theta_true = rng.standard_normal(x.shape[1])
    target = x @ theta_true

    flat = delta.solve_gls(knots, points, target, np.ones(n), bn, lam=1e-10)
    varied = delta.solve_gls(knots, points, target, rng.uniform(0.5, 5.0, n), bn, lam=1e-10)
    np.testing.assert_allclose(flat["theta"], varied["theta"], rtol=1e-5, atol=1e-5)


def test_effective_dof_small_lambda_approaches_full_rank():
    knots = delta.grid_knots(0.0, 0.0, 100.0, 100.0, 4, 4)  # k = 16
    rng = np.random.default_rng(2)
    n = 500
    points = rng.uniform(0, 100, size=(n, 2))
    bn = rng.standard_normal((n, 3))
    x = _build_design(knots, points, bn)
    target = x @ rng.standard_normal(x.shape[1]) + 0.01 * rng.standard_normal(n)

    res = delta.solve_gls(knots, points, target, np.ones(n), bn, lam=1e-12)
    p = (3 + 1) * 16  # (nc + 1) * k
    assert abs(res["effective_dof"] - p) < 1e-3


def test_effective_dof_large_lambda_approaches_affine_nullspace():
    knots = delta.grid_knots(0.0, 0.0, 100.0, 100.0, 4, 4)
    rng = np.random.default_rng(3)
    n = 500
    points = rng.uniform(0, 100, size=(n, 2))
    bn = rng.standard_normal((n, 3))
    x = _build_design(knots, points, bn)
    target = x @ rng.standard_normal(x.shape[1]) + 0.01 * rng.standard_normal(n)

    res = delta.solve_gls(knots, points, target, np.ones(n), bn, lam=1e14)
    # Only the 3-D affine null space per field survives infinite smoothing.
    affine_dof = 3 * (3 + 1)  # 3 affine terms x (nc + 1) fields
    assert abs(res["effective_dof"] - affine_dof) < 0.5


def test_matches_numpy_normal_equations_at_fixed_lambda():
    knots = delta.grid_knots(0.0, 0.0, 60.0, 40.0, 4, 4)
    rng = np.random.default_rng(4)
    n = 350
    points = rng.uniform([0, 0], [60, 40], size=(n, 2))
    bn = rng.standard_normal((n, 2))
    x = _build_design(knots, points, bn)
    theta_true = rng.standard_normal(x.shape[1])
    target = x @ theta_true + 0.1 * rng.standard_normal(n)
    weights = rng.uniform(0.5, 2.0, n)
    lam = 3.7

    # Reference: penalised weighted normal equations with block-diag penalty.
    pen = delta.tps_penalty(knots)  # (k, k)
    k = pen.shape[0]
    p = np.zeros((x.shape[1], x.shape[1]))
    for b in range(3):  # nc + 1 = 2 + 1 blocks
        p[b * k : (b + 1) * k, b * k : (b + 1) * k] = pen
    w = np.diag(weights)
    a = x.T @ w @ x + lam * p
    theta_ref = np.linalg.solve(a, x.T @ w @ target)

    res = delta.solve_gls(knots, points, target, weights, bn, lam=lam)
    np.testing.assert_allclose(res["theta"], theta_ref, rtol=1e-7, atol=1e-7)


def test_gcv_search_returns_curve_minimum():
    knots = delta.grid_knots(0.0, 0.0, 100.0, 100.0, 5, 5)
    rng = np.random.default_rng(5)
    n = 600
    points = rng.uniform(0, 100, size=(n, 2))
    bn = rng.standard_normal((n, 2))
    x = _build_design(knots, points, bn)
    target = x @ rng.standard_normal(x.shape[1]) + 0.2 * rng.standard_normal(n)

    grid = np.logspace(-8, 8, 25, dtype=np.float64)
    res = delta.solve_gls_gcv(knots, points, target, np.ones(n), bn, grid)

    curve = res["gcv_curve"]
    assert curve.shape == grid.shape
    np.testing.assert_allclose(res["lambda_grid"], grid)
    # The reported solution sits at the grid minimum.
    imin = int(np.argmin(curve))
    assert res["lambda"] == grid[imin]
    assert res["gcv"] == curve[imin]
    np.testing.assert_allclose(res["gcv"], curve.min())


def test_gcv_fixed_lambda_matches_solve_gls():
    knots = delta.grid_knots(0.0, 0.0, 80.0, 80.0, 4, 4)
    rng = np.random.default_rng(6)
    n = 400
    points = rng.uniform(0, 80, size=(n, 2))
    bn = rng.standard_normal((n, 2))
    x = _build_design(knots, points, bn)
    target = x @ rng.standard_normal(x.shape[1]) + 0.1 * rng.standard_normal(n)
    weights = rng.uniform(0.5, 2.0, n)

    single = delta.solve_gls(knots, points, target, weights, bn, lam=12.5)
    via_gcv = delta.solve_gls_gcv(knots, points, target, weights, bn, np.array([12.5]))
    np.testing.assert_allclose(single["theta"], via_gcv["theta"], rtol=1e-9)
    np.testing.assert_allclose(single["gcv"], via_gcv["gcv"], rtol=1e-9)
    np.testing.assert_allclose(single["effective_dof"], via_gcv["effective_dof"], rtol=1e-9)


def test_cv_group_size_mismatch_raises():
    knots = delta.grid_knots(0.0, 0.0, 80.0, 80.0, 4, 4)
    rng = np.random.default_rng(7)
    n = 200
    points = rng.uniform(0, 80, size=(n, 2))
    bn = rng.standard_normal((n, 2))
    x = _build_design(knots, points, bn)
    target = x @ rng.standard_normal(x.shape[1])
    weights = np.ones(n)
    grid = np.logspace(-4, 4, 5, dtype=np.float64)

    short_group = np.zeros(n - 1, dtype=np.int32)
    with pytest.raises(RuntimeError, match="group.size"):
        delta.solve_gls_cv(knots, points, target, weights, bn, grid, short_group, 5)


def test_cv_selects_lambda_from_grid():
    knots = delta.grid_knots(0.0, 0.0, 80.0, 80.0, 4, 4)
    rng = np.random.default_rng(8)
    n = 400
    points = rng.uniform(0, 80, size=(n, 2))
    bn = rng.standard_normal((n, 2))
    x = _build_design(knots, points, bn)
    target = x @ rng.standard_normal(x.shape[1]) + 0.2 * rng.standard_normal(n)
    weights = np.ones(n)
    grid = np.logspace(-4, 4, 9, dtype=np.float64)
    n_groups = 5
    group = (np.arange(n) % n_groups).astype(np.int32)

    res = delta.solve_gls_cv(knots, points, target, weights, bn, grid, group, n_groups)
    assert res["lambda"] in grid
    assert res["gcv_curve"].shape == grid.shape


def test_cv_matches_numpy_leave_group_out():
    # N large enough to span several of solve_gls_cv's internal row chunks per fold
    # (kMaxFoldChunk=2048), so this exercises the chunked whitened-design build
    # (issue #49) across chunk boundaries, not just a single-chunk fold.
    knots = delta.grid_knots(0.0, 0.0, 100.0, 100.0, 5, 5)
    rng = np.random.default_rng(11)
    n = 6000
    points = rng.uniform(0, 100, size=(n, 2))
    bn = rng.standard_normal((n, 3))
    x = _build_design(knots, points, bn)
    theta_true = rng.standard_normal(x.shape[1])
    target = x @ theta_true + 0.2 * rng.standard_normal(n)
    weights = rng.uniform(0.5, 2.0, n)
    grid = np.logspace(-4, 4, 9, dtype=np.float64)
    n_groups = 5
    group = (np.arange(n) % n_groups).astype(np.int32)

    res = delta.solve_gls_cv(knots, points, target, weights, bn, grid, group, n_groups)

    # Independent NumPy leave-one-group-out CV over the same block design as
    # _build_design, penalised normal equations matching solve_at()'s A = M + lam*P0.
    pen = delta.tps_penalty(knots)
    k = pen.shape[0]
    p = x.shape[1]
    nc = bn.shape[1]
    pmat = np.zeros((p, p))
    for b in range(nc + 1):
        pmat[b * k : (b + 1) * k, b * k : (b + 1) * k] = pen

    def cv_error(lam):
        total = 0.0
        for g in range(n_groups):
            train, hold = group != g, group == g
            xg, wg, tg = x[train], weights[train], target[train]
            a = xg.T @ (wg[:, None] * xg) + lam * pmat
            th = np.linalg.solve(a, xg.T @ (wg * tg))
            xh, wh, th_t = x[hold], weights[hold], target[hold]
            total += np.sum(wh * (th_t - xh @ th) ** 2)
        return total

    # solve_gls_cv's coarse-to-fine lambda search doesn't evaluate every grid
    # point (see cv_finish); only compare the curve where it did.
    evaluated = np.isfinite(res["gcv_curve"])
    ref_curve = np.array([cv_error(lam) for lam in grid[evaluated]])
    np.testing.assert_allclose(res["gcv_curve"][evaluated], ref_curve, rtol=1e-6)
    assert res["lambda"] == grid[np.argmin(res["gcv_curve"])]

    a_full = x.T @ (weights[:, None] * x) + res["lambda"] * pmat
    theta_ref = np.linalg.solve(a_full, x.T @ (weights * target))
    np.testing.assert_allclose(res["theta"], theta_ref, rtol=1e-6, atol=1e-6)
