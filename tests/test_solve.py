"""M4b tests: penalised generalized least-squares solver + GCV lambda search.

The solver fits the factorized Alard & Lupton model (SPEC §3.2-3.3): for each
kernel component n and spatial term m the design column is psi_m(x,y)*B_n(x,y),
plus a background block psi_m(x,y). We validate against a design matrix built
independently in NumPy from ``tps_design`` and the limiting behaviour of the
effective degrees of freedom.
"""

import delta
import numpy as np


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

    grid = np.logspace(-8, 8, 25)
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


def test_fit_background_false_zeroes_background_block_and_matches_reduced_design():
    # fit_background=False must (a) return theta at the documented (nc+1)*k length
    # with the trailing k-block exactly zero, and (b) match a solve against a
    # design with the background block dropped entirely (not just fixed at zero).
    knots = delta.grid_knots(0.0, 0.0, 80.0, 80.0, 4, 4)
    k = knots.shape[0]
    rng = np.random.default_rng(9)
    n = 500
    points = rng.uniform(0, 80, size=(n, 2))
    bn = rng.standard_normal((n, 3))
    x_full = _build_design(knots, points, bn)
    x_nobg = x_full[:, : 3 * k]  # drop the background block
    theta_true = rng.standard_normal(x_nobg.shape[1])
    target = x_nobg @ theta_true + 0.05 * rng.standard_normal(n)
    weights = rng.uniform(0.5, 2.0, n)

    res = delta.solve_gls(knots, points, target, weights, bn, lam=1e-6, fit_background=False)
    assert res["theta"].shape == (4 * k,)
    np.testing.assert_array_equal(res["theta"][3 * k :], 0.0)

    ref = np.linalg.lstsq(
        x_nobg.T @ (weights[:, None] * x_nobg) + 1e-6 * np.eye(3 * k),
        x_nobg.T @ (weights * target),
        rcond=None,
    )[0]
    np.testing.assert_allclose(res["theta"][: 3 * k], ref, rtol=1e-4, atol=1e-4)


def test_fit_background_false_gcv_and_cv_also_zero_background():
    knots = delta.grid_knots(0.0, 0.0, 80.0, 80.0, 4, 4)
    k = knots.shape[0]
    rng = np.random.default_rng(10)
    n = 400
    points = rng.uniform(0, 80, size=(n, 2))
    bn = rng.standard_normal((n, 2))
    grid = np.logspace(-6, 6, 13)
    target = rng.standard_normal(n)
    weights = np.ones(n)

    gcv = delta.solve_gls_gcv(knots, points, target, weights, bn, grid, fit_background=False)
    assert gcv["theta"].shape == (3 * k,)
    np.testing.assert_array_equal(gcv["theta"][2 * k :], 0.0)

    group = (np.arange(n) % 5).astype(np.int32)
    cv = delta.solve_gls_cv(
        knots, points, target, weights, bn, grid, group, 5, fit_background=False
    )
    assert cv["theta"].shape == (3 * k,)
    np.testing.assert_array_equal(cv["theta"][2 * k :], 0.0)
