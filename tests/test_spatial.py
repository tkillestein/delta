"""M4a tests: low-rank thin-plate regression spline (basis, penalty, fit).

The weighted GLS + GCV come later (M4); here we validate the basis/penalty and
an unweighted penalised fit against known fields and limiting behaviour.
"""

import delta
import numpy as np


def test_grid_knots_shape_and_span():
    knots = delta.grid_knots(0.0, 0.0, 100.0, 50.0, 5, 4)
    assert knots.shape == (20, 2)
    assert knots[:, 0].min() == 0.0 and knots[:, 0].max() == 100.0
    assert knots[:, 1].min() == 0.0 and knots[:, 1].max() == 50.0


def test_design_and_penalty_shapes():
    knots = delta.grid_knots(0.0, 0.0, 1.0, 1.0, 4, 4)  # k = 16
    pts = np.random.default_rng(0).uniform(0, 1, size=(30, 2))
    design = delta.tps_design(knots, pts)
    penalty = delta.tps_penalty(knots)
    assert design.shape == (30, 16)
    assert penalty.shape == (16, 16)


def test_penalty_is_symmetric_psd_with_unpenalised_affine():
    knots = delta.grid_knots(0.0, 0.0, 1.0, 1.0, 4, 4)  # k = 16
    penalty = delta.tps_penalty(knots)
    np.testing.assert_allclose(penalty, penalty.T, atol=1e-9)
    # The 3 affine columns/rows (last three) are unpenalised.
    np.testing.assert_allclose(penalty[-3:, :], 0.0, atol=1e-9)
    np.testing.assert_allclose(penalty[:, -3:], 0.0, atol=1e-9)
    # Radial block is positive (semi-)definite.
    eig = np.linalg.eigvalsh(penalty[:-3, :-3])
    assert eig.min() > -1e-8
    assert eig.max() > 0.0


def test_evaluate_matches_design_times_coeffs():
    knots = delta.grid_knots(0.0, 0.0, 10.0, 10.0, 5, 5)
    rng = np.random.default_rng(1)
    pts = rng.uniform(0, 10, size=(40, 2))
    coeffs = rng.standard_normal(knots.shape[0])
    via_design = delta.tps_design(knots, pts) @ coeffs
    via_eval = delta.tps_evaluate(knots, pts, coeffs)
    np.testing.assert_allclose(via_eval, via_design, rtol=1e-10, atol=1e-10)


def test_affine_field_recovered_exactly():
    # An affine field lives in the (unpenalised) null space -> exact for any lam.
    knots = delta.grid_knots(0.0, 0.0, 100.0, 80.0, 5, 5)
    rng = np.random.default_rng(2)
    pts = rng.uniform([0, 0], [100, 80], size=(200, 2))
    truth = 3.0 + 0.5 * pts[:, 0] - 0.2 * pts[:, 1]
    coeffs = delta.tps_fit(knots, pts, truth, lam=1.0)

    test_pts = rng.uniform([0, 0], [100, 80], size=(50, 2))
    pred = delta.tps_evaluate(knots, test_pts, coeffs)
    expected = 3.0 + 0.5 * test_pts[:, 0] - 0.2 * test_pts[:, 1]
    np.testing.assert_allclose(pred, expected, rtol=1e-6, atol=1e-6)


def test_smooth_field_recovered_with_small_lambda():
    domain = 100.0
    knots = delta.grid_knots(0.0, 0.0, domain, domain, 10, 10)
    rng = np.random.default_rng(3)
    pts = rng.uniform(0, domain, size=(800, 2))

    def field(p):
        return np.sin(2 * np.pi * p[:, 0] / domain) + np.cos(2 * np.pi * p[:, 1] / domain)

    coeffs = delta.tps_fit(knots, pts, field(pts), lam=1e-8)
    test_pts = rng.uniform(0, domain, size=(300, 2))
    pred = delta.tps_evaluate(knots, test_pts, coeffs)
    assert np.max(np.abs(pred - field(test_pts))) < 0.05


def test_large_lambda_approaches_ols_plane():
    knots = delta.grid_knots(0.0, 0.0, 50.0, 50.0, 6, 6)
    rng = np.random.default_rng(4)
    pts = rng.uniform(0, 50, size=(300, 2))
    values = np.sin(pts[:, 0] / 5.0) + 0.3 * pts[:, 1]  # not affine

    coeffs = delta.tps_fit(knots, pts, values, lam=1e12)
    test_pts = rng.uniform(0, 50, size=(60, 2))
    pred = delta.tps_evaluate(knots, test_pts, coeffs)

    # Ordinary least-squares plane fit in the original coordinates.
    amat = np.column_stack([np.ones(len(pts)), pts[:, 0], pts[:, 1]])
    plane, *_ = np.linalg.lstsq(amat, values, rcond=None)
    test_amat = np.column_stack([np.ones(len(test_pts)), test_pts[:, 0], test_pts[:, 1]])
    np.testing.assert_allclose(pred, test_amat @ plane, rtol=1e-4, atol=1e-4)
