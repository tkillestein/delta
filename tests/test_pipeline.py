"""M8 tests: high-level pipeline, astropy interop, solution serialization.

A synthetic pair shares a star field at two seeings; the science frame carries an
extra transient. A correct subtraction removes the static stars and leaves the
transient, with the convolution direction auto-selected from the seeing.
"""

import delta
import numpy as np
import pytest


def _stars(h, w, positions, total, sigma):
    """Render Gaussian sources of fixed total flux (so flux is seeing-invariant)."""
    img = np.zeros((h, w), np.float64)
    ys, xs = np.mgrid[0:h, 0:w]
    amp = total / (2 * np.pi * sigma**2)
    for x, y in positions:
        img += amp * np.exp(-((xs - x) ** 2 + (ys - y) ** 2) / (2 * sigma**2))
    return img, amp


def _pair(sig_ref, sig_sci, transient_xy=(60, 140), seed=0):
    h = w = 200
    rng = np.random.default_rng(seed)
    grid = [(x, y) for x in (40, 80, 120, 160) for y in (40, 80, 120, 160)]
    ref, _ = _stars(h, w, grid, total=12000.0, sigma=sig_ref)
    sci, _ = _stars(h, w, grid, total=12000.0, sigma=sig_sci)
    trans, amp_t = _stars(h, w, [transient_xy], total=6000.0, sigma=sig_sci)
    sci = sci + trans
    bg = 100.0
    ref = ref + bg + rng.normal(0, 3.0, (h, w))
    sci = sci + bg + rng.normal(0, 3.0, (h, w))
    return ref.astype(np.float32), sci.astype(np.float32), amp_t


def _window_max(img, xy, r=8):
    x, y = xy
    return img[y - r : y + r + 1, x - r : x + r + 1].max()


def test_subtract_removes_statics_keeps_transient_reference_sharper(preview):
    ref, sci, amp_t = _pair(sig_ref=1.6, sig_sci=2.4)
    res = delta.subtract(sci, ref, gain=1.5, read_noise=4.0, n_knots=4, stamp_radius=12)
    preview("pipeline_reference_sharper", science=sci, reference=ref, difference=res.difference)

    assert isinstance(res, delta.DiffResult)
    # Reference is sharper -> it is the convolved image.
    assert res.solution.direction == "reference"
    assert np.isfinite(res.solution.gcv)

    # Static star is suppressed; the transient stands out and is positive.
    static_resid = np.abs(_window_max(res.difference, (80, 80)))
    transient_peak = _window_max(res.difference, (60, 140))
    assert static_resid < 0.25 * amp_t
    assert transient_peak > 0.4 * amp_t


def test_direction_flips_when_science_sharper(preview):
    # Swap seeings: science sharper -> it is convolved, sign handled so the
    # transient still comes out positive.
    ref, sci, amp_t = _pair(sig_ref=2.4, sig_sci=1.6, transient_xy=(140, 60), seed=1)
    res = delta.subtract(sci, ref, gain=1.5, read_noise=4.0, n_knots=4, stamp_radius=12)
    preview("pipeline_science_sharper", science=sci, reference=ref, difference=res.difference)

    assert res.solution.direction == "science"
    assert _window_max(res.difference, (140, 60)) > 0.4 * amp_t
    assert np.abs(_window_max(res.difference, (80, 80))) < 0.25 * amp_t


def test_variance_and_mask_propagate():
    ref, sci, _ = _pair(sig_ref=1.6, sig_sci=2.2)
    sci_mask = np.zeros(sci.shape, np.uint8)
    sci_mask[100, 100] = 1
    res = delta.subtract(sci, ref, gain=2.0, n_knots=3, stamp_radius=12, science_mask=sci_mask)
    assert res.variance is not None and res.variance.shape == sci.shape
    assert res.mask is not None
    assert res.mask[100, 100] & 1  # science bad pixel propagated


def test_decorrelate_and_score_paths_run(preview):
    ref, sci, _ = _pair(sig_ref=1.6, sig_sci=2.4)
    res = delta.subtract(
        sci,
        ref,
        gain=1.5,
        read_noise=4.0,
        n_knots=4,
        stamp_radius=12,
        decorrelate=True,
        score=True,
        block=64,
    )
    preview(
        "pipeline_decorrelate_score",
        science=sci,
        reference=ref,
        difference=res.difference,
        score=res.score,
    )
    assert res.score is not None and res.score.shape == sci.shape
    assert np.all(np.isfinite(res.difference))
    # Score noise is order-unity in a source-free corner.
    corner = res.score[10:60, 10:60]
    assert 0.3 < corner.std() < 3.0


def test_astropy_ccddata_matches_ndarray():
    ccddata = pytest.importorskip("astropy.nddata").CCDData
    ref, sci, _ = _pair(sig_ref=1.6, sig_sci=2.4)

    plain = delta.subtract(sci, ref, gain=1.5, n_knots=3, stamp_radius=12)
    sci_cc = ccddata(sci, unit="adu")
    ref_cc = ccddata(ref, unit="adu")
    viacc = delta.subtract(sci_cc, ref_cc, gain=1.5, n_knots=3, stamp_radius=12)

    np.testing.assert_allclose(plain.difference, viacc.difference, rtol=1e-5, atol=1e-5)


def test_solution_save_load_roundtrip(tmp_path):
    ref, sci, _ = _pair(sig_ref=1.6, sig_sci=2.4)
    sol = delta.Subtractor(n_knots=4, stamp_radius=12).fit(sci, ref, gain=1.5)

    path = str(tmp_path / "sol.npz")
    sol.save(path)
    loaded = delta.KernelSolution.load(path)

    assert loaded.direction == sol.direction
    assert loaded.n_components == sol.n_components
    np.testing.assert_allclose(loaded.theta, sol.theta)
    np.testing.assert_allclose(loaded.knots, sol.knots)
    np.testing.assert_allclose(loaded.photometric_scale(), sol.photometric_scale(), rtol=1e-6)


def test_write_fits_products(tmp_path):
    fits = pytest.importorskip("astropy.io.fits")
    ref, sci, _ = _pair(sig_ref=1.6, sig_sci=2.4)
    res = delta.subtract(sci, ref, gain=1.5, n_knots=3, stamp_radius=12)

    path = str(tmp_path / "diff.fits")
    res.write(path)
    with fits.open(path) as hdul:
        assert hdul[0].header["DLTCONV"] == "reference"
        assert "VARIANCE" in hdul
        np.testing.assert_allclose(hdul[0].data, res.difference, rtol=1e-6)
