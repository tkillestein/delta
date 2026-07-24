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
        n_max=2,  # this path test fixes the basis order, independent of the default
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


def _robust_reduced_chi2(difference, variance, mask):
    good = (variance > 0) & np.isfinite(difference) & np.isfinite(variance)
    if mask is not None:
        good &= mask == 0
    z2 = difference[good].astype(np.float64) ** 2 / variance[good].astype(np.float64)
    return np.median(z2) / 0.4549364231195724


def test_rescale_variance_forces_reduced_chi2_to_one():
    from delta._inputs import synth_variance

    ref, sci, _ = _pair(sig_ref=1.6, sig_sci=2.2)
    # Fit with the correct gain (good kernel solve), but feed the subtraction a
    # deliberately under-estimated variance map -> the un-rescaled diff-image
    # reduced chi2 should sit well above 1.
    gain = 1.5
    bad_factor = 0.02
    sci_var = synth_variance(sci, gain, 4.0) * bad_factor
    ref_var = synth_variance(ref, gain, 4.0) * bad_factor

    common = dict(science_var=sci_var, reference_var=ref_var, n_knots=3, stamp_radius=12)
    baseline = delta.subtract(sci, ref, **common)
    assert _robust_reduced_chi2(baseline.difference, baseline.variance, baseline.mask) > 1.5

    res = delta.subtract(sci, ref, **common, rescale_variance=True)
    assert res.variance_scale is not None and res.variance_scale > 1.0
    assert res.variance is not None and baseline.variance is not None
    np.testing.assert_allclose(res.variance, baseline.variance * res.variance_scale, rtol=1e-6)
    assert _robust_reduced_chi2(res.difference, res.variance, res.mask) == pytest.approx(
        1.0, abs=0.15
    )


def test_rescale_variance_off_by_default():
    ref, sci, _ = _pair(sig_ref=1.6, sig_sci=2.2)
    res = delta.subtract(sci, ref, gain=1.5, n_knots=3, stamp_radius=12)
    assert res.variance_scale is None


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


def test_apply_matches_subtract():
    # A frozen solution applied to the same pair reproduces the fit-then-subtract
    # difference exactly (same C++ subtract call, no re-fit).
    ref, sci, _ = _pair(sig_ref=1.6, sig_sci=2.4)
    sub = delta.Subtractor(n_knots=4, stamp_radius=12)
    res = sub.subtract(sci, ref, gain=1.5)
    applied = sub.apply(res.solution, sci, ref, gain=1.5)

    np.testing.assert_array_equal(applied.difference, res.difference)
    assert applied.solution is res.solution


def test_apply_reuses_solution_across_frames(tmp_path):
    # Fit on one pair, save, reload, and apply to a fresh science frame sharing
    # the reference. The static field is removed and the new transient survives.
    ref, sci, amp_t = _pair(sig_ref=1.6, sig_sci=2.4)
    sol = delta.Subtractor(n_knots=4, stamp_radius=12).fit(sci, ref, gain=1.5)

    path = str(tmp_path / "sol.npz")
    sol.save(path)
    loaded = delta.KernelSolution.load(path)

    ref2, sci2, amp_t2 = _pair(sig_ref=1.6, sig_sci=2.4, transient_xy=(160, 40), seed=7)
    res = delta.apply(loaded, sci2, ref2, gain=1.5)

    assert res.solution.direction == "reference"
    assert np.abs(_window_max(res.difference, (80, 80))) < 0.25 * amp_t2
    assert _window_max(res.difference, (160, 40)) > 0.4 * amp_t2


def test_apply_shape_mismatch_errors():
    ref, sci, _ = _pair(sig_ref=1.6, sig_sci=2.4)
    sol = delta.Subtractor(n_knots=4, stamp_radius=12).fit(sci, ref, gain=1.5)
    small = sci[:100, :100].copy()
    with pytest.raises(ValueError, match="solution shape"):
        delta.apply(sol, small, small, gain=1.5)


def test_write_fits_products(tmp_path):
    fits = pytest.importorskip("astropy.io.fits")
    ref, sci, _ = _pair(sig_ref=1.6, sig_sci=2.4)
    res = delta.subtract(sci, ref, gain=1.5, n_knots=3, stamp_radius=12)

    # Default (compressed) layout: provenance in the dataless primary, the
    # difference in a DIFFERENCE extension. Compression is quantised, so the
    # round-trip is approximate.
    path = str(tmp_path / "diff.fits")
    res.write(path)
    with fits.open(path) as hdul:
        assert hdul[0].header["DLTCONV"] == "reference"
        assert hdul[0].data is None
        assert "DIFFERENCE" in hdul
        assert "VARIANCE" in hdul
        scale = np.median(np.abs(res.difference)) + 1e-6
        np.testing.assert_allclose(hdul["DIFFERENCE"].data, res.difference, atol=0.1 * scale)

    # Legacy (uncompressed) layout: difference in the primary, exact round-trip.
    plain = str(tmp_path / "diff_plain.fits")
    res.write(plain, compress=False)
    with fits.open(plain) as hdul:
        assert hdul[0].header["DLTCONV"] == "reference"
        assert "VARIANCE" in hdul
        np.testing.assert_allclose(hdul[0].data, res.difference, rtol=1e-6)


def test_write_fits_provenance(tmp_path):
    fits = pytest.importorskip("astropy.io.fits")
    ref, sci, _ = _pair(sig_ref=1.6, sig_sci=2.4)
    res = delta.subtract(sci, ref, gain=1.5, n_knots=3, stamp_radius=12)

    assert res.elapsed > 0.0
    assert res.config["DLTSRAD"] == 12

    path = str(tmp_path / "diff.fits")
    extra: dict[str, object] = {"DLTSCI": "sci.fits", "DLTCMD": "delta subtract sci.fits ref.fits"}
    res.write(path, extra_cards=extra)
    header = fits.getheader(path)
    # Fit provenance (KernelSolution.header_cards).
    assert header["DLTCONV"] == "reference"
    # Config provenance (Subtractor.config_cards), not duplicated from the fit.
    assert header["DLTSRAD"] == 12
    # Environment provenance (delta._provenance.environment_cards).
    assert header["DLTVERS"] == delta.__version__
    assert "DLTHOST" in header
    assert header["DLTELAP"] > 0.0
    # Caller-supplied extra_cards, e.g. input paths / invocation, from the CLI.
    assert header["DLTSCI"] == "sci.fits"
    assert "delta subtract" in header["DLTCMD"]


def _few_stars_pair(n_stars, h=180, w=180, seed=0, sig_ref=1.6, sig_sci=1.6, bg=100.0):
    """A sparse point-source field (few detections, no differential background) --
    the regime issue #54 reports: a small/galaxy-dominated frame where only a
    handful of stamps clear the detection threshold."""
    rng = np.random.default_rng(seed)
    margin = 25
    pos = rng.uniform(margin, min(h, w) - margin, size=(n_stars, 2))
    ref, _ = _stars(h, w, pos, total=20000.0, sigma=sig_ref)
    sci, _ = _stars(h, w, pos, total=20000.0, sigma=sig_sci)
    ref = ref + bg + rng.normal(0, 3.0, (h, w))
    sci = sci + bg + rng.normal(0, 3.0, (h, w))
    return ref.astype(np.float32), sci.astype(np.float32)


def test_fit_background_false_zeroes_background_field():
    # With matched (zero differential) sky levels, disabling the background fit
    # must not change the difference image meaningfully, and the background block
    # of theta must be exactly zero (issue #54: an unconstrained background field
    # otherwise tends to overfit a sparsely-sampled/small frame).
    ref, sci = _few_stars_pair(6, seed=2)
    kw = dict(n_knots=4, stamp_radius=10, threshold_sigma=5.0)
    on = delta.subtract(sci, ref, gain=2.0, **kw)
    off = delta.subtract(sci, ref, gain=2.0, fit_background=False, **kw)

    k = off.solution.knots.shape[0]
    nc = off.solution.n_components
    assert off.solution.theta.shape == ((nc + 1) * k,)
    np.testing.assert_array_equal(off.solution.theta[nc * k :], 0.0)

    # Both fits stay well-behaved -- dropping the (unneeded, here) background
    # block must not blow up the fit.
    assert np.all(np.isfinite(off.difference))
    assert off.solution.reduced_chi2 < 5.0
    assert on.solution.reduced_chi2 < 5.0


def test_min_stamps_per_knot_caps_sparse_frame_knots():
    # A handful of stamps can't constrain a fine knot grid; the cap should pull
    # the requested 6x6 grid down. Disabling the cap (<= 0) keeps the full grid.
    ref, sci = _few_stars_pair(4, seed=3)
    capped = delta.Subtractor(
        n_knots=6, stamp_radius=8, threshold_sigma=5.0, min_stamps_per_knot=2.0
    ).fit(sci, ref, gain=2.0)
    uncapped = delta.Subtractor(
        n_knots=6, stamp_radius=8, threshold_sigma=5.0, min_stamps_per_knot=0.0
    ).fit(sci, ref, gain=2.0)

    assert uncapped.knots.shape[0] == 36
    assert capped.knots.shape[0] < 36
    assert capped.knots.shape[0] >= 9  # floors at 3x3
