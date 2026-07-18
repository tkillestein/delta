"""Source catalog from the score image (SPEC §3.7, issue #8).

Connected-component labelling of synthetic score images, exercising the
FWHM-consistency filter (noise blips vs. genuine PSF-shaped blobs vs.
oversized blends), dipole flagging, mask-pixel exclusion/boundary reporting,
edge-safety, the default-on `DiffResult.source_catalog`, and the on-demand
`DiffResult.build_catalog()` path.
"""

import delta
import numpy as np
import pytest
from delta import catalog, validation

# Mirrors include/delta/image.hpp's MaskFlag bit values.
_MASK_BAD = 1 << 0
_MASK_COSMIC = 1 << 2


def _gaussian_blob(shape, x, y, peak, sigma):
    """A Gaussian bump of given peak amplitude (not total flux)."""
    flux = peak * 2.0 * np.pi * sigma**2
    return validation.gaussian_psf(shape, x, y, flux, sigma).astype(np.float32)


def _gaussian_stamp(size, sigma):
    """A small, unit-sum Gaussian PSF stamp (odd `size`, centred)."""
    r = size // 2
    ys, xs = np.mgrid[-r : r + 1, -r : r + 1]
    g = np.exp(-(xs**2 + ys**2) / (2.0 * sigma**2))
    return (g / g.sum()).astype(np.float32)


def _static_pair(sig_ref=1.6, sig_sci=2.4, sigma_noise=3.0, seed=0):
    h = w = 240
    rng = np.random.default_rng(seed)
    grid = [(x, y) for x in (50, 110, 170, 200) for y in (50, 110, 170, 200)]
    ref = np.full((h, w), 120.0)
    sci = np.full((h, w), 120.0)
    for x, y in grid:
        ref = ref + validation.gaussian_psf((h, w), x, y, 12000.0, sig_ref)
        sci = sci + validation.gaussian_psf((h, w), x, y, 12000.0, sig_sci)
    var = np.full((h, w), sigma_noise**2, np.float32)
    return ref, sci, var, sig_sci, rng, sigma_noise


def _finalize(ref, sci, rng, sigma_noise):
    ref = (ref + rng.normal(0, sigma_noise, ref.shape)).astype(np.float32)
    sci = (sci + rng.normal(0, sigma_noise, sci.shape)).astype(np.float32)
    return ref, sci


def test_clean_psf_blob_is_recovered_and_consistent():
    shape = (80, 80)
    score = _gaussian_blob(shape, 40, 40, 8.0, 2.0)
    diff = score.copy()
    cat = catalog.build_catalog(
        score, diff, expected_fwhm=2.0 * 2.354820045030949, threshold_sigma=5.0
    )
    assert len(cat) == 1
    assert bool(cat["fwhm_consistent"][0])
    assert abs(cat["x"][0] - 40) < 1.0
    assert abs(cat["y"][0] - 40) < 1.0
    assert cat["peak_snr"][0] > 7.5


def test_isolated_spike_is_flagged_inconsistent():
    """A single-pixel spike has far too few pixels for its peak S/N."""
    shape = (60, 60)
    score = np.zeros(shape, np.float32)
    score[30, 30] = 8.0
    diff = score.copy()
    cat = catalog.build_catalog(score, diff, expected_fwhm=4.0, threshold_sigma=5.0)
    assert len(cat) == 1
    assert cat["n_pix"][0] == 1
    assert not bool(cat["fwhm_consistent"][0])
    assert cat["quality"][0] & 0b001


def test_extended_blend_is_flagged_inconsistent():
    """An artificially widened blob has far too many pixels for its peak S/N."""
    shape = (60, 60)
    score = np.zeros(shape, np.float32)
    score[20:35, 20:35] = 6.0  # 15x15 block
    diff = score.copy()
    cat = catalog.build_catalog(score, diff, expected_fwhm=2.0, threshold_sigma=5.0)
    assert len(cat) == 1
    assert cat["n_pix"][0] == 225
    assert not bool(cat["fwhm_consistent"][0])
    assert cat["fwhm_ratio"][0] > 3.0


def test_dipole_pair_flags_is_dipole():
    shape = (60, 60)
    score = np.zeros(shape, np.float32)
    score[28:33, 28:33] = 8.0  # positive 5x5 block, cols/rows 28..32
    score[30, 33] = -4.0  # negative pixel touching the block's east edge
    diff = score.copy()
    cat = catalog.build_catalog(
        score, diff, expected_fwhm=3.0, threshold_sigma=5.0, threshold_sigma_dipole=3.0
    )
    assert len(cat) == 1
    assert bool(cat["is_dipole"][0])
    assert cat["quality"][0] & 0b010


def test_isolated_blob_without_negative_counterpart_is_not_dipole():
    shape = (60, 60)
    score = np.zeros(shape, np.float32)
    score[28:33, 28:33] = 8.0
    diff = score.copy()
    cat = catalog.build_catalog(
        score, diff, expected_fwhm=3.0, threshold_sigma=5.0, threshold_sigma_dipole=3.0
    )
    assert len(cat) == 1
    assert not bool(cat["is_dipole"][0])


def test_false_positive_rate_in_source_free_region():
    rng = np.random.default_rng(7)
    shape = (300, 300)
    score = rng.normal(0.0, 1.0, shape).astype(np.float32)
    diff = np.zeros(shape, np.float32)
    cat = catalog.build_catalog(score, diff, expected_fwhm=3.5, threshold_sigma=5.0)
    # Expected spurious-blob rate over 9e4 pixels at 5 sigma is well under 1.
    assert len(cat) <= 5


def test_edge_of_frame_aperture_is_bounds_safe():
    shape = (40, 40)
    score = np.zeros(shape, np.float32)
    score[0:3, 0:3] = 8.0  # blob touching the top-left corner
    diff = score.copy()
    cat = catalog.build_catalog(
        score, diff, expected_fwhm=3.0, threshold_sigma=5.0, aperture_radius=10
    )
    assert len(cat) == 1
    assert np.isfinite(cat["flux"][0])
    assert cat["flux"][0] > 0


def test_masked_pixels_are_excluded_from_growth():
    shape = (40, 40)
    score = np.zeros(shape, np.float32)
    score[18:23, 18:23] = 8.0  # cols/rows 18..22
    mask = np.zeros(shape, np.uint8)
    mask[18:23, 20] = _MASK_BAD  # column straight through the middle
    diff = score.copy()
    cat = catalog.build_catalog(score, diff, mask=mask, expected_fwhm=3.0, threshold_sigma=5.0)
    # The masked column severs 8-connectivity between cols 18-19 and 21-22.
    assert len(cat) == 2
    assert np.all(cat["mask_flags"] != 0)


def test_mask_flags_or_over_boundary_without_membership():
    shape = (40, 40)
    score = np.zeros(shape, np.float32)
    score[18:23, 18:23] = 8.0  # last column of the block is col 22
    mask = np.zeros(shape, np.uint8)
    mask[20, 23] = _MASK_COSMIC  # touches the block's east edge, not a member
    diff = score.copy()
    cat = catalog.build_catalog(score, diff, mask=mask, expected_fwhm=3.0, threshold_sigma=5.0)
    assert len(cat) == 1
    assert cat["mask_flags"][0] & _MASK_COSMIC


def test_threshold_sigma_dipole_independent_of_threshold_sigma():
    shape = (40, 40)
    score = np.zeros(shape, np.float32)
    score[18:23, 18:23] = 8.0
    score[20, 23] = -3.5  # marginal negative pixel, touching the block
    diff = score.copy()
    cat_low = catalog.build_catalog(
        score, diff, expected_fwhm=3.0, threshold_sigma=5.0, threshold_sigma_dipole=3.0
    )
    cat_high = catalog.build_catalog(
        score, diff, expected_fwhm=3.0, threshold_sigma=5.0, threshold_sigma_dipole=4.0
    )
    assert bool(cat_low["is_dipole"][0])
    assert not bool(cat_high["is_dipole"][0])


def test_aperture_radius_zero_resolves_from_expected_fwhm():
    shape = (60, 60)
    score = np.zeros(shape, np.float32)
    score[28:33, 28:33] = 8.0
    diff = np.ones(shape, np.float32)
    cat_small = catalog.build_catalog(
        score, diff, expected_fwhm=2.0, threshold_sigma=5.0, aperture_radius=0
    )
    cat_large = catalog.build_catalog(
        score, diff, expected_fwhm=8.0, threshold_sigma=5.0, aperture_radius=0
    )
    assert cat_large["flux"][0] > cat_small["flux"][0]


def test_catalog_dtype_and_as_table_roundtrip():
    shape = (40, 40)
    score = np.zeros(shape, np.float32)
    score[18:23, 18:23] = 8.0
    diff = score.copy()
    cat = catalog.build_catalog(score, diff, expected_fwhm=3.0, threshold_sigma=5.0)
    assert cat.dtype == catalog.CATALOG_DTYPE

    pytest.importorskip("astropy")
    table = catalog.build_catalog(
        score, diff, expected_fwhm=3.0, threshold_sigma=5.0, as_table=True
    )
    assert list(table.colnames) == list(catalog.CATALOG_DTYPE.names or ())
    assert len(table) == len(cat)


def test_diffresult_build_catalog_method_end_to_end():
    ref, sci, var, sig_sci, rng, sigma_noise = _static_pair(seed=11)
    position = (180, 180)
    sci = validation.inject(sci, [position], [30000.0], sig_sci)
    ref, sci = _finalize(ref, sci, rng, sigma_noise)

    res = delta.subtract(
        sci,
        ref,
        science_var=var,
        reference_var=var,
        n_knots=4,
        stamp_radius=12,
        decorrelate=True,
        score=True,
        block=128,
    )
    cat = res.build_catalog(threshold_sigma=5.0)
    assert len(cat) >= 1
    dist = np.hypot(cat["x"] - position[0], cat["y"] - position[1])
    assert dist.min() < 3.0


def test_build_catalog_raises_without_score():
    ref, sci, var, _, rng, sigma_noise = _static_pair(seed=12)
    ref, sci = _finalize(ref, sci, rng, sigma_noise)
    res = delta.subtract(sci, ref, science_var=var, reference_var=var, n_knots=4, stamp_radius=12)
    with pytest.raises(ValueError):
        res.build_catalog()


def test_source_catalog_is_auto_populated_when_score_is_on():
    ref, sci, var, sig_sci, rng, sigma_noise = _static_pair(seed=13)
    position = (180, 180)
    sci = validation.inject(sci, [position], [30000.0], sig_sci)
    ref, sci = _finalize(ref, sci, rng, sigma_noise)

    res = delta.subtract(
        sci,
        ref,
        science_var=var,
        reference_var=var,
        n_knots=4,
        stamp_radius=12,
        decorrelate=True,
        score=True,
        block=128,
    )
    assert res.source_catalog is not None
    assert res.source_catalog.dtype == catalog.CATALOG_DTYPE
    dist = np.hypot(res.source_catalog["x"] - position[0], res.source_catalog["y"] - position[1])
    assert dist.min() < 3.0


def test_source_catalog_is_none_without_score():
    ref, sci, var, _, rng, sigma_noise = _static_pair(seed=14)
    ref, sci = _finalize(ref, sci, rng, sigma_noise)
    res = delta.subtract(sci, ref, science_var=var, reference_var=var, n_knots=4, stamp_radius=12)
    assert res.source_catalog is None


def test_source_catalog_disabled_via_source_catalog_false():
    ref, sci, var, _, rng, sigma_noise = _static_pair(seed=15)
    ref, sci = _finalize(ref, sci, rng, sigma_noise)
    res = delta.subtract(
        sci,
        ref,
        science_var=var,
        reference_var=var,
        n_knots=4,
        stamp_radius=12,
        decorrelate=True,
        score=True,
        source_catalog=False,
        block=128,
    )
    assert res.score is not None
    assert res.source_catalog is None


def test_write_includes_catalog_extension(tmp_path):
    pytest.importorskip("astropy")
    from astropy.io import fits

    ref, sci, var, sig_sci, rng, sigma_noise = _static_pair(seed=16)
    position = (180, 180)
    sci = validation.inject(sci, [position], [30000.0], sig_sci)
    ref, sci = _finalize(ref, sci, rng, sigma_noise)

    res = delta.subtract(
        sci,
        ref,
        science_var=var,
        reference_var=var,
        n_knots=4,
        stamp_radius=12,
        decorrelate=True,
        score=True,
        block=128,
    )
    assert res.source_catalog is not None and len(res.source_catalog) >= 1
    path = tmp_path / "diff.fits"
    res.write(str(path))
    with fits.open(path) as hdul:
        assert "CATALOG" in hdul
        assert len(hdul["CATALOG"].data) == len(res.source_catalog)


# ---- PSF-shape chi^2 fit (opt-in false-positive vetting) -------------------


def test_fit_psf_shape_recovers_amplitude_for_a_clean_source():
    shape = (60, 60)
    psf = _gaussian_stamp(11, 1.8)
    amplitude = 500.0
    r = psf.shape[0] // 2
    diff = np.zeros(shape, np.float32)
    diff[30 - r : 30 + r + 1, 30 - r : 30 + r + 1] = amplitude * psf
    score = _gaussian_blob(shape, 30, 30, 8.0, 1.8)
    variance = np.ones(shape, np.float32)

    cat = catalog.build_catalog(
        score, diff, psf=psf, threshold_sigma=5.0, fit_psf_shape=True, variance=variance
    )
    assert len(cat) == 1
    assert cat["psf_chi2_dof"][0] > 0
    assert cat["psf_chi2"][0] / cat["psf_chi2_dof"][0] < 1.0
    assert bool(cat["shape_consistent"][0])
    assert cat["quality"][0] & catalog.QUALITY_SHAPE_INCONSISTENT == 0
    assert abs(cat["psf_amplitude"][0] - amplitude) / amplitude < 0.05
    assert cat["psf_amplitude_err"][0] > 0


def test_fit_psf_shape_flags_a_non_psf_shaped_spike():
    """A single-pixel spike (cosmic ray / hot pixel) fits a broad PSF badly."""
    shape = (60, 60)
    psf = _gaussian_stamp(11, 1.8)
    diff = np.zeros(shape, np.float32)
    diff[30, 30] = 5000.0  # all flux in one pixel, none of the PSF's wings
    score = np.zeros(shape, np.float32)
    score[30, 30] = 8.0
    variance = np.ones(shape, np.float32)

    cat = catalog.build_catalog(
        score, diff, psf=psf, threshold_sigma=5.0, fit_psf_shape=True, variance=variance
    )
    assert len(cat) == 1
    assert not bool(cat["shape_consistent"][0])
    assert cat["quality"][0] & catalog.QUALITY_SHAPE_INCONSISTENT


def test_fit_psf_shape_requires_psf_not_expected_fwhm():
    shape = (40, 40)
    score = np.zeros(shape, np.float32)
    score[18:23, 18:23] = 8.0
    diff = score.copy()
    with pytest.raises(ValueError, match="expected_fwhm"):
        catalog.build_catalog(
            score,
            diff,
            expected_fwhm=3.0,
            threshold_sigma=5.0,
            fit_psf_shape=True,
            variance=np.ones(shape, np.float32),
        )


def test_fit_psf_shape_requires_variance():
    shape = (40, 40)
    score = np.zeros(shape, np.float32)
    score[18:23, 18:23] = 8.0
    diff = score.copy()
    with pytest.raises(ValueError, match="variance"):
        catalog.build_catalog(
            score, diff, psf=_gaussian_stamp(11, 1.8), threshold_sigma=5.0, fit_psf_shape=True
        )


def test_max_shape_fit_guard_skips_enrichment_above_the_cap():
    shape = (60, 60)
    psf = _gaussian_stamp(11, 1.8)
    amplitude = 500.0
    r = psf.shape[0] // 2
    diff = np.zeros(shape, np.float32)
    diff[30 - r : 30 + r + 1, 30 - r : 30 + r + 1] = amplitude * psf
    score = _gaussian_blob(shape, 30, 30, 8.0, 1.8)
    variance = np.ones(shape, np.float32)

    cat = catalog.build_catalog(
        score,
        diff,
        psf=psf,
        threshold_sigma=5.0,
        fit_psf_shape=True,
        variance=variance,
        max_shape_fit=0,  # cap below the 1 candidate present -> guard trips
    )
    assert len(cat) == 1
    assert cat["psf_chi2_dof"][0] == 0
    assert bool(cat["shape_consistent"][0])  # default, not evidence of a fit


# ---- Bright-star proximity flag --------------------------------------------


def test_near_bright_star_flag_within_radius():
    shape = (60, 60)
    score = np.zeros(shape, np.float32)
    score[28:33, 28:33] = 8.0  # centroid ~ (30, 30)
    diff = score.copy()
    cat = catalog.build_catalog(
        score,
        diff,
        expected_fwhm=3.0,
        threshold_sigma=5.0,
        bright_x=np.array([32.0]),
        bright_y=np.array([30.0]),
        bright_star_radius=5.0,
    )
    assert len(cat) == 1
    assert cat["bright_star_dist"][0] == pytest.approx(2.0, abs=0.5)
    assert bool(cat["near_bright_star"][0])
    assert cat["quality"][0] & catalog.QUALITY_NEAR_BRIGHT_STAR


def test_far_from_bright_star_is_not_flagged():
    shape = (60, 60)
    score = np.zeros(shape, np.float32)
    score[28:33, 28:33] = 8.0
    diff = score.copy()
    cat = catalog.build_catalog(
        score,
        diff,
        expected_fwhm=3.0,
        threshold_sigma=5.0,
        bright_x=np.array([59.0]),
        bright_y=np.array([59.0]),
        bright_star_radius=5.0,
    )
    assert len(cat) == 1
    assert not bool(cat["near_bright_star"][0])
    assert np.isfinite(cat["bright_star_dist"][0])


def test_bright_star_dist_is_inf_without_bright_star_input():
    shape = (60, 60)
    score = np.zeros(shape, np.float32)
    score[28:33, 28:33] = 8.0
    diff = score.copy()
    cat = catalog.build_catalog(score, diff, expected_fwhm=3.0, threshold_sigma=5.0)
    assert np.isinf(cat["bright_star_dist"][0])
    assert not bool(cat["near_bright_star"][0])


# ---- Dual-polarity catalog --------------------------------------------------


def test_polarity_negative_returns_the_negative_blob_catalog():
    shape = (60, 60)
    score = np.zeros(shape, np.float32)
    score[28:33, 28:33] = -8.0  # a fading/disappearing source
    diff = -score.copy()
    cat_pos = catalog.build_catalog(
        score, diff, expected_fwhm=3.0, threshold_sigma=5.0, polarity="positive"
    )
    cat_neg = catalog.build_catalog(
        score, diff, expected_fwhm=3.0, threshold_sigma=5.0, polarity="negative"
    )
    assert len(cat_pos) == 0
    assert len(cat_neg) == 1
    assert cat_neg["peak_snr"][0] < 0


def test_is_dipole_is_symmetric_across_polarities():
    """A dipole pair should be flagged on both its positive and negative row
    in a `polarity="both"` catalog, not just the positive one."""
    shape = (60, 60)
    score = np.zeros(shape, np.float32)
    score[28:33, 28:33] = 8.0  # positive block, cols/rows 28..32
    score[30, 33] = -4.0  # negative pixel touching the block's east edge
    diff = score.copy()
    cat = catalog.build_catalog(
        score,
        diff,
        expected_fwhm=3.0,
        threshold_sigma=5.0,
        threshold_sigma_dipole=3.0,
        polarity="both",
    )
    assert len(cat) == 2
    assert np.all(cat["is_dipole"])


def test_polarity_both_merges_with_a_sign_column():
    shape = (80, 80)
    score = np.zeros(shape, np.float32)
    score[18:23, 18:23] = 8.0
    score[58:63, 58:63] = -8.0
    diff = score.copy()
    cat = catalog.build_catalog(
        score, diff, expected_fwhm=3.0, threshold_sigma=5.0, polarity="both"
    )
    assert len(cat) == 2
    assert "sign" in cat.dtype.names
    assert set(cat["sign"]) == {1, -1}
    assert set(cat.dtype.names) == set(catalog.CATALOG_DTYPE.names or ()) | {"sign"}


def test_polarity_invalid_value_raises():
    shape = (40, 40)
    score = np.zeros(shape, np.float32)
    diff = score.copy()
    with pytest.raises(ValueError, match="polarity"):
        catalog.build_catalog(
            score, diff, expected_fwhm=3.0, threshold_sigma=5.0, polarity="sideways"
        )


# ---- DiffResult.build_catalog() wiring for the new options -----------------


def test_diffresult_build_catalog_fit_psf_shape_end_to_end():
    ref, sci, var, sig_sci, rng, sigma_noise = _static_pair(seed=21)
    position = (180, 180)
    sci = validation.inject(sci, [position], [30000.0], sig_sci)
    ref, sci = _finalize(ref, sci, rng, sigma_noise)

    res = delta.subtract(
        sci,
        ref,
        science_var=var,
        reference_var=var,
        n_knots=4,
        stamp_radius=12,
        decorrelate=True,
        score=True,
        block=128,
    )
    cat = res.build_catalog(threshold_sigma=5.0, fit_psf_shape=True)
    dist = np.hypot(cat["x"] - position[0], cat["y"] - position[1])
    injected = cat[dist.argmin()]
    assert dist.min() < 3.0
    assert injected["psf_chi2_dof"] > 0
    # Not asserting shape_consistent here: `self.psf` is one representative,
    # frame-centre-whitened profile (matching what the matched filter itself
    # scored with -- see catalog.build_catalog's fit_psf_shape docs), so a
    # real injected source under real spatial kernel variation can
    # legitimately land above the chi^2 tolerance. What matters is that the
    # fit ran and recovered a sane, positive, well-determined amplitude.
    assert injected["psf_amplitude"] > 0
    assert 0 < injected["psf_amplitude_err"] < injected["psf_amplitude"]


def test_diffresult_build_catalog_auto_sources_bright_stars_from_solution():
    """Omitting bright_x/bright_y should default to solution.stamp_x/stamp_y
    (a free input already computed by the fit), not to "no bright-star check".
    """
    ref, sci, var, sig_sci, rng, sigma_noise = _static_pair(seed=22)
    ref, sci = _finalize(ref, sci, rng, sigma_noise)

    res = delta.subtract(
        sci,
        ref,
        science_var=var,
        reference_var=var,
        n_knots=4,
        stamp_radius=12,
        decorrelate=True,
        score=True,
        block=128,
    )
    assert res.solution.stamp_x is not None and len(res.solution.stamp_x) > 0
    assert res.solution.stamp_y is not None

    cat_auto = res.build_catalog(threshold_sigma=5.0, bright_star_radius=5.0)
    cat_explicit = res.build_catalog(
        threshold_sigma=5.0,
        bright_star_radius=5.0,
        bright_x=res.solution.stamp_x.astype(float),
        bright_y=res.solution.stamp_y.astype(float),
    )
    cat_opt_out = res.build_catalog(
        threshold_sigma=5.0,
        bright_star_radius=5.0,
        bright_x=np.array([]),
        bright_y=np.array([]),
    )
    np.testing.assert_array_equal(cat_auto["bright_star_dist"], cat_explicit["bright_star_dist"])
    assert np.all(np.isinf(cat_opt_out["bright_star_dist"]))
