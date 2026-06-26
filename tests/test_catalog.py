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
