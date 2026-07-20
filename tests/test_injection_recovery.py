"""M9 injection-recovery suite (SPEC §10): completeness, photometry, statistics.

Inject synthetic transients across a range of fluxes into the science frame, run
the full pipeline with decorrelation + score, and check that bright sources are
recovered, faint ones are not, recovered flux tracks injected flux, and the
source-free score is ~unit-Gaussian.

Noise is generated to match the variance maps handed to the pipeline, so the
match-filtered score is correctly normalised to unit variance.
"""

import delta
import numpy as np
from delta import catalog, validation


def _static_pair(sig_ref=1.6, sig_sci=2.4, sigma_noise=3.0, seed=0):
    """Static star field at two seeings plus matched constant-variance noise."""
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


def test_injection_recovery_completeness_and_photometry(preview):
    ref, sci, var, sig_sci, rng, sigma_noise = _static_pair()
    positions = [(x, y) for x in (70, 110, 150) for y in (70, 140, 200)]
    fluxes = np.sort(np.logspace(np.log10(50.0), np.log10(40000.0), len(positions)))
    sci = validation.inject(sci, positions, fluxes, sig_sci)
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
    assert res.score is not None
    preview(
        "injection_recovery",
        science=sci,
        reference=ref,
        difference=res.difference,
        score=res.score,
    )

    # Completeness: the brightest third all detected, the faintest third barely.
    recovered = validation.completeness(res.score, positions, threshold=5.0, radius=4)
    third = len(positions) // 3
    assert recovered[-third:].all()
    assert recovered[:third].sum() <= 1

    # Photometry: aperture flux on the difference tracks injected flux.
    ap_r = int(round(3 * sig_sci))
    measured = np.array([validation.aperture_flux(res.difference, xy, ap_r) for xy in positions])
    assert np.corrcoef(fluxes, measured)[0, 1] > 0.97
    # Astrometry: the brightest source peaks within ~2px of injection.
    _, found = validation.peak_near(res.difference, positions[-1], radius=5)
    assert abs(found[0] - positions[-1][0]) <= 2
    assert abs(found[1] - positions[-1][1]) <= 2


def test_score_is_unit_gaussian_in_source_free_region():
    ref, sci, var, _, rng, sigma_noise = _static_pair(seed=3)
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
    # A source-free corner of the score image should be ~N(0, 1): unit-variance
    # (the whitening check) and roughly centred.
    assert res.score is not None
    corner = res.score[10:70, 10:70]
    assert abs(corner.mean()) < 0.5
    assert 0.6 < corner.std() < 1.5


# ---- Catalog-level injection-recovery (SPEC §3.7) --------------------------
#
# The tests above validate the score image directly. These validate the actual
# end-user product -- `source_catalog` / `build_catalog()` -- the same way:
# completeness vs. injected flux, purity on a clean field, and whether the
# dipole/shape/mask vetoes actually distinguish a genuine source from a
# manufactured subtraction artifact on real (not synthetic-array) pipeline
# output.


def test_catalog_completeness_across_flux_range():
    ref, sci, var, sig_sci, rng, sigma_noise = _static_pair(seed=30)
    positions = [(x, y) for x in (70, 110, 150) for y in (70, 140, 200)]
    fluxes = np.sort(np.logspace(np.log10(50.0), np.log10(40000.0), len(positions)))
    sci = validation.inject(sci, positions, fluxes, sig_sci)
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
    cat = res.source_catalog
    assert cat is not None

    def recovered(xy, radius=4):
        if len(cat) == 0:
            return False
        dist = np.hypot(cat["x"] - xy[0], cat["y"] - xy[1])
        return bool(dist.min() <= radius)

    flags = np.array([recovered(p) for p in positions])
    third = len(positions) // 3
    # Same completeness shape as the score-based check, but through the
    # thresholded/labelled catalog product actual callers use.
    assert flags[-third:].all()
    assert flags[:third].sum() <= 1


def test_catalog_purity_on_a_clean_field_with_no_injected_transients():
    """No injected transients, a well-matched PSF: the catalog built from a
    real pipeline run (not a synthetic score array) should be near-empty, not
    full of subtraction-residual false positives."""
    ref, sci, var, _, rng, sigma_noise = _static_pair(seed=31)
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
    assert len(res.source_catalog) <= 2


def test_catalog_flags_a_misregistration_dipole_but_not_a_clean_source():
    """A source injected at slightly different (x, y) in science vs. reference
    is the classic misregistration failure mode: the mismatch doesn't cancel
    on subtraction and leaves a positive/negative dipole residual. A genuine
    transient of similar brightness (injected into science only -- present in
    one epoch, not the other, exactly like the completeness-test sources)
    should come back without that flag."""
    sig_ref = 1.6
    ref, sci, var, sig_sci, rng, sigma_noise = _static_pair(seed=32, sig_ref=sig_ref)
    clean_pos = (80.0, 150.0)
    dipole_sci_pos = (150.0, 80.0)
    dipole_ref_pos = (153.0, 80.0)  # 3px offset: beyond match_radius, so this
    # isn't cross-matched into a fit stamp -- it's purely a post-fit
    # convolution/subtraction artifact, not a corrupted kernel solve.
    flux = 20000.0
    # clean_pos: science only (a real transient -- present in one epoch).
    # dipole_*_pos: both epochs but offset (a non-varying star that fails to
    # cancel because of the position mismatch, not because it's a transient).
    sci = validation.inject(sci, [clean_pos, dipole_sci_pos], [flux, flux], sig_sci)
    ref = validation.inject(ref, [dipole_ref_pos], [flux], sig_ref)
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
    cat = res.build_catalog(threshold_sigma=5.0, threshold_sigma_dipole=3.0)

    dist_clean = np.hypot(cat["x"] - clean_pos[0], cat["y"] - clean_pos[1])
    dist_dipole = np.hypot(cat["x"] - dipole_sci_pos[0], cat["y"] - dipole_sci_pos[1])
    assert dist_clean.min() < 3.0
    assert dist_dipole.min() < 3.0
    clean_entry = cat[dist_clean.argmin()]
    dipole_entry = cat[dist_dipole.argmin()]

    assert not bool(clean_entry["is_dipole"])
    assert bool(dipole_entry["is_dipole"])
    assert dipole_entry["quality"] & catalog.QUALITY_DIPOLE
