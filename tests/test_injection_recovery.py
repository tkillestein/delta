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
from delta import validation


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
