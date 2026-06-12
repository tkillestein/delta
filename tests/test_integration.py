"""Large-frame end-to-end integration test (SPEC §13.3).

The unit suite only exercises ~200px frames. This differences a survey-scale frame
built from a *realistic* random star field (random positions, power-law fluxes —
`delta.validation.sample_starfield`) at two seeings, with an injected transient and
a shot-noise-aware variance model, then checks that:

  * the static field is suppressed across *all* bright stars (not just one),
  * the injected transient is recovered in both the difference and the decorrelated
    score image,
  * a source-free region of the score is ~unit-Gaussian (no field-wide artifacts),
  * the run completes within a generous wall-clock budget (an O(N²) regression guard).

Frame size defaults to 1536px and is overridable via ``DELTA_INTEGRATION_SIZE`` —
set it to 8192 to validate at the full reference frame size (SPEC §13.3).
"""

import os
import time

import delta
import numpy as np
from delta import validation

SIZE = int(os.environ.get("DELTA_INTEGRATION_SIZE", "1536"))
SIGMA_BORDER = 24  # keep sources clear of the frame edge / grown mask


def _starfield_pair(size, sig_ref, sig_sci, gain, read_noise, rng):
    """A static star field rendered at two seeings, with matched shot+read noise.

    Returns the noisy science/reference frames, their (true) variance maps, and the
    star catalog (positions, per-image peak amplitudes) for residual checks. The
    variance is the true per-pixel noise so the match-filtered score is correctly
    normalised; noise is drawn to match it.
    """
    shape = (size, size)
    background = 200.0
    # ~1 star per (70px)^2 cell, kept loosely separated so stamps stay isolated.
    n_stars = max(16, (size // 70) ** 2)
    positions, fluxes = validation.sample_starfield(
        shape,
        n_stars,
        rng,
        flux_range=(300.0, 80000.0),
        border=2 * SIGMA_BORDER,
        min_separation=24.0,
    )

    ref_signal = background + validation.render_stars(shape, positions, fluxes, sig_ref)
    sci_signal = background + validation.render_stars(shape, positions, fluxes, sig_sci)
    return positions, fluxes, ref_signal, sci_signal


def _add_noise(signal, gain, read_noise, rng):
    """True shot+read variance from a noiseless signal, plus a matching noise draw."""
    var = np.maximum(signal, 0.0) / gain + read_noise**2
    noisy = signal + rng.normal(0.0, np.sqrt(var))
    return noisy.astype(np.float32), var.astype(np.float32)


def test_large_frame_end_to_end(preview):
    rng = np.random.default_rng(7)
    gain, read_noise = 1.6, 5.0
    sig_ref, sig_sci = 1.7, 2.6  # reference sharper -> it is the convolved image

    positions, fluxes, ref_signal, sci_signal = _starfield_pair(
        SIZE, sig_ref, sig_sci, gain, read_noise, rng
    )

    # Inject one bright transient into the science frame, in a clear patch.
    tx, ty = int(0.62 * SIZE), int(0.37 * SIZE)
    trans_flux = 30000.0
    sci_signal = sci_signal + validation.render_stars(
        (SIZE, SIZE), [(tx, ty)], [trans_flux], sig_sci
    )
    trans_peak = trans_flux / (2.0 * np.pi * sig_sci**2)

    sci, sci_var = _add_noise(sci_signal, gain, read_noise, rng)
    ref, ref_var = _add_noise(ref_signal, gain, read_noise, rng)

    t0 = time.perf_counter()
    res = delta.subtract(
        sci,
        ref,
        science_var=sci_var,
        reference_var=ref_var,
        n_knots=5,
        stamp_radius=15,
        decorrelate=True,
        score=True,
        block=256,
    )
    elapsed = time.perf_counter() - t0

    preview(
        "integration_large_frame",
        science=sci,
        reference=ref,
        difference=res.difference,
        score=res.score,
    )

    # --- shapes / direction / finiteness ----------------------------------------
    assert res.difference.shape == (SIZE, SIZE)
    assert res.solution.direction == "reference"
    assert res.variance is not None and res.score is not None
    # Difference is finite everywhere outside the grown edge mask.
    interior = res.difference[SIGMA_BORDER:-SIGMA_BORDER, SIGMA_BORDER:-SIGMA_BORDER]
    assert np.all(np.isfinite(interior))

    # --- static suppression across the whole field ------------------------------
    # For each bright star, the difference residual at its centre should be a small
    # fraction of its (per-image) peak amplitude. Check the population, not one star.
    peak_amp = fluxes / (2.0 * np.pi * sig_sci**2)
    bright = peak_amp > 0.2 * trans_peak
    ratios = []
    for (x, y), amp in zip(positions[bright], peak_amp[bright], strict=True):
        resid, _ = validation.peak_near(np.abs(res.difference), (int(x), int(y)), radius=4)
        ratios.append(resid / amp)
    ratios = np.array(ratios)
    assert np.median(ratios) < 0.15
    assert np.percentile(ratios, 90) < 0.40

    # --- transient recovery ------------------------------------------------------
    diff_peak, _ = validation.peak_near(res.difference, (tx, ty), radius=5)
    assert diff_peak > 0.4 * trans_peak  # positive and prominent in the difference
    score_peak, found = validation.peak_near(res.score, (tx, ty), radius=5)
    assert score_peak > 5.0  # detected at >5 sigma in the calibrated score
    assert abs(found[0] - tx) <= 2 and abs(found[1] - ty) <= 2  # astrometry

    # --- score statistics in a source-free corner -------------------------------
    corner = res.score[
        SIGMA_BORDER : SIGMA_BORDER + 200, SIZE - 200 - SIGMA_BORDER : SIZE - SIGMA_BORDER
    ]
    assert abs(corner.mean()) < 0.5
    assert 0.6 < corner.std() < 1.6

    # --- runtime guard (catches catastrophic O(N^2) regressions, not perf drift) -
    budget = 60.0 + SIZE * SIZE / 5.0e4
    assert elapsed < budget, f"subtract took {elapsed:.1f}s (> {budget:.0f}s budget)"
