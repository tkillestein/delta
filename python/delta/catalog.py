"""Source/transient catalog from the score image (SPEC §3.7).

Thin wrapper around the C++ connected-component labeller (`_core.build_catalog`):
coerces inputs, resolves the expected PSF FWHM used for the consistency filter
and default aperture radius, and packs the result into a NumPy structured
array (or an `astropy.table.Table`).
"""

from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import NDArray

from . import _core
from .validation import FWHM_TO_SIGMA

CATALOG_DTYPE = np.dtype(
    [
        ("x", np.float64),
        ("y", np.float64),
        ("peak_x", np.int32),
        ("peak_y", np.int32),
        ("peak_snr", np.float64),
        ("n_pix", np.int32),
        ("flux", np.float64),
        ("expected_n_pix", np.float64),
        ("fwhm_ratio", np.float64),
        ("fwhm_consistent", np.bool_),
        ("mask_flags", np.uint8),
        ("is_dipole", np.bool_),
        # PSF-shape chi^2 fit (populated only when `fit_psf_shape=True`; a
        # 1-parameter weighted fit of `psf` against the pixels around the
        # candidate, on the pixel grid -- no subpixel registration). Left at
        # (0.0, 0, 0.0, 0.0, True) otherwise: absence of evidence, not
        # evidence of a good fit.
        ("psf_chi2", np.float64),
        ("psf_chi2_dof", np.int32),
        ("psf_amplitude", np.float64),
        ("psf_amplitude_err", np.float64),
        ("shape_consistent", np.bool_),
        # Bright-star proximity (populated only when `bright_x`/`bright_y`
        # are supplied). A flag, not a rejection -- real transients do occur
        # near bright stars.
        ("bright_star_dist", np.float64),
        ("near_bright_star", np.bool_),
        ("quality", np.uint8),
    ]
)

# quality bit layout (see include/delta/catalog.hpp CatalogEntry::quality):
QUALITY_FWHM_INCONSISTENT = 1 << 0
QUALITY_DIPOLE = 1 << 1
QUALITY_MASKED = 1 << 2
QUALITY_SHAPE_INCONSISTENT = 1 << 3  # only meaningful when fit_psf_shape=True
QUALITY_NEAR_BRIGHT_STAR = 1 << 4  # only meaningful when bright_x/bright_y given


def _measure_psf_fwhm(psf: NDArray[np.floating]) -> float:
    """Second-moment FWHM (px) of a small, background-subtracted PSF stamp."""
    psf = np.asarray(psf, dtype=np.float64)
    h, w = psf.shape
    ys, xs = np.mgrid[0:h, 0:w]
    weight = np.clip(psf, 0.0, None)
    total = weight.sum()
    if total <= 0:
        raise ValueError("PSF stamp has no positive flux")
    mx = (weight * xs).sum() / total
    my = (weight * ys).sum() / total
    varx = (weight * (xs - mx) ** 2).sum() / total
    vary = (weight * (ys - my) ** 2).sum() / total
    sigma = np.sqrt(0.5 * (varx + vary))
    return float(sigma / FWHM_TO_SIGMA)


def _pack(raw: _core.CatalogResult) -> NDArray[np.void]:
    raw_columns = cast("dict[str, NDArray]", raw)
    n = raw_columns["x"].shape[0]
    # `np.empty(n, dtype=CATALOG_DTYPE)` isn't recognised as a structured
    # (void-dtype) array by static type checkers since CATALOG_DTYPE is a
    # runtime np.dtype value, not a literal -- cast explicitly so the by-name
    # field assignment below type-checks.
    cat = cast("NDArray[np.void]", np.empty(n, dtype=CATALOG_DTYPE))
    for name in CATALOG_DTYPE.names or ():
        cat[name] = raw_columns[name]
    return cat


def build_catalog(
    score: NDArray[np.float32],
    difference: NDArray[np.float32],
    *,
    mask: NDArray[np.uint8] | None = None,
    psf: NDArray[np.floating] | None = None,
    expected_fwhm: float | None = None,
    threshold_sigma: float = 5.0,
    threshold_sigma_dipole: float = 3.0,
    fwhm_tolerance_lo: float = 0.3,
    fwhm_tolerance_hi: float = 3.0,
    aperture_radius: int = 0,
    exclude_bad_pixels: bool = True,
    fit_psf_shape: bool = False,
    shape_chi2_tolerance: float = 3.0,
    variance: NDArray[np.floating] | None = None,
    bright_x: NDArray[np.floating] | NDArray[np.integer] | None = None,
    bright_y: NDArray[np.floating] | NDArray[np.integer] | None = None,
    bright_star_radius: float = 0.0,
    max_shape_fit: int = 5000,
    polarity: str = "positive",
    as_table: bool = False,
):
    """Build a source catalog from a match-filtered score image (SPEC §3.7).

    Connected-component labelling (8-connected) of `score` at
    `score >= threshold_sigma`, with a parallel negative-blob pass at
    `score <= -threshold_sigma_dipole` used purely to flag dipoles
    (`is_dipole`) -- positive components that enclose or touch a negative one,
    evidence of a bad subtraction rather than a transient.

    Exactly one of `psf` (a PSF stamp, e.g. from `pipeline._estimate_psf`, used
    to measure the expected FWHM via second moments) or `expected_fwhm` (a
    precomputed FWHM in pixels) must be supplied; it drives the
    FWHM-consistency filter and the default aperture radius.

    **Choosing `threshold_sigma`.** The score image is normalised so
    source-free pixels are ~unit-Gaussian, so a naive threshold has a
    calibrated (but not zero) false-alarm rate from noise alone: the number
    of independent resolution elements on a frame is roughly
    ``n_pixels / (pi * (expected_fwhm / 2.3548)**2)``, so on a survey-scale
    8000x6000 frame at the default `expected_fwhm=3.5`, ``threshold_sigma=5``
    leaves ~1-2 pure-noise candidates per frame, `5.5` leaves ~0.1, `6` is
    negligible. In practice, most false positives on real data are *not*
    Gaussian-tail noise but structured subtraction defects (cosmic rays,
    misregistration dipoles, bright-star residuals) that don't shrink much
    with a higher threshold -- `is_dipole`, `fit_psf_shape`, `mask_flags` and
    `bright_x`/`bright_y` target those directly and matter more than pushing
    `threshold_sigma` up, which mainly costs faint real detections.

    **`fit_psf_shape`** (opt-in; needs `psf` -- not `expected_fwhm` -- and
    `variance`) adds a per-candidate weighted 1-parameter fit of the PSF
    stamp against the pixels around each candidate: `psf_chi2`/`psf_chi2_dof`
    (a real shape statistic, unlike the pixel-count-based `fwhm_ratio`),
    `psf_amplitude`/`psf_amplitude_err` (a calibrated flux, complementing the
    shape-agnostic aperture `flux`), and `shape_consistent` (reduced chi^2 <=
    `shape_chi2_tolerance`). It reuses the exact PSF profile the matched
    filter scored with (pass `DiffResult.psf`), and only runs over the
    already-thresholded candidate list -- cost tracks detection count, not
    frame size -- capped at `max_shape_fit` candidates as a guard against a
    pathologically low threshold or a broken frame turning that list huge.

    **`bright_x`/`bright_y`** (opt-in; e.g. `KernelSolution.stamp_x/stamp_y`,
    the bright stars actually used for the kernel fit) add `bright_star_dist`
    and `near_bright_star` (within `bright_star_radius`, default
    `3 * expected_fwhm`): subtraction residuals cluster near bright stars
    beyond what `mask_flags` catches. A flag, not a rejection -- real
    transients do occur near bright stars.

    **`polarity`** selects `"positive"` (default; transients get brighter),
    `"negative"` (transients get fainter/disappear), or `"both"` (concatenates
    both, with an added `sign` column, +1/-1). The negative-polarity pass is
    seeded/grown at `threshold_sigma_dipole`, not `threshold_sigma` -- it
    reuses the same negative-blob pass the dipole check already runs, rather
    than adding a third threshold. Set `threshold_sigma_dipole ==
    threshold_sigma` for a symmetric significance cut across both polarities.
    `is_dipole` is reported symmetrically for both polarities (a component of
    either sign touching/enclosing an opposite-sign one).

    Returns a structured NumPy array (`CATALOG_DTYPE`, or `CATALOG_DTYPE` plus
    a `sign` column for `polarity="both"`), or an `astropy.table.Table` if
    `as_table=True` (optional dependency).
    """
    if (psf is None) == (expected_fwhm is None):
        raise ValueError("supply exactly one of `psf` or `expected_fwhm`")
    if fit_psf_shape and psf is None:
        raise ValueError("fit_psf_shape=True needs `psf` (not `expected_fwhm`)")
    if fit_psf_shape and variance is None:
        raise ValueError("fit_psf_shape=True needs `variance`")
    if polarity not in ("positive", "negative", "both"):
        raise ValueError('polarity must be one of "positive", "negative", "both"')
    fwhm = _measure_psf_fwhm(psf) if psf is not None else float(expected_fwhm or 0.0)

    score = np.ascontiguousarray(score, dtype=np.float32)
    difference = np.ascontiguousarray(difference, dtype=np.float32)
    if mask is not None:
        mask = np.ascontiguousarray(mask, dtype=np.uint8)
    if variance is not None:
        variance = np.ascontiguousarray(variance, dtype=np.float32)
    psf_arr = np.ascontiguousarray(psf, dtype=np.float32) if psf is not None else None
    bx = np.ascontiguousarray(bright_x, dtype=np.float64) if bright_x is not None else None
    by = np.ascontiguousarray(bright_y, dtype=np.float64) if bright_y is not None else None

    def _run(return_negative: bool) -> _core.CatalogResult:
        return _core.build_catalog(
            score,
            difference,
            mask=mask,
            threshold_sigma=threshold_sigma,
            threshold_sigma_dipole=threshold_sigma_dipole,
            expected_fwhm=fwhm,
            fwhm_tolerance_lo=fwhm_tolerance_lo,
            fwhm_tolerance_hi=fwhm_tolerance_hi,
            aperture_radius=aperture_radius,
            exclude_bad_pixels=exclude_bad_pixels,
            return_negative=return_negative,
            fit_psf_shape=fit_psf_shape,
            shape_chi2_tolerance=shape_chi2_tolerance,
            variance=variance,
            psf=psf_arr if fit_psf_shape else None,
            bright_star_radius=bright_star_radius,
            max_shape_fit=max_shape_fit,
            bright_x=bx,
            bright_y=by,
        )

    if polarity == "positive":
        cat = _pack(_run(False))
    elif polarity == "negative":
        cat = _pack(_run(True))
    else:
        pos = _pack(_run(False))
        neg = _pack(_run(True))
        both_dtype = np.dtype([*CATALOG_DTYPE.descr, ("sign", np.int8)])
        cat = cast("NDArray[np.void]", np.empty(len(pos) + len(neg), dtype=both_dtype))
        for name in CATALOG_DTYPE.names or ():
            cat[name] = np.concatenate([pos[name], neg[name]])
        cat["sign"] = np.concatenate([np.ones(len(pos), np.int8), -np.ones(len(neg), np.int8)])

    if as_table:
        from astropy.table import Table  # optional dependency

        return Table(cat)
    return cat
