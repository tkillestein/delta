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
        ("quality", np.uint8),
    ]
)


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

    Returns a structured NumPy array (`CATALOG_DTYPE`), or an
    `astropy.table.Table` if `as_table=True` (optional dependency).
    """
    if (psf is None) == (expected_fwhm is None):
        raise ValueError("supply exactly one of `psf` or `expected_fwhm`")
    fwhm = _measure_psf_fwhm(psf) if psf is not None else float(expected_fwhm or 0.0)

    score = np.ascontiguousarray(score, dtype=np.float32)
    difference = np.ascontiguousarray(difference, dtype=np.float32)
    if mask is not None:
        mask = np.ascontiguousarray(mask, dtype=np.uint8)

    raw = _core.build_catalog(
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
    )
    raw_columns = cast("dict[str, NDArray]", dict(raw))
    n = raw_columns["x"].shape[0]
    cat = np.empty(n, dtype=CATALOG_DTYPE)
    for name in CATALOG_DTYPE.names or ():
        cat[name] = raw_columns[name]

    if as_table:
        from astropy.table import Table  # optional dependency

        return Table(cat)
    return cat
