"""Synthetic source injection and recovery measurement (SPEC §10).

Reusable helpers for the injection–recovery validation suite and benchmarks:
render Gaussian point sources, inject them into a frame, and measure peaks /
aperture fluxes / completeness on the difference or score image.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# 1 / (2*sqrt(2 ln 2)); matches kFwhmPerSigma in src/detect.cpp.
FWHM_TO_SIGMA = 1.0 / 2.354820045030949


def gaussian_psf(
    shape: tuple[int, int], x: float, y: float, flux: float, sigma: float
) -> NDArray[np.float64]:
    """A Gaussian point source of given total `flux` and `sigma`, on a blank frame."""
    h, w = shape
    ys, xs = np.mgrid[0:h, 0:w]
    amp = flux / (2.0 * np.pi * sigma**2)
    return amp * np.exp(-((xs - x) ** 2 + (ys - y) ** 2) / (2.0 * sigma**2))


def inject(image, positions, fluxes, sigma) -> NDArray:
    """Return a copy of `image` with Gaussian sources added at `positions`."""
    out = np.array(image, dtype=np.float64, copy=True)
    out += render_stars(out.shape, positions, fluxes, sigma)
    return out.astype(image.dtype, copy=False)


def render_stars(shape, positions, fluxes, sigma, radius: int | None = None) -> NDArray[np.float64]:
    """Render Gaussian point sources onto a blank frame.

    Each source is drawn only within a local box (``radius`` ~ ``5*sigma`` by
    default), so this stays cheap on survey-scale frames where a full-frame mgrid
    per star would be prohibitive. Positions may be sub-pixel.
    """
    h, w = shape
    out = np.zeros((h, w), dtype=np.float64)
    if radius is None:
        radius = int(np.ceil(5.0 * sigma))
    norm = 1.0 / (2.0 * np.pi * sigma**2)
    two_s2 = 2.0 * sigma**2
    for (x, y), f in zip(positions, fluxes, strict=True):
        ix, iy = int(round(x)), int(round(y))
        x0, x1 = max(0, ix - radius), min(w, ix + radius + 1)
        y0, y1 = max(0, iy - radius), min(h, iy + radius + 1)
        if x0 >= x1 or y0 >= y1:
            continue
        ys, xs = np.mgrid[y0:y1, x0:x1]
        out[y0:y1, x0:x1] += (f * norm) * np.exp(-((xs - x) ** 2 + (ys - y) ** 2) / two_s2)
    return out


def sample_starfield(
    shape,
    n_stars: int,
    rng,
    flux_range: tuple[float, float] = (200.0, 60000.0),
    slope: float = 1.8,
    border: int = 16,
    min_separation: float = 0.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Sample a realistic star field: random positions, power-law fluxes.

    Returns ``(positions (N, 2), fluxes (N,))``. Fluxes follow a bounded power law
    ``p(F) ∝ F**-slope`` over ``flux_range`` (many faint, few bright — a stellar
    luminosity function), far more realistic than a regular grid of equal-flux
    sources. Positions are uniform within a ``border`` margin; ``min_separation``
    rejection-samples to limit crowding so stamp selection has isolated sources.
    """
    rng = np.random.default_rng(rng)
    h, w = shape
    fmin, fmax = flux_range

    # Inverse-CDF sampling of a bounded power law p(F) ∝ F**-slope.
    u = rng.random(n_stars)
    if abs(slope - 1.0) < 1e-9:
        fluxes = fmin * (fmax / fmin) ** u
    else:
        a = 1.0 - slope
        fluxes = (fmin**a + u * (fmax**a - fmin**a)) ** (1.0 / a)

    xs = np.empty(n_stars)
    ys = np.empty(n_stars)
    min_sep2 = min_separation**2
    count = 0
    attempts = 0
    max_attempts = 100 * n_stars
    while count < n_stars and attempts < max_attempts:
        attempts += 1
        x = rng.uniform(border, w - 1 - border)
        y = rng.uniform(border, h - 1 - border)
        if min_sep2 > 0.0 and count > 0:
            d2 = (xs[:count] - x) ** 2 + (ys[:count] - y) ** 2
            if d2.min() < min_sep2:
                continue
        xs[count], ys[count] = x, y
        count += 1

    positions = np.column_stack([xs[:count], ys[:count]])
    return positions, fluxes[:count]


def peak_near(image, xy, radius: int) -> tuple[float, tuple[int, int]]:
    """Peak value and its (x, y) location within `radius` of `xy`."""
    x, y = xy
    win = image[y - radius : y + radius + 1, x - radius : x + radius + 1]
    iy, ix = np.unravel_index(int(np.argmax(win)), win.shape)
    return float(win[iy, ix]), (x - radius + int(ix), y - radius + int(iy))


def aperture_flux(image, xy, radius: int) -> float:
    """Sum of pixels within a circular aperture of `radius` about `xy`."""
    x, y = xy
    ys, xs = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    disk = xs**2 + ys**2 <= radius**2
    win = image[y - radius : y + radius + 1, x - radius : x + radius + 1]
    return float(np.asarray(win, dtype=np.float64)[disk].sum())


def completeness(score, positions, threshold: float, radius: int = 3) -> NDArray[np.bool_]:
    """Boolean recovery flag per position: score peak within `radius` >= threshold."""
    return np.array([peak_near(score, xy, radius)[0] >= threshold for xy in positions], dtype=bool)
