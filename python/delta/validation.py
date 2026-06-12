"""Synthetic source injection and recovery measurement (SPEC §10).

Reusable helpers for the injection–recovery validation suite and benchmarks:
render Gaussian point sources, inject them into a frame, and measure peaks /
aperture fluxes / completeness on the difference or score image.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FWHM_TO_SIGMA = 1.0 / 2.35482


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
    for (x, y), f in zip(positions, fluxes, strict=True):
        out += gaussian_psf(out.shape, x, y, f, sigma)
    return out.astype(image.dtype, copy=False)


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
