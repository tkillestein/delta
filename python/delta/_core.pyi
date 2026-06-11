"""Type stubs for the compiled ``delta._core`` extension.

Hand-written for the M0 scaffold; once the API grows these can be generated from
nanobind (``nanobind_add_stub``).
"""

from typing import TypedDict

import numpy as np
from numpy.typing import NDArray

__version__: str

class FitsLayers(TypedDict):
    data: NDArray[np.float32]
    variance: NDArray[np.float32] | None
    mask: NDArray[np.uint8] | None

class DetectedStamps(TypedDict):
    x: NDArray[np.int32]
    y: NDArray[np.int32]
    flux: NDArray[np.float64]
    snr: NDArray[np.float64]
    fwhm: NDArray[np.float64]

class StampSelection(TypedDict):
    x: NDArray[np.int32]
    y: NDArray[np.int32]
    fwhm_science: NDArray[np.float64]
    fwhm_reference: NDArray[np.float64]
    median_fwhm_science: float
    median_fwhm_reference: float
    direction: str

def weighted_mean(
    data: NDArray[np.float32],
    variance: NDArray[np.float32],
    mask: NDArray[np.uint8],
) -> float:
    """Inverse-variance weighted mean over unmasked pixels."""

def read_fits(path: str) -> FitsLayers:
    """Read a FITS file into {'data', 'variance', 'mask'} arrays."""

def write_fits(
    path: str,
    data: NDArray[np.float32],
    variance: NDArray[np.float32] | None = ...,
    mask: NDArray[np.uint8] | None = ...,
) -> None:
    """Write data plus optional variance/mask layers to a FITS file."""

def gauss_hermite_basis1d(beta: float, n_max: int, radius: int = ...) -> NDArray[np.float64]:
    """1-D sampled Gauss-Hermite basis functions, shape (n_max+1, ksize)."""

def gauss_hermite_kernels(
    beta: float, n_max: int, radius: int = ...
) -> tuple[NDArray[np.int32], NDArray[np.float32]]:
    """(orders, kernels) for the 2-D Gauss-Hermite components."""

def basis_convolve(
    image: NDArray[np.float32], beta: float, n_max: int, radius: int = ...
) -> NDArray[np.float32]:
    """Convolve image with each basis component; returns (ncomp, H, W)."""

def estimate_background(
    data: NDArray[np.float32], mask: NDArray[np.uint8] | None = ...
) -> tuple[float, float]:
    """Robust (median, MAD-sigma) background over unmasked pixels."""

def detect_stamps(
    data: NDArray[np.float32],
    mask: NDArray[np.uint8] | None = ...,
    stamp_radius: int = ...,
    threshold_sigma: float = ...,
    max_stamps: int = ...,
    saturation: float = ...,
    isolation_radius: int = ...,
    border: int = ...,
) -> DetectedStamps:
    """Detect bright, isolated, unsaturated point-source stamps."""

def select_stamps(
    science: NDArray[np.float32],
    reference: NDArray[np.float32],
    science_mask: NDArray[np.uint8] | None = ...,
    reference_mask: NDArray[np.uint8] | None = ...,
    catalog_x: NDArray[np.int32] | None = ...,
    catalog_y: NDArray[np.int32] | None = ...,
    stamp_radius: int = ...,
    threshold_sigma: float = ...,
    max_stamps: int = ...,
    saturation: float = ...,
    isolation_radius: int = ...,
    border: int = ...,
) -> StampSelection:
    """Select matched stamps across both images and the convolution direction."""
