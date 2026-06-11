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
