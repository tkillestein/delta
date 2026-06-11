"""Type stubs for the compiled ``delta._core`` extension.

Hand-written for the M0 scaffold; once the API grows these can be generated from
nanobind (``nanobind_add_stub``).
"""

import numpy as np
from numpy.typing import NDArray

__version__: str

def weighted_mean(
    data: NDArray[np.float32],
    variance: NDArray[np.float32],
    mask: NDArray[np.uint8],
) -> float:
    """Inverse-variance weighted mean over unmasked pixels."""
