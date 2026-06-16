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

GlsResult = TypedDict(
    "GlsResult",
    {
        "theta": NDArray[np.float64],
        "n_components": int,
        "n_spatial": int,
        "lambda": float,
        "gcv": float,
        "effective_dof": float,
        "rss": float,
        "lambda_grid": NDArray[np.float64],
        "gcv_curve": NDArray[np.float64],
    },
)

KernelFit = TypedDict(
    "KernelFit",
    {
        "theta": NDArray[np.float64],
        "n_components": int,
        "n_spatial": int,
        "lambda": float,
        "gcv": float,
        "effective_dof": float,
        "rss": float,
        "lambda_grid": NDArray[np.float64],
        "gcv_curve": NDArray[np.float64],
        "component_sums": NDArray[np.float64],
        "n_pixels": int,
        "n_stamps_used": int,
        "n_stamps_total": int,
        "n_stamps_rejected": int,
        "reduced_chi2": float,
        "stamp_x": NDArray[np.int32],
        "stamp_y": NDArray[np.int32],
        "stamp_chi2": NDArray[np.float64],
        "stamp_accepted": NDArray[np.uint8],
        # Opt-in C++ sub-stage timings (DELTA_TIMING); None otherwise.
        "timing": dict[str, float] | None,
    },
)

class DiffProducts(TypedDict):
    difference: NDArray[np.float32]
    variance: NDArray[np.float32] | None
    mask: NDArray[np.uint8] | None
    # Opt-in C++ sub-stage timings (DELTA_TIMING); None otherwise.
    timing: dict[str, float] | None

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

def grid_knots(x0: float, y0: float, x1: float, y1: float, nx: int, ny: int) -> NDArray[np.float64]:
    """Regular nx*ny grid of knots over [x0,x1]x[y0,y1], shape (nx*ny, 2)."""

def tps_design(knots: NDArray[np.float64], points: NDArray[np.float64]) -> NDArray[np.float64]:
    """Thin-plate regression-spline design matrix, shape (m, n_basis)."""

def tps_penalty(knots: NDArray[np.float64]) -> NDArray[np.float64]:
    """Thin-plate bending-energy penalty, shape (n_basis, n_basis)."""

def tps_fit(
    knots: NDArray[np.float64],
    points: NDArray[np.float64],
    values: NDArray[np.float64],
    lam: float,
) -> NDArray[np.float64]:
    """Unweighted penalised fit: (D^T D + lam P) theta = D^T values."""

def tps_evaluate(
    knots: NDArray[np.float64],
    points: NDArray[np.float64],
    coeffs: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Evaluate a fitted TPS field at points: design(points) @ coeffs."""

def solve_gls(
    knots: NDArray[np.float64],
    points: NDArray[np.float64],
    target: NDArray[np.float64],
    weights: NDArray[np.float64],
    bn: NDArray[np.float64],
    lam: float,
) -> GlsResult:
    """Penalised GLS at fixed lambda over the factorized A&L model."""

def solve_gls_gcv(
    knots: NDArray[np.float64],
    points: NDArray[np.float64],
    target: NDArray[np.float64],
    weights: NDArray[np.float64],
    bn: NDArray[np.float64],
    lambda_grid: NDArray[np.float64],
) -> GlsResult:
    """Penalised GLS selecting lambda by GCV over lambda_grid."""

def solve_gls_cv(
    knots: NDArray[np.float64],
    points: NDArray[np.float64],
    target: NDArray[np.float64],
    weights: NDArray[np.float64],
    bn: NDArray[np.float64],
    lambda_grid: NDArray[np.float64],
    group: NDArray[np.int32],
    n_groups: int,
) -> GlsResult:
    """Penalised GLS selecting lambda by k-fold group cross-validation."""

def subtract(
    science: NDArray[np.float32],
    reference: NDArray[np.float32],
    knots: NDArray[np.float64],
    theta: NDArray[np.float64],
    beta: float,
    n_max: int,
    radius: int = ...,
    saturation: float = ...,
    science_var: NDArray[np.float32] | None = ...,
    reference_var: NDArray[np.float32] | None = ...,
    science_mask: NDArray[np.uint8] | None = ...,
    reference_mask: NDArray[np.uint8] | None = ...,
) -> DiffProducts:
    """Full-frame spatially-varying subtraction with variance/mask propagation;
    saturation>0 masks and grows bright/saturated cores."""

def fit_kernel(
    science: NDArray[np.float32],
    reference: NDArray[np.float32],
    knots: NDArray[np.float64],
    stamp_x: NDArray[np.int32],
    stamp_y: NDArray[np.int32],
    stamp_radius: int,
    beta: float,
    n_max: int,
    lambda_grid: NDArray[np.float64],
    radius: int = ...,
    clip_sigma: float = ...,
    clip_iterations: int = ...,
    min_stamps: int = ...,
    cv_folds: int = ...,
    science_var: NDArray[np.float32] | None = ...,
    reference_var: NDArray[np.float32] | None = ...,
    science_mask: NDArray[np.uint8] | None = ...,
    reference_mask: NDArray[np.uint8] | None = ...,
) -> KernelFit:
    """Fit the matching kernel + background via penalised GLS with per-stamp
    sigma clipping. The returned dict adds reduced_chi2, n_stamps_total/used/
    rejected and per-stamp stamp_x/stamp_y/stamp_chi2/stamp_accepted."""

def photometric_scale(
    knots: NDArray[np.float64],
    theta: NDArray[np.float64],
    component_sums: NDArray[np.float64],
    height: int,
    width: int,
) -> NDArray[np.float32]:
    """Per-pixel photometric scale field sum_n a_n(x,y) S_n, shape (H, W)."""

def photometric_scale_at(
    knots: NDArray[np.float64],
    theta: NDArray[np.float64],
    component_sums: NDArray[np.float64],
    points: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Photometric scale evaluated at points, shape (m,)."""

def decorrelate_block(
    image: NDArray[np.float32],
    kernel: NDArray[np.float32],
    var_science: float,
    var_reference: float,
) -> NDArray[np.float32]:
    """Whiten one square block with a constant kernel/noise (FFT core)."""

def decorrelation_kernel(
    kernel: NDArray[np.float32],
    var_science: float,
    var_reference: float,
    n: int,
) -> NDArray[np.float32]:
    """Real-space decorrelation kernel (centred, n x n) for QA."""

def decorrelate(
    difference: NDArray[np.float32],
    knots: NDArray[np.float64],
    theta: NDArray[np.float64],
    beta: float,
    n_max: int,
    var_science: NDArray[np.float32],
    var_reference: NDArray[np.float32],
    block: int = ...,
    radius: int = ...,
    kernel_cell_blocks: int = ...,
) -> NDArray[np.float32]:
    """Spatially-varying noise decorrelation via apodized FFT blocks."""

def matched_filter(
    image: NDArray[np.float32],
    psf: NDArray[np.float32],
    variance: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Match-filtered score image (per-pixel S/N map). variance must be a
    same-shape float32 image of per-pixel noise variance."""
