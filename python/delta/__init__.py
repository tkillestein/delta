"""delta — high-performance astronomical difference imaging.

A modern, statistically-principled reformulation of the Alard & Lupton (1998)
PSF-matching algorithm. See ``docs/SPEC.md`` for the design.
"""

from __future__ import annotations

from . import validation
from ._core import (
    __version__,
    basis_convolve,
    decorrelate,
    decorrelate_block,
    decorrelation_kernel,
    detect_stamps,
    estimate_background,
    fit_kernel,
    gauss_hermite_basis1d,
    gauss_hermite_kernels,
    grid_knots,
    matched_filter,
    photometric_scale,
    photometric_scale_at,
    read_fits,
    select_stamps,
    solve_gls,
    solve_gls_cv,
    solve_gls_gcv,
    tps_design,
    tps_evaluate,
    tps_fit,
    tps_penalty,
    weighted_mean,
    write_fits,
)
from ._core import subtract as subtract_model
from ._inputs import as_layers, synth_variance
from .pipeline import DiffResult, Subtractor, subtract
from .solution import KernelSolution

__all__ = [
    "DiffResult",
    "KernelSolution",
    "Subtractor",
    "__version__",
    "as_layers",
    "basis_convolve",
    "decorrelate",
    "decorrelate_block",
    "decorrelation_kernel",
    "detect_stamps",
    "estimate_background",
    "fit_kernel",
    "gauss_hermite_basis1d",
    "gauss_hermite_kernels",
    "grid_knots",
    "matched_filter",
    "photometric_scale",
    "photometric_scale_at",
    "read_fits",
    "select_stamps",
    "solve_gls",
    "solve_gls_cv",
    "solve_gls_gcv",
    "subtract",
    "subtract_model",
    "synth_variance",
    "tps_design",
    "tps_evaluate",
    "tps_fit",
    "tps_penalty",
    "validation",
    "weighted_mean",
    "write_fits",
]
