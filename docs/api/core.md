# Core (low-level)

The C++ core, exposed as `delta._core` and re-exported at the top level. These are the
building blocks the [high-level pipeline](pipeline.md) orchestrates — basis evaluation,
convolution, the penalized GLS solver, thin-plate splines, stamp detection, noise
decorrelation, and FITS I/O.

!!! warning "Advanced"
    Most users should use the [high-level pipeline](pipeline.md) instead. These
    functions are documented for advanced use and for understanding the internals.
    Signatures and docstrings are sourced from the `_core.pyi` type stub. The compiled
    `_core.subtract` is re-exported as `delta.subtract_model` to avoid colliding with
    the high-level [`delta.subtract`][delta.subtract].

## Basis & convolution

::: delta._core.gauss_hermite_basis1d

::: delta._core.gauss_hermite_kernels

::: delta._core.basis_convolve

## Spatial model (thin-plate splines)

::: delta._core.grid_knots

::: delta._core.tps_design

::: delta._core.tps_penalty

::: delta._core.tps_fit

::: delta._core.tps_evaluate

## Solver

::: delta._core.solve_gls

::: delta._core.solve_gls_gcv

::: delta._core.solve_gls_cv

::: delta._core.weighted_mean

## Detection & stamps

::: delta._core.detect_stamps

::: delta._core.select_stamps

## Fit & subtract

::: delta._core.fit_kernel

::: delta._core.subtract

::: delta._core.photometric_scale

::: delta._core.photometric_scale_at

::: delta._core.estimate_background

## Noise: decorrelation & scoring

::: delta._core.decorrelation_kernel

::: delta._core.decorrelate_block

::: delta._core.decorrelate

::: delta._core.matched_filter

## FITS I/O

::: delta._core.read_fits

::: delta._core.write_fits
