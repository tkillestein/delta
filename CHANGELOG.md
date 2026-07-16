# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.1]

### Added
- `apply` API + CLI command: apply a saved `KernelSolution` to a new image pair
  without re-fitting.
- Connected-component source catalog (`build_catalog`) from the match-filtered
  score image, with FWHM-consistency filtering, dipole flagging, and mask-flag
  aggregation (SPEC §3.7).
- Full run provenance (`DLT*` FITS header cards): fit, config, environment,
  and runtime layers, merged by `DiffResult.write()`.
- A scalar variance rescale so the difference-image reduced chi2 is forced to 1.

### Changed
- The kernel fit is now weighted by `K²⊗Var` instead of `Q·Var_c`.
- The decorrelated difference and its variance are now statistically consistent.

## [1.0.0] - 2026-06-18

First public release. The API is now considered stable under semantic versioning.

### Changed
- The FFT backend for noise decorrelation is now the vendored header-only
  **PocketFFT** (BSD-3) instead of system **FFTW**. This removes the GPL build
  dependency and a system dependency; the decorrelation stage is no slower.
- `DELTA_NATIVE` (`-march=native`) now defaults **OFF** so a from-source build is
  portable by default. Set `-DDELTA_NATIVE=ON` for a host-tuned local build.
- Eigen is now required at version ≥ 3.4.

### Fixed
- The FWHM median used for convolution-direction selection no longer poisons on NaN
  inputs from degenerate stamps (non-finite samples are dropped before the median).

### Added
- The low-level `write_fits` binding now takes an `overwrite` argument (default
  `False`) instead of always clobbering an existing file.
- `CITATION.cff`, a References/Related-work section, third-party attribution
  (`THIRD_PARTY.md`), `CONTRIBUTING.md`, and this changelog.
- Direct tests for the convolution engine and the FITS `DIFFERENCE`-extension fallback.

## [0.0.1]

- Initial pre-release: end-to-end difference-imaging pipeline (kernel solve →
  spatially-varying subtraction → variance/mask propagation → noise decorrelation →
  match-filtered score), nanobind Python API, astropy interop, and CLI.
