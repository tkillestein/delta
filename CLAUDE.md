# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`delta` is an astronomical difference-imaging engine: a modern reformulation of the
Alard & Lupton (1998) PSF-matching algorithm, intended as a HOTPANTS replacement. A
C++20 core (`delta::` namespace) does the numerics; a thin Python layer (zero-copy
nanobind bindings) provides the user-facing API. `docs/SPEC.md` is the authoritative
design document — section numbers (e.g. "SPEC §3.6") are referenced throughout the
code and are the best place to understand intent.

## Build & dev commands

The toolchain is `uv` + `ruff` + `ty`; the C++ extension is built by scikit-build-core
via `uv sync`.

```sh
uv sync                       # create venv, build the C++ core, install dev deps
uv run pytest                 # run the full test suite
uv run pytest tests/test_fit.py::test_name   # run a single test
uv run ruff check             # lint
uv run ruff format            # format
uv run ty check               # type-check
```

After editing any C++ source/header, re-run `uv sync` (or `uv run pytest`, which
rebuilds) to recompile the `_core` extension — Python-only changes do not need a rebuild.

System deps (via `pkg-config`): C++20 compiler, CMake ≥ 3.18, Eigen, CFITSIO, FFTW
(`fftw3f`). OpenMP is used if found.

Enable the pre-commit hook once per clone (`git config core.hooksPath .githooks`); it
runs ruff fix/format + `ty check` on staged Python and gates the commit.

## Architecture

The work is split into C++ modules (header in `include/delta/<name>.hpp`, impl in
`src/<name>.cpp`), exposed through `src/bindings.cpp` as the `_core` extension, and
orchestrated by the Python package in `python/delta/`.

**C++ core modules** (each maps to a SPEC §4 module):
- `io` — CFITSIO read/write of data + variance + mask FITS layers.
- `basis` — separable Cartesian Gauss-Hermite kernel basis (scale `beta`, order `n_max`).
- `convolve` — separable/SIMD convolution; precomputes `B_n = φ_n ⊗ R`.
- `spatial` — low-rank thin-plate regression spline: knots, design, bending-energy penalty.
- `solve` — penalized GLS normal equations (Eigen) + GCV λ selection.
- `detect` — stamp detection/selection, FWHM and convolution-direction estimation.
- `subtract` — full-frame model evaluation, difference, variance/mask propagation.
- `noise` — ZOGY-style decorrelation (apodized FFT blocks, FFTW) + match-filter score.
- `fit` — ties stamps + basis + spatial together into the kernel solve.

**Python layer** (`python/delta/`):
- `pipeline.py` — `Subtractor` class and `subtract()` convenience wrapper. This is the
  orchestration brain: coerces inputs, selects stamps, picks convolution direction and
  `beta`, calls `_core.fit_kernel` / `_core.subtract`, optional decorrelate + score.
- `solution.py` — `KernelSolution` dataclass: the serializable (`.npz`) fit result
  (basis params, knots, `theta`, GCV diagnostics) + FITS provenance header cards.
- `_inputs.py` — input coercion (`as_layers`, `synth_variance`); numpy/astropy interop.
- `validation.py` — QA/validation helpers.
- `_core.pyi` — type stub for the compiled extension.

### The central algorithmic idea (SPEC §3.2)

Spatial kernel variation **factors out of the convolution**:
`[K ⊗ R](x,y) = Σ_n a_n(x,y) · B_n(x,y)`, where `B_n = φ_n ⊗ R` is precomputed once per
basis component and `a_n(x,y)` is a thin-plate spline field. This gives one *global*,
exactly-continuous spatially-varying fit (no tiling, no subregion seams) and makes the
solve linear in `θ = {c_nm, b_m}`. `theta` is ordered `[c_0 | … | c_{nc-1} | b]`.

### Conventions that span the codebase

- **Image storage** (`include/delta/image.hpp`): contiguous row-major, `index = y*width + x`.
  `Image<T>` carries optional variance and mask vectors; `ImageF` (float) is the working type.
- **Masks** are first-class `uint8` bitmasks (`MaskFlag` enum; `0` = good). Bad pixels are
  excluded from the fit; the reference mask is dilated by the kernel half-width on subtraction.
- **Convolution direction is auto-selected**: the sharper (narrower-FWHM) image is convolved
  to match the broader. `direction == "science"` flips the difference sign so transients stay
  positive — see `pipeline.py` `_fit`.
- **Variance/masks flow through the whole pipeline** (SPEC §3.6), not bolted on; if no variance
  is supplied, it is synthesized from `gain` + `read_noise`.
- **Bindings**: results cross to Python as zero-copy NumPy arrays (`to_numpy` moves a
  `std::vector` into a capsule-owned ndarray). C++ functions return `nb::dict`s that the
  Python layer unpacks.

## Tests & benchmarks

- `tests/` mirrors the modules (`test_basis.py`, `test_solve.py`, `test_fit.py`, …);
  `test_pipeline.py` and `test_injection_recovery.py` are end-to-end. `test_hotpants.py`
  validates against HOTPANTS when available.
- The `preview` fixture (`tests/conftest.py`) writes side-by-side JPEGs to
  `tests/artifacts/` for eyeballing results; it is a no-op without Pillow.
- `benchmarks/` holds HOTPANTS head-to-head comparison and timing scripts (needs the
  `benchmarks` extra: `astropy`, `matplotlib`).
