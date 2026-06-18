# delta

A high-performance astronomical difference-imaging engine — a modern,
statistically-principled reformulation of the Alard & Lupton (1998) PSF-matching
algorithm, intended as a replacement for HOTPANTS.

A C++20 core does the numerics; a thin Python layer (zero-copy
[nanobind](https://nanobind.readthedocs.io) bindings) provides the user-facing API.

!!! note "Status"
    Stable (v1.0). The end-to-end pipeline works (kernel solve → spatially-varying
    subtraction → variance/mask propagation → noise decorrelation → match-filtered
    score), with astropy interop and a CLI. The public API follows semantic
    versioning.

## Key ideas

- **Cartesian Gauss-Hermite (shapelet) kernel basis** (Refregier 2003) — the 2-D
  Gauss-Hermite functions are orthogonal under the L² inner product, so the matching
  solve is far better conditioned than the classic Gaussian×polynomial basis (a known
  pitfall; Becker et al. 2012) and needs far less tuning.
- **Continuous, adaptive spatial variation** via low-rank thin-plate regression
  splines, with the smoothing parameter chosen automatically by cross-validation.
  One global field — no subregion seams or PSF discontinuities.
- **Masks and noise layers are first-class**: inverse-variance weighting in the
  fit, bad regions excluded, masks grown across the kernel footprint.
- **Valid detection statistics**: optional difference-image noise decorrelation and
  a match-filtered S/N score image.
- **Modern engineering**: multicore + SIMD, NumPy/astropy interop.

## The central idea

Spatial kernel variation *factors out of the convolution*:

$$[K \otimes R](x,y) = \sum_n a_n(x,y)\, B_n(x,y), \qquad B_n = \varphi_n \otimes R$$

where `B_n` is precomputed once per basis component and `a_n(x, y)` is a thin-plate
spline field. This gives one *global*, exactly-continuous spatially-varying fit (no
tiling, no subregion seams) and makes the solve linear in the coefficients. See the
[design spec](design.md) (§3.2) for the full derivation.

## Quickstart

```python
import delta

result = delta.subtract(
    science, reference,          # numpy arrays or astropy CCDData / HDU
    gain=1.5, read_noise=4.0,    # or pass science_var / reference_var
    decorrelate=True, score=True,
)
result.difference   # PSF-matched difference (transients positive)
result.score        # match-filtered S/N detection map
result.write("diff.fits")
```

The convolution direction is auto-selected from the measured seeing and the
smoothing parameter is chosen automatically — no per-field tuning. Continue to the
[usage guide](usage.md) for inputs, noise models, and tuning, or the
[API reference](api/index.md).

## Installation

`delta` builds a C++ extension, so it needs a C++20 compiler, CMake ≥ 3.18, and
CFITSIO and Eigen (≥ 3.4) available via `pkg-config`. The FFT is vendored
(header-only PocketFFT), so no system FFT library is needed. The Python dev
stack uses [uv](https://docs.astral.sh/uv/).

```sh
uv sync                 # create the venv, build the C++ core, install dev tools
uv sync --extra cli     # also install the standalone `delta` CLI (Typer + Rich)
```

Or with pip from a checkout:

```sh
pip install .            # core library
pip install '.[cli]'     # with the command-line interface
```

## License

MIT — see the [`LICENSE`](https://github.com/tkillestein/delta/blob/main/LICENSE)
file.
