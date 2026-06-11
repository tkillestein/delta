# delta

A high-performance astronomical difference-imaging engine — a modern,
statistically-principled reformulation of the Alard & Lupton (1998) PSF-matching
algorithm, intended as a replacement for HOTPANTS.

High-performance C++ core with zero-copy [nanobind](https://nanobind.readthedocs.io)
Python bindings.

## Key ideas

- **Cartesian Gauss-Hermite kernel basis** — same function space as classic A&L but
  near-orthogonal, so the matching solve is far better conditioned and needs far less
  tuning.
- **Continuous, adaptive spatial variation** via low-rank thin-plate regression
  splines, with the smoothing parameter chosen automatically by generalized
  cross-validation (GCV). One global field — no subregion seams or PSF
  discontinuities.
- **Masks and noise layers are first-class**: inverse-variance weighting in the fit,
  bad regions excluded, masks grown across the kernel footprint.
- **Valid detection statistics**: optional difference-image noise decorrelation and a
  match-filtered S/N score image.
- **Modern engineering**: multicore + SIMD, NumPy/astropy interop.

See [`docs/SPEC.md`](docs/SPEC.md) for the full design and roadmap.

> **Status:** early development (M0 scaffold). APIs are unstable.

## Building

Requires a C++20 compiler, CMake ≥ 3.18, and CFITSIO, Eigen, and FFTW
(via `pkg-config`). The Python dev stack uses [uv](https://docs.astral.sh/uv/),
[ruff](https://docs.astral.sh/ruff/), and [ty](https://github.com/astral-sh/ty).

```sh
uv sync          # create the venv, build the C++ core, install dev tools
uv run pytest    # run the smoke tests
uv run ruff check # lint
uv run ty check  # type-check
```

## License

MIT — see [`LICENSE`](LICENSE).
