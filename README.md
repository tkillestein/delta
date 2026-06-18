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

Full documentation is published at **<https://tkillestein.github.io/delta/>** (usage
guide, API reference, CLI, and the design spec). See [`docs/SPEC.md`](docs/SPEC.md) for
the design and roadmap and [`docs/usage.md`](docs/usage.md) for the user guide.

> **Status:** end-to-end pipeline working (kernel solve → spatially-varying
> subtraction → variance/mask propagation → noise decorrelation → match-filtered
> score), with astropy interop. APIs may still change.

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
smoothing parameter is chosen automatically — no per-field tuning. See the
[usage guide](docs/usage.md).

## Command line

Install the `cli` extra (Typer + Rich) for a standalone `delta` command:

```sh
uv sync --extra cli            # or: pip install 'delta[cli]'

delta subtract science.fits reference.fits -o diff.fits \
    --gain 1.5 --read-noise 4 --decorrelate --score -v
delta info science.fits        # inspect data/variance/mask layers
delta subtract --help          # full option list
```

It reads FITS inputs, runs the pipeline, and writes a multi-extension FITS
(difference + variance/mask/score) with provenance header cards. Progress and
per-stage timings are logged to stderr ([loguru](https://github.com/Delgan/loguru));
control verbosity with `-v` (debug) / `-q` (warnings only). The library itself
stays silent unless a host enables the `delta` log namespace.

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

### Documentation

The docs site is built with [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/)
and [mkdocstrings](https://mkdocstrings.github.io/) (API reference generated from the
source docstrings). Build or preview it locally:

```sh
uv run --group docs mkdocs serve    # live preview at http://127.0.0.1:8000
uv run --group docs mkdocs build    # static build into site/
```

A GitHub Actions workflow (`.github/workflows/docs.yml`) builds and deploys to GitHub
Pages on push to `main`.

### Git hooks

A pre-commit hook (`.githooks/pre-commit`) auto-runs `ruff check --fix`,
`ruff format`, and `ty check --fix` on staged Python, and blocks the commit on
any remaining lint/type errors. Enable it once per clone:

```sh
git config core.hooksPath .githooks
```

## License

MIT — see [`LICENSE`](LICENSE).
