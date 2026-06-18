# delta

[![CI](https://github.com/tkillestein/delta/actions/workflows/ci.yml/badge.svg)](https://github.com/tkillestein/delta/actions/workflows/ci.yml)
[![docs](https://github.com/tkillestein/delta/actions/workflows/docs.yml/badge.svg)](https://tkilleste.in/delta)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow)](LICENSE)

A high-performance astronomical difference-imaging engine — a modern,
statistically-principled reformulation of the Alard & Lupton (1998) PSF-matching
algorithm, intended as a replacement for HOTPANTS.

A C++20 core does the numerics; a thin Python layer (zero-copy
[nanobind](https://nanobind.readthedocs.io) bindings) provides the user-facing API.

## Key ideas

- **Cartesian Gauss-Hermite (shapelet) kernel basis** ([Refregier 2003][refregier])
  — the 2-D Gauss-Hermite functions are orthogonal under the L² inner product, so
  the matching solve is far better conditioned than the classic Gaussian×polynomial
  basis, whose conditioning is a known pitfall ([Becker et al. 2012][becker2012]).
  It spans a function space comparable to a single Alard & Lupton Gaussian component
  and needs far less per-field tuning.
- **Continuous, adaptive spatial variation** via low-rank thin-plate regression
  splines ([Wood 2003][wood]; [Wahba 1990][wahba]), with the smoothing parameter
  chosen automatically by generalized cross-validation ([GCV][gcv]; Golub, Heath &
  Wahba 1979). One global field — no subregion seams or PSF discontinuities.
- **Masks and noise layers are first-class**: inverse-variance weighting in the fit,
  bad regions excluded, masks grown across the kernel footprint.
- **Valid detection statistics**: optional difference-image noise decorrelation
  (whitening, following [ZOGY][zogy]; Zackay, Ofek & Gal-Yam 2016) and a
  match-filtered S/N score image.
- **Modern engineering**: multicore + SIMD, NumPy/astropy interop.

Full documentation is published at **<https://tkilleste.in/delta>** (usage
guide, API reference, CLI, and the design spec). See [`docs/SPEC.md`](docs/SPEC.md) for
the design and [`docs/usage.md`](docs/usage.md) for the user guide.

> **Status:** stable (v1.0). The full pipeline — kernel solve → spatially-varying
> subtraction → variance/mask propagation → noise decorrelation → match-filtered
> score — works end to end, with astropy interop and a CLI. The public API follows
> [semantic versioning](https://semver.org).

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

Requires a C++20 compiler, CMake ≥ 3.18, and CFITSIO and Eigen (≥ 3.4, via
`pkg-config`). The FFT is vendored (header-only PocketFFT), so no system FFT
library is needed. The Python dev stack uses [uv](https://docs.astral.sh/uv/),
[ruff](https://docs.astral.sh/ruff/), and [ty](https://github.com/astral-sh/ty).

```sh
uv sync           # create the venv, build the C++ core, install dev tools
uv run pytest     # run the test suite
uv run ruff check # lint
uv run ty check   # type-check
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

## Related work

`delta` sits in a long line of difference-imaging methods. The optimal PSF-matching
kernel idea is due to [Alard & Lupton (1998)][al1998], extended to a spatially
varying kernel by [Alard (2000)][alard2000]; **HOTPANTS** ([Becker 2015][hotpants])
is the widely used C implementation `delta` is benchmarked against. An alternative
to basis decomposition is the discrete/delta-function "numerical kernel" of
[Bramich (2008)][bramich2008] and [Miller, Pennypacker & White (2008)][miller2008],
with automatic kernel design via information criteria in
[Bramich et al. (2016)][bramich2016]. [Becker et al. (2012)][becker2012] study how
the choice of kernel basis and regularization affects conditioning — directly
relevant to `delta`'s near-orthogonal basis. For optimal *statistics*,
[Zackay, Ofek & Gal-Yam (2016, ZOGY)][zogy] derive the proper subtraction and the
noise-whitening filter `delta` uses; [Hu et al. (2022, SFFT)][sfft] perform the
subtraction in Fourier space with a δ-function basis for an order-of-magnitude
speed-up. `delta`'s contribution is to recast the Alard & Lupton solve as a single
**global, penalized GLS spline fit** in a near-orthogonal shapelet basis, with
automatic smoothing and first-class noise/mask propagation.

## References

- <a id="al1998"></a>Alard, C. & Lupton, R. H. (1998), *A Method for Optimal Image Subtraction*, ApJ **503**, 325. [doi:10.1086/305984](https://doi.org/10.1086/305984)
- <a id="alard2000"></a>Alard, C. (2000), *Image subtraction using a space-varying kernel*, A&AS **144**, 363. [doi:10.1051/aas:2000214](https://doi.org/10.1051/aas:2000214)
- <a id="becker2012"></a>Becker, A. C. et al. (2012), *Regularization Techniques for PSF-Matching Kernels — I. Choice of Kernel Basis*, MNRAS **425**, 1341. [doi:10.1111/j.1365-2966.2012.21542.x](https://doi.org/10.1111/j.1365-2966.2012.21542.x)
- <a id="hotpants"></a>Becker, A. C. (2015), *HOTPANTS: High Order Transform of PSF ANd Template Subtraction*, Astrophysics Source Code Library, [ascl:1504.004](https://ascl.net/1504.004).
- <a id="bramich2008"></a>Bramich, D. M. (2008), *A new algorithm for difference image analysis*, MNRAS **386**, L77. [doi:10.1111/j.1745-3933.2008.00464.x](https://doi.org/10.1111/j.1745-3933.2008.00464.x)
- <a id="bramich2016"></a>Bramich, D. M. et al. (2016), *Difference image analysis: automatic kernel design using information criteria*, MNRAS **457**, 542. [doi:10.1093/mnras/stv2910](https://doi.org/10.1093/mnras/stv2910)
- <a id="gcv"></a>Golub, G. H., Heath, M. & Wahba, G. (1979), *Generalized Cross-Validation as a Method for Choosing a Good Ridge Parameter*, Technometrics **21**, 215. [doi:10.1080/00401706.1979.10489751](https://doi.org/10.1080/00401706.1979.10489751)
- <a id="sfft"></a>Hu, L. et al. (2022), *Image Subtraction in Fourier Space (SFFT)*, ApJ **936**, 157. [doi:10.3847/1538-4357/ac7394](https://doi.org/10.3847/1538-4357/ac7394)
- <a id="miller2008"></a>Miller, J. P., Pennypacker, C. R. & White, G. L. (2008), *Optimal Image Subtraction Method: Summary Derivations, Applications, and Publicly Shared Application Using IDL*, PASP **120**, 449. [doi:10.1086/588258](https://doi.org/10.1086/588258)
- <a id="refregier"></a>Refregier, A. (2003), *Shapelets — I. A method for image analysis*, MNRAS **338**, 35. [doi:10.1046/j.1365-8711.2003.05901.x](https://doi.org/10.1046/j.1365-8711.2003.05901.x)
- <a id="wahba"></a>Wahba, G. (1990), *Spline Models for Observational Data*, SIAM. [doi:10.1137/1.9781611970128](https://doi.org/10.1137/1.9781611970128)
- <a id="wood"></a>Wood, S. N. (2003), *Thin plate regression splines*, J. R. Stat. Soc. B **65**, 95. [doi:10.1111/1467-9868.00374](https://doi.org/10.1111/1467-9868.00374)
- <a id="zogy"></a>Zackay, B., Ofek, E. O. & Gal-Yam, A. (2016), *Proper Image Subtraction — Optimal Transient Detection, Photometry, and Hypothesis Testing*, ApJ **830**, 27. [doi:10.3847/0004-637X/830/1/27](https://doi.org/10.3847/0004-637X/830/1/27)

[al1998]: #al1998
[alard2000]: #alard2000
[becker2012]: #becker2012
[hotpants]: #hotpants
[bramich2008]: #bramich2008
[bramich2016]: #bramich2016
[gcv]: #gcv
[sfft]: #sfft
[miller2008]: #miller2008
[refregier]: #refregier
[wahba]: #wahba
[wood]: #wood
[zogy]: #zogy

## License

MIT — see [`LICENSE`](LICENSE). Third-party components and their licenses (all
permissive; no GPL dependency) are listed in [`THIRD_PARTY.md`](THIRD_PARTY.md).
