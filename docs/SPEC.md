# delta — Specification & Implementation Plan

A high-performance astronomical difference-imaging engine: a modern,
statistically-principled reformulation of the Alard & Lupton (1998) PSF-matching
algorithm, intended as a replacement for HOTPANTS. High-performance C++ core with
nanobind Python bindings.

---

## 1. Context & Motivation

Difference imaging — subtracting a reference/template image from a science image to
reveal transient and variable sources — is foundational to time-domain astronomy.
The dominant open tool, HOTPANTS (an implementation of Alard & Lupton 1998; Becker
2015), is effective but dated: a non-orthogonal sum-of-Gaussians × polynomial kernel
basis whose conditioning is a known pitfall (Becker et al. 2012) and which requires
hand-tuning of σ's and polynomial degrees; global
polynomial spatial variation that oscillates and can't adapt to localized PSF
structure; correlated noise in the output difference that breaks naive detection
statistics; and a single-threaded C codebase that does not exploit modern hardware.

`delta` keeps the proven A&L idea — solve for a convolution kernel that matches one
image's PSF to the other, then subtract — but recasts it as a **penalized
generalized least-squares (spline) fit**:

- **Near-orthogonal kernel basis.** Cartesian Gauss-Hermite (shapelet) functions
  (Refregier 2003) replace the ad-hoc Gaussian×polynomial basis. They are orthogonal
  under the L² inner product (they are eigenstates of the 2-D quantum harmonic
  oscillator), spanning a function space comparable to a single A&L Gaussian
  component but far better conditioned (Becker et al. 2012), with one scale parameter
  β + max order `n_max` instead of a basket of knobs.
- **Adaptive, continuous spatial variation.** Each kernel coefficient is a 2-D field
  modeled by a **low-rank thin-plate regression spline** (Wahba 1990; Wood 2003),
  with the smoothing parameter λ chosen automatically by **generalized
  cross-validation (GCV)** (Golub, Heath & Wahba 1979) — no manual polynomial-degree
  tuning, no subregion seams.
- **Valid detection statistics.** Optional difference-image **noise decorrelation**
  (whitening, following Zackay, Ofek & Gal-Yam 2016, "ZOGY") and a **match-filtered
  S/N score image** for direct transient detection — features HOTPANTS lacks.
- **Modern engineering.** Multicore + SIMD C++ core, zero-copy NumPy/astropy Python
  API.

**Intended outcome:** a faster, better-conditioned, lower-tuning difference-imaging
engine that produces statistically valid difference and detection products on
large monolithic frames (primary target ~8000×6000 px) at survey cadence.

---

## 2. Scope

**In scope (decided with user):**
- Core engine: kernel solve → spatially-varying convolution → subtraction → noise
  propagation. Assumes inputs are already astrometrically registered onto a common
  pixel grid.
- Jointly-fit differential **background/photometric alignment** (flux scaling comes
  from the kernel sum; background is a fitted spatial field in the same GLS).
- Noise products: propagated variance map, decorrelated difference, match-filtered
  score image.
- Stamp selection: internal source detection **and** caller-supplied external catalog.
- Auto-selected convolution direction (convolve the narrower-PSF image to match the
  broader one) from measured stamp FWHM.

**Out of scope (for now; see §16 Future):**
- WCS-based registration/resampling (inputs assumed pre-registered; WCSLIB not a
  dependency).
- Source detection/measurement *on the difference* beyond producing the score image.
- GPU backends, ZOGY engine, common-PSF pre-matching, multi-chip global fields.

---

## 3. Algorithm

### 3.1 Model

With science `S`, reference `R` (the image chosen to be convolved), spatially-varying
matching kernel `K`, and differential background `B`, the difference image is:

```
D(x,y) = S(x,y) − [K ⊗ R](x,y) − B(x,y)
```

The kernel is expanded in a **Cartesian Gauss-Hermite** basis of scale β, order ≤ n_max,
over a small footprint (u,v); each coefficient is a smooth field over the frame:

```
K(u,v; x,y) = Σ_n  a_n(x,y) · φ_n(u,v)          φ_n = Gauss-Hermite basis
a_n(x,y)    = Σ_m  c_{nm} · ψ_m(x,y)            ψ_m = low-rank TPS basis (knots + affine null space)
B(x,y)      = Σ_m  b_m   · ψ_m(x,y)             (its own TPS field)
```

### 3.2 The factorization trick (continuous, seam-free, no FFT)

Convolution is linear and `a_n(x,y)` multiplies pixelwise, so spatial variation
**factors out of the convolution**:

```
[K ⊗ R](x,y) = Σ_n a_n(x,y) · (φ_n ⊗ R)(x,y) = Σ_n a_n(x,y) · B_n(x,y)
```

- Precompute `B_n = φ_n ⊗ R` once per basis component — **separable** (Hermite⊗Hermite),
  SIMD-vectorized, threaded over rows/tiles purely for parallelism (not for kernel
  variation). Cost = n_basis separable passes over the frame.
- The full-frame model is then `Σ_n a_n(x,y)·B_n(x,y) + B(x,y)`, evaluated per pixel
  with `a_n` from the low-rank fields (k terms). **Exactly continuous across the whole
  8000×6000 frame — no tiling, no join lines, no PSF discontinuities.**

### 3.3 The solve

Linear in the unknowns `{c_{nm}, b_m}`: each design column for `(n,m)` is
`ψ_m(x,y)·B_n(x,y)`, sampled over the pixels of selected PSF stamps. Solve a single
**penalized GLS** over all stamps:

```
minimize  ‖W^½ (d − X θ)‖²  +  λ θᵀ P θ
  X  = design matrix  (GaussHermite ⊗ TPS columns, + background columns)
  W  = inverse-variance weights (proper noise model)
  P  = thin-plate bending-energy penalty (per a_n field)
  θ  = {c_{nm}, b_m}
```

- **Low-rank** TPS knots (grid/clustered over the frame) keep one *global* continuous
  fit tractable: O(N_stamps·k²) instead of dense O(N_stamps³).
- **λ via GCV:** minimize `GCV(λ) = n·RSS(λ) / (n − tr S(λ))²`; effective dof
  `tr S(λ)` is cheap for the low-rank smoother. (REML selector and per-coefficient λ
  are future options — §16.)
- Solved with **Eigen** (Cholesky/LDLT on the penalized normal equations, QR fallback
  for conditioning).
- **Photometric scale** = Σ K(x,y) over the footprint, reported per location; flux
  matching is intrinsic to the kernel, background absorbed by `B(x,y)`.

### 3.4 Output difference & noise

- **Difference:** `D = S − Σ_n a_n B_n − B`, full frame, continuous.
- **Variance propagation:** from input variance maps (or synthesized from gain +
  read-noise), `Var(D) = Var(S) + Σ over kernel of K²·Var(R)` accounting for the
  convolution; produces a per-pixel variance/weight map.
- **Decorrelation (optional):** the convolution correlates difference-image noise.
  Apply a whitening (decorrelation) kernel à la ZOGY so noise is ~uncorrelated —
  required for valid thresholding. Spatially-varying → **apodized overlapping FFT
  blocks** (vendored PocketFFT); the noise field varies on the knot length-scale,
  which is much coarser than the block stride, so the Hann overlap-add blending
  between adjacent blocks introduces no visible seam.
- **Score image (optional):** PSF-matched filter of the decorrelated difference →
  per-pixel S/N map for direct detection/thresholding.

### 3.5 Stamps & convolution direction

- Stamps from **internal detection** (bright, isolated, unsaturated sources) or a
  **caller-supplied catalog**; saturated/masked pixels excluded.
- **Auto-pick direction:** measure stamp FWHM in both images; convolve the
  narrower-PSF image to match the broader (broadening kernels are well-behaved).
  Single global direction (common-PSF pre-match deferred — §16); document the
  limitation when seeing crosses within a frame.

### 3.6 Masks & noise layers (first-class)

Every image carries an optional **variance/noise layer** and a **bad-pixel mask**;
both flow through the whole pipeline rather than being bolted on at the end.

- **Noise-weighted fitting.** Per-pixel inverse variance from the input noise layers
  (or synthesized from gain + read-noise) forms the GLS weight matrix
  `W = diag(1/Var)`. Stamp pixels are weighted by their actual noise, so low-S/N and
  high-S/N stamps contribute correctly — no flat weighting.
- **Exclude bad areas from the fit.** Masked pixels (saturation, cosmic rays, bad
  columns, NaN/inf, user flags) are dropped from stamp selection and removed as rows
  from the GLS design matrix `X`. Stamps with too large a masked fraction are rejected
  entirely. Nothing bad enters the kernel/background solve.
- **Propagate masks across the kernel.** A bad pixel in the convolved image `R`
  contaminates every output pixel within the kernel footprint, so the reference mask
  is **dilated by the kernel half-width** before subtraction; the science mask
  propagates one-to-one. The output difference mask is the grown union, plus a frame
  border of one kernel half-width.
- **Variance through convolution.** `Var(D) = Var(S) + (K² ⊗ Var(R))` — the kernel is
  squared and convolved against the reference variance (assuming independent input
  pixels), giving a per-pixel output variance/weight map consistent with the
  decorrelation and score stages.

---

## 4. Architecture (C++ core)

Namespace `delta::`. Modules:

- **`io`** — CFITSIO wrappers: read/write FITS images, headers, variance & mask
  extensions. Lightweight `Image<T>` (contiguous, row-major, with variance + mask).
- **`basis`** — Gauss-Hermite footprint generation (separable kernels), scale β,
  order n_max.
- **`convolve`** — separable + SIMD convolution; `B_n = φ_n ⊗ R` precompute; threaded
  tiling.
- **`spatial`** — low-rank thin-plate regression spline: knot placement, ψ basis,
  bending-energy penalty P, field evaluation.
- **`solve`** — assemble penalized GLS design/normal equations, Eigen solve, GCV λ
  search.
- **`subtract`** — full-frame model evaluation, difference, variance propagation.
- **`detect`** — stamp source detection & selection, FWHM/direction estimation.
- **`noise`** — decorrelation kernel + apodized block FFT (PocketFFT), match-filter score.
- **`pipeline`** — orchestration: end-to-end `subtract(science, reference, options)`.

Key data structures: `Image<T>`, `Stamp`, `KernelBasis`, `SpatialModel`,
`KernelSolution` (serializable: basis params, knots, θ, λ, photometric scale),
`DiffProducts` (difference, variance, mask, decorrelated, score).

---

## 5. Parallelism & Performance

- **Threading:** OpenMP (or TBB) — over rows/tiles in convolution and score; over
  stamps when assembling normal-equation contributions (reduce into the system).
- **SIMD:** vectorized separable convolution inner loops; rely on Eigen's vectorized
  kernels for the linear algebra.
- **Memory:** stream large frames; pre-allocate the n_basis `B_n` buffers once; avoid
  per-stamp allocation churn (arena/pool). Target single-pass over the frame for the
  subtraction stage.
- **FFT** (PocketFFT): one workspace per thread, reused across blocks; real-to-complex transforms.

---

## 6. Dependencies & Build

- **CFITSIO** — FITS I/O.
- **Eigen** (≥ 3.4) — dense/sparse linear algebra for the penalized GLS (header-only).
- **PocketFFT** (vendored, header-only, BSD-3) — used by the decorrelation + score
  post-filter stage. Matching convolution itself needs no FFT, and there is no
  external/system FFT (or GPL) dependency.
- **OpenMP/TBB** — threading.
- **nanobind** — Python bindings.
- **Build:** CMake; **scikit-build-core** for the Python wheel. CI builds + tests on
  Linux (primary), macOS.

---

## 7. Python API (nanobind + NumPy/astropy)

Zero-copy NumPy arrays in/out; accept astropy `CCDData`/FITS/HDU conveniences.

```python
import delta

result = delta.subtract(
    science, reference,           # numpy arrays or astropy CCDData
    science_var=..., reference_var=...,   # or gain/read-noise to synthesize
    mask=...,
    catalog=None,                 # None → internal detection
    n_max=..., beta=...,          # basis; sensible auto-defaults
    n_knots=...,                  # low-rank TPS knots
    decorrelate=True, score=True,
)
result.difference   # numpy
result.variance
result.score        # match-filtered S/N map
result.kernel_solution  # serializable (save/load, QA)
```

Object-oriented path (`delta.Subtractor`) for reusing a solution / batch runs.

---

## 8. Output Products

- **Difference image** + **propagated variance map** (always).
- **Decorrelated difference** (whitened noise).
- **Match-filtered score image** (S/N map for detection).
- Output **mask** (grown bad/saturated/edge pixels) and serializable
  **kernel/background solution** available for QA/reuse (not headline products but
  produced internally).
- FITS output with provenance in headers: fit params (basis β/n_max, knots, λ, GCV
  score, convolution direction), `Subtractor` configuration (stamp/clipping/CV
  knobs), software/host environment (version, git commit, hostname, user, platform,
  UTC timestamp), and run timing — enough to reproduce a run without external notes.
  See `docs/usage.md` "Provenance headers" for the full card reference.

---

## 9. Project Layout

```
delta/
  CMakeLists.txt
  pyproject.toml            # scikit-build-core
  include/delta/            # public C++ headers
  src/                      # io, basis, convolve, spatial, solve, subtract, detect, noise, pipeline
  python/delta/             # Python package (nanobind ext + .py wrappers)
  tests/                    # C++ (ctest) + Python (pytest)
  benchmarks/
  docs/
```

---

## 10. Testing & Validation

- **Unit:** Gauss-Hermite basis correctness/near-orthogonality; separable convolution
  vs reference NumPy convolution; TPS field reproduces known smooth functions; GCV
  recovers the right λ on synthetic data; variance propagation vs Monte Carlo.
- **Injection–recovery:** inject synthetic sources (range of flux, position,
  varying PSF), measure recovery completeness, astrometric/photometric accuracy, and
  residual statistics around bright stars (dipole suppression).
- **Noise whitening check:** autocorrelation of the difference is flat after
  decorrelation; score-image pixel distribution is ~unit-Gaussian in source-free
  regions.
- **HOTPANTS head-to-head harness:** run both on identical real data; compare
  residual RMS, subtraction artifacts, photometric scatter, and runtime. (This is the
  "modern replacement" evidence; does not require carrying the A&L basis in the
  production solver.)
- **Performance benchmarks:** end-to-end wall-clock and scaling on 8000×6000 frames,
  thread-count scaling.

---

## 11. Milestones

- **M0** Scaffolding: CMake + scikit-build-core + nanobind hello-world, CI, layout.
- **M1** `io` + `Image<T>` (FITS images, variance, mask).
- **M2** `basis` + `convolve`: Gauss-Hermite footprint, separable SIMD convolution,
  `B_n` precompute (validated vs NumPy).
- **M3** `detect`: stamp detection/selection, FWHM, auto convolution direction;
  external-catalog ingestion.
- **M4** `spatial` + `solve`: low-rank TPS, penalized GLS via Eigen, GCV λ search.
- **M5** `subtract`: full-frame continuous spatially-varying subtraction + variance
  propagation.
- **M6** Jointly-fit background/photometric term; kernel-sum flux scale reporting.
- **M7** `noise`: decorrelation (apodized block FFT, PocketFFT) + match-filtered score image.
- **M8** nanobind Python API + astropy interop + packaging.
- **M9** HOTPANTS validation harness, injection-recovery suite, benchmarks, docs.

---

## 12. Future / Out-of-Scope Extensions

GPU backend (CUDA/SYCL); ZOGY Fourier engine as an alternative method; common-PSF
pre-matching for seeing that crosses within a frame; REML and per-coefficient λ
selectors; true multi-chip mosaic handling (per-detector fields with WCS); WCS-based
registration/resampling; difference-image source measurement.

---

## 13. Verification (how we'll know it works)

1. **Build & import:** `pip install -e .` builds the C++ core and imports `delta`;
   CI green on Linux/macOS.
2. **Unit + injection-recovery** pass (`ctest`, `pytest`), including the noise-whitening
   and GCV tests.
3. **End-to-end on real 8000×6000 data:** `delta.subtract(...)` produces a clean
   difference with suppressed stellar residuals, a flat decorrelated-noise
   autocorrelation, and a unit-Gaussian score image in source-free regions.
4. **Head-to-head vs HOTPANTS** on identical data: comparable-or-better residual RMS
   and photometric scatter, at lower runtime, with no hand-tuning of basis knobs.
5. **Benchmark** the full pipeline meets the survey-cadence latency target on the
   reference frame size and scales with thread count.

---

## 14. References

- Alard, C. & Lupton, R. H. (1998), *A Method for Optimal Image Subtraction*, ApJ **503**, 325.
- Alard, C. (2000), *Image subtraction using a space-varying kernel*, A&AS **144**, 363.
- Becker, A. C. et al. (2012), *Regularization Techniques for PSF-Matching Kernels — I. Choice of Kernel Basis*, MNRAS **425**, 1341.
- Becker, A. C. (2015), *HOTPANTS: High Order Transform of PSF ANd Template Subtraction*, ascl:1504.004.
- Bramich, D. M. (2008), *A new algorithm for difference image analysis*, MNRAS **386**, L77.
- Bramich, D. M. et al. (2016), *Difference image analysis: automatic kernel design using information criteria*, MNRAS **457**, 542.
- Golub, G. H., Heath, M. & Wahba, G. (1979), *Generalized Cross-Validation as a Method for Choosing a Good Ridge Parameter*, Technometrics **21**, 215.
- Hu, L. et al. (2022), *Image Subtraction in Fourier Space (SFFT)*, ApJ **936**, 157.
- Miller, J. P., Pennypacker, C. R. & White, G. L. (2008), *Optimal Image Subtraction Method*, PASP **120**, 449.
- Refregier, A. (2003), *Shapelets — I. A method for image analysis*, MNRAS **338**, 35.
- Wahba, G. (1990), *Spline Models for Observational Data*, SIAM.
- Wood, S. N. (2003), *Thin plate regression splines*, J. R. Stat. Soc. B **65**, 95.
- Zackay, B., Ofek, E. O. & Gal-Yam, A. (2016), *Proper Image Subtraction (ZOGY)*, ApJ **830**, 27.
