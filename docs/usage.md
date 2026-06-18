# Usage

`delta` performs PSF-matched difference imaging: it solves for a spatially-varying
convolution kernel that matches one image's PSF to the other, subtracts, and
(optionally) decorrelates the noise and produces a match-filtered detection score.
Inputs are assumed to be astrometrically registered onto a common pixel grid.

## Quickstart

```python
import delta

result = delta.subtract(
    science, reference,          # numpy arrays or astropy CCDData / HDU
    gain=1.5, read_noise=4.0,    # synthesize variance, or pass *_var explicitly
    decorrelate=True, score=True,
)

result.difference   # PSF-matched difference (transients positive)
result.variance     # propagated per-pixel variance
result.mask         # grown bad/edge mask
result.score        # match-filtered S/N map (when score=True)
result.solution     # serializable KernelSolution (QA / reuse)

result.write("diff.fits")        # multi-extension FITS with provenance (needs astropy)
```

`delta.subtract` auto-detects PSF-matching stamps, measures the seeing in both
frames, and convolves the sharper image to match the broader one — the convolution
direction is recorded in `result.solution.direction` (`"reference"` or `"science"`)
and the difference sign is set so transients are always positive.

See [`delta.subtract`][delta.subtract] and [`DiffResult`][delta.DiffResult] in the
API reference.

## Inputs and noise

- **Arrays or astropy.** `science` / `reference` may be NumPy arrays, astropy
  `CCDData`, or FITS HDUs; variance and mask carried by a `CCDData` are picked up
  automatically. Explicit `science_var`, `reference_var`, `science_mask`,
  `reference_mask` override them.
- **Variance.** Provide variance maps directly, or pass `gain` (and optional
  `read_noise`) to synthesize `Var = max(data, 0)/gain + read_noise²`. See
  [`synth_variance`][delta.synth_variance].
- **Masks.** Bad pixels are excluded from the fit, grown by the kernel footprint on
  the convolved image, and bordered by one kernel half-width in the output. A
  boolean mask follows the convention `True == bad`.

## Tuning

Most runs need no tuning — the smoothing parameter is selected automatically by
cross-validation and the convolution direction is chosen from the measured seeing.
The most useful knobs:

| keyword | meaning | default |
|---|---|---|
| `n_max` | Gauss-Hermite max total order (kernel flexibility) | `6` |
| `beta` | kernel basis scale (pixels); auto from FWHM if `None` | `None` |
| `n_knots` | thin-plate knots per axis (spatial flexibility) | `5` |
| `radius` | kernel footprint half-width; `0` auto-sizes from `beta`/`n_max` | `0` |
| `stamp_radius` | half-width of PSF-matching stamps | `15` |
| `threshold_sigma` | stamp detection threshold (σ) | `5.0` |
| `max_stamps` | cap on the number of stamps used | `200` |
| `saturation` | detector saturation level; reject + mask above (`0` = off) | `0.0` |
| `bright_mask` | output-only level masking bright-star cores (`0` = off) | `0.0` |
| `lambda_grid` | smoothing-parameter search grid | `logspace(-6, 6, 25)` |
| `cv_folds` | leave-stamp-out k-fold CV folds for λ selection (`>=2`) | `5` |
| `spatial_scale` | minimum knot spacing (px); coarsens the spatial basis | `None` |
| `decorrelate` / `score` | noise whitening / S/N map | `False` |
| `block` | FFT block size for decorrelation | `256` |
| `catalog` | `(N, 2)` stamp positions instead of detection | `None` |

A few notes on the less-obvious knobs:

- **`n_max`** — `6` is a robust default; `8` sharpens bright-star cores on
  well-sampled frames but overfits sparse ones.
- **`saturation` vs `bright_mask`** — `saturation` is the *instrument* detector
  level: genuinely saturated (non-linear) pixels are rejected from stamp selection
  *and* masked in the output. `bright_mask` is a *data-driven, output-only* level:
  it masks the cores of bright (but linear) stars whose few-percent PSF-match
  residual would otherwise dominate the difference, without removing those high-S/N
  stars from the kernel fit. Both grow by the kernel radius in the output.
- **`spatial_scale`** — a physical prior: the kernel varies on scales much larger
  than a stamp, so knots are placed no finer than `spatial_scale` pixels apart. A
  coarse spatial basis cannot over-fit small-scale structure. `None` keeps
  `n_knots`.

The full set of knobs is documented on [`Subtractor`][delta.Subtractor].

## Object-oriented / batch use

Construct a [`Subtractor`][delta.Subtractor] once and reuse it across many image
pairs (it carries the configuration; the images are passed per call):

```python
sub = delta.Subtractor(n_max=6, n_knots=6, decorrelate=True, score=True)

solution = sub.fit(science, reference, gain=1.5)      # just the kernel solution
result = sub.subtract(science, reference, gain=1.5)   # fit + difference products

solution.save("kernel.npz")
solution = delta.KernelSolution.load("kernel.npz")
solution.photometric_scale()      # per-pixel flux-scale field Σ_n a_n S_n
```

See [`KernelSolution`][delta.KernelSolution] for the serialized fit result and its
diagnostics.

## Validation and benchmarks

- [`delta.validation`][delta.validation] provides synthetic injection/recovery
  helpers (`inject`, `peak_near`, `aperture_flux`, `completeness`) used by the test
  suite.
- The test suite writes side-by-side JPG previews (science | reference | difference
  | score) to `tests/artifacts/` for eyeballing end-to-end results.
- `python -m benchmarks.bench_subtract [size]` times the full pipeline (honours
  `OMP_NUM_THREADS`); see [Performance](performance.md).
- `python -m benchmarks.compare_hotpants` runs the HOTPANTS head-to-head when the
  `hotpants` binary is on `PATH`.
