# Using delta

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

## Inputs and noise

- **Arrays or astropy.** `science` / `reference` may be NumPy arrays, astropy
  `CCDData`, or FITS HDUs; variance and mask carried by a `CCDData` are picked up
  automatically. Explicit `science_var`, `reference_var`, `science_mask`,
  `reference_mask` override them.
- **Variance.** Provide variance maps directly, or pass `gain` (and optional
  `read_noise`) to synthesize `Var = max(data, 0)/gain + read_noise²`.
- **Masks.** Bad pixels are excluded from the fit, grown by the kernel footprint on
  the convolved image, and bordered by one kernel half-width in the output.

## Tuning

| keyword | meaning | default |
|---|---|---|
| `n_max` | Gauss-Hermite max total order (kernel flexibility) | `2` |
| `beta` | kernel basis scale (pixels); auto from FWHM if `None` | `None` |
| `n_knots` | thin-plate knots per axis (spatial flexibility) | `5` |
| `stamp_radius` | half-width of PSF-matching stamps | `15` |
| `threshold_sigma` | stamp detection threshold | `5.0` |
| `lambda_grid` | GCV smoothing search grid | `logspace(-6, 6, 25)` |
| `decorrelate` / `score` | noise whitening / S/N map | `False` |
| `catalog` | `(N, 2)` stamp positions instead of detection | `None` |

The smoothing parameter λ is selected automatically by generalized
cross-validation — there are no polynomial-degree knobs to tune.

## Object-oriented / batch use

```python
sub = delta.Subtractor(n_max=2, n_knots=6, decorrelate=True, score=True)
solution = sub.fit(science, reference, gain=1.5)   # just the kernel solution
result = sub.subtract(science, reference, gain=1.5)

solution.save("kernel.npz")
solution = delta.KernelSolution.load("kernel.npz")
solution.photometric_scale()      # per-pixel flux-scale field sum_n a_n S_n
```

## Validation and benchmarks

- `delta.validation` provides injection/recovery helpers (`inject`, `peak_near`,
  `aperture_flux`, `completeness`) used by the test suite.
- The test suite writes side-by-side JPG previews (science | reference | difference
  | score) to `tests/artifacts/` for eyeballing end-to-end results.
- `python -m benchmarks.bench_subtract [size]` times the full pipeline
  (honours `OMP_NUM_THREADS`).
- `python -m benchmarks.compare_hotpants` runs the HOTPANTS head-to-head when the
  `hotpants` binary is on `PATH`.
