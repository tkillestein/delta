# Performance: reference-frame latency and thread scaling

This note pins down where `delta` spends its time on the reference survey frame and
where the plausible optimisations lie. It backs SPEC §5 / §13.4 / §13.5: a
survey-cadence latency target on an ~8000×6000 frame, scaling with thread count, and
*lower runtime than HOTPANTS*.

## Running it

```sh
# Reference 8000×6000 frame, full per-stage breakdown (uses all cores):
uv run python -m benchmarks.bench_subtract

# Thread-scaling sweep (one fresh process per point — OpenMP latches the thread
# count at the first parallel region, so the count cannot be changed in-process):
uv run python -m benchmarks.bench_subtract --threads 1,2,4,8,16

# Smaller frame for a quick check; --record appends a SHA-tagged line to
# benchmarks/results/history.jsonl for tracking regressions across commits:
uv run python -m benchmarks.bench_subtract --size 4096 --record
```

Per-stage timings are read back out of the pipeline's own `log_timing`
instrumentation (a temporary loguru sink), so the breakdown stays in lock-step with
the stages the pipeline actually reports.

## Reference numbers

The headline is host-specific (treat the *shape* of the breakdown, not the absolute
seconds, as the durable signal). Captured on a 14-core machine, 8000×6000 frame
(48 Mpix), 1200 stars, decorrelation + score on, with `DELTA_TIMING=1` for the
indented C++ sub-stage splits:

| stage                              | time (s) | share |
|------------------------------------|---------:|------:|
| stamp selection                    |     1.54 |  4.5% |
| kernel solve                       |     5.53 | 16.2% |
| &nbsp;&nbsp;↳ stamp `B_n` convolve |     0.20 |  0.6% |
| &nbsp;&nbsp;↳ GLS solve            |     4.94 | 14.5% |
| full-frame subtraction             |    16.98 | 49.9% |
| &nbsp;&nbsp;↳ spatial fields       |     6.75 | 19.8% |
| &nbsp;&nbsp;↳ model convolve       |     4.16 | 12.2% |
| &nbsp;&nbsp;↳ variance propagation |     4.76 | 14.0% |
| noise decorrelation                |     5.76 | 16.9% |
| matched-filter score               |     3.88 | 11.4% |
| orchestration                      |     0.35 |  1.0% |
| **total**                          |**34.03** |  100% |

Two findings fall straight out of the splits:

- **The biggest single sub-stage is `spatial fields` (~20%)** — evaluating the
  thin-plate-spline coefficient maps `aₙ(x,y)` and background per pixel, *not* the
  convolution. The RBF design matrix is rebuilt for every frame row inside
  `evaluate_fields` (`src/subtract.cpp`). This was invisible before the sub-timers.
- **`B_n` precompute is negligible (~0.6%)** because the fit convolves only the stamp
  windows, not the whole frame — so the kernel solve is essentially all *solve*
  (the λ-grid × CV-fold factorisations), not convolution.

The balance shifts with frame size: **the GLS solve is a fixed cost** (set by stamp
count and the λ-grid × CV-fold solve count, independent of frame area), so on small
frames it dominates (~76% at 1024²), while on the reference frame the
**area-proportional `full-frame subtraction` dominates at ~50%**.

## Thread scaling (4096², no warm-up)

Same 14-core host, 4096×4096 (16.8 Mpix), 419 stars, no warm-up:

| threads | wall (s) | speedup | efficiency |
|--------:|---------:|--------:|-----------:|
|       1 |    23.25 |   1.00x |       100% |
|       2 |    16.73 |   1.39x |        70% |
|       4 |    14.62 |   1.59x |        40% |
|       8 |    13.93 |   1.67x |        21% |
|      14 |    13.69 |   1.70x |        12% |

Scaling is far from ideal: it saturates at ~1.7× by 4 threads and barely moves after.
The convolution/subtract paths (SPEC §5: *threaded over rows/tiles purely for
parallelism*) do parallelise, but a large serial/memory-bound fraction (Amdahl) caps
the win — at this frame size the full-frame passes are memory-bandwidth bound, so
extra cores contend for the same bus rather than adding throughput. This is itself a
finding: the headline lever is per-core efficiency (SIMD/cache/GPU), not more threads.
See the per-stage gaps below.

## Sub-stage instrumentation (`DELTA_TIMING`)

The five Python-orchestrated stages are timed by the pipeline's loguru `log_timing`.
The finer C++ splits above come from opt-in RAII timers
(`include/delta/timing.hpp`): set `DELTA_TIMING=1` and they are accumulated per label,
returned through the `fit_kernel` / `subtract` result dicts under `"timing"`, and
logged by the pipeline in the same `"<label> done in <s>s"` form the benchmark
captures. When `DELTA_TIMING` is unset the timers compile down to one cached bool
check, so production runs are unaffected. To split a stage further, add a
`DELTA_TIME("…")` scope in the relevant `src/*.cpp` and a label to `SUBSTAGES` in
`bench_subtract.py`.

## Where plausible improvements could lie

Ordered by measured share of reference-frame wall time. Each big lever has a tracking
GitHub issue (search the repo issues for "perf:").

### `subtract: spatial fields` — ~20% (`src/subtract.cpp`, `evaluate_fields`)
**The single biggest lever, and a surprise** — it is coefficient-field evaluation, not
convolution. `evaluate_fields` rebuilds the thin-plate-spline design matrix
`spatial.design(points)` for *every frame row* and multiplies by the coefficient
matrix. The RBF design depends only on (x,y) and the fixed knots, so it is recomputed
~6000× over identical structure.
- **Evaluate on a coarse grid + interpolate**: the fields `aₙ(x,y)` vary on knot
  scales (≫ a pixel), so evaluating them on a coarse lattice and bilinearly
  interpolating to full resolution would be near-exact and orders of magnitude cheaper.
- **Cache / factor the design** across rows (the knot-distance RBF terms are shared),
  or evaluate the spline incrementally.

### `kernel solve` (GLS) — ~15% (and dominant on small frames) (`src/solve.cpp`, `src/fit.cpp`)
Now confirmed to be ~all *solve* (`B_n` precompute is ~0.6%). Builds `M = XᵀWX`
(threaded symmetric `rankUpdate` over row chunks) and solves it across the
**λ-grid × CV-fold** grid: `np.logspace(-6, 6, 25)` × `cv_folds=5` ⇒ up to ~125
factorise/solve passes of the normal equations.
- **Reuse factorisations** across the λ-grid (the penalty changes `M` by a known
  low-rank/diagonal term), **warm-start CV folds**, or prune the λ range adaptively.

### `subtract: variance propagation` — ~14% (`src/subtract.cpp`)
The *block-effective squared-kernel* convolution (freeze `K=Σaₙφₙ` per tile, square it,
convolve `Var(R)`). A near-full-frame second convolution.
- Where the kernel is near-constant a **coarser tile grid** (or skipping re-freeze
  between adjacent tiles) trades negligible accuracy for time; a cheaper variance
  approximation is possible when `Var(R)` is near-flat.

### `subtract: model convolve` — ~12% (`src/subtract.cpp`, `src/convolve.cpp`)
The actual separable model convolution (`#pragma omp simd` today). Shares the
convolution-throughput story with score and decorrelation (below): the natural
first **GPU / wider-SIMD / FFT-for-large-kernels** offload candidate, and a place
**fusing with the variance pass** would cut repeated reads of the same neighbourhoods.

### `noise decorrelation` — ~17% (`src/noise.cpp`)
Apodised FFT blocks (FFTW, per-thread reused plans, threaded over blocks).
- Sensitive to `block` size (default 256) — sweep it; larger blocks amortise plan
  overhead and apodisation but raise per-block FFT cost.
- It is optional (`decorrelate=False`); for latency-bound runs the cost/benefit of
  skipping or coarsening it is worth documenting.

### `matched-filter score` — ~11% (`src/noise.cpp`)
A PSF convolution of the (decorrelated) difference. Shares the convolution story with
`full-frame subtraction` — any SIMD/GPU/FFT convolution win lands here too. Also
optional (`score=False`).

### `stamp selection` — ~5% (`src/detect.cpp`)
Smallest share; a full-frame threshold/FWHM scan. Low priority on wall time, but its
*absolute* time was observed to rise slightly with thread count in small-frame sweeps,
which would make it a relative serial bottleneck once the convolutions are sped up —
worth confirming it parallelises (or stays cheap) before it becomes the tall pole.

## Recording for regression tracking

`--record` appends one JSON object per run to `benchmarks/results/history.jsonl`
(git SHA, host, platform, per-stage timings). Diffing the latest two lines after a
change makes a regression — or a win — visible without re-deriving the baseline.
