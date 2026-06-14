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
| stamp selection                    |     1.99 |  8.6% |
| kernel solve                       |     5.22 | 22.4% |
| &nbsp;&nbsp;↳ stamp `B_n` convolve |     0.20 |  0.8% |
| &nbsp;&nbsp;↳ GLS solve            |     4.61 | 19.8% |
| full-frame subtraction             |    13.67 | 58.8% |
| &nbsp;&nbsp;↳ spatial fields       |     6.21 | 26.7% |
| &nbsp;&nbsp;↳ model convolve       |     2.20 |  9.5% |
| &nbsp;&nbsp;↳ variance propagation |     4.13 | 17.8% |
| noise decorrelation                |     1.26 |  5.4% |
| matched-filter score               |     0.80 |  3.4% |
| orchestration                      |     0.32 |  1.4% |
| **total**                          |**23.25** |  100% |

These numbers are *post* the convolution-path optimisation (issue #19): cache-blocked
model convolve and matched filter, plus a de-`cos`'d decorrelation blend (history
below). The previous breakdown totalled **34.03 s**; the three convolution paths it
called out are now markedly cheaper — model convolve 4.16→2.20 s, matched filter
3.88→0.80 s, decorrelation 5.76→1.26 s.

Two findings fall straight out of the splits:

- **The biggest single sub-stage is now `spatial fields` (~27%)** — evaluating the
  thin-plate-spline coefficient maps `aₙ(x,y)` and background per pixel, *not* the
  convolution. The RBF design matrix is rebuilt for every frame row inside
  `evaluate_fields` (`src/subtract.cpp`). With the convolutions sped up it is, by some
  margin, the tall pole.
- **`B_n` precompute is negligible (~0.8%)** because the fit convolves only the stamp
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
|       1 |    20.83 |   1.00x |       100% |
|       2 |    14.54 |   1.43x |        72% |
|       4 |    12.97 |   1.61x |        40% |
|       8 |    12.31 |   1.69x |        21% |
|      14 |    12.29 |   1.69x |        12% |

Scaling is far from ideal: it saturates at ~1.7× by 4 threads and barely moves after.
The convolution/subtract paths (SPEC §5: *threaded over rows/tiles purely for
parallelism*) do parallelise, but a large serial/memory-bound fraction (Amdahl) caps
the win — at this frame size the full-frame passes are memory-bandwidth bound, so
extra cores contend for the same bus rather than adding throughput. This confirmed the
headline lever is **per-core efficiency (SIMD/cache), not more threads** — acted on in
issue #19 (cache-blocking the convolutions): the single-thread baseline dropped from
23.25→20.83 s here while the scaling *shape* is unchanged, because the remaining tall
poles (GLS solve, spatial-field evaluation) are not convolutions. See the per-stage
gaps below.

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

### `subtract: model convolve` — ~10% (`src/subtract.cpp`, `src/convolve.cpp`)
The separable model convolution, now **2D-tiled and fully fused** (issue #19): the
per-component loop runs *inside* a 128×128 tile loop, so a tile-local difference
accumulator plus the shared x-passes for the tile's rows stay in L2 while every
component is swept. The earlier version looped components on the outside with a
full-frame y-pass each, read-modify-writing the whole `diff` `nc`(=28)× and re-reading
each `tx[nx]` from DRAM per component — memory-bandwidth bound. Per-pixel accumulation
order is unchanged, so the result is bit-identical. Remaining levers: **fusing with the
variance pass** (same neighbourhoods, read once) and wider SIMD.

### `noise decorrelation` — ~5% (`src/noise.cpp`)
Apodised FFT blocks (FFTW, per-thread reused plans, threaded over blocks). The FFT is
genuine (the ZOGY decorrelation filter is non-compact in Fourier space), so it stays.
The recent win (issue #19) was in the *blend*, not the FFT: the overlap-add Hann window
is precomputed once (it was evaluating two `std::cos` per pixel of every overlapping
block, ~370M calls) and the weight normalisation is computed as a **separable outer
product** rather than accumulating a `w·h` weight image in the serialised critical
section. Remaining levers: sweep `block` size (default 256); it is optional
(`decorrelate=False`).

### `matched-filter score` — ~3% (`src/noise.cpp`)
A PSF correlation of the (decorrelated) difference, now **cache-blocked** (issue #19):
each tile gathers its haloed input window once into a contiguous buffer (kept hot in
L1/L2 across the `ks²` taps) and applies the PSF taps as unit-stride `#pragma omp simd`
multiply-adds — mirroring the variance pass. The earlier output-outer/kernel-inner loop
re-read a full `ks²` window per output pixel straight off the strided frame. Also
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
