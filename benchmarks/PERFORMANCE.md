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
| stamp selection                    |     2.16 | 13.8% |
| kernel solve                       |     4.24 | 27.0% |
| &nbsp;&nbsp;↳ stamp `B_n` convolve |     0.19 |  1.2% |
| &nbsp;&nbsp;↳ GLS solve            |     3.62 | 23.1% |
| full-frame subtraction             |     6.68 | 42.5% |
| &nbsp;&nbsp;↳ spatial fields       |     2.72 | 17.3% |
| &nbsp;&nbsp;↳ model convolve       |     2.26 | 14.4% |
| &nbsp;&nbsp;↳ variance propagation |     0.47 |  3.0% |
| noise decorrelation                |     1.41 |  9.0% |
| matched-filter score               |     0.87 |  5.5% |
| orchestration                      |     0.34 |  2.2% |
| **total**                          |**15.70** |  100% |

These numbers are *post* the GLS-solve and variance-propagation optimisations (issues
#17 and #18); the convolution-path (#19) and spatial-field coarse-grid (#16) wins are
folded in. Re-captured on this host, the pre-#17/#18 baseline was **21.33 s** (the
absolute seconds are host-specific — an earlier capture on a different 14-core machine
read 19.70 s for the same pre-#17/#18 code; treat the *ratios* as durable). The two tall
poles it called out are now markedly cheaper: **GLS solve 5.26→3.62 s** (#17) and
**variance propagation 4.46→0.47 s** (#18), dropping the total 21.33→15.70 s (−26%) and
flipping `full-frame subtraction` from 50% to ~42% of wall time.

Two findings fall straight out of the splits:

- **No single tall pole remains**: the largest sub-stages — `GLS solve` (~23%),
  `spatial fields` (~17%), and `model convolve` (~14%) — are now within ~1.6× of each
  other, so further wins must spread across them rather than chase one. `variance
  propagation`, the previous #2 at ~23%, has collapsed to ~3% (#18); `spatial fields`
  (#16) and the convolution paths (#19) were cut earlier.
- **`B_n` precompute is negligible (~1.2%)** because the fit convolves only the stamp
  windows, not the whole frame — so the kernel solve is essentially all *solve*
  (the λ-grid × CV-fold factorisations), not convolution.

The balance shifts with frame size: **the GLS solve is a fixed cost** (set by stamp
count and the λ-grid × CV-fold solve count, independent of frame area), so on small
frames it dominates (~62% at 2048²), while on the reference frame the
**area-proportional `full-frame subtraction` dominates at ~42%**.

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
23.25→20.83 s here while the scaling *shape* is unchanged, because the dominant tall
pole (the GLS solve) is not a convolution. (Spatial-field evaluation, the other large
non-convolution cost at the time, has since been cut by issue #16.) See the per-stage
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

Ordered by the share each lever had *when first identified* (completed ones are kept,
marked **DONE**, for the history). Each big lever has a tracking GitHub issue (search
the repo issues for "perf:").

### `subtract: spatial fields` — ~13%, was ~27% (`src/subtract.cpp`, `evaluate_fields`) — **DONE (issue #16)**
Was the single biggest lever, and a surprise — coefficient-field evaluation, not
convolution: `evaluate_fields` rebuilt the thin-plate-spline design matrix
`spatial.design(points)` for *every frame row* (~6000× over identical structure), an
`h·w·k` count of `log()` evaluations. Fixed by **coarse-grid evaluation +
bilinear interpolation**: the fields `aₙ(x,y)` vary on the knot length-scale (≫ a
pixel), so they are now evaluated exactly on a coarse lattice and bilinearly
interpolated to full resolution. The stride is chosen adaptively from the smallest knot
spacing (`ThinPlateBasis::min_knot_spacing()`, ~16 samples per knot interval, capped at
64 px), so the relative error stays well below 1e-3; small frames fall back to the exact
per-pixel path (bit-for-bit). This cut the stage 6.21→2.51 s (~2.5×). What remains is
the memory-bound interpolation + full-frame field materialisation (the downstream model
convolve reads `aₙ` per pixel), not the RBF/`log` cost — a further win would need to
fuse field evaluation into the convolve so the full-resolution fields are never
materialised.

### `kernel solve` (GLS) — ~18%, was ~24% (and dominant on small frames) (`src/solve.cpp`, `src/fit.cpp`) — **DONE (issue #17)**
~All *solve* (`B_n` precompute is ~1.0%). Builds `M = XᵀWX` (threaded symmetric
`rankUpdate` over row chunks) and solves it across the **λ-grid × CV-fold** grid:
`np.logspace(-6, 6, 25)` × `cv_folds=5` ⇒ up to ~125 factorise/solve passes of the
normal equations. Two *exact* levers cut it **5.26→3.62 s** (−31%):

- **Cholesky (`LLT`) instead of the pivoted `LDLT`** for the per-(λ, fold) factorisation
  of the SPD penalised normal matrix `A = M_train + λP₀` (with an `LDLT` fallback if a
  fold's system loses positive-definiteness at tiny λ). This was the larger win — on the
  real per-stamp normal matrices Eigen's `LDLT` is far slower than its `LLT` (≈ −44% on
  the solve), though on random matrices the two are within ~6%, so it only showed up
  end-to-end.
- **Coarse-to-fine λ search** in the CV path: the held-out CV error is unimodal in
  log-λ, so a stride-2 coarse scan plus a ±2 bracket refinement around its minimum
  evaluates 15 of 25 grid points and selects the *identical* λ as the full sweep for any
  unimodal curve. The reduced-χ²/effective-dof diagnostics come from the exact final
  solve, so they are untouched.

The hypothesised **factorisation reuse across the λ-grid** (a single generalised
eigendecomposition of the `(M, P₀)` pencil, then O(P)-per-λ diagnostics) was implemented
and measured but *does not pay off* at the relevant size: one symmetric eigensolve at
P = (nc+1)k ≈ 812 costs ≈ 25–50 Cholesky factorisations, so it is a net loss for both
the 25-point GCV grid and the 5-fold CV path. Left out.

### `subtract: variance propagation` — ~3%, was ~21% (`src/subtract.cpp`) — **DONE (issue #18)**
The *block-effective squared-kernel* convolution (freeze `K=Σaₙφₙ` per tile, square it,
convolve `Var(R)`) was a near-full-frame second convolution. Cut **4.46→0.47 s** (~9.6×)
by the **flat-`Var(R)` shortcut**: where the reference variance is near-constant over the
kernel footprint — the sky-dominated common case — `K² ⊗ Var(R) ≈ Var(R)·Σᵤ K²(u)`, i.e.
a per-pixel scale by the kernel sum-of-squares `Q`, with no `ks²` convolution at all. Each
tile gathers its `Var(R)` window (tile + kernel-radius halo) and takes the shortcut only
when the window's relative spread `(max−min)/max` is below `1e-3`, which bounds the
shortcut's relative error by the same; otherwise it falls through to the exact
convolution. Because the *haloed* window is the exact footprint the convolution reads, a
tile near a bright source (a Poisson variance spike) reads non-flat and stays exact —
verified to ~1e-6 relative against a direct `K²` convolution on a flat-sky-plus-bump map.
The ~9.6× here is the constant-`Var` best case (every tile flat); a frame with structure
near sources sees less, but the sky tiles that dominate a survey frame still get it.
- Remaining lever: **adaptive tile size** where the kernel itself is near-constant, and
  **fusing with the model-convolve pass** (same neighbourhoods, read once).

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
