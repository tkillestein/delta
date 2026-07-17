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

### Measurement hygiene — use `--reps` (warm min-of-N)

**A single warm sample is not trustworthy on this host.** First-touch/page-fault
cost and thermal throttling drift any one run by 20-40%, and back-to-back A/B
comparisons are worse: the second variant runs on a hotter chip, so whichever you
measure second looks slower. This is not hypothetical — an early "the GLS solve is
nearly serial (1.04× over 14 threads)" finding was pure artifact of cold,
single-sample runs; warm min-of-N showed the same code already scaling ~1.8×.

The robust estimator is **warm min-of-N**: warm up once, run N timed iterations,
take the *minimum* per stage (the least-contended sample — the floor a regression
has to beat). Pass `--reps N` (≥5 for A/B work). For thread scaling, drive
`OMP_NUM_THREADS` across separate processes (OpenMP latches the count at the first
parallel region) and let the per-process min absorb the thermal drift::

    OMP_NUM_THREADS=14 python -m benchmarks.bench_subtract --size 4096 --reps 7

## Reference numbers

The headline is host-specific (treat the *shape* of the breakdown, not the absolute
seconds, as the durable signal). Captured on a 14-core machine, 8000×6000 frame
(48 Mpix), 1200 stars, decorrelation + score on, warm min-of-7 (`--reps 7`), with
`DELTA_TIMING=1` for the indented C++ sub-stage splits:

| stage                              | time (s) | share |
|------------------------------------|---------:|------:|
| stamp selection                    |     0.39 |  7.6% |
| kernel solve                       |     1.17 | 22.7% |
| &nbsp;&nbsp;↳ stamp `B_n` convolve |     0.09 |  1.7% |
| &nbsp;&nbsp;↳ GLS solve            |     0.55 | 10.7% |
| full-frame subtraction             |     1.49 | 28.9% |
| &nbsp;&nbsp;↳ spatial fields       |     0.00 |  0.0% |
| &nbsp;&nbsp;↳ sanitise reference   |     0.10 |  1.9% |
| &nbsp;&nbsp;↳ model convolve       |     0.60 | 11.6% |
| &nbsp;&nbsp;↳ variance propagation |     0.21 |  4.0% |
| noise decorrelation                |     1.30 | 25.3% |
| matched-filter score               |     0.56 | 10.8% |
| orchestration                      |     0.25 |  4.8% |
| **wall**                           | **5.15** |  100% |

(Per-stage figures are each an independent warm min, so they need not sum to the min
wall — the floor each stage has to beat is what a regression is measured against. The
sub-stage splits also do not fully account for their parent: `kernel solve` carries
~0.5 s of Python-side stamp prep + CV orchestration outside the two C++ timers, and
`full-frame subtraction` carries mask growth + difference assembly.)

This is *post* the full optimisation arc — spatial-field coarse grid (#16), GLS solve
(#17), variance propagation (#18), convolution-path cache-blocking (#19) — **and** the
later round folded in on this branch. Treat the *shape*, not the absolute seconds, as
durable (an earlier capture of the pre-#17/#18 code on this class of host read ~21 s;
the ratios are the signal). The later round, all bit-identical or validated within
tolerance:

- **`spatial fields` → ~0 and `model convolve` 2.26→0.60 s**: the coefficient fields are
  no longer materialised full-frame — they are interpolated tile-local *inside* the
  model convolve, and the separable x-pass is streamed per tile rather than staged to
  DRAM (issue #32). The separate `spatial fields` stage is now effectively free.
- **`noise decorrelation` 1.41→1.30 s**: per-cell kernel-power cache (one kernel FFT per
  coarse cell) + subsampled block-noise median, after the overlap-add blend was
  de-serialised (#28).
- **`sanitise reference` fused** into the model-convolve gather (the full-frame median-
  filled copy — a 512 MB round-trip — is gone; only the fill scalar is computed up front).
- **`matched-filter score` 0.87→0.56 s**: the dense PSF correlation reordered to
  output-row-outer so the accumulator stays L1-resident (it was L1-bandwidth bound, not
  tap bound — a separable PSF was tried and ran *slower*).
- **orchestration** trimmed: the redundant full-frame difference copy/`astype` removed
  and the rms log made lazy (the silent-by-default library no longer computes a 48 Mpix
  `np.std` for a message no sink emits).

### Final pre-deployment pass (2026-07)

A further sweep, A/B'd on a 4-core host (same commit pair, same box, warm
min-of-5, 8000×6000): **wall 8.35 → 6.29 s (1.33×)**. Stage deltas:

| stage                  | before (s) | after (s) | delta |
|------------------------|-----------:|----------:|------:|
| stamp selection        |       0.33 |      0.23 |  −30% |
| kernel solve           |       1.67 |      0.90 |  −46% |
| full-frame subtraction |       2.74 |      2.28 |  −17% |
| noise decorrelation    |       1.56 |      1.25 |  −20% |
| matched-filter score   |       1.25 |      0.83 |  −33% |

What landed (all bit-identical — verified by output-hash probes — except the
effective-dof identity, which is exact in real arithmetic and shifts only
roundoff):

- **Zero-copy input views** (`ImageView` in `image.hpp`): `subtract`,
  `fit_kernel`, `decorrelate`, and `matched_filter` read the caller's NumPy
  buffers directly instead of copying every full-frame layer into an owning
  `ImageF` (~190 MB/layer); the one-pass range constructor covers the
  remaining owning marshals (`write_fits` etc.). This is most of the score /
  decorrelation stage deltas — their C++ kernels were already tight.
- **Mask growth fused + threaded** (`subtract.cpp`): five serial full-frame
  sweeps + two dilations became one threaded flag pass + one combined
  dilation (OR-dilation is per-bit independent), with the dilated footprint
  folded into the existing edge/non-finite pass (~2× on 4 threads, more on
  wider hosts).
- **IRLS reweight threaded and the thin-plate design cached** across IRLS
  passes (`fit.cpp`; the design was rebuilt inside the solver every pass and
  again for residuals — the saving scales as O(N·k²) with knot count).
- **Detect peak scan** band-parallelised with a deterministic band-order merge.
- **Effective-dof trace identity** in `solve_at`:
  `tr(A⁻¹M) = P − λ‖L⁻¹F‖²_F` with `P0 = FFᵀ` (`System::pen_half`) — one
  triangular solve over rank(P0) columns instead of the full `A⁻¹M` solve,
  ~1.6× per factorisation across every λ-selection path.
- Assorted: convolve row-zeroing deduplicated, `basis_convolve` kernel
  sampling hoisted, merge vectors reserved, per-fold CV training matrices
  hoisted out of the (λ, fold) grid.

Two negative results, recorded so they are not re-attempted: **Demmler-Reinsch
λ-grid amortisation is numerically unsound here** (with a realistic
ill-conditioned thin-plate `M`, the spectral GCV curve misranks λ — up to
~400× curve error on a P=704 probe; see the NOTE in `solve.cpp`), and
**nanobind LTO measured ±1.5%** — indistinguishable from run noise.

**The build now defaults to `-march=x86-64-v3`** (AVX2 + FMA; Haswell 2013+ /
Excavator 2015+, skipped automatically on non-x86 targets). On the same 4-core
host this took the 8000×6000 wall from 6.29 → **4.26 s** (model convolve and
score roughly halved — these loops were written to vectorise and the SSE2
baseline was the limiter), i.e. **1.96× cumulative** over the pre-pass code.
`DELTA_X86_64_V3=OFF` restores the SSE2 baseline for pre-2013 CPUs;
`DELTA_NATIVE=ON` still supersedes both for host-tuned builds (AVX-512 etc.).
Unlike the rest of the pass this shifts float rounding (~1 ulp, FMA
contraction); the full suite passes unchanged. The decorrelate output pair is
also now adopted from the overlap-add accumulators (no fresh full-frame
allocation), and `catalog` is attributed as its own benchmark stage instead of
landing in "other".

Two findings fall straight out of the splits:

- **`noise decorrelation` is now the single tallest stage (~25%)**, ahead of `model
  convolve` (~12%), `matched-filter score` (~11%), and `GLS solve` (~11%) — which are
  themselves within ~1.1× of each other. The FFT is genuine (the ZOGY filter is
  non-compact in Fourier space) so it stays; further decorrelation wins need
  approximation, and the cheap structural levers across the other stages are largely
  spent. `variance propagation` (~4%) and `spatial fields` (~0%) have collapsed from
  their former top-two status.
- **`B_n` precompute is negligible (~1.7%)** because the fit convolves only the stamp
  windows, not the whole frame — so the kernel solve is essentially all *solve*
  (the λ-grid × CV-fold factorisations), not convolution.

The balance shifts with frame size: **the GLS solve is a fixed cost** (set by stamp
count and the λ-grid × CV-fold solve count, independent of frame area), so on small
frames it dominates (~62% at 2048²), while on the reference frame the
**area-proportional `full-frame subtraction` dominates at ~42%**.

## Thread scaling (4096², warm min-of-5)

Same 14-core host, 4096×4096 (16.8 Mpix), 419 stars, warm min-of-5 (`--reps 5`):

| threads | wall (s) | speedup | efficiency |
|--------:|---------:|--------:|-----------:|
|       1 |     4.74 |   1.00x |       100% |
|       2 |     3.14 |   1.51x |        76% |
|       4 |     2.67 |   1.78x |        44% |
|       8 |     2.23 |   2.13x |        27% |
|      14 |     2.43 |   1.95x |        14% |

Scaling is far from ideal: it climbs to ~2.1× by 8 threads, then *regresses* at 14 as
the extra cores contend for the bus (and oversubscribe alongside the Eigen thread
team). The cap is **memory bandwidth, not a serial section**: at this frame size the
full-frame convolution/subtract passes (SPEC §5: *threaded over rows/tiles purely for
parallelism*) are bandwidth-bound, so beyond ~4–8 threads extra cores contend for the
same bus rather than adding throughput. The headline lever is therefore **per-core
efficiency (SIMD/cache), not more threads** — the cache-blocking and tile-local fusion
across the convolution/score paths (#19, #32) cut the single-thread baseline hard while
leaving the scaling *shape* bandwidth-limited.

Note this is a *wall-time* ceiling on the area-proportional passes, **not** evidence
that any one stage is serial. The GLS solve in particular — long mis-read as serial
from cold single-sample runs — scales ~1.8× warm via its threaded fold M-build (see
the GLS section), the same bandwidth-limited ceiling as the rest. Measure stage
scaling with `--reps` (warm min-of-N); a cold sample will not show it. See the
per-stage gaps below.

## Sub-stage instrumentation (`DELTA_TIMING`)

The five Python-orchestrated stages are timed by the pipeline's loguru `log_timing`.
The finer C++ splits above come from opt-in RAII timers
(`include/delta/timing.hpp`): set `DELTA_TIMING=1` and they are accumulated per label,
returned through the `fit_kernel` / `subtract` / `decorrelate` result dicts under
`"timing"` (for `matched_filter`, which returns a bare array, the pipeline drains them
via `_core.drain_timing()`), and logged by the pipeline in the same
`"<label> done in <s>s"` form the benchmark captures. When `DELTA_TIMING` is unset the timers compile down to one cached bool
check, so production runs are unaffected. To split a stage further, add a
`DELTA_TIME("…")` scope in the relevant `src/*.cpp` and a label to `SUBSTAGES` in
`bench_subtract.py`.

## Where plausible improvements could lie

Ordered by the share each lever had *when first identified* (completed ones are kept,
marked **DONE**, for the history). Each big lever has a tracking GitHub issue (search
the repo issues for "perf:").

### `subtract: spatial fields` — ~0%, was ~27% (`src/subtract.cpp`, `evaluate_fields`) — **DONE (issues #16, #32)**
Was the single biggest lever, and a surprise — coefficient-field evaluation, not
convolution: `evaluate_fields` rebuilt the thin-plate-spline design matrix
`spatial.design(points)` for *every frame row* (~6000× over identical structure), an
`h·w·k` count of `log()` evaluations. First fixed (#16) by **coarse-grid evaluation +
bilinear interpolation**: the fields `aₙ(x,y)` vary on the knot length-scale (≫ a
pixel), so they are evaluated exactly on a coarse lattice and bilinearly interpolated to
full resolution (stride chosen adaptively from `ThinPlateBasis::min_knot_spacing()`, ~16
samples per knot interval, capped at 64 px, keeping the relative error well below 1e-3;
small frames fall back to the exact per-pixel path, bit-for-bit). That cut it 6.21→2.51 s.
The remaining materialisation cost was then **eliminated (#32)** by fusing the field
interpolation *into* the model convolve: the coarse lattice (`CoarseFields`) is
interpolated tile-local inside the convolution loop (point-sampled in variance prop), so
the full-resolution `aₙ`/background are never written to DRAM. The separate stage is now
effectively free (~0 s); its former cost moved into — and is hidden by — the model
convolve, which dropped overall (below).

### `kernel solve` (GLS) — ~11%, was ~24% (and dominant on small frames) (`src/solve.cpp`, `src/fit.cpp`) — **DONE (issue #17)**
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

Two further *exact* refinements (this branch) trim it again, ~5-8% on solve-min each
measured with `--reps`:

- **Row-chunked CV fold M-build.** Building each fold's `M_g = Xs_gᵀ Xs_g` was
  threaded one-fold-per-thread, capping parallelism at `cv_folds` (=5) regardless of
  core count — and the build (O(N·P²)) is the solve's *dominant* cost. It now mirrors
  the full-data `assemble()`: the fold loop is serial on the outside but each fold's
  symmetric `rankUpdate` is split over 2048-row chunks across *all* threads (per-thread
  gather buffers hoisted out of the chunk loop — allocating them per chunk regressed the
  1-thread path). This is what lifts the warm solve scaling to ~1.8× (below); the earlier
  "solve is serial" read was the cold-sample artifact, not the M-build.
- **λ-search warm start across IRLS passes.** The σ-clip loop re-runs the whole CV solve
  each pass, but a pass only clips a handful of stamps off a curve the previous pass
  already located — the optimum barely moves. Passes after the first now seed a ±1
  bracket at the previous pass's CV-selected index and descend to the interior minimum
  (a handful of factorisations) instead of the cold coarse-to-fine sweep. The CV curve is
  unimodal in log-λ, so the descent lands on the *identical* λ (verified: same selected λ
  and difference image to the threaded-reduction FP-noise floor, ~1e-5, which two *cold*
  runs already differ by). First pass stays cold-start. Smaller than hoped because the
  M-build, not the λ-grid factorisations (O(P³), small P), is the solve's floor — see
  below.

**Where the solve's floor now is.** With both above, the per-fold M-build dominates: it
runs once per IRLS pass regardless of how many λ points are factorised, so cutting λ
work only trims the tail. The next real lever is avoiding the rebuild on passes that
clip few pixels — a rank downdate of the clipped rows out of the existing `M_g` rather
than a full re-accumulate — which is materially more complex than these two changes.

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

### `subtract: model convolve` — ~12%, was ~14% (`src/subtract.cpp`, `src/convolve.cpp`) — **DONE (issues #19, #32)**
The separable model convolution, **2D-tiled and fully fused** (#19): the per-component
loop runs *inside* a 96×96 tile loop, so a tile-local difference accumulator plus the
shared x-passes for the tile's rows stay in L2 while every component is swept. The
earlier version looped components on the outside with a full-frame y-pass each,
read-modify-writing the whole `diff` `nc`(=28)× and re-reading each `tx[nx]` from DRAM
per component — memory-bandwidth bound. **#32** then removed the two remaining DRAM
round-trips: the separable x-pass is computed *per tile* into a thread-local buffer
(never staged full-frame, ~1.3 GB saved) and the coefficient fields are interpolated
tile-local (the `spatial fields` ~5.4 GB materialisation, above). The **reference
sanitisation is also fused in** (its own stage below): each tile gathers its haloed
reference window with the median fill applied in place, so the convolution reads it with
no per-tap bounds test and the full-frame sanitised copy never materialises. Per-pixel
accumulation order is unchanged throughout, so the result stays bit-identical. y-pass
micro-opts (register-blocking, tap-reorder) are duds — compute/L1-bound at a local
optimum. Remaining lever: **fusing with the variance pass** (same neighbourhoods).

### `subtract: sanitise reference` — ~2% (`src/subtract.cpp`) — **DONE (fused)**
Masked / non-finite reference pixels are replaced by the median background before the
convolution (a zero fill would ring the model at every mask boundary; SPEC §3.6). This
used to materialise a full-frame sanitised *copy* (256 MB write + 256 MB read) whose only
consumer was the model convolve. Now only the median fill **scalar** is computed up front
(stride-7 sample → bit-identical fill) and the substitution is fused into the model-
convolve tile gather. What remains in the stage is just the serial median sample;
coarsening its stride would speed it further but changes the fill value, so it is kept at
7 to preserve bit-identity.

### `noise decorrelation` — ~25% (now the tallest stage) (`src/noise.cpp`)
Apodised FFT blocks (vendored PocketFFT, one workspace per thread, threaded over blocks). The FFT is
genuine (the ZOGY decorrelation filter is non-compact in Fourier space), so it stays.
Wins so far: the overlap-add *blend* was de-serialised via a 9-colour lock-free scheme
and the Hann window precomputed (#28); then a **per-cell kernel-power cache** — `|K̂|²`
and `ΣK²` depend only on the matching kernel, which varies on the knot scale ≫ the block
stride, so they are cached one kernel FFT per coarse cell (auto-sized from knot spacing)
and reused across the cell's blocks, dropping one of the three FFTs per block — plus a
**subsampled block-noise median** (`block_variance` ran an `nth_element` over 65 k floats
twice per block; now ~4 k samples). Validated within tolerance against the exact
per-block path (`kernel_cell_blocks=1`). As the now-tallest stage with the FFT
irreducible, further wins need a coarser approximation; it is optional
(`decorrelate=False`).

### `matched-filter score` — ~11% (`src/noise.cpp`) — **DONE (reorder)**
A dense PSF correlation of the (decorrelated) difference. Each tile gathers its haloed
input window once into a contiguous L1/L2-hot buffer (#19). The hot loop is now
**output-row-outer** (`oy, ly, lx, ox`): a single output row's accumulator stays
L1/register-resident while all `ks²` taps reduce into it, and each tap reads only the
`ks` window rows directly above the row. The earlier tap-outer order swept the whole tile
per tap, so the accumulator and haloed window together overran L1 and the window
thrashed — the stage was **L1-bandwidth bound, not tap bound** (a separable low-rank PSF
was tried and ran *slower* for exactly this reason, and is not worth the approximation).
Same per-pixel `(ly,lx)` sum order → bit-identical. Also optional (`score=False`).

### `stamp selection` — ~8% (`src/detect.cpp`)
Smallest share; a full-frame threshold/FWHM scan. Low priority on wall time, but its
*absolute* time was observed to rise slightly with thread count in small-frame sweeps,
which would make it a relative serial bottleneck once the convolutions are sped up —
worth confirming it parallelises (or stays cheap) before it becomes the tall pole.

## Recording for regression tracking

`--record` appends one JSON object per run to `benchmarks/results/history.jsonl`
(git SHA, host, platform, per-stage timings). Diffing the latest two lines after a
change makes a regression — or a win — visible without re-deriving the baseline.
