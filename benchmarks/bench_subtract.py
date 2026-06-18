"""End-to-end subtraction benchmark with per-stage breakdown and thread scaling.

Pins down absolute pipeline latency at the reference frame size (SPEC §5, §13.5:
the ~8000×6000 survey frame) and measures OpenMP thread scaling, so regressions
and the HOTPANTS runtime target (SPEC §13.4) are visible over time.

Two modes:

* single run — synthesize a frame, run the full pipeline (after a warm-up), and
  report a per-stage timing breakdown::

      python -m benchmarks.bench_subtract                 # 8000x6000 reference
      python -m benchmarks.bench_subtract --size 4096     # square shortcut
      OMP_NUM_THREADS=8 python -m benchmarks.bench_subtract

  Pass ``--reps N`` for the robust **warm min-of-N** estimator: warm up once, time
  N iterations, and report the *minimum* per stage. Use this for any A/B or scaling
  comparison — a single warm sample drifts 20-40% under thermal throttling and
  first-touch cost, enough to invent or mask a regression (it once made the GLS
  solve look serial). The min is the least-contended sample, which is what a real
  change has to beat::

      python -m benchmarks.bench_subtract --size 4096 --reps 7

* thread sweep — re-run the single mode in subprocesses across a range of
  ``OMP_NUM_THREADS`` and report speedup vs the single-thread baseline (OpenMP
  reads the thread count once at the first parallel region, so each point needs a
  fresh process)::

      python -m benchmarks.bench_subtract --threads 1,2,4,8

Set ``DELTA_TIMING=1`` to additionally surface the C++ core's sub-stage timers
(``B_n`` convolve vs GLS solve; model-convolve vs variance vs mask passes),
printed indented under their parent stage.

Pass ``--record`` to append a JSON line to ``benchmarks/results/`` tagged with the
git SHA, host, and frame size, for tracking regressions across commits.

See ``benchmarks/PERFORMANCE.md`` for the stage breakdown and where plausible
optimisations lie.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import delta
import numpy as np
from delta import validation

# log_timing emits "<label> done in <seconds>s" at INFO; parse the stage timings
# back out of the library's own instrumentation rather than re-timing by hand.
_TIMING_RE = re.compile(r"^(?P<label>.+?) done in (?P<seconds>[\d.]+)s$")

# Stage labels the pipeline reports, in execution order (see pipeline.py).
STAGES = (
    "stamp selection",
    "kernel solve",
    "full-frame subtraction",
    "noise decorrelation",
    "matched-filter score",
)

# C++ sub-stage timers nested under their parent (emitted only with DELTA_TIMING
# set; see include/delta/timing.hpp). Shown indented under the parent stage.
SUBSTAGES = {
    "kernel solve": ("fit: stamp B_n convolve", "fit: GLS solve"),
    "full-frame subtraction": (
        "subtract: spatial fields",
        "subtract: sanitise reference",
        "subtract: model convolve",
        "subtract: variance propagation",
        "subtract: mask growth",
    ),
}

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def build_pair(
    width: int, height: int, n_stars: int, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A realistic science/reference pair: power-law star field, two PSF widths.

    Uses the local-box renderer (``render_stars``) rather than a full-frame mgrid
    per source, so synthesizing the 48 Mpix reference frame stays cheap.
    """
    rng = np.random.default_rng(seed)
    positions, fluxes = validation.sample_starfield(
        (height, width), n_stars, rng, border=24, min_separation=12.0
    )
    sky = 120.0
    ref = sky + validation.render_stars((height, width), positions, fluxes, 1.6)
    sci = sky + validation.render_stars((height, width), positions, fluxes, 2.4)
    var = np.full((height, width), 9.0, np.float32)
    ref = (ref + rng.normal(0, 3.0, (height, width))).astype(np.float32)
    sci = (sci + rng.normal(0, 3.0, (height, width))).astype(np.float32)
    return sci, ref, var


@contextmanager
def capture_stage_timings() -> Iterator[dict[str, float]]:
    """Capture the pipeline's ``log_timing`` durations into a ``{label: seconds}``.

    Adds a temporary loguru sink, enables the ``delta`` namespace for the block,
    and restores the prior (silent) configuration on exit.
    """
    from delta._log import logger

    timings: dict[str, float] = {}

    def sink(message) -> None:
        match = _TIMING_RE.match(message.record["message"])
        if match:
            timings[match["label"]] = float(match["seconds"])

    # DEBUG so the opt-in C++ sub-stage timers (logged at DEBUG) are captured too.
    sink_id = logger.add(sink, level="DEBUG", format="{message}")
    logger.enable("delta")
    try:
        yield timings
    finally:
        logger.remove(sink_id)
        logger.disable("delta")


@dataclass
class RunResult:
    width: int
    height: int
    n_stars: int
    threads: int
    direction: str
    beta: float
    lam: float
    wall_s: float
    mpix_per_s: float
    stages: dict[str, float] = field(default_factory=dict)


def run_once(
    width: int,
    height: int,
    n_stars: int,
    *,
    decorrelate: bool,
    score: bool,
    warmup: bool,
    reps: int = 1,
    seed: int = 0,
) -> RunResult:
    """Synthesize a frame and time the subtraction, reporting the warm min over `reps`.

    Benchmarking this pipeline cold or single-shot is unreliable: page-fault /
    first-touch cost and thermal-throttle spikes inflate a single sample by 20-40%
    and made earlier scaling reads (e.g. "the GLS solve is serial") pure artifact.
    The robust estimator is **warm min-of-N**: warm up once, run `reps` timed
    iterations, and take the *minimum* per stage (the least-contended sample, which
    is what a regression actually has to beat). Drive thread scaling from
    ``OMP_NUM_THREADS`` across separate processes; min absorbs the within-process
    thermal drift that corrupts back-to-back A/B comparisons.
    """
    sci, ref, var = build_pair(width, height, n_stars, seed)
    kwargs = dict(
        science_var=var,
        reference_var=var,
        n_knots=5,
        stamp_radius=12,
        decorrelate=decorrelate,
        score=score,
    )
    if warmup:
        # First-touch allocation, FFT workspace warmup: run once and discard.
        delta.subtract(sci, ref, **kwargs)

    wall = float("inf")
    stage_min: dict[str, float] = {}
    res = None
    for _ in range(max(1, reps)):
        with capture_stage_timings() as stages:
            t0 = time.perf_counter()
            res = delta.subtract(sci, ref, **kwargs)
            w = time.perf_counter() - t0
        wall = min(wall, w)
        # Per-stage min independently: each stage's least-contended sample. (The
        # column won't sum to the min wall, by construction -- that's intended; we
        # want each stage's floor, not one run's snapshot.)
        for label, secs in stages.items():
            stage_min[label] = min(stage_min.get(label, float("inf")), secs)

    assert res is not None  # loop runs at least once
    mpix = width * height / 1e6
    threads = int(os.environ.get("OMP_NUM_THREADS", "0")) or os.cpu_count() or 1
    return RunResult(
        width=width,
        height=height,
        n_stars=n_stars,
        threads=threads,
        direction=res.solution.direction,
        beta=float(res.solution.beta),
        lam=float(res.solution.lam),
        wall_s=wall,
        mpix_per_s=mpix / wall,
        stages=stage_min,
    )


def print_run(result: RunResult) -> None:
    mpix = result.width * result.height / 1e6
    print(f"frame      : {result.width}x{result.height} ({mpix:.1f} Mpix), {result.n_stars} stars")
    print(f"threads    : OMP_NUM_THREADS={result.threads}")
    print(f"direction  : {result.direction}, beta={result.beta:.2f}, lambda={result.lam:.2e}")
    print("stage breakdown:")
    accounted = 0.0
    for label in STAGES:
        if label in result.stages:
            secs = result.stages[label]
            accounted += secs
            print(f"  {label:<26} {secs:8.3f} s  ({100 * secs / result.wall_s:5.1f}%)")
            # Indented C++ sub-stage breakdown when DELTA_TIMING captured it.
            for child in SUBSTAGES.get(label, ()):
                if child in result.stages:
                    cs = result.stages[child]
                    print(f"    {child:<30} {cs:8.3f} s  ({100 * cs / result.wall_s:5.1f}%)")
    other = result.wall_s - accounted
    print(f"  {'other (orchestration)':<26} {other:8.3f} s  ({100 * other / result.wall_s:5.1f}%)")
    print(f"wall time  : {result.wall_s:.3f} s   ({result.mpix_per_s:.1f} Mpix/s)")


def thread_sweep(
    threads: list[int],
    width: int,
    height: int,
    n_stars: int,
    *,
    decorrelate: bool,
    score: bool,
    reps: int = 1,
) -> list[RunResult]:
    """Re-run single mode in a subprocess per thread count, parsing emitted JSON."""
    results: list[RunResult] = []
    for n in threads:
        env = dict(os.environ, OMP_NUM_THREADS=str(n))
        cmd = [
            sys.executable,
            "-m",
            "benchmarks.bench_subtract",
            "--width",
            str(width),
            "--height",
            str(height),
            "--n-stars",
            str(n_stars),
            "--reps",
            str(reps),
            "--emit-json",
        ]
        if not decorrelate:
            cmd.append("--no-decorrelate")
        if not score:
            cmd.append("--no-score")
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, check=True)
        results.append(RunResult(**json.loads(proc.stdout.strip().splitlines()[-1])))
    return results


def print_sweep(results: list[RunResult]) -> None:
    baseline = results[0]
    print(f"\nthread scaling ({baseline.width}x{baseline.height}, {baseline.n_stars} stars):")
    print(f"  {'threads':>8}  {'wall (s)':>10}  {'speedup':>8}  {'efficiency':>10}")
    for r in results:
        speedup = baseline.wall_s / r.wall_s
        efficiency = speedup / (r.threads / baseline.threads)
        print(f"  {r.threads:>8}  {r.wall_s:>10.3f}  {speedup:>7.2f}x  {100 * efficiency:>9.0f}%")


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def record_results(results: list[RunResult]) -> Path:
    """Append a timestamped, SHA-tagged JSON line to ``benchmarks/results/``."""
    RESULTS_DIR.mkdir(exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "host": platform.node(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "runs": [asdict(r) for r in results],
    }
    path = RESULTS_DIR / "history.jsonl"
    with path.open("a") as fh:
        fh.write(json.dumps(record) + "\n")
    return path


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=8000, help="frame width (default: 8000)")
    parser.add_argument("--height", type=int, default=6000, help="frame height (default: 6000)")
    parser.add_argument("--size", type=int, help="square frame shortcut (overrides w/h)")
    parser.add_argument("--n-stars", type=int, help="default: width*height/40000")
    parser.add_argument(
        "--threads",
        type=str,
        help="comma-separated thread counts for a scaling sweep, e.g. 1,2,4,8",
    )
    parser.add_argument("--no-decorrelate", action="store_true")
    parser.add_argument("--no-score", action="store_true")
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument(
        "--reps",
        type=int,
        default=1,
        help="timed warm iterations; report the per-stage min (default: 1)",
    )
    parser.add_argument("--record", action="store_true", help="append results to history.jsonl")
    parser.add_argument(
        "--emit-json", action="store_true", help="print one JSON line to stdout (sweep subprocess)"
    )
    args = parser.parse_args(argv[1:])

    width = args.size if args.size else args.width
    height = args.size if args.size else args.height
    n_stars = args.n_stars if args.n_stars else max(50, width * height // 40000)
    decorrelate = not args.no_decorrelate
    score = not args.no_score

    if args.threads:
        threads = [int(t) for t in args.threads.split(",")]
        results = thread_sweep(
            threads,
            width,
            height,
            n_stars,
            decorrelate=decorrelate,
            score=score,
            reps=args.reps,
        )
        for r in results:
            print_run(r)
            print()
        print_sweep(results)
        if args.record:
            print(f"\nrecorded -> {record_results(results)}")
        return 0

    result = run_once(
        width,
        height,
        n_stars,
        decorrelate=decorrelate,
        score=score,
        warmup=not args.no_warmup,
        reps=args.reps,
    )
    if args.emit_json:
        # Logs go to the captured sink; stdout stays pure JSON for the parent.
        print(json.dumps(asdict(result)))
        return 0

    print_run(result)
    if args.record:
        print(f"\nrecorded -> {record_results([result])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
