"""End-to-end subtraction benchmark (SPEC §10, §13.5).

Times the full pipeline on a synthetic frame and reports throughput. Thread count
follows OMP_NUM_THREADS. Usage:

    python -m benchmarks.bench_subtract [size] [n_stars]
    OMP_NUM_THREADS=8 python -m benchmarks.bench_subtract 4096
"""

from __future__ import annotations

import os
import sys
import time

import delta
import numpy as np
from delta import validation


def build_pair(size: int, n_stars: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    xs = rng.integers(20, size - 20, n_stars)
    ys = rng.integers(20, size - 20, n_stars)
    fluxes = rng.uniform(4000.0, 20000.0, n_stars)
    ref = np.full((size, size), 120.0)
    sci = np.full((size, size), 120.0)
    for x, y, f in zip(xs, ys, fluxes, strict=True):
        ref += validation.gaussian_psf((size, size), x, y, f, 1.6)
        sci += validation.gaussian_psf((size, size), x, y, f, 2.4)
    var = np.full((size, size), 9.0, np.float32)
    ref = (ref + rng.normal(0, 3.0, (size, size))).astype(np.float32)
    sci = (sci + rng.normal(0, 3.0, (size, size))).astype(np.float32)
    return sci, ref, var


def main(argv: list[str]) -> int:
    size = int(argv[1]) if len(argv) > 1 else 1024
    n_stars = int(argv[2]) if len(argv) > 2 else max(50, size // 8)
    threads = os.environ.get("OMP_NUM_THREADS", "(default)")

    sci, ref, var = build_pair(size, n_stars)
    # Warm up (plan caches, first-touch allocation).
    delta.subtract(sci, ref, science_var=var, reference_var=var, n_knots=4, stamp_radius=12)

    t0 = time.perf_counter()
    res = delta.subtract(
        sci,
        ref,
        science_var=var,
        reference_var=var,
        n_knots=5,
        stamp_radius=12,
        decorrelate=True,
        score=True,
    )
    dt = time.perf_counter() - t0

    mpix = size * size / 1e6
    print(f"frame      : {size}x{size} ({mpix:.1f} Mpix), {n_stars} stars")
    print(f"threads    : OMP_NUM_THREADS={threads}")
    print(
        f"direction  : {res.solution.direction}, beta={res.solution.beta:.2f}, "
        f"lambda={res.solution.lam:.2e}"
    )
    print(f"wall time  : {dt:.3f} s   ({mpix / dt:.1f} Mpix/s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
