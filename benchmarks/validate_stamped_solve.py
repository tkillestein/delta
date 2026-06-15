"""Validate + benchmark the per-stamp factorised GLS M-build (handoff item #1).

Compares the stamped approximation (DELTA_STAMP_APPROX=1) against the exact
per-row solve (=0) on a realistic frame: how close are the fitted kernel theta and
the difference image, does injection recovery hold, and how much faster is it.

Run: uv run python benchmarks/validate_stamped_solve.py
"""

import os
import time

import delta
import numpy as np
from delta import validation

from benchmarks.bench_subtract import build_pair


def run(mode, sci, ref, var):
    os.environ["DELTA_STAMP_APPROX"] = mode
    sub = delta.Subtractor(n_knots=5, stamp_radius=15, cv_folds=5, decorrelate=True, score=True)
    t0 = time.perf_counter()
    res = sub.subtract(sci, ref, science_var=var, reference_var=var)
    dt = time.perf_counter() - t0
    return res, dt


def main():
    W = H = 4096
    n_stars = W * H // 40000
    sci, ref, var = build_pair(W, H, n_stars, seed=2)

    # Inject a flux ladder of transients into the science frame.
    pos = [
        (int(x), int(y)) for x in np.linspace(400, W - 400, 5) for y in np.linspace(400, H - 400, 5)
    ]
    fluxes = np.logspace(np.log10(200.0), np.log10(50000.0), len(pos))
    sci = validation.inject(sci.astype(np.float64), pos, fluxes, 2.4).astype(np.float32)

    # Warm both paths once (build caches, etc.), then time.
    for m in ("0", "1"):
        run(m, sci, ref, var)

    res_x, t_x = run("0", sci, ref, var)
    res_s, t_s = run("1", sci, ref, var)
    del os.environ["DELTA_STAMP_APPROX"]

    tx = res_x.solution.theta
    ts = res_s.solution.theta
    theta_rel = np.linalg.norm(ts - tx) / np.linalg.norm(tx)

    dx, ds = res_x.difference, res_s.difference
    diff_rel = np.linalg.norm(ds - dx) / np.linalg.norm(dx)

    ap = 7
    fx = np.array([validation.aperture_flux(dx, p, ap) for p in pos])
    fs = np.array([validation.aperture_flux(ds, p, ap) for p in pos])
    corr_x = np.corrcoef(fluxes, fx)[0, 1]
    corr_s = np.corrcoef(fluxes, fs)[0, 1]

    cx_score = res_x.score[200:600, 200:600]
    cs_score = res_s.score[200:600, 200:600]

    print(f"frame {W}x{H}, {n_stars} stars + {len(pos)} injected\n")
    print(f"{'':22} {'exact':>12} {'stamped':>12}")
    print(f"{'lambda (CV)':22} {res_x.solution.lam:>12.3g} {res_s.solution.lam:>12.3g}")
    cx2, cs2 = res_x.solution.reduced_chi2, res_s.solution.reduced_chi2
    print(f"{'reduced chi2':22} {cx2:>12.4f} {cs2:>12.4f}")
    print(f"{'flux corr (inj)':22} {corr_x:>12.4f} {corr_s:>12.4f}")
    print(f"{'score mean':22} {cx_score.mean():>12.4f} {cs_score.mean():>12.4f}")
    print(f"{'score std':22} {cx_score.std():>12.4f} {cs_score.std():>12.4f}")
    print(f"{'subtract wall (s)':22} {t_x:>12.2f} {t_s:>12.2f}")
    print()
    print(f"theta relative diff (stamped vs exact): {theta_rel:.3e}")
    print(f"difference-image relative diff        : {diff_rel:.3e}")
    print(f"difference-image RMS exact/stamped     : {dx.std():.4f} / {ds.std():.4f}")
    print(f"speedup (exact/stamped wall)           : {t_x / t_s:.2f}x")


if __name__ == "__main__":
    main()
