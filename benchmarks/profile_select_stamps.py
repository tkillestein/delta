"""Profile stamp selection (handoff item #3) at reference frame scale.

select_stamps is reported at ~12% of the pipeline and is wholly serial (no OpenMP
in detect.cpp). This isolates its cost and splits detect-vs-match to find where the
time goes before deciding whether it's worth optimizing.

Run: uv run python benchmarks/profile_select_stamps.py
"""

import time

from delta import _core

from benchmarks.bench_subtract import build_pair


def t(fn, reps=5):
    fn()  # warm
    best = min(_time(fn) for _ in range(reps))
    return best


def _time(fn):
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def main():
    for w, h in ((4096, 4096), (8000, 6000)):
        n_stars = max(50, w * h // 40000)
        sci, ref, var = build_pair(w, h, n_stars)
        mpix = w * h / 1e6
        print(f"\n=== {w}x{h} ({mpix:.0f} Mpix), {n_stars} stars ===")

        # Full selection (detect both frames + match), as the pipeline calls it.
        sel_ms = (
            t(
                lambda s=sci, r=ref: _core.select_stamps(
                    s,
                    r,
                    stamp_radius=12,
                    threshold_sigma=5.0,
                    max_stamps=200,
                )
            )
            * 1e3
        )

        # Single-frame detection in isolation (half the selection's detect work).
        det_ms = (
            t(
                lambda s=sci: _core.detect_stamps(
                    s, stamp_radius=12, threshold_sigma=5.0, max_stamps=200
                )
            )
            * 1e3
        )

        out = _core.select_stamps(sci, ref, stamp_radius=12, threshold_sigma=5.0, max_stamps=200)
        n_sel = len(out["x"])
        print(f"  select_stamps (detect x2 + match): {sel_ms:8.1f} ms")
        print(f"  detect_stamps (one frame)        : {det_ms:8.1f} ms")
        print(
            f"  -> ~2x detect = {2 * det_ms:.1f} ms; match/overhead ~ {sel_ms - 2 * det_ms:.1f} ms"
        )
        print(f"  stamps selected: {n_sel}")
        print(f"  per-pixel detect throughput: {mpix / (det_ms / 1e3):.0f} Mpix/s")


if __name__ == "__main__":
    main()
