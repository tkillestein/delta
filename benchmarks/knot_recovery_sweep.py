"""Injection-recovery vs global n_knots (handoff item #1, follow-up).

The knot-budget probe showed the real flop lever is the *global* knot count, not a
per-field budget. This sweep answers the operative question: does recovery hold as
n_knots drops? It runs the full pipeline (decorrelate + score) under a genuinely
spatially-varying (curved) PSF mismatch -- the regime where knots actually matter --
injects transients across a flux ladder, and reports the same metrics the
injection-recovery test asserts on, per n_knots, alongside fit wall-time.

Run: uv run python benchmarks/knot_recovery_sweep.py
"""

import time

import delta
import numpy as np
from delta import validation


def curved_pair_with_injections(positions, fluxes, h=240, w=240, sigma_noise=3.0, seed=0):
    """Curved-seeing static field + injected transients, then matched noise.

    Mirrors tests/test_injection_recovery, but the science PSF width has a central
    bump (non-affine) so the matching kernel genuinely varies across the frame.
    Returns (ref, sci, var, sig_at_center) with noise added after injection.
    """
    rng = np.random.default_rng(seed)
    ref = np.full((h, w), 120.0)
    sci = np.full((h, w), 120.0)
    sig_ref = 1.6
    cx, cy = w / 2.0, h / 2.0

    def sig_sci_at(x, y):
        r2 = (x - cx) ** 2 + (y - cy) ** 2
        return 2.2 + 1.2 * np.exp(-r2 / (2.0 * (0.22 * w) ** 2))

    grid = [(x, y) for x in range(30, w, 30) for y in range(30, h, 30)]
    for x, y in grid:
        ref = ref + validation.gaussian_psf((h, w), x, y, 12000.0, sig_ref)
        sci = sci + validation.gaussian_psf((h, w), x, y, 12000.0, sig_sci_at(x, y))

    # Inject transients into the science frame at the local science seeing.
    for (x, y), flux in zip(positions, fluxes, strict=False):
        sci = sci + validation.gaussian_psf((h, w), x, y, flux, sig_sci_at(x, y))

    var = np.full((h, w), sigma_noise**2, np.float32)
    ref = (ref + rng.normal(0, sigma_noise, ref.shape)).astype(np.float32)
    sci = (sci + rng.normal(0, sigma_noise, sci.shape)).astype(np.float32)
    return ref, sci, var, sig_sci_at(cx, cy)


def main():
    positions = [(x, y) for x in (70, 110, 150) for y in (70, 140, 200)]
    fluxes = np.sort(np.logspace(np.log10(50.0), np.log10(40000.0), len(positions)))
    third = len(positions) // 3
    seeds = (0, 1, 2, 3, 4)
    nc = 28  # n_max=6 default

    print(
        f"curved seeing, {len(positions)} injections, flux "
        f"{fluxes.min():.0f}..{fluxes.max():.0f}, averaged over {len(seeds)} seeds\n"
    )
    hdr = (
        f"{'n_knots':>7} {'P':>5} {'chi2':>13} {'bright':>7} {'faint':>6} "
        f"{'flux_corr':>10} {'astrom':>7} {'score_std':>13} {'fit_ms':>8}"
    )
    print(hdr)
    print("-" * len(hdr))

    for nk in (3, 4, 5, 6):
        chi2, sstd, smean, bright, faint, fcorr, astrom, ms = ([] for _ in range(8))
        for seed in seeds:
            ref, sci, var, sig_c = curved_pair_with_injections(positions, fluxes, seed=seed)
            ap_r = int(round(3 * sig_c))
            t0 = time.perf_counter()
            res = delta.subtract(
                sci,
                ref,
                science_var=var,
                reference_var=var,
                n_knots=nk,
                stamp_radius=12,
                decorrelate=True,
                score=True,
                block=128,
            )
            ms.append((time.perf_counter() - t0) * 1e3)
            assert res.score is not None
            rec = validation.completeness(res.score, positions, threshold=5.0, radius=4)
            bright.append(rec[-third:].mean())
            faint.append(int(rec[:third].sum()))
            measured = np.array(
                [validation.aperture_flux(res.difference, xy, ap_r) for xy in positions]
            )
            fcorr.append(np.corrcoef(fluxes, measured)[0, 1])
            _, found = validation.peak_near(res.difference, positions[-1], radius=5)
            astrom.append(max(abs(found[0] - positions[-1][0]), abs(found[1] - positions[-1][1])))
            corner = res.score[10:70, 10:70]
            sstd.append(corner.std())
            smean.append(corner.mean())
            chi2.append(res.solution.reduced_chi2)

        print(
            f"{nk:>7} {(nc + 1) * nk * nk:>5} "
            f"{np.mean(chi2):>6.3f}+-{np.std(chi2):<5.3f} "
            f"{np.mean(bright):>7.2f} {max(faint):>6d} "
            f"{np.mean(fcorr):>10.3f} {max(astrom):>7d} "
            f"{np.mean(sstd):>6.3f}+-{np.std(sstd):<5.3f} {np.mean(ms):>8.0f}"
        )

    print(
        "\nmetrics (test thresholds): bright=1.0, faint<=1, flux_corr>0.97, "
        "astrom<=2, score_std in 0.6..1.5 (ideal 1.0)"
    )


if __name__ == "__main__":
    main()
