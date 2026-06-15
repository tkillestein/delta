"""Measure-first probe for the per-field knot budget (handoff item #1).

The GLS M-build is O(N P^2) with P = (nc+1) k = 725: every one of the nc kernel
coefficient fields *and* the background get the full k = n_knots^2 thin-plate
basis. The hypothesis is that the high-order Gauss-Hermite fields and the
background don't need that many knots, so a per-field knot budget would shrink P
and the M-build (scales with P^2) for free.

That's a modeling change, so before building the (substantial) ragged-k solver we
measure the *premise*: under a genuinely spatially-varying PSF mismatch, how much
of its 25-knot spatial freedom does each component field actually use? A near-flat
high-order/background field is provably wasting its knots.

Run: uv run python benchmarks/knot_budget_probe.py
"""

import time

import delta
import numpy as np
from delta import _core, validation


def varying_seeing_pair(h=240, w=240, sigma_noise=3.0, seed=0, pattern="linear"):
    """Star field whose science-frame seeing varies across the frame.

    The reference has a constant PSF; the science PSF width varies with position,
    so the matching kernel is genuinely spatially varying and the low-order fields
    must use their spatial DOF. A constant-seeing pair makes *every* field flat and
    would rig the answer.

    `pattern` controls the spatial form of the mismatch:
      - "linear": seeing grows along the diagonal. The truth is affine, so even
        the low-order radial knots aren't needed -- a baseline that shows what an
        affine-sufficient field looks like.
      - "curved": a central seeing bump (radial Gaussian). Non-affine, so the
        low-order fields *must* engage their radial knots; the test is then
        whether the high-order/background fields still don't.
    """
    rng = np.random.default_rng(seed)
    ref = np.full((h, w), 120.0)
    sci = np.full((h, w), 120.0)
    sig_ref = 1.6
    cx, cy = w / 2.0, h / 2.0
    grid = [(x, y) for x in range(30, w, 30) for y in range(30, h, 30)]
    for x, y in grid:
        if pattern == "linear":
            sig_sci = 2.2 + 1.0 * (x + y) / (w + h)  # ~2.2 -> ~3.2 diagonally
        else:  # "curved": central bump
            r2 = (x - cx) ** 2 + (y - cy) ** 2
            sig_sci = 2.2 + 1.2 * np.exp(-r2 / (2.0 * (0.22 * w) ** 2))
        ref = ref + validation.gaussian_psf((h, w), x, y, 12000.0, sig_ref)
        sci = sci + validation.gaussian_psf((h, w), x, y, 12000.0, sig_sci)
    var = np.full((h, w), sigma_noise**2, np.float32)
    ref = (ref + rng.normal(0, sigma_noise, ref.shape)).astype(np.float32)
    sci = (sci + rng.normal(0, sigma_noise, sci.shape)).astype(np.float32)
    return ref, sci, var


def gh_orders(n_max):
    """(nx, ny) per component, in the C++ basis order: total then ny."""
    out = []
    for total in range(n_max + 1):
        for ny in range(total + 1):
            out.append((total - ny, ny))
    return out


def field_stats(knots, coeffs, points):
    """Spatial RMS variation and mean magnitude of a TPS field over `points`."""
    f = _core.tps_evaluate(knots, points, coeffs)
    return float(np.std(f)), float(np.mean(np.abs(f)))


def main():
    n_max = 6
    n_knots = 5
    for pattern in ("linear", "curved"):
        print("=" * 78)
        print(f"PATTERN: {pattern}")
        print("=" * 78)
        analyze(pattern, n_max, n_knots)
        print()


def analyze(pattern, n_max, n_knots):
    ref, sci, var = varying_seeing_pair(pattern=pattern)

    sub = delta.Subtractor(n_max=n_max, n_knots=n_knots, stamp_radius=12)
    sol = sub.fit(sci, ref, science_var=var, reference_var=var)
    theta = sol.theta
    knots = sol.knots
    k = sol.n_spatial
    nc = sol.n_components
    S = sol.component_sums
    orders = gh_orders(n_max)
    assert len(orders) == nc, (len(orders), nc)

    # Evaluate every field on a coarse frame lattice.
    h, w = sci.shape
    xs, ys = np.meshgrid(np.linspace(0, w - 1, 24), np.linspace(0, h - 1, 24))
    pts = np.column_stack([xs.ravel(), ys.ravel()]).astype(np.float64)

    print(f"n_max={n_max}  nc={nc}  k={k}  P=(nc+1)k={(nc + 1) * k}")
    print(f"selected lambda={sol.lam:.3g}  reduced_chi2={sol.reduced_chi2:.3f}\n")

    n_radial = k - 3  # penalised (knot-dependent) coeffs; last 3 are affine
    rows = []
    for n in range(nc):
        blk = theta[n * k : (n + 1) * k]
        radial_rms = float(np.sqrt(np.mean(blk[:n_radial] ** 2)))
        affine_rms = float(np.sqrt(np.mean(blk[n_radial:] ** 2)))
        std, mag = field_stats(knots, blk, pts)
        nx, ny = orders[n]
        # Weight the field's spatial variation by the component's footprint sum:
        # how much spatial structure this component injects into the actual model.
        rows.append(
            (nx + ny, n, nx, ny, std, mag, radial_rms, affine_rms, abs(S[n]), std * abs(S[n]))
        )

    # Background field (trailing block).
    bg = theta[nc * k : (nc + 1) * k]
    bg_std, bg_mag = field_stats(knots, bg, pts)
    bg_radial = float(np.sqrt(np.mean(bg[:n_radial] ** 2)))
    bg_affine = float(np.sqrt(np.mean(bg[n_radial:] ** 2)))

    print("per-component field usage (sorted by total order):")
    print(
        f"{'ord':>3} {'nx,ny':>6} {'fieldstd':>10} {'fieldmag':>10} "
        f"{'rad_rms':>9} {'aff_rms':>9} {'|S_n|':>9} {'std*|S|':>10}"
    )
    for tot, _n, nx, ny, std, mag, rr, ar, aS, w_std in sorted(rows):
        print(
            f"{tot:>3} {f'{nx},{ny}':>6} {std:>10.3g} {mag:>10.3g} "
            f"{rr:>9.3g} {ar:>9.3g} {aS:>9.3g} {w_std:>10.3g}"
        )
    print(f"{'bg':>3} {'-':>6} {bg_std:>10.3g} {bg_mag:>10.3g} {bg_radial:>9.3g} {bg_affine:>9.3g}")

    # Aggregate by order: fraction of total weighted spatial structure carried by
    # each order band. If high orders carry ~0, their knots are wasted.
    print("\nweighted spatial structure (std*|S_n|) by total order:")
    by_ord = {}
    for tot, _, _, _, _, _, _, _, _, w_std in rows:
        by_ord.setdefault(tot, 0.0)
        by_ord[tot] += w_std
    total_ws = sum(by_ord.values()) or 1.0
    cum = 0.0
    for tot in sorted(by_ord):
        cum += by_ord[tot]
        print(
            f"  order {tot}: {by_ord[tot]:>10.3g}  "
            f"({100 * by_ord[tot] / total_ws:5.1f}%)  cum {100 * cum / total_ws:5.1f}%"
        )

    if pattern != "linear":
        return
    # Secondary: solve cost vs global n_knots, to bound the payoff ceiling.
    print("\nfit wall-time vs global n_knots (P = (nc+1)*n_knots^2):")
    for nk in (3, 4, 5, 6):
        s = delta.Subtractor(n_max=n_max, n_knots=nk, stamp_radius=12)
        t0 = time.perf_counter()
        for _ in range(3):
            s.fit(sci, ref, science_var=var, reference_var=var)
        dt = (time.perf_counter() - t0) / 3
        print(f"  n_knots={nk}  P={(nc + 1) * nk * nk:>4}  {dt * 1e3:7.1f} ms")


if __name__ == "__main__":
    main()
