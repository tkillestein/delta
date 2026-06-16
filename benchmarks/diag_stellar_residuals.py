"""Diagnostic: compare delta vs HOTPANTS stellar residuals on a large frame.

Generates PNG plots:
  diag_residual_ratios.png    -- per-star |diff|/peak_amp scatter, both engines
  diag_chi2_map.png           -- per-stamp fit chi2 across the frame
  diag_kernel_corners.png     -- recovered kernel at 5 representative locations
  diag_score_hist.png         -- score histogram in a source-free region

Run:
    uv run python -m benchmarks.diag_stellar_residuals [SIZE]
SIZE defaults to 1536 (matches the integration test); use 512 for a quick run.
"""

from __future__ import annotations

import sys
import time

import matplotlib
import numpy as np

matplotlib.use("Agg")
from pathlib import Path

import delta
import matplotlib.pyplot as plt
from delta import validation

from benchmarks import compare_hotpants

OUT = Path("tests/artifacts")
OUT.mkdir(parents=True, exist_ok=True)

SIGMA_BORDER = 24


def make_frame(size, rng):
    gain, read_noise = 1.6, 5.0
    sig_ref, sig_sci = 1.7, 2.6

    shape = (size, size)
    background = 200.0
    n_stars = max(16, (size // 70) ** 2)
    positions, fluxes = validation.sample_starfield(
        shape,
        n_stars,
        rng,
        flux_range=(300.0, 80000.0),
        border=2 * SIGMA_BORDER,
        min_separation=24.0,
    )
    ref_signal = background + validation.render_stars(shape, positions, fluxes, sig_ref)
    sci_signal = background + validation.render_stars(shape, positions, fluxes, sig_sci)

    # Inject a bright transient.
    tx, ty = int(0.62 * size), int(0.37 * size)
    trans_flux = 30000.0
    sci_signal += validation.render_stars(shape, [(tx, ty)], [trans_flux], sig_sci)
    trans_peak = trans_flux / (2.0 * np.pi * sig_sci**2)

    def _noise(signal):
        var = np.maximum(signal, 0.0) / gain + read_noise**2
        noisy = signal + rng.normal(0.0, np.sqrt(var))
        return noisy.astype(np.float32), var.astype(np.float32)

    sci, sci_var = _noise(sci_signal)
    ref, ref_var = _noise(ref_signal)
    return sci, ref, sci_var, ref_var, positions, fluxes, sig_sci, trans_peak, (tx, ty)


def residual_ratios(diff, positions, fluxes, sig_sci, trans_peak, size):
    peak_amp = fluxes / (2.0 * np.pi * sig_sci**2)
    bright = peak_amp > 0.2 * trans_peak
    ratios, xs, ys = [], [], []
    for (x, y), amp in zip(positions[bright], peak_amp[bright], strict=False):
        xi, yi = int(x), int(y)
        in_border = (
            SIGMA_BORDER <= xi < size - SIGMA_BORDER and SIGMA_BORDER <= yi < size - SIGMA_BORDER
        )
        if in_border:
            resid, _ = validation.peak_near(np.abs(diff), (xi, yi), radius=4)
            ratios.append(resid / amp)
            xs.append(x)
            ys.append(y)
    return np.array(ratios), np.array(xs), np.array(ys)


def run(size: int) -> None:
    rng = np.random.default_rng(7)
    print(f"Building {size}×{size} frame...")
    sci, ref, sci_var, ref_var, positions, fluxes, sig_sci, trans_peak, (tx, ty) = make_frame(
        size, rng
    )

    # --- delta ---
    print("Running delta...")
    t0 = time.perf_counter()
    res = delta.subtract(
        sci,
        ref,
        science_var=sci_var,
        reference_var=ref_var,
        n_knots=5,
        stamp_radius=15,
        decorrelate=True,
        score=True,
        block=256,
    )
    dt_delta = time.perf_counter() - t0
    sol = res.solution
    n_used = sol.n_stamps_used
    n_rej = sol.n_stamps_rejected
    print(
        f"  delta: {dt_delta:.1f}s  reduced_chi2={sol.reduced_chi2:.3f}"
        f"  lambda={sol.lam:.3g}  stamps used={n_used}/{n_used + n_rej}"
    )

    # --- HOTPANTS ---
    print("Running HOTPANTS...")
    t0 = time.perf_counter()
    hp_diff = compare_hotpants.run_hotpants(sci, ref)
    dt_hp = time.perf_counter() - t0
    print(f"  hotpants: {dt_hp:.1f}s")

    delta_ratios, star_x, star_y = residual_ratios(
        res.difference, positions, fluxes, sig_sci, trans_peak, size
    )
    hp_ratios, _, _ = residual_ratios(hp_diff, positions, fluxes, sig_sci, trans_peak, size)

    print(
        f"\nDelta   median ratio: {np.median(delta_ratios):.3f}"
        f"  p90: {np.percentile(delta_ratios, 90):.3f}"
    )
    print(
        f"HOTPANTS median ratio: {np.median(hp_ratios):.3f}"
        f"  p90: {np.percentile(hp_ratios, 90):.3f}"
    )

    # ------------------------------------------------------------------ #
    # Plot 1: per-star residual ratios scatter
    # ------------------------------------------------------------------ #
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, ratios, label in [
        (axes[0], delta_ratios, "delta"),
        (axes[1], hp_ratios, "HOTPANTS"),
    ]:
        sc = ax.scatter(
            star_x,
            star_y,
            c=ratios,
            cmap="hot_r",
            vmin=0,
            vmax=0.5,
            s=20,
            edgecolors="none",
        )
        plt.colorbar(sc, ax=ax, label="|resid|/peak_amp")
        ax.set_xlim(0, size)
        ax.set_ylim(0, size)
        ax.set_aspect("equal")
        ax.set_title(
            f"{label}  median={np.median(ratios):.3f}  p90={np.percentile(ratios, 90):.3f}"
        )
        ax.set_xlabel("x [px]")
        ax.set_ylabel("y [px]")
    fig.suptitle(f"Per-star |residual|/peak_amp  ({size}×{size} frame)", fontsize=13)
    fig.tight_layout()
    out = OUT / "diag_residual_ratios.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"Saved {out}")

    # ------------------------------------------------------------------ #
    # Plot 2: per-stamp chi2 map
    # ------------------------------------------------------------------ #
    sx = np.asarray(sol.stamp_x, dtype=float)
    sy = np.asarray(sol.stamp_y, dtype=float)
    chi2 = np.asarray(sol.stamp_chi2)
    accepted = np.asarray(sol.stamp_accepted, dtype=bool)

    fig, ax = plt.subplots(figsize=(7, 7))
    sc = ax.scatter(
        sx[accepted],
        sy[accepted],
        c=np.clip(chi2[accepted], 0, 10),
        cmap="YlOrRd",
        vmin=0,
        vmax=10,
        s=25,
        label="accepted",
    )
    if (~accepted).any():
        ax.scatter(sx[~accepted], sy[~accepted], marker="x", c="navy", s=40, label="rejected")
    plt.colorbar(sc, ax=ax, label="per-stamp reduced chi²")
    ax.set_xlim(0, size)
    ax.set_ylim(0, size)
    ax.set_aspect("equal")
    ax.set_title(f"Per-stamp chi²  (global reduced chi²={sol.reduced_chi2:.3f})")
    ax.set_xlabel("x [px]")
    ax.set_ylabel("y [px]")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    out = OUT / "diag_chi2_map.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"Saved {out}")

    # ------------------------------------------------------------------ #
    # Plot 3: kernel at 5 locations across the frame
    # ------------------------------------------------------------------ #
    from delta import _core

    locs = [
        (size // 4, size // 4),
        (3 * size // 4, size // 4),
        (size // 2, size // 2),
        (size // 4, 3 * size // 4),
        (3 * size // 4, 3 * size // 4),
    ]
    labels = ["BL", "BR", "C", "TL", "TR"]

    # K(u,v; x,y) = Σ_n a_n(x,y) · φ_n(u,v)
    # a_n = tps_design(knots, point) @ C[:, n], C = theta.reshape(k, nc+1)[:, :nc]
    _orders, _phi = _core.gauss_hermite_kernels(sol.beta, sol.n_max, sol.radius)
    _phi = np.asarray(_phi)  # (nc, ks, ks)
    nc_k = _phi.shape[0]
    k_sp = sol.n_spatial  # number of TPS basis functions
    # Eigen Map is column-major: theta is stored as (k, nc+1) in Fortran order.
    C = np.asarray(sol.theta).reshape(k_sp, nc_k + 1, order="F")  # k x (nc+1)

    kernels = []
    for kx, ky in locs:
        pt = np.asfortranarray([[float(kx), float(ky)]])
        design = np.asarray(_core.tps_design(sol.knots, pt))  # (1, k)
        a_n = (design @ C[:, :nc_k]).ravel()  # (nc,)
        k2d = np.einsum("n,nij->ij", a_n, _phi)
        kernels.append(k2d)

    fig, axes = plt.subplots(1, 5, figsize=(14, 3.5))
    vmax = max(np.abs(k).max() for k in kernels)
    for ax, k, lbl in zip(axes, kernels, labels, strict=False):
        im = ax.imshow(k, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(lbl)
        ax.axis("off")
    fig.colorbar(im, ax=axes[-1], fraction=0.046, pad=0.04)
    fig.suptitle(f"Kernel at 5 frame locations  (β={sol.beta:.2f}, n_max={sol.n_max})", fontsize=12)
    fig.tight_layout()
    out = OUT / "diag_kernel_corners.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"Saved {out}")

    # ------------------------------------------------------------------ #
    # Plot 4: score histogram in a source-free corner
    # ------------------------------------------------------------------ #
    if res.score is not None:
        corner = res.score[
            SIGMA_BORDER : SIGMA_BORDER + 200,
            size - 200 - SIGMA_BORDER : size - SIGMA_BORDER,
        ]
        flat = corner.ravel()
        fig, ax = plt.subplots(figsize=(6, 4))
        bins = np.linspace(-5, 5, 80)
        ax.hist(flat, bins=bins.tolist(), density=True, alpha=0.7, label="score (source-free)")
        xs = np.linspace(-5, 5, 200)
        gauss = np.exp(-0.5 * xs**2) / np.sqrt(2.0 * np.pi)
        ax.plot(xs, gauss, "r--", lw=1.5, label="N(0,1)")
        ax.set_xlabel("Score σ")
        ax.set_ylabel("Density")
        ax.set_title(f"Score distribution  μ={flat.mean():.3f}  σ={flat.std():.3f}")
        ax.legend()
        fig.tight_layout()
        out = OUT / "diag_score_hist.png"
        fig.savefig(out, dpi=130)
        plt.close(fig)
        print(f"Saved {out}")

    # ------------------------------------------------------------------ #
    # Plot 5: side-by-side difference cutouts around the worst stars
    # ------------------------------------------------------------------ #
    worst_idx = np.argsort(delta_ratios)[-6:]
    fig, axes = plt.subplots(len(worst_idx), 3, figsize=(9, 2.5 * len(worst_idx)))
    r_cut = 20
    for row, idx in enumerate(worst_idx):
        xi, yi = int(star_x[idx]), int(star_y[idx])
        nearest_stamp = int(np.argmin((sx - xi) ** 2 + (sy - yi) ** 2))
        panels = [
            (sci, "science"),
            (res.difference, f"Δ delta  χ²={chi2[nearest_stamp]:.2f}"),
            (hp_diff, "Δ HOTPANTS"),
        ]
        for col, (img, lbl) in enumerate(panels):
            ax = axes[row, col]
            cut = img[
                max(0, yi - r_cut) : yi + r_cut + 1,
                max(0, xi - r_cut) : xi + r_cut + 1,
            ]
            v = np.percentile(np.abs(cut), 98)
            ax.imshow(cut, origin="lower", cmap="RdBu_r", vmin=-v, vmax=v)
            ax.set_title(lbl if row == 0 else "", fontsize=8)
            ax.axis("off")
        axes[row, 0].set_ylabel(f"#{idx}", fontsize=8)
    fig.suptitle("Worst-residual stars (delta): science | Δdelta | ΔHP", fontsize=11)
    fig.tight_layout()
    out = OUT / "diag_worst_stars.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"Saved {out}")

    print("\nAll diagnostics written to tests/artifacts/")


if __name__ == "__main__":
    size = int(sys.argv[1]) if len(sys.argv) > 1 else 1536
    run(size)
