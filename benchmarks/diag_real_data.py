"""Diagnostic: delta vs HOTPANTS on t2_s115088_ut2.fits (real survey data).

Uses the SCIENCE_PHOTOMETRY catalog from the MEF for stellar positions and the
DIFFERENCE extension as the HOTPANTS reference. Operates on the same central
2048×2048 window as plot_vs_hotpants.py / core_resid.py.

Outputs (to tests/artifacts/):
  diag_real_residual_ratios.png  -- per-star core |resid| scatter, delta vs HP
  diag_real_chi2_map.png         -- per-stamp fit chi² across the frame
  diag_real_kernel_corners.png   -- recovered kernel at 5 frame locations
  diag_real_pull_dist.png        -- pull (diff/noise) histogram vs N(0,1)
  diag_real_worst_stars.png      -- cutouts of worst-residual stars

Run:
    uv run python -m benchmarks.diag_real_data
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import delta
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from delta import _core

ART = Path(__file__).resolve().parent.parent / "artifacts"
FITS_PATH = ART / "t2_s115088_ut2.fits"
OUT = Path(__file__).resolve().parent.parent / "tests" / "artifacts"
OUT.mkdir(parents=True, exist_ok=True)

# The 2048×2048 analysis window used in core_resid.py / plot_vs_hotpants.py.
WIN = (slice(2200, 4248), slice(3200, 5248))
Y0, X0 = WIN[0].start, WIN[1].start


# ── data loading ─────────────────────────────────────────────────────────────


def _total_error(data, bkg_rms, gain):
    bkg = np.nan_to_num(bkg_rms.astype(np.float64), nan=0.0)
    src = np.clip(np.nan_to_num(data, nan=0.0), 0.0, None) / gain
    return np.sqrt(bkg**2 + src)


def load_plane(h, image, mask, rms):
    d = np.asarray(h[image].data, np.float32)
    r = np.asarray(h[rms].data, np.float32)
    m = np.asarray(h[mask].data)
    et = float(h[image].header["EXPTIME"])
    var = _total_error(d, r, et).astype(np.float32) ** 2
    bad = (m != 0) | ~np.isfinite(d) | ~np.isfinite(r) | (r <= 0)
    d = np.where(np.isfinite(d), d, 0.0).astype(np.float32)
    var = np.where(np.isfinite(var) & (var > 0), var, 1e30).astype(np.float32)
    return d, var, bad.astype(np.uint8)


def cut(a):
    ny, nx = a.shape[0], a.shape[1]
    return np.ascontiguousarray(a[:ny, :nx][WIN])


# ── stellar residuals ─────────────────────────────────────────────────────────


def core_residuals(diff, lx, ly, rad=2):
    """Per-star median |diff| in a (2rad+1)^2 box; returns array over stars."""
    vals = np.empty(len(lx))
    for i, (x, y) in enumerate(zip(lx, ly, strict=False)):
        box = np.abs(diff[y - rad : y + rad + 1, x - rad : x + rad + 1])
        vals[i] = float(np.nanmedian(box))
    return vals


# ── kernel evaluation ─────────────────────────────────────────────────────────


def eval_kernel_at(sol, x, y):
    """Recover the 2-D matching kernel at frame position (x, y)."""
    _, _phi = _core.gauss_hermite_kernels(sol.beta, sol.n_max, sol.radius)
    _phi = np.asarray(_phi)  # (nc, ks, ks)
    nc_k = _phi.shape[0]
    k_sp = sol.n_spatial
    # theta is stored as column-major (Eigen Map) -> Fortran reshape
    C = np.asarray(sol.theta).reshape(k_sp, nc_k + 1, order="F")
    pt = np.asfortranarray([[float(x), float(y)]])
    design = np.asarray(_core.tps_design(sol.knots, pt))  # (1, k_sp)
    a_n = (design @ C[:, :nc_k]).ravel()  # (nc,)
    return np.einsum("n,nij->ij", a_n, _phi)


# ── main ──────────────────────────────────────────────────────────────────────


def main():
    print(f"Loading {FITS_PATH.name} ({FITS_PATH.stat().st_size / 1e6:.0f} MB)…")
    with fits.open(FITS_PATH, memmap=True) as h:
        sci, svar, sbad = load_plane(h, "SCIENCE", "SCIENCE_MASK", "SCIENCE_BACKGROUND_RMS")
        tmpl, tvar, tbad = load_plane(h, "TEMPLATE", "TEMPLATE_MASK", "TEMPLATE_BACKGROUND_RMS")
        hp_full = np.asarray(h["DIFFERENCE"].data, np.float64)
        hp_rms_full = np.asarray(h["DIFFERENCE_BACKGROUND_RMS"].data, np.float64)
        phot = h["SCIENCE_PHOTOMETRY"].data
        xpk = np.asarray(phot["xpeak"])
        ypk = np.asarray(phot["ypeak"])
        peak = np.asarray(phot["peak"])

    sci, svar, sbad = cut(sci), cut(svar), cut(sbad)
    tmpl, tvar, tbad = cut(tmpl), cut(tvar), cut(tbad)
    hp = cut(hp_full).astype(np.float64)
    hprms = cut(hp_rms_full).astype(np.float64)
    H, W = sci.shape
    print(f"Working frame: {W}×{H} px")

    # Stars from the catalog that fall inside the window and pass brightness/
    # saturation cuts (mirroring core_resid.py).
    border = 10
    sel = (
        (peak > 40)
        & (peak < 520)
        & (xpk >= X0 + border)
        & (xpk < X0 + W - border)
        & (ypk >= Y0 + border)
        & (ypk < Y0 + H - border)
    )
    lx = (xpk[sel] - X0).astype(int)
    ly = (ypk[sel] - Y0).astype(int)
    pk = peak[sel]
    order = np.argsort(pk)[::-1][:250]
    lx, ly, pk = lx[order], ly[order], pk[order]
    print(f"{len(lx)} catalog stars (peak 40–520) in window")

    # ── run delta ─────────────────────────────────────────────────────────────
    print("Running delta (n_max=8, n_knots=4, saturation=552)…")
    res = delta.Subtractor(
        n_max=8,
        n_knots=4,
        stamp_radius=15,
        threshold_sigma=8.0,
        max_stamps=300,
        saturation=552.0,
        decorrelate=True,
        score=True,
    ).subtract(
        sci,
        tmpl,
        science_var=svar,
        reference_var=tvar,
        science_mask=sbad,
        reference_mask=tbad,
    )

    sol = res.solution
    dd = np.where(res.mask == 0, res.difference, np.nan)
    print(
        f"  beta={sol.beta:.3f}  n_max={sol.n_max}  lambda={sol.lam:.3g}  "
        f"chi2={sol.reduced_chi2:.3f}  "
        f"stamps={sol.n_stamps_used}/{sol.n_stamps_used + sol.n_stamps_rejected}"
    )

    # ── residual arrays ───────────────────────────────────────────────────────
    hp_resid = core_residuals(hp, lx, ly)
    dlt_resid = core_residuals(dd, lx, ly)
    print(
        f"HOTPANTS  core|resid| median={np.nanmedian(hp_resid):.4f}  "
        f"p90={np.nanpercentile(hp_resid, 90):.4f}"
    )
    print(
        f"delta     core|resid| median={np.nanmedian(dlt_resid):.4f}  "
        f"p90={np.nanpercentile(dlt_resid, 90):.4f}"
    )

    # ── Plot 1: per-star residual scatter ─────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    vmax = float(np.nanpercentile(np.concatenate([hp_resid, dlt_resid]), 97))
    for ax, vals, label in [
        (axes[0], hp_resid, "HOTPANTS (pipeline)"),
        (axes[1], dlt_resid, "delta"),
    ]:
        finite = np.isfinite(vals)
        sc = ax.scatter(
            lx[finite],
            ly[finite],
            c=vals[finite],
            cmap="hot_r",
            vmin=0,
            vmax=vmax,
            s=22,
            edgecolors="none",
        )
        plt.colorbar(sc, ax=ax, label="core |resid| [ADU]")
        ax.set_xlim(0, W)
        ax.set_ylim(0, H)
        ax.set_aspect("equal")
        ax.set_title(
            f"{label}\nmedian={np.nanmedian(vals):.4f}  p90={np.nanpercentile(vals, 90):.4f}"
        )
        ax.set_xlabel("x [px]")
        ax.set_ylabel("y [px]")
    fig.suptitle("Per-star core |residual|  —  t2_s115088_ut2  (2048×2048 window)", fontsize=12)
    fig.tight_layout()
    out = OUT / "diag_real_residual_ratios.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"Saved {out.name}")

    # ── Plot 2: per-stamp chi2 map ────────────────────────────────────────────
    sx = np.asarray(sol.stamp_x, float)
    sy = np.asarray(sol.stamp_y, float)
    chi2 = np.asarray(sol.stamp_chi2)
    accepted = np.asarray(sol.stamp_accepted, bool)

    fig, ax = plt.subplots(figsize=(7, 7))
    sc = ax.scatter(
        sx[accepted],
        sy[accepted],
        c=np.clip(chi2[accepted], 0, 15),
        cmap="YlOrRd",
        vmin=0,
        vmax=15,
        s=30,
        label="accepted",
    )
    if (~accepted).any():
        ax.scatter(sx[~accepted], sy[~accepted], marker="x", c="navy", s=45, label="rejected")
    plt.colorbar(sc, ax=ax, label="per-stamp reduced chi²")
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.set_title(
        f"Per-stamp chi²  (global chi²={sol.reduced_chi2:.3f},  "
        f"λ={sol.lam:.3g},  eff.dof={sol.effective_dof:.1f})"
    )
    ax.set_xlabel("x [px]")
    ax.set_ylabel("y [px]")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    out = OUT / "diag_real_chi2_map.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"Saved {out.name}")

    # ── Plot 3: kernel at 5 locations ─────────────────────────────────────────
    locs = [
        (W // 4, H // 4),
        (3 * W // 4, H // 4),
        (W // 2, H // 2),
        (W // 4, 3 * H // 4),
        (3 * W // 4, 3 * H // 4),
    ]
    klabels = ["BL", "BR", "C", "TL", "TR"]
    kernels = [eval_kernel_at(sol, kx, ky) for kx, ky in locs]

    vmax_k = max(np.abs(k).max() for k in kernels)
    fig, axes = plt.subplots(1, 5, figsize=(14, 3.5))
    for ax, k, lbl in zip(axes, kernels, klabels, strict=False):
        im = ax.imshow(k, origin="lower", cmap="RdBu_r", vmin=-vmax_k, vmax=vmax_k)
        ax.set_title(lbl)
        ax.axis("off")
    fig.colorbar(im, ax=axes[-1], fraction=0.046, pad=0.04)
    fig.suptitle(
        f"Kernel at 5 locations  (β={sol.beta:.2f}, n_max={sol.n_max}, "
        f"r={sol.radius}px, dir={sol.direction})",
        fontsize=11,
    )
    fig.tight_layout()
    out = OUT / "diag_real_kernel_corners.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"Saved {out.name}")

    # ── Plot 4: pull distributions ────────────────────────────────────────────
    assert res.variance is not None
    dpull = np.where(
        (res.mask == 0) & (res.variance > 0),
        res.difference / np.sqrt(np.maximum(res.variance, 1e-12)),
        np.nan,
    )
    hppull = np.where(hprms > 0, hp / np.maximum(hprms, 1e-12), np.nan)

    bins = np.linspace(-8, 8, 160)
    gauss = np.exp(-0.5 * bins**2) / math.sqrt(2 * math.pi)
    fig, ax = plt.subplots(figsize=(7, 5))
    for arr, lbl, col in [(hppull, "HOTPANTS", "C1"), (dpull, "delta", "C0")]:
        v = arr[np.isfinite(arr)]
        sig = float(np.std(v))
        med = float(np.median(v))
        ax.hist(
            v,
            bins=bins.tolist(),
            density=True,
            histtype="step",
            lw=1.5,
            color=col,
            label=f"{lbl}  μ={med:.3f}  σ={sig:.3f}",
        )
    ax.plot(bins, gauss, "k--", lw=1, label="N(0,1)")
    ax.set_yscale("log")
    ax.set_ylim(1e-5, 1)
    ax.set_xlabel("pull (diff / noise σ)")
    ax.set_ylabel("density")
    ax.set_title("Pull distributions  —  full 2048×2048 window")
    ax.legend(fontsize=9)
    fig.tight_layout()
    out = OUT / "diag_real_pull_dist.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"Saved {out.name}")

    # ── Plot 5: worst-residual star cutouts ───────────────────────────────────
    ratio = dlt_resid / np.maximum(hp_resid, 1e-9)
    finite = np.isfinite(ratio)
    worst_idx = np.argsort(ratio[finite])[-6:]
    orig_idx = np.where(finite)[0][worst_idx]

    r_cut = 20
    fig, axes = plt.subplots(len(orig_idx), 3, figsize=(9, 2.6 * len(orig_idx)))
    for row, idx in enumerate(orig_idx):
        xi, yi = int(lx[idx]), int(ly[idx])
        nearest_stamp = int(np.argmin((sx - xi) ** 2 + (sy - yi) ** 2))
        stamp_chi2_val = chi2[nearest_stamp] if accepted[nearest_stamp] else float("nan")
        for col, (img, lbl) in enumerate(
            [
                (sci.astype(np.float64), "science"),
                (dd, f"Δ delta  χ²~{stamp_chi2_val:.1f}"),
                (hp, "Δ HOTPANTS"),
            ]
        ):
            ax = axes[row, col]
            patch = img[
                max(0, yi - r_cut) : yi + r_cut + 1,
                max(0, xi - r_cut) : xi + r_cut + 1,
            ]
            v = np.nanpercentile(np.abs(patch), 98) or 1.0
            if col == 0:
                lo, hi = np.nanpercentile(patch, [5, 99.5])
                ax.imshow(patch, origin="lower", cmap="gray", vmin=lo, vmax=hi)
            else:
                ax.imshow(patch, origin="lower", cmap="RdBu_r", vmin=-v, vmax=v)
            ax.set_title(lbl if row == 0 else "", fontsize=8)
            ax.axis("off")
        axes[row, 0].set_ylabel(f"#{idx}  pk={pk[idx]:.0f}\nΔ/HP={ratio[idx]:.2f}", fontsize=7)
    fig.suptitle(
        "Stars where delta residual / HOTPANTS residual is largest\nscience | Δdelta | Δhotpants",
        fontsize=10,
    )
    fig.tight_layout()
    out = OUT / "diag_real_worst_stars.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"Saved {out.name}")

    # ── Sweep: basis parameters (beta × n_max) ────────────────────────────────
    print("\nRunning basis parameter sweep (beta × n_max)…")

    def _run_basis(beta, n_max):
        r = delta.Subtractor(
            beta=beta,
            n_max=n_max,
            n_knots=4,
            stamp_radius=15,
            threshold_sigma=8.0,
            max_stamps=300,
            saturation=552.0,
            lambda_grid=np.logspace(3, 12, 28, dtype=np.float64),
            decorrelate=False,
            score=False,
        ).subtract(
            sci,
            tmpl,
            science_var=svar,
            reference_var=tvar,
            science_mask=sbad,
            reference_mask=tbad,
        )
        diff_w = np.where(r.mask == 0, r.difference, np.nan)
        resids = core_residuals(diff_w, lx, ly)
        s = r.solution
        return {
            "beta": s.beta,
            "n_max": n_max,
            "lam": s.lam,
            "chi2": s.reduced_chi2,
            "median": float(np.nanmedian(resids)),
            "p90": float(np.nanpercentile(resids, 90)),
            "radius": s.radius,
        }

    betas = [None, 0.8, 1.0, 1.263, 1.5, 2.0, 2.5]  # None = auto
    n_maxes = [4, 6, 8, 10]
    basis_rows = []
    for nm in n_maxes:
        for b in betas:
            r = _run_basis(b, nm)
            basis_rows.append(r)
            print(
                f"  beta={r['beta']:.3f}  n_max={nm:2d}  r={r['radius']:2d}px  "
                f"λ={r['lam']:.2g}  chi2={r['chi2']:.3f}  "
                f"median={r['median']:.4f}  p90={r['p90']:.4f}"
            )

    # ── Plot: basis sweep heatmaps ────────────────────────────────────────────
    unique_betas = sorted({r["beta"] for r in basis_rows})
    unique_nmaxes = sorted({r["n_max"] for r in basis_rows})
    nb_, nm_ = len(unique_betas), len(unique_nmaxes)

    def _grid(key):
        g = np.full((nm_, nb_), np.nan)
        for r in basis_rows:
            i = unique_nmaxes.index(r["n_max"])
            j = unique_betas.index(r["beta"])
            g[i, j] = r[key]
        return g

    med_grid = _grid("median")
    p90_grid = _grid("p90")
    chi_grid = _grid("chi2")

    beta_labels = [f"{b:.2f}" for b in unique_betas]
    nmax_labels = [str(n) for n in unique_nmaxes]

    hp_med = np.nanmedian(hp_resid)
    hp_p90 = np.nanpercentile(hp_resid, 90)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, grid, title, fmt in [
        (axes[0], med_grid, f"median core |resid|  (HP={hp_med:.3f})", ".3f"),
        (axes[1], p90_grid, f"p90 core |resid|  (HP={hp_p90:.3f})", ".3f"),
        (axes[2], chi_grid, "reduced chi²", ".3f"),
    ]:
        im = ax.imshow(
            grid,
            aspect="auto",
            origin="lower",
            cmap="RdYlGn_r" if "chi" not in title else "RdYlGn",
        )
        plt.colorbar(im, ax=ax)
        ax.set_xticks(range(nb_))
        ax.set_xticklabels(beta_labels, fontsize=8, rotation=45)
        ax.set_yticks(range(nm_))
        ax.set_yticklabels(nmax_labels, fontsize=8)
        ax.set_xlabel("beta")
        ax.set_ylabel("n_max")
        ax.set_title(title)
        for i in range(nm_):
            for j in range(nb_):
                v = grid[i, j]
                if np.isfinite(v):
                    ax.text(
                        j,
                        i,
                        f"{v:{fmt}}",
                        ha="center",
                        va="center",
                        fontsize=6.5,
                        color="black",
                    )
    fig.suptitle("Basis sweep: beta × n_max  —  t2_s115088_ut2", fontsize=12)
    fig.tight_layout()
    out = OUT / "diag_real_basis_sweep.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"Saved {out.name}")

    # ── Sweep: lambda grid ceiling × n_knots ─────────────────────────────────
    print("\nRunning lambda-ceiling × n_knots sweep…")

    def _run_knots(n_knots, lam_lo, lam_hi, n_pts):
        lgrid = np.logspace(lam_lo, lam_hi, n_pts)
        r = delta.Subtractor(
            n_max=8,
            n_knots=n_knots,
            stamp_radius=15,
            threshold_sigma=8.0,
            max_stamps=300,
            saturation=552.0,
            lambda_grid=lgrid,
            decorrelate=False,
            score=False,
        ).subtract(
            sci,
            tmpl,
            science_var=svar,
            reference_var=tvar,
            science_mask=sbad,
            reference_mask=tbad,
        )
        diff_w = np.where(r.mask == 0, r.difference, np.nan)
        resids = core_residuals(diff_w, lx, ly)
        s = r.solution
        return {
            "lam": s.lam,
            "edof": s.effective_dof,
            "chi2": s.reduced_chi2,
            "median": float(np.nanmedian(resids)),
            "p90": float(np.nanpercentile(resids, 90)),
            "n_knots": n_knots,
            "lam_hi": lam_hi,
        }

    sweep_configs = [
        (4, 6, 12, 28),
        (4, 3, 12, 28),
        (6, 3, 12, 28),
        (8, 3, 12, 28),
        (10, 3, 12, 28),
    ]
    rows = []
    for n_knots, lo, hi, n_pts in sweep_configs:
        r = _run_knots(n_knots, lo, hi, n_pts)
        rows.append(r)
        print(
            f"  n_knots={n_knots:2d}  λ=[1e{lo},1e{hi}]  "
            f"→ λ_sel={r['lam']:.3g}  edof={r['edof']:.1f}  "
            f"chi2={r['chi2']:.3f}  median={r['median']:.4f}  p90={r['p90']:.4f}"
        )

    # ── Plot 6: sweep summary ─────────────────────────────────────────────────
    labels = [f"nk={r['n_knots']}" for r in rows]
    medians = [r["median"] for r in rows]
    p90s = [r["p90"] for r in rows]
    lam_sels = [r["lam"] for r in rows]
    edofs = [r["edof"] for r in rows]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    x = np.arange(len(rows))

    ax = axes[0, 0]
    ax.bar(x, medians, color="C0", alpha=0.8)
    ax.axhline(np.nanmedian(hp_resid), color="C1", lw=1.5, ls="--", label="HOTPANTS median")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("median core |resid| [ADU]")
    ax.set_title("Median residual vs config")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.bar(x, p90s, color="C2", alpha=0.8)
    ax.axhline(np.nanpercentile(hp_resid, 90), color="C1", lw=1.5, ls="--", label="HOTPANTS p90")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("p90 core |resid| [ADU]")
    ax.set_title("p90 residual vs config")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.bar(x, np.log10(np.clip(lam_sels, 1e-9, None)), color="C3", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("log₁₀(λ selected)")
    ax.set_title("CV-selected λ")
    for i, r in enumerate(rows):
        if abs(math.log10(max(r["lam"], 1e-9)) - 12) < 0.1:
            ax.text(
                i,
                math.log10(r["lam"]) + 0.1,
                "⚠ ceiling",
                ha="center",
                fontsize=7,
                color="red",
            )

    ax = axes[1, 1]
    ax.bar(x, edofs, color="C4", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("effective DOF")
    ax.set_title("Spatial effective DOF")

    fig.suptitle("Lambda-ceiling × n_knots sweep  —  t2_s115088_ut2", fontsize=12)
    fig.tight_layout()
    out = OUT / "diag_real_sweep.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"Saved {out.name}")

    print("\nAll diagnostics written to tests/artifacts/")


if __name__ == "__main__":
    main()
