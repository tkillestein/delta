"""Visual HOTPANTS head-to-head (companion to compare_hotpants).

Runs delta and HOTPANTS on the same science/reference pair and writes a panel
figure to the artifacts/ folder for human validation: inputs, both difference
images on a shared stretch, and their residual histograms away from sources.

    python -m benchmarks.visualize_hotpants            # synthetic demo
    python -m benchmarks.visualize_hotpants sci.fits ref.fits
"""

from __future__ import annotations

import sys
from pathlib import Path

import delta
import numpy as np
from delta import validation

from benchmarks.compare_hotpants import (
    hotpants_available,
    residual_rms,
    run_hotpants,
)

ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"


def _synthetic():
    h = w = 256
    rng = np.random.default_rng(0)
    grid = [(x, y) for x in (50, 110, 170, 220) for y in (50, 110, 170, 220)]
    ref = np.full((h, w), 120.0)
    sci = np.full((h, w), 120.0)
    for x, y in grid:
        ref += validation.gaussian_psf((h, w), x, y, 12000.0, 1.6)
        sci += validation.gaussian_psf((h, w), x, y, 12000.0, 2.4)
    var = np.full((h, w), 9.0, np.float32)
    ref_n = (ref + rng.normal(0, 3.0, (h, w))).astype(np.float32)
    sci_n = (sci + rng.normal(0, 3.0, (h, w))).astype(np.float32)
    region = (slice(75, 105), slice(75, 105))
    res = delta.subtract(
        sci_n, ref_n, science_var=var, reference_var=var, n_knots=4, stamp_radius=12
    )
    return sci_n, ref_n, res.difference, region


def _from_files(sci_path: str, ref_path: str):
    from astropy.io import fits

    sci = fits.getdata(sci_path).astype(np.float32)
    ref = fits.getdata(ref_path).astype(np.float32)
    region = (slice(0, sci.shape[0] // 8), slice(0, sci.shape[1] // 8))
    res = delta.subtract(sci, ref)
    return sci, ref, res.difference, region


def make_figure(sci, ref, delta_diff, region, hp_diff, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    have_hp = hp_diff is not None
    delta_rms = residual_rms(delta_diff, region)
    hp_rms = residual_rms(hp_diff, region) if have_hp else None

    # Shared symmetric stretch for the difference panels.
    diffs = [delta_diff] + ([hp_diff] if have_hp else [])
    span = np.percentile(np.abs(np.concatenate([d.ravel() for d in diffs])), 99)
    in_lo, in_hi = np.percentile(np.concatenate([sci.ravel(), ref.ravel()]), [1, 99])

    fig, axes = plt.subplots(2, 3, figsize=(13, 8.5))

    def show(ax, img, title, **kw):
        im = ax.imshow(img, origin="lower", **kw)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    show(axes[0, 0], sci, "science", cmap="gray", vmin=in_lo, vmax=in_hi)
    show(axes[0, 1], ref, "reference", cmap="gray", vmin=in_lo, vmax=in_hi)

    show(
        axes[0, 2],
        delta_diff,
        f"delta difference (RMS={delta_rms:.3f})",
        cmap="RdBu_r",
        vmin=-span,
        vmax=span,
    )
    if have_hp:
        show(
            axes[1, 0],
            hp_diff,
            f"HOTPANTS difference (RMS={hp_rms:.3f})",
            cmap="RdBu_r",
            vmin=-span,
            vmax=span,
        )
    else:
        axes[1, 0].text(0.5, 0.5, "HOTPANTS\nnot on PATH", ha="center", va="center")
        axes[1, 0].axis("off")

    # Mark the source-free measurement region on delta's difference.
    ys, xs = region
    axes[0, 2].add_patch(
        Rectangle(
            (xs.start, ys.start),
            xs.stop - xs.start,
            ys.stop - ys.start,
            fill=False,
            edgecolor="lime",
            lw=1.5,
        )
    )

    # delta - hotpants pixel difference.
    if have_hp:
        d = delta_diff - hp_diff
        dspan = np.percentile(np.abs(d), 99)
        show(
            axes[1, 1],
            d,
            f"delta - HOTPANTS (RMS={np.std(d):.3f})",
            cmap="PuOr",
            vmin=-dspan,
            vmax=dspan,
        )
    else:
        axes[1, 1].axis("off")

    # Residual histograms over the source-free region.
    ax = axes[1, 2]
    bins = np.linspace(-4 * span, 4 * span, 80)
    ax.hist(
        delta_diff[region].ravel(),
        bins=bins,
        histtype="step",
        label=f"delta ({delta_rms:.3f})",
        color="C0",
    )
    if have_hp:
        ax.hist(
            hp_diff[region].ravel(),
            bins=bins,
            histtype="step",
            label=f"HOTPANTS ({hp_rms:.3f})",
            color="C1",
        )
    ax.set_title("residual distribution (source-free)", fontsize=10)
    ax.set_xlabel("pixel value")
    ax.legend(fontsize=8)

    fig.suptitle("delta vs HOTPANTS head-to-head", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main(argv: list[str]) -> int:
    if len(argv) == 3:
        sci, ref, delta_diff, region = _from_files(argv[1], argv[2])
    else:
        sci, ref, delta_diff, region = _synthetic()

    hp_diff = None
    if hotpants_available():
        hp_diff = run_hotpants(sci, ref)
    else:
        print("hotpants not on PATH — figure will show delta only.")

    ARTIFACTS.mkdir(exist_ok=True)
    out_path = ARTIFACTS / "hotpants_head_to_head.png"
    make_figure(sci, ref, delta_diff, region, hp_diff, out_path)

    print(f"delta    residual RMS = {residual_rms(delta_diff, region):.3f}")
    if hp_diff is not None:
        print(f"hotpants residual RMS = {residual_rms(hp_diff, region):.3f}")
    print(f"figure written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
