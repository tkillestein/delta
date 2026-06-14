"""Numerically compare a delta difference against the bundled HOTPANTS difference.

The real MEF (artifacts/t2_s115088_ut2.fits) already carries the optimised
HOTPANTS products: HDU ``DIFFERENCE`` and ``DIFFERENCE_BACKGROUND_RMS``, the
``SCIENCE_PHOTOMETRY`` catalogue, and the exact HOTPANTS call in the
``DIFFERENCE`` header (``DIFFCMD``). We run delta on the same SCIENCE/TEMPLATE
planes (cropped to a common window) and score three things head-to-head:

* **Gaussianity** of the pull (difference / its 1-sigma): a well-matched,
  whitened difference has a source-free-core pull with sigma -> 1 and
  skew / excess-kurtosis -> 0.
* **Whiteness**: the source-free noise autocorrelation (lag 1..4). delta's
  decorrelation should drive this below HOTPANTS (which does not whiten). The
  *input* science/template background ACF is also measured -- it is the floor,
  since resampling onto a common astrometric grid pre-correlates the pixels.
* **Bright-star cores**: the raw |residual| in counts at unsaturated bright
  stars (directly comparable between methods), which is dominated by how well
  the matching kernel reproduces the PSF.

Usage:  python -m benchmarks.compare_gaussianity
"""

from __future__ import annotations

import math
from pathlib import Path

import delta
import numpy as np
from astropy.io import fits

ART = Path(__file__).resolve().parents[1] / "artifacts"
FITS = ART / "t2_s115088_ut2.fits"

# A central window with a healthy source density (rows, cols on the science grid).
WIN = (slice(2200, 4248), slice(3200, 5248))

_erf = np.vectorize(math.erf)


def _normcdf(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + _erf(z / math.sqrt(2.0)))


def _sigma_clip(x: np.ndarray, nsig: float = 4.0, iters: int = 6) -> np.ndarray:
    x = x[np.isfinite(x)]
    for _ in range(iters):
        m = np.median(x)
        s = 1.4826 * np.median(np.abs(x - m))
        if s <= 0:
            break
        keep = np.abs(x - m) < nsig * s
        if keep.all():
            break
        x = x[keep]
    return x


def gaussianity(pull: np.ndarray, label: str) -> None:
    """Pull-distribution Gaussianity of the source-free core."""
    p = pull[np.isfinite(pull)]
    med = np.median(p)
    rsig = 1.4826 * np.median(np.abs(p - med))  # robust sigma of the *pull*
    core = _sigma_clip(p)
    m, s = core.mean(), core.std()
    skew = float(((core - m) ** 3).mean() / s**3)
    exkurt = float(((core - m) ** 4).mean() / s**4 - 3.0)
    z = (core - m) / s
    rng = np.random.default_rng(0)
    z = np.sort(rng.choice(z, size=min(40000, z.size), replace=False))
    cdf = np.arange(1, z.size + 1) / z.size
    ks = float(np.max(np.abs(cdf - _normcdf(z))))
    print(
        f"  {label:26s} pull_sigma={rsig:6.3f}  skew={skew:+.3f}  "
        f"exkurt={exkurt:+.3f}  KS={ks:.4f}  n={p.size}"
    )


def acf(diff: np.ndarray, mask: np.ndarray | None = None, lags: int = 4) -> np.ndarray:
    """Source-free noise autocorrelation at lags 1..L (x/y averaged)."""
    a = diff.astype(np.float64).copy()
    if mask is not None:
        a[mask != 0] = np.nan
    m = np.nanmedian(a)
    s = 1.4826 * np.nanmedian(np.abs(a - m))
    a[np.abs(a - m) > 4 * s] = np.nan  # clip sources
    a = a - np.nanmedian(a)
    var = np.nanmean(a * a)
    out = []
    for lag in range(1, lags + 1):
        cx = np.nanmean(a[:, :-lag] * a[:, lag:])
        cy = np.nanmean(a[:-lag, :] * a[lag:, :])
        out.append(0.5 * (cx + cy) / var)
    return np.array(out)


def core_residual(diff: np.ndarray, xs: np.ndarray, ys: np.ndarray, rad: int = 2):
    """median / p90 of the median |residual| in a (2 rad + 1) box at each star."""
    vals = []
    for x, y in zip(xs, ys, strict=True):
        box = np.abs(diff[y - rad : y + rad + 1, x - rad : x + rad + 1])
        if np.isfinite(box).any():
            vals.append(np.nanmedian(box))
    return float(np.nanmedian(vals)), float(np.nanpercentile(vals, 90))


def _total_error(data, bkg_rms, gain):
    bkg = np.nan_to_num(bkg_rms.astype(np.float64), nan=0.0)
    src = np.clip(np.nan_to_num(data, nan=0.0), 0.0, None) / gain
    return np.sqrt(bkg**2 + src)


def _load_plane(h, image, mask, rms):
    data = np.asarray(h[image].data, np.float32)
    rms_map = np.asarray(h[rms].data, np.float32)
    raw = np.asarray(h[mask].data)
    et = float(h[image].header["EXPTIME"])
    var = _total_error(data, rms_map, et).astype(np.float32) ** 2
    bad = (raw != 0) | ~np.isfinite(data) | ~np.isfinite(rms_map) | (rms_map <= 0)
    data = np.where(np.isfinite(data), data, 0.0).astype(np.float32)
    var = np.where(np.isfinite(var) & (var > 0), var, 1e30).astype(np.float32)
    return data, var, bad.astype(np.uint8)


def main() -> int:
    y0, x0 = WIN[0].start, WIN[1].start
    with fits.open(FITS, memmap=True) as h:
        sci, svar, sbad = _load_plane(h, "SCIENCE", "SCIENCE_MASK", "SCIENCE_BACKGROUND_RMS")
        tmpl, tvar, tbad = _load_plane(h, "TEMPLATE", "TEMPLATE_MASK", "TEMPLATE_BACKGROUND_RMS")
        hp_diff = np.asarray(h["DIFFERENCE"].data, np.float64)
        hp_rms = np.asarray(h["DIFFERENCE_BACKGROUND_RMS"].data, np.float64)
        phot = h["SCIENCE_PHOTOMETRY"].data
        xpk, ypk, peak = (np.asarray(phot[k]) for k in ("xpeak", "ypeak", "peak"))

    ny = min(sci.shape[0], tmpl.shape[0])
    nx = min(sci.shape[1], tmpl.shape[1])
    cut = lambda a: np.ascontiguousarray(a[:ny, :nx][WIN])  # noqa: E731
    sci, svar, sbad = cut(sci), cut(svar), cut(sbad)
    tmpl, tvar, tbad = cut(tmpl), cut(tvar), cut(tbad)
    hp_diff, hp_rms = cut(hp_diff), cut(hp_rms)
    h_, w_ = sci.shape

    # Bright, unsaturated stars inside the window (away from the edge).
    sel = (peak > 40) & (peak < 520) & (xpk >= x0 + 8) & (xpk < x0 + w_ - 8)
    sel &= (ypk >= y0 + 8) & (ypk < y0 + h_ - 8)
    lx = (xpk[sel] - x0).astype(int)
    ly = (ypk[sel] - y0).astype(int)
    order = np.argsort(peak[sel])[::-1][:250]
    lx, ly = lx[order], ly[order]
    print(f"window {WIN}  ({w_}x{h_} px), {lx.size} bright stars\n")

    # --- input-noise floor: resampling pre-correlates the pixels -------------
    print("input background autocorrelation (the whiteness floor):")
    print(f"  science  ACF lag1..4 = {np.round(acf(sci, sbad), 4)}")
    print(f"  template ACF lag1..4 = {np.round(acf(tmpl, tbad), 4)}\n")

    # --- HOTPANTS reference --------------------------------------------------
    hp_pull = np.where(hp_rms > 0, hp_diff / np.maximum(hp_rms, 1e-12), np.nan)
    print("HOTPANTS (bundled):")
    gaussianity(hp_pull, "pull")
    print(f"  whiteness ACF lag1..4 = {np.round(acf(hp_diff), 4)}")
    hm, hp90 = core_residual(hp_diff, lx, ly)
    print(f"  core |resid| median / p90 = {hm:.3f} / {hp90:.3f}\n")

    # --- delta (recommended config) ------------------------------------------
    for dec in (False, True):
        res = delta.subtract(
            sci,
            tmpl,
            science_var=svar,
            reference_var=tvar,
            science_mask=sbad,
            reference_mask=tbad,
            n_max=8,
            n_knots=4,
            stamp_radius=15,
            threshold_sigma=8.0,
            max_stamps=300,
            saturation=552.0,
            decorrelate=dec,
        )
        assert res.mask is not None and res.variance is not None
        var = np.asarray(res.variance)
        good = (var > 0) & (res.mask == 0)
        dpull = np.where(good, res.difference / np.sqrt(np.maximum(var, 1e-12)), np.nan)
        s = res.solution
        print(
            f"delta n_max=8 decorrelate={dec} (chi2={s.reduced_chi2:.3f}, "
            f"{s.n_stamps_used} stamps):"
        )
        gaussianity(dpull, "pull")
        print(f"  whiteness ACF lag1..4 = {np.round(acf(res.difference, res.mask), 4)}")
        dm = np.where(res.mask == 0, res.difference, np.nan)
        cm, c90 = core_residual(dm, lx, ly)
        print(f"  core |resid| median / p90 = {cm:.3f} / {c90:.3f}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
