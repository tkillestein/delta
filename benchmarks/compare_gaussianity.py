"""Numerically compare a delta difference against the bundled HOTPANTS difference,
scoring how Gaussian each residual is.

The real MEF (artifacts/t2_s115088_ut2.fits) already carries the optimised
HOTPANTS products: HDU ``DIFFERENCE`` and its ``DIFFERENCE_BACKGROUND_RMS``, plus
the exact HOTPANTS call in the ``DIFFERENCE`` header (``DIFFCMD``). We run delta on
the same SCIENCE/TEMPLATE planes (cropped to a common window), form the pull
(difference / its 1-sigma) for both methods, and report Gaussianity of the
source-free core: a well-matched, properly-whitened difference has pull
sigma -> 1 with skew/excess-kurtosis -> 0.

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


def gaussianity(pull: np.ndarray, label: str) -> dict:
    """Pull-distribution Gaussianity of the source-free core."""
    p = pull[np.isfinite(pull)]
    med = np.median(p)
    rsig = 1.4826 * np.median(np.abs(p - med))  # robust sigma of the *pull*
    core = _sigma_clip(p)
    m, s = core.mean(), core.std()
    skew = float(((core - m) ** 3).mean() / s**3)
    exkurt = float(((core - m) ** 4).mean() / s**4 - 3.0)
    # KS vs N(0,1) on the standardised core (subsampled for speed).
    z = (core - m) / s
    rng = np.random.default_rng(0)
    z = np.sort(rng.choice(z, size=min(40000, z.size), replace=False))
    F = np.arange(1, z.size + 1) / z.size
    ks = float(np.max(np.abs(F - _normcdf(z))))
    # Tail fractions vs the Gaussian expectations (0.317 / 0.0455 / 0.0027).
    out1 = float(np.mean(np.abs(p - med) > 1 * rsig))
    out3 = float(np.mean(np.abs(p - med) > 3 * rsig))
    print(
        f"  {label:28s} pull_sigma={rsig:6.3f}  skew={skew:+.3f}  "
        f"exkurt={exkurt:+7.2f}  KS={ks:.4f}  >1s={out1:.3f} >3s={out3:.4f}  n={p.size}"
    )
    return dict(label=label, pull_sigma=rsig, skew=skew, exkurt=exkurt, ks=ks)


def total_error(data, bkg_rms, gain):
    bkg = np.nan_to_num(bkg_rms.astype(np.float64), nan=0.0)
    src = np.clip(np.nan_to_num(data, nan=0.0), 0.0, None) / gain
    return np.sqrt(bkg**2 + src)


def load_plane(h, image, mask, rms):
    data = np.asarray(h[image].data, np.float32)
    rms_map = np.asarray(h[rms].data, np.float32)
    raw = np.asarray(h[mask].data)
    et = float(h[image].header["EXPTIME"])
    var = total_error(data, rms_map, et).astype(np.float32) ** 2
    bad = (raw != 0) | ~np.isfinite(data) | ~np.isfinite(rms_map) | (rms_map <= 0)
    data = np.where(np.isfinite(data), data, 0.0).astype(np.float32)
    var = np.where(np.isfinite(var) & (var > 0), var, 1e30).astype(np.float32)
    return data, var, bad.astype(np.uint8)


def crop(a, win):
    return np.ascontiguousarray(a[win])


def main() -> int:
    with fits.open(FITS, memmap=True) as h:
        sci, sci_var, sci_bad = load_plane(h, "SCIENCE", "SCIENCE_MASK", "SCIENCE_BACKGROUND_RMS")
        tmpl, tmpl_var, tmpl_bad = load_plane(
            h, "TEMPLATE", "TEMPLATE_MASK", "TEMPLATE_BACKGROUND_RMS"
        )
        hp_diff = np.asarray(h["DIFFERENCE"].data, np.float64)
        hp_rms = np.asarray(h["DIFFERENCE_BACKGROUND_RMS"].data, np.float64)

    ny = min(sci.shape[0], tmpl.shape[0])
    nx = min(sci.shape[1], tmpl.shape[1])
    sci, sci_var, sci_bad = sci[:ny, :nx], sci_var[:ny, :nx], sci_bad[:ny, :nx]
    tmpl, tmpl_var, tmpl_bad = tmpl[:ny, :nx], tmpl_var[:ny, :nx], tmpl_bad[:ny, :nx]
    hp_diff, hp_rms = hp_diff[:ny, :nx], hp_rms[:ny, :nx]

    csci, csvar, csbad = crop(sci, WIN), crop(sci_var, WIN), crop(sci_bad, WIN)
    ctmpl, ctvar, ctbad = crop(tmpl, WIN), crop(tmpl_var, WIN), crop(tmpl_bad, WIN)
    chp, chp_rms = crop(hp_diff, WIN), crop(hp_rms, WIN)

    # HOTPANTS reference pull (its own difference / background RMS).
    hp_pull = np.where(chp_rms > 0, chp / np.maximum(chp_rms, 1e-12), np.nan)
    print(f"window {WIN}  ({csci.shape[1]}x{csci.shape[0]} px)\n")
    gaussianity(hp_pull, "HOTPANTS (bundled)")

    # delta configs to compare (instrument saturation 552 from HOTPANTS -tu).
    configs = {
        "delta default": dict(n_knots=8),
        "delta sat+brightmask": dict(n_knots=8, saturation=552.0, bright_mask=60.0),
        "delta +coarse knots": dict(n_knots=4, saturation=552.0, bright_mask=60.0),
        "delta +r13": dict(n_knots=4, saturation=552.0, bright_mask=60.0, radius=13),
    }
    for name, kw in configs.items():
        res = delta.subtract(
            csci,
            ctmpl,
            science_var=csvar,
            reference_var=ctvar,
            science_mask=csbad,
            reference_mask=ctbad,
            n_max=6,
            stamp_radius=15,
            threshold_sigma=8.0,
            max_stamps=300,
            decorrelate=True,
            **kw,
        )
        assert res.variance is not None and res.mask is not None
        m = res.mask
        var = np.asarray(res.variance)
        dpull = np.where(
            (var > 0) & (m == 0), res.difference / np.sqrt(np.maximum(var, 1e-12)), np.nan
        )
        s = res.solution
        print(
            f"\n[{name}]  beta={s.beta:.2f} lam={s.lam:.2g} chi2={s.reduced_chi2:.3f} "
            f"used={s.n_stamps_used} rej={s.n_stamps_rejected} "
            f"masked={100 * np.mean(m != 0):.1f}%"
        )
        gaussianity(dpull, name + " pull")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
