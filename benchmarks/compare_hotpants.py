"""HOTPANTS head-to-head harness (SPEC §10, §13.4).

Runs delta and HOTPANTS on the same science/reference pair and compares residual
RMS away from sources. HOTPANTS is an external binary; if it is not on PATH this
module still provides delta's side and the residual metric. Usage:

    python -m benchmarks.compare_hotpants            # synthetic demo
    python -m benchmarks.compare_hotpants sci.fits ref.fits
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import delta
import numpy as np
from delta import validation


def hotpants_available() -> bool:
    return shutil.which("hotpants") is not None


def run_hotpants(science: np.ndarray, reference: np.ndarray, **kwargs) -> np.ndarray:
    """Run HOTPANTS on two arrays and return its difference image.

    Raises RuntimeError if the binary is unavailable. Extra keyword arguments are
    passed through as ``-key value`` CLI flags.
    """
    if not hotpants_available():
        raise RuntimeError("hotpants not found on PATH")
    from astropy.io import fits

    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        fits.PrimaryHDU(science.astype(np.float32)).writeto(d / "sci.fits")
        fits.PrimaryHDU(reference.astype(np.float32)).writeto(d / "ref.fits")
        cmd = [
            "hotpants",
            "-inim",
            str(d / "sci.fits"),
            "-tmplim",
            str(d / "ref.fits"),
            "-outim",
            str(d / "diff.fits"),
        ]
        for key, value in kwargs.items():
            cmd += [f"-{key}", str(value)]
        subprocess.run(cmd, check=True, capture_output=True)
        with fits.open(d / "diff.fits") as hdul:
            return np.asarray(hdul[0].data, dtype=np.float64)


def residual_rms(difference: np.ndarray, source_free: tuple[slice, slice]) -> float:
    """RMS of the difference over a source-free region (lower is better)."""
    return float(np.std(difference[source_free]))


def _demo() -> int:
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

    res = delta.subtract(
        sci_n, ref_n, science_var=var, reference_var=var, n_knots=4, stamp_radius=12
    )
    region = (slice(75, 105), slice(75, 105))  # between stars
    print(f"delta    residual RMS = {residual_rms(res.difference, region):.3f}")
    if hotpants_available():
        hp = run_hotpants(sci_n, ref_n)
        print(f"hotpants residual RMS = {residual_rms(hp, region):.3f}")
    else:
        print("hotpants not on PATH — install it to run the head-to-head.")
    return 0


def main(argv: list[str]) -> int:
    if len(argv) == 3:
        from astropy.io import fits

        sci = fits.getdata(argv[1]).astype(np.float32)
        ref = fits.getdata(argv[2]).astype(np.float32)
        res = delta.subtract(sci, ref)
        region = (slice(0, sci.shape[0] // 8), slice(0, sci.shape[1] // 8))
        print(f"delta residual RMS (corner) = {residual_rms(res.difference, region):.3f}")
        if hotpants_available():
            hp = run_hotpants(sci, ref)
            print(f"hotpants residual RMS (corner) = {residual_rms(hp, region):.3f}")
        return 0
    return _demo()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
