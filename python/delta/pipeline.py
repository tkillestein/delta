"""High-level difference-imaging pipeline (SPEC §7).

Ties the C++ core together: coerce inputs, select stamps and the convolution
direction, fit the spatially-varying matching kernel + differential background,
subtract, and optionally decorrelate the noise and build a match-filtered score.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from . import _core
from ._inputs import as_layers, synth_variance
from .solution import KernelSolution

_FWHM_TO_SIGMA = 1.0 / 2.35482


@dataclass
class DiffResult:
    """Products of a subtraction run."""

    difference: NDArray[np.float32]
    variance: NDArray[np.float32] | None
    mask: NDArray[np.uint8] | None
    score: NDArray[np.float32] | None
    solution: KernelSolution

    def write(self, path: str, overwrite: bool = False) -> None:
        """Write products to a multi-extension FITS with provenance (needs astropy)."""
        from astropy.io import fits  # optional dependency

        cards = self.solution.header_cards()
        primary = fits.PrimaryHDU(self.difference)
        for key, value in cards.items():
            primary.header[key] = value
        hdus: list[fits.hdu.base.ExtensionHDU | fits.PrimaryHDU] = [primary]
        if self.variance is not None:
            hdus.append(fits.ImageHDU(self.variance, name="VARIANCE"))
        if self.mask is not None:
            hdus.append(fits.ImageHDU(self.mask, name="MASK"))
        if self.score is not None:
            hdus.append(fits.ImageHDU(self.score, name="SCORE"))
        fits.HDUList(hdus).writeto(path, overwrite=overwrite)


def _default_beta(fwhm_a: float, fwhm_b: float) -> float:
    """Heuristic kernel scale from the two median FWHMs."""
    fwhms = [f for f in (fwhm_a, fwhm_b) if np.isfinite(f) and f > 0]
    if not fwhms:
        return 2.0
    broad, narrow = max(fwhms), min(fwhms)
    sig_broad = broad * _FWHM_TO_SIGMA
    # Width of the broadening kernel sqrt(sig_broad^2 - sig_narrow^2), with a
    # floor so an already-matched pair still gets a usable basis scale.
    diff = sig_broad**2 - (narrow * _FWHM_TO_SIGMA) ** 2
    k_sigma = np.sqrt(diff) if diff > 0 else 0.0
    return float(max(1.0, k_sigma if k_sigma > 0.5 else 0.5 * sig_broad))


def _estimate_psf(image: NDArray[np.float32], xs, ys, radius: int) -> NDArray[np.float32]:
    """Median, unit-sum postage stamp of the listed sources (a crude PSF)."""
    h, w = image.shape
    stamps = []
    for x, y in zip(xs, ys, strict=True):
        if x - radius < 0 or x + radius >= w or y - radius < 0 or y + radius >= h:
            continue
        cut = image[y - radius : y + radius + 1, x - radius : x + radius + 1].astype(np.float64)
        cut = cut - np.median(cut)
        total = cut.sum()
        if total > 0:
            stamps.append(cut / total)
    if not stamps:
        raise ValueError("no usable sources for PSF estimation")
    psf = np.median(np.stack(stamps), axis=0)
    psf = np.clip(psf, 0.0, None)
    psf /= psf.sum()
    return np.ascontiguousarray(psf, dtype=np.float32)


class Subtractor:
    """Reusable difference-imaging configuration (SPEC §7 object-oriented path)."""

    def __init__(
        self,
        *,
        n_max: int = 2,
        beta: float | None = None,
        n_knots: int = 5,
        stamp_radius: int = 15,
        threshold_sigma: float = 5.0,
        max_stamps: int = 200,
        saturation: float = 0.0,
        lambda_grid: NDArray[np.float64] | None = None,
        decorrelate: bool = False,
        score: bool = False,
        block: int = 256,
    ) -> None:
        self.n_max = n_max
        self.beta = beta
        self.n_knots = n_knots
        self.stamp_radius = stamp_radius
        self.threshold_sigma = threshold_sigma
        self.max_stamps = max_stamps
        self.saturation = saturation
        self.lambda_grid = (
            np.logspace(-6.0, 6.0, 25) if lambda_grid is None else np.asarray(lambda_grid)
        )
        self.decorrelate = decorrelate
        self.score = score
        self.block = block

    # -- internals -----------------------------------------------------------

    def _select(self, sci, ref, sci_mask, ref_mask, catalog):
        cat_x = cat_y = None
        if catalog is not None:
            cat = np.asarray(catalog, dtype=np.int32)
            cat_x = np.ascontiguousarray(cat[:, 0])
            cat_y = np.ascontiguousarray(cat[:, 1])
        return _core.select_stamps(
            sci,
            ref,
            science_mask=sci_mask,
            reference_mask=ref_mask,
            catalog_x=cat_x,
            catalog_y=cat_y,
            stamp_radius=self.stamp_radius,
            threshold_sigma=self.threshold_sigma,
            max_stamps=self.max_stamps,
            saturation=self.saturation,
        )

    def _fit(
        self,
        science,
        reference,
        *,
        science_var,
        reference_var,
        science_mask,
        reference_mask,
        gain,
        read_noise,
        catalog,
        direction,
    ):
        sci, svar, smask = as_layers(science, science_var, science_mask)
        ref, rvar, rmask = as_layers(reference, reference_var, reference_mask)
        if sci.shape != ref.shape:
            raise ValueError("science and reference must share a shape")
        if svar is None and gain is not None:
            svar = synth_variance(sci, gain, read_noise)
        if rvar is None and gain is not None:
            rvar = synth_variance(ref, gain, read_noise)

        sel = self._select(sci, ref, smask, rmask, catalog)
        if direction is None:
            direction = sel["direction"]

        # The convolved image is the sharper one; the other is the fit target.
        # direction == "science" means science is convolved -> reference is the
        # target and the difference sign is flipped so transients stay positive.
        if direction == "science":
            target, tvar, tmask = ref, rvar, rmask
            conv, cvar, cmask = sci, svar, smask
            sign = -1.0
        else:
            target, tvar, tmask = sci, svar, smask
            conv, cvar, cmask = ref, rvar, rmask
            sign = 1.0

        beta = self.beta
        if beta is None:
            beta = _default_beta(sel["median_fwhm_science"], sel["median_fwhm_reference"])

        h, w = sci.shape
        knots = _core.grid_knots(0.0, 0.0, w - 1.0, h - 1.0, self.n_knots, self.n_knots)
        stamp_x = np.ascontiguousarray(sel["x"], dtype=np.int32)
        stamp_y = np.ascontiguousarray(sel["y"], dtype=np.int32)

        fit = _core.fit_kernel(
            target,
            conv,
            knots,
            stamp_x,
            stamp_y,
            self.stamp_radius,
            beta,
            self.n_max,
            self.lambda_grid,
            science_var=tvar,
            reference_var=cvar,
            science_mask=tmask,
            reference_mask=cmask,
        )
        solution = KernelSolution(
            beta=beta,
            n_max=self.n_max,
            radius=0,
            knots=knots,
            theta=fit["theta"],
            n_components=fit["n_components"],
            n_spatial=fit["n_spatial"],
            lam=fit["lambda"],
            gcv=fit["gcv"],
            effective_dof=fit["effective_dof"],
            rss=fit["rss"],
            component_sums=fit["component_sums"],
            direction=direction,
            shape=(h, w),
        )
        layers = (target, conv, tvar, cvar, tmask, cmask, sign, stamp_x, stamp_y)
        return solution, layers

    # -- public API ----------------------------------------------------------

    def fit(
        self,
        science,
        reference,
        *,
        science_var=None,
        reference_var=None,
        science_mask=None,
        reference_mask=None,
        gain=None,
        read_noise=0.0,
        catalog=None,
        direction=None,
    ) -> KernelSolution:
        """Select stamps and fit the kernel/background, returning the solution."""
        solution, _ = self._fit(
            science,
            reference,
            science_var=science_var,
            reference_var=reference_var,
            science_mask=science_mask,
            reference_mask=reference_mask,
            gain=gain,
            read_noise=read_noise,
            catalog=catalog,
            direction=direction,
        )
        return solution

    def subtract(
        self,
        science,
        reference,
        *,
        science_var=None,
        reference_var=None,
        science_mask=None,
        reference_mask=None,
        gain=None,
        read_noise=0.0,
        catalog=None,
        direction=None,
    ) -> DiffResult:
        """Fit and produce the difference (and optional decorrelation/score)."""
        solution, layers = self._fit(
            science,
            reference,
            science_var=science_var,
            reference_var=reference_var,
            science_mask=science_mask,
            reference_mask=reference_mask,
            gain=gain,
            read_noise=read_noise,
            catalog=catalog,
            direction=direction,
        )
        target, conv, tvar, cvar, tmask, cmask, sign, stamp_x, stamp_y = layers

        diff = _core.subtract(
            target,
            conv,
            solution.knots,
            solution.theta,
            solution.beta,
            solution.n_max,
            science_var=tvar,
            reference_var=cvar,
            science_mask=tmask,
            reference_mask=cmask,
        )
        difference = (sign * diff["difference"]).astype(np.float32)
        variance = diff["variance"]
        mask = diff["mask"]

        whitened = difference
        if self.decorrelate and tvar is not None and cvar is not None:
            whitened = _core.decorrelate(
                difference,
                solution.knots,
                solution.theta,
                solution.beta,
                solution.n_max,
                tvar,
                cvar,
                block=self.block,
            )

        score = None
        if self.score:
            psf_radius = min(self.stamp_radius, 8)
            psf = _estimate_psf(target, stamp_x, stamp_y, psf_radius)
            noise_var = (
                float(np.median(variance[variance > 0]))
                if variance is not None
                else float(np.var(whitened))
            )
            score = _core.matched_filter(whitened, psf, noise_var)

        out_diff = whitened if self.decorrelate else difference
        return DiffResult(
            difference=out_diff,
            variance=variance,
            mask=mask,
            score=score,
            solution=solution,
        )


def subtract(science, reference, **kwargs) -> DiffResult:
    """Convenience wrapper: configure a :class:`Subtractor` and run it once.

    Configuration keywords (``n_max``, ``beta``, ``n_knots``, ``stamp_radius``,
    ``threshold_sigma``, ``max_stamps``, ``saturation``, ``lambda_grid``,
    ``decorrelate``, ``score``, ``block``) are split from the per-call inputs
    (``science_var``, ``reference_var``, ``science_mask``, ``reference_mask``,
    ``gain``, ``read_noise``, ``catalog``, ``direction``).
    """
    config_keys = {
        "n_max",
        "beta",
        "n_knots",
        "stamp_radius",
        "threshold_sigma",
        "max_stamps",
        "saturation",
        "lambda_grid",
        "decorrelate",
        "score",
        "block",
    }
    config = {k: v for k, v in kwargs.items() if k in config_keys}
    call = {k: v for k, v in kwargs.items() if k not in config_keys}
    return Subtractor(**config).subtract(science, reference, **call)
