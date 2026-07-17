"""High-level difference-imaging pipeline (SPEC §7).

Ties the C++ core together: coerce inputs, select stamps and the convolution
direction, fit the spatially-varying matching kernel + differential background,
subtract, and optionally decorrelate the noise and build a match-filtered score.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from . import _core
from ._inputs import as_layers, synth_variance
from ._log import log_timing, logger
from ._provenance import environment_cards
from .solution import KernelSolution

# 1 / (2*sqrt(2 ln 2)); matches kFwhmPerSigma in src/detect.cpp.
_FWHM_TO_SIGMA = 1.0 / 2.354820045030949

# Keyword names that configure a `Subtractor` (vs. per-call image inputs); used by
# the `subtract` / `apply` convenience wrappers to split **kwargs.
_CONFIG_KEYS = frozenset(
    {
        "n_max",
        "beta",
        "n_knots",
        "radius",
        "stamp_radius",
        "threshold_sigma",
        "max_stamps",
        "saturation",
        "match_radius",
        "fwhm_radius",
        "bright_mask",
        "lambda_grid",
        "clip_sigma",
        "clip_iterations",
        "min_stamps",
        "cv_folds",
        "spatial_scale",
        "decorrelate",
        "score",
        "source_catalog",
        "block",
        "rescale_variance",
    }
)

# Warn when the exact (non-stamped) k-fold CV path's rebuilt-every-IRLS-pass
# whitened design exceeds this size (see fit_kernel's cv_exact_design_bytes).
_CV_EXACT_DESIGN_WARN_BYTES = 256 * 1024 * 1024

# Median of a chi-square_1 (squared standard normal) distribution. Used to turn a
# robust median(z^2) into a reduced-chi2-equivalent scale factor without the
# mean's sensitivity to high-|z| outliers (genuine transients).
_MEDIAN_CHI2_1 = 0.4549364231195724


def _variance_rescale_factor(
    difference: NDArray[np.float32],
    variance: NDArray[np.float32],
    mask: NDArray[np.uint8] | None,
) -> float:
    """Scalar by which to multiply ``variance`` so the diff-image reduced chi2 -> 1.

    ``median(difference^2 / variance) / median(chi2_1)`` is the median-based
    analogue of a reduced chi2: under a correctly-scaled Gaussian noise model
    ``z = difference / sqrt(variance)`` is standard normal, so ``z^2`` has
    median 0.4549. Using the median rather than the mean keeps real transients
    (which are deliberately high-|z| outliers, not noise) from inflating the
    correction.
    """
    good = (variance > 0) & np.isfinite(difference) & np.isfinite(variance)
    if mask is not None:
        good &= mask == 0
    if not np.any(good):
        return 1.0
    z2 = difference[good].astype(np.float64) ** 2 / variance[good].astype(np.float64)
    return float(np.median(z2) / _MEDIAN_CHI2_1)


def _log_core_timings(timing: dict | None) -> None:
    """Emit the C++ core's opt-in sub-stage timers (set ``DELTA_TIMING``).

    The core returns a ``{label: seconds}`` dict (or ``None`` when ``DELTA_TIMING``
    is unset) splitting e.g. the kernel solve into ``B_n`` convolve vs GLS solve.
    They are logged in the same ``"<label> done in <s>s"`` form as
    :func:`log_timing` so timing consumers (the benchmark) pick them up uniformly.
    """
    if not timing:
        return
    for label, seconds in timing.items():
        logger.debug("{} done in {:.3f}s", label, seconds)


@dataclass
class DiffResult:
    """Products of a subtraction run."""

    # Difference image (whitened when decorrelate=True) and its matching
    # per-pixel variance: post-whitening σ_D² after decorrelate, else the
    # propagated Var(D) = Var(S) + (K² ⊗ Var(R)).
    difference: NDArray[np.float32]
    variance: NDArray[np.float32] | None
    mask: NDArray[np.uint8] | None
    score: NDArray[np.float32] | None
    solution: KernelSolution
    # The PSF stamp built for the match filter (None unless `score=True`);
    # carried so `build_catalog()` can derive the expected FWHM without
    # recomputing `_estimate_psf` from raw images, which `DiffResult` no
    # longer has.
    psf: NDArray[np.float32] | None = None
    # Source catalog from the score image (SPEC §3.7), auto-built by default
    # whenever `score=True` (toggle: `Subtractor(source_catalog=False)`). A
    # structured NumPy array (`delta.catalog.CATALOG_DTYPE`), or None when
    # `score` or `source_catalog` was off. Use `build_catalog()` to rebuild
    # with custom thresholds without re-running `subtract()`.
    source_catalog: NDArray | None = None
    # Reproducibility metadata, set by Subtractor.subtract()/apply(): the
    # configuration knobs that aren't already baked into `solution` (see
    # Subtractor.config_cards), and the fit+subtract wall time in seconds.
    config: dict[str, object] = field(default_factory=dict)
    elapsed: float = float("nan")
    # Scalar applied to `variance` (post-subtraction) to force the diff-image
    # reduced chi2 to 1; None when `rescale_variance` was not requested.
    variance_scale: float | None = None

    def header_cards(self) -> dict[str, object]:
        """Full provenance card set: fit + config + environment + runtime.

        Combines :meth:`KernelSolution.header_cards`, the producing
        ``Subtractor``'s :meth:`~Subtractor.config_cards`, and
        :func:`delta._provenance.environment_cards`, enough on its own to
        reproduce this run (modulo the input pixel data itself).
        """
        return {
            **self.solution.header_cards(),
            **self.config,
            **environment_cards(),
            "DLTELAP": self.elapsed,
        }

    def write(
        self,
        path: str,
        overwrite: bool = False,
        *,
        compress: bool = True,
        extra_cards: dict[str, object] | None = None,
    ) -> None:
        """Write products to a multi-extension FITS with provenance (needs astropy).

        With ``compress`` (the default) each image layer is stored as a tile-
        compressed ``CompImageHDU`` (RICE), which shrinks noise-dominated
        difference frames by roughly an order of magnitude. A FITS primary HDU
        cannot itself be compressed, so the provenance header lives in a
        dataless primary and the difference moves to a ``DIFFERENCE`` extension.
        Pass ``compress=False`` for the plain layout (difference in the primary).

        ``extra_cards`` adds/overrides header cards on top of
        :meth:`header_cards` — e.g. input file paths or the invoking command
        line, which this method has no way to know on its own.
        """
        from astropy.io import fits  # optional dependency

        cards = self.header_cards()
        if self.variance_scale is not None:
            cards["DLTVSCL"] = self.variance_scale
        if extra_cards:
            cards.update(extra_cards)
        layers = [("VARIANCE", self.variance), ("MASK", self.mask), ("SCORE", self.score)]
        hdus: list[fits.hdu.base._BaseHDU]
        if compress:
            primary = fits.PrimaryHDU()
            for key, value in cards.items():
                primary.header[key] = value
            hdus = [primary, fits.CompImageHDU(self.difference, name="DIFFERENCE")]
            hdus += [fits.CompImageHDU(a, name=n) for n, a in layers if a is not None]
        else:
            primary = fits.PrimaryHDU(self.difference)
            for key, value in cards.items():
                primary.header[key] = value
            hdus = [primary]
            hdus += [fits.ImageHDU(a, name=n) for n, a in layers if a is not None]
        if self.source_catalog is not None:
            hdus.append(fits.BinTableHDU(self.source_catalog, name="CATALOG"))
        fits.HDUList(hdus).writeto(path, overwrite=overwrite)

    def build_catalog(
        self,
        *,
        threshold_sigma: float = 5.0,
        threshold_sigma_dipole: float = 3.0,
        fwhm_tolerance_lo: float = 0.3,
        fwhm_tolerance_hi: float = 3.0,
        aperture_radius: int = 0,
        exclude_bad_pixels: bool = True,
        as_table: bool = False,
    ):
        """Rebuild the source catalog from `self.score` with custom thresholds
        (SPEC §3.7), instead of the defaults used to auto-populate
        `self.source_catalog`.

        The expected PSF FWHM is measured from `self.psf` (the stamp built for
        the match filter); requires the originating ``subtract()`` /
        ``Subtractor.subtract()`` call to have used ``score=True``.
        """
        if self.score is None or self.psf is None:
            raise ValueError(
                "build_catalog() requires a score image and PSF stamp; pass score=True "
                "to subtract()"
            )
        from .catalog import build_catalog as _build_catalog

        return _build_catalog(
            self.score,
            self.difference,
            mask=self.mask,
            psf=self.psf,
            threshold_sigma=threshold_sigma,
            threshold_sigma_dipole=threshold_sigma_dipole,
            fwhm_tolerance_lo=fwhm_tolerance_lo,
            fwhm_tolerance_hi=fwhm_tolerance_hi,
            aperture_radius=aperture_radius,
            exclude_bad_pixels=exclude_bad_pixels,
            as_table=as_table,
        )


def _default_beta(fwhm_a: float, fwhm_b: float) -> float:
    """Heuristic kernel scale from the two median FWHMs."""
    fwhms = [f for f in (fwhm_a, fwhm_b) if np.isfinite(f) and f > 0]
    if not fwhms:
        return 2.0
    broad, narrow = max(fwhms), min(fwhms)
    sig_broad = broad * _FWHM_TO_SIGMA
    # Width of the Gaussian broadening kernel: K ≈ Gaussian(k_sigma).
    diff = sig_broad**2 - (narrow * _FWHM_TO_SIGMA) ** 2
    k_sigma = np.sqrt(diff) if diff > 0 else 0.0
    # Real PSFs share wing structure that largely cancels in the matching kernel,
    # making the effective kernel narrower than the Gaussian model predicts.
    # Empirically, 0.8 × k_sigma captures the matching kernel scale better than
    # k_sigma itself on realistic ground-based data. The 0.5 px floor prevents
    # degenerate footprints for nearly-matched or undersampled PSF pairs.
    return float(max(0.8 * k_sigma, 0.5))


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
        n_max: int = 6,  # 6 is a robust default; 8 sharpens bright-star cores on
        beta: float | None = None,  # well-sampled frames (overfits sparse ones).
        n_knots: int = 5,
        radius: int = 0,
        stamp_radius: int = 15,
        threshold_sigma: float = 5.0,
        max_stamps: int = 200,
        saturation: float = 0.0,
        match_radius: int = 2,
        fwhm_radius: int = 0,
        bright_mask: float = 0.0,
        lambda_grid: NDArray[np.float64] | None = None,
        clip_sigma: float = 4.0,
        clip_iterations: int = 5,
        min_stamps: int = 5,
        cv_folds: int = 5,
        spatial_scale: float | None = None,
        decorrelate: bool = False,
        score: bool = False,
        source_catalog: bool = True,
        block: int = 256,
        rescale_variance: bool = False,
    ) -> None:
        self.n_max = n_max
        self.beta = beta
        self.n_knots = n_knots
        # Kernel footprint half-width; 0 auto-sizes from beta/n_max. Larger values
        # capture more of the matching-kernel wings (comparable to HOTPANTS' -r,
        # its convolution-kernel half-width).
        self.radius = radius
        self.stamp_radius = stamp_radius
        self.threshold_sigma = threshold_sigma
        self.max_stamps = max_stamps
        # Science/reference stamp cross-match tolerance in pixels; widen this if
        # inputs are astrometrically misregistered, which otherwise fails stamp
        # selection with "no stamps selected" and no obvious knob to reach for.
        self.match_radius = match_radius
        # FWHM moment window; 0 auto-sizes to max(5, stamp_radius/2).
        self.fwhm_radius = fwhm_radius
        # `saturation` is the instrument detector level: genuinely saturated
        # (non-linear) pixels, rejected from stamp selection AND masked in the
        # output. `bright_mask` is a data-driven, output-only level: it masks the
        # cores of bright (but linear) stars whose few-percent PSF-match residual
        # would otherwise dominate the difference, without removing those high-SNR
        # stars from the kernel fit. Both grow by the kernel radius in the output.
        self.saturation = saturation
        self.bright_mask = bright_mask
        self.lambda_grid = (
            np.logspace(-6.0, 6.0, 25, dtype=np.float64)
            if lambda_grid is None
            else np.asarray(lambda_grid, dtype=np.float64)
        )
        self.clip_sigma = clip_sigma
        self.clip_iterations = clip_iterations
        self.min_stamps = min_stamps
        # cv_folds >= 2 selects the spatial smoothing by leave-stamp-out k-fold CV
        # (more robust than GCV, which under-smooths correlated stamp pixels).
        self.cv_folds = cv_folds
        # Physical prior: the kernel varies on scales >> a stamp, so knots are
        # placed no finer than `spatial_scale` pixels apart (a coarse spatial
        # basis can't over-fit small-scale structure). None keeps `n_knots`.
        self.spatial_scale = spatial_scale
        self.decorrelate = decorrelate
        self.score = score
        # Source catalog from the score image (SPEC §3.7), built by default
        # whenever `score` is also on; set False to skip (e.g. for callers
        # who only want the score image itself, or care about latency).
        self.source_catalog = source_catalog
        self.block = block
        # Forces the diff-image reduced chi2 to 1 by rescaling `variance` with a
        # single robust scalar (SPEC §3.4) -- a post-hoc correction for noise
        # models (gain/read-noise, or propagated science/reference variance)
        # that under- or over-estimate the true pixel noise.
        self.rescale_variance = rescale_variance

    def config_cards(self) -> dict[str, object]:
        """Reproducibility cards for knobs not already in KernelSolution.header_cards()."""
        return {
            "DLTSRAD": self.stamp_radius,
            "DLTTHSIG": self.threshold_sigma,
            "DLTMAXST": self.max_stamps,
            "DLTSAT": self.saturation,
            "DLTMRAD": self.match_radius,
            "DLTFRAD": self.fwhm_radius,
            "DLTBMASK": self.bright_mask,
            "DLTCLPSG": self.clip_sigma,
            "DLTCLPIT": self.clip_iterations,
            "DLTMINST": self.min_stamps,
            "DLTCVF": self.cv_folds,
            "DLTSSCL": self.spatial_scale if self.spatial_scale is not None else 0.0,
            "DLTDECOR": self.decorrelate,
            "DLTSCORE": self.score,
            "DLTSCAT": self.source_catalog,
            "DLTBLK": self.block,
        }

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
            match_radius=self.match_radius,
            fwhm_radius=self.fwhm_radius,
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

        logger.info("fitting {} x {} frame", sci.shape[1], sci.shape[0])
        with log_timing("stamp selection"):
            sel = self._select(sci, ref, smask, rmask, catalog)
        n_stamps = int(np.asarray(sel["x"]).size)
        logger.info(
            "selected {} stamps (median FWHM: science={:.2f}, reference={:.2f} px)",
            n_stamps,
            sel["median_fwhm_science"],
            sel["median_fwhm_reference"],
        )
        if n_stamps == 0:
            raise ValueError(
                "no stamps selected; lower threshold_sigma, supply a catalog, or "
                "widen match_radius if science/reference are astrometrically "
                "misregistered"
            )
        if direction is None:
            direction = sel["direction"]
        elif direction not in ("science", "reference"):
            raise ValueError(
                f"direction must be 'science', 'reference', or None, got {direction!r}"
            )
        logger.info("convolution direction: {}", direction)

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
            logger.debug("auto-selected basis scale beta={:.3f}", beta)
        logger.info(
            "kernel basis: beta={:.3f}, n_max={} ({} components), {}^2 knots",
            beta,
            self.n_max,
            (self.n_max + 1) * (self.n_max + 2) // 2,
            self.n_knots,
        )

        h, w = sci.shape
        # Knot grid: honour `spatial_scale` (don't place knots finer than the
        # expected variation scale) so the spatial basis stays coarse.
        nkx, nky = self.n_knots, self.n_knots
        if self.spatial_scale is not None and self.spatial_scale > 0:
            nkx = int(np.clip(round((w - 1) / self.spatial_scale) + 1, 3, self.n_knots))
            nky = int(np.clip(round((h - 1) / self.spatial_scale) + 1, 3, self.n_knots))
            logger.debug("spatial_scale={:.0f}px -> {}x{} knots", self.spatial_scale, nkx, nky)
        knots = _core.grid_knots(0.0, 0.0, w - 1.0, h - 1.0, nkx, nky)
        stamp_x = np.ascontiguousarray(sel["x"], dtype=np.int32)
        stamp_y = np.ascontiguousarray(sel["y"], dtype=np.int32)

        with log_timing("kernel solve"):
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
                radius=self.radius,
                clip_sigma=self.clip_sigma,
                clip_iterations=self.clip_iterations,
                min_stamps=self.min_stamps,
                cv_folds=self.cv_folds,
                science_var=tvar,
                reference_var=cvar,
                science_mask=tmask,
                reference_mask=cmask,
            )
        _log_core_timings(fit.get("timing"))
        selector = f"{self.cv_folds}-fold CV" if self.cv_folds > 1 else "GCV"
        logger.info(
            "kernel fit: lambda={:.3g} ({}), eff.dof={:.1f}, "
            "reduced chi2={:.3f} ({} stamps used, {} rejected of {}, {} px)",
            fit["lambda"],
            selector,
            fit["effective_dof"],
            fit["reduced_chi2"],
            fit["n_stamps_used"],
            fit["n_stamps_rejected"],
            fit["n_stamps_total"],
            fit["n_pixels"],
        )
        lam_grid = np.asarray(fit["lambda_grid"])
        if lam_grid.size > 1 and fit["lambda"] in (lam_grid[0], lam_grid[-1]):
            edge = "lower" if fit["lambda"] == lam_grid[0] else "upper"
            logger.warning(
                "selected lambda={:.3g} ({}) sits at the {} edge of lambda_grid "
                "[{:.3g}, {:.3g}] -- the search may be under-constrained; consider "
                "widening lambda_grid to confirm this isn't clipping the true "
                "optimum (under-smoothing at the lower edge, over-smoothing at "
                "the upper edge).",
                fit["lambda"],
                selector,
                edge,
                lam_grid[0],
                lam_grid[-1],
            )
        design_bytes = fit.get("cv_exact_design_bytes", 0)
        if design_bytes > _CV_EXACT_DESIGN_WARN_BYTES:
            logger.warning(
                "exact (non-stamped) k-fold CV path used a {:.2f} GB whitened "
                "design, rebuilt every IRLS pass -- the 16x knot-spacing gate to "
                "the cheaper stamped path was not met (fine knots relative to "
                "stamp_radius). Consider fewer/coarser knots, a larger "
                "spatial_scale, or a smaller stamp_radius if this is slow.",
                design_bytes / 1e9,
            )
        if fit["reduced_chi2"] > 3.0:
            logger.warning(
                "kernel fit reduced chi2={:.2f} >> 1: stellar residuals likely. "
                "Check registration (dipoles), basis (n_max/beta), or knots.",
                fit["reduced_chi2"],
            )
        solution = KernelSolution(
            beta=beta,
            n_max=self.n_max,
            radius=self.radius,
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
            reduced_chi2=fit["reduced_chi2"],
            n_stamps_used=fit["n_stamps_used"],
            n_stamps_rejected=fit["n_stamps_rejected"],
            stamp_x=fit["stamp_x"],
            stamp_y=fit["stamp_y"],
            stamp_chi2=fit["stamp_chi2"],
            stamp_accepted=fit["stamp_accepted"],
        )
        layers = (target, conv, tvar, cvar, tmask, cmask, sign, stamp_x, stamp_y)
        return solution, layers

    def _coerce_apply_layers(
        self,
        solution,
        science,
        reference,
        *,
        science_var,
        reference_var,
        science_mask,
        reference_mask,
        gain,
        read_noise,
    ):
        """Build the apply-path ``layers`` tuple from a frozen ``solution``.

        Mirrors the input coercion and direction/sign assignment in :meth:`_fit`,
        but takes the convolution direction from ``solution`` instead of selecting
        stamps. ``stamp_x``/``stamp_y`` come from the solution's accepted stamps
        (used only for matched-filter PSF estimation).
        """
        sci, svar, smask = as_layers(science, science_var, science_mask)
        ref, rvar, rmask = as_layers(reference, reference_var, reference_mask)
        if sci.shape != ref.shape:
            raise ValueError("science and reference must share a shape")
        if sci.shape != solution.shape:
            raise ValueError(
                f"solution shape {solution.shape} does not match input frame shape {sci.shape}"
            )
        if svar is None and gain is not None:
            svar = synth_variance(sci, gain, read_noise)
        if rvar is None and gain is not None:
            rvar = synth_variance(ref, gain, read_noise)

        # Same convention as `_fit`: direction == "science" convolves science to
        # match reference (the target) and flips the difference sign so transients
        # stay positive; otherwise reference is convolved to match science.
        if solution.direction == "science":
            target, tvar, tmask = ref, rvar, rmask
            conv, cvar, cmask = sci, svar, smask
            sign = -1.0
        else:
            target, tvar, tmask = sci, svar, smask
            conv, cvar, cmask = ref, rvar, rmask
            sign = 1.0

        stamp_x = solution.stamp_x
        stamp_y = solution.stamp_y
        return (target, conv, tvar, cvar, tmask, cmask, sign, stamp_x, stamp_y)

    def _subtract(self, solution, layers) -> DiffResult:
        """Run the full-frame subtraction (+ optional decorrelate/score) for a
        given ``solution`` and coerced ``layers``. Shared by :meth:`subtract`
        (fit-then-apply) and :meth:`apply` (frozen solution)."""
        target, conv, tvar, cvar, tmask, cmask, sign, stamp_x, stamp_y = layers

        # Output bright-core mask threshold: the lowest positive of the
        # instrument saturation and the data-level bright mask (stamp selection
        # already used `saturation` alone, so bright_mask never culls the fit).
        out_levels = [v for v in (self.saturation, self.bright_mask) if v and v > 0]
        out_saturation = min(out_levels) if out_levels else 0.0
        with log_timing("full-frame subtraction"):
            diff = _core.subtract(
                target,
                conv,
                solution.knots,
                solution.theta,
                solution.beta,
                solution.n_max,
                radius=solution.radius,
                saturation=out_saturation,
                science_var=tvar,
                reference_var=cvar,
                science_mask=tmask,
                reference_mask=cmask,
            )
        _log_core_timings(diff.get("timing"))
        # diff["difference"] is already a fresh float32 array owned by this
        # result; the sign flip (transients-positive convention) is the only
        # transform needed. Negate in place when flipping and alias otherwise --
        # the old `(sign * arr).astype(float32)` allocated a full-frame temporary
        # and re-cast an already-float32 array on every call (and copied for free
        # when sign == +1).
        difference = diff["difference"]
        if sign < 0:
            np.negative(difference, out=difference)
        variance = diff["variance"]
        mask = diff["mask"]
        variance_scale = None
        if self.rescale_variance and variance is not None:
            variance_scale = _variance_rescale_factor(difference, variance, mask)
            variance = variance * variance_scale
            logger.info(
                "variance rescaled by {:.4f} to force diff-image reduced chi2=1",
                variance_scale,
            )
        elif self.rescale_variance:
            logger.warning("variance rescale requested but no variance available; skipping")
        # rms / masked-fraction are log-only and span the full frame (np.std over
        # 64 Mpix is ~0.4s); evaluate them lazily so the silent-by-default library
        # never pays for a message no sink will emit.
        if mask is not None:
            logger.opt(lazy=True).info(
                "difference: rms={rms:.4g}, {n} masked px ({pct:.2f}%)",
                rms=lambda: float(np.std(difference)),
                n=lambda: int(np.count_nonzero(mask)),
                pct=lambda: 100.0 * int(np.count_nonzero(mask)) / difference.size,
            )
        else:
            logger.opt(lazy=True).info(
                "difference: rms={rms:.4g}", rms=lambda: float(np.std(difference))
            )

        whitened = difference
        white_var = None
        if self.decorrelate:
            if tvar is not None and cvar is not None:
                with log_timing("noise decorrelation"):
                    dec = _core.decorrelate(
                        difference,
                        solution.knots,
                        solution.theta,
                        solution.beta,
                        solution.n_max,
                        tvar,
                        cvar,
                        block=self.block,
                    )
                whitened = dec["difference"]
                white_var = dec["variance"]
                _log_core_timings(dec.get("timing"))
                # Propagate any raw-Var(D) rescale onto the whitened noise level:
                # vs/vr mis-scaling that forced variance_scale on Var(D) applies
                # equally to σ_D² = vs + vr ΣK².
                if variance_scale is not None:
                    # In place: white_var is the C++-owned buffer moved out of
                    # decorrelate, not caller-visible, so no full-frame
                    # temporary is needed (unlike the input `variance` above,
                    # which may view the caller's array).
                    white_var *= variance_scale
            else:
                logger.warning("decorrelation requested but no variance available; skipping")

        score = None
        psf = None
        if self.score:
            with log_timing("matched-filter score"):
                if stamp_x is None or np.asarray(stamp_x).size == 0:
                    raise ValueError(
                        "score requested but the solution carries no stamp positions "
                        "for PSF estimation"
                    )
                psf_radius = min(self.stamp_radius, 8)
                psf = _estimate_psf(target, stamp_x, stamp_y, psf_radius)
                if white_var is not None:
                    # Whitened difference: normalise with post-whitening variance
                    # and the Phi-filtered PSF (ZOGY score profile).
                    score_var = np.asarray(white_var, dtype=np.float32)
                    if tvar is not None and cvar is not None:
                        psf = _core.whiten_score_psf(
                            psf,
                            solution.knots,
                            solution.theta,
                            solution.beta,
                            solution.n_max,
                            tvar,
                            cvar,
                            self.block,
                            radius=solution.radius,
                        )
                elif variance is not None:
                    score_var = np.asarray(variance, dtype=np.float32)
                else:
                    fallback = float(np.var(whitened))
                    score_var = np.full(whitened.shape, fallback, dtype=np.float32)
                score = _core.matched_filter(whitened, psf, score_var)
            # matched_filter returns a bare array, so drain its DELTA_TIMING
            # sub-stage timers explicitly (fit/subtract/decorrelate attach
            # theirs to their result dicts).
            _log_core_timings(_core.drain_timing())

        out_diff = whitened
        # When decorrelated, return the post-whitening variance so difference and
        # variance describe the same product (pulls D/√V stay valid).
        out_var = white_var if white_var is not None else variance

        source_catalog = None
        if score is not None and self.source_catalog:
            with log_timing("catalog"):
                from .catalog import build_catalog as _build_catalog

                source_catalog = _build_catalog(score, out_diff, mask=mask, psf=psf)

        return DiffResult(
            difference=out_diff,
            variance=out_var,
            mask=mask,
            score=score,
            solution=solution,
            psf=psf,
            source_catalog=source_catalog,
            variance_scale=variance_scale,
        )

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
        start = time.perf_counter()
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
        result = self._subtract(solution, layers)
        result.config = self.config_cards()
        result.elapsed = time.perf_counter() - start
        return result

    def apply(
        self,
        solution: KernelSolution,
        science,
        reference,
        *,
        science_var=None,
        reference_var=None,
        science_mask=None,
        reference_mask=None,
        gain=None,
        read_noise=0.0,
    ) -> DiffResult:
        """Apply a frozen ``solution`` to a new pair without re-fitting.

        Runs only the full-frame subtraction (+ optional decorrelation/score)
        using the supplied :class:`KernelSolution` — the convolution direction,
        kernel basis, knots and coefficients all come from ``solution``. The
        inputs must match ``solution.shape``. Use this to reuse one fit across a
        stack of frames, or to re-apply a saved solution for QA/reproducibility.
        """
        start = time.perf_counter()
        layers = self._coerce_apply_layers(
            solution,
            science,
            reference,
            science_var=science_var,
            reference_var=reference_var,
            science_mask=science_mask,
            reference_mask=reference_mask,
            gain=gain,
            read_noise=read_noise,
        )
        result = self._subtract(solution, layers)
        result.config = self.config_cards()
        result.elapsed = time.perf_counter() - start
        return result


def subtract(science, reference, **kwargs) -> DiffResult:
    """Convenience wrapper: configure a :class:`Subtractor` and run it once.

    Configuration keywords (``n_max``, ``beta``, ``n_knots``, ``stamp_radius``,
    ``threshold_sigma``, ``max_stamps``, ``saturation``, ``bright_mask``,
    ``lambda_grid``, ``clip_sigma``, ``clip_iterations``, ``min_stamps``, ``cv_folds``,
    ``spatial_scale``, ``decorrelate``, ``score``, ``block``, ``rescale_variance``)
    are split from the per-call inputs
    (``science_var``, ``reference_var``, ``science_mask``, ``reference_mask``,
    ``gain``, ``read_noise``, ``catalog``, ``direction``).
    """
    config = {k: v for k, v in kwargs.items() if k in _CONFIG_KEYS}
    call = {k: v for k, v in kwargs.items() if k not in _CONFIG_KEYS}
    return Subtractor(**config).subtract(science, reference, **call)


def apply(solution: KernelSolution, science, reference, **kwargs) -> DiffResult:
    """Convenience wrapper: apply a frozen ``solution`` without re-fitting.

    Configuration keywords (a subset of those accepted by :func:`subtract`:
    ``saturation``, ``bright_mask``, ``stamp_radius``, ``decorrelate``, ``score``,
    ``block`` — the fit-only knobs are ignored) are split from the per-call inputs
    (``science_var``, ``reference_var``, ``science_mask``, ``reference_mask``,
    ``gain``, ``read_noise``). The convolution direction, kernel basis, knots and
    coefficients all come from ``solution``; the inputs must match
    ``solution.shape``. See :meth:`Subtractor.apply`.
    """
    config = {k: v for k, v in kwargs.items() if k in _CONFIG_KEYS}
    call = {k: v for k, v in kwargs.items() if k not in _CONFIG_KEYS}
    return Subtractor(**config).apply(solution, science, reference, **call)
