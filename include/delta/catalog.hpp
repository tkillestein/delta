#pragma once

#include <cstdint>
#include <limits>
#include <vector>

#include "delta/image.hpp"

namespace delta {

// A single connected-component catalog entry, built from the match-filtered
// score image (SPEC §3.7).
struct CatalogEntry {
  double x = 0.0;             // flux-weighted centroid column (on |score|)
  double y = 0.0;             // flux-weighted centroid row
  int peak_x = 0;             // integer peak-pixel column
  int peak_y = 0;             // integer peak-pixel row
  double peak_snr = 0.0;      // signed score value at the peak pixel
  int n_pix = 0;               // footprint pixel count
  double flux = 0.0;           // aperture flux on `difference`
  double expected_n_pix = 0.0; // predicted pixel count from expected_fwhm
  double fwhm_ratio = 0.0;     // n_pix / expected_n_pix (1.0 == consistent)
  bool fwhm_consistent = false;
  std::uint8_t mask_flags = 0; // OR of MaskFlag over the footprint + boundary
  bool is_dipole = false;      // an enclosed/touching opposite-polarity blob

  // PSF-shape chi^2 fit (opt-in, CatalogParams::fit_psf_shape; needs
  // CatalogAux::psf + CatalogAux::variance). A 1-parameter weighted linear fit
  // of the PSF stamp against the pixels around the peak: amplitude and its
  // formal error come from the same normal equations as the chi^2, so all
  // four fields are a byproduct of one small windowed pass, not separate
  // work. Left at these defaults (chi2=0, dof=0, consistent=true) when the
  // fit did not run -- absence of evidence, not evidence of a good fit.
  double psf_chi2 = 0.0;
  int psf_chi2_dof = 0;
  double psf_amplitude = 0.0;
  double psf_amplitude_err = 0.0;
  bool shape_consistent = true; // true when not fit, or psf_chi2/dof <= tolerance

  // Bright-star proximity (opt-in; needs CatalogAux::bright_x/bright_y).
  // Subtraction residuals (imperfect kernel match, diffraction spikes, bleed)
  // cluster near bright stars beyond what `mask_flags`'s footprint+boundary
  // check reaches. This is a flag, not a rejection -- real transients do
  // occur near bright stars.
  double bright_star_dist = std::numeric_limits<double>::infinity();
  bool near_bright_star = false;

  std::uint8_t quality = 0;    // bit0: !fwhm_consistent, bit1: is_dipole,
                               // bit2: mask_flags != 0, bit3: !shape_consistent
                               // (only meaningful when fit_psf_shape ran),
                               // bit4: near_bright_star
};

// Tuning for catalog construction (SPEC §3.7).
struct CatalogParams {
  double threshold_sigma = 5.0;        // positive-blob seed/grow threshold
  double threshold_sigma_dipole = 3.0; // |score| seed/grow threshold for the
                                       // negative blobs used in dipole checks
  double expected_fwhm = 3.5;          // PSF FWHM (px) for the consistency
                                       // filter and default aperture radius
  double fwhm_tolerance_lo = 0.3;      // fwhm_ratio lower bound for "consistent"
  double fwhm_tolerance_hi = 3.0;      // fwhm_ratio upper bound for "consistent"
  int aperture_radius = 0;             // 0 -> resolve from expected_fwhm
  bool exclude_bad_pixels = true;      // exclude masked pixels from growth

  // Per-candidate enrichment (opt-in; see CatalogAux). Both only ever run on
  // the already-thresholded candidate list -- O(components), not O(frame) --
  // so their cost tracks detection count, not image size.
  bool fit_psf_shape = false;       // PSF chi^2 fit (needs aux.psf + aux.variance)
  double shape_chi2_tolerance = 3.0; // reduced-chi^2 cutoff for shape_consistent
  double bright_star_radius = 0.0;   // 0 -> 3 * expected_fwhm

  // Pathological-frame guard: a mis-set threshold_sigma (or a genuinely
  // broken frame) can turn "hundreds of candidates" into "hundreds of
  // thousands of one-pixel blobs". Per-candidate enrichment (shape fit +
  // bright-star distance) is skipped above this many positive components --
  // the base catalog (seed/label/summarize) is unaffected and always runs.
  int max_shape_fit = 5000;
};

// Optional auxiliary inputs for per-candidate enrichment (SPEC §3.7
// extension). Each piece is independently optional: `psf` + `variance`
// drive the PSF-shape chi^2 fit (CatalogParams::fit_psf_shape); `bright_x` +
// `bright_y` drive the bright-star proximity flag. Absent inputs leave the
// corresponding CatalogEntry fields at their defaults.
struct CatalogAux {
  const ImageF* variance = nullptr;        // per-pixel Var(difference)
  const std::vector<float>* psf = nullptr; // square PSF stamp, side psf_size
                                           // (the same profile matched_filter
                                           // scored with -- already whitened
                                           // when decorrelation was used)
  int psf_size = 0;
  const std::vector<double>* bright_x = nullptr; // known bright-star positions
  const std::vector<double>* bright_y = nullptr; // (e.g. KernelSolution.stamp_*)
};

// Connected-component source catalog from a match-filtered score image
// (SPEC §3.7). `score` is the per-pixel S/N map (DiffResult.score);
// `difference` supplies the flux measurement. Positive blobs are seeded and
// grown (8-connected) at `score >= threshold_sigma`; negative blobs are
// independently labelled at `score <= -threshold_sigma_dipole` purely to
// flag dipoles -- components of either polarity that enclose or directly
// touch an opposite-polarity component are marked `is_dipole` symmetrically
// (bad-subtraction evidence, not a transient). When `score.has_mask()`,
// pixels with `mask != kMaskGood` are excluded from seeding/growth entirely
// if `params.exclude_bad_pixels` (the default); `mask_flags` still reports
// the bitwise OR of mask bits touched by the footprint's 8-connected
// boundary, so a clean candidate adjacent to bad pixels remains visible.
// Returns positive-blob entries (seeded at `threshold_sigma`) unless
// `return_negative`, in which case the negative-blob catalog (seeded at
// `threshold_sigma_dipole`, not `threshold_sigma` -- it reuses the dipole
// pass rather than a third threshold) is returned instead.
//
// `aux` carries optional per-candidate enrichment inputs (PSF-shape chi^2,
// bright-star proximity -- see CatalogAux); omitting it (or leaving fields
// null) reproduces the base catalog exactly, at the base cost.
std::vector<CatalogEntry> build_catalog(const ImageF& score,
                                        const ImageF& difference,
                                        const CatalogParams& params,
                                        bool return_negative = false,
                                        const CatalogAux& aux = {});

}  // namespace delta
