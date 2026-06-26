#pragma once

#include <cstdint>
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
  std::uint8_t quality = 0;    // bit0: !fwhm_consistent, bit1: is_dipole,
                               // bit2: mask_flags != 0
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
};

// Connected-component source catalog from a match-filtered score image
// (SPEC §3.7). `score` is the per-pixel S/N map (DiffResult.score);
// `difference` supplies the flux measurement. Positive blobs are seeded and
// grown (8-connected) at `score >= threshold_sigma`; negative blobs are
// independently labelled at `score <= -threshold_sigma_dipole` purely to
// flag dipoles -- positive components that enclose or directly touch a
// negative component are marked `is_dipole` (bad-subtraction evidence, not a
// transient). When `score.has_mask()`, pixels with `mask != kMaskGood` are
// excluded from seeding/growth entirely if `params.exclude_bad_pixels`
// (the default); `mask_flags` still reports the bitwise OR of mask bits
// touched by the footprint's 8-connected boundary, so a clean candidate
// adjacent to bad pixels remains visible. Returns positive-blob entries only
// unless `return_negative`, in which case the negative-blob catalog (at the
// dipole threshold) is returned instead.
std::vector<CatalogEntry> build_catalog(const ImageF& score,
                                        const ImageF& difference,
                                        const CatalogParams& params,
                                        bool return_negative = false);

}  // namespace delta
