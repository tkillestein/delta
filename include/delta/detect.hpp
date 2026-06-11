#pragma once

#include <vector>

#include "delta/image.hpp"

namespace delta {

// A candidate PSF-matching stamp: a bright, isolated, unsaturated point source.
struct Stamp {
  int x = 0;             // peak pixel column (footprint centre)
  int y = 0;             // peak pixel row
  double flux = 0.0;     // background-subtracted peak value
  double snr = 0.0;      // peak / background noise
  double fwhm = 0.0;     // estimated PSF FWHM in pixels (NaN if unavailable)
};

// Robust background level and noise from the median / MAD of good pixels.
struct BackgroundStats {
  double median = 0.0;
  double sigma = 0.0;  // 1.4826 * MAD
};

// Which image gets convolved to match the other (SPEC §3.5): always the
// narrower-PSF (sharper) image, broadened to the wider one's seeing.
enum class ConvolveDirection {
  kConvolveReference,  // reference is sharper -> convolve it to science seeing
  kConvolveScience,    // science is sharper   -> convolve it to reference seeing
};

const char* to_string(ConvolveDirection direction);

// Detection / selection tuning. Zero-valued knobs resolve to defaults derived
// from `stamp_radius` (see detect.cpp).
struct DetectParams {
  int stamp_radius = 15;        // half-width of the stamp footprint
  double threshold_sigma = 5.0; // peak detection threshold above background
  int max_stamps = 200;         // cap on returned stamps (brightest kept)
  double saturation = 0.0;      // reject footprints with pixels >= this (<=0: off)
  int isolation_radius = 0;     // min separation between stamps (0 -> 2*stamp_radius)
  int border = 0;               // edge exclusion (0 -> stamp_radius)
  int fwhm_radius = 0;          // FWHM moment window (0 -> max(5, stamp_radius/2))
  int match_radius = 2;         // science/reference cross-match tolerance (px)
};

// The outcome of selecting matched stamps across both images.
struct StampSelection {
  std::vector<int> x;
  std::vector<int> y;
  std::vector<double> fwhm_science;    // per stamp
  std::vector<double> fwhm_reference;  // per stamp
  double median_fwhm_science = 0.0;
  double median_fwhm_reference = 0.0;
  ConvolveDirection direction = ConvolveDirection::kConvolveReference;
};

// Robust background statistics over unmasked, finite pixels.
BackgroundStats estimate_background(const ImageF& image);

// Estimate PSF FWHM at (x, y) from flux-weighted second moments of
// (value - background) within a window of the given radius. Returns NaN when
// the window carries no positive flux.
double estimate_fwhm(const ImageF& image, int x, int y, int radius,
                     double background);

// Detect candidate stamps: 3x3 local maxima above threshold, edge-safe,
// unsaturated/unmasked over the footprint, and isolated (greedy brightest-first
// suppression within isolation_radius). Returns stamps sorted by descending SNR.
std::vector<Stamp> detect_stamps(const ImageF& image, const DetectParams& params);

// Decide the convolution direction from two median FWHM values.
ConvolveDirection choose_direction(double fwhm_science, double fwhm_reference);

// Detect on both images and cross-match to keep sources present in both,
// estimating per-image FWHM and the resulting convolution direction.
StampSelection select_stamps(const ImageF& science, const ImageF& reference,
                             const DetectParams& params);

// As above but using caller-supplied (x, y) positions instead of detection
// (SPEC §3.5): each position is validated (edge/mask/saturation) in both images.
StampSelection select_stamps_from_catalog(const ImageF& science,
                                          const ImageF& reference,
                                          const std::vector<int>& xs,
                                          const std::vector<int>& ys,
                                          const DetectParams& params);

}  // namespace delta
