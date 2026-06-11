#include "delta/detect.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace delta {
namespace {

constexpr double kFwhmPerSigma = 2.354820045030949;  // 2*sqrt(2 ln 2)

bool is_finite(float v) { return std::isfinite(v); }

double median_of(std::vector<float>& v) {
  if (v.empty()) return 0.0;
  const std::size_t mid = v.size() / 2;
  std::nth_element(v.begin(), v.begin() + mid, v.end());
  return static_cast<double>(v[mid]);
}

double median_of(std::vector<double> v) {
  if (v.empty()) return std::numeric_limits<double>::quiet_NaN();
  const std::size_t mid = v.size() / 2;
  std::nth_element(v.begin(), v.begin() + mid, v.end());
  return v[mid];
}

int resolved_border(const DetectParams& p) {
  return p.border > 0 ? p.border : p.stamp_radius;
}
int resolved_isolation(const DetectParams& p) {
  return p.isolation_radius > 0 ? p.isolation_radius : 2 * p.stamp_radius;
}
int resolved_fwhm_radius(const DetectParams& p) {
  return p.fwhm_radius > 0 ? p.fwhm_radius
                           : std::max(5, p.stamp_radius / 2);
}

// True if the stamp footprint around (x,y) is fully on-frame, unmasked, and
// (when a saturation level is set) unsaturated.
bool footprint_ok(const ImageF& img, int x, int y, int radius,
                  double saturation) {
  const int w = static_cast<int>(img.width());
  const int h = static_cast<int>(img.height());
  if (x - radius < 0 || x + radius >= w || y - radius < 0 || y + radius >= h)
    return false;

  const bool has_mask = img.has_mask();
  const bool check_sat = saturation > 0.0;
  for (int dy = -radius; dy <= radius; ++dy) {
    const std::size_t row = static_cast<std::size_t>(y + dy) * w;
    for (int dx = -radius; dx <= radius; ++dx) {
      const std::size_t idx = row + (x + dx);
      const float v = img.data()[idx];
      if (!is_finite(v)) return false;
      if (has_mask && img.mask()[idx] != kMaskGood) return false;
      if (check_sat && static_cast<double>(v) >= saturation) return false;
    }
  }
  return true;
}

}  // namespace

const char* to_string(ConvolveDirection direction) {
  return direction == ConvolveDirection::kConvolveReference ? "reference"
                                                            : "science";
}

BackgroundStats estimate_background(const ImageF& image) {
  const std::size_t n = image.size();
  const bool has_mask = image.has_mask();

  std::vector<float> good;
  good.reserve(n);
  for (std::size_t i = 0; i < n; ++i) {
    const float v = image.data()[i];
    if (!is_finite(v)) continue;
    if (has_mask && image.mask()[i] != kMaskGood) continue;
    good.push_back(v);
  }
  if (good.empty()) return {0.0, 0.0};

  const double med = median_of(good);
  for (float& v : good) v = std::fabs(v - static_cast<float>(med));
  const double mad = median_of(good);
  return {med, 1.4826 * mad};
}

double estimate_fwhm(const ImageF& image, int x, int y, int radius,
                     double background) {
  const int w = static_cast<int>(image.width());
  const int h = static_cast<int>(image.height());

  double sum = 0.0, sx = 0.0, sy = 0.0;
  for (int dy = -radius; dy <= radius; ++dy) {
    const int yy = y + dy;
    if (yy < 0 || yy >= h) continue;
    for (int dx = -radius; dx <= radius; ++dx) {
      const int xx = x + dx;
      if (xx < 0 || xx >= w) continue;
      const double f =
          static_cast<double>(image.data()[static_cast<std::size_t>(yy) * w + xx]) -
          background;
      const double weight = f > 0.0 ? f : 0.0;
      sum += weight;
      sx += weight * dx;
      sy += weight * dy;
    }
  }
  if (sum <= 0.0) return std::numeric_limits<double>::quiet_NaN();

  const double mx = sx / sum, my = sy / sum;
  double sxx = 0.0, syy = 0.0;
  for (int dy = -radius; dy <= radius; ++dy) {
    const int yy = y + dy;
    if (yy < 0 || yy >= h) continue;
    for (int dx = -radius; dx <= radius; ++dx) {
      const int xx = x + dx;
      if (xx < 0 || xx >= w) continue;
      const double f =
          static_cast<double>(image.data()[static_cast<std::size_t>(yy) * w + xx]) -
          background;
      const double weight = f > 0.0 ? f : 0.0;
      sxx += weight * (dx - mx) * (dx - mx);
      syy += weight * (dy - my) * (dy - my);
    }
  }
  const double var = 0.5 * (sxx / sum + syy / sum);
  if (var <= 0.0) return std::numeric_limits<double>::quiet_NaN();
  return kFwhmPerSigma * std::sqrt(var);
}

std::vector<Stamp> detect_stamps(const ImageF& image,
                                 const DetectParams& params) {
  const int w = static_cast<int>(image.width());
  const int h = static_cast<int>(image.height());
  const BackgroundStats bg = estimate_background(image);
  const double threshold =
      bg.median + params.threshold_sigma * (bg.sigma > 0.0 ? bg.sigma : 1.0);
  const int border = resolved_border(params);
  const int fwhm_radius = resolved_fwhm_radius(params);

  // 3x3 local maxima above threshold, with a valid stamp footprint.
  std::vector<Stamp> candidates;
  for (int y = border; y < h - border; ++y) {
    for (int x = border; x < w - border; ++x) {
      const double v = static_cast<double>(image.data()[static_cast<std::size_t>(y) * w + x]);
      if (v <= threshold) continue;

      bool peak = true;
      for (int dy = -1; dy <= 1 && peak; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
          if (dx == 0 && dy == 0) continue;
          const double nv = static_cast<double>(
              image.data()[static_cast<std::size_t>(y + dy) * w + (x + dx)]);
          if (nv > v) { peak = false; break; }
        }
      }
      if (!peak) continue;
      if (!footprint_ok(image, x, y, params.stamp_radius, params.saturation))
        continue;

      Stamp s;
      s.x = x;
      s.y = y;
      s.flux = v - bg.median;
      s.snr = bg.sigma > 0.0 ? s.flux / bg.sigma : s.flux;
      s.fwhm = estimate_fwhm(image, x, y, fwhm_radius, bg.median);
      candidates.push_back(s);
    }
  }

  // Brightest-first greedy isolation (non-maximum suppression).
  std::sort(candidates.begin(), candidates.end(),
            [](const Stamp& a, const Stamp& b) { return a.flux > b.flux; });
  const long iso2 = static_cast<long>(resolved_isolation(params)) *
                    resolved_isolation(params);

  std::vector<Stamp> kept;
  for (const Stamp& c : candidates) {
    if (static_cast<int>(kept.size()) >= params.max_stamps) break;
    bool isolated = true;
    for (const Stamp& k : kept) {
      const long ddx = c.x - k.x;
      const long ddy = c.y - k.y;
      if (ddx * ddx + ddy * ddy < iso2) { isolated = false; break; }
    }
    if (isolated) kept.push_back(c);
  }
  return kept;
}

ConvolveDirection choose_direction(double fwhm_science, double fwhm_reference) {
  return fwhm_reference <= fwhm_science ? ConvolveDirection::kConvolveReference
                                        : ConvolveDirection::kConvolveScience;
}

namespace {

StampSelection finalize_selection(StampSelection sel) {
  sel.median_fwhm_science = median_of(sel.fwhm_science);
  sel.median_fwhm_reference = median_of(sel.fwhm_reference);
  sel.direction =
      choose_direction(sel.median_fwhm_science, sel.median_fwhm_reference);
  return sel;
}

}  // namespace

StampSelection select_stamps(const ImageF& science, const ImageF& reference,
                             const DetectParams& params) {
  const std::vector<Stamp> det_sci = detect_stamps(science, params);
  const std::vector<Stamp> det_ref = detect_stamps(reference, params);
  const long tol2 =
      static_cast<long>(params.match_radius) * params.match_radius;

  StampSelection sel;
  for (const Stamp& r : det_ref) {
    const Stamp* best = nullptr;
    long best_d2 = tol2 + 1;
    for (const Stamp& s : det_sci) {
      const long ddx = r.x - s.x;
      const long ddy = r.y - s.y;
      const long d2 = ddx * ddx + ddy * ddy;
      if (d2 <= tol2 && d2 < best_d2) {
        best_d2 = d2;
        best = &s;
      }
    }
    if (best == nullptr) continue;
    sel.x.push_back(r.x);
    sel.y.push_back(r.y);
    sel.fwhm_science.push_back(best->fwhm);
    sel.fwhm_reference.push_back(r.fwhm);
  }
  return finalize_selection(std::move(sel));
}

StampSelection select_stamps_from_catalog(const ImageF& science,
                                          const ImageF& reference,
                                          const std::vector<int>& xs,
                                          const std::vector<int>& ys,
                                          const DetectParams& params) {
  const BackgroundStats bg_sci = estimate_background(science);
  const BackgroundStats bg_ref = estimate_background(reference);
  const int fwhm_radius = resolved_fwhm_radius(params);

  StampSelection sel;
  const std::size_t n = std::min(xs.size(), ys.size());
  for (std::size_t i = 0; i < n; ++i) {
    const int x = xs[i], y = ys[i];
    if (!footprint_ok(science, x, y, params.stamp_radius, params.saturation))
      continue;
    if (!footprint_ok(reference, x, y, params.stamp_radius, params.saturation))
      continue;
    sel.x.push_back(x);
    sel.y.push_back(y);
    sel.fwhm_science.push_back(
        estimate_fwhm(science, x, y, fwhm_radius, bg_sci.median));
    sel.fwhm_reference.push_back(
        estimate_fwhm(reference, x, y, fwhm_radius, bg_ref.median));
  }
  return finalize_selection(std::move(sel));
}

}  // namespace delta
