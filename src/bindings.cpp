#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <vector>

#include "delta/basis.hpp"
#include "delta/catalog.hpp"
#include "delta/convolve.hpp"
#include "delta/detect.hpp"
#include "delta/fit.hpp"
#include "delta/image.hpp"
#include "delta/io.hpp"
#include "delta/noise.hpp"
#include "delta/solve.hpp"
#include "delta/spatial.hpp"
#include "delta/subtract.hpp"
#include "delta/timing.hpp"
#include "delta/version.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace {

// Dimension contract: the C++ core indexes pixels with std::size_t, but image
// *axis extents* (width/height, kernel/psf side) are carried as `int` throughout
// the convolution / variance / FFT loops. Each axis must therefore fit in `int`
// (< 2^31 ~ 2.1e9 px per side). That bounds a single image axis well above any
// realistic detector/mosaic dimension; total pixel counts beyond 2^31 are fine
// (they use std::size_t). Callers handing in a single axis near INT_MAX are out of
// contract and would narrow on the static_cast<int>(shape(...)) below.
template <typename T>
using InArray =
    nb::ndarray<const T, nb::ndim<2>, nb::c_contig, nb::device::cpu>;

template <typename T>
using InArray1D =
    nb::ndarray<const T, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

// Build an ImageF from a 2-D float array and an optional bad-pixel mask.
//
// Uses ImageF's range constructor (one allocate+copy pass) rather than
// `ImageF(w,h)` followed by a fill: the latter's zero-initialisation is
// wasted work the caller immediately overwrites, and at survey-cadence sizes
// (8000x6000, ~48M px) that redundant pass was measured (DELTA_TIMING) to
// cost a third of build_catalog's total. Likewise `mask().assign(...)`
// (single pass) instead of `allocate_mask()` (zero-init) + a fill.
delta::ImageF make_image(InArray<float> data,
                         std::optional<InArray<std::uint8_t>> mask) {
  const std::size_t h = data.shape(0);
  const std::size_t w = data.shape(1);
  delta::ImageF img(w, h, data.data());
  if (mask) {
    if (mask->shape(0) != h || mask->shape(1) != w)
      throw std::runtime_error("mask shape does not match data");
    img.mask().assign(mask->data(), mask->data() + h * w);
  }
  return img;
}

delta::DetectParams make_params(int stamp_radius, double threshold_sigma,
                                int max_stamps, double saturation,
                                int isolation_radius, int border) {
  delta::DetectParams p;
  p.stamp_radius = stamp_radius;
  p.threshold_sigma = threshold_sigma;
  p.max_stamps = max_stamps;
  p.saturation = saturation;
  p.isolation_radius = isolation_radius;
  p.border = border;
  return p;
}

// Move a std::vector into a NumPy array that owns the buffer (zero copy).
template <typename T>
nb::ndarray<nb::numpy, T> to_numpy(std::vector<T>&& vec,
                                   std::vector<std::size_t> shape) {
  auto* heap = new std::vector<T>(std::move(vec));
  nb::capsule owner(heap, [](void* p) noexcept {
    delete static_cast<std::vector<T>*>(p);
  });
  return nb::ndarray<nb::numpy, T>(heap->data(), shape.size(), shape.data(),
                                   owner);
}

int resolve_radius(double beta, int n_max, int radius) {
  return radius > 0 ? radius
                    : delta::GaussHermiteBasis::default_radius(beta, n_max);
}

// ---- mask/noise demo -------------------------------------------------------

// Inverse-variance weighted mean over good (unmasked) pixels. Exercises the
// zero-copy NumPy path and the mask/noise-weighting contract (SPEC §3.6).
double weighted_mean(InArray<float> data, InArray<float> variance,
                     InArray<std::uint8_t> mask) {
  const std::size_t n = data.size();
  const float* d = data.data();
  const float* v = variance.data();
  const std::uint8_t* m = mask.data();

  double wsum = 0.0;
  double sum = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    if (m[i] != delta::kMaskGood) continue;  // exclude bad pixels
    if (!(v[i] > 0.0f)) continue;            // require positive variance
    const double w = 1.0 / static_cast<double>(v[i]);
    wsum += w;
    sum += w * static_cast<double>(d[i]);
  }
  return wsum > 0.0 ? sum / wsum : 0.0;
}

// ---- IO --------------------------------------------------------------------

nb::dict read_fits(const std::string& path) {
  delta::ImageF image = delta::io::read_image(path);
  const std::size_t h = image.height();
  const std::size_t w = image.width();

  nb::dict out;
  out["data"] = to_numpy<float>(std::move(image.pixels()), {h, w});
  if (image.has_variance())
    out["variance"] = to_numpy<float>(std::move(image.variance()), {h, w});
  else
    out["variance"] = nb::none();
  if (image.has_mask())
    out["mask"] = to_numpy<std::uint8_t>(std::move(image.mask()), {h, w});
  else
    out["mask"] = nb::none();
  return out;
}

void write_fits(const std::string& path, InArray<float> data,
                std::optional<InArray<float>> variance,
                std::optional<InArray<std::uint8_t>> mask,
                bool overwrite) {
  const std::size_t h = data.shape(0);
  const std::size_t w = data.shape(1);

  delta::ImageF image(w, h);
  std::copy(data.data(), data.data() + h * w, image.pixels().begin());

  if (variance) {
    if (variance->shape(0) != h || variance->shape(1) != w)
      throw std::runtime_error("variance shape does not match data");
    image.allocate_variance();
    std::copy(variance->data(), variance->data() + h * w,
              image.variance().begin());
  }
  if (mask) {
    if (mask->shape(0) != h || mask->shape(1) != w)
      throw std::runtime_error("mask shape does not match data");
    image.allocate_mask();
    std::copy(mask->data(), mask->data() + h * w, image.mask().begin());
  }

  delta::io::write_image(path, image, overwrite);
}

// ---- basis / convolve ------------------------------------------------------

// 1-D sampled Gauss-Hermite basis functions, shape (n_max+1, ksize), float64.
nb::ndarray<nb::numpy, double> gauss_hermite_basis1d(double beta, int n_max,
                                                     int radius) {
  const delta::GaussHermiteBasis b(beta, n_max,
                                   resolve_radius(beta, n_max, radius));
  const std::size_t ks = static_cast<std::size_t>(b.ksize());
  std::vector<double> flat;
  flat.reserve((n_max + 1) * ks);
  for (int n = 0; n <= n_max; ++n) {
    const auto& v = b.basis1d(n);
    flat.insert(flat.end(), v.begin(), v.end());
  }
  return to_numpy<double>(std::move(flat),
                          {static_cast<std::size_t>(n_max + 1), ks});
}

// 2-D kernels with their orders: (orders int32 (ncomp,2), kernels f32 (n,k,k)).
nb::object gauss_hermite_kernels(double beta, int n_max, int radius) {
  const delta::GaussHermiteBasis b(beta, n_max,
                                   resolve_radius(beta, n_max, radius));
  const std::size_t nc = static_cast<std::size_t>(b.component_count());
  const std::size_t ks = static_cast<std::size_t>(b.ksize());

  std::vector<std::int32_t> orders;
  orders.reserve(nc * 2);
  std::vector<float> kernels;
  kernels.reserve(nc * ks * ks);
  for (std::size_t c = 0; c < nc; ++c) {
    const auto [nx, ny] = b.orders()[c];
    orders.push_back(nx);
    orders.push_back(ny);
    const auto k = b.kernel2d(static_cast<int>(c));
    kernels.insert(kernels.end(), k.begin(), k.end());
  }
  return nb::make_tuple(to_numpy<std::int32_t>(std::move(orders), {nc, 2}),
                        to_numpy<float>(std::move(kernels), {nc, ks, ks}));
}

// Convolve `image` with every basis component -> B_n stack, shape (ncomp,H,W).
nb::ndarray<nb::numpy, float> basis_convolve(InArray<float> image, double beta,
                                             int n_max, int radius) {
  const std::size_t h = image.shape(0);
  const std::size_t w = image.shape(1);

  delta::ImageF img(w, h);
  std::copy(image.data(), image.data() + h * w, img.pixels().begin());

  const delta::GaussHermiteBasis b(beta, n_max,
                                   resolve_radius(beta, n_max, radius));
  const std::vector<delta::ImageF> stack = delta::basis_convolve(img, b);

  const std::size_t nc = stack.size();
  std::vector<float> flat;
  flat.reserve(nc * h * w);
  for (const auto& im : stack) {
    flat.insert(flat.end(), im.pixels().begin(), im.pixels().end());
  }
  return to_numpy<float>(std::move(flat), {nc, h, w});
}

// ---- detect ----------------------------------------------------------------

// Robust (median, MAD-sigma) background over unmasked pixels.
nb::object estimate_background(InArray<float> data,
                               std::optional<InArray<std::uint8_t>> mask) {
  const delta::ImageF img = make_image(data, mask);
  const delta::BackgroundStats bg = delta::estimate_background(img);
  return nb::make_tuple(bg.median, bg.sigma);
}

// Detect stamps -> {'x','y','flux','snr','fwhm'} arrays.
nb::dict detect_stamps(InArray<float> data,
                       std::optional<InArray<std::uint8_t>> mask,
                       int stamp_radius, double threshold_sigma, int max_stamps,
                       double saturation, int isolation_radius, int border) {
  const delta::ImageF img = make_image(data, mask);
  const delta::DetectParams p = make_params(stamp_radius, threshold_sigma,
                                            max_stamps, saturation,
                                            isolation_radius, border);
  const std::vector<delta::Stamp> stamps = delta::detect_stamps(img, p);
  const std::size_t n = stamps.size();

  std::vector<std::int32_t> xs, ys;
  std::vector<double> flux, snr, fwhm;
  xs.reserve(n);
  ys.reserve(n);
  flux.reserve(n);
  snr.reserve(n);
  fwhm.reserve(n);
  for (const delta::Stamp& s : stamps) {
    xs.push_back(s.x);
    ys.push_back(s.y);
    flux.push_back(s.flux);
    snr.push_back(s.snr);
    fwhm.push_back(s.fwhm);
  }

  nb::dict out;
  out["x"] = to_numpy<std::int32_t>(std::move(xs), {n});
  out["y"] = to_numpy<std::int32_t>(std::move(ys), {n});
  out["flux"] = to_numpy<double>(std::move(flux), {n});
  out["snr"] = to_numpy<double>(std::move(snr), {n});
  out["fwhm"] = to_numpy<double>(std::move(fwhm), {n});
  return out;
}

// Select matched stamps across both images (detection or supplied catalog) and
// the resulting convolution direction.
nb::dict select_stamps(InArray<float> science, InArray<float> reference,
                       std::optional<InArray<std::uint8_t>> science_mask,
                       std::optional<InArray<std::uint8_t>> reference_mask,
                       std::optional<InArray1D<std::int32_t>> catalog_x,
                       std::optional<InArray1D<std::int32_t>> catalog_y,
                       int stamp_radius, double threshold_sigma, int max_stamps,
                       double saturation, int isolation_radius, int border) {
  const delta::ImageF sci = make_image(science, science_mask);
  const delta::ImageF ref = make_image(reference, reference_mask);
  const delta::DetectParams p = make_params(stamp_radius, threshold_sigma,
                                            max_stamps, saturation,
                                            isolation_radius, border);

  delta::StampSelection sel;
  if (catalog_x && catalog_y) {
    const std::vector<int> xs(catalog_x->data(),
                              catalog_x->data() + catalog_x->size());
    const std::vector<int> ys(catalog_y->data(),
                              catalog_y->data() + catalog_y->size());
    sel = delta::select_stamps_from_catalog(sci, ref, xs, ys, p);
  } else {
    sel = delta::select_stamps(sci, ref, p);
  }

  const std::size_t n = sel.x.size();
  nb::dict out;
  out["x"] = to_numpy<std::int32_t>(
      std::vector<std::int32_t>(sel.x.begin(), sel.x.end()), {n});
  out["y"] = to_numpy<std::int32_t>(
      std::vector<std::int32_t>(sel.y.begin(), sel.y.end()), {n});
  out["fwhm_science"] = to_numpy<double>(std::move(sel.fwhm_science), {n});
  out["fwhm_reference"] = to_numpy<double>(std::move(sel.fwhm_reference), {n});
  out["median_fwhm_science"] = sel.median_fwhm_science;
  out["median_fwhm_reference"] = sel.median_fwhm_reference;
  out["direction"] = delta::to_string(sel.direction);
  return out;
}

// Drain the opt-in C++ sub-stage timers (DELTA_TIMING) into a {label: seconds}
// dict, or None when timing is disabled (so callers can branch cheaply).
// Forward-declared here, defined below (used by both `build_catalog` and
// `subtract`).
nb::object timing_dict();

// ---- catalog (connected-component source catalog from the score image) ---

delta::CatalogParams make_catalog_params(double threshold_sigma,
                                         double threshold_sigma_dipole,
                                         double expected_fwhm,
                                         double fwhm_tolerance_lo,
                                         double fwhm_tolerance_hi,
                                         int aperture_radius,
                                         bool exclude_bad_pixels) {
  delta::CatalogParams p;
  p.threshold_sigma = threshold_sigma;
  p.threshold_sigma_dipole = threshold_sigma_dipole;
  p.expected_fwhm = expected_fwhm;
  p.fwhm_tolerance_lo = fwhm_tolerance_lo;
  p.fwhm_tolerance_hi = fwhm_tolerance_hi;
  p.aperture_radius = aperture_radius;
  p.exclude_bad_pixels = exclude_bad_pixels;
  return p;
}

// Connected-component source catalog from a match-filtered score image
// (SPEC §3.7) -> column arrays.
nb::dict build_catalog(InArray<float> score, InArray<float> difference,
                       std::optional<InArray<std::uint8_t>> mask,
                       double threshold_sigma, double threshold_sigma_dipole,
                       double expected_fwhm, double fwhm_tolerance_lo,
                       double fwhm_tolerance_hi, int aperture_radius,
                       bool exclude_bad_pixels, bool return_negative) {
  if (difference.shape(0) != score.shape(0) ||
      difference.shape(1) != score.shape(1)) {
    throw std::runtime_error("build_catalog: difference shape must match score");
  }
  delta::timing::clear();
  delta::ImageF score_img, diff_img;
  {
    DELTA_TIME("catalog: marshal input");
    score_img = make_image(score, mask);
    diff_img = delta::ImageF(difference.shape(1), difference.shape(0), difference.data());
  }

  const delta::CatalogParams p = make_catalog_params(
      threshold_sigma, threshold_sigma_dipole, expected_fwhm,
      fwhm_tolerance_lo, fwhm_tolerance_hi, aperture_radius, exclude_bad_pixels);
  const std::vector<delta::CatalogEntry> entries =
      delta::build_catalog(score_img, diff_img, p, return_negative);

  const std::size_t n = entries.size();
  std::vector<double> x, y, peak_snr, flux, expected_n_pix, fwhm_ratio;
  std::vector<std::int32_t> peak_x, peak_y, n_pix;
  std::vector<std::uint8_t> mask_flags, fwhm_consistent, is_dipole, quality;
  {
    DELTA_TIME("catalog: marshal output");
    x.reserve(n); y.reserve(n); peak_x.reserve(n); peak_y.reserve(n);
    peak_snr.reserve(n); n_pix.reserve(n); flux.reserve(n);
    expected_n_pix.reserve(n); fwhm_ratio.reserve(n);
    mask_flags.reserve(n); fwhm_consistent.reserve(n); is_dipole.reserve(n);
    quality.reserve(n);
    for (const delta::CatalogEntry& e : entries) {
      x.push_back(e.x);
      y.push_back(e.y);
      peak_x.push_back(e.peak_x);
      peak_y.push_back(e.peak_y);
      peak_snr.push_back(e.peak_snr);
      n_pix.push_back(e.n_pix);
      flux.push_back(e.flux);
      expected_n_pix.push_back(e.expected_n_pix);
      fwhm_ratio.push_back(e.fwhm_ratio);
      mask_flags.push_back(e.mask_flags);
      fwhm_consistent.push_back(e.fwhm_consistent ? 1 : 0);
      is_dipole.push_back(e.is_dipole ? 1 : 0);
      quality.push_back(e.quality);
    }
  }

  // Drained after every DELTA_TIME scope above has destructed (each timer
  // records on scope exit), so this captures the full breakdown.
  nb::object timing = timing_dict();

  nb::dict out;
  out["x"] = to_numpy<double>(std::move(x), {n});
  out["y"] = to_numpy<double>(std::move(y), {n});
  out["peak_x"] = to_numpy<std::int32_t>(std::move(peak_x), {n});
  out["peak_y"] = to_numpy<std::int32_t>(std::move(peak_y), {n});
  out["peak_snr"] = to_numpy<double>(std::move(peak_snr), {n});
  out["n_pix"] = to_numpy<std::int32_t>(std::move(n_pix), {n});
  out["flux"] = to_numpy<double>(std::move(flux), {n});
  out["expected_n_pix"] = to_numpy<double>(std::move(expected_n_pix), {n});
  out["fwhm_ratio"] = to_numpy<double>(std::move(fwhm_ratio), {n});
  out["fwhm_consistent"] = to_numpy<std::uint8_t>(std::move(fwhm_consistent), {n});
  out["mask_flags"] = to_numpy<std::uint8_t>(std::move(mask_flags), {n});
  out["is_dipole"] = to_numpy<std::uint8_t>(std::move(is_dipole), {n});
  out["quality"] = to_numpy<std::uint8_t>(std::move(quality), {n});
  out["timing"] = timing;
  return out;
}

// ---- spatial (thin-plate regression spline) --------------------------------

Eigen::MatrixXd tps_design(const Eigen::MatrixXd& knots,
                           const Eigen::MatrixXd& points) {
  return delta::ThinPlateBasis(knots).design(points);
}

Eigen::MatrixXd tps_penalty(const Eigen::MatrixXd& knots) {
  return delta::ThinPlateBasis(knots).penalty();
}

Eigen::VectorXd tps_fit(const Eigen::MatrixXd& knots,
                        const Eigen::MatrixXd& points,
                        const Eigen::VectorXd& values, double lam) {
  return delta::tps_fit(delta::ThinPlateBasis(knots), points, values, lam);
}

Eigen::VectorXd tps_evaluate(const Eigen::MatrixXd& knots,
                             const Eigen::MatrixXd& points,
                             const Eigen::VectorXd& coeffs) {
  return delta::ThinPlateBasis(knots).design(points) * coeffs;
}

// ---- solve (penalised GLS + GCV) -------------------------------------------

// Pack a GlsResult into a Python dict. The GCV-curve fields are only filled by
// the GCV search; for a fixed-lambda solve they come back empty.
nb::dict gls_result_to_dict(const delta::GlsResult& r) {
  nb::dict out;
  out["theta"] = Eigen::VectorXd(r.theta);
  out["n_components"] = r.n_components;
  out["n_spatial"] = r.n_spatial;
  out["lambda"] = r.lambda;
  out["gcv"] = r.gcv;
  out["effective_dof"] = r.effective_dof;
  out["rss"] = r.rss;
  out["lambda_grid"] = to_numpy<double>(
      std::vector<double>(r.lambda_grid), {r.lambda_grid.size()});
  out["gcv_curve"] =
      to_numpy<double>(std::vector<double>(r.gcv_curve), {r.gcv_curve.size()});
  return out;
}

// Penalised GLS at a fixed smoothing lambda. `bn` is the per-pixel
// basis-convolved template stack (N, nc); see delta::solve_gls.
nb::dict solve_gls(const Eigen::MatrixXd& knots, const Eigen::MatrixXd& points,
                   const Eigen::VectorXd& target,
                   const Eigen::VectorXd& weights, const Eigen::MatrixXd& bn,
                   double lam) {
  const delta::ThinPlateBasis basis(knots);
  return gls_result_to_dict(
      delta::solve_gls(points, target, weights, bn, basis, lam));
}

// Penalised GLS selecting lambda by GCV over `lambda_grid`.
nb::dict solve_gls_gcv(const Eigen::MatrixXd& knots,
                       const Eigen::MatrixXd& points,
                       const Eigen::VectorXd& target,
                       const Eigen::VectorXd& weights,
                       const Eigen::MatrixXd& bn,
                       const Eigen::VectorXd& lambda_grid) {
  const delta::ThinPlateBasis basis(knots);
  const std::vector<double> grid(lambda_grid.data(),
                                 lambda_grid.data() + lambda_grid.size());
  return gls_result_to_dict(
      delta::solve_gls_gcv(points, target, weights, bn, basis, grid));
}

// As solve_gls_gcv but selecting lambda by k-fold group cross-validation; `group`
// is the fold id per pixel (gcv_curve in the result holds the CV-error curve).
nb::dict solve_gls_cv(const Eigen::MatrixXd& knots, const Eigen::MatrixXd& points,
                      const Eigen::VectorXd& target,
                      const Eigen::VectorXd& weights, const Eigen::MatrixXd& bn,
                      const Eigen::VectorXd& lambda_grid,
                      InArray1D<std::int32_t> group, int n_groups) {
  const delta::ThinPlateBasis basis(knots);
  const std::vector<double> grid(lambda_grid.data(),
                                 lambda_grid.data() + lambda_grid.size());
  const std::vector<int> grp(group.data(), group.data() + group.size());
  return gls_result_to_dict(
      delta::solve_gls_cv(points, target, weights, bn, basis, grid, grp, n_groups));
}

// ---- subtract (full-frame spatially-varying subtraction) -------------------

// Build an ImageF from data with optional variance and mask layers.
delta::ImageF make_image_vm(InArray<float> data,
                            std::optional<InArray<float>> variance,
                            std::optional<InArray<std::uint8_t>> mask) {
  const std::size_t h = data.shape(0);
  const std::size_t w = data.shape(1);
  delta::ImageF img(w, h);
  std::copy(data.data(), data.data() + h * w, img.pixels().begin());
  if (variance) {
    if (variance->shape(0) != h || variance->shape(1) != w)
      throw std::runtime_error("variance shape does not match data");
    img.allocate_variance();
    std::copy(variance->data(), variance->data() + h * w,
              img.variance().begin());
  }
  if (mask) {
    if (mask->shape(0) != h || mask->shape(1) != w)
      throw std::runtime_error("mask shape does not match data");
    img.allocate_mask();
    std::copy(mask->data(), mask->data() + h * w, img.mask().begin());
  }
  return img;
}

// Drain the opt-in C++ sub-stage timers (DELTA_TIMING) into a {label: seconds}
// dict, or None when timing is disabled (so callers can branch cheaply).
nb::object timing_dict() {
  if (!delta::timing::enabled()) return nb::none();
  nb::dict out;
  for (const auto& [label, seconds] : delta::timing::drain())
    out[label.c_str()] = seconds;
  return out;
}

// Full-frame subtraction -> {'difference', 'variance', 'mask'} (the latter two
// present only when the inputs carry the corresponding layers).
nb::dict subtract(InArray<float> science, InArray<float> reference,
                  const Eigen::MatrixXd& knots, const Eigen::VectorXd& theta,
                  double beta, int n_max, int radius, double saturation,
                  std::optional<InArray<float>> science_var,
                  std::optional<InArray<float>> reference_var,
                  std::optional<InArray<std::uint8_t>> science_mask,
                  std::optional<InArray<std::uint8_t>> reference_mask) {
  const delta::ImageF sci = make_image_vm(science, science_var, science_mask);
  const delta::ImageF ref =
      make_image_vm(reference, reference_var, reference_mask);
  const delta::ThinPlateBasis spatial(knots);
  const delta::GaussHermiteBasis basis(beta, n_max,
                                       resolve_radius(beta, n_max, radius));

  delta::timing::clear();
  delta::ImageF diff = delta::subtract(sci, ref, spatial, theta, basis, saturation);
  const std::size_t h = diff.height();
  const std::size_t w = diff.width();

  nb::dict out;
  out["timing"] = timing_dict();
  out["difference"] = to_numpy<float>(std::move(diff.pixels()), {h, w});
  if (diff.has_variance())
    out["variance"] = to_numpy<float>(std::move(diff.variance()), {h, w});
  else
    out["variance"] = nb::none();
  if (diff.has_mask())
    out["mask"] = to_numpy<std::uint8_t>(std::move(diff.mask()), {h, w});
  else
    out["mask"] = nb::none();
  return out;
}

// ---- fit (kernel solve from stamps) + photometric scale --------------------

// Fit the matching kernel from stamp pixels -> solver result + reporting.
nb::dict fit_kernel(InArray<float> science, InArray<float> reference,
                    const Eigen::MatrixXd& knots,
                    InArray1D<std::int32_t> stamp_x,
                    InArray1D<std::int32_t> stamp_y, int stamp_radius,
                    double beta, int n_max,
                    const Eigen::VectorXd& lambda_grid, int radius,
                    double clip_sigma, int clip_iterations, int min_stamps,
                    int cv_folds,
                    std::optional<InArray<float>> science_var,
                    std::optional<InArray<float>> reference_var,
                    std::optional<InArray<std::uint8_t>> science_mask,
                    std::optional<InArray<std::uint8_t>> reference_mask) {
  const delta::ImageF sci = make_image_vm(science, science_var, science_mask);
  const delta::ImageF ref =
      make_image_vm(reference, reference_var, reference_mask);
  const delta::ThinPlateBasis spatial(knots);
  const delta::GaussHermiteBasis basis(beta, n_max,
                                       resolve_radius(beta, n_max, radius));
  const std::vector<int> sx(stamp_x.data(), stamp_x.data() + stamp_x.size());
  const std::vector<int> sy(stamp_y.data(), stamp_y.data() + stamp_y.size());
  const std::vector<double> grid(lambda_grid.data(),
                                 lambda_grid.data() + lambda_grid.size());

  delta::timing::clear();
  const delta::KernelFit fit =
      delta::fit_kernel(sci, ref, spatial, basis, sx, sy, stamp_radius, grid,
                        clip_sigma, clip_iterations, min_stamps, cv_folds);

  nb::dict out = gls_result_to_dict(fit.gls);
  out["timing"] = timing_dict();
  out["component_sums"] = to_numpy<double>(
      std::vector<double>(fit.component_sums), {fit.component_sums.size()});
  out["n_pixels"] = fit.n_pixels;
  out["n_stamps_used"] = fit.n_stamps_used;
  out["n_stamps_total"] = fit.n_stamps_total;
  out["n_stamps_rejected"] = fit.n_stamps_rejected;
  out["reduced_chi2"] = fit.reduced_chi2;
  out["stamp_x"] = to_numpy<int>(std::vector<int>(fit.stamp_x),
                                 {fit.stamp_x.size()});
  out["stamp_y"] = to_numpy<int>(std::vector<int>(fit.stamp_y),
                                 {fit.stamp_y.size()});
  out["stamp_chi2"] = to_numpy<double>(std::vector<double>(fit.stamp_chi2),
                                       {fit.stamp_chi2.size()});
  out["stamp_accepted"] = to_numpy<std::uint8_t>(
      std::vector<std::uint8_t>(fit.stamp_accepted), {fit.stamp_accepted.size()});
  return out;
}

// Per-pixel photometric scale field, shape (height, width).
nb::ndarray<nb::numpy, float> photometric_scale(
    const Eigen::MatrixXd& knots, const Eigen::VectorXd& theta,
    InArray1D<double> component_sums, int height, int width) {
  const delta::ThinPlateBasis spatial(knots);
  const std::vector<double> sums(component_sums.data(),
                                 component_sums.data() + component_sums.size());
  delta::ImageF scale = delta::photometric_scale(
      spatial, theta, sums, static_cast<std::size_t>(width),
      static_cast<std::size_t>(height));
  const std::size_t h = scale.height();
  const std::size_t w = scale.width();
  return to_numpy<float>(std::move(scale.pixels()), {h, w});
}

// Photometric scale at arbitrary points (m,).
Eigen::VectorXd photometric_scale_at(const Eigen::MatrixXd& knots,
                                     const Eigen::VectorXd& theta,
                                     InArray1D<double> component_sums,
                                     const Eigen::MatrixXd& points) {
  const delta::ThinPlateBasis spatial(knots);
  const std::vector<double> sums(component_sums.data(),
                                 component_sums.data() + component_sums.size());
  return delta::photometric_scale_at(spatial, theta, sums, points);
}

// ---- noise (decorrelation + match-filtered score) --------------------------

// Decorrelate a single square (n x n) block with a constant kernel/noise: the
// FFT whitening core (one circular block), useful for QA and as a building unit.
nb::ndarray<nb::numpy, float> decorrelate_block(InArray<float> image,
                                                InArray<float> kernel,
                                                double var_science,
                                                double var_reference) {
  const std::size_t n = image.shape(0);
  if (image.shape(1) != n)
    throw std::runtime_error("decorrelate_block: image must be square");
  const int ks = static_cast<int>(kernel.shape(0));
  if (kernel.shape(1) != static_cast<std::size_t>(ks))
    throw std::runtime_error("decorrelate_block: kernel must be square");

  delta::ImageF block(n, n);
  std::copy(image.data(), image.data() + n * n, block.pixels().begin());
  const std::vector<float> kern(kernel.data(),
                                kernel.data() + kernel.size());
  const std::vector<float> filter = delta::decorrelation_filter(
      kern, ks, var_science, var_reference, static_cast<int>(n));
  delta::ImageF out =
      delta::apply_filter_fft(block, filter, static_cast<int>(n));
  return to_numpy<float>(std::move(out.pixels()), {n, n});
}

// Real-space decorrelation kernel (centred, n x n) for QA.
nb::ndarray<nb::numpy, float> decorrelation_kernel(InArray<float> kernel,
                                                   double var_science,
                                                   double var_reference, int n) {
  const int ks = static_cast<int>(kernel.shape(0));
  const std::vector<float> kern(kernel.data(),
                                kernel.data() + kernel.size());
  const std::vector<float> filter =
      delta::decorrelation_filter(kern, ks, var_science, var_reference, n);
  delta::ImageF out = delta::decorrelation_kernel_image(filter, n);
  return to_numpy<float>(std::move(out.pixels()),
                         {static_cast<std::size_t>(n),
                          static_cast<std::size_t>(n)});
}

// Spatially-varying decorrelation of a difference image (apodized FFT blocks).
// Returns {'difference', 'variance'} where variance is the post-whitening noise
// level (σ_D² = v_S + v_R ΣK² per block, Hann-blended) — use it for pulls and
// the match-filtered score of the whitened difference.
nb::dict decorrelate(InArray<float> difference, const Eigen::MatrixXd& knots,
                     const Eigen::VectorXd& theta, double beta, int n_max,
                     InArray<float> var_science, InArray<float> var_reference,
                     int block, int radius, int kernel_cell_blocks) {
  const std::size_t h = difference.shape(0);
  const std::size_t w = difference.shape(1);
  delta::ImageF diff(w, h);
  std::copy(difference.data(), difference.data() + h * w,
            diff.pixels().begin());
  delta::ImageF vs(w, h), vr(w, h);
  std::copy(var_science.data(), var_science.data() + h * w,
            vs.pixels().begin());
  std::copy(var_reference.data(), var_reference.data() + h * w,
            vr.pixels().begin());

  const delta::ThinPlateBasis spatial(knots);
  const delta::GaussHermiteBasis basis(beta, n_max,
                                       resolve_radius(beta, n_max, radius));
  delta::ImageF out = delta::decorrelate(diff, spatial, theta, basis, vs, vr,
                                         block, kernel_cell_blocks);
  nb::dict result;
  result["difference"] = to_numpy<float>(std::move(out.pixels()), {h, w});
  result["variance"] = to_numpy<float>(std::move(out.variance()), {h, w});
  return result;
}

// Whiten a PSF stamp with the ZOGY decorrelation filter at a representative
// location (frame centre by default), for match-filtering a whitened difference.
nb::ndarray<nb::numpy, float> whiten_score_psf(
    InArray<float> psf, const Eigen::MatrixXd& knots,
    const Eigen::VectorXd& theta, double beta, int n_max,
    InArray<float> var_science, InArray<float> var_reference, int block,
    int radius) {
  const int ps = static_cast<int>(psf.shape(0));
  if (psf.shape(1) != static_cast<std::size_t>(ps))
    throw std::runtime_error("whiten_score_psf: psf must be square");
  if (block < ps)
    throw std::runtime_error("whiten_score_psf: block must be >= psf side");

  const std::size_t h = var_science.shape(0);
  const std::size_t w = var_science.shape(1);
  if (var_reference.shape(0) != h || var_reference.shape(1) != w)
    throw std::runtime_error("whiten_score_psf: variance shapes disagree");

  const delta::ThinPlateBasis spatial(knots);
  const delta::GaussHermiteBasis basis(beta, n_max,
                                       resolve_radius(beta, n_max, radius));
  const int nc = basis.component_count();
  std::vector<std::vector<float>> phi(nc);
  for (int n = 0; n < nc; ++n) phi[n] = basis.kernel2d(n);

  // Matching kernel + robust noise at the frame centre (knot-scale fields;
  // a single representative Phi is enough for the compact score PSF).
  const double cx = 0.5 * static_cast<double>(w - 1);
  const double cy = 0.5 * static_cast<double>(h - 1);
  const int k = spatial.n_basis();
  const Eigen::MatrixXd c =
      Eigen::Map<const Eigen::MatrixXd>(theta.data(), k, nc + 1);
  Eigen::MatrixXd point(1, 2);
  point(0, 0) = cx;
  point(0, 1) = cy;
  const Eigen::MatrixXd a = spatial.design(point) * c;
  const int ks = basis.ksize();
  std::vector<float> kernel(static_cast<std::size_t>(ks) * ks, 0.0f);
  for (int n = 0; n < nc; ++n) {
    const float an = static_cast<float>(a(0, n));
    const float* p = phi[n].data();
    for (std::size_t i = 0; i < kernel.size(); ++i) kernel[i] += an * p[i];
  }

  delta::ImageF vs_img(w, h), vr_img(w, h);
  std::copy(var_science.data(), var_science.data() + h * w,
            vs_img.pixels().begin());
  std::copy(var_reference.data(), var_reference.data() + h * w,
            vr_img.pixels().begin());
  // Reuse the same robust block-median estimator as decorrelate (via a local
  // window centred on the frame). Fall back to a 1-pixel sample if the frame
  // is smaller than `block`.
  const int bx = std::max(0, static_cast<int>(w) / 2 - block / 2);
  const int by = std::max(0, static_cast<int>(h) / 2 - block / 2);
  // Inline median of positive finite samples over the central block (mirrors
  // noise.cpp::block_variance without exposing it).
  auto block_var = [&](const delta::ImageF& var) {
    const int step = std::max(1, block / 64);
    std::vector<float> vals;
    vals.reserve(static_cast<std::size_t>(block / step + 1) *
                 (block / step + 1));
    for (int y = by; y < by + block; y += step) {
      if (y < 0 || y >= static_cast<int>(h)) continue;
      for (int x = bx; x < bx + block; x += step) {
        if (x < 0 || x >= static_cast<int>(w)) continue;
        const float v =
            var.data()[static_cast<std::size_t>(y) * w + x];
        if (std::isfinite(v) && v > 0.0f) vals.push_back(v);
      }
    }
    if (vals.empty()) return 0.0;
    const std::size_t mid = vals.size() / 2;
    std::nth_element(vals.begin(), vals.begin() + mid, vals.end());
    return static_cast<double>(vals[mid]);
  };
  const double vs = block_var(vs_img);
  const double vr = block_var(vr_img);

  const std::vector<float> p(psf.data(), psf.data() + psf.size());
  std::vector<float> out =
      delta::whiten_psf(p, ps, kernel, ks, vs, vr, block);
  return to_numpy<float>(std::move(out),
                         {static_cast<std::size_t>(ps),
                          static_cast<std::size_t>(ps)});
}

// Match-filtered score image (S/N map). `variance` is a same-shape per-pixel
// noise variance image; each output pixel is normalised by sqrt(var * sumpsf2).
nb::ndarray<nb::numpy, float> matched_filter(InArray<float> image,
                                             InArray<float> psf,
                                             InArray<float> variance) {
  const std::size_t h = image.shape(0);
  const std::size_t w = image.shape(1);
  const int ps = static_cast<int>(psf.shape(0));
  if (psf.shape(1) != static_cast<std::size_t>(ps))
    throw std::runtime_error("matched_filter: psf must be square");
  if (variance.shape(0) != h || variance.shape(1) != w)
    throw std::runtime_error("matched_filter: variance must match image shape");

  delta::ImageF img(w, h);
  std::copy(image.data(), image.data() + h * w, img.pixels().begin());
  delta::ImageF var(w, h);
  std::copy(variance.data(), variance.data() + h * w, var.pixels().begin());
  const std::vector<float> p(psf.data(), psf.data() + psf.size());
  delta::ImageF out = delta::matched_filter(img, p, ps, var);
  return to_numpy<float>(std::move(out.pixels()), {h, w});
}

}  // namespace

NB_MODULE(_core, m) {
  m.doc() = "delta C++ core (Alard & Lupton difference imaging)";
  m.attr("__version__") = delta::version;

  m.def("weighted_mean", &weighted_mean, "data"_a, "variance"_a, "mask"_a,
        "Inverse-variance weighted mean over unmasked pixels.");

  m.def("read_fits", &read_fits, "path"_a,
        "Read a FITS file into {'data', 'variance', 'mask'} NumPy arrays.");
  m.def("write_fits", &write_fits, "path"_a, "data"_a,
        "variance"_a = nb::none(), "mask"_a = nb::none(),
        "overwrite"_a = false,
        "Write data plus optional variance/mask layers to a FITS file. Raises if "
        "the file exists unless overwrite=True.");

  m.def("gauss_hermite_basis1d", &gauss_hermite_basis1d, "beta"_a, "n_max"_a,
        "radius"_a = 0,
        "1-D sampled Gauss-Hermite basis functions, shape (n_max+1, ksize).");
  m.def("gauss_hermite_kernels", &gauss_hermite_kernels, "beta"_a, "n_max"_a,
        "radius"_a = 0,
        "(orders, kernels) for the 2-D Gauss-Hermite components.");
  m.def("basis_convolve", &basis_convolve, "image"_a, "beta"_a, "n_max"_a,
        "radius"_a = 0,
        "Convolve image with each basis component; returns (ncomp, H, W).");

  m.def("estimate_background", &estimate_background, "data"_a,
        "mask"_a = nb::none(),
        "Robust (median, MAD-sigma) background over unmasked pixels.");
  m.def("detect_stamps", &detect_stamps, "data"_a, "mask"_a = nb::none(),
        "stamp_radius"_a = 15, "threshold_sigma"_a = 5.0, "max_stamps"_a = 200,
        "saturation"_a = 0.0, "isolation_radius"_a = 0, "border"_a = 0,
        "Detect bright, isolated, unsaturated point-source stamps.");
  m.def("select_stamps", &select_stamps, "science"_a, "reference"_a,
        "science_mask"_a = nb::none(), "reference_mask"_a = nb::none(),
        "catalog_x"_a = nb::none(), "catalog_y"_a = nb::none(),
        "stamp_radius"_a = 15, "threshold_sigma"_a = 5.0, "max_stamps"_a = 200,
        "saturation"_a = 0.0, "isolation_radius"_a = 0, "border"_a = 0,
        "Select matched stamps across both images and the convolution "
        "direction.");
  m.def("build_catalog", &build_catalog, "score"_a, "difference"_a,
        "mask"_a = nb::none(), "threshold_sigma"_a = 5.0,
        "threshold_sigma_dipole"_a = 3.0, "expected_fwhm"_a = 3.5,
        "fwhm_tolerance_lo"_a = 0.3, "fwhm_tolerance_hi"_a = 3.0,
        "aperture_radius"_a = 0, "exclude_bad_pixels"_a = true,
        "return_negative"_a = false,
        "Connected-component source catalog from a match-filtered score "
        "image (SPEC §3.7): 8-connected positive/negative blobs, "
        "FWHM-consistency filter, dipole flagging, mask-flag aggregation, "
        "aperture flux on `difference`.");

  m.def("grid_knots", &delta::grid_knots, "x0"_a, "y0"_a, "x1"_a, "y1"_a,
        "nx"_a, "ny"_a, "Regular nx*ny grid of knots over [x0,x1]x[y0,y1].");
  m.def("tps_design", &tps_design, "knots"_a, "points"_a,
        "Thin-plate regression-spline design matrix (m, n_basis).");
  m.def("tps_penalty", &tps_penalty, "knots"_a,
        "Thin-plate bending-energy penalty matrix (n_basis, n_basis).");
  m.def("tps_fit", &tps_fit, "knots"_a, "points"_a, "values"_a, "lam"_a,
        "Unweighted penalised fit: (D^T D + lam P) theta = D^T values.");
  m.def("tps_evaluate", &tps_evaluate, "knots"_a, "points"_a, "coeffs"_a,
        "Evaluate a fitted TPS field at points: design(points) @ coeffs.");

  m.def("solve_gls", &solve_gls, "knots"_a, "points"_a, "target"_a, "weights"_a,
        "bn"_a, "lam"_a,
        "Penalised GLS at fixed lambda over the factorized A&L model.");
  m.def("solve_gls_gcv", &solve_gls_gcv, "knots"_a, "points"_a, "target"_a,
        "weights"_a, "bn"_a, "lambda_grid"_a,
        "Penalised GLS selecting lambda by GCV over lambda_grid.");
  m.def("solve_gls_cv", &solve_gls_cv, "knots"_a, "points"_a, "target"_a,
        "weights"_a, "bn"_a, "lambda_grid"_a, "group"_a, "n_groups"_a,
        "Penalised GLS selecting lambda by k-fold group cross-validation "
        "(group = fold id per pixel); gcv_curve holds the CV-error curve.");

  m.def("subtract", &subtract, "science"_a, "reference"_a, "knots"_a, "theta"_a,
        "beta"_a, "n_max"_a, "radius"_a = 0, "saturation"_a = 0.0,
        "science_var"_a = nb::none(), "reference_var"_a = nb::none(),
        "science_mask"_a = nb::none(), "reference_mask"_a = nb::none(),
        "Full-frame spatially-varying subtraction with variance/mask "
        "propagation; saturation>0 masks+grows bright cores. Returns "
        "{'difference','variance','mask'}.");

  m.def("fit_kernel", &fit_kernel, "science"_a, "reference"_a, "knots"_a,
        "stamp_x"_a, "stamp_y"_a, "stamp_radius"_a, "beta"_a, "n_max"_a,
        "lambda_grid"_a, "radius"_a = 0, "clip_sigma"_a = 4.0,
        "clip_iterations"_a = 5, "min_stamps"_a = 5, "cv_folds"_a = 0,
        "science_var"_a = nb::none(), "reference_var"_a = nb::none(),
        "science_mask"_a = nb::none(), "reference_mask"_a = nb::none(),
        "Fit the matching kernel + background from stamp pixels via penalised "
        "GLS, with iterative per-stamp sigma clipping. Returns the solver result "
        "plus component_sums, n_pixels, per-stamp chi^2 and goodness-of-fit "
        "diagnostics (reduced_chi2, n_stamps_used/total/rejected).");
  m.def("photometric_scale", &photometric_scale, "knots"_a, "theta"_a,
        "component_sums"_a, "height"_a, "width"_a,
        "Per-pixel photometric scale field sum_n a_n(x,y) S_n, shape (H,W).");
  m.def("photometric_scale_at", &photometric_scale_at, "knots"_a, "theta"_a,
        "component_sums"_a, "points"_a,
        "Photometric scale evaluated at points, shape (m,).");

  m.def("decorrelate_block", &decorrelate_block, "image"_a, "kernel"_a,
        "var_science"_a, "var_reference"_a,
        "Whiten one square block with a constant kernel/noise (FFT core).");
  m.def("decorrelation_kernel", &decorrelation_kernel, "kernel"_a,
        "var_science"_a, "var_reference"_a, "n"_a,
        "Real-space decorrelation kernel (centred, n x n) for QA.");
  m.def("decorrelate", &decorrelate, "difference"_a, "knots"_a, "theta"_a,
        "beta"_a, "n_max"_a, "var_science"_a, "var_reference"_a, "block"_a = 256,
        "radius"_a = 0, "kernel_cell_blocks"_a = 0,
        "Spatially-varying noise decorrelation via apodized FFT blocks. "
        "Returns {'difference', 'variance'} where variance is the "
        "post-whitening noise level for pulls and the match-filtered score.");
  m.def("whiten_score_psf", &whiten_score_psf, "psf"_a, "knots"_a, "theta"_a,
        "beta"_a, "n_max"_a, "var_science"_a, "var_reference"_a, "block"_a,
        "radius"_a = 0,
        "Apply the ZOGY decorrelation filter to a PSF stamp (frame-centre "
        "kernel/noise) for match-filtering a whitened difference.");
  m.def("matched_filter", &matched_filter, "image"_a, "psf"_a, "variance"_a,
        "Match-filtered score image (per-pixel S/N map). variance must be a "
        "same-shape float32 image of per-pixel noise variance.");
}
