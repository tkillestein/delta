#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <vector>

#include "delta/basis.hpp"
#include "delta/convolve.hpp"
#include "delta/detect.hpp"
#include "delta/image.hpp"
#include "delta/io.hpp"
#include "delta/solve.hpp"
#include "delta/spatial.hpp"
#include "delta/subtract.hpp"
#include "delta/version.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace {

template <typename T>
using InArray =
    nb::ndarray<const T, nb::ndim<2>, nb::c_contig, nb::device::cpu>;

template <typename T>
using InArray1D =
    nb::ndarray<const T, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

// Build an ImageF from a 2-D float array and an optional bad-pixel mask.
delta::ImageF make_image(InArray<float> data,
                         std::optional<InArray<std::uint8_t>> mask) {
  const std::size_t h = data.shape(0);
  const std::size_t w = data.shape(1);
  delta::ImageF img(w, h);
  std::copy(data.data(), data.data() + h * w, img.pixels().begin());
  if (mask) {
    if (mask->shape(0) != h || mask->shape(1) != w)
      throw std::runtime_error("mask shape does not match data");
    img.allocate_mask();
    std::copy(mask->data(), mask->data() + h * w, img.mask().begin());
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
                std::optional<InArray<std::uint8_t>> mask) {
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

  delta::io::write_image(path, image);
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

// Full-frame subtraction -> {'difference', 'variance', 'mask'} (the latter two
// present only when the inputs carry the corresponding layers).
nb::dict subtract(InArray<float> science, InArray<float> reference,
                  const Eigen::MatrixXd& knots, const Eigen::VectorXd& theta,
                  double beta, int n_max, int radius,
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

  delta::ImageF diff = delta::subtract(sci, ref, spatial, theta, basis);
  const std::size_t h = diff.height();
  const std::size_t w = diff.width();

  nb::dict out;
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
        "Write data plus optional variance/mask layers to a FITS file.");

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

  m.def("subtract", &subtract, "science"_a, "reference"_a, "knots"_a, "theta"_a,
        "beta"_a, "n_max"_a, "radius"_a = 0, "science_var"_a = nb::none(),
        "reference_var"_a = nb::none(), "science_mask"_a = nb::none(),
        "reference_mask"_a = nb::none(),
        "Full-frame spatially-varying subtraction with variance/mask "
        "propagation; returns {'difference','variance','mask'}.");
}
