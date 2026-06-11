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
#include "delta/image.hpp"
#include "delta/io.hpp"
#include "delta/version.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace {

template <typename T>
using InArray =
    nb::ndarray<const T, nb::ndim<2>, nb::c_contig, nb::device::cpu>;

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
}
