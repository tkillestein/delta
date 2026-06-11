#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>

#include <algorithm>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <vector>

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
nb::ndarray<nb::numpy, T, nb::ndim<2>> vec_to_numpy(std::vector<T>&& vec,
                                                    std::size_t height,
                                                    std::size_t width) {
  auto* heap = new std::vector<T>(std::move(vec));
  nb::capsule owner(heap, [](void* p) noexcept {
    delete static_cast<std::vector<T>*>(p);
  });
  return nb::ndarray<nb::numpy, T, nb::ndim<2>>(heap->data(), {height, width},
                                                owner);
}

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

// Read a FITS file into {'data', 'variance', 'mask'} NumPy arrays (variance and
// mask are None when the corresponding extension is absent).
nb::dict read_fits(const std::string& path) {
  delta::ImageF image = delta::io::read_image(path);
  const std::size_t h = image.height();
  const std::size_t w = image.width();

  nb::dict out;
  out["data"] = vec_to_numpy<float>(std::move(image.pixels()), h, w);
  if (image.has_variance())
    out["variance"] = vec_to_numpy<float>(std::move(image.variance()), h, w);
  else
    out["variance"] = nb::none();
  if (image.has_mask())
    out["mask"] = vec_to_numpy<std::uint8_t>(std::move(image.mask()), h, w);
  else
    out["mask"] = nb::none();
  return out;
}

// Write data plus optional variance / mask layers to a multi-extension FITS.
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
}
