#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <cstdint>

#include "delta/image.hpp"
#include "delta/version.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace {

// Inverse-variance weighted mean over good (unmasked) pixels.
//
// This is the M0 scaffold's proof of life: it exercises zero-copy NumPy access
// and the mask/noise-weighting contract the rest of the pipeline is built on
// (SPEC §3.6) -- masked pixels are excluded, the remainder weighted by 1/var.
double weighted_mean(
    nb::ndarray<const float, nb::ndim<2>, nb::c_contig, nb::device::cpu> data,
    nb::ndarray<const float, nb::ndim<2>, nb::c_contig, nb::device::cpu> variance,
    nb::ndarray<const std::uint8_t, nb::ndim<2>, nb::c_contig, nb::device::cpu>
        mask) {
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

}  // namespace

NB_MODULE(_core, m) {
  m.doc() = "delta C++ core (Alard & Lupton difference imaging)";
  m.attr("__version__") = delta::version;

  m.def("weighted_mean", &weighted_mean, "data"_a, "variance"_a, "mask"_a,
        "Inverse-variance weighted mean over unmasked pixels.");
}
