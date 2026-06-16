#include "delta/convolve.hpp"

#include <cstddef>

namespace delta {

// Row-parallelism is only worth its fork/join cost above this much work. The
// stamp fit convolves ~35-row patches per component (fit.cpp), where spawning a
// full thread team per call is pure overhead -- it made the `B_n` precompute
// *slower* with more threads. Full-frame convolutions (subtract.cpp, ~6000 rows)
// clear this by orders of magnitude and stay threaded. Gating on total work
// (height*width) keeps both: small patch -> serial, full frame -> parallel.
namespace {
constexpr std::size_t kParallelMinWork = 1u << 15;  // ~32k elements
}

// 1-D convolution along x (fast axis), zero-padded, 'same' size.
// out[y,x] = sum_j k[j] * in[y, x + h - j]   (j = m + h, h = ksize/2).
//
// Tap-outer accumulation: for each kernel tap we add a shifted, scaled copy of
// the row. The inner loop is a contiguous float axpy that the compiler
// auto-vectorises (SIMD); out-of-range source indices are simply skipped, which
// is exactly the zero-padding contract.
ImageF convolve_x(const ImageF& in, const std::vector<float>& k) {
  const int width = static_cast<int>(in.width());
  const int height = static_cast<int>(in.height());
  const int ks = static_cast<int>(k.size());
  const int h = (ks - 1) / 2;

  ImageF out(in.width(), in.height());
  const float* src = in.data();
  float* dst = out.data();

#pragma omp parallel for schedule(static) \
    if (static_cast<std::size_t>(height) * width >= kParallelMinWork)
  for (int y = 0; y < height; ++y) {
    const std::size_t row = static_cast<std::size_t>(y) * width;
    float* drow = dst + row;
    const float* srow = src + row;
    for (int x = 0; x < width; ++x) drow[x] = 0.0f;
    for (int j = 0; j < ks; ++j) {
      const float kc = k[j];
      const int sh = h - j;  // src index = x + sh
      const int xlo = sh < 0 ? -sh : 0;
      const int xhi = (width - 1 - sh) < (width - 1) ? (width - 1 - sh) : (width - 1);
#pragma omp simd
      for (int x = xlo; x <= xhi; ++x) drow[x] += kc * srow[x + sh];
    }
  }
  return out;
}

// 1-D convolution along y, zero-padded, 'same' size.
// out[y,x] = sum_j k[j] * in[y + h - j, x].  Tap-outer over input rows so the
// inner loop stays a contiguous float axpy across x (SIMD-friendly).
ImageF convolve_y(const ImageF& in, const std::vector<float>& k) {
  const int width = static_cast<int>(in.width());
  const int height = static_cast<int>(in.height());
  const int ks = static_cast<int>(k.size());
  const int h = (ks - 1) / 2;

  ImageF out(in.width(), in.height());
  const float* src = in.data();
  float* dst = out.data();

#pragma omp parallel for schedule(static) \
    if (static_cast<std::size_t>(height) * width >= kParallelMinWork)
  for (int y = 0; y < height; ++y) {
    float* drow = dst + static_cast<std::size_t>(y) * width;
    for (int x = 0; x < width; ++x) drow[x] = 0.0f;
    const int jmin = std::max(0, y + h - height + 1);
    const int jmax = std::min(ks - 1, y + h);
    for (int j = jmin; j <= jmax; ++j) {
      const int sy = y + h - j;
      const float kc = k[j];
      const float* srow = src + static_cast<std::size_t>(sy) * width;
#pragma omp simd
      for (int x = 0; x < width; ++x) drow[x] += kc * srow[x];
    }
  }
  return out;
}

ImageF convolve_separable(const ImageF& image, const std::vector<float>& kx,
                          const std::vector<float>& ky) {
  return convolve_y(convolve_x(image, kx), ky);
}

std::vector<ImageF> basis_convolve(const ImageF& image,
                                   const GaussHermiteBasis& basis) {
  const int n_max = basis.n_max();

  // Share the x-pass across all components with the same nx.
  std::vector<ImageF> tx;
  tx.reserve(n_max + 1);
  for (int nx = 0; nx <= n_max; ++nx) {
    tx.push_back(convolve_x(image, basis.kernel1d(nx)));
  }

  std::vector<ImageF> out;
  out.reserve(basis.component_count());
  for (const auto& [nx, ny] : basis.orders()) {
    out.push_back(convolve_y(tx[nx], basis.kernel1d(ny)));
  }
  return out;
}

}  // namespace delta
