#include "delta/convolve.hpp"

#include <cstddef>

namespace delta {
namespace {

// 1-D convolution along x (fast axis), zero-padded, 'same' size.
// out[y,x] = sum_m k[m+h] * in[y, x-m].
ImageF convolve_x(const ImageF& in, const std::vector<float>& k) {
  const int width = static_cast<int>(in.width());
  const int height = static_cast<int>(in.height());
  const int h = static_cast<int>(k.size() - 1) / 2;

  ImageF out(in.width(), in.height());
  const float* src = in.data();
  float* dst = out.data();

#pragma omp parallel for schedule(static)
  for (int y = 0; y < height; ++y) {
    const std::size_t row = static_cast<std::size_t>(y) * width;
    for (int x = 0; x < width; ++x) {
      double acc = 0.0;
      const int m_lo = (x - (width - 1) > -h) ? (x - (width - 1)) : -h;
      const int m_hi = (x < h) ? x : h;
      for (int m = m_lo; m <= m_hi; ++m) {
        acc += static_cast<double>(k[m + h]) * src[row + (x - m)];
      }
      dst[row + x] = static_cast<float>(acc);
    }
  }
  return out;
}

// 1-D convolution along y, zero-padded, 'same' size.
// out[y,x] = sum_n k[n+h] * in[y-n, x].
ImageF convolve_y(const ImageF& in, const std::vector<float>& k) {
  const int width = static_cast<int>(in.width());
  const int height = static_cast<int>(in.height());
  const int h = static_cast<int>(k.size() - 1) / 2;

  ImageF out(in.width(), in.height());
  const float* src = in.data();
  float* dst = out.data();

#pragma omp parallel for schedule(static)
  for (int y = 0; y < height; ++y) {
    const std::size_t row = static_cast<std::size_t>(y) * width;
    const int n_lo = (y - (height - 1) > -h) ? (y - (height - 1)) : -h;
    const int n_hi = (y < h) ? y : h;
    for (int x = 0; x < width; ++x) {
      double acc = 0.0;
      for (int n = n_lo; n <= n_hi; ++n) {
        acc += static_cast<double>(k[n + h]) *
               src[static_cast<std::size_t>(y - n) * width + x];
      }
      dst[row + x] = static_cast<float>(acc);
    }
  }
  return out;
}

}  // namespace

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
