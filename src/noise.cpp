#include "delta/noise.hpp"

#include <fftw3.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace delta {

namespace {

constexpr double kPi = 3.14159265358979323846;

// RAII for FFTW single-precision buffers and plans (one block's worth).
struct Fft2d {
  int n;
  int nc;  // n/2 + 1 complex columns
  float* real;
  fftwf_complex* freq;
  fftwf_plan fwd;
  fftwf_plan inv;

  explicit Fft2d(int n_)
      : n(n_),
        nc(n_ / 2 + 1),
        real(fftwf_alloc_real(static_cast<std::size_t>(n_) * n_)),
        freq(fftwf_alloc_complex(static_cast<std::size_t>(n_) * (n_ / 2 + 1))) {
    fwd = fftwf_plan_dft_r2c_2d(n, n, real, freq, FFTW_ESTIMATE);
    inv = fftwf_plan_dft_c2r_2d(n, n, freq, real, FFTW_ESTIMATE);
  }
  ~Fft2d() {
    fftwf_destroy_plan(fwd);
    fftwf_destroy_plan(inv);
    fftwf_free(real);
    fftwf_free(freq);
  }
  Fft2d(const Fft2d&) = delete;
  Fft2d& operator=(const Fft2d&) = delete;
};

// |Khat(k)|^2 over the half-spectrum, for a kernel embedded zero-phase in n x n.
std::vector<double> kernel_power(const std::vector<float>& kernel, int ksize,
                                 int n, Fft2d& fft) {
  const std::size_t nn = static_cast<std::size_t>(n) * n;
  std::fill(fft.real, fft.real + nn, 0.0f);
  const int r = ksize / 2;
  for (int ky = 0; ky < ksize; ++ky) {
    const int gy = ((ky - r) % n + n) % n;  // zero-phase wrap
    for (int kx = 0; kx < ksize; ++kx) {
      const int gx = ((kx - r) % n + n) % n;
      fft.real[static_cast<std::size_t>(gy) * n + gx] =
          kernel[static_cast<std::size_t>(ky) * ksize + kx];
    }
  }
  fftwf_execute(fft.fwd);
  std::vector<double> power(static_cast<std::size_t>(n) * fft.nc);
  for (std::size_t i = 0; i < power.size(); ++i) {
    const double re = fft.freq[i][0], im = fft.freq[i][1];
    power[i] = re * re + im * im;
  }
  return power;
}

// Separable Hann window weight at (iy, ix) within an n x n block.
inline double hann2d(int iy, int ix, int n) {
  const double wy = 0.5 * (1.0 - std::cos(2.0 * kPi * (iy + 0.5) / n));
  const double wx = 0.5 * (1.0 - std::cos(2.0 * kPi * (ix + 0.5) / n));
  return wy * wx;
}

}  // namespace

std::vector<float> decorrelation_filter(const std::vector<float>& kernel,
                                        int ksize, double var_science,
                                        double var_reference, int n) {
  if (static_cast<int>(kernel.size()) != ksize * ksize)
    throw std::runtime_error("decorrelation_filter: kernel size mismatch");
  Fft2d fft(n);
  const std::vector<double> power = kernel_power(kernel, ksize, n, fft);

  double sumk2 = 0.0;
  for (float v : kernel) sumk2 += static_cast<double>(v) * v;
  const double num = var_science + var_reference * sumk2;
  const double cnum = std::sqrt(std::max(num, 0.0));

  std::vector<float> filter(power.size());
  for (std::size_t i = 0; i < power.size(); ++i) {
    const double denom = var_science + var_reference * power[i];
    filter[i] = denom > 0.0 ? static_cast<float>(cnum / std::sqrt(denom)) : 0.0f;
  }
  return filter;
}

ImageF apply_filter_fft(const ImageF& block, const std::vector<float>& filter,
                        int n) {
  if (static_cast<int>(block.width()) != n ||
      static_cast<int>(block.height()) != n)
    throw std::runtime_error("apply_filter_fft: block must be n x n");
  Fft2d fft(n);
  const std::size_t nn = static_cast<std::size_t>(n) * n;
  std::copy(block.data(), block.data() + nn, fft.real);
  fftwf_execute(fft.fwd);
  const std::size_t nspec = static_cast<std::size_t>(n) * fft.nc;
  if (filter.size() != nspec)
    throw std::runtime_error("apply_filter_fft: filter size mismatch");
  for (std::size_t i = 0; i < nspec; ++i) {
    fft.freq[i][0] *= filter[i];
    fft.freq[i][1] *= filter[i];
  }
  fftwf_execute(fft.inv);
  ImageF out(n, n);
  const float inv_n2 = 1.0f / static_cast<float>(nn);
  for (std::size_t i = 0; i < nn; ++i) out.data()[i] = fft.real[i] * inv_n2;
  return out;
}

ImageF decorrelation_kernel_image(const std::vector<float>& filter, int n) {
  Fft2d fft(n);
  const std::size_t nspec = static_cast<std::size_t>(n) * fft.nc;
  if (filter.size() != nspec)
    throw std::runtime_error("decorrelation_kernel_image: filter size mismatch");
  for (std::size_t i = 0; i < nspec; ++i) {
    fft.freq[i][0] = filter[i];
    fft.freq[i][1] = 0.0f;
  }
  fftwf_execute(fft.inv);
  const std::size_t nn = static_cast<std::size_t>(n) * n;
  const float inv_n2 = 1.0f / static_cast<float>(nn);
  // fft-shift so the (zero-phase) kernel is centred.
  ImageF out(n, n);
  const int half = n / 2;
  for (int y = 0; y < n; ++y) {
    const int sy = (y + half) % n;
    for (int x = 0; x < n; ++x) {
      const int sx = (x + half) % n;
      out.data()[static_cast<std::size_t>(y) * n + x] =
          fft.real[static_cast<std::size_t>(sy) * n + sx] * inv_n2;
    }
  }
  return out;
}

namespace {

// Reconstruct the local matching kernel K at integer location (cx, cy) from the
// spatial solution: K = sum_n a_n(cx,cy) phi_n, returned as a ksize^2 footprint.
std::vector<float> local_kernel(const ThinPlateBasis& spatial,
                                const Eigen::Ref<const Eigen::VectorXd>& theta,
                                const GaussHermiteBasis& basis, double cx,
                                double cy) {
  const int nc = basis.component_count();
  const int k = spatial.n_basis();
  const Eigen::MatrixXd c = Eigen::Map<const Eigen::MatrixXd>(theta.data(), k,
                                                              nc + 1);
  Eigen::MatrixXd point(1, 2);
  point(0, 0) = cx;
  point(0, 1) = cy;
  const Eigen::MatrixXd a = spatial.design(point) * c;  // 1 x (nc+1)

  const int ks = basis.ksize();
  std::vector<float> kernel(static_cast<std::size_t>(ks) * ks, 0.0f);
  for (int n = 0; n < nc; ++n) {
    const std::vector<float> phi = basis.kernel2d(n);
    const float an = static_cast<float>(a(0, n));
    for (std::size_t i = 0; i < kernel.size(); ++i) kernel[i] += an * phi[i];
  }
  return kernel;
}

// Mean of a variance map over the block region clipped to the image.
double block_mean(const ImageF& var, int bx, int by, int block, int w, int h) {
  double sum = 0.0;
  int count = 0;
  for (int y = by; y < by + block; ++y) {
    if (y < 0 || y >= h) continue;
    for (int x = bx; x < bx + block; ++x) {
      if (x < 0 || x >= w) continue;
      sum += var.data()[static_cast<std::size_t>(y) * w + x];
      ++count;
    }
  }
  return count > 0 ? sum / count : 0.0;
}

}  // namespace

ImageF decorrelate(const ImageF& difference, const ThinPlateBasis& spatial,
                   const Eigen::Ref<const Eigen::VectorXd>& theta,
                   const GaussHermiteBasis& basis, const ImageF& var_science,
                   const ImageF& var_reference, int block) {
  const int w = static_cast<int>(difference.width());
  const int h = static_cast<int>(difference.height());
  if (block < 8 || (block & (block - 1)) != 0)
    throw std::runtime_error("decorrelate: block must be a power of two >= 8");
  const int ks = basis.ksize();

  std::vector<double> acc(static_cast<std::size_t>(w) * h, 0.0);
  std::vector<double> wsum(static_cast<std::size_t>(w) * h, 0.0);

  const int stride = block / 2;
  // Block origins covering the frame, last block flush with the far edge.
  auto origins = [](int extent, int blk, int strd) {
    std::vector<int> o;
    if (extent <= blk) {
      o.push_back(0);
      return o;
    }
    for (int p = 0; p + blk <= extent; p += strd) o.push_back(p);
    if (o.back() + blk < extent) o.push_back(extent - blk);
    return o;
  };
  const std::vector<int> oxs = origins(w, block, stride);
  const std::vector<int> oys = origins(h, block, stride);

  // Flatten the block grid so the (independent, FFT-heavy) blocks distribute
  // over threads. Each thread builds its own FFTW plans/buffers once -- planning
  // is not thread-safe (guarded) and was previously redone for every block via
  // the per-call decorrelation_filter / apply_filter_fft helpers.
  std::vector<std::pair<int, int>> origins_xy;
  origins_xy.reserve(oxs.size() * oys.size());
  for (int by : oys)
    for (int bx : oxs) origins_xy.emplace_back(bx, by);

  const std::size_t nn = static_cast<std::size_t>(block) * block;
  const float inv_n2 = 1.0f / static_cast<float>(nn);

#pragma omp parallel
  {
    Fft2d* fftp = nullptr;
#pragma omp critical(delta_fftw_plan)
    { fftp = new Fft2d(block); }
    Fft2d& fft = *fftp;
    const std::size_t nspec = static_cast<std::size_t>(block) * fft.nc;
    std::vector<float> buf(nn);
    std::vector<float> filter(nspec);

#pragma omp for schedule(dynamic)
    for (std::size_t bi = 0; bi < origins_xy.size(); ++bi) {
      const int bx = origins_xy[bi].first, by = origins_xy[bi].second;

      // Reflect-pad the block region into a local buffer.
      for (int iy = 0; iy < block; ++iy) {
        int sy = by + iy;
        if (sy < 0) sy = -sy - 1;
        if (sy >= h) sy = 2 * h - sy - 1;
        sy = std::clamp(sy, 0, h - 1);
        for (int ix = 0; ix < block; ++ix) {
          int sx = bx + ix;
          if (sx < 0) sx = -sx - 1;
          if (sx >= w) sx = 2 * w - sx - 1;
          sx = std::clamp(sx, 0, w - 1);
          buf[static_cast<std::size_t>(iy) * block + ix] =
              difference.data()[static_cast<std::size_t>(sy) * w + sx];
        }
      }

      const double cx = std::clamp(bx + block / 2.0, 0.0, w - 1.0);
      const double cy = std::clamp(by + block / 2.0, 0.0, h - 1.0);
      const std::vector<float> kern = local_kernel(spatial, theta, basis, cx, cy);
      const double vs = block_mean(var_science, bx, by, block, w, h);
      const double vr = block_mean(var_reference, bx, by, block, w, h);

      // Decorrelation filter from |Khat|^2 (kernel_power reuses the thread's FFT
      // buffers, so it must run before the block data is loaded into fft.real).
      const std::vector<double> power = kernel_power(kern, ks, block, fft);
      double sumk2 = 0.0;
      for (float v : kern) sumk2 += static_cast<double>(v) * v;
      const double cnum = std::sqrt(std::max(vs + vr * sumk2, 0.0));
      for (std::size_t i = 0; i < nspec; ++i) {
        const double denom = vs + vr * power[i];
        filter[i] = denom > 0.0 ? static_cast<float>(cnum / std::sqrt(denom)) : 0.0f;
      }

      // Apply the filter to this block.
      std::copy(buf.begin(), buf.end(), fft.real);
      fftwf_execute(fft.fwd);
      for (std::size_t i = 0; i < nspec; ++i) {
        fft.freq[i][0] *= filter[i];
        fft.freq[i][1] *= filter[i];
      }
      fftwf_execute(fft.inv);

      // Hann-weighted blend into the shared accumulators (cheap; serialised).
#pragma omp critical(delta_decorr_blend)
      for (int iy = 0; iy < block; ++iy) {
        const int y = by + iy;
        if (y < 0 || y >= h) continue;
        for (int ix = 0; ix < block; ++ix) {
          const int x = bx + ix;
          if (x < 0 || x >= w) continue;
          const double wgt = hann2d(iy, ix, block);
          const std::size_t i = static_cast<std::size_t>(y) * w + x;
          acc[i] += wgt * fft.real[static_cast<std::size_t>(iy) * block + ix] * inv_n2;
          wsum[i] += wgt;
        }
      }
    }

#pragma omp critical(delta_fftw_plan)
    { delete fftp; }
  }

  ImageF out(w, h);
  for (std::size_t i = 0; i < acc.size(); ++i)
    out.data()[i] = wsum[i] > 0.0 ? static_cast<float>(acc[i] / wsum[i]) : 0.0f;
  return out;
}

ImageF matched_filter(const ImageF& image, const std::vector<float>& psf,
                      int psf_size, double noise_var) {
  if (static_cast<int>(psf.size()) != psf_size * psf_size)
    throw std::runtime_error("matched_filter: psf size mismatch");
  const int w = static_cast<int>(image.width());
  const int h = static_cast<int>(image.height());
  const int r = psf_size / 2;

  double sumpsf2 = 0.0;
  for (float v : psf) sumpsf2 += static_cast<double>(v) * v;
  const double norm = std::sqrt(std::max(noise_var, 0.0) * sumpsf2);

  ImageF out(w, h);
#pragma omp parallel for schedule(static)
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      double sum = 0.0;
      for (int dy = -r; dy <= r; ++dy) {
        const int sy = y + dy;
        if (sy < 0 || sy >= h) continue;
        for (int dx = -r; dx <= r; ++dx) {
          const int sx = x + dx;
          if (sx < 0 || sx >= w) continue;
          const float p = psf[static_cast<std::size_t>(dy + r) * psf_size + (dx + r)];
          sum += p * image.data()[static_cast<std::size_t>(sy) * w + sx];
        }
      }
      out.data()[static_cast<std::size_t>(y) * w + x] =
          norm > 0.0 ? static_cast<float>(sum / norm) : 0.0f;
    }
  }
  return out;
}

}  // namespace delta
