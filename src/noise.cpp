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

// Periodic Hann window samples over an n-point block (the separable factor used
// for overlap-add blending). Precomputed once: the blend touches every pixel of
// every overlapping block, so evaluating cos per pixel dominated the pass.
inline std::vector<double> hann1d(int n) {
  std::vector<double> w(n);
  for (int i = 0; i < n; ++i)
    w[i] = 0.5 * (1.0 - std::cos(2.0 * kPi * (i + 0.5) / n));
  return w;
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
                                const GaussHermiteBasis& basis,
                                const std::vector<std::vector<float>>& phi,
                                double cx, double cy) {
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
    const float an = static_cast<float>(a(0, n));
    const float* p = phi[n].data();
    for (std::size_t i = 0; i < kernel.size(); ++i) kernel[i] += an * p[i];
  }
  return kernel;
}

// Robust block-level noise variance: the median of the (finite, positive)
// variance over the block region clipped to the image. The median (rather than
// the mean) is deliberate: callers flag bad/masked pixels with a large sentinel
// variance (e.g. 1e30), and a plain mean lets even a handful of those poison the
// whole block -- a hugely inflated `vr` makes the decorrelation filter blow up at
// high frequencies (where |Khat|^2 -> 0) and paints large artifacts around masked
// regions and the frame edge. The median is immune to a minority of fill pixels
// and is also a cleaner white-noise level estimate than the mean (which sources
// and the sentinel both bias high).
double block_variance(const ImageF& var, int bx, int by, int block, int w,
                      int h) {
  // The block noise level is a single robust scalar over thousands of pixels;
  // a regular subsample of the block estimates the same median to far better
  // than the decorrelation filter cares about, at a fraction of the gather +
  // nth_element cost (this runs twice per overlap-add block). Step so large
  // blocks draw ~64 samples per axis (~4k total -- ample for a median that is
  // immune to a minority of fill/sentinel pixels) while small blocks stay
  // exact.
  const int step = std::max(1, block / 64);
  std::vector<float> vals;
  vals.reserve(static_cast<std::size_t>(block / step + 1) * (block / step + 1));
  for (int y = by; y < by + block; y += step) {
    if (y < 0 || y >= h) continue;
    for (int x = bx; x < bx + block; x += step) {
      if (x < 0 || x >= w) continue;
      const float v = var.data()[static_cast<std::size_t>(y) * w + x];
      if (std::isfinite(v) && v > 0.0f) vals.push_back(v);
    }
  }
  if (vals.empty()) return 0.0;
  const std::size_t mid = vals.size() / 2;
  std::nth_element(vals.begin(), vals.begin() + mid, vals.end());
  return static_cast<double>(vals[mid]);
}

}  // namespace

ImageF decorrelate(const ImageF& difference, const ThinPlateBasis& spatial,
                   const Eigen::Ref<const Eigen::VectorXd>& theta,
                   const GaussHermiteBasis& basis, const ImageF& var_science,
                   const ImageF& var_reference, int block,
                   int kernel_cell_blocks) {
  const int w = static_cast<int>(difference.width());
  const int h = static_cast<int>(difference.height());
  if (block < 8 || (block & (block - 1)) != 0)
    throw std::runtime_error("decorrelate: block must be a power of two >= 8");
  const int ks = basis.ksize();

  std::vector<double> acc(static_cast<std::size_t>(w) * h, 0.0);

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

  // Component footprints, built once and shared across every block's local-kernel
  // reconstruction (they are frame-invariant; rebuilding all nc per block churned
  // ~nc allocations per block over the thousands of overlap-add blocks).
  const int nc = basis.component_count();
  std::vector<std::vector<float>> phi(nc);
  for (int n = 0; n < nc; ++n) phi[n] = basis.kernel2d(n);

  // Flatten the block grid so the (independent, FFT-heavy) blocks distribute
  // over threads. Each thread builds its own FFTW plans/buffers once -- planning
  // is not thread-safe (guarded) and was previously redone for every block via
  // the per-call decorrelation_filter / apply_filter_fft helpers.
  //
  // The Hann overlap-add writes into the shared `acc`, and adjacent blocks'
  // 50%-overlapping footprints touch the same pixels, so a naive blend has to be
  // serialised against a lock. Instead, 9-colour the block grid by
  // (ix % 3, iy % 3): two distinct same-colour blocks either share a column
  // (|dix| = 0 => |diy| >= 3) or differ by >= 3 columns, and a 3-index
  // separation along either axis is >= block even for the irregular flush origin
  // (origins[L] - origins[L-3] >= 2*stride = block). So same-colour blocks never
  // touch the same pixel and blend lock-free; a barrier between colours keeps
  // cross-colour writes from racing.
  // --- coarse kernel-power cache -------------------------------------------------
  // |Khat(k)|^2 and sum K^2 (the only kernel-derived inputs to the filter) depend
  // solely on the matching kernel K = sum_n a_n(x,y) phi_n, which varies on the knot
  // length-scale -- far slower than the block stride. Cache them per coarse cell of
  // kgroup x kgroup blocks (one kernel FFT per cell) and reuse across the cell's
  // blocks: this drops the per-block kernel FFT (1 of the 3 FFTs/block) while the two
  // data FFTs and the per-block noise levels vs, vr stay exact. K is frozen on the
  // cell lattice -- the same block-effective approximation as variance propagation.
  const int nox = static_cast<int>(oxs.size());
  const int noy = static_cast<int>(oys.size());
  int kgroup = kernel_cell_blocks;
  if (kgroup <= 0) {
    // Auto: aim for ~kKernelCellSamplesPerKnot cells per knot interval. The kernel's
    // spatial variation is band-limited to the knot scale, so a few samples per knot
    // resolve it; this is much coarser than the field interpolation lattice because
    // it only freezes the (smooth) kernel, not the per-block noise.
    constexpr int kKernelCellSamplesPerKnot = 4;
    const double spacing = spatial.min_knot_spacing();
    kgroup = 1;
    if (spacing > 0.0 && stride > 0) {
      const double cell_px = spacing / kKernelCellSamplesPerKnot;
      kgroup = std::max(1, static_cast<int>(std::lround(cell_px / stride)));
    }
  }
  const int ncellx = (nox + kgroup - 1) / kgroup;
  const int ncelly = (noy + kgroup - 1) / kgroup;
  const int ncell = ncellx * ncelly;

  const int fnc = block / 2 + 1;
  const std::size_t nspec = static_cast<std::size_t>(block) * fnc;

  // Per-cell cached |Khat|^2 (nspec each) and sum K^2, evaluated at the cell's
  // central block. Built up front (one FFT per cell) over the thread team.
  std::vector<double> cell_power(static_cast<std::size_t>(ncell) * nspec);
  std::vector<double> cell_sumk2(ncell);
#pragma omp parallel
  {
    Fft2d* fftp = nullptr;
#pragma omp critical(delta_fftw_plan)
    { fftp = new Fft2d(block); }
    Fft2d& fft = *fftp;
#pragma omp for schedule(dynamic)
    for (int ci = 0; ci < ncell; ++ci) {
      const int cix = ci % ncellx, ciy = ci / ncellx;
      const int repx = std::min(cix * kgroup + kgroup / 2, nox - 1);
      const int repy = std::min(ciy * kgroup + kgroup / 2, noy - 1);
      const double cx = std::clamp(oxs[repx] + block / 2.0, 0.0, w - 1.0);
      const double cy = std::clamp(oys[repy] + block / 2.0, 0.0, h - 1.0);
      const std::vector<float> kern =
          local_kernel(spatial, theta, basis, phi, cx, cy);
      const std::vector<double> power = kernel_power(kern, ks, block, fft);
      std::copy(power.begin(), power.end(),
                cell_power.begin() + static_cast<std::size_t>(ci) * nspec);
      double sumk2 = 0.0;
      for (float v : kern) sumk2 += static_cast<double>(v) * v;
      cell_sumk2[ci] = sumk2;
    }
#pragma omp critical(delta_fftw_plan)
    { delete fftp; }
  }

  struct BlockJob {
    int bx, by, cell;
  };
  std::vector<std::vector<BlockJob>> color_blocks(9);
  for (int iy = 0; iy < noy; ++iy)
    for (int ix = 0; ix < nox; ++ix) {
      const int cell = (iy / kgroup) * ncellx + (ix / kgroup);
      color_blocks[(ix % 3) * 3 + (iy % 3)].push_back({oxs[ix], oys[iy], cell});
    }

  const std::size_t nn = static_cast<std::size_t>(block) * block;
  const float inv_n2 = 1.0f / static_cast<float>(nn);

  // The overlap-add weight sum is separable: the blocks form a full oxs x oys
  // grid, so wsum(x,y) = (sum_bx hann(x-bx)) * (sum_by hann(y-by)). Build the two
  // 1-D total-weight profiles once instead of accumulating a w*h weight image in
  // the (serialised) blend.
  const std::vector<double> hwin = hann1d(block);
  std::vector<double> wx_total(w, 0.0), wy_total(h, 0.0);
  for (int bx : oxs)
    for (int ix = 0; ix < block; ++ix) {
      const int x = bx + ix;
      if (x >= 0 && x < w) wx_total[x] += hwin[ix];
    }
  for (int by : oys)
    for (int iy = 0; iy < block; ++iy) {
      const int y = by + iy;
      if (y >= 0 && y < h) wy_total[y] += hwin[iy];
    }

#pragma omp parallel
  {
    Fft2d* fftp = nullptr;
#pragma omp critical(delta_fftw_plan)
    { fftp = new Fft2d(block); }
    Fft2d& fft = *fftp;
    std::vector<float> buf(nn);
    std::vector<float> filter(nspec);

    // One worksharing pass per colour; the implicit barrier at the end of each
    // `omp for` orders the colours so cross-colour writes never overlap in time.
    for (const auto& blocks : color_blocks) {
#pragma omp for schedule(dynamic)
      for (std::size_t bi = 0; bi < blocks.size(); ++bi) {
        const int bx = blocks[bi].bx, by = blocks[bi].by;

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

        const double vs = block_variance(var_science, bx, by, block, w, h);
        const double vr = block_variance(var_reference, bx, by, block, w, h);

        // Decorrelation filter from the cell's cached |Khat|^2 and sum K^2; only the
        // per-block noise levels vs, vr enter here, so the (slowly varying) kernel
        // power is shared across the cell without an FFT per block.
        const double* power =
            cell_power.data() + static_cast<std::size_t>(blocks[bi].cell) * nspec;
        const double cnum =
            std::sqrt(std::max(vs + vr * cell_sumk2[blocks[bi].cell], 0.0));
        for (std::size_t i = 0; i < nspec; ++i) {
          const double denom = vs + vr * power[i];
          filter[i] =
              denom > 0.0 ? static_cast<float>(cnum / std::sqrt(denom)) : 0.0f;
        }

        // Apply the filter to this block.
        std::copy(buf.begin(), buf.end(), fft.real);
        fftwf_execute(fft.fwd);
        for (std::size_t i = 0; i < nspec; ++i) {
          fft.freq[i][0] *= filter[i];
          fft.freq[i][1] *= filter[i];
        }
        fftwf_execute(fft.inv);

        // Hann-weighted overlap-add into the shared accumulator. The weight is
        // the precomputed separable window; wsum is handled analytically (see
        // above), so the blend only touches acc. Lock-free: same-colour blocks
        // are pairwise disjoint, so no two threads in this pass write a pixel.
        for (int iy = 0; iy < block; ++iy) {
          const int y = by + iy;
          if (y < 0 || y >= h) continue;
          const double wy = hwin[iy] * inv_n2;
          const float* frow = fft.real + static_cast<std::size_t>(iy) * block;
          double* arow = acc.data() + static_cast<std::size_t>(y) * w;
          for (int ix = 0; ix < block; ++ix) {
            const int x = bx + ix;
            if (x < 0 || x >= w) continue;
            arow[x] += wy * hwin[ix] * frow[ix];
          }
        }
      }
    }

#pragma omp critical(delta_fftw_plan)
    { delete fftp; }
  }

  ImageF out(w, h);
#pragma omp parallel for schedule(static)
  for (int y = 0; y < h; ++y) {
    const double wy = wy_total[y];
    const double* arow = acc.data() + static_cast<std::size_t>(y) * w;
    float* orow = out.data() + static_cast<std::size_t>(y) * w;
    for (int x = 0; x < w; ++x) {
      const double wsum = wy * wx_total[x];
      orow[x] = wsum > 0.0 ? static_cast<float>(arow[x] / wsum) : 0.0f;
    }
  }
  return out;
}

ImageF matched_filter(const ImageF& image, const std::vector<float>& psf,
                      int psf_size, const ImageF& variance) {
  if (static_cast<int>(psf.size()) != psf_size * psf_size)
    throw std::runtime_error("matched_filter: psf size mismatch");
  const int w = static_cast<int>(image.width());
  const int h = static_cast<int>(image.height());
  if (static_cast<int>(variance.width()) != w ||
      static_cast<int>(variance.height()) != h)
    throw std::runtime_error("matched_filter: variance size mismatch");
  const int ks = psf_size;
  const int r = psf_size / 2;

  float sumpsf2 = 0.0f;
  for (float v : psf) sumpsf2 += v * v;

  ImageF out(w, h);

  // Cache-blocked correlation: out(x) = (1/norm(x)) sum_u psf(u) image(x+u).
  // Done naively (output-outer, kernel-inner) every output pixel re-reads the
  // whole ks x ks window straight off the strided frame, thrashing the cache.
  // Instead tile the frame, gather each tile's haloed input window once into a
  // small contiguous buffer (so it stays hot in L1/L2 across the ks^2 taps),
  // and apply the PSF taps as unit-stride multiply-adds. Unlike the variance
  // pass this is a correlation, so the PSF is applied directly (no axis mirror).
  // Normalisation is per-pixel: norm(x) = sqrt(var(x) * sumpsf2).
  const float* img = image.data();
  const float* var = variance.data();
  constexpr int bsz = 64;
  const int ibw = bsz + 2 * r;  // gathered window side (tile + halo)
  std::vector<std::pair<int, int>> blocks;
  for (int by = 0; by < h; by += bsz)
    for (int bx = 0; bx < w; bx += bsz) blocks.emplace_back(bx, by);

#pragma omp parallel
  {
    std::vector<float> win(static_cast<std::size_t>(ibw) * ibw);
    std::vector<float> accrow(bsz);
#pragma omp for schedule(dynamic)
    for (std::size_t bi = 0; bi < blocks.size(); ++bi) {
      const int bx = blocks[bi].first, by = blocks[bi].second;
      const int bw = std::min(bsz, w - bx);
      const int bh = std::min(bsz, h - by);

      // Gather the (bh+2r) x (bw+2r) input window, zero-padded at the frame
      // edge (matching the original zero-padding contract), into a contiguous
      // buffer.
      const int iwh = bh + 2 * r;
      const int iww = bw + 2 * r;
      for (int iy = 0; iy < iwh; ++iy) {
        const int gy = by - r + iy;
        float* wrow = win.data() + static_cast<std::size_t>(iy) * ibw;
        if (gy < 0 || gy >= h) {
          std::fill(wrow, wrow + iww, 0.0f);
          continue;
        }
        const float* irow = img + static_cast<std::size_t>(gy) * w;
        for (int ix = 0; ix < iww; ++ix) {
          const int gx = bx - r + ix;
          wrow[ix] = (gx < 0 || gx >= w) ? 0.0f : irow[gx];
        }
      }

      // Output-row-outer correlation on the cache-resident window. The earlier
      // tap-outer order swept the *whole tile* once per (ly,lx) tap, so `acc`
      // (bh*bw) and `win` together overran L1 and `win` thrashed across the ks^2
      // taps. Here a single output row's accumulator (`accrow`, bw floats) stays
      // L1/register-resident while all ks^2 taps reduce into it, and each tap
      // reads only the ks haloed `win` rows above this output row (reused across
      // the lx shifts). Each output pixel still sums the taps in the same (ly,lx)
      // order, so the result is bit-identical. The per-row normalise is folded in
      // so `acc` is never re-streamed.
      for (int oy = 0; oy < bh; ++oy) {
        std::fill(accrow.begin(), accrow.begin() + bw, 0.0f);
        for (int ly = 0; ly < ks; ++ly) {
          const float* prow = psf.data() + static_cast<std::size_t>(ly) * ks;
          const float* wrow =
              win.data() + static_cast<std::size_t>(oy + ly) * ibw;
          for (int lx = 0; lx < ks; ++lx) {
            const float p = prow[lx];
            if (p == 0.0f) continue;
            const float* src = wrow + lx;
#pragma omp simd
            for (int ox = 0; ox < bw; ++ox) accrow[ox] += p * src[ox];
          }
        }
        // Per-pixel normalise + scatter: score = conv / sqrt(var * sumpsf2).
        float* orow = out.data() + static_cast<std::size_t>(by + oy) * w + bx;
        const float* vrow = var + static_cast<std::size_t>(by + oy) * w + bx;
        for (int ox = 0; ox < bw; ++ox) {
          const float v = vrow[ox];
          orow[ox] = v > 0.0f ? accrow[ox] / std::sqrt(v * sumpsf2) : 0.0f;
        }
      }
    }
  }
  return out;
}

ImageF matched_filter(const ImageF& image, const std::vector<float>& psf,
                      int psf_size, double noise_var) {
  const int w = static_cast<int>(image.width());
  const int h = static_cast<int>(image.height());
  ImageF var(w, h);
  std::fill(var.pixels().begin(), var.pixels().end(),
            static_cast<float>(std::max(noise_var, 0.0)));
  return matched_filter(image, psf, psf_size, var);
}

}  // namespace delta
