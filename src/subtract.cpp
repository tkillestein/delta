#include "delta/subtract.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "delta/convolve.hpp"
#include "delta/timing.hpp"

namespace delta {

namespace {

// Reshape theta into the (k x (nc+1)) coefficient matrix C whose column j is
// theta[j*k : (j+1)*k]. theta is laid out as consecutive field blocks, so a
// column-major map is exactly that reshape.
Eigen::MatrixXd coeff_matrix(const Eigen::Ref<const Eigen::VectorXd>& theta,
                             int k, int n_fields) {
  if (theta.size() != static_cast<Eigen::Index>(k) * n_fields)
    throw std::runtime_error("subtract: theta size does not match k*(nc+1)");
  return Eigen::Map<const Eigen::MatrixXd>(theta.data(), k, n_fields);
}

// A copy of the reference with masked / non-finite pixels replaced by the image
// median (the background level). Convolving the raw reference smears NaNs and the
// spurious values under bad pixels across the kernel footprint, polluting B_n
// within a kernel radius of any defect (SPEC §3.6). Filling with the *median*
// rather than zero matters at mask boundaries: a zero fill convolves a hard step
// (0 vs background) and rings the model just outside every masked region, which
// shows up as cruft around the masked template and the frame edge. Median fill
// removes that step, so the model stays smooth right up to the (dilated-masked)
// boundary. The median is estimated from a strided sample of the good pixels.
ImageF sanitised_reference(const ImageF& reference) {
  ImageF clean(reference.width(), reference.height());
  const std::size_t n = reference.size();
  const float* src = reference.data();
  float* dst = clean.data();
  const bool has_mask = reference.has_mask();
  const MaskType* m = has_mask ? reference.mask().data() : nullptr;

  // Background estimate: median of a strided sample of finite, unmasked pixels.
  std::vector<float> sample;
  sample.reserve(n / 16 + 1);
  for (std::size_t i = 0; i < n; i += 7) {
    const float v = src[i];
    if (std::isfinite(v) && !(has_mask && m[i] != kMaskGood)) sample.push_back(v);
  }
  float fill = 0.0f;
  if (!sample.empty()) {
    std::nth_element(sample.begin(), sample.begin() + sample.size() / 2,
                     sample.end());
    fill = sample[sample.size() / 2];
  }

  for (std::size_t i = 0; i < n; ++i) {
    const float v = src[i];
    const bool bad = (has_mask && m[i] != kMaskGood) || !std::isfinite(v);
    dst[i] = bad ? fill : v;
  }
  return clean;
}

// Separable OR-dilation of a uint8 bitmask by Chebyshev radius r, in place.
// A square (box) structuring element factors into an x-pass then a y-pass; each
// output pixel is the bitwise-OR of its neighbours within +/- r along that axis.
// This is the exact same footprint as the naive nested-box fill but runs in
// O(N r) instead of O(N r^2), and preserves which flag bits were set.
void dilate_mask(std::vector<MaskType>& mask, int w, int h, int r) {
  if (r <= 0) return;
  std::vector<MaskType> tmp(mask.size(), kMaskGood);
  // x-pass: mask -> tmp
  for (int y = 0; y < h; ++y) {
    const std::size_t row = static_cast<std::size_t>(y) * w;
    for (int x = 0; x < w; ++x) {
      const int x0 = std::max(0, x - r), x1 = std::min(w - 1, x + r);
      MaskType acc = kMaskGood;
      for (int xx = x0; xx <= x1; ++xx) acc |= mask[row + xx];
      tmp[row + x] = acc;
    }
  }
  // y-pass: tmp -> mask
  for (int y = 0; y < h; ++y) {
    const int y0 = std::max(0, y - r), y1 = std::min(h - 1, y + r);
    for (int x = 0; x < w; ++x) {
      MaskType acc = kMaskGood;
      for (int yy = y0; yy <= y1; ++yy)
        acc |= tmp[static_cast<std::size_t>(yy) * w + x];
      mask[static_cast<std::size_t>(y) * w + x] = acc;
    }
  }
}

}  // namespace

namespace {

// Coarse evaluation lattice for spatial-field evaluation. The thin-plate fields vary
// on the knot length-scale (>> a pixel; SPEC §3.2/§3.5), so evaluating them exactly
// on a coarse lattice and bilinearly interpolating to full resolution is near-exact
// at a fraction of the cost. Bilinear error scales as ~(stride / knot-spacing)^2, so
// the stride is taken as a fraction of the smallest knot spacing (>= ~16 samples per
// knot interval keeps the relative error well below 1e-3), capped so it stays cheap.
constexpr int kSamplesPerKnotInterval = 16;
constexpr int kMaxFieldStride = 64;
// Below this stride the coarse grid is barely coarser than full resolution, so the
// exact per-pixel path is used instead (it is also bit-for-bit for tests).
constexpr int kMinFieldStride = 4;

int choose_field_stride(const ThinPlateBasis& spatial) {
  const double spacing = spatial.min_knot_spacing();
  if (!(spacing > 0.0)) return 1;
  const int s = static_cast<int>(spacing / kSamplesPerKnotInterval);
  return std::clamp(s, 1, kMaxFieldStride);
}

// Exact per-pixel field evaluation: for each row, design(points) * C scattered into
// the full-resolution coefficient/background images. The reference implementation,
// used directly for small frames where interpolation buys nothing.
void evaluate_fields_exact(const ThinPlateBasis& spatial, const Eigen::MatrixXd& c,
                           int n_components, int w, int h, SpatialFields& out) {
#pragma omp parallel for schedule(static)
  for (int y = 0; y < h; ++y) {
    Eigen::MatrixXd points(w, 2);
    for (int x = 0; x < w; ++x) {
      points(x, 0) = static_cast<double>(x);
      points(x, 1) = static_cast<double>(y);
    }
    const Eigen::MatrixXd fields = spatial.design(points) * c;  // w x n_fields
    for (int n = 0; n < n_components; ++n) {
      float* row = out.coeff[n].data() + static_cast<std::size_t>(y) * w;
      for (int x = 0; x < w; ++x) row[x] = static_cast<float>(fields(x, n));
    }
    float* bg = out.background.data() + static_cast<std::size_t>(y) * w;
    for (int x = 0; x < w; ++x)
      bg[x] = static_cast<float>(fields(x, n_components));
  }
}

// Coarse sample coordinates for one axis: 0, S, 2S, ... with the last entry pinned
// to len-1 so interpolation brackets every full-resolution pixel.
std::vector<int> coarse_coords(int len, int stride) {
  std::vector<int> g;
  for (int p = 0; p < len - 1; p += stride) g.push_back(p);
  g.push_back(len - 1);
  return g;
}

}  // namespace

SpatialFields evaluate_fields(const ThinPlateBasis& spatial,
                              const Eigen::Ref<const Eigen::VectorXd>& theta,
                              int n_components, std::size_t width,
                              std::size_t height) {
  const int k = spatial.n_basis();
  const int n_fields = n_components + 1;  // kernel coeffs + background
  const Eigen::MatrixXd c = coeff_matrix(theta, k, n_fields);

  SpatialFields out;
  out.coeff.assign(n_components, ImageF(width, height));
  out.background = ImageF(width, height);

  const int w = static_cast<int>(width);
  const int h = static_cast<int>(height);

  // Stride chosen from the knot spacing; a frame coarser than the lattice or a
  // sub-threshold stride falls back to exact per-pixel evaluation (also bit-for-bit
  // for tests). 2*stride guarantees >= 3 coarse samples per axis so the bilinear
  // brackets are always non-degenerate.
  const int stride = choose_field_stride(spatial);
  if (stride < kMinFieldStride || w <= 2 * stride || h <= 2 * stride) {
    evaluate_fields_exact(spatial, c, n_components, w, h, out);
    return out;
  }

  // --- evaluate fields exactly on a coarse lattice -------------------------------
  const std::vector<int> gx = coarse_coords(w, stride);
  const std::vector<int> gy = coarse_coords(h, stride);
  const int ncx = static_cast<int>(gx.size());
  const int ncy = static_cast<int>(gy.size());

  // coarse[(field*ncy + iy)*ncx + ix] -- one contiguous (ncy x ncx) block per field.
  std::vector<double> coarse(static_cast<std::size_t>(n_fields) * ncy * ncx);
#pragma omp parallel for schedule(static)
  for (int iy = 0; iy < ncy; ++iy) {
    Eigen::MatrixXd points(ncx, 2);
    for (int ix = 0; ix < ncx; ++ix) {
      points(ix, 0) = static_cast<double>(gx[ix]);
      points(ix, 1) = static_cast<double>(gy[iy]);
    }
    const Eigen::MatrixXd fields = spatial.design(points) * c;  // ncx x n_fields
    for (int f = 0; f < n_fields; ++f) {
      double* dst = coarse.data() +
                    (static_cast<std::size_t>(f) * ncy + iy) * ncx;
      for (int ix = 0; ix < ncx; ++ix) dst[ix] = fields(ix, f);
    }
  }

  // --- bilinearly interpolate to full resolution ---------------------------------
  // Per-axis lookup: containing coarse cell + fractional weight, computed once.
  std::vector<int> ix0(w);
  std::vector<float> wx(w);
  for (int x = 0; x < w; ++x) {
    int c0 = std::min(x / stride, ncx - 2);
    const int x0 = gx[c0], x1 = gx[c0 + 1];
    ix0[x] = c0;
    wx[x] = x1 > x0 ? static_cast<float>(x - x0) / (x1 - x0) : 0.0f;
  }
  std::vector<int> iy0(h);
  std::vector<float> wy(h);
  for (int y = 0; y < h; ++y) {
    int c0 = std::min(y / stride, ncy - 2);
    const int y0 = gy[c0], y1 = gy[c0 + 1];
    iy0[y] = c0;
    wy[y] = y1 > y0 ? static_cast<float>(y - y0) / (y1 - y0) : 0.0f;
  }

#pragma omp parallel for schedule(static)
  for (int y = 0; y < h; ++y) {
    const int jy0 = iy0[y], jy1 = jy0 + 1;
    const float fy = wy[y];
    for (int f = 0; f < n_fields; ++f) {
      const double* c00 = coarse.data() +
                          (static_cast<std::size_t>(f) * ncy + jy0) * ncx;
      const double* c10 = coarse.data() +
                          (static_cast<std::size_t>(f) * ncy + jy1) * ncx;
      float* dst = (f < n_components ? out.coeff[f].data()
                                     : out.background.data()) +
                   static_cast<std::size_t>(y) * w;
      for (int x = 0; x < w; ++x) {
        const int jx0 = ix0[x], jx1 = jx0 + 1;
        const float fx = wx[x];
        const double top = c00[jx0] + (c00[jx1] - c00[jx0]) * fx;
        const double bot = c10[jx0] + (c10[jx1] - c10[jx0]) * fx;
        dst[x] = static_cast<float>(top + (bot - top) * fy);
      }
    }
  }
  return out;
}

ImageF subtract(const ImageF& science, const ImageF& reference,
                const ThinPlateBasis& spatial,
                const Eigen::Ref<const Eigen::VectorXd>& theta,
                const GaussHermiteBasis& basis, double saturation) {
  const std::size_t w = science.width();
  const std::size_t h = science.height();
  if (reference.width() != w || reference.height() != h)
    throw std::runtime_error("subtract: science/reference shape mismatch");

  // D = S - b - sum_n a_n (phi_n (x) R)   (SPEC §3.2). The reference is sanitised
  // first so masked/non-finite pixels do not bleed into the model.
  const int nc = basis.component_count();
  const SpatialFields fields = [&] {
    DELTA_TIME("subtract: spatial fields");
    return evaluate_fields(spatial, theta, nc, w, h);
  }();
  const ImageF clean = [&] {
    DELTA_TIME("subtract: sanitise reference");
    return sanitised_reference(reference);
  }();
  const int hh = static_cast<int>(h);
  const int ww = static_cast<int>(w);

  // Share the separable x-pass across components with the same nx (n_max+1 of
  // them), then fuse every component's y-pass + a_n accumulation into the running
  // difference. Materialising the full nc-component B_n stack (and re-reading it
  // in a combine pass) was memory-bound; this never materialises B_n at all.
  ImageF diff(w, h);
  {
    DELTA_TIME("subtract: model convolve");
    const int n_max = basis.n_max();
    std::vector<ImageF> tx;
    tx.reserve(n_max + 1);
    for (int nx = 0; nx <= n_max; ++nx)
      tx.push_back(convolve_x(clean, basis.kernel1d(nx)));

    const auto& orders = basis.orders();
    std::vector<std::vector<float>> ky(nc);
    for (int n = 0; n < nc; ++n) ky[n] = basis.kernel1d(orders[n].second);
    const int kh = (basis.ksize() - 1) / 2;

    const float* s = science.data();
    const float* bg = fields.background.data();

    // Cache-blocked, fully fused model convolve. The earlier version looped
    // components on the outside and ran a full-frame y-pass each, so the running
    // diff was read-modify-written nc(=28) times and every tx[nx] was re-read
    // from DRAM per component -- memory-bandwidth bound (the pipeline's headline
    // bottleneck). Here we tile the frame: a tile-local diff accumulator and the
    // shared x-passes for the tile's rows (+halo) all fit in L2, so we sweep all
    // components over the tile while diff stays hot and write it out once. The
    // per-pixel accumulation order (component order) is unchanged, so the result
    // is bit-identical to the streamed version.
    constexpr int tw = 128;
    constexpr int th = 128;
    std::vector<std::pair<int, int>> tiles;
    for (int by = 0; by < hh; by += th)
      for (int bx = 0; bx < ww; bx += tw) tiles.emplace_back(bx, by);

#pragma omp parallel
    {
      std::vector<float> dtile(static_cast<std::size_t>(tw) * th);
      std::vector<float> bn(tw);
#pragma omp for schedule(dynamic)
      for (std::size_t ti = 0; ti < tiles.size(); ++ti) {
        const int bx = tiles[ti].first, by = tiles[ti].second;
        const int bw = std::min(tw, ww - bx);
        const int bh = std::min(th, hh - by);

        // Initialise the tile accumulator to S - b.
        for (int ly = 0; ly < bh; ++ly) {
          const std::size_t off = static_cast<std::size_t>(by + ly) * w + bx;
          float* drow = dtile.data() + static_cast<std::size_t>(ly) * bw;
          for (int lx = 0; lx < bw; ++lx) drow[lx] = s[off + lx] - bg[off + lx];
        }

        // Fuse every component's y-pass + a_n accumulate over the tile. tx[nx]
        // for the tile rows (+kh halo) is read once and reused across the (up to
        // n_max+1) components that share nx.
        for (int n = 0; n < nc; ++n) {
          const float* src = tx[orders[n].first].data();
          const std::vector<float>& kyn = ky[n];
          const int ks = static_cast<int>(kyn.size());
          const float* a = fields.coeff[n].data();
          for (int ly = 0; ly < bh; ++ly) {
            const int y = by + ly;
            std::fill(bn.begin(), bn.begin() + bw, 0.0f);
            for (int j = 0; j < ks; ++j) {
              const int sy = y + kh - j;
              if (sy < 0 || sy >= hh) continue;
              const float c = kyn[j];
              const float* srow = src + static_cast<std::size_t>(sy) * w + bx;
#pragma omp simd
              for (int lx = 0; lx < bw; ++lx) bn[lx] += c * srow[lx];
            }
            const float* arow = a + static_cast<std::size_t>(y) * w + bx;
            float* drow = dtile.data() + static_cast<std::size_t>(ly) * bw;
#pragma omp simd
            for (int lx = 0; lx < bw; ++lx) drow[lx] -= arow[lx] * bn[lx];
          }
        }

        // Write the tile's difference out once.
        for (int ly = 0; ly < bh; ++ly) {
          float* drow = diff.data() + static_cast<std::size_t>(by + ly) * w + bx;
          const float* trow = dtile.data() + static_cast<std::size_t>(ly) * bw;
          for (int lx = 0; lx < bw; ++lx) drow[lx] = trow[lx];
        }
      }
    }
  }

  // ---- variance propagation: Var(D) = Var(S) + (K^2 (x) Var(R)) ------------
  if (reference.has_variance() || science.has_variance()) {
    DELTA_TIME("subtract: variance propagation");
    diff.allocate_variance();
    std::vector<float>& var = diff.variance();

    if (reference.has_variance()) {
      // Block-effective squared-kernel convolution (SPEC §3.4).
      //
      //   Var(D) gets  (K^2 (x) Var(R)),  K(x,y) = sum_n a_n(x,y) phi_n .
      //
      // The exact expansion sum_{n,m} a_n a_m [(phi_n phi_m) (x) Var(R)] needs
      // nc(nc+1)/2 separable convolutions of the whole frame (406 at n_max=6) --
      // it dominated the subtraction. Since K varies smoothly, we instead tile
      // the frame, freeze K at each tile's centre, square it into a dense ks x ks
      // footprint K^2, and convolve Var(R) with it in a single O(ks^2)/pixel pass.
      // That is ~30x fewer FLOPs at n_max=6 and allocates no full-frame temporaries.
      // For a spatially constant field every tile yields the same K, so the result
      // is identical to the exact expansion; for a varying field it is a per-tile
      // piecewise-constant approximation of the (slowly varying) coefficient maps.
      const std::vector<float>& vr = reference.variance();
      const int ks = basis.ksize();
      const int rk = basis.radius();

      std::vector<std::vector<float>> phi(nc);
      for (int n = 0; n < nc; ++n) phi[n] = basis.kernel2d(n);

      // Tiles are kept small (so the gathered input window + output tile sit in
      // L1/L2 and stay hot across the ks^2 taps) and the input window is gathered
      // once into a contiguous, zero-padded buffer -- a 256-wide tile straight off
      // the strided frame thrashed the cache and was the dominant cost.
      constexpr int bsz = 64;
      const int ibw = bsz + 2 * rk;  // gathered window side (>= tile + halo)
      std::vector<std::pair<int, int>> blocks;
      for (int by = 0; by < hh; by += bsz)
        for (int bx = 0; bx < ww; bx += bsz) blocks.emplace_back(bx, by);

#pragma omp parallel
      {
        std::vector<float> k2(static_cast<std::size_t>(ks) * ks);
        std::vector<float> kf(static_cast<std::size_t>(ks) * ks);
        std::vector<float> win(static_cast<std::size_t>(ibw) * ibw);
        std::vector<float> out(static_cast<std::size_t>(bsz) * bsz);
#pragma omp for schedule(dynamic)
        for (std::size_t bi = 0; bi < blocks.size(); ++bi) {
          const int bx = blocks[bi].first, by = blocks[bi].second;
          const int bw = std::min(bsz, ww - bx);
          const int bh = std::min(bsz, hh - by);
          const std::size_t cidx =
              static_cast<std::size_t>(std::min(by + bh / 2, hh - 1)) * w +
              std::min(bx + bw / 2, ww - 1);

          // Effective kernel at the tile centre, squared elementwise, then
          // mirrored in both axes (kf) so out(x) = sum K^2(u) Var(x-u) becomes a
          // unit-stride correlation against the gathered window.
          std::fill(k2.begin(), k2.end(), 0.0f);
          for (int n = 0; n < nc; ++n) {
            const float an = fields.coeff[n].data()[cidx];
            const float* p = phi[n].data();
            for (std::size_t i = 0; i < k2.size(); ++i) k2[i] += an * p[i];
          }
          for (std::size_t i = 0; i < k2.size(); ++i) {
            const float v = k2[i];
            kf[k2.size() - 1 - i] = v * v;
          }

          // Gather the (bh+2rk) x (bw+2rk) input window, zero-padded at the frame
          // edge, into a contiguous buffer of row stride ibw.
          const int iwh = bh + 2 * rk;
          const int iww = bw + 2 * rk;
          for (int iy = 0; iy < iwh; ++iy) {
            const int gy = by - rk + iy;
            float* wrow = win.data() + static_cast<std::size_t>(iy) * ibw;
            if (gy < 0 || gy >= hh) {
              std::fill(wrow, wrow + iww, 0.0f);
              continue;
            }
            const float* vrow = vr.data() + static_cast<std::size_t>(gy) * w;
            for (int ix = 0; ix < iww; ++ix) {
              const int gx = bx - rk + ix;
              wrow[ix] = (gx < 0 || gx >= ww) ? 0.0f : vrow[gx];
            }
          }

          // Tap-outer dense convolution on the cache-resident window/tile: each
          // tap is a contiguous multiply-add over the tile row (auto-vectorised).
          std::fill(out.begin(), out.begin() + static_cast<std::size_t>(bh) * bw,
                    0.0f);
          for (int ly = 0; ly < ks; ++ly) {
            const float* krow = kf.data() + static_cast<std::size_t>(ly) * ks;
            for (int lx = 0; lx < ks; ++lx) {
              const float c = krow[lx];
              if (c == 0.0f) continue;
              for (int oy = 0; oy < bh; ++oy) {
                const float* src =
                    win.data() + static_cast<std::size_t>(oy + ly) * ibw + lx;
                float* dst = out.data() + static_cast<std::size_t>(oy) * bw;
#pragma omp simd
                for (int ox = 0; ox < bw; ++ox) dst[ox] += c * src[ox];
              }
            }
          }

          // Scatter the tile's reference-variance contribution into Var(D).
          for (int oy = 0; oy < bh; ++oy) {
            float* vout = var.data() + static_cast<std::size_t>(by + oy) * w + bx;
            const float* orow = out.data() + static_cast<std::size_t>(oy) * bw;
            for (int ox = 0; ox < bw; ++ox) vout[ox] += orow[ox];
          }
        }
      }
    }
    if (science.has_variance()) {
      const std::vector<float>& vs = science.variance();
      for (std::size_t i = 0; i < var.size(); ++i) var[i] += vs[i];
    }
  }

  // ---- mask growth (SPEC §3.6) --------------------------------------------
  if (reference.has_mask() || science.has_mask() || saturation > 0.0) {
    DELTA_TIME("subtract: mask growth");
    diff.allocate_mask();
    std::vector<MaskType>& out = diff.mask();
    const int r = basis.radius();

    // Science mask propagates one-to-one.
    if (science.has_mask()) {
      const std::vector<MaskType>& sm = science.mask();
      for (std::size_t i = 0; i < out.size(); ++i) out[i] |= sm[i];
    }
    // Reference mask is dilated by the kernel half-width: a bad reference pixel
    // contaminates every output pixel within the kernel footprint (separable
    // OR-dilation; same footprint as a nested box fill, far cheaper).
    if (reference.has_mask()) {
      std::vector<MaskType> grown = reference.mask();
      dilate_mask(grown, ww, hh, r);
      for (std::size_t i = 0; i < out.size(); ++i) out[i] |= grown[i];
    }
    // `exact_bad` records the un-dilated pixels whose difference value is truly
    // garbage (no valid input data, or non-linear), as opposed to pixels merely
    // flagged by mask *growth*. Only these exact pixels are overwritten; the grown
    // footprint is preserved (see the resolution loop below).
    std::vector<char> exact_bad(out.size(), 0);

    // Bright/saturated cores are non-linear and never match the model; flag them
    // so their large residual stays out of the difference (SPEC §3.6). A saturated
    // *reference* (convolved) pixel contaminates a whole kernel footprint, so it
    // is grown by the kernel radius; a saturated *science* (target) pixel is not
    // convolved, so it propagates one-to-one -- growing it too is what produced
    // the oversized boxes around every bright star.
    if (saturation > 0.0) {
      const float sat = static_cast<float>(saturation);
      const float* rd = reference.data();
      const float* sd = science.data();
      std::vector<MaskType> hot(out.size(), kMaskGood);
      for (std::size_t i = 0; i < out.size(); ++i)
        if (rd[i] >= sat) hot[i] = kMaskSaturated;
      dilate_mask(hot, ww, hh, r);
      for (std::size_t i = 0; i < out.size(); ++i) {
        if (sd[i] >= sat) hot[i] |= kMaskSaturated;  // target pixel: 1:1
        if (rd[i] >= sat || sd[i] >= sat) exact_bad[i] = 1;
        out[i] |= hot[i];
      }
    }
    // No-overlap / no-data pixels: a location directly flagged in the *input*
    // science or reference mask has no valid science-template overlap there, so
    // its difference is meaningless (typically a huge value at the registered
    // frame border). These are exact (un-dilated) bad pixels -- the reference
    // mask's dilated footprint is handled separately above and is preserved.
    if (science.has_mask()) {
      const std::vector<MaskType>& sm = science.mask();
      for (std::size_t i = 0; i < out.size(); ++i)
        if (sm[i] != kMaskGood) exact_bad[i] = 1;
    }
    if (reference.has_mask()) {
      const std::vector<MaskType>& rm = reference.mask();
      for (std::size_t i = 0; i < out.size(); ++i)
        if (rm[i] != kMaskGood) exact_bad[i] = 1;
    }
    // Edge border of one kernel half-width, and any non-finite difference pixel
    // (also garbage, so it joins the exact-fill set).
    for (int y = 0; y < hh; ++y) {
      for (int x = 0; x < ww; ++x) {
        const std::size_t i = static_cast<std::size_t>(y) * w + x;
        if (x < r || x >= ww - r || y < r || y >= hh - r)
          out[i] |= kMaskEdge;
        if (!std::isfinite(diff.data()[i])) {
          out[i] |= kMaskNonFinite;
          exact_bad[i] = 1;
        }
      }
    }
    // Background median of the good difference, for filling the garbage pixels.
    float fill = 0.0f;
    const bool has_var = diff.has_variance();
    float* dd = diff.data();
    float* vv = has_var ? diff.variance().data() : nullptr;
    {
      std::vector<float> good;
      good.reserve(out.size() / 7 + 1);
      for (std::size_t i = 0; i < out.size(); i += 7)
        if (out[i] == kMaskGood && std::isfinite(dd[i])) good.push_back(dd[i]);
      if (!good.empty()) {
        std::nth_element(good.begin(), good.begin() + good.size() / 2, good.end());
        fill = good[good.size() / 2];
      }
    }
    // Resolve masked-pixel values. Only genuinely meaningless pixels are
    // overwritten with the background median: the exact (un-dilated) saturated
    // cores, the no-overlap / input-masked pixels, and any non-finite pixel.
    // Every other masked pixel keeps its real difference (and variance): the mask
    // grows to flag the affected footprint, but the subtraction itself is
    // preserved so PSF wings and transients close to bright stars / mask edges
    // stay assessable. (Setting the whole grown footprint to a constant boxed out
    // exactly those wings.) Filled pixels remain masked -- the value is cosmetic.
    for (std::size_t i = 0; i < out.size(); ++i) {
      if (out[i] == kMaskGood) continue;
      if (exact_bad[i]) {
        dd[i] = fill;
        if (has_var) vv[i] = 0.0f;
      }
      // otherwise: masked by mask growth / edge -> preserve.
    }
  }

  return diff;
}

}  // namespace delta
