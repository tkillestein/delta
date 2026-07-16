#include "delta/subtract.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "delta/convolve.hpp"
#include "delta/timing.hpp"

namespace delta {

// Only the fill *scalar* is produced here; the substitution itself is fused into
// the model-convolve tile gather (see below), so the sanitised reference never
// materialises full-frame -- that copy was a 512 MB DRAM round-trip whose only
// consumer was the convolution. The strided sample (stride 7, as before) leaves
// the fill value, and hence the whole result, bit-for-bit identical to the
// previous full-frame sanitised copy.
float reference_fill_value(const ImageF& reference) {
  const std::size_t n = reference.size();
  const float* src = reference.data();
  const bool has_mask = reference.has_mask();
  const MaskType* m = has_mask ? reference.mask().data() : nullptr;

  std::vector<float> sample;
  sample.reserve(n / 7 + 1);
  for (std::size_t i = 0; i < n; i += 7) {
    const float v = src[i];
    if (std::isfinite(v) && !(has_mask && m[i] != kMaskGood)) sample.push_back(v);
  }
  if (sample.empty()) return 0.0f;
  std::nth_element(sample.begin(), sample.begin() + sample.size() / 2,
                   sample.end());
  return sample[sample.size() / 2];
}

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

// Separable OR-dilation of a uint8 bitmask by Chebyshev radius r, in place.
// A square (box) structuring element factors into an x-pass then a y-pass; each
// output pixel is the bitwise-OR of its neighbours within +/- r along that axis.
// This is the exact same footprint as the naive nested-box fill but runs in
// O(N r) instead of O(N r^2), and preserves which flag bits were set.
void dilate_mask(std::vector<MaskType>& mask, int w, int h, int r) {
  if (r <= 0) return;
  std::vector<MaskType> tmp(mask.size(), kMaskGood);
  // x-pass: mask -> tmp. Each output row reads only its own input row, and the
  // y-pass reads a +/-r band of `tmp` it never writes, so both passes are
  // row-independent -- thread them (this O(N r) dilation is the last sizeable
  // serial cost in mask growth, run once per reference/saturation layer).
#pragma omp parallel for schedule(static)
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
#pragma omp parallel for schedule(static)
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
      float* row = out.coeff[n].get() + static_cast<std::size_t>(y) * w;
      for (int x = 0; x < w; ++x) row[x] = static_cast<float>(fields(x, n));
    }
    float* bg = out.background.get() + static_cast<std::size_t>(y) * w;
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

// Coarse-lattice field representation for the *subtraction* path: the thin-plate
// fields are evaluated exactly on a coarse lattice (or full-frame for small frames)
// and bilinearly interpolated to full resolution lazily, one tile at a time, inside
// the model-convolve loop. This avoids materialising the ~nc full-frame coefficient
// arrays (~5.4 GB at the reference frame) and the DRAM round-trip of writing them in
// a separate pass and streaming them back during the convolution -- the tile-local
// interpolation stays in cache. The interpolation formula is identical to the
// full-frame materialisation in evaluate_fields(), so the result is bit-identical to
// the previously-materialised path (and exact, for the small-frame fallback).
struct CoarseFields {
  bool exact = false;
  int n_components = 0;
  int w = 0, h = 0;

  // Exact (small-frame) fallback: full-resolution fields.
  SpatialFields full;

  // Coarse path: coarse[(f*ncy+iy)*ncx+ix], f in [0, n_components] (n_components is
  // the background), plus the per-axis bracketing cell index + fractional weight.
  std::vector<double> coarse;
  int ncx = 0, ncy = 0;
  std::vector<int> gx;      // ncx coarse x coordinates
  std::vector<int> cellx;   // w:  bracketing coarse x cell
  std::vector<float> wx;    // w:  fractional x weight within that cell
  std::vector<int> celly;   // h:  bracketing coarse y cell
  std::vector<float> wy;    // h:  fractional y weight within that cell

  // Interpolated value of field f at integer (x,y); f == n_components is the
  // background. Matches the coarse-cell bilinear blend used by fill_tile().
  float point(int f, int x, int y) const {
    if (exact) {
      const float* a =
          (f < n_components ? full.coeff[f].get() : full.background.get());
      return a[static_cast<std::size_t>(y) * w + x];
    }
    const int jx = cellx[x], jy = celly[y];
    const double* c0 = coarse.data() + (static_cast<std::size_t>(f) * ncy + jy) * ncx;
    const double* c1 = c0 + ncx;
    const float a00 = static_cast<float>(c0[jx]);
    const float a01 = static_cast<float>(c0[jx + 1]);
    const float a10 = static_cast<float>(c1[jx]);
    const float a11 = static_cast<float>(c1[jx + 1]);
    const float fy = wy[y];
    const float base = a00 + (a10 - a00) * fy;
    const float top_slope = a01 - a00;
    const float slope = top_slope + ((a11 - a10) - top_slope) * fy;
    return base + slope * wx[x];
  }

  // Interpolate field f over a (bh x bw, row stride bw) tile at origin (bx,by) into
  // dst. Within a coarse cell the bilinear value is affine in x, so each cell reduces
  // to a unit-stride FMA ramp `base + slope*wx[x]` over the precomputed x-weight --
  // the same per-cell ramp evaluate_fields() uses for full-frame materialisation.
  void fill_tile(int f, int bx, int by, int bw, int bh, float* dst) const {
    if (exact) {
      const float* a =
          (f < n_components ? full.coeff[f].get() : full.background.get());
      for (int ly = 0; ly < bh; ++ly) {
        const float* src = a + static_cast<std::size_t>(by + ly) * w + bx;
        std::copy(src, src + bw, dst + static_cast<std::size_t>(ly) * bw);
      }
      return;
    }
    for (int ly = 0; ly < bh; ++ly) {
      const int y = by + ly;
      const int jy = celly[y];
      const float fy = wy[y];
      const double* c0row =
          coarse.data() + (static_cast<std::size_t>(f) * ncy + jy) * ncx;
      const double* c1row = c0row + ncx;
      float* drow = dst + static_cast<std::size_t>(ly) * bw;
      const int xend = bx + bw;
      for (int ix = cellx[bx]; ix <= ncx - 2; ++ix) {
        const int xa = std::max(bx, gx[ix]);
        if (xa >= xend) break;
        const int xb_cell = (ix == ncx - 2) ? w : gx[ix + 1];
        const int xb = std::min(xend, xb_cell);
        const float a00 = static_cast<float>(c0row[ix]);
        const float a01 = static_cast<float>(c0row[ix + 1]);
        const float a10 = static_cast<float>(c1row[ix]);
        const float a11 = static_cast<float>(c1row[ix + 1]);
        const float base = a00 + (a10 - a00) * fy;
        const float top_slope = a01 - a00;
        const float slope = top_slope + ((a11 - a10) - top_slope) * fy;
#pragma omp simd
        for (int x = xa; x < xb; ++x) drow[x - bx] = base + slope * wx[x];
        if (xb >= xend) break;
      }
    }
  }
};

// Build the coarse-lattice (or exact, for small frames) field representation used by
// subtract(). Mirrors evaluate_fields()'s stride choice and coarse evaluation, but
// keeps the lattice rather than materialising the full-frame fields.
CoarseFields build_coarse_fields(const ThinPlateBasis& spatial,
                                 const Eigen::Ref<const Eigen::VectorXd>& theta,
                                 int n_components, int w, int h) {
  const int k = spatial.n_basis();
  const int n_fields = n_components + 1;
  const Eigen::MatrixXd c = coeff_matrix(theta, k, n_fields);

  CoarseFields cf;
  cf.n_components = n_components;
  cf.w = w;
  cf.h = h;

  const int stride = choose_field_stride(spatial);
  if (stride < kMinFieldStride || w <= 2 * stride || h <= 2 * stride) {
    cf.exact = true;
    const std::size_t npix = static_cast<std::size_t>(w) * h;
    cf.full.coeff.resize(n_components);
    for (int n = 0; n < n_components; ++n)
      cf.full.coeff[n] = std::unique_ptr<float[]>(new float[npix]);
    cf.full.background = std::unique_ptr<float[]>(new float[npix]);
    evaluate_fields_exact(spatial, c, n_components, w, h, cf.full);
    return cf;
  }

  cf.gx = coarse_coords(w, stride);
  const std::vector<int> gy = coarse_coords(h, stride);
  cf.ncx = static_cast<int>(cf.gx.size());
  cf.ncy = static_cast<int>(gy.size());
  const int ncx = cf.ncx, ncy = cf.ncy;

  cf.coarse.assign(static_cast<std::size_t>(n_fields) * ncy * ncx, 0.0);
#pragma omp parallel for schedule(static)
  for (int iy = 0; iy < ncy; ++iy) {
    Eigen::MatrixXd points(ncx, 2);
    for (int ix = 0; ix < ncx; ++ix) {
      points(ix, 0) = static_cast<double>(cf.gx[ix]);
      points(ix, 1) = static_cast<double>(gy[iy]);
    }
    const Eigen::MatrixXd fields = spatial.design(points) * c;
    for (int f = 0; f < n_fields; ++f) {
      double* dst =
          cf.coarse.data() + (static_cast<std::size_t>(f) * ncy + iy) * ncx;
      for (int ix = 0; ix < ncx; ++ix) dst[ix] = fields(ix, f);
    }
  }

  cf.wx.resize(w);
  cf.cellx.resize(w);
  for (int x = 0; x < w; ++x) {
    const int c0 = std::min(x / stride, ncx - 2);
    const int x0 = cf.gx[c0], x1 = cf.gx[c0 + 1];
    cf.cellx[x] = c0;
    cf.wx[x] = x1 > x0 ? static_cast<float>(x - x0) / (x1 - x0) : 0.0f;
  }
  cf.wy.resize(h);
  cf.celly.resize(h);
  for (int y = 0; y < h; ++y) {
    const int c0 = std::min(y / stride, ncy - 2);
    const int y0 = gy[c0], y1 = gy[c0 + 1];
    cf.celly[y] = c0;
    cf.wy[y] = y1 > y0 ? static_cast<float>(y - y0) / (y1 - y0) : 0.0f;
  }
  return cf;
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
  const std::size_t npix = width * height;
  out.coeff.resize(n_components);
  for (int n = 0; n < n_components; ++n)
    out.coeff[n] = std::unique_ptr<float[]>(new float[npix]);
  out.background = std::unique_ptr<float[]>(new float[npix]);

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
  // Per-axis fractional weight within the containing coarse cell, precomputed once
  // and shared across every field/row. The x-weight is what lets the inner loop run
  // as a pure FMA ramp over a coarse cell (no per-pixel gather); the y-weight blends
  // the two bracketing coarse rows.
  std::vector<float> wx(w);
  for (int x = 0; x < w; ++x) {
    const int c0 = std::min(x / stride, ncx - 2);
    const int x0 = gx[c0], x1 = gx[c0 + 1];
    wx[x] = x1 > x0 ? static_cast<float>(x - x0) / (x1 - x0) : 0.0f;
  }
  std::vector<int> iy0(h);
  std::vector<float> wy(h);
  for (int y = 0; y < h; ++y) {
    const int c0 = std::min(y / stride, ncy - 2);
    const int y0 = gy[c0], y1 = gy[c0 + 1];
    iy0[y] = c0;
    wy[y] = y1 > y0 ? static_cast<float>(y - y0) / (y1 - y0) : 0.0f;
  }

  // Collapsing the field and row loops keeps each iteration writing one field's row
  // end-to-end. The earlier field-innermost order juggled n_fields simultaneous
  // output streams, which throttled the interpolation to a fraction of memory
  // bandwidth and refused to scale; here a thread's static chunk sweeps one field's
  // rows sequentially. Within a row the bilinear value is affine in x across each
  // coarse cell, so the cell reduces to dst[x] = base + slope*wx[x] -- a unit-stride
  // FMA over the precomputed weight, with no per-pixel gather or index lookup.
#pragma omp parallel for collapse(2) schedule(static)
  for (int f = 0; f < n_fields; ++f) {
    for (int y = 0; y < h; ++y) {
      const int jy0 = iy0[y], jy1 = jy0 + 1;
      const float fy = wy[y];
      const double* c0row =
          coarse.data() + (static_cast<std::size_t>(f) * ncy + jy0) * ncx;
      const double* c1row =
          coarse.data() + (static_cast<std::size_t>(f) * ncy + jy1) * ncx;
      float* dst = (f < n_components ? out.coeff[f].get()
                                     : out.background.get()) +
                   static_cast<std::size_t>(y) * w;
      for (int ix = 0; ix < ncx - 1; ++ix) {
        const float a00 = static_cast<float>(c0row[ix]);
        const float a01 = static_cast<float>(c0row[ix + 1]);
        const float a10 = static_cast<float>(c1row[ix]);
        const float a11 = static_cast<float>(c1row[ix + 1]);
        // val(fx) = base + slope*fx, with base/slope the y-blended cell corners.
        const float base = a00 + (a10 - a00) * fy;
        const float top_slope = a01 - a00;
        const float slope = top_slope + ((a11 - a10) - top_slope) * fy;
        const int xa = gx[ix];
        const int xb = (ix == ncx - 2) ? w : gx[ix + 1];
#pragma omp simd
        for (int x = xa; x < xb; ++x) dst[x] = base + slope * wx[x];
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
  // Coarse field lattice: a_n(x,y) and b(x,y) are interpolated lazily, tile-local,
  // inside the model-convolve loop (and point-sampled in variance prop) rather than
  // materialised full-frame -- see CoarseFields. This drops the ~5.4 GB full-frame
  // coefficient write + the matching streamed read during the convolution.
  const CoarseFields fields = [&] {
    DELTA_TIME("subtract: spatial fields");
    return build_coarse_fields(spatial, theta, nc, static_cast<int>(w),
                               static_cast<int>(h));
  }();
  // Reference sanitisation is fused into the model-convolve gather below; only
  // the median fill scalar is computed up front (the full-frame substitution
  // copy is gone -- see reference_fill_value).
  const float ref_fill = [&] {
    DELTA_TIME("subtract: sanitise reference");
    return reference_fill_value(reference);
  }();
  const bool ref_has_mask = reference.has_mask();
  const MaskType* ref_mask = ref_has_mask ? reference.mask().data() : nullptr;
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
    const auto& orders = basis.orders();
    std::vector<std::vector<float>> kx(n_max + 1);
    for (int nx = 0; nx <= n_max; ++nx) kx[nx] = basis.kernel1d(nx);
    std::vector<std::vector<float>> ky(nc);
    for (int n = 0; n < nc; ++n) ky[n] = basis.kernel1d(orders[n].second);
    const int ks = basis.ksize();
    const int kh = (ks - 1) / 2;

    const float* rd = reference.data();
    const float* s = science.data();
    const int bg_field = nc;  // background is field index nc in CoarseFields

    // Cache-blocked, fully fused model convolve. The earlier version looped
    // components on the outside and ran a full-frame y-pass each, so the running
    // diff was read-modify-written nc(=28) times and every tx[nx] was re-read
    // from DRAM per component -- memory-bandwidth bound (the pipeline's headline
    // bottleneck). It then materialised the whole (n_max+1)-deep x-pass stack
    // (~1.3 GB at the reference frame) and streamed it back per tile.
    //
    // Here the shared x-passes are computed *per tile* into a thread-local buffer
    // covering the tile rows + kh y-halo, then fused in place: the x-pass stack
    // never touches DRAM (it stays in L2 and is consumed immediately by the
    // y-pass), trading the 1.3 GB write + repeated streaming reads for a ~1.17x
    // halo re-convolution of the reference. The per-pixel accumulation order (x
    // taps 0..ks-1, then component order) is unchanged, so the result is
    // bit-identical to the materialised-B_n version.
    //
    // Reference sanitisation is fused in: each tile first gathers its haloed
    // reference window (tile cols +/- kh, rows +/- kh) into a thread-local `rtile`
    // buffer, substituting the median fill for masked / non-finite pixels and 0
    // for off-frame columns. The x-pass then reads `rtile` with no per-tap bounds
    // test (off-frame already zeroed) -- so the full-frame sanitised copy and its
    // 512 MB DRAM round-trip are gone, and the gather largely replaces the read
    // the x-pass already did from the reference. `rtile` matches the old `clean`
    // bit-for-bit (good pixels are an exact float copy), keeping the result
    // identical.
    //
    // 96x96 tiles measured fastest on a 14-thread Ultra 7 (vs 64/128/192): small
    // enough that the per-thread working set (x-pass stack + tile a_n/b/diff
    // buffers) stays L2-resident, large enough that the kh y-halo re-convolution
    // overhead (2*kh/th) stays modest.
    constexpr int tw = 96;
    constexpr int th = 96;
    std::vector<std::pair<int, int>> tiles;
    for (int by = 0; by < hh; by += th)
      for (int bx = 0; bx < ww; bx += tw) tiles.emplace_back(bx, by);

    const int nrows_max = th + 2 * kh;
    const int gbw_max = tw + 2 * kh;  // gathered reference window width
    const std::size_t per_nx_max = static_cast<std::size_t>(nrows_max) * tw;

#pragma omp parallel
    {
      std::vector<float> dtile(static_cast<std::size_t>(tw) * th);
      std::vector<float> bn(tw);
      // Thread-local x-pass stack: (n_max+1) buffers of (bh+2kh) x bw rows.
      std::vector<float> txt(static_cast<std::size_t>(n_max + 1) * per_nx_max);
      // Thread-local gathered + sanitised reference window: (bh+2kh) x (bw+2kh).
      std::vector<float> rtile(static_cast<std::size_t>(nrows_max) * gbw_max);
      // Tile-local interpolated field buffers (one component's a_n; the background).
      std::vector<float> abuf(static_cast<std::size_t>(tw) * th);
      std::vector<float> bgbuf(static_cast<std::size_t>(tw) * th);
#pragma omp for schedule(dynamic)
      for (std::size_t ti = 0; ti < tiles.size(); ++ti) {
        const int bx = tiles[ti].first, by = tiles[ti].second;
        const int bw = std::min(tw, ww - bx);
        const int bh = std::min(th, hh - by);
        const int nrows = bh + 2 * kh;
        const int gbw = bw + 2 * kh;  // gathered window width
        const int hy0 = by - kh;      // global y of buffer row 0
        const int gx0 = bx - kh;      // global x of gathered column 0
        const std::size_t per_nx = static_cast<std::size_t>(nrows) * bw;

        // Gather the haloed reference window once, sanitising in place: masked /
        // non-finite in-frame pixels take the median fill; off-frame columns take
        // 0 (the zero-pad contract). Each pixel is read once per tile here instead
        // of once per nx from the full frame.
        for (int r = 0; r < nrows; ++r) {
          float* rrow = rtile.data() + static_cast<std::size_t>(r) * gbw;
          const int gy = hy0 + r;
          if (gy < 0 || gy >= hh) {
            std::fill(rrow, rrow + gbw, 0.0f);
            continue;
          }
          const float* srow = rd + static_cast<std::size_t>(gy) * w;
          const MaskType* mrow =
              ref_has_mask ? ref_mask + static_cast<std::size_t>(gy) * w : nullptr;
          for (int c = 0; c < gbw; ++c) {
            const int gx = gx0 + c;
            if (gx < 0 || gx >= ww) {
              rrow[c] = 0.0f;
              continue;
            }
            const float v = srow[gx];
            const bool bad =
                (mrow && mrow[gx] != kMaskGood) || !std::isfinite(v);
            rrow[c] = bad ? ref_fill : v;
          }
        }

        // Tile-local separable x-pass for every nx over the tile rows + y-halo.
        // Tap-outer accumulation in the same order as convolve_x, reading the
        // pre-gathered (already zero-padded) window so no per-tap bounds test is
        // needed. Rows whose global y is off-frame are never read by the y-pass
        // (its jmin/jmax clamp), so they are just zeroed.
        for (int nx = 0; nx <= n_max; ++nx) {
          const std::vector<float>& kxn = kx[nx];
          float* base = txt.data() + static_cast<std::size_t>(nx) * per_nx;
          for (int r = 0; r < nrows; ++r) {
            float* orow = base + static_cast<std::size_t>(r) * bw;
            std::fill(orow, orow + bw, 0.0f);
            const int gy = hy0 + r;
            if (gy < 0 || gy >= hh) continue;
            const float* rrow = rtile.data() + static_cast<std::size_t>(r) * gbw;
            for (int j = 0; j < ks; ++j) {
              const float kc = kxn[j];
              const float* src = rrow + (2 * kh - j);  // col = lx + (2kh - j)
#pragma omp simd
              for (int lx = 0; lx < bw; ++lx) orow[lx] += kc * src[lx];
            }
          }
        }

        // Initialise the tile accumulator to S - b, interpolating b tile-local.
        fields.fill_tile(bg_field, bx, by, bw, bh, bgbuf.data());
        for (int ly = 0; ly < bh; ++ly) {
          const std::size_t off = static_cast<std::size_t>(by + ly) * w + bx;
          const float* brow = bgbuf.data() + static_cast<std::size_t>(ly) * bw;
          float* drow = dtile.data() + static_cast<std::size_t>(ly) * bw;
          for (int lx = 0; lx < bw; ++lx) drow[lx] = s[off + lx] - brow[lx];
        }

        // Fuse every component's y-pass + a_n accumulate over the tile, reading
        // the tile-local x-passes (one per nx, shared across the components that
        // use that nx) and the tile-local interpolated a_n field.
        for (int n = 0; n < nc; ++n) {
          const float* src = txt.data() +
                             static_cast<std::size_t>(orders[n].first) * per_nx;
          const std::vector<float>& kyn = ky[n];
          const int kys = static_cast<int>(kyn.size());
          fields.fill_tile(n, bx, by, bw, bh, abuf.data());
          for (int ly = 0; ly < bh; ++ly) {
            const int y = by + ly;
            std::fill(bn.begin(), bn.begin() + bw, 0.0f);
            const int jmin = std::max(0, y + kh - hh + 1);
            const int jmax = std::min(kys - 1, y + kh);
            for (int j = jmin; j <= jmax; ++j) {
              const int sy = y + kh - j;
              const float c = kyn[j];
              const float* srow =
                  src + static_cast<std::size_t>(sy - hy0) * bw;
#pragma omp simd
              for (int lx = 0; lx < bw; ++lx) bn[lx] += c * srow[lx];
            }
            const float* arow = abuf.data() + static_cast<std::size_t>(ly) * bw;
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
      // A tile takes the flat-Var(R) shortcut when its window's relative spread is
      // below this; it bounds the shortcut's relative error (see below). 1e-3 keeps
      // it well inside the propagated-variance tolerance while still catching the
      // smooth sky that dominates a survey frame.
      constexpr float kVarFlatTol = 1e-3f;

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
          const int cpx = std::min(bx + bw / 2, ww - 1);
          const int cpy = std::min(by + bh / 2, hh - 1);

          // Effective kernel at the tile centre, squared elementwise, then
          // mirrored in both axes (kf) so out(x) = sum K^2(u) Var(x-u) becomes a
          // unit-stride correlation against the gathered window.
          std::fill(k2.begin(), k2.end(), 0.0f);
          for (int n = 0; n < nc; ++n) {
            const float an = fields.point(n, cpx, cpy);
            const float* p = phi[n].data();
            for (std::size_t i = 0; i < k2.size(); ++i) k2[i] += an * p[i];
          }
          // K^2 mirrored for the correlation, and its sum Q = sum_u K^2(u) -- the
          // total weight the convolution applies to Var(R), reused by the flat-var
          // shortcut below.
          double qsum = 0.0;
          for (std::size_t i = 0; i < k2.size(); ++i) {
            const float v = k2[i];
            kf[k2.size() - 1 - i] = v * v;
            qsum += static_cast<double>(v) * v;
          }

          // Shrink the K^2 footprint to the smallest centred (2*hw+1)^2 box that
          // holds (1 - kVarBoxTol) of qsum. Because K is squared, K^2 is far more
          // concentrated than K itself -- for a smooth matching kernel ~99.99% of
          // the weight sits within ~rk/2, so the dense correlation only needs the
          // central box, not the full ks^2 footprint (a ~4x tap reduction here).
          // The dropped outer taps carry < kVarBoxTol of the weight, well below the
          // block-effective freeze's own error.
          constexpr float kVarBoxTol = 1e-4f;
          int hw = 0;
          {
            const double target = (1.0 - kVarBoxTol) * qsum;
            double box = 0.0;
            for (; hw < rk; ++hw) {
              box = 0.0;
              const int lo = rk - hw, hi = rk + hw;
              for (int yy = lo; yy <= hi; ++yy) {
                const float* krow = kf.data() + static_cast<std::size_t>(yy) * ks;
                for (int xx = lo; xx <= hi; ++xx) box += krow[xx];
              }
              if (box >= target) break;
            }
          }
          const int lo = rk - hw, hi = rk + hw;  // dense-conv tap bounds

          // Gather the (bh+2rk) x (bw+2rk) input window, zero-padded at the frame
          // edge, into a contiguous buffer of row stride ibw. Track the min/max of
          // the gathered Var(R) to decide whether it is flat over the footprint.
          // Unlike fit.cpp's conv_var_at (the stamp-local equivalent used during
          // the kernel solve), masked reference pixels are not excluded here --
          // harmless, since the corresponding output pixels are mask-grown by the
          // kernel radius regardless (see mask growth above).
          const int iwh = bh + 2 * rk;
          const int iww = bw + 2 * rk;
          float wmin = std::numeric_limits<float>::infinity();
          float wmax = -std::numeric_limits<float>::infinity();
          for (int iy = 0; iy < iwh; ++iy) {
            const int gy = by - rk + iy;
            float* wrow = win.data() + static_cast<std::size_t>(iy) * ibw;
            if (gy < 0 || gy >= hh) {
              std::fill(wrow, wrow + iww, 0.0f);
              wmin = std::min(wmin, 0.0f);
              wmax = std::max(wmax, 0.0f);
              continue;
            }
            const float* vrow = vr.data() + static_cast<std::size_t>(gy) * w;
            for (int ix = 0; ix < iww; ++ix) {
              const int gx = bx - rk + ix;
              const float v = (gx < 0 || gx >= ww) ? 0.0f : vrow[gx];
              wrow[ix] = v;
              wmin = std::min(wmin, v);
              wmax = std::max(wmax, v);
            }
          }

          // Flat-Var(R) shortcut (SPEC §3.4): where the reference variance is
          // near-constant over the kernel footprint -- the sky-dominated common
          // case -- K^2 (x) Var(R) ~= Var(R) * sum_u K^2(u) = Q * Var(R), so the
          // ks^2 convolution collapses to a per-pixel scale by Q. The relative
          // error is bounded by the window's (max-min)/max, so we take it only when
          // that is below kVarFlatTol (zero-padded edge tiles read non-flat and fall
          // through to the exact convolution, which tapers correctly there).
          if (wmax > 0.0f && (wmax - wmin) <= kVarFlatTol * wmax) {
            const float q = static_cast<float>(qsum);
            for (int oy = 0; oy < bh; ++oy) {
              const float* vrow =
                  win.data() + static_cast<std::size_t>(oy + rk) * ibw + rk;
              float* vout = var.data() + static_cast<std::size_t>(by + oy) * w + bx;
              for (int ox = 0; ox < bw; ++ox) vout[ox] += q * vrow[ox];
            }
            continue;
          }

          // Tap-outer dense convolution on the cache-resident window/tile: each
          // tap is a contiguous multiply-add over the tile row (auto-vectorised).
          // Only the central [lo,hi]^2 box of the K^2 footprint is swept (see the
          // box-shrink above); the gathered window keeps its full rk halo so the
          // src indexing is unchanged.
          std::fill(out.begin(), out.begin() + static_cast<std::size_t>(bh) * bw,
                    0.0f);
          for (int ly = lo; ly <= hi; ++ly) {
            const float* krow = kf.data() + static_cast<std::size_t>(ly) * ks;
            for (int lx = lo; lx <= hi; ++lx) {
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
#pragma omp parallel for schedule(static)
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
    // (also garbage, so it joins the exact-fill set). The non-finite scan reads
    // the whole float difference, so thread it over rows (each row independent).
#pragma omp parallel for schedule(static)
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
#pragma omp parallel for schedule(static)
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
