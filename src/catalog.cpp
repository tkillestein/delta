#include "delta/catalog.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <utility>
#include <vector>

#include "delta/timing.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace delta {
namespace {

constexpr double kFwhmPerSigma = 2.354820045030949;  // 2*sqrt(2 ln 2)
constexpr double kPi = 3.14159265358979323846;

// Row-parallelism is only worth its fork/join cost above this much work
// (mirrors convolve.cpp's kParallelMinWork): small frames stay serial, full
// survey-cadence frames (~6000x8000) clear this by orders of magnitude.
constexpr std::size_t kParallelMinWork = 1u << 15;  // ~32k elements

bool is_finite(float v) { return std::isfinite(v); }

// Disjoint-set over a caller-chosen index space (path compression + union by
// rank). Used both per-block (phase 1, local labels) and once globally over
// the much smaller set of per-block component roots (phase 2, stitching).
class UnionFind {
public:
  explicit UnionFind(std::size_t n) : parent_(n), rank_(n, 0) {
    for (std::size_t i = 0; i < n; ++i) parent_[i] = i;
  }
  std::size_t find(std::size_t i) {
    while (parent_[i] != i) { parent_[i] = parent_[parent_[i]]; i = parent_[i]; }
    return i;
  }
  void unite(std::size_t a, std::size_t b) {
    a = find(a); b = find(b);
    if (a == b) return;
    if (rank_[a] < rank_[b]) std::swap(a, b);
    parent_[b] = a;
    if (rank_[a] == rank_[b]) ++rank_[a];
  }

private:
  std::vector<std::size_t> parent_;
  std::vector<std::uint8_t> rank_;  // union-by-rank height is O(log n); 1 byte suffices
};

// Per-pixel "is candidate" predicate: |score| over threshold in the requested
// direction (polarity>0: score>=threshold; polarity<0: score<=-threshold),
// finite, and (when exclude_bad and a mask is present) unmasked. Branch-free
// per-element body (a flat comparison/bitwise-AND into a byte) over an
// embarrassingly parallel index range -- threaded the same way as
// convolve.cpp's row loops, and auto-vectorizable within each thread's chunk.
std::vector<std::uint8_t> seed_mask(const ImageF& score, double threshold,
                                    int polarity, bool exclude_bad) {
  const std::size_t n = score.size();
  std::vector<std::uint8_t> out(n, 0);
  const bool has_mask = score.has_mask();
  const bool exclude = exclude_bad && has_mask;
  const float* s = score.data();
  const MaskType* m = has_mask ? score.mask().data() : nullptr;
  std::uint8_t* o = out.data();
  const float thr = static_cast<float>(threshold);

#pragma omp parallel for schedule(static) if (n >= kParallelMinWork)
  for (std::size_t i = 0; i < n; ++i) {
    const float v = s[i];
    bool pass = std::isfinite(v) && (polarity > 0 ? v >= thr : v <= -thr);
    if (exclude) pass = pass && (m[i] == kMaskGood);
    o[i] = pass ? 1 : 0;
  }
  return out;
}

struct LabelResult {
  std::vector<std::int32_t> labels;  // 0 = background, 1..n_components
  int n_components = 0;
};

// Parallel 8-connected connected-component labelling: 3 phases.
//
// Phase 1 (parallel): split the frame into row-blocks (one per thread) and run
// an independent two-pass union-find *local to each block* -- the block's own
// top/bottom edges are treated as frame edges, so a component spanning a block
// seam is locally split into two pieces. Each block renumbers its own roots to
// a dense block-local id space.
// Phase 2 (serial, over block roots only -- not pixels): stitch block-local
// ids back together by checking 8-adjacency across each block boundary's one
// seam-row pair, via a small union-find sized to the *number of provisional
// components* rather than the frame.
// Phase 3 (parallel): remap every pixel's provisional label to its final
// dense id.
//
// This is the standard block-decomposed parallel connected-components
// approach (cf. OpenCV's parallel labelling): the expensive full-frame pass is
// threaded, and the only serial step is cheap because it scales with the
// component count, not the pixel count.
LabelResult label_components(const std::vector<std::uint8_t>& seed, int w, int h) {
  const std::size_t n = seed.size();
  std::vector<std::int32_t> labels(n, 0);
  if (w <= 0 || h <= 0) return {std::move(labels), 0};

  int n_threads = 1;
#ifdef _OPENMP
  n_threads = omp_get_max_threads();
#endif
  constexpr int kMinRowsPerBlock = 8;  // avoid degenerate tiny blocks
  int n_blocks = std::max(1, std::min(n_threads, h / kMinRowsPerBlock));
  if (static_cast<std::size_t>(h) * static_cast<std::size_t>(w) < kParallelMinWork)
    n_blocks = 1;

  std::vector<int> block_start(static_cast<std::size_t>(n_blocks) + 1);
  for (int b = 0; b <= n_blocks; ++b)
    block_start[static_cast<std::size_t>(b)] =
        static_cast<int>((static_cast<long long>(h) * b) / n_blocks);

  // Phase 1: independent per-block local labelling.
  std::vector<int> block_n_components(static_cast<std::size_t>(n_blocks), 0);
#pragma omp parallel for schedule(dynamic) if (n_blocks > 1)
  for (int b = 0; b < n_blocks; ++b) {
    const int y0 = block_start[static_cast<std::size_t>(b)];
    const int y1 = block_start[static_cast<std::size_t>(b) + 1];
    if (y0 >= y1) continue;
    const std::size_t block_n = static_cast<std::size_t>(y1 - y0) * static_cast<std::size_t>(w);
    const auto local_idx = [w, y0](int x, int y) {
      return static_cast<std::size_t>(y - y0) * static_cast<std::size_t>(w) + static_cast<std::size_t>(x);
    };

    // Compact this block's seed pixels to a dense [0, k) index space before
    // sizing the union-find: most pixels in a block are typically *not*
    // seeded (sparse point sources over a mostly-empty score image), so a
    // union-find sized to the whole block pays its full O(block_n)
    // allocation/zero-init cost (two arrays) for pixels that never
    // participate in a union. `seed_index` is the one full-block-sized array
    // this still costs; everything downstream is sized to k instead.
    std::vector<std::int32_t> seed_index(block_n, -1);
    std::size_t k = 0;
    for (int y = y0; y < y1; ++y) {
      const std::size_t row = static_cast<std::size_t>(y) * static_cast<std::size_t>(w);
      for (int x = 0; x < w; ++x) {
        if (seed[row + static_cast<std::size_t>(x)]) seed_index[local_idx(x, y)] = static_cast<std::int32_t>(k++);
      }
    }
    if (k == 0) continue;

    UnionFind uf(k);
    for (int y = y0; y < y1; ++y) {
      const std::size_t row = static_cast<std::size_t>(y) * static_cast<std::size_t>(w);
      for (int x = 0; x < w; ++x) {
        const std::size_t idx = row + static_cast<std::size_t>(x);
        if (!seed[idx]) continue;
        const std::size_t me = static_cast<std::size_t>(seed_index[local_idx(x, y)]);
        if (x > 0 && seed[idx - 1])
          uf.unite(me, static_cast<std::size_t>(seed_index[local_idx(x - 1, y)]));
        if (y > y0) {
          const std::size_t above = idx - static_cast<std::size_t>(w);
          if (x > 0 && seed[above - 1])
            uf.unite(me, static_cast<std::size_t>(seed_index[local_idx(x - 1, y - 1)]));
          if (seed[above]) uf.unite(me, static_cast<std::size_t>(seed_index[local_idx(x, y - 1)]));
          if (x + 1 < w && seed[above + 1])
            uf.unite(me, static_cast<std::size_t>(seed_index[local_idx(x + 1, y - 1)]));
        }
      }
    }

    // `root` is a UnionFind index over [0, k), so direct-address the
    // renumbering table by root instead of hashing -- this matters here:
    // a frame full of single-pixel noise blips makes every seed pixel its own
    // root, so this loop runs once per seed pixel and a hash map's overhead
    // would dominate.
    std::vector<std::int32_t> root_to_local(k, 0);
    std::int32_t next_local = 1;
    for (int y = y0; y < y1; ++y) {
      const std::size_t row = static_cast<std::size_t>(y) * static_cast<std::size_t>(w);
      for (int x = 0; x < w; ++x) {
        const std::size_t idx = row + static_cast<std::size_t>(x);
        if (!seed[idx]) continue;
        const std::size_t root = uf.find(static_cast<std::size_t>(seed_index[local_idx(x, y)]));
        std::int32_t& slot = root_to_local[root];
        if (slot == 0) slot = next_local++;
        labels[idx] = slot;  // block-local label, globalised below
      }
    }
    block_n_components[static_cast<std::size_t>(b)] = next_local - 1;
  }

  std::vector<std::int32_t> block_offset(static_cast<std::size_t>(n_blocks) + 1, 0);
  for (int b = 0; b < n_blocks; ++b)
    block_offset[static_cast<std::size_t>(b) + 1] =
        block_offset[static_cast<std::size_t>(b)] + block_n_components[static_cast<std::size_t>(b)];
  const std::int32_t total_provisional = block_offset[static_cast<std::size_t>(n_blocks)];
  if (total_provisional == 0) return {std::move(labels), 0};

  // Offset each block's local labels into a shared, still-provisional id space.
#pragma omp parallel for schedule(static) if (n_blocks > 1)
  for (int b = 0; b < n_blocks; ++b) {
    const int y0 = block_start[static_cast<std::size_t>(b)];
    const int y1 = block_start[static_cast<std::size_t>(b) + 1];
    const std::int32_t offset = block_offset[static_cast<std::size_t>(b)];
    for (int y = y0; y < y1; ++y) {
      const std::size_t row = static_cast<std::size_t>(y) * static_cast<std::size_t>(w);
      for (int x = 0; x < w; ++x) {
        std::int32_t& lbl = labels[row + static_cast<std::size_t>(x)];
        if (lbl > 0) lbl += offset;
      }
    }
  }

  // Phase 2 (serial, O(n_blocks * w) work, not O(n)): stitch components that
  // cross a block seam by checking 8-adjacency across each boundary's one
  // seam-row pair.
  UnionFind global_uf(static_cast<std::size_t>(total_provisional) + 1);
  for (int b = 0; b + 1 < n_blocks; ++b) {
    const int yA = block_start[static_cast<std::size_t>(b) + 1] - 1;  // last row of block b
    const int yB = block_start[static_cast<std::size_t>(b) + 1];      // first row of block b+1
    if (yA < block_start[static_cast<std::size_t>(b)] || yB >= block_start[static_cast<std::size_t>(b) + 2])
      continue;
    for (int x = 0; x < w; ++x) {
      const std::size_t idxA = static_cast<std::size_t>(yA) * static_cast<std::size_t>(w) + static_cast<std::size_t>(x);
      const std::int32_t la = labels[idxA];
      if (la <= 0) continue;
      for (int dx = -1; dx <= 1; ++dx) {
        const int xb = x + dx;
        if (xb < 0 || xb >= w) continue;
        const std::size_t idxB = static_cast<std::size_t>(yB) * static_cast<std::size_t>(w) + static_cast<std::size_t>(xb);
        const std::int32_t lb = labels[idxB];
        if (lb > 0) global_uf.unite(static_cast<std::size_t>(la), static_cast<std::size_t>(lb));
      }
    }
  }

  // As in phase 1, `root` is bounded to [0, total_provisional], so a direct-
  // addressed memo table (keyed by root) beats a hash map; `final_label`
  // (keyed by provisional id `p`) is a separate array since the two index
  // domains overlap.
  std::vector<std::int32_t> root_seen(static_cast<std::size_t>(total_provisional) + 1, 0);
  std::vector<std::int32_t> final_label(static_cast<std::size_t>(total_provisional) + 1, 0);
  std::int32_t next_final = 1;
  for (std::int32_t p = 1; p <= total_provisional; ++p) {
    const std::size_t root = global_uf.find(static_cast<std::size_t>(p));
    std::int32_t& slot = root_seen[root];
    if (slot == 0) slot = next_final++;
    final_label[static_cast<std::size_t>(p)] = slot;
  }
  const int n_components = next_final - 1;

  // Phase 3: parallel remap from provisional global id to final dense id.
#pragma omp parallel for schedule(static) if (n >= kParallelMinWork)
  for (std::size_t i = 0; i < n; ++i) {
    std::int32_t& lbl = labels[i];
    if (lbl > 0) lbl = final_label[static_cast<std::size_t>(lbl)];
  }
  return {std::move(labels), n_components};
}

// Counting-sort bucketing of pixel indices by label: `flat[offsets[lbl] ..
// offsets[lbl+1])` holds label `lbl`'s member pixel indices (lbl in
// 1..n_components). Built via a parallel atomic histogram + (serial, cheap)
// prefix sum + parallel atomic scatter, so the O(n) passes are threaded while
// avoiding the per-bucket heap-allocation/locking a vector<vector<>> would need.
struct Buckets {
  std::vector<std::size_t> flat;
  std::vector<std::size_t> offsets;  // size n_components + 2
};

Buckets bucket_by_label(const std::vector<std::int32_t>& labels, int n_components) {
  const std::size_t n = labels.size();
  const std::size_t nc = static_cast<std::size_t>(n_components);

  std::vector<std::atomic<std::size_t>> counts(nc + 1);
  for (auto& c : counts) c.store(0, std::memory_order_relaxed);
#pragma omp parallel for schedule(static) if (n >= kParallelMinWork)
  for (std::size_t i = 0; i < n; ++i) {
    const std::int32_t lbl = labels[i];
    if (lbl > 0) counts[static_cast<std::size_t>(lbl)].fetch_add(1, std::memory_order_relaxed);
  }

  std::vector<std::size_t> offsets(nc + 2, 0);
  for (std::size_t lbl = 1; lbl <= nc; ++lbl)
    offsets[lbl + 1] = offsets[lbl] + counts[lbl].load(std::memory_order_relaxed);

  std::vector<std::atomic<std::size_t>> cursor(nc + 1);
  for (std::size_t lbl = 1; lbl <= nc; ++lbl) cursor[lbl].store(offsets[lbl], std::memory_order_relaxed);

  std::vector<std::size_t> flat(offsets[nc + 1]);
#pragma omp parallel for schedule(static) if (n >= kParallelMinWork)
  for (std::size_t i = 0; i < n; ++i) {
    const std::int32_t lbl = labels[i];
    if (lbl > 0) {
      const std::size_t slot = cursor[static_cast<std::size_t>(lbl)].fetch_add(1, std::memory_order_relaxed);
      flat[slot] = i;
    }
  }
  return {std::move(flat), std::move(offsets)};
}

// Edge-bounds-safe circular aperture sum on `difference` about (cx, cy).
// Unlike python/delta/validation.py's aperture_flux, this never reads past
// the frame edge -- auto-detected candidates near the border are routine.
double aperture_flux_checked(const ImageF& difference, double cx, double cy,
                             int radius) {
  const int w = static_cast<int>(difference.width());
  const int h = static_cast<int>(difference.height());
  const int x0 = static_cast<int>(std::lround(cx));
  const int y0 = static_cast<int>(std::lround(cy));
  const double r2 = static_cast<double>(radius) * radius;
  double sum = 0.0;
  for (int dy = -radius; dy <= radius; ++dy) {
    const int yy = y0 + dy;
    if (yy < 0 || yy >= h) continue;
    for (int dx = -radius; dx <= radius; ++dx) {
      if (dx * dx + dy * dy > r2) continue;
      const int xx = x0 + dx;
      if (xx < 0 || xx >= w) continue;
      const float v = difference.data()[static_cast<std::size_t>(yy) * w + xx];
      if (is_finite(v)) sum += static_cast<double>(v);
    }
  }
  return sum;
}

// Summarize one labelled component into a CatalogEntry: flux-weighted
// centroid and peak (on |score|), pixel count, FWHM-consistency, mask-flag
// aggregation over the footprint plus its 8-connected boundary ring, and
// aperture flux on `difference`. `polarity` selects which threshold
// (`threshold_sigma` vs `threshold_sigma_dipole`) the component was seeded
// at, for the expected-pixel-count formula. Called once per component from a
// parallel loop over (disjoint, pre-bucketed) component pixel ranges, so this
// itself does no further threading.
CatalogEntry summarize_component(const std::size_t* pixels, std::size_t count,
                                 const ImageF& score, const ImageF& difference,
                                 const std::vector<std::int32_t>& labels,
                                 std::int32_t label_id, int w, int h,
                                 int polarity, const CatalogParams& params) {
  CatalogEntry e;
  e.n_pix = static_cast<int>(count);

  double sum = 0.0, sx = 0.0, sy = 0.0, best_abs = -1.0, peak_signed = 0.0;
  std::size_t best_idx = 0;
  std::uint8_t mask_flags = 0;
  const bool has_mask = score.has_mask();

  // The component's member-pixel list is bucketed via a parallel atomic
  // scatter (bucket_by_label), so its element order is run-to-run
  // nondeterministic. A plain `weight > best_abs` peak pick would then make
  // the *reported* peak pixel nondeterministic on score plateaus (multiple
  // pixels tied at the max |score|); break ties on linear pixel index so the
  // result never depends on traversal order.
  for (std::size_t k = 0; k < count; ++k) {
    const std::size_t idx = pixels[k];
    const int x = static_cast<int>(idx % static_cast<std::size_t>(w));
    const int y = static_cast<int>(idx / static_cast<std::size_t>(w));
    const double v = static_cast<double>(score.data()[idx]);
    const double weight = std::fabs(v);
    sum += weight;
    sx += weight * x;
    sy += weight * y;
    if (weight > best_abs || (weight == best_abs && idx < best_idx)) {
      best_abs = weight;
      best_idx = idx;
      peak_signed = v;
      e.peak_x = x;
      e.peak_y = y;
    }
    if (has_mask) {
      mask_flags |= score.mask()[idx];
      for (int dy = -1; dy <= 1; ++dy) {
        const int yy = y + dy;
        if (yy < 0 || yy >= h) continue;
        for (int dx = -1; dx <= 1; ++dx) {
          if (dx == 0 && dy == 0) continue;
          const int xx = x + dx;
          if (xx < 0 || xx >= w) continue;
          const std::size_t nidx = static_cast<std::size_t>(yy) * w + xx;
          if (labels[nidx] != label_id) mask_flags |= score.mask()[nidx];
        }
      }
    }
  }

  e.x = sum > 0.0 ? sx / sum
                  : (count > 0 ? static_cast<double>(pixels[0] % static_cast<std::size_t>(w)) : 0.0);
  e.y = sum > 0.0 ? sy / sum
                  : (count > 0 ? static_cast<double>(pixels[0] / static_cast<std::size_t>(w)) : 0.0);
  e.peak_snr = peak_signed;
  e.mask_flags = mask_flags;

  const double sigma = params.expected_fwhm / kFwhmPerSigma;
  const double threshold =
      polarity > 0 ? params.threshold_sigma : params.threshold_sigma_dipole;
  const double ratio = threshold > 0.0 ? std::fabs(e.peak_snr) / threshold : 0.0;
  const double log_term = ratio > 0.0 ? std::log(ratio * ratio) : 0.0;
  e.expected_n_pix = kPi * sigma * sigma * std::max(0.0, log_term);
  e.fwhm_ratio = e.expected_n_pix > 0.0
                     ? e.n_pix / e.expected_n_pix
                     : std::numeric_limits<double>::infinity();
  e.fwhm_consistent = e.fwhm_ratio >= params.fwhm_tolerance_lo &&
                      e.fwhm_ratio <= params.fwhm_tolerance_hi;

  const int radius = params.aperture_radius > 0
                         ? params.aperture_radius
                         : std::max(1, static_cast<int>(std::lround(3.0 * sigma)));
  e.flux = aperture_flux_checked(difference, e.x, e.y, radius);

  std::uint8_t q = 0;
  if (!e.fwhm_consistent) q |= 1u << 0;
  if (e.mask_flags != 0) q |= 1u << 2;
  e.quality = q;  // bit1 (is_dipole) set by the caller once both polarities
                  // have been labelled.
  return e;
}

// For each negative-threshold pixel, mark every positive label touched by
// its 3x3 neighborhood (covers both "fully enclosed" and "directly
// touching"). Dipole detection only needs to know whether a positive
// component touches *any* negative-threshold pixel, not that pixel's
// component identity, so this runs directly off the boolean seed mask --
// skipping the negative side's connected-component labelling/bucketing
// entirely whenever the caller doesn't also want the negative-blob catalog
// itself (the overwhelmingly common case). A flat parallel pass over the
// full negative-seed image with lock-free relaxed atomic stores into a small
// per-positive-label flag array -- writes only ever set `1`, so concurrent
// writers racing on the same flag are harmless.
std::vector<std::uint8_t> flag_dipole_positives(
    const std::vector<std::uint8_t>& neg_seed,
    const std::vector<std::int32_t>& pos_labels, int n_pos, int w, int h) {
  std::vector<std::atomic<std::uint8_t>> touched(static_cast<std::size_t>(n_pos) + 1);
  for (auto& t : touched) t.store(0, std::memory_order_relaxed);

  const std::size_t n = neg_seed.size();
#pragma omp parallel for schedule(static) if (n >= kParallelMinWork)
  for (std::size_t idx = 0; idx < n; ++idx) {
    if (!neg_seed[idx]) continue;
    const int x = static_cast<int>(idx % static_cast<std::size_t>(w));
    const int y = static_cast<int>(idx / static_cast<std::size_t>(w));
    for (int dy = -1; dy <= 1; ++dy) {
      const int yy = y + dy;
      if (yy < 0 || yy >= h) continue;
      for (int dx = -1; dx <= 1; ++dx) {
        const int xx = x + dx;
        if (xx < 0 || xx >= w) continue;
        const std::int32_t lbl = pos_labels[static_cast<std::size_t>(yy) * w + xx];
        if (lbl > 0) touched[static_cast<std::size_t>(lbl)].store(1, std::memory_order_relaxed);
      }
    }
  }

  std::vector<std::uint8_t> out(touched.size());
  for (std::size_t i = 0; i < touched.size(); ++i) out[i] = touched[i].load(std::memory_order_relaxed);
  return out;
}

// Components are usually numerous and individually cheap (a frame of noise
// blips makes most components single pixels), so a per-component dynamic
// schedule with chunk size 1 would pay OpenMP's claim/dispatch overhead once
// per component; a small fixed chunk amortises that while still adapting to
// the rare frame with one outsized blended component.
constexpr int kComponentChunk = 32;

}  // namespace

std::vector<CatalogEntry> build_catalog(const ImageF& score,
                                        const ImageF& difference,
                                        const CatalogParams& params,
                                        bool return_negative) {
  const int w = static_cast<int>(score.width());
  const int h = static_cast<int>(score.height());

  std::vector<std::uint8_t> pos_seed, neg_seed;
  {
    DELTA_TIME("catalog: seed");
    pos_seed = seed_mask(score, params.threshold_sigma, +1, params.exclude_bad_pixels);
    neg_seed = seed_mask(score, params.threshold_sigma_dipole, -1, params.exclude_bad_pixels);
  }

  LabelResult pos_lr;
  { DELTA_TIME("catalog: label positive"); pos_lr = label_components(pos_seed, w, h); }
  Buckets pos_buckets;
  { DELTA_TIME("catalog: bucket positive"); pos_buckets = bucket_by_label(pos_lr.labels, pos_lr.n_components); }

  std::vector<CatalogEntry> pos_entries(static_cast<std::size_t>(pos_lr.n_components));
  {
    DELTA_TIME("catalog: summarize positive");
#pragma omp parallel for schedule(dynamic, kComponentChunk) if (pos_lr.n_components > 1)
    for (int lbl = 1; lbl <= pos_lr.n_components; ++lbl) {
      const std::size_t off = pos_buckets.offsets[static_cast<std::size_t>(lbl)];
      const std::size_t cnt = pos_buckets.offsets[static_cast<std::size_t>(lbl) + 1] - off;
      pos_entries[static_cast<std::size_t>(lbl - 1)] =
          summarize_component(pos_buckets.flat.data() + off, cnt, score, difference,
                              pos_lr.labels, lbl, w, h, +1, params);
    }
  }

  // Dipole detection only needs the boolean negative-seed mask (see
  // flag_dipole_positives), so the negative side's own connected-component
  // labelling/bucketing/summarisation -- each as expensive as the positive
  // side's -- is skipped unless the caller explicitly wants the negative-blob
  // catalog back.
  {
    DELTA_TIME("catalog: dipole flag");
    const std::vector<std::uint8_t> touched =
        flag_dipole_positives(neg_seed, pos_lr.labels, pos_lr.n_components, w, h);
#pragma omp parallel for schedule(static) if (pos_lr.n_components > 1)
    for (int lbl = 1; lbl <= pos_lr.n_components; ++lbl) {
      CatalogEntry& e = pos_entries[static_cast<std::size_t>(lbl - 1)];
      e.is_dipole = touched[static_cast<std::size_t>(lbl)] != 0;
      if (e.is_dipole) e.quality |= 1u << 1;
    }
  }

  if (!return_negative) return pos_entries;

  LabelResult neg_lr;
  { DELTA_TIME("catalog: label negative"); neg_lr = label_components(neg_seed, w, h); }
  Buckets neg_buckets;
  { DELTA_TIME("catalog: bucket negative"); neg_buckets = bucket_by_label(neg_lr.labels, neg_lr.n_components); }
  std::vector<CatalogEntry> neg_entries(static_cast<std::size_t>(neg_lr.n_components));
  {
    DELTA_TIME("catalog: summarize negative");
#pragma omp parallel for schedule(dynamic, kComponentChunk) if (neg_lr.n_components > 1)
    for (int lbl = 1; lbl <= neg_lr.n_components; ++lbl) {
      const std::size_t off = neg_buckets.offsets[static_cast<std::size_t>(lbl)];
      const std::size_t cnt = neg_buckets.offsets[static_cast<std::size_t>(lbl) + 1] - off;
      neg_entries[static_cast<std::size_t>(lbl - 1)] =
          summarize_component(neg_buckets.flat.data() + off, cnt, score, difference,
                              neg_lr.labels, lbl, w, h, -1, params);
    }
  }
  return neg_entries;
}

}  // namespace delta
