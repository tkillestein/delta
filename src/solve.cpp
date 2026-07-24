#include "delta/solve.hpp"

#include <Eigen/Cholesky>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace delta {

namespace {

// Assembled, lambda-independent pieces of the penalised normal equations.
//
// The factorized model (SPEC §3.2-3.3) writes each design column for kernel
// component n and spatial term m as psi_m(x,y) * B_n(x,y), plus a background
// block psi_m(x,y). With D = basis.design(points) (N x k) the full design is
//
//   X = [ diag(B_0) D | diag(B_1) D | ... | diag(B_{nc-1}) D | D ]   (N x P)
//
// where P = (nc + 1) * k. We only ever need the weighted normal equations, so
// we store M = X^T W X (P x P), rhs = X^T W d, the penalty unit P0, and the
// scalar y^T W y for the residual sum of squares.
struct System {
  Eigen::MatrixXd m;   // X^T W X
  Eigen::VectorXd rhs;  // X^T W d
  Eigen::MatrixXd p0;  // block-diagonal penalty (one block per component + bg)
  double ywy = 0.0;    // d^T W d
  int n = 0;           // number of pixels (rows)
  int nc = 0;          // number of kernel components
  int k = 0;           // spatial terms per field
  int p = 0;           // total unknowns = (nc + 1) * k
};

System assemble(const Eigen::Ref<const Eigen::MatrixXd>& points,
                const Eigen::Ref<const Eigen::VectorXd>& target,
                const Eigen::Ref<const Eigen::VectorXd>& weights,
                const Eigen::Ref<const Eigen::MatrixXd>& bn,
                const ThinPlateBasis& basis, bool fit_background) {
  const int n = static_cast<int>(points.rows());
  const int nc = static_cast<int>(bn.cols());
  if (target.size() != n || weights.size() != n || bn.rows() != n)
    throw std::runtime_error("solve_gls: row counts of points/target/weights/bn disagree");

  const Eigen::MatrixXd d = basis.design(points);  // N x k
  const int k = static_cast<int>(d.cols());
  const int n_fields = fit_background ? nc + 1 : nc;
  const int p = n_fields * k;

  // Build the pre-whitened design Xs = W^{1/2} X (N x P): each component block is
  // D scaled row-wise by sqrt(weight) * B_n; the trailing block (omitted when
  // fit_background is false) is sqrt(weight) D. With Xs the normal equations are
  // M = Xs^T Xs and rhs = Xs^T ds (ds = W^{1/2} d), so M is a symmetric rank
  // update -- half the FLOPs of the general X^T W X.
  const Eigen::ArrayXd sw = weights.array().sqrt();
  Eigen::MatrixXd xs(n, p);
  for (int c = 0; c < nc; ++c)
    xs.block(0, c * k, n, k) =
        (d.array().colwise() * (bn.col(c).array() * sw)).matrix();
  if (fit_background) xs.block(0, nc * k, n, k) = (d.array().colwise() * sw).matrix();
  const Eigen::VectorXd ds = (sw * target.array()).matrix();

  System s;
  s.n = n;
  s.nc = nc;
  s.k = k;
  s.p = p;

  // M = Xs^T Xs (P x P) and rhs = Xs^T ds. This O(N P^2) work dominates the fit
  // (P = (nc+1)k can be ~700) and Eigen does not reliably thread it here, so we
  // accumulate over row chunks and reduce; each chunk uses a symmetric rankUpdate
  // (lower triangle only). Each thread holds a private P x P partial.
  // Note: the per-thread P x P partial below is allocated *inside* the parallel
  // region. A C++ exception (e.g. Eigen std::bad_alloc on a very large P) must not
  // escape an `omp parallel` body -- that is undefined behavior / std::terminate.
  // The dominant allocation is this fixed-size P x P partial; P = (nc+1)k is known
  // before the region and the same allocation is made serially as `m_acc` just
  // above, so an out-of-memory condition is hit (and throws cleanly) here in serial
  // first. If P ever becomes caller-controlled enough to risk per-thread OOM, add
  // an explicit capacity check before this region.
  Eigen::MatrixXd m_acc = Eigen::MatrixXd::Zero(p, p);
  Eigen::VectorXd rhs_acc = Eigen::VectorXd::Zero(p);
  constexpr int kChunk = 2048;
#pragma omp parallel
  {
    Eigen::MatrixXd m_loc = Eigen::MatrixXd::Zero(p, p);
    Eigen::VectorXd rhs_loc = Eigen::VectorXd::Zero(p);
#pragma omp for schedule(dynamic) nowait
    for (int r0 = 0; r0 < n; r0 += kChunk) {
      const int nr = std::min(kChunk, n - r0);
      const auto xb = xs.middleRows(r0, nr);
      m_loc.selfadjointView<Eigen::Lower>().rankUpdate(xb.transpose());
      rhs_loc.noalias() += xb.transpose() * ds.segment(r0, nr);
    }
#pragma omp critical
    {
      m_acc.triangularView<Eigen::Lower>() += m_loc;
      rhs_acc += rhs_loc;
    }
  }
  // Mirror the accumulated lower triangle into a full symmetric matrix.
  s.m = m_acc.selfadjointView<Eigen::Lower>();
  s.rhs = std::move(rhs_acc);
  s.ywy = (weights.array() * target.array().square()).sum();

  // Penalty unit: the same k x k bending-energy block for every field
  // (the nc kernel-coefficient fields and, when fit, the background field).
  s.p0 = Eigen::MatrixXd::Zero(p, p);
  const Eigen::MatrixXd& pen = basis.penalty();
  for (int b = 0; b < n_fields; ++b) s.p0.block(b * k, b * k, k, k) = pen;
  return s;
}

// Appends a zero background block so `theta` always has the documented
// `(nc+1)*k` length, even when fit_background dropped it from the solve --
// callers reconstruct the model from theta unconditionally, and a zero
// background field is exactly "no differential background".
void pad_zero_background(GlsResult& r, int k) {
  const Eigen::Index old_size = r.theta.size();
  r.theta.conservativeResize(old_size + k);
  r.theta.tail(k).setZero();
}

GlsResult solve_at(const System& s, double lambda) {
  const Eigen::MatrixXd a = s.m + lambda * s.p0;

  GlsResult r;
  r.n_components = s.nc;
  r.n_spatial = s.k;
  r.lambda = lambda;

  // A = X^T W X + lambda P0 is symmetric positive-definite for any lambda>0, so a
  // plain Cholesky (LLT) factorises it ~1.5x faster than the pivoted LDLT -- and
  // this factorisation, repeated across the lambda grid, is the dominant cost of
  // the fit (benchmarks/PERFORMANCE.md). The rare near-singular system (an
  // under-constrained field at lambda~0) loses positive-definiteness mid-Cholesky;
  // there we fall back to the robust LDLT. Both give theta and tr S(lambda) =
  // tr(A^{-1} X^T W X) (the effective dof) identically.
  const Eigen::LLT<Eigen::MatrixXd> llt(a);
  if (llt.info() == Eigen::Success) {
    r.theta = llt.solve(s.rhs);
    r.effective_dof = (llt.solve(s.m)).trace();
  } else {
    const Eigen::LDLT<Eigen::MatrixXd> ldlt(a);
    r.theta = ldlt.solve(s.rhs);
    r.effective_dof = (ldlt.solve(s.m)).trace();
  }

  // Weighted residual sum of squares: ||W^1/2 (d - X theta)||^2
  //   = y^T W y - 2 theta^T (X^T W d) + theta^T (X^T W X) theta.
  double rss = s.ywy - 2.0 * r.theta.dot(s.rhs) + r.theta.dot(s.m * r.theta);
  if (rss < 0.0) rss = 0.0;  // guard against round-off near a perfect fit
  r.rss = rss;

  const double denom = static_cast<double>(s.n) - r.effective_dof;
  r.gcv = denom > 0.0 ? static_cast<double>(s.n) * rss / (denom * denom)
                      : std::numeric_limits<double>::infinity();
  return r;
}

}  // namespace

GlsResult solve_gls(const Eigen::Ref<const Eigen::MatrixXd>& points,
                    const Eigen::Ref<const Eigen::VectorXd>& target,
                    const Eigen::Ref<const Eigen::VectorXd>& weights,
                    const Eigen::Ref<const Eigen::MatrixXd>& bn,
                    const ThinPlateBasis& basis, double lambda,
                    bool fit_background) {
  GlsResult r =
      solve_at(assemble(points, target, weights, bn, basis, fit_background), lambda);
  if (!fit_background) pad_zero_background(r, r.n_spatial);
  return r;
}

GlsResult solve_gls_gcv(const Eigen::Ref<const Eigen::MatrixXd>& points,
                        const Eigen::Ref<const Eigen::VectorXd>& target,
                        const Eigen::Ref<const Eigen::VectorXd>& weights,
                        const Eigen::Ref<const Eigen::MatrixXd>& bn,
                        const ThinPlateBasis& basis,
                        const std::vector<double>& lambda_grid,
                        bool fit_background) {
  if (lambda_grid.empty())
    throw std::runtime_error("solve_gls_gcv: lambda_grid is empty");

  const System s = assemble(points, target, weights, bn, basis, fit_background);

  // Each lambda is an independent LDLT solve + trace on the shared (read-only)
  // system; evaluate the grid in parallel, then reduce to the GCV-minimising fit.
  const int ng = static_cast<int>(lambda_grid.size());
  std::vector<GlsResult> results(ng);
#pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < ng; ++i) results[i] = solve_at(s, lambda_grid[i]);

  std::vector<double> curve(ng);
  int best_i = 0;
  double best_gcv = std::numeric_limits<double>::infinity();
  for (int i = 0; i < ng; ++i) {
    curve[i] = results[i].gcv;
    if (results[i].gcv < best_gcv) {
      best_gcv = results[i].gcv;
      best_i = i;
    }
  }

  GlsResult best = results[best_i];
  if (!fit_background) pad_zero_background(best, best.n_spatial);
  best.lambda_grid = lambda_grid;
  best.gcv_curve = std::move(curve);
  return best;
}

namespace {

// Shared tail of the group-CV solve: given each fold's pre-built normal-equation
// pieces (M_g, rhs_g, held-out ds_g^T ds_g), assemble the full-data system, select
// lambda by the unimodal CV-error search (warm or coarse-to-fine), and return the
// final all-data fit. Both the exact (per-row) and the per-stamp builders feed this
// the same mf/rf/ysq, so the selection logic lives in one place.
GlsResult cv_finish(const std::vector<Eigen::MatrixXd>& mf,
                    const std::vector<Eigen::VectorXd>& rf,
                    const std::vector<double>& ysq, int n_groups,
                    const ThinPlateBasis& basis, int n, int nc, int k,
                    const std::vector<double>& lambda_grid, int warm_start,
                    bool fit_background) {
  const int n_fields = fit_background ? nc + 1 : nc;
  const int p = n_fields * k;

  // Full-data system (sum of folds) + the block-diagonal bending penalty.
  System s;
  s.n = n;
  s.nc = nc;
  s.k = k;
  s.p = p;
  s.m = Eigen::MatrixXd::Zero(p, p);
  s.rhs = Eigen::VectorXd::Zero(p);
  s.ywy = 0.0;
  for (int g = 0; g < n_groups; ++g) {
    s.m += mf[g];
    s.rhs += rf[g];
    s.ywy += ysq[g];
  }
  s.p0 = Eigen::MatrixXd::Zero(p, p);
  const Eigen::MatrixXd& pen = basis.penalty();
  for (int b = 0; b < n_fields; ++b) s.p0.block(b * k, b * k, k, k) = pen;

  // Held-out weighted SSE for every (lambda, fold): fit on M_all - M_g, predict
  // fold g via ds_g^T ds_g - 2 theta^T rhs_g + theta^T M_g theta.
  const int nl = static_cast<int>(lambda_grid.size());
  const double kInf = std::numeric_limits<double>::infinity();
  std::vector<double> curve(nl, kInf);
  std::vector<char> done(nl, 0);

  // Evaluate the total CV error at a set of lambda indices. Each (lambda, fold) is
  // an independent small dense solve -- this nl*n_groups factorisation grid is the
  // dominant cost of the fit (benchmarks/PERFORMANCE.md), so the work is the
  // (collapsed) parallel loop below.
  auto evaluate = [&](const std::vector<int>& idx) {
    const int m = static_cast<int>(idx.size());
    Eigen::MatrixXd err(m, n_groups);
#pragma omp parallel for collapse(2) schedule(dynamic)
    for (int ii = 0; ii < m; ++ii) {
      for (int g = 0; g < n_groups; ++g) {
        const Eigen::MatrixXd a = (s.m - mf[g]) + lambda_grid[idx[ii]] * s.p0;
        // SPD training normal matrix -> Cholesky (LLT); fall back to the robust
        // pivoted LDLT only when a fold's system is not positive-definite (an
        // under-constrained field at tiny lambda). See solve_at().
        const Eigen::VectorXd rhs_g = s.rhs - rf[g];
        const Eigen::LLT<Eigen::MatrixXd> llt(a);
        Eigen::VectorXd th;
        if (llt.info() == Eigen::Success)
          th = llt.solve(rhs_g);
        else
          th = Eigen::LDLT<Eigen::MatrixXd>(a).solve(rhs_g);
        const double e = ysq[g] - 2.0 * th.dot(rf[g]) + th.dot(mf[g] * th);
        err(ii, g) = e > 0.0 ? e : 0.0;
      }
    }
    for (int ii = 0; ii < m; ++ii) {
      curve[idx[ii]] = err.row(ii).sum();
      done[idx[ii]] = 1;
    }
  };

  // Both search strategies below exploit the same property: the held-out CV error
  // is unimodal in log-lambda (the bias-variance tradeoff is convex over a log
  // grid), so the global minimum is found without sweeping the whole grid.
  auto argmin_done = [&]() {
    int b = -1;
    double bc = kInf;
    for (int li = 0; li < nl; ++li)
      if (done[li] && curve[li] < bc) {
        bc = curve[li];
        b = li;
      }
    return b;
  };

  if (warm_start >= 0 && warm_start < nl) {
    // Warm start: a later IRLS pass clips a handful of pixels off a curve already
    // located the previous pass, so the optimum barely moves. Seed a +/-1 bracket
    // at the hint and descend -- each step evaluates the unexplored neighbour of
    // the current argmin, walking downhill until both neighbours (or the grid
    // edges) are evaluated and no lower. On a unimodal curve that interior
    // minimum is the global one, identical to the coarse-to-fine pick, but in a
    // handful of factorisations instead of ~half the grid.
    std::vector<int> seed;
    for (int li = std::max(0, warm_start - 1);
         li <= std::min(nl - 1, warm_start + 1); ++li)
      seed.push_back(li);
    evaluate(seed);
    while (true) {
      const int b = argmin_done();
      std::vector<int> step;
      if (b > 0 && !done[b - 1]) step.push_back(b - 1);
      if (b < nl - 1 && !done[b + 1]) step.push_back(b + 1);
      if (step.empty()) break;  // both neighbours pinned -> interior minimum
      evaluate(step);
    }
  } else {
    // Cold start: coarse-to-fine. Evaluate a stride-`kCoarseStep` coarse subset,
    // then refine the bracket around its minimum. With stride 2 the true grid
    // minimum is always within +/-2 of the coarse argmin (an interior minimum
    // lies between, or on, the two bracketing coarse samples), so the selected
    // lambda is identical to the full-grid sweep for any unimodal curve -- at
    // ~0.6x the factorisations. The reduced-chi2 / effective-dof diagnostics come
    // from the exact final solve below, not the curve, so they are unaffected.
    constexpr int kCoarseStep = 2;
    std::vector<int> coarse;
    for (int li = 0; li < nl; li += kCoarseStep) coarse.push_back(li);
    if (coarse.empty() || coarse.back() != nl - 1) coarse.push_back(nl - 1);
    evaluate(coarse);

    int coarse_best = coarse.front();
    for (int li : coarse)
      if (curve[li] < curve[coarse_best]) coarse_best = li;

    std::vector<int> refine;
    for (int li = std::max(0, coarse_best - kCoarseStep);
         li <= std::min(nl - 1, coarse_best + kCoarseStep); ++li)
      if (!done[li]) refine.push_back(li);
    if (!refine.empty()) evaluate(refine);
  }

  int best_i = 0;
  double best_cv = kInf;
  for (int li = 0; li < nl; ++li)
    if (done[li] && curve[li] < best_cv) {
      best_cv = curve[li];
      best_i = li;
    }

  // Final fit on all pixels at the CV-selected lambda.
  GlsResult best = solve_at(s, lambda_grid[best_i]);
  if (!fit_background) pad_zero_background(best, k);
  best.lambda_grid = lambda_grid;
  best.gcv_curve = std::move(curve);  // holds the CV-error curve here
  return best;
}

}  // namespace

GlsResult solve_gls_cv(const Eigen::Ref<const Eigen::MatrixXd>& points,
                       const Eigen::Ref<const Eigen::VectorXd>& target,
                       const Eigen::Ref<const Eigen::VectorXd>& weights,
                       const Eigen::Ref<const Eigen::MatrixXd>& bn,
                       const ThinPlateBasis& basis,
                       const std::vector<double>& lambda_grid,
                       const std::vector<int>& group, int n_groups,
                       int warm_start, bool fit_background) {
  if (lambda_grid.empty())
    throw std::runtime_error("solve_gls_cv: lambda_grid is empty");
  const int n = static_cast<int>(points.rows());
  if (n_groups < 2 || static_cast<int>(group.size()) != n)
    return solve_gls_gcv(points, target, weights, bn, basis, lambda_grid,
                         fit_background);

  const int nc = static_cast<int>(bn.cols());
  const Eigen::MatrixXd d = basis.design(points);  // N x k
  const int k = static_cast<int>(d.cols());
  const int n_fields = fit_background ? nc + 1 : nc;
  const int p = n_fields * k;

  // Whitened design Xs = W^{1/2} X and target ds = W^{1/2} y (see assemble()).
  // Xs is an N x P matrix rebuilt every IRLS pass; building it serially was ~19%
  // of the GLS solve. The n_fields column blocks are independent (block c is D scaled
  // row-wise by sqrt(w)*B_c, the trailing background block -- omitted when
  // fit_background is false -- by sqrt(w) alone), so build them in parallel.
  const Eigen::ArrayXd sw = weights.array().sqrt();
  Eigen::MatrixXd xs(n, p);
#pragma omp parallel for schedule(static)
  for (int c = 0; c < n_fields; ++c) {
    const Eigen::ArrayXd col =
        (c < nc) ? (bn.col(c).array() * sw).eval() : sw;
    xs.block(0, c * k, n, k) = (d.array().colwise() * col).matrix();
  }
  const Eigen::VectorXd ds = (sw * target.array()).matrix();

  // Row indices per fold, then each fold's normal-equation contribution
  // M_g = Xs_g^T Xs_g, rhs_g = Xs_g^T ds_g, and held-out sum of squares ds_g^T ds_g.
  std::vector<std::vector<int>> rows(n_groups);
  for (int i = 0; i < n; ++i) {
    const int g = group[i];
    if (g < 0 || g >= n_groups)
      throw std::runtime_error("solve_gls_cv: group id out of range");
    rows[g].push_back(i);
  }
  std::vector<Eigen::MatrixXd> mf(n_groups);
  std::vector<Eigen::VectorXd> rf(n_groups);
  std::vector<double> ysq(n_groups, 0.0);
  // Build each fold's M_g = Xs_g^T Xs_g with full row-chunked thread parallelism:
  // the fold loop is serial on the outside, but each fold's rank update is split
  // over row chunks across *all* threads (mirroring assemble()). The earlier
  // one-thread-per-fold loop capped parallelism at n_groups (=5) regardless of core
  // count -- the build is O(N P^2) and was the solve's main scaling limiter.
  //
  // The chunk size is adaptive: a fold holds only ~N/n_groups rows, so a fixed
  // coarse chunk leaves most cores idle (4 chunks over 14 threads capped the build
  // at ~1.3x and made it the solve's headline cost), while a fixed fine chunk pays
  // per-call rankUpdate overhead that slows the serial path. Targeting ~3 chunks
  // per thread engages every core when threaded yet stays coarse at low thread
  // counts -- 1.7x faster at 14 cores with no single-thread regression.
  int n_threads = 1;
#ifdef _OPENMP
  n_threads = omp_get_max_threads();
#endif
  constexpr int kMaxFoldChunk = 2048;
  for (int g = 0; g < n_groups; ++g) {
    const std::vector<int>& idx = rows[g];
    const int nf = static_cast<int>(idx.size());
    const int fold_chunk =
        std::clamp(nf / (3 * n_threads), 256, kMaxFoldChunk);
    Eigen::MatrixXd m_acc = Eigen::MatrixXd::Zero(p, p);
    Eigen::VectorXd rhs_acc = Eigen::VectorXd::Zero(p);
    double ysq_acc = 0.0;
#pragma omp parallel
    {
      Eigen::MatrixXd m_loc = Eigen::MatrixXd::Zero(p, p);
      Eigen::VectorXd rhs_loc = Eigen::VectorXd::Zero(p);
      double ysq_loc = 0.0;
      // Reused per-thread gather buffers (sized to the largest possible chunk).
      Eigen::MatrixXd xb(kMaxFoldChunk, p);
      Eigen::VectorXd yb(kMaxFoldChunk);
#pragma omp for schedule(dynamic) nowait
      for (int r0 = 0; r0 < nf; r0 += fold_chunk) {
        const int nr = std::min(fold_chunk, nf - r0);
        for (int j = 0; j < nr; ++j) {
          xb.row(j) = xs.row(idx[r0 + j]);
          yb(j) = ds(idx[r0 + j]);
        }
        const auto xt = xb.topRows(nr).transpose();
        m_loc.selfadjointView<Eigen::Lower>().rankUpdate(xt);
        rhs_loc.noalias() += xt * yb.head(nr);
        ysq_loc += yb.head(nr).squaredNorm();
      }
#pragma omp critical
      {
        m_acc.triangularView<Eigen::Lower>() += m_loc;
        rhs_acc += rhs_loc;
        ysq_acc += ysq_loc;
      }
    }
    mf[g] = m_acc.selfadjointView<Eigen::Lower>();
    rf[g] = std::move(rhs_acc);
    ysq[g] = ysq_acc;
  }

  return cv_finish(mf, rf, ysq, n_groups, basis, n, nc, k, lambda_grid,
                   warm_start, fit_background);
}

GlsResult solve_gls_cv_stamped(
    const Eigen::Ref<const Eigen::VectorXd>& target,
    const Eigen::Ref<const Eigen::VectorXd>& weights,
    const Eigen::Ref<const Eigen::MatrixXd>& bn,
    const Eigen::Ref<const Eigen::MatrixXd>& stamp_design,
    const std::vector<int>& stamp_id, const std::vector<int>& stamp_fold,
    const ThinPlateBasis& basis, const std::vector<double>& lambda_grid,
    int n_groups, int warm_start, bool fit_background) {
  if (lambda_grid.empty())
    throw std::runtime_error("solve_gls_cv_stamped: lambda_grid is empty");
  const int n = static_cast<int>(target.size());
  const int nc = static_cast<int>(bn.cols());
  const int n_fields = fit_background ? nc + 1 : nc;
  const int k = static_cast<int>(stamp_design.cols());
  const int ns = static_cast<int>(stamp_design.rows());
  const int p = n_fields * k;
  if (weights.size() != n || bn.rows() != n ||
      static_cast<int>(stamp_id.size()) != n)
    throw std::runtime_error("solve_gls_cv_stamped: row counts disagree");
  if (static_cast<int>(stamp_fold.size()) != ns)
    throw std::runtime_error("solve_gls_cv_stamped: stamp_fold size mismatch");

  // Per-stamp moments (the only O(N) pass): A_s = Σ_{i∈s} w_i B_i B_i^T with
  // B_i = [bn_i, (1 if fit_background)] (n_fields), r_s = Σ w_i t_i B_i, and
  // ds2_s = Σ w_i t_i^2.
  std::vector<Eigen::MatrixXd> a_stamp(ns, Eigen::MatrixXd::Zero(n_fields, n_fields));
  std::vector<Eigen::VectorXd> r_stamp(ns, Eigen::VectorXd::Zero(n_fields));
  std::vector<double> ds2(ns, 0.0);
  Eigen::VectorXd b(n_fields);
  for (int i = 0; i < n; ++i) {
    const int s = stamp_id[i];
    if (s < 0 || s >= ns)
      throw std::runtime_error("solve_gls_cv_stamped: stamp_id out of range");
    const double wi = weights(i);
    const double ti = target(i);
    for (int c = 0; c < nc; ++c) b(c) = bn(i, c);
    if (fit_background) b(nc) = 1.0;
    a_stamp[s].selfadjointView<Eigen::Lower>().rankUpdate(b, wi);
    r_stamp[s].noalias() += (wi * ti) * b;
    ds2[s] += wi * ti * ti;
  }
  for (int s = 0; s < ns; ++s)
    a_stamp[s] = a_stamp[s].selfadjointView<Eigen::Lower>();

  // Per-fold normal equations via the Kronecker factorisation:
  //   M_g = Σ_{s∈g} A_s ⊗ (d_s d_s^T),  rhs_g(c-block) = Σ_{s∈g} r_s(c) d_s,
  //   ysq_g = Σ_{s∈g} ds2_s. Folds are independent, so build them in parallel.
  std::vector<Eigen::MatrixXd> mf(n_groups, Eigen::MatrixXd::Zero(p, p));
  std::vector<Eigen::VectorXd> rf(n_groups, Eigen::VectorXd::Zero(p));
  std::vector<double> ysq(n_groups, 0.0);
#pragma omp parallel for schedule(dynamic)
  for (int g = 0; g < n_groups; ++g) {
    Eigen::MatrixXd& m = mf[g];
    Eigen::VectorXd& r = rf[g];
    double y = 0.0;
    for (int s = 0; s < ns; ++s) {
      if (stamp_fold[s] != g) continue;
      const Eigen::VectorXd d = stamp_design.row(s).transpose();
      const Eigen::MatrixXd dd = d * d.transpose();  // k x k, rank-1
      const Eigen::MatrixXd& as = a_stamp[s];
      for (int c = 0; c < n_fields; ++c) {
        for (int cp = 0; cp < n_fields; ++cp)
          m.block(c * k, cp * k, k, k).noalias() += as(c, cp) * dd;
        r.segment(c * k, k).noalias() += r_stamp[s](c) * d;
      }
      y += ds2[s];
    }
    ysq[g] = y;
  }

  return cv_finish(mf, rf, ysq, n_groups, basis, n, nc, k, lambda_grid,
                   warm_start, fit_background);
}

}  // namespace delta
