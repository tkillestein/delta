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
                const ThinPlateBasis& basis) {
  const int n = static_cast<int>(points.rows());
  const int nc = static_cast<int>(bn.cols());
  if (target.size() != n || weights.size() != n || bn.rows() != n)
    throw std::runtime_error("solve_gls: row counts of points/target/weights/bn disagree");

  const Eigen::MatrixXd d = basis.design(points);  // N x k
  const int k = static_cast<int>(d.cols());
  const int p = (nc + 1) * k;

  // Build the pre-whitened design Xs = W^{1/2} X (N x P): each component block is
  // D scaled row-wise by sqrt(weight) * B_n; the trailing block is sqrt(weight) D.
  // With Xs the normal equations are M = Xs^T Xs and rhs = Xs^T ds (ds = W^{1/2} d),
  // so M is a symmetric rank update -- half the FLOPs of the general X^T W X.
  const Eigen::ArrayXd sw = weights.array().sqrt();
  Eigen::MatrixXd xs(n, p);
  for (int c = 0; c < nc; ++c)
    xs.block(0, c * k, n, k) =
        (d.array().colwise() * (bn.col(c).array() * sw)).matrix();
  xs.block(0, nc * k, n, k) = (d.array().colwise() * sw).matrix();
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
  // (the nc kernel-coefficient fields and the background field).
  s.p0 = Eigen::MatrixXd::Zero(p, p);
  const Eigen::MatrixXd& pen = basis.penalty();
  for (int b = 0; b <= nc; ++b) s.p0.block(b * k, b * k, k, k) = pen;
  return s;
}

GlsResult solve_at(const System& s, double lambda) {
  const Eigen::MatrixXd a = s.m + lambda * s.p0;
  const Eigen::LDLT<Eigen::MatrixXd> ldlt(a);

  GlsResult r;
  r.theta = ldlt.solve(s.rhs);
  r.n_components = s.nc;
  r.n_spatial = s.k;
  r.lambda = lambda;

  // Weighted residual sum of squares: ||W^1/2 (d - X theta)||^2
  //   = y^T W y - 2 theta^T (X^T W d) + theta^T (X^T W X) theta.
  double rss = s.ywy - 2.0 * r.theta.dot(s.rhs) + r.theta.dot(s.m * r.theta);
  if (rss < 0.0) rss = 0.0;  // guard against round-off near a perfect fit
  r.rss = rss;

  // Effective degrees of freedom tr S(lambda) = tr(A^{-1} X^T W X).
  r.effective_dof = (ldlt.solve(s.m)).trace();

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
                    const ThinPlateBasis& basis, double lambda) {
  return solve_at(assemble(points, target, weights, bn, basis), lambda);
}

GlsResult solve_gls_gcv(const Eigen::Ref<const Eigen::MatrixXd>& points,
                        const Eigen::Ref<const Eigen::VectorXd>& target,
                        const Eigen::Ref<const Eigen::VectorXd>& weights,
                        const Eigen::Ref<const Eigen::MatrixXd>& bn,
                        const ThinPlateBasis& basis,
                        const std::vector<double>& lambda_grid) {
  if (lambda_grid.empty())
    throw std::runtime_error("solve_gls_gcv: lambda_grid is empty");

  const System s = assemble(points, target, weights, bn, basis);

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
  best.lambda_grid = lambda_grid;
  best.gcv_curve = std::move(curve);
  return best;
}

GlsResult solve_gls_cv(const Eigen::Ref<const Eigen::MatrixXd>& points,
                       const Eigen::Ref<const Eigen::VectorXd>& target,
                       const Eigen::Ref<const Eigen::VectorXd>& weights,
                       const Eigen::Ref<const Eigen::MatrixXd>& bn,
                       const ThinPlateBasis& basis,
                       const std::vector<double>& lambda_grid,
                       const std::vector<int>& group, int n_groups) {
  if (lambda_grid.empty())
    throw std::runtime_error("solve_gls_cv: lambda_grid is empty");
  const int n = static_cast<int>(points.rows());
  if (n_groups < 2 || static_cast<int>(group.size()) != n)
    return solve_gls_gcv(points, target, weights, bn, basis, lambda_grid);

  const int nc = static_cast<int>(bn.cols());
  const Eigen::MatrixXd d = basis.design(points);  // N x k
  const int k = static_cast<int>(d.cols());
  const int p = (nc + 1) * k;

  // Whitened design Xs = W^{1/2} X and target ds = W^{1/2} y (see assemble()).
  const Eigen::ArrayXd sw = weights.array().sqrt();
  Eigen::MatrixXd xs(n, p);
  for (int c = 0; c < nc; ++c)
    xs.block(0, c * k, n, k) =
        (d.array().colwise() * (bn.col(c).array() * sw)).matrix();
  xs.block(0, nc * k, n, k) = (d.array().colwise() * sw).matrix();
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
#pragma omp parallel for schedule(dynamic)
  for (int g = 0; g < n_groups; ++g) {
    const std::vector<int>& idx = rows[g];
    const int nf = static_cast<int>(idx.size());
    Eigen::MatrixXd xf(nf, p);
    Eigen::VectorXd yf(nf);
    for (int j = 0; j < nf; ++j) {
      xf.row(j) = xs.row(idx[j]);
      yf(j) = ds(idx[j]);
    }
    Eigen::MatrixXd m = Eigen::MatrixXd::Zero(p, p);
    if (nf > 0) m.selfadjointView<Eigen::Lower>().rankUpdate(xf.transpose());
    mf[g] = m.selfadjointView<Eigen::Lower>();
    rf[g] = nf > 0 ? Eigen::VectorXd(xf.transpose() * yf)
                   : Eigen::VectorXd::Zero(p);
    ysq[g] = yf.squaredNorm();
  }

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
  for (int b = 0; b <= nc; ++b) s.p0.block(b * k, b * k, k, k) = pen;

  // Held-out weighted SSE for every (lambda, fold): fit on M_all - M_g, predict
  // fold g via ds_g^T ds_g - 2 theta^T rhs_g + theta^T M_g theta. Independent
  // tasks, run in parallel.
  const int nl = static_cast<int>(lambda_grid.size());
  Eigen::MatrixXd err(nl, n_groups);
#pragma omp parallel for collapse(2) schedule(dynamic)
  for (int li = 0; li < nl; ++li) {
    for (int g = 0; g < n_groups; ++g) {
      const Eigen::MatrixXd a = (s.m - mf[g]) + lambda_grid[li] * s.p0;
      const Eigen::LDLT<Eigen::MatrixXd> ldlt(a);
      const Eigen::VectorXd th = ldlt.solve(s.rhs - rf[g]);
      double e = ysq[g] - 2.0 * th.dot(rf[g]) + th.dot(mf[g] * th);
      err(li, g) = e > 0.0 ? e : 0.0;
    }
  }

  std::vector<double> curve(nl);
  int best_i = 0;
  double best_cv = std::numeric_limits<double>::infinity();
  for (int li = 0; li < nl; ++li) {
    curve[li] = err.row(li).sum();
    if (curve[li] < best_cv) {
      best_cv = curve[li];
      best_i = li;
    }
  }

  // Final fit on all pixels at the CV-selected lambda.
  GlsResult best = solve_at(s, lambda_grid[best_i]);
  best.lambda_grid = lambda_grid;
  best.gcv_curve = std::move(curve);  // holds the CV-error curve here
  return best;
}

}  // namespace delta
