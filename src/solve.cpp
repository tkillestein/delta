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

}  // namespace delta
