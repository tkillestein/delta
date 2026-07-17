#pragma once

#include <Eigen/Dense>
#include <vector>

#include "delta/spatial.hpp"

namespace delta {

// Result of the penalized GLS fit (SPEC §3.3).
//
// `theta` is ordered [ c_0 (k) | c_1 (k) | ... | c_{nc-1} (k) | b (k) ], i.e.
// the k spatial coefficients for each of the nc kernel components followed by
// the k background coefficients. The spatial field for component n at a point p
// is design(p) · theta[n*k : (n+1)*k]; the background is the last k block.
struct GlsResult {
  Eigen::VectorXd theta;
  int n_components = 0;       // nc
  int n_spatial = 0;          // k
  double lambda = 0.0;
  double gcv = 0.0;
  double effective_dof = 0.0;  // tr S(lambda)
  double rss = 0.0;            // weighted residual sum of squares
  std::vector<double> lambda_grid;  // populated by the GCV search
  std::vector<double> gcv_curve;    // GCV at each grid lambda
};

// Solve the penalized GLS at a fixed smoothing lambda.
//   points  : (N, 2) pixel coordinates
//   target  : (N)    image being matched to (science or reference)
//   weights : (N)    inverse-variance weights (W = diag(weights))
//   bn      : (N, nc) per-pixel basis-convolved template values B_n
GlsResult solve_gls(const Eigen::Ref<const Eigen::MatrixXd>& points,
                    const Eigen::Ref<const Eigen::VectorXd>& target,
                    const Eigen::Ref<const Eigen::VectorXd>& weights,
                    const Eigen::Ref<const Eigen::MatrixXd>& bn,
                    const ThinPlateBasis& basis, double lambda);

// As above but selecting lambda by minimising GCV over `lambda_grid`.
GlsResult solve_gls_gcv(const Eigen::Ref<const Eigen::MatrixXd>& points,
                        const Eigen::Ref<const Eigen::VectorXd>& target,
                        const Eigen::Ref<const Eigen::VectorXd>& weights,
                        const Eigen::Ref<const Eigen::MatrixXd>& bn,
                        const ThinPlateBasis& basis,
                        const std::vector<double>& lambda_grid);

// Design-taking variant: `design` is the precomputed N x k spatial design
// basis.design(points). The point-based overload is a thin wrapper around
// this. Exists so an IRLS caller whose pixel coordinates are fixed across
// passes (fit_kernel) can evaluate the O(N k^2) design once and re-use it,
// instead of rebuilding it inside the solver every pass.
GlsResult solve_gls_gcv_design(const Eigen::Ref<const Eigen::MatrixXd>& design,
                               const Eigen::Ref<const Eigen::VectorXd>& target,
                               const Eigen::Ref<const Eigen::VectorXd>& weights,
                               const Eigen::Ref<const Eigen::MatrixXd>& bn,
                               const ThinPlateBasis& basis,
                               const std::vector<double>& lambda_grid);

// Select lambda by k-fold *group* cross-validation rather than GCV. `group[i]`
// assigns pixel i to a fold (0..n_groups-1); folds should be whole stamps so the
// held-out prediction error respects the strong within-stamp pixel correlation
// that makes GCV under-smooth (and over-fit the spatial field). For each lambda
// every fold is predicted from a fit on the others; the lambda minimising total
// held-out weighted SSE is chosen, then the final fit uses all pixels. The
// per-fold normal equations are formed once and reused (M_train = M_all - M_fold),
// so the cost is k*|grid| small dense solves, not k*|grid| re-assemblies.
//
// `warm_start` optionally seeds the lambda search at a known-good grid index (the
// previous IRLS pass's optimum, whose clipping barely moves the curve). When >= 0
// the search evaluates a bracket around the hint and descends to the interior
// minimum instead of sweeping a coarse subset -- a handful of factorisations vs.
// ~half the grid. The CV error is unimodal in log-lambda, so the descent lands on
// the same lambda the full coarse-to-fine search would pick. Pass -1 for the
// cold-start (first-pass) coarse-to-fine search.
GlsResult solve_gls_cv(const Eigen::Ref<const Eigen::MatrixXd>& points,
                       const Eigen::Ref<const Eigen::VectorXd>& target,
                       const Eigen::Ref<const Eigen::VectorXd>& weights,
                       const Eigen::Ref<const Eigen::MatrixXd>& bn,
                       const ThinPlateBasis& basis,
                       const std::vector<double>& lambda_grid,
                       const std::vector<int>& group, int n_groups,
                       int warm_start = -1);

// Design-taking variant of solve_gls_cv (see solve_gls_gcv_design): `design`
// is the precomputed N x k spatial design basis.design(points).
GlsResult solve_gls_cv_design(const Eigen::Ref<const Eigen::MatrixXd>& design,
                              const Eigen::Ref<const Eigen::VectorXd>& target,
                              const Eigen::Ref<const Eigen::VectorXd>& weights,
                              const Eigen::Ref<const Eigen::MatrixXd>& bn,
                              const ThinPlateBasis& basis,
                              const std::vector<double>& lambda_grid,
                              const std::vector<int>& group, int n_groups,
                              int warm_start = -1);

// Per-stamp factorised group-CV solve (SPEC §3.2-3.3 approximation). Stamps are
// small (≈30 px) relative to the knot spacing (which sets the field length-scale),
// so the spatial design row d(x,y) is ~constant across a stamp. Freezing it at the
// stamp's representative row d_s collapses the design row to the Kronecker product
// B_i ⊗ d_s, so the O(N·P²) normal-equation build factorises into
//
//   M = Σ_s A_s ⊗ (d_s d_s^T),   A_s = Σ_{i∈s} w_i B_i B_i^T   (B_i = [bn_i, 1]),
//
// i.e. an O(N·nc²) per-stamp accumulate plus an O(S·P²) Kronecker assembly --
// hundreds× fewer FLOPs than the exact per-row rankUpdate when stamps ≪ knots. The
// caller supplies the representative spatial design per stamp (`stamp_design`, S×k,
// e.g. the design at each stamp's pixel centroid), the stamp index per pixel
// (`stamp_id`, in [0,S)), and the CV fold per stamp (`stamp_fold`, in [0,n_groups)).
// The lambda search and final fit are identical to solve_gls_cv. Only valid when the
// stamp is small vs the knot spacing; the caller gates on that and falls back to the
// exact solve otherwise.
GlsResult solve_gls_cv_stamped(
    const Eigen::Ref<const Eigen::VectorXd>& target,
    const Eigen::Ref<const Eigen::VectorXd>& weights,
    const Eigen::Ref<const Eigen::MatrixXd>& bn,
    const Eigen::Ref<const Eigen::MatrixXd>& stamp_design,
    const std::vector<int>& stamp_id, const std::vector<int>& stamp_fold,
    const ThinPlateBasis& basis, const std::vector<double>& lambda_grid,
    int n_groups, int warm_start = -1);

}  // namespace delta
