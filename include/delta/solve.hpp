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

// Select lambda by k-fold *group* cross-validation rather than GCV. `group[i]`
// assigns pixel i to a fold (0..n_groups-1); folds should be whole stamps so the
// held-out prediction error respects the strong within-stamp pixel correlation
// that makes GCV under-smooth (and over-fit the spatial field). For each lambda
// every fold is predicted from a fit on the others; the lambda minimising total
// held-out weighted SSE is chosen, then the final fit uses all pixels. The
// per-fold normal equations are formed once and reused (M_train = M_all - M_fold),
// so the cost is k*|grid| small dense solves, not k*|grid| re-assemblies.
GlsResult solve_gls_cv(const Eigen::Ref<const Eigen::MatrixXd>& points,
                       const Eigen::Ref<const Eigen::VectorXd>& target,
                       const Eigen::Ref<const Eigen::VectorXd>& weights,
                       const Eigen::Ref<const Eigen::MatrixXd>& bn,
                       const ThinPlateBasis& basis,
                       const std::vector<double>& lambda_grid,
                       const std::vector<int>& group, int n_groups);

}  // namespace delta
