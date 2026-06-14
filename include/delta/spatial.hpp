#pragma once

#include <Eigen/Dense>

namespace delta {

// Low-rank thin-plate regression spline (SPEC §3.3).
//
// Models a smooth 2-D field over k knots as
//
//   f(x,y) = radial(x,y) · delta  +  [1, u, v] · a
//
// where the radial part uses the thin-plate Green's function U(r) = r^2 log r
// between the (normalised) evaluation point and each knot, filtered onto the
// null space of the affine trend so the affine part [1, u, v] is unpenalised.
// Coordinates are normalised by the knot bounding box for conditioning. The
// basis has k columns: (k-3) radial (penalised) followed by 3 affine.
//
// This is the spatial half of the penalised GLS; the weighted solve and GCV
// selection of the smoothing parameter live in the solver (M4).
class ThinPlateBasis {
 public:
  explicit ThinPlateBasis(const Eigen::Ref<const Eigen::MatrixXd>& knots);

  int n_knots() const { return static_cast<int>(knots_.rows()); }
  int n_basis() const { return n_knots(); }

  // Design matrix (m, n_basis) at the (m, 2) points: [ radial | 1, u, v ].
  Eigen::MatrixXd design(const Eigen::Ref<const Eigen::MatrixXd>& points) const;

  // Bending-energy penalty (n_basis, n_basis): block-diag(Z^T E Z, 0_3).
  const Eigen::MatrixXd& penalty() const { return penalty_; }

  // Smallest centre-to-centre knot spacing in the original (input) coordinate
  // units. The fields vary on this length-scale; callers use it to choose a
  // coarse evaluation lattice (see evaluate_fields).
  double min_knot_spacing() const;

 private:
  Eigen::MatrixXd knots_;    // (k, 2) normalised knot coordinates
  Eigen::MatrixXd nullspace_;  // (k, k-3) affine null space (Z)
  Eigen::MatrixXd penalty_;  // (k, k)
  double cx_ = 0.0, cy_ = 0.0, scale_ = 1.0;
};

// Regular nx*ny grid of knots spanning [x0, x1] x [y0, y1]; returns (nx*ny, 2).
Eigen::MatrixXd grid_knots(double x0, double y0, double x1, double y1, int nx,
                           int ny);

// Unweighted penalised fit: solve (D^T D + lambda P) theta = D^T values.
// (The inverse-variance-weighted GLS is M4.)
Eigen::VectorXd tps_fit(const ThinPlateBasis& basis,
                        const Eigen::Ref<const Eigen::MatrixXd>& points,
                        const Eigen::Ref<const Eigen::VectorXd>& values,
                        double lambda);

}  // namespace delta
