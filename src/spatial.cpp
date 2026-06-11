#include "delta/spatial.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace delta {
namespace {

// Thin-plate Green's function U(r) = r^2 log r, taking squared distance.
// U(r) = 0.5 r^2 log(r^2), and 0 at r = 0.
double tps_green(double r2) {
  if (r2 <= 0.0) return 0.0;
  return 0.5 * r2 * std::log(r2);
}

}  // namespace

ThinPlateBasis::ThinPlateBasis(const Eigen::Ref<const Eigen::MatrixXd>& knots) {
  if (knots.cols() != 2) throw std::invalid_argument("knots must be (k, 2)");
  const int k = static_cast<int>(knots.rows());
  if (k < 3) throw std::invalid_argument("need at least 3 knots");

  // Normalise coordinates by the knot bounding box (centre + max half-extent).
  cx_ = knots.col(0).mean();
  cy_ = knots.col(1).mean();
  const double sx = (knots.col(0).array() - cx_).abs().maxCoeff();
  const double sy = (knots.col(1).array() - cy_).abs().maxCoeff();
  scale_ = std::max({sx, sy, 1e-12});

  knots_.resize(k, 2);
  knots_.col(0) = (knots.col(0).array() - cx_) / scale_;
  knots_.col(1) = (knots.col(1).array() - cy_) / scale_;

  // Knot-to-knot thin-plate matrix E (bending energy of the radial part).
  Eigen::MatrixXd e(k, k);
  for (int i = 0; i < k; ++i) {
    for (int j = 0; j < k; ++j) {
      const double dx = knots_(i, 0) - knots_(j, 0);
      const double dy = knots_(i, 1) - knots_(j, 1);
      e(i, j) = tps_green(dx * dx + dy * dy);
    }
  }

  // Affine trend T = [1, u, v]; its null space carries the radial coefficients.
  Eigen::MatrixXd t(k, 3);
  t.col(0).setOnes();
  t.col(1) = knots_.col(0);
  t.col(2) = knots_.col(1);

  // Z = last (k-3) columns of Q from the QR of T, an orthonormal basis for
  // null(T^T) = {c : T^T c = 0}, the bending-energy domain.
  const Eigen::HouseholderQR<Eigen::MatrixXd> qr(t);
  const Eigen::MatrixXd q =
      qr.householderQ() * Eigen::MatrixXd::Identity(k, k);
  nullspace_ = q.rightCols(k - 3);

  penalty_ = Eigen::MatrixXd::Zero(k, k);
  if (k > 3) {
    penalty_.topLeftCorner(k - 3, k - 3) =
        nullspace_.transpose() * e * nullspace_;
  }
  penalty_ = (0.5 * (penalty_ + penalty_.transpose())).eval();  // symmetrise
}

Eigen::MatrixXd ThinPlateBasis::design(
    const Eigen::Ref<const Eigen::MatrixXd>& points) const {
  if (points.cols() != 2) throw std::invalid_argument("points must be (m, 2)");
  const int m = static_cast<int>(points.rows());
  const int k = n_knots();
  const int nr = k - 3;

  Eigen::MatrixXd un(m, 2);
  un.col(0) = (points.col(0).array() - cx_) / scale_;
  un.col(1) = (points.col(1).array() - cy_) / scale_;

  Eigen::MatrixXd sigma(m, k);
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < k; ++j) {
      const double dx = un(i, 0) - knots_(j, 0);
      const double dy = un(i, 1) - knots_(j, 1);
      sigma(i, j) = tps_green(dx * dx + dy * dy);
    }
  }

  Eigen::MatrixXd d(m, k);
  if (nr > 0) d.leftCols(nr) = sigma * nullspace_;
  d.col(nr).setOnes();
  d.col(nr + 1) = un.col(0);
  d.col(nr + 2) = un.col(1);
  return d;
}

Eigen::MatrixXd grid_knots(double x0, double y0, double x1, double y1, int nx,
                           int ny) {
  if (nx < 1 || ny < 1) throw std::invalid_argument("nx, ny must be >= 1");
  Eigen::MatrixXd knots(static_cast<long>(nx) * ny, 2);
  int idx = 0;
  for (int iy = 0; iy < ny; ++iy) {
    const double fy = ny == 1 ? 0.5 : static_cast<double>(iy) / (ny - 1);
    const double y = y0 + fy * (y1 - y0);
    for (int ix = 0; ix < nx; ++ix) {
      const double fx = nx == 1 ? 0.5 : static_cast<double>(ix) / (nx - 1);
      knots(idx, 0) = x0 + fx * (x1 - x0);
      knots(idx, 1) = y;
      ++idx;
    }
  }
  return knots;
}

Eigen::VectorXd tps_fit(const ThinPlateBasis& basis,
                        const Eigen::Ref<const Eigen::MatrixXd>& points,
                        const Eigen::Ref<const Eigen::VectorXd>& values,
                        double lambda) {
  const Eigen::MatrixXd d = basis.design(points);
  const Eigen::MatrixXd a =
      d.transpose() * d + lambda * basis.penalty();
  const Eigen::VectorXd b = d.transpose() * values;
  return a.ldlt().solve(b);
}

}  // namespace delta
