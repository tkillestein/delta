#pragma once

#include <utility>
#include <vector>

namespace delta {

// Cartesian Gauss-Hermite (shapelet) kernel basis (SPEC §3.1; Refregier 2003,
// MNRAS 338, 35).
//
// The 2-D basis functions are separable:  phi_{nx,ny}(x,y) = g_nx(x) g_ny(y),
// where the 1-D function of order n and scale beta is
//
//   g_n(x) = [2^n n! sqrt(pi) beta]^(-1/2) H_n(x/beta) exp(-x^2 / 2 beta^2)
//
// with H_n the physicists' Hermite polynomial. Components are truncated by total
// order: nx + ny <= n_max, giving (n_max+1)(n_max+2)/2 components. Separability
// is what lets the convolution engine share work across components (SPEC §3.2).
class GaussHermiteBasis {
 public:
  // `radius` is the kernel footprint half-width; the footprint is
  // (2*radius+1) x (2*radius+1).
  GaussHermiteBasis(double beta, int n_max, int radius);

  // Heuristic footprint half-width covering the largest characteristic scale
  // ~ beta*sqrt(n_max+1).
  static int default_radius(double beta, int n_max);

  double beta() const { return beta_; }
  int n_max() const { return n_max_; }
  int radius() const { return radius_; }
  int ksize() const { return 2 * radius_ + 1; }
  int component_count() const { return static_cast<int>(orders_.size()); }

  // Component c -> (nx, ny) orders, ordered by total order then ny.
  const std::vector<std::pair<int, int>>& orders() const { return orders_; }

  // 1-D sampled basis function for order n (length ksize), double precision.
  const std::vector<double>& basis1d(int n) const { return basis1d_.at(n); }
  // Same, as a float copy for the convolution kernels.
  std::vector<float> kernel1d(int n) const;

  // Full 2-D footprint for component c, row-major (y-major), length ksize^2.
  std::vector<float> kernel2d(int c) const;

  // Footprint sum of each component, S_c = sum_{u,v} phi_c(u,v). This is the
  // per-unit-coefficient photometric scale: the kernel sum at a location is
  // sum_n a_n(x,y) S_n (SPEC §3.3, kernel-sum flux scale).
  std::vector<double> component_sums() const;

 private:
  double beta_;
  int n_max_;
  int radius_;
  std::vector<std::vector<double>> basis1d_;     // [order][sample]
  std::vector<std::pair<int, int>> orders_;      // component -> (nx, ny)
};

}  // namespace delta
