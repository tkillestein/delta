#include "delta/basis.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace delta {

int GaussHermiteBasis::default_radius(double beta, int n_max) {
  const double scale = beta * std::sqrt(static_cast<double>(n_max + 1));
  return std::max(static_cast<int>(std::ceil(3.0 * scale)), 3);
}

GaussHermiteBasis::GaussHermiteBasis(double beta, int n_max, int radius)
    : beta_(beta), n_max_(n_max), radius_(radius) {
  if (beta_ <= 0.0) throw std::invalid_argument("beta must be > 0");
  if (n_max_ < 0) throw std::invalid_argument("n_max must be >= 0");
  if (radius_ < 1) throw std::invalid_argument("radius must be >= 1");

  const int ks = ksize();
  const double pi = std::acos(-1.0);
  const double ln2 = std::log(2.0);
  const double half_ln_pi = 0.5 * std::log(pi);
  const double ln_beta = std::log(beta_);

  // log of the per-order normalisation constant (stable for larger n).
  std::vector<double> log_norm(n_max_ + 1);
  for (int n = 0; n <= n_max_; ++n) {
    log_norm[n] =
        -0.5 * (n * ln2 + std::lgamma(n + 1.0) + half_ln_pi + ln_beta);
  }

  basis1d_.assign(n_max_ + 1, std::vector<double>(ks, 0.0));
  for (int i = 0; i < ks; ++i) {
    const double t = static_cast<double>(i - radius_) / beta_;
    const double gauss = std::exp(-0.5 * t * t);

    // Physicists' Hermite recurrence: H_0=1, H_1=2t, H_{n+1}=2t H_n - 2n H_{n-1}.
    double h_prev = 1.0;
    basis1d_[0][i] = std::exp(log_norm[0]) * h_prev * gauss;
    if (n_max_ >= 1) {
      double h_curr = 2.0 * t;
      basis1d_[1][i] = std::exp(log_norm[1]) * h_curr * gauss;
      for (int n = 2; n <= n_max_; ++n) {
        const double h_next = 2.0 * t * h_curr - 2.0 * (n - 1) * h_prev;
        basis1d_[n][i] = std::exp(log_norm[n]) * h_next * gauss;
        h_prev = h_curr;
        h_curr = h_next;
      }
    }
  }

  // Components by total order, then by ny.
  for (int total = 0; total <= n_max_; ++total) {
    for (int ny = 0; ny <= total; ++ny) {
      orders_.emplace_back(total - ny, ny);  // (nx, ny)
    }
  }
}

std::vector<float> GaussHermiteBasis::kernel1d(int n) const {
  const auto& v = basis1d_.at(n);
  return std::vector<float>(v.begin(), v.end());
}

std::vector<float> GaussHermiteBasis::kernel2d(int c) const {
  const auto [nx, ny] = orders_.at(c);
  const int ks = ksize();
  const auto& gx = basis1d_[nx];
  const auto& gy = basis1d_[ny];
  std::vector<float> k(static_cast<std::size_t>(ks) * ks);
  for (int iy = 0; iy < ks; ++iy) {
    for (int ix = 0; ix < ks; ++ix) {
      k[static_cast<std::size_t>(iy) * ks + ix] =
          static_cast<float>(gy[iy] * gx[ix]);
    }
  }
  return k;
}

}  // namespace delta
