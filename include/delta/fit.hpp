#pragma once

#include <Eigen/Dense>
#include <cstddef>
#include <vector>

#include "delta/basis.hpp"
#include "delta/image.hpp"
#include "delta/solve.hpp"
#include "delta/spatial.hpp"

namespace delta {

// Outcome of fitting the matching kernel from image stamps (SPEC §3.3).
struct KernelFit {
  GlsResult gls;                       // theta + GCV diagnostics
  std::vector<double> component_sums;  // S_n = sum phi_n footprint
  int n_pixels = 0;                    // stamp pixels entering the solve
  int n_stamps_used = 0;               // stamps contributing >= 1 pixel
};

// Fit the spatially-varying matching kernel + differential background by
// assembling the penalized GLS over the pixels of the supplied stamps.
//
// For each stamp centre (stamp_x[i], stamp_y[i]) the (2*stamp_radius+1)^2 pixel
// box is gathered (clipped to the frame). Pixels flagged bad in either mask, or
// non-finite, are excluded (SPEC §3.6). The target is the science value; the
// design row is [B_0(x,y) ... B_{nc-1}(x,y)] with B_n = phi_n (x) reference; the
// weight is the inverse of the summed input variances (1 if none supplied). The
// smoothing lambda is selected by GCV over `lambda_grid`.
KernelFit fit_kernel(const ImageF& science, const ImageF& reference,
                     const ThinPlateBasis& spatial,
                     const GaussHermiteBasis& basis,
                     const std::vector<int>& stamp_x,
                     const std::vector<int>& stamp_y, int stamp_radius,
                     const std::vector<double>& lambda_grid);

// Per-pixel photometric scale field m(x,y) = sum_n a_n(x,y) S_n over a
// (width x height) grid (SPEC §3.3). `component_sums` are the S_n; theta is the
// full solver vector (its background block is ignored here). n_components is
// inferred from component_sums.size().
ImageF photometric_scale(const ThinPlateBasis& spatial,
                         const Eigen::Ref<const Eigen::VectorXd>& theta,
                         const std::vector<double>& component_sums,
                         std::size_t width, std::size_t height);

// Photometric scale evaluated at arbitrary points (e.g. stamp centres).
Eigen::VectorXd photometric_scale_at(
    const ThinPlateBasis& spatial,
    const Eigen::Ref<const Eigen::VectorXd>& theta,
    const std::vector<double>& component_sums,
    const Eigen::Ref<const Eigen::MatrixXd>& points);

}  // namespace delta
