#pragma once

#include <Eigen/Dense>
#include <cstddef>
#include <cstdint>
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
  int n_pixels = 0;                    // stamp pixels in the final solve
  int n_stamps_used = 0;               // stamps kept in the final solve
  int n_stamps_total = 0;              // stamps that contributed any pixels
  int n_stamps_rejected = 0;           // stamps dropped by sigma clipping
  double reduced_chi2 = 0.0;           // rss / (n_pixels - effective_dof)
  // Per-stamp goodness-of-fit, one entry per stamp that contributed pixels
  // (parallel arrays). `stamp_chi2` is the per-stamp reduced chi^2 under the
  // final solution; `stamp_accepted` is 1 if the stamp survived clipping.
  std::vector<int> stamp_x, stamp_y;
  std::vector<double> stamp_chi2;
  std::vector<std::uint8_t> stamp_accepted;
};

// Fit the spatially-varying matching kernel + differential background by
// assembling the penalized GLS over the pixels of the supplied stamps.
//
// For each stamp centre (stamp_x[i], stamp_y[i]) the (2*stamp_radius+1)^2 pixel
// box is gathered (clipped to the frame). Pixels flagged bad in either mask, or
// non-finite, are excluded (SPEC §3.6). The reference patch includes a
// kernel-radius halo that is median-filled where masked/non-finite so defects
// just outside the stamp do not ring B_n. The target is the science value; the
// design row is [B_0(x,y) ... B_{nc-1}(x,y)] with B_n = phi_n (x) reference.
// With variance maps, IRLS weights use
//   w = 1 / (Var_target + (K² ⊗ Var_reference))
// with K frozen at each stamp centre (first pass uses the cruder
// 1/(Var_t+Var_c)). Without variance, unit weights. The smoothing lambda is
// selected by GCV over `lambda_grid`.
//
// Bad stamps (variable sources, dipoles from misregistration, cosmics, saturated
// cores) bias the global kernel, so the solve is iterated with per-stamp sigma
// clipping: after each fit the per-stamp reduced chi^2 is computed and stamps
// beyond `clip_sigma` robust deviations above the median are dropped, up to
// `clip_iterations` times. `clip_sigma <= 0` or `clip_iterations <= 0` disables
// clipping (single solve). At least `min_stamps` stamps are always retained.
//
// `cv_folds` chooses the spatial smoothing lambda: 0/1 keeps GCV, while >= 2
// selects lambda by k-fold *group* cross-validation that holds out whole stamps
// (folds = stamp index mod cv_folds). Group CV respects the within-stamp pixel
// correlation that makes GCV under-smooth and over-fit the spatial field.
//
// `fit_background` (default true) fits the differential-background spatial
// field jointly with the kernel, as documented above. Set false to skip it
// entirely (e.g. when the caller's own pipeline already matches sky levels
// upstream): the returned theta still carries the background block, zeroed,
// so callers that evaluate the model unconditionally see a zero background
// field rather than needing to special-case its absence.
KernelFit fit_kernel(const ImageF& science, const ImageF& reference,
                     const ThinPlateBasis& spatial,
                     const GaussHermiteBasis& basis,
                     const std::vector<int>& stamp_x,
                     const std::vector<int>& stamp_y, int stamp_radius,
                     const std::vector<double>& lambda_grid,
                     double clip_sigma = 4.0, int clip_iterations = 5,
                     int min_stamps = 5, int cv_folds = 0,
                     bool fit_background = true);

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
