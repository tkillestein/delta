#pragma once

#include <Eigen/Dense>
#include <cstddef>
#include <memory>
#include <vector>

#include "delta/basis.hpp"
#include "delta/image.hpp"
#include "delta/spatial.hpp"

namespace delta {

// Spatially-varying coefficient fields evaluated over the full frame.
//
// `coeff[n](x,y)` is a_n(x,y), the field multiplying basis-convolved template
// B_n; `background(x,y)` is the differential background b(x,y) (SPEC §3.2).
struct SpatialFields {
  // Raw row-major (index = y*width + x) buffers, allocated uninitialised: every
  // pixel is written by evaluate_fields before any read, so the value-initialising
  // pass an ImageF would do is pure redundant write traffic over a ~1.9 GB set of
  // fields per full frame. coeff[n] is a_n(x,y); background is b(x,y) (SPEC §3.2).
  std::vector<std::unique_ptr<float[]>> coeff;
  std::unique_ptr<float[]> background;
};

// Evaluate a_n(x,y) and b(x,y) from a fitted theta on a (width x height) grid.
//
// `theta` is the solver output, ordered [ c_0 | c_1 | ... | c_{nc-1} | b ] with
// k = spatial.n_basis() coefficients per field; n_components = nc. The fields
// are continuous across the whole frame (no tiling seams).
//
// The thin-plate fields vary on the knot length-scale (>> a pixel; SPEC §3.2/§3.5),
// so for large frames they are evaluated exactly on a coarse lattice and bilinearly
// interpolated to full resolution -- near-exact (error ~ (stride/knot-spacing)^2) and
// far cheaper than the per-pixel design-matrix rebuild. Small frames fall back to the
// exact per-pixel evaluation.
SpatialFields evaluate_fields(const ThinPlateBasis& spatial,
                              const Eigen::Ref<const Eigen::VectorXd>& theta,
                              int n_components, std::size_t width,
                              std::size_t height);

// Full-frame spatially-varying subtraction (SPEC §3.2, §3.4, §3.6).
//
//   D(x,y) = S(x,y) - sum_n a_n(x,y) B_n(x,y) - b(x,y),   B_n = phi_n (x) R
//
// If either input carries a variance layer the difference carries propagated
// variance Var(S) + (K^2 (x) Var(R)), where the spatially-varying kernel-square
// convolution is evaluated via the separable factorisation
//   (K^2 (x) Var(R))(x,y) = sum_{n,m} a_n a_m [(phi_n phi_m) (x) Var(R)](x,y).
// If either input carries a mask the difference carries the science mask unioned
// with the reference mask dilated by the kernel radius, plus a one-kernel-radius
// edge border (kMaskEdge); non-finite difference pixels are flagged.
// `saturation` (> 0) flags pixels at or above that level in either input as
// kMaskSaturated and grows them by the kernel radius: a bright/saturated stellar
// core is non-linear and never matches the model, so propagating it leaves a
// large spurious residual; masking the affected footprint keeps it out of the
// difference (and downstream detection). 0 disables it.
ImageF subtract(const ImageF& science, const ImageF& reference,
                const ThinPlateBasis& spatial,
                const Eigen::Ref<const Eigen::VectorXd>& theta,
                const GaussHermiteBasis& basis, double saturation = 0.0);

}  // namespace delta
