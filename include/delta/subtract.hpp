#pragma once

#include <Eigen/Dense>
#include <cstddef>
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
  std::vector<ImageF> coeff;  // one field per kernel component
  ImageF background;
};

// Evaluate a_n(x,y) and b(x,y) from a fitted theta on a (width x height) grid.
//
// `theta` is the solver output, ordered [ c_0 | c_1 | ... | c_{nc-1} | b ] with
// k = spatial.n_basis() coefficients per field; n_components = nc. The fields
// are continuous across the whole frame (no tiling seams): each pixel row is a
// design block times the reshaped coefficient matrix.
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
ImageF subtract(const ImageF& science, const ImageF& reference,
                const ThinPlateBasis& spatial,
                const Eigen::Ref<const Eigen::VectorXd>& theta,
                const GaussHermiteBasis& basis);

}  // namespace delta
