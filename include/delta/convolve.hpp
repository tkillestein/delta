#pragma once

#include <vector>

#include "delta/basis.hpp"
#include "delta/image.hpp"

namespace delta {

// Separable 2-D convolution with zero padding and 'same' output size.
// `kx` is applied along x (fast axis), `ky` along y; both must have odd length
// and are centred on their middle sample. This is a true convolution (the
// kernel is mirrored), so odd-order Gauss-Hermite components behave correctly.
ImageF convolve_separable(const ImageF& image, const std::vector<float>& kx,
                          const std::vector<float>& ky);

// Convolve `image` with every Gauss-Hermite basis component, producing
// B_n = phi_n ⊗ R (SPEC §3.2), one ImageF per component in `basis.orders()`
// order. The x-pass is shared across components that share an nx.
std::vector<ImageF> basis_convolve(const ImageF& image,
                                   const GaussHermiteBasis& basis);

}  // namespace delta
