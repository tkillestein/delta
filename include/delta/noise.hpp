#pragma once

#include <Eigen/Dense>
#include <cstddef>
#include <vector>

#include "delta/basis.hpp"
#include "delta/image.hpp"
#include "delta/spatial.hpp"

namespace delta {

// Noise decorrelation (whitening) and the match-filtered score image (SPEC §3.4).
// The whitening filter below follows the proper-subtraction construction of
// Zackay, Ofek & Gal-Yam 2016 ("ZOGY", ApJ 830, 27).
//
// The matching convolution K (x) R correlates the difference-image noise, so a
// naive threshold on the difference is statistically invalid. With white science
// noise (variance vs) and white reference noise (variance vr), the difference
// noise power spectrum is
//
//   P(k) = vs + vr |Khat(k)|^2 ,
//
// and the zero-phase decorrelation filter that flattens it is
//
//   Phi(k) = sqrt(vs + vr * sum K^2) / sqrt(vs + vr |Khat(k)|^2) .
//
// The numerator normalises Phi to ~1 at the average magnitude, so the whitened
// difference keeps the same per-pixel noise variance as the input difference but
// with (approximately) uncorrelated noise.

// Build Phi(k) over an (n x (n/2+1)) real-to-complex half-spectrum (row-major),
// for a kernel footprint `kernel` of side `ksize` and local noise variances.
std::vector<float> decorrelation_filter(const std::vector<float>& kernel,
                                        int ksize, double var_science,
                                        double var_reference, int n);

// Apply a precomputed Fourier filter to an (n x n) real block (circular).
ImageF apply_filter_fft(const ImageF& block, const std::vector<float>& filter,
                        int n);

// Real-space decorrelation kernel: inverse transform of `filter`, fft-shifted to
// be centred in an (n x n) image (for inspection / QA).
ImageF decorrelation_kernel_image(const std::vector<float>& filter, int n);

// Spatially-varying decorrelation of a difference image via apodized overlapping
// FFT blocks (SPEC §3.4). For each block the matching kernel is reconstructed at
// the block centre from the spatial solution (theta over the Gauss-Hermite +
// thin-plate model) and the local noise from the (per-pixel) variance maps;
// blocks are blended with a Hann window. `block` is the FFT block side; the
// stride is block/2.
//
// The matching kernel varies on the knot length-scale (>> the block stride), so
// |Khat|^2 -- the only kernel-derived input to the filter -- is cached on a coarse
// lattice of `kernel_cell_blocks` x `kernel_cell_blocks` blocks (one kernel FFT per
// cell instead of per block; the per-block noise levels and the two data FFTs stay
// exact). `kernel_cell_blocks <= 0` auto-selects the cell size from the knot
// spacing; `1` recomputes the kernel per block (exact, used for validation).
ImageF decorrelate(const ImageF& difference, const ThinPlateBasis& spatial,
                   const Eigen::Ref<const Eigen::VectorXd>& theta,
                   const GaussHermiteBasis& basis, const ImageF& var_science,
                   const ImageF& var_reference, int block,
                   int kernel_cell_blocks = 0);

// Match-filtered score image: the normalised correlation of `image` with the
// point-source profile `psf` (side `psf_size`), giving a per-pixel S/N map.
// `variance` is a same-shape image of per-pixel noise variance (e.g. the
// propagated difference-image variance). Each output pixel is normalised by
// sqrt(var(x,y) * sum(psf^2)), so source-free noise is a unit Gaussian even
// under spatially-varying noise. Pixels with zero or negative variance yield 0.
ImageF matched_filter(const ImageF& image, const std::vector<float>& psf,
                      int psf_size, const ImageF& variance);

// Convenience overload with a spatially-constant noise variance.
ImageF matched_filter(const ImageF& image, const std::vector<float>& psf,
                      int psf_size, double noise_var);

}  // namespace delta
