#pragma once

#include <Eigen/Dense>
#include <cstddef>
#include <vector>

#include "delta/basis.hpp"
#include "delta/image.hpp"
#include "delta/spatial.hpp"

namespace delta {

// Noise decorrelation (whitening) and the match-filtered score image (SPEC §3.4).
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
ImageF decorrelate(const ImageF& difference, const ThinPlateBasis& spatial,
                   const Eigen::Ref<const Eigen::VectorXd>& theta,
                   const GaussHermiteBasis& basis, const ImageF& var_science,
                   const ImageF& var_reference, int block);

// Match-filtered score image: the normalised correlation of `image` with the
// point-source profile `psf` (side `psf_size`), giving a per-pixel S/N map for
// white noise of variance `noise_var`. A source of amplitude A with profile psf
// produces a peak of A * sqrt(sum psf^2 / noise_var); source-free noise is unit
// Gaussian.
ImageF matched_filter(const ImageF& image, const std::vector<float>& psf,
                      int psf_size, double noise_var);

}  // namespace delta
