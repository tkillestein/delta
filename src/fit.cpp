#include "delta/fit.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "delta/convolve.hpp"
#include "delta/subtract.hpp"

namespace delta {

namespace {

// Combine the (k x (nc+1)) coefficient matrix from theta, as in subtract.
Eigen::MatrixXd coeff_matrix(const Eigen::Ref<const Eigen::VectorXd>& theta,
                             int k, int n_fields) {
  if (theta.size() != static_cast<Eigen::Index>(k) * n_fields)
    throw std::runtime_error("photometric_scale: theta size mismatch");
  return Eigen::Map<const Eigen::MatrixXd>(theta.data(), k, n_fields);
}

}  // namespace

KernelFit fit_kernel(const ImageF& science, const ImageF& reference,
                     const ThinPlateBasis& spatial,
                     const GaussHermiteBasis& basis,
                     const std::vector<int>& stamp_x,
                     const std::vector<int>& stamp_y, int stamp_radius,
                     const std::vector<double>& lambda_grid) {
  const std::size_t w = science.width();
  const std::size_t h = science.height();
  if (reference.width() != w || reference.height() != h)
    throw std::runtime_error("fit_kernel: science/reference shape mismatch");
  if (stamp_x.size() != stamp_y.size())
    throw std::runtime_error("fit_kernel: stamp_x/stamp_y length mismatch");

  // B_n = phi_n (x) reference, but only the gathered stamp pixels are used by
  // the solve, so convolving the whole frame is wasteful. For each stamp we
  // convolve just its window plus a one-kernel-radius halo: with the halo in
  // place the zero-padded patch convolution reproduces the full-frame B_n
  // exactly inside the stamp (and the halo clamps to the frame edge, where the
  // zero padding is the same as the full-frame engine's).
  const int nc = basis.component_count();
  const int r = basis.radius();
  const int iw = static_cast<int>(w);
  const int ih = static_cast<int>(h);

  std::vector<double> px, py, target, weights, bn_flat;  // bn_flat row-major (N x nc)
  int n_stamps_used = 0;

  for (std::size_t s = 0; s < stamp_x.size(); ++s) {
    const int cx = stamp_x[s], cy = stamp_y[s];
    const int x0 = std::max(0, cx - stamp_radius);
    const int x1 = std::min(iw - 1, cx + stamp_radius);
    const int y0 = std::max(0, cy - stamp_radius);
    const int y1 = std::min(ih - 1, cy + stamp_radius);
    bool used = false;

    // Reference patch covering the stamp plus a kernel-radius halo, and its
    // per-component basis convolution (patch-local, zero-padded at patch edges).
    const int ex0 = std::max(0, x0 - r), ex1 = std::min(iw - 1, x1 + r);
    const int ey0 = std::max(0, y0 - r), ey1 = std::min(ih - 1, y1 + r);
    const int pw = ex1 - ex0 + 1, ph = ey1 - ey0 + 1;
    ImageF patch(pw, ph);
    for (int yy = 0; yy < ph; ++yy) {
      const float* rrow = reference.data() + static_cast<std::size_t>(ey0 + yy) * w + ex0;
      std::copy(rrow, rrow + pw, patch.data() + static_cast<std::size_t>(yy) * pw);
    }
    const std::vector<ImageF> bn = basis_convolve(patch, basis);

    for (int y = y0; y <= y1; ++y) {
      for (int x = x0; x <= x1; ++x) {
        const std::size_t i = static_cast<std::size_t>(y) * w + x;
        const std::size_t li =
            static_cast<std::size_t>(y - ey0) * pw + (x - ex0);
        if (science.has_mask() && science.mask()[i] != kMaskGood) continue;
        if (reference.has_mask() && reference.mask()[i] != kMaskGood) continue;
        const float sv = science.data()[i];
        if (!std::isfinite(sv)) continue;

        // Inverse-variance weight from the input noise layers (SPEC §3.6).
        double var = 0.0;
        bool have_var = false;
        if (science.has_variance()) { var += science.variance()[i]; have_var = true; }
        if (reference.has_variance()) { var += reference.variance()[i]; have_var = true; }
        double weight = 1.0;
        if (have_var) {
          if (!(var > 0.0)) continue;  // drop non-positive variance
          weight = 1.0 / var;
        }

        bool finite_bn = true;
        for (int n = 0; n < nc; ++n)
          if (!std::isfinite(bn[n].data()[li])) { finite_bn = false; break; }
        if (!finite_bn) continue;

        px.push_back(static_cast<double>(x));
        py.push_back(static_cast<double>(y));
        target.push_back(static_cast<double>(sv));
        weights.push_back(weight);
        for (int n = 0; n < nc; ++n)
          bn_flat.push_back(static_cast<double>(bn[n].data()[li]));
        used = true;
      }
    }
    if (used) ++n_stamps_used;
  }

  const int npix = static_cast<int>(target.size());
  if (npix == 0) throw std::runtime_error("fit_kernel: no usable stamp pixels");

  Eigen::MatrixXd points(npix, 2);
  Eigen::VectorXd tgt(npix), wts(npix);
  Eigen::MatrixXd bn_mat(npix, nc);
  for (int i = 0; i < npix; ++i) {
    points(i, 0) = px[i];
    points(i, 1) = py[i];
    tgt(i) = target[i];
    wts(i) = weights[i];
    for (int n = 0; n < nc; ++n)
      bn_mat(i, n) = bn_flat[static_cast<std::size_t>(i) * nc + n];
  }

  KernelFit out;
  out.gls = solve_gls_gcv(points, tgt, wts, bn_mat, spatial, lambda_grid);
  out.component_sums = basis.component_sums();
  out.n_pixels = npix;
  out.n_stamps_used = n_stamps_used;
  return out;
}

ImageF photometric_scale(const ThinPlateBasis& spatial,
                         const Eigen::Ref<const Eigen::VectorXd>& theta,
                         const std::vector<double>& component_sums,
                         std::size_t width, std::size_t height) {
  const int nc = static_cast<int>(component_sums.size());
  const SpatialFields fields = evaluate_fields(spatial, theta, nc, width, height);

  ImageF scale(width, height);
  const std::size_t n = scale.size();
  float* out = scale.data();
  for (std::size_t i = 0; i < n; ++i) out[i] = 0.0f;
  for (int c = 0; c < nc; ++c) {
    const float s = static_cast<float>(component_sums[c]);
    const float* a = fields.coeff[c].data();
    for (std::size_t i = 0; i < n; ++i) out[i] += s * a[i];
  }
  return scale;
}

Eigen::VectorXd photometric_scale_at(
    const ThinPlateBasis& spatial,
    const Eigen::Ref<const Eigen::VectorXd>& theta,
    const std::vector<double>& component_sums,
    const Eigen::Ref<const Eigen::MatrixXd>& points) {
  const int nc = static_cast<int>(component_sums.size());
  const int k = spatial.n_basis();
  const Eigen::MatrixXd c = coeff_matrix(theta, k, nc + 1);
  // Fields at the points: design (m x k) * C (k x (nc+1)) -> (m x (nc+1)).
  const Eigen::MatrixXd fields = spatial.design(points) * c;
  Eigen::VectorXd s = Eigen::Map<const Eigen::VectorXd>(component_sums.data(), nc);
  return fields.leftCols(nc) * s;
}

}  // namespace delta
