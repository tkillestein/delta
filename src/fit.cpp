#include "delta/fit.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "delta/convolve.hpp"
#include "delta/subtract.hpp"
#include "delta/timing.hpp"

namespace delta {

namespace {

// Combine the (k x (nc+1)) coefficient matrix from theta, as in subtract.
Eigen::MatrixXd coeff_matrix(const Eigen::Ref<const Eigen::VectorXd>& theta,
                             int k, int n_fields) {
  if (theta.size() != static_cast<Eigen::Index>(k) * n_fields)
    throw std::runtime_error("photometric_scale: theta size mismatch");
  return Eigen::Map<const Eigen::MatrixXd>(theta.data(), k, n_fields);
}

double median_copy(std::vector<double> v) {
  if (v.empty()) return std::numeric_limits<double>::quiet_NaN();
  const std::size_t mid = v.size() / 2;
  std::nth_element(v.begin(), v.begin() + mid, v.end());
  return v[mid];
}

}  // namespace

KernelFit fit_kernel(const ImageF& science, const ImageF& reference,
                     const ThinPlateBasis& spatial,
                     const GaussHermiteBasis& basis,
                     const std::vector<int>& stamp_x,
                     const std::vector<int>& stamp_y, int stamp_radius,
                     const std::vector<double>& lambda_grid, double clip_sigma,
                     int clip_iterations, int min_stamps, int cv_folds) {
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

  // Gather all candidate stamp pixels, tagging each with the index of the
  // contributing stamp it belongs to so whole stamps can be clipped together.
  // Target- and convolved-image variances are kept separately: the residual is
  // target - K (x) conv, whose variance is Var_target + (sum K^2) Var_conv, so
  // the convolved layer's contribution must be scaled by the (fitted) kernel.
  const bool have_var = science.has_variance() || reference.has_variance();
  std::vector<double> px, py, target, var_t, var_c, bn_flat;  // bn_flat (N x nc)
  std::vector<int> pix_stamp;     // contributing-stamp index per pixel
  std::vector<int> stamp_cx, stamp_cy;  // centre of each contributing stamp

  {
   DELTA_TIME("fit: stamp B_n convolve");
   for (std::size_t s = 0; s < stamp_x.size(); ++s) {
    const int cx = stamp_x[s], cy = stamp_y[s];
    const int x0 = std::max(0, cx - stamp_radius);
    const int x1 = std::min(iw - 1, cx + stamp_radius);
    const int y0 = std::max(0, cy - stamp_radius);
    const int y1 = std::min(ih - 1, cy + stamp_radius);
    const int stamp_idx = static_cast<int>(stamp_cx.size());
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

        // Per-pixel target / convolved-image variances (SPEC §3.6).
        const double vt =
            science.has_variance() ? static_cast<double>(science.variance()[i]) : 0.0;
        const double vc = reference.has_variance()
                              ? static_cast<double>(reference.variance()[i])
                              : 0.0;
        if (have_var && !(vt + vc > 0.0)) continue;  // drop non-positive variance

        bool finite_bn = true;
        for (int n = 0; n < nc; ++n)
          if (!std::isfinite(bn[n].data()[li])) { finite_bn = false; break; }
        if (!finite_bn) continue;

        px.push_back(static_cast<double>(x));
        py.push_back(static_cast<double>(y));
        target.push_back(static_cast<double>(sv));
        var_t.push_back(vt);
        var_c.push_back(vc);
        pix_stamp.push_back(stamp_idx);
        for (int n = 0; n < nc; ++n)
          bn_flat.push_back(static_cast<double>(bn[n].data()[li]));
        used = true;
      }
    }
    if (used) {
      stamp_cx.push_back(cx);
      stamp_cy.push_back(cy);
    }
   }
  }

  const int npix_all = static_cast<int>(target.size());
  if (npix_all == 0) throw std::runtime_error("fit_kernel: no usable stamp pixels");
  const int n_stamps = static_cast<int>(stamp_cx.size());
  const int k = spatial.n_basis();

  // Kernel-footprint inner products T_nm = sum_u phi_n(u) phi_m(u); the kernel
  // sum-of-squares at a pixel is Q(x) = a(x)^T T a(x), used to scale the
  // convolved-image noise into the residual variance (and so the weights).
  Eigen::MatrixXd tmat(nc, nc);
  {
    std::vector<std::vector<float>> phi(nc);
    for (int n = 0; n < nc; ++n) phi[n] = basis.kernel2d(n);
    const std::size_t ksz = phi[0].size();
    for (int n = 0; n < nc; ++n)
      for (int m = 0; m <= n; ++m) {
        double s = 0.0;
        for (std::size_t u = 0; u < ksz; ++u)
          s += static_cast<double>(phi[n][u]) * phi[m][u];
        tmat(n, m) = s;
        tmat(m, n) = s;
      }
  }

  // Per-pixel weights, updated each pass from the current kernel (IRLS):
  //   w_i = 1 / (Var_target,i + Q_i Var_conv,i).
  // The first pass uses the cruder 1/(Var_target+Var_conv); without variance,
  // unit weights (the reduced chi^2 is then unnormalised).
  std::vector<double> weights(npix_all, 1.0);
  if (have_var)
    for (int i = 0; i < npix_all; ++i) weights[i] = 1.0 / (var_t[i] + var_c[i]);

  // Iteratively re-fit, dropping stamps whose reduced chi^2 is a robust outlier
  // (variable/dipole/cosmic/saturated sources otherwise bias the global kernel),
  // while the weights converge so the fit's reduced chi^2 approaches 1.
  std::vector<std::uint8_t> accepted(n_stamps, 1);
  std::vector<double> stamp_chi2(n_stamps,
                                 std::numeric_limits<double>::quiet_NaN());
  const int clip_passes = (clip_sigma > 0.0 && clip_iterations > 0)
                              ? clip_iterations
                              : 0;
  // The iterative refit does both IRLS reweighting (so reduced chi^2 -> 1) and
  // per-stamp clipping; it is governed by clip_iterations. With it disabled the
  // fit is a single raw solve under the crude 1/(Var_t+Var_c) weights.
  const int do_iterate = (clip_iterations > 0) ? 1 : 0;
  const int min_pass = (do_iterate && have_var) ? 1 : 0;
  const int hard_max = std::max(clip_passes, min_pass);
  const int floor_stamps = std::max(1, std::min(min_stamps, n_stamps));

  GlsResult gls;
  int npix = 0;
  double reduced_chi2 = std::numeric_limits<double>::quiet_NaN();
  // Carries the previous pass's CV-selected lambda index into the next pass so the
  // search warm-starts at it (the clip barely moves the curve). -1 = cold start.
  int warm_start = -1;
  delta::timing::ScopedTimer solve_timer("fit: GLS solve");
  for (int iter = 0;; ++iter) {
    // Active pixel rows = pixels whose stamp is still accepted.
    std::vector<int> rows;
    rows.reserve(npix_all);
    for (int i = 0; i < npix_all; ++i)
      if (accepted[pix_stamp[i]]) rows.push_back(i);
    npix = static_cast<int>(rows.size());
    if (npix == 0) throw std::runtime_error("fit_kernel: all stamp pixels clipped");

    Eigen::MatrixXd points(npix, 2);
    Eigen::VectorXd tgt(npix), wts(npix);
    Eigen::MatrixXd bn_mat(npix, nc);
    for (int j = 0; j < npix; ++j) {
      const int i = rows[j];
      points(j, 0) = px[i];
      points(j, 1) = py[i];
      tgt(j) = target[i];
      wts(j) = weights[i];
      for (int n = 0; n < nc; ++n)
        bn_mat(j, n) = bn_flat[static_cast<std::size_t>(i) * nc + n];
    }

    if (cv_folds > 1) {
      // Group folds: pixels inherit their stamp's fold, so whole stamps are
      // held out (leave-stamp-out CV).
      std::vector<int> grp(npix);
      for (int j = 0; j < npix; ++j) grp[j] = pix_stamp[rows[j]] % cv_folds;
      gls = solve_gls_cv(points, tgt, wts, bn_mat, spatial, lambda_grid, grp,
                         cv_folds, warm_start);
      // Recover the selected grid index (min of the CV-error curve; unevaluated
      // entries are +inf) to warm-start the next IRLS pass.
      warm_start = static_cast<int>(
          std::min_element(gls.gcv_curve.begin(), gls.gcv_curve.end()) -
          gls.gcv_curve.begin());
    } else {
      gls = solve_gls_gcv(points, tgt, wts, bn_mat, spatial, lambda_grid);
    }

    // Model and per-pixel residual under this solution; fields = design * C.
    const Eigen::MatrixXd design = spatial.design(points);
    const Eigen::MatrixXd c = coeff_matrix(gls.theta, k, nc + 1);
    const Eigen::MatrixXd fields = design * c;  // npix x (nc+1)
    const Eigen::MatrixXd a = fields.leftCols(nc);
    // Q_j = a_j^T T a_j (kernel sum-of-squares at each pixel).
    const Eigen::VectorXd qvec =
        (a * tmat).cwiseProduct(a).rowwise().sum();

    std::vector<double> sum_chi(n_stamps, 0.0);
    std::vector<int> cnt(n_stamps, 0);
    for (int j = 0; j < npix; ++j) {
      double model = fields(j, nc);
      for (int n = 0; n < nc; ++n) model += fields(j, n) * bn_mat(j, n);
      const double resid = tgt(j) - model;
      const int s = pix_stamp[rows[j]];
      sum_chi[s] += wts(j) * resid * resid;
      ++cnt[s];
    }
    std::fill(stamp_chi2.begin(), stamp_chi2.end(),
              std::numeric_limits<double>::quiet_NaN());
    for (int s = 0; s < n_stamps; ++s)
      if (cnt[s] > 0) stamp_chi2[s] = sum_chi[s] / cnt[s];
    const double denom = static_cast<double>(npix) - gls.effective_dof;
    reduced_chi2 = denom > 0.0 ? gls.rss / denom
                               : std::numeric_limits<double>::quiet_NaN();

    // Refine the weights with the kernel sum-of-squares for the next pass.
    if (have_var)
      for (int j = 0; j < npix; ++j) {
        const int i = rows[j];
        const double rv = var_t[i] + qvec(j) * var_c[i];
        if (rv > 0.0) weights[i] = 1.0 / rv;
      }

    // Robust per-stamp clip: drop accepted stamps with chi^2 above
    // median + clip_sigma * 1.4826 * MAD (worst-first, never below the floor).
    int rejected_now = 0;
    if (clip_passes > 0 && iter < clip_passes) {
      std::vector<double> active_chi2;
      active_chi2.reserve(n_stamps);
      for (int s = 0; s < n_stamps; ++s)
        if (accepted[s] && cnt[s] > 0) active_chi2.push_back(stamp_chi2[s]);
      const int n_active = static_cast<int>(active_chi2.size());
      if (n_active > floor_stamps) {
        const double med = median_copy(active_chi2);
        std::vector<double> dev;
        dev.reserve(active_chi2.size());
        for (double v : active_chi2) dev.push_back(std::fabs(v - med));
        const double thr = med + clip_sigma * 1.4826 * median_copy(dev);
        std::vector<int> cand;
        for (int s = 0; s < n_stamps; ++s)
          if (accepted[s] && cnt[s] > 0 && stamp_chi2[s] > thr) cand.push_back(s);
        std::sort(cand.begin(), cand.end(),
                  [&](int aa, int bb) { return stamp_chi2[aa] > stamp_chi2[bb]; });
        for (int s : cand) {
          if (n_active - rejected_now <= floor_stamps) break;
          accepted[s] = 0;
          ++rejected_now;
        }
      }
    }

    if (rejected_now == 0 && iter >= min_pass) break;
    if (iter >= hard_max) break;
  }

  int n_used = 0;
  for (int s = 0; s < n_stamps; ++s)
    if (accepted[s] && std::isfinite(stamp_chi2[s])) ++n_used;

  KernelFit out;
  out.gls = gls;
  out.component_sums = basis.component_sums();
  out.n_pixels = npix;
  out.n_stamps_total = n_stamps;
  out.n_stamps_used = n_used;
  out.n_stamps_rejected = n_stamps - n_used;
  out.reduced_chi2 = reduced_chi2;
  out.stamp_x = stamp_cx;
  out.stamp_y = stamp_cy;
  out.stamp_chi2 = stamp_chi2;
  out.stamp_accepted = accepted;
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
    const float* a = fields.coeff[c].get();
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
