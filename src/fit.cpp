#include "delta/fit.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
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

// (K² ⊗ Var)(x,y) with K frozen and zero-pad at the frame edge. Matches the
// exact independent-pixel convolution variance when a is constant over the
// kernel footprint (SPEC §3.6). Masked reference pixels contribute 0 (they are
// median-filled in the model, so they carry no Poisson noise into B_n).
double conv_var_at(int x, int y, const std::vector<float>& k2, int ks, int r,
                   const float* var, int iw, int ih, std::size_t w,
                   const MaskType* mask) {
  double acc = 0.0;
  for (int dv = -r; dv <= r; ++dv) {
    const int sy = y - dv;
    if (sy < 0 || sy >= ih) continue;
    const float* krow = k2.data() + static_cast<std::size_t>(dv + r) * ks;
    const float* vrow = var + static_cast<std::size_t>(sy) * w;
    const MaskType* mrow =
        mask ? mask + static_cast<std::size_t>(sy) * w : nullptr;
    for (int du = -r; du <= r; ++du) {
      const int sx = x - du;
      if (sx < 0 || sx >= iw) continue;
      if (mrow && mrow[sx] != kMaskGood) continue;
      const float v = vrow[sx];
      if (!(v > 0.0f) || !std::isfinite(v)) continue;
      acc += static_cast<double>(krow[du + r]) * v;
    }
  }
  return acc;
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
  const int ks = basis.ksize();
  const int iw = static_cast<int>(w);
  const int ih = static_cast<int>(h);

  // Gather all candidate stamp pixels, tagging each with the index of the
  // contributing stamp it belongs to so whole stamps can be clipped together.
  // Target- and convolved-image variances are kept separately: the residual is
  // target - K (x) conv, whose variance is Var_target + (K² ⊗ Var_conv), so the
  // convolved layer must be weighted with the kernel-convolved noise map.
  const bool have_var = science.has_variance() || reference.has_variance();
  const bool have_cvar = reference.has_variance();
  std::vector<double> px, py, target, var_t, var_c, bn_flat;  // bn_flat (N x nc)
  std::vector<int> pix_stamp;     // contributing-stamp index per pixel
  std::vector<int> stamp_cx, stamp_cy;  // centre of each contributing stamp

  const float ref_fill = reference_fill_value(reference);
  const bool ref_has_mask = reference.has_mask();
  const MaskType* ref_mask = ref_has_mask ? reference.mask().data() : nullptr;
  const float* ref_var = have_cvar ? reference.variance().data() : nullptr;

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

    // Reference patch covering the stamp plus a kernel-radius halo, sanitised
    // like full-frame subtract: masked / non-finite pixels take the median fill
    // so they do not ring B_n inside the stamp (detect only clears the stamp
    // itself, not the convolution halo).
    const int ex0 = std::max(0, x0 - r), ex1 = std::min(iw - 1, x1 + r);
    const int ey0 = std::max(0, y0 - r), ey1 = std::min(ih - 1, y1 + r);
    const int pw = ex1 - ex0 + 1, ph = ey1 - ey0 + 1;
    ImageF patch(pw, ph);
    for (int yy = 0; yy < ph; ++yy) {
      const int gy = ey0 + yy;
      const float* rrow = reference.data() + static_cast<std::size_t>(gy) * w;
      float* prow = patch.data() + static_cast<std::size_t>(yy) * pw;
      for (int xx = 0; xx < pw; ++xx) {
        const int gx = ex0 + xx;
        const float v = rrow[gx];
        const bool bad =
            (ref_has_mask &&
             ref_mask[static_cast<std::size_t>(gy) * w + gx] != kMaskGood) ||
            !std::isfinite(v);
        prow[xx] = bad ? ref_fill : v;
      }
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

        // Per-pixel target variance; reference variance is retained for the
        // first-pass crude weight and as a positivity gate. IRLS replaces the
        // Q·Var_c term with (K² ⊗ Var_c) using the full reference variance map.
        const double vt =
            science.has_variance() ? static_cast<double>(science.variance()[i])
                                   : 0.0;
        const double vc = have_cvar ? static_cast<double>(ref_var[i]) : 0.0;
        if (have_var && !(vt + vc > 0.0)) continue;  // drop non-positive variance

        bool finite_bn = true;
        for (int n = 0; n < nc; ++n)
          if (!std::isfinite(bn[n].data()[li])) {
            finite_bn = false;
            break;
          }
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

  // Basis footprints (shared across IRLS passes) for reconstructing K = Σ a_n φ_n
  // at each stamp centre when building (K² ⊗ Var_c).
  std::vector<std::vector<float>> phi;
  if (have_cvar) {
    phi.resize(nc);
    for (int n = 0; n < nc; ++n) phi[n] = basis.kernel2d(n);
  }

  // Per-pixel weights, updated each pass from the current kernel (IRLS):
  //   w_i = 1 / (Var_target,i + (K² ⊗ Var_conv)_i).
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
      // Stamps (≈2*stamp_radius wide) are far smaller than the knot spacing (the
      // field length-scale), so the spatial design is ~constant across a stamp;
      // freezing it per stamp lets the O(N·P²) normal-equation build factorise
      // (solve_gls_cv_stamped). Only safe when stamp ≪ knot spacing; otherwise the
      // exact per-row solve is used. DELTA_STAMP_APPROX forces the choice (1/0) for
      // validation; unset = the automatic scale gate.
      // Gate: require the knot spacing to be >= 16x the stamp width. The frozen-
      // design error grows steeply as the stamp approaches the knot scale; 16x
      // keeps the difference-image relative error below ~1e-3 (well under the
      // image noise floor) -- measured in benchmarks/validate_stamped_solve.py.
      const double knot_spacing = spatial.min_knot_spacing();
      const double stamp_width = 2.0 * stamp_radius + 1.0;
      bool use_stamped = knot_spacing > 0.0 && knot_spacing >= 16.0 * stamp_width;
      if (const char* e = std::getenv("DELTA_STAMP_APPROX"))
        use_stamped = std::atoi(e) != 0;

      if (use_stamped) {
        // Map the active (unclipped) stamps to contiguous local ids, accumulate
        // each stamp's pixel centroid (the representative spatial location), and
        // record its CV fold (== global stamp index mod cv_folds, matching the
        // leave-stamp-out grouping the exact path uses).
        std::vector<int> g2l(n_stamps, -1);
        std::vector<int> stamp_id(npix);
        std::vector<double> sx, sy;
        std::vector<int> cnt, stamp_fold;
        for (int j = 0; j < npix; ++j) {
          const int gid = pix_stamp[rows[j]];
          int loc = g2l[gid];
          if (loc < 0) {
            loc = static_cast<int>(sx.size());
            g2l[gid] = loc;
            sx.push_back(0.0);
            sy.push_back(0.0);
            cnt.push_back(0);
            stamp_fold.push_back(gid % cv_folds);
          }
          stamp_id[j] = loc;
          sx[loc] += points(j, 0);
          sy[loc] += points(j, 1);
          cnt[loc] += 1;
        }
        const int s_active = static_cast<int>(sx.size());
        Eigen::MatrixXd centroids(s_active, 2);
        for (int s = 0; s < s_active; ++s) {
          centroids(s, 0) = sx[s] / cnt[s];
          centroids(s, 1) = sy[s] / cnt[s];
        }
        const Eigen::MatrixXd stamp_design = spatial.design(centroids);  // S x k
        gls = solve_gls_cv_stamped(tgt, wts, bn_mat, stamp_design, stamp_id,
                                   stamp_fold, spatial, lambda_grid, cv_folds,
                                   warm_start);
      } else {
        // Group folds: pixels inherit their stamp's fold, so whole stamps are
        // held out (leave-stamp-out CV).
        std::vector<int> grp(npix);
        for (int j = 0; j < npix; ++j) grp[j] = pix_stamp[rows[j]] % cv_folds;
        gls = solve_gls_cv(points, tgt, wts, bn_mat, spatial, lambda_grid, grp,
                           cv_folds, warm_start);
      }
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

    // Refine weights: Var(resid) = Var_t + (K² ⊗ Var_c). Freeze K at each stamp
    // centre (stamps ≪ knot scale, same block-effective idea as subtract) and
    // convolve the reference variance; the old Q·Var_c(x) form mis-weights
    // Poisson cores.
    if (have_var) {
      if (have_cvar) {
        std::vector<float> k2(static_cast<std::size_t>(ks) * ks);
        std::vector<std::vector<float>> stamp_k2(n_stamps);
        for (int s = 0; s < n_stamps; ++s) {
          if (!accepted[s] || cnt[s] == 0) continue;
          Eigen::MatrixXd pt(1, 2);
          pt(0, 0) = static_cast<double>(stamp_cx[s]);
          pt(0, 1) = static_cast<double>(stamp_cy[s]);
          const Eigen::MatrixXd a_s = spatial.design(pt) * c;  // 1 x (nc+1)
          std::fill(k2.begin(), k2.end(), 0.0f);
          for (int n = 0; n < nc; ++n) {
            const float an = static_cast<float>(a_s(0, n));
            const float* p = phi[n].data();
            for (std::size_t u = 0; u < k2.size(); ++u) k2[u] += an * p[u];
          }
          for (float& v : k2) v *= v;  // elementwise K²
          stamp_k2[s] = k2;
        }
        for (int j = 0; j < npix; ++j) {
          const int i = rows[j];
          const int s = pix_stamp[i];
          const double vc = conv_var_at(
              static_cast<int>(px[i]), static_cast<int>(py[i]), stamp_k2[s], ks,
              r, ref_var, iw, ih, w, ref_mask);
          const double rv = var_t[i] + vc;
          if (rv > 0.0) weights[i] = 1.0 / rv;
        }
      } else {
        // Science variance only: no convolution term.
        for (int j = 0; j < npix; ++j) {
          const int i = rows[j];
          if (var_t[i] > 0.0) weights[i] = 1.0 / var_t[i];
        }
      }
    }

    // Robust per-stamp clip: drop accepted stamps with chi^2 above
    // median + clip_sigma * 1.4826 * MAD (worst-first, never below the floor).
    // Rejections are only *applied* below if another pass will actually refit
    // on them — otherwise `accepted`/`stamp_accepted` would describe a stamp
    // set the returned `theta` was never fitted on.
    std::vector<int> to_reject;
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
          if (n_active - static_cast<int>(to_reject.size()) <= floor_stamps) break;
          to_reject.push_back(s);
        }
      }
    }

    const int rejected_now = static_cast<int>(to_reject.size());
    if (rejected_now == 0 && iter >= min_pass) break;
    if (iter >= hard_max) break;  // no further pass; leave `accepted` untouched.

    for (int s : to_reject) accepted[s] = 0;
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
