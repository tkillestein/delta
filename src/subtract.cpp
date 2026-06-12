#include "delta/subtract.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "delta/convolve.hpp"

namespace delta {

namespace {

// Reshape theta into the (k x (nc+1)) coefficient matrix C whose column j is
// theta[j*k : (j+1)*k]. theta is laid out as consecutive field blocks, so a
// column-major map is exactly that reshape.
Eigen::MatrixXd coeff_matrix(const Eigen::Ref<const Eigen::VectorXd>& theta,
                             int k, int n_fields) {
  if (theta.size() != static_cast<Eigen::Index>(k) * n_fields)
    throw std::runtime_error("subtract: theta size does not match k*(nc+1)");
  return Eigen::Map<const Eigen::MatrixXd>(theta.data(), k, n_fields);
}

// Per-pixel inverse-variance not needed here; this wraps a variance layer into
// an ImageF so the separable convolution engine can act on it.
ImageF variance_image(const ImageF& src) {
  ImageF v(src.width(), src.height());
  std::copy(src.variance().begin(), src.variance().end(), v.pixels().begin());
  return v;
}

}  // namespace

SpatialFields evaluate_fields(const ThinPlateBasis& spatial,
                              const Eigen::Ref<const Eigen::VectorXd>& theta,
                              int n_components, std::size_t width,
                              std::size_t height) {
  const int k = spatial.n_basis();
  const int n_fields = n_components + 1;  // kernel coeffs + background
  const Eigen::MatrixXd c = coeff_matrix(theta, k, n_fields);

  SpatialFields out;
  out.coeff.assign(n_components, ImageF(width, height));
  out.background = ImageF(width, height);

  const int w = static_cast<int>(width);
  const int h = static_cast<int>(height);

#pragma omp parallel for schedule(static)
  for (int y = 0; y < h; ++y) {
    // Design block for this row (w x k), then fields = design * C (w x n_fields).
    Eigen::MatrixXd points(w, 2);
    for (int x = 0; x < w; ++x) {
      points(x, 0) = static_cast<double>(x);
      points(x, 1) = static_cast<double>(y);
    }
    const Eigen::MatrixXd fields = spatial.design(points) * c;  // w x n_fields
    for (int n = 0; n < n_components; ++n) {
      float* row = out.coeff[n].data() + static_cast<std::size_t>(y) * w;
      for (int x = 0; x < w; ++x) row[x] = static_cast<float>(fields(x, n));
    }
    float* bg = out.background.data() + static_cast<std::size_t>(y) * w;
    for (int x = 0; x < w; ++x)
      bg[x] = static_cast<float>(fields(x, n_components));
  }
  return out;
}

ImageF subtract(const ImageF& science, const ImageF& reference,
                const ThinPlateBasis& spatial,
                const Eigen::Ref<const Eigen::VectorXd>& theta,
                const GaussHermiteBasis& basis) {
  const std::size_t w = science.width();
  const std::size_t h = science.height();
  if (reference.width() != w || reference.height() != h)
    throw std::runtime_error("subtract: science/reference shape mismatch");

  // B_n = phi_n (x) R, one per basis component (SPEC §3.2).
  const std::vector<ImageF> bn = basis_convolve(reference, basis);
  const int nc = static_cast<int>(bn.size());
  const SpatialFields fields = evaluate_fields(spatial, theta, nc, w, h);

  ImageF diff(w, h);
  const int hh = static_cast<int>(h);
  const int ww = static_cast<int>(w);

#pragma omp parallel for schedule(static)
  for (int y = 0; y < hh; ++y) {
    const std::size_t off = static_cast<std::size_t>(y) * w;
    float* d = diff.data() + off;
    const float* s = science.data() + off;
    const float* bg = fields.background.data() + off;
    for (int x = 0; x < ww; ++x) {
      double model = bg[x];
      for (int n = 0; n < nc; ++n)
        model += static_cast<double>(fields.coeff[n].data()[off + x]) *
                 bn[n].data()[off + x];
      d[x] = s[x] - static_cast<float>(model);
    }
  }

  // ---- variance propagation: Var(D) = Var(S) + (K^2 (x) Var(R)) ------------
  if (reference.has_variance() || science.has_variance()) {
    diff.allocate_variance();
    std::vector<float>& var = diff.variance();

    if (reference.has_variance()) {
      const ImageF var_ref = variance_image(reference);
      const auto& orders = basis.orders();
      // Pairwise separable accumulation. phi_n phi_m is separable with x-kernel
      // g_{nx} g_{mx} and y-kernel g_{ny} g_{my}; symmetric in (n,m).
      for (int n = 0; n < nc; ++n) {
        for (int m = n; m < nc; ++m) {
          const auto [nxn, nyn] = orders[n];
          const auto [nxm, nym] = orders[m];
          const auto& gxn = basis.basis1d(nxn);
          const auto& gxm = basis.basis1d(nxm);
          const auto& gyn = basis.basis1d(nyn);
          const auto& gym = basis.basis1d(nym);
          const std::size_t ks = gxn.size();
          std::vector<float> kx(ks), ky(ks);
          for (std::size_t i = 0; i < ks; ++i) {
            kx[i] = static_cast<float>(gxn[i] * gxm[i]);
            ky[i] = static_cast<float>(gyn[i] * gym[i]);
          }
          const ImageF v = convolve_separable(var_ref, kx, ky);
          const float factor = (n == m) ? 1.0f : 2.0f;
          const float* an = fields.coeff[n].data();
          const float* am = fields.coeff[m].data();
          const float* vv = v.data();
          for (std::size_t i = 0; i < var.size(); ++i)
            var[i] += factor * an[i] * am[i] * vv[i];
        }
      }
    }
    if (science.has_variance()) {
      const std::vector<float>& vs = science.variance();
      for (std::size_t i = 0; i < var.size(); ++i) var[i] += vs[i];
    }
  }

  // ---- mask growth (SPEC §3.6) --------------------------------------------
  if (reference.has_mask() || science.has_mask()) {
    diff.allocate_mask();
    std::vector<MaskType>& out = diff.mask();
    const int r = basis.radius();

    // Science mask propagates one-to-one.
    if (science.has_mask()) {
      const std::vector<MaskType>& sm = science.mask();
      for (std::size_t i = 0; i < out.size(); ++i) out[i] |= sm[i];
    }
    // Reference mask is dilated by the kernel half-width: a bad reference pixel
    // contaminates every output pixel within the kernel footprint.
    if (reference.has_mask()) {
      const std::vector<MaskType>& rm = reference.mask();
      for (int y = 0; y < hh; ++y) {
        for (int x = 0; x < ww; ++x) {
          const MaskType flag = rm[static_cast<std::size_t>(y) * w + x];
          if (flag == kMaskGood) continue;
          const int y0 = std::max(0, y - r), y1 = std::min(hh - 1, y + r);
          const int x0 = std::max(0, x - r), x1 = std::min(ww - 1, x + r);
          for (int yy = y0; yy <= y1; ++yy)
            for (int xx = x0; xx <= x1; ++xx)
              out[static_cast<std::size_t>(yy) * w + xx] |= flag;
        }
      }
    }
    // Edge border of one kernel half-width, and any non-finite difference pixel.
    for (int y = 0; y < hh; ++y) {
      for (int x = 0; x < ww; ++x) {
        const std::size_t i = static_cast<std::size_t>(y) * w + x;
        if (x < r || x >= ww - r || y < r || y >= hh - r)
          out[i] |= kMaskEdge;
        if (!std::isfinite(diff.data()[i])) out[i] |= kMaskNonFinite;
      }
    }
  }

  return diff;
}

}  // namespace delta
