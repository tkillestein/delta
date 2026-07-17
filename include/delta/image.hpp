#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace delta {

// Per-pixel bad-pixel flags. 0 == good; otherwise a bitmask of reasons.
// Masks are first-class: they are excluded from the fit and grown by the
// kernel footprint during subtraction (see SPEC §3.6).
using MaskType = std::uint8_t;

enum MaskFlag : MaskType {
  kMaskGood      = 0,
  kMaskBad       = 1u << 0,  // generic bad pixel
  kMaskSaturated = 1u << 1,
  kMaskCosmic    = 1u << 2,
  kMaskNonFinite = 1u << 3,  // NaN / inf
  kMaskEdge      = 1u << 4,  // frame border / grown from kernel footprint
  kMaskUser      = 1u << 5,
};

// A 2-D image with optional per-pixel variance and bad-pixel mask.
// Storage is contiguous and row-major (y-major): index = y * width + x.
template <typename T>
class Image {
public:
  Image() = default;
  Image(std::size_t width, std::size_t height)
      : width_(width), height_(height), data_(width * height, T{}) {}

  // Construct directly from a contiguous source range: one allocate+copy
  // pass, unlike `Image(width, height)` followed by a separate fill, which
  // wastes a full zero-initialisation pass over data the caller is about to
  // overwrite anyway. Matters when marshalling an already-ready buffer
  // across a language boundary (e.g. a NumPy array) on a large image, where
  // the redundant pass is a measurable fraction of the call.
  Image(std::size_t width, std::size_t height, const T* src)
      : width_(width), height_(height), data_(src, src + width * height) {}

  // Adopt an existing buffer (no allocation, no copy, no zero-fill): for
  // producers that already hold the finished pixels in a vector (e.g. an
  // accumulator finalised in place) and would otherwise pay a full-frame
  // zero-initialisation plus a copy to get them into an Image.
  Image(std::size_t width, std::size_t height, std::vector<T>&& data)
      : width_(width), height_(height), data_(std::move(data)) {
    if (data_.size() != width * height) data_.resize(width * height, T{});
  }

  std::size_t width() const { return width_; }
  std::size_t height() const { return height_; }
  std::size_t size() const { return width_ * height_; }

  T* data() { return data_.data(); }
  const T* data() const { return data_.data(); }

  // Underlying contiguous buffer (row-major). Exposed so callers can move it
  // out, e.g. into a zero-copy NumPy array at the Python boundary.
  std::vector<T>& pixels() { return data_; }
  const std::vector<T>& pixels() const { return data_; }

  T& operator()(std::size_t x, std::size_t y) { return data_[y * width_ + x]; }
  const T& operator()(std::size_t x, std::size_t y) const {
    return data_[y * width_ + x];
  }

  bool has_variance() const { return !variance_.empty(); }
  bool has_mask() const { return !mask_.empty(); }

  std::vector<T>& variance() { return variance_; }
  const std::vector<T>& variance() const { return variance_; }
  std::vector<MaskType>& mask() { return mask_; }
  const std::vector<MaskType>& mask() const { return mask_; }

  void allocate_variance() { variance_.assign(size(), T{}); }
  void allocate_mask() { mask_.assign(size(), kMaskGood); }

private:
  std::size_t width_ = 0;
  std::size_t height_ = 0;
  std::vector<T> data_;
  std::vector<T> variance_;     // optional, same shape as data_
  std::vector<MaskType> mask_;  // optional, same shape as data_
};

using ImageF = Image<float>;
using ImageD = Image<double>;

// Non-owning, read-only view of an image with optional variance and mask
// layers. The read-only entry points (subtract, fit_kernel, decorrelate,
// matched_filter) take views so the Python boundary can pass NumPy buffers
// through without the full-frame copy an owning Image would cost (~190 MB per
// layer at survey-cadence sizes). Borrow contract: the caller guarantees the
// buffers outlive the call -- at the bindings, nanobind holds the input
// ndarrays alive for the call duration and the GIL-released region does not
// outlive it. An owning Image converts implicitly, so internal callers that
// hold an Image (or tests) are unaffected.
template <typename T>
class ImageView {
public:
  ImageView() = default;
  ImageView(std::size_t width, std::size_t height, const T* data,
            const T* variance = nullptr, const MaskType* mask = nullptr)
      : width_(width),
        height_(height),
        data_(data),
        variance_(variance),
        mask_(mask) {}
  ImageView(const Image<T>& img)  // NOLINT(google-explicit-constructor)
      : width_(img.width()),
        height_(img.height()),
        data_(img.data()),
        variance_(img.has_variance() ? img.variance().data() : nullptr),
        mask_(img.has_mask() ? img.mask().data() : nullptr) {}

  std::size_t width() const { return width_; }
  std::size_t height() const { return height_; }
  std::size_t size() const { return width_ * height_; }

  const T* data() const { return data_; }
  bool has_variance() const { return variance_ != nullptr; }
  bool has_mask() const { return mask_ != nullptr; }
  const T* variance() const { return variance_; }
  const MaskType* mask() const { return mask_; }

private:
  std::size_t width_ = 0;
  std::size_t height_ = 0;
  const T* data_ = nullptr;
  const T* variance_ = nullptr;
  const MaskType* mask_ = nullptr;
};

using ImageViewF = ImageView<float>;

}  // namespace delta
