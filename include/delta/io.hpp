#pragma once

#include <string>

#include "delta/image.hpp"

namespace delta::io {

// Read a FITS file into an ImageF.
//
// The primary HDU supplies the pixel data. If IMAGE extensions named
// "VARIANCE" and/or "MASK" are present, they populate the image's variance and
// mask layers respectively (SPEC §3.6). The primary image must be 2-D.
ImageF read_image(const std::string& path);

// Write an ImageF to a multi-extension FITS file: primary HDU = pixel data,
// plus "VARIANCE" and "MASK" image extensions when those layers are present.
// When `overwrite` is true an existing file at `path` is replaced.
void write_image(const std::string& path, const ImageF& image,
                 bool overwrite = true);

}  // namespace delta::io
