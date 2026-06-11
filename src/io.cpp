#include "delta/io.hpp"

#include <fitsio.h>

#include <stdexcept>
#include <string>

namespace delta::io {
namespace {

[[noreturn]] void throw_fits(int status, const std::string& context) {
  char msg[FLEN_STATUS] = {0};
  fits_get_errstatus(status, msg);
  throw std::runtime_error("delta::io FITS error (" + context + "): " + msg);
}

void check(int status, const std::string& context) {
  if (status != 0) throw_fits(status, context);
}

// Move to a named 2-D IMAGE extension, reading its pixels into `dst` with the
// given CFITSIO datatype. Returns false (without error) if no such extension
// exists; the search always restarts from the primary HDU.
template <typename T>
bool read_named_extension(fitsfile* fptr, const char* extname, int datatype,
                          T* dst, long nelements) {
  int status = 0;
  int hdutype = 0;
  fits_movabs_hdu(fptr, 1, &hdutype, &status);  // back to primary first
  check(status, "rewind to primary");

  if (fits_movnam_hdu(fptr, IMAGE_HDU, const_cast<char*>(extname), 0, &status)) {
    return false;  // extension absent; status intentionally ignored
  }
  long fpixel[2] = {1, 1};
  fits_read_pix(fptr, datatype, fpixel, nelements, nullptr, dst, nullptr,
                &status);
  check(status, std::string("read ") + extname);
  return true;
}

}  // namespace

ImageF read_image(const std::string& path) {
  fitsfile* fptr = nullptr;
  int status = 0;

  fits_open_file(&fptr, path.c_str(), READONLY, &status);
  check(status, "open " + path);

  int naxis = 0;
  fits_get_img_dim(fptr, &naxis, &status);
  check(status, "image dimensions");
  if (naxis != 2) {
    fits_close_file(fptr, &status);
    throw std::runtime_error(
        "delta::io::read_image expects a 2-D primary image, got NAXIS=" +
        std::to_string(naxis));
  }

  long naxes[2] = {0, 0};
  fits_get_img_size(fptr, 2, naxes, &status);
  check(status, "image size");
  const auto width = static_cast<std::size_t>(naxes[0]);
  const auto height = static_cast<std::size_t>(naxes[1]);

  ImageF image(width, height);
  const long n = static_cast<long>(image.size());
  long fpixel[2] = {1, 1};
  fits_read_pix(fptr, TFLOAT, fpixel, n, nullptr, image.data(), nullptr,
                &status);
  check(status, "read primary pixels");

  // Optional noise / mask layers (SPEC §3.6).
  image.allocate_variance();
  if (!read_named_extension(fptr, "VARIANCE", TFLOAT, image.variance().data(),
                            n)) {
    image.variance().clear();  // absent → leave unset
  }
  image.allocate_mask();
  if (!read_named_extension(fptr, "MASK", TBYTE, image.mask().data(), n)) {
    image.mask().clear();
  }

  fits_close_file(fptr, &status);
  check(status, "close " + path);
  return image;
}

void write_image(const std::string& path, const ImageF& image, bool overwrite) {
  fitsfile* fptr = nullptr;
  int status = 0;

  const std::string name = (overwrite ? "!" : "") + path;
  fits_create_file(&fptr, name.c_str(), &status);
  check(status, "create " + path);

  long naxes[2] = {static_cast<long>(image.width()),
                   static_cast<long>(image.height())};
  long fpixel[2] = {1, 1};
  const long n = static_cast<long>(image.size());

  fits_create_img(fptr, FLOAT_IMG, 2, naxes, &status);
  check(status, "create primary");
  fits_write_pix(fptr, TFLOAT, fpixel, n, const_cast<float*>(image.data()),
                 &status);
  check(status, "write primary");

  if (image.has_variance()) {
    fits_create_img(fptr, FLOAT_IMG, 2, naxes, &status);
    check(status, "create variance");
    fits_write_key_str(fptr, "EXTNAME", "VARIANCE",
                       "per-pixel variance (noise layer)", &status);
    fits_write_pix(fptr, TFLOAT, fpixel, n,
                   const_cast<float*>(image.variance().data()), &status);
    check(status, "write variance");
  }

  if (image.has_mask()) {
    fits_create_img(fptr, BYTE_IMG, 2, naxes, &status);
    check(status, "create mask");
    fits_write_key_str(fptr, "EXTNAME", "MASK", "bad-pixel mask (0 = good)",
                       &status);
    fits_write_pix(fptr, TBYTE, fpixel, n,
                   const_cast<MaskType*>(image.mask().data()), &status);
    check(status, "write mask");
  }

  fits_close_file(fptr, &status);
  check(status, "close " + path);
}

}  // namespace delta::io
