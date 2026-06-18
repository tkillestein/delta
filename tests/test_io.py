"""M1 IO tests: FITS round-trip of data, variance, and mask layers."""

import delta
import numpy as np
import pytest


def test_roundtrip_data_only(tmp_path):
    data = np.arange(12, dtype=np.float32).reshape(3, 4)
    path = str(tmp_path / "data_only.fits")

    delta.write_fits(path, data)
    out = delta.read_fits(path)

    np.testing.assert_array_equal(out["data"], data)
    assert out["variance"] is None
    assert out["mask"] is None


def test_roundtrip_with_variance_and_mask(tmp_path):
    rng = np.random.default_rng(0)
    data = rng.standard_normal((5, 7)).astype(np.float32)
    variance = rng.uniform(0.5, 2.0, size=(5, 7)).astype(np.float32)
    mask = np.zeros((5, 7), dtype=np.uint8)
    mask[2, 3] = 1  # flag one pixel as bad
    path = str(tmp_path / "full.fits")

    delta.write_fits(path, data, variance=variance, mask=mask)
    out = delta.read_fits(path)

    out_mask = out["mask"]
    assert out_mask is not None
    np.testing.assert_array_equal(out["data"], data)
    np.testing.assert_array_equal(out["variance"], variance)
    np.testing.assert_array_equal(out_mask, mask)
    assert out_mask[2, 3] == mask[2, 3]


def test_shape_preserved_non_square(tmp_path):
    # Guards against width/height (NAXIS1/NAXIS2) transposition.
    data = np.arange(6, dtype=np.float32).reshape(2, 3)
    path = str(tmp_path / "rect.fits")

    delta.write_fits(path, data)
    out = delta.read_fits(path)

    assert out["data"].shape == (2, 3)
    np.testing.assert_array_equal(out["data"], data)


def test_mask_shape_mismatch_raises(tmp_path):
    data = np.zeros((3, 4), dtype=np.float32)
    bad_mask = np.zeros((4, 3), dtype=np.uint8)
    path = str(tmp_path / "bad.fits")

    with pytest.raises(Exception):  # noqa: B017 - shape validation in C++
        delta.write_fits(path, data, mask=bad_mask)


def test_write_fits_no_overwrite_by_default(tmp_path):
    data = np.zeros((3, 4), dtype=np.float32)
    path = str(tmp_path / "x.fits")

    delta.write_fits(path, data)
    # Default overwrite=False: a second write to an existing path must raise.
    with pytest.raises(Exception):  # noqa: B017 - CFITSIO file-exists error
        delta.write_fits(path, data)
    # ...but overwrite=True replaces it.
    delta.write_fits(path, data, overwrite=True)
    assert delta.read_fits(path)["data"].shape == (3, 4)


def test_read_difference_extension_fallback(tmp_path):
    # Compressed layouts leave a dataless primary HDU and move the frame to a
    # "DIFFERENCE" image extension; read_image must fall back to it (io.cpp).
    fits = pytest.importorskip("astropy.io.fits")
    data = np.arange(12, dtype=np.float32).reshape(3, 4)
    path = str(tmp_path / "diff_ext.fits")
    hdul = fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(data=data, name="DIFFERENCE")])
    hdul.writeto(path)

    out = delta.read_fits(path)
    np.testing.assert_array_equal(out["data"], data)


def test_read_no_image_no_difference_raises(tmp_path):
    fits = pytest.importorskip("astropy.io.fits")
    path = str(tmp_path / "empty.fits")
    fits.HDUList([fits.PrimaryHDU()]).writeto(path)  # dataless, no DIFFERENCE

    with pytest.raises(Exception):  # noqa: B017 - explicit throw in io.cpp
        delta.read_fits(path)
