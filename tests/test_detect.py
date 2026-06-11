"""M3 tests: background estimation, stamp detection, FWHM, convolution direction.

Validated on synthetic star fields with known positions and PSF widths.
"""

import delta
import numpy as np

FWHM_PER_SIGMA = 2.354820045030949


def add_star(img, x0, y0, sigma, flux):
    """Add a normalised 2-D Gaussian of total `flux` centred at (x0, y0)."""
    h, w = img.shape
    yy, xx = np.mgrid[0:h, 0:w]
    g = np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2.0 * sigma**2))
    img += (flux * g / g.sum()).astype(img.dtype)


def peak_flux(peak, sigma):
    """Total flux giving a Gaussian peak value of `peak`."""
    return peak * 2.0 * np.pi * sigma**2


def test_estimate_background_recovers_level_and_noise():
    rng = np.random.default_rng(1)
    img = (100.0 + rng.normal(0.0, 2.0, size=(256, 256))).astype(np.float32)
    median, sigma = delta.estimate_background(img)
    assert abs(median - 100.0) < 0.3
    assert abs(sigma - 2.0) < 0.3


def test_detect_finds_injected_stars_and_positions():
    rng = np.random.default_rng(2)
    img = (100.0 + rng.normal(0.0, 1.0, size=(200, 200))).astype(np.float32)
    positions = [(40, 50), (120, 60), (80, 150), (160, 140)]
    for x0, y0 in positions:
        add_star(img, x0, y0, sigma=2.0, flux=peak_flux(400.0, 2.0))

    stamps = delta.detect_stamps(img, stamp_radius=12, threshold_sigma=5.0)
    found = set(zip(stamps["x"].tolist(), stamps["y"].tolist(), strict=True))
    for x0, y0 in positions:
        assert any(abs(x0 - fx) <= 1 and abs(y0 - fy) <= 1 for fx, fy in found)
    assert len(stamps["x"]) == len(positions)


def test_detect_rejects_edge_and_saturated():
    rng = np.random.default_rng(3)
    img = (100.0 + rng.normal(0.0, 1.0, size=(200, 200))).astype(np.float32)
    add_star(img, 100, 100, sigma=2.0, flux=peak_flux(300.0, 2.0))  # ok
    add_star(img, 4, 100, sigma=2.0, flux=peak_flux(300.0, 2.0))  # within border
    add_star(img, 150, 60, sigma=2.0, flux=peak_flux(900.0, 2.0))  # saturated

    stamps = delta.detect_stamps(img, stamp_radius=12, threshold_sigma=5.0, saturation=500.0)
    found = set(zip(stamps["x"].tolist(), stamps["y"].tolist(), strict=True))
    assert any(abs(100 - fx) <= 1 and abs(100 - fy) <= 1 for fx, fy in found)
    assert all(fx > 12 for fx, _ in found)  # edge star excluded
    # No surviving stamp sits on the saturated source.
    assert not any(abs(150 - fx) <= 2 and abs(60 - fy) <= 2 for fx, fy in found)


def test_fwhm_estimate_recovers_injected_width():
    img = np.full((200, 200), 100.0, dtype=np.float32)
    sigma = 2.5
    add_star(img, 100, 100, sigma=sigma, flux=peak_flux(500.0, sigma))
    stamps = delta.detect_stamps(img, stamp_radius=16, threshold_sigma=5.0)
    assert len(stamps["x"]) == 1
    expected = FWHM_PER_SIGMA * sigma
    assert abs(stamps["fwhm"][0] - expected) / expected < 0.2


def test_select_stamps_direction_reference_when_science_broader():
    rng = np.random.default_rng(4)
    science = (100.0 + rng.normal(0, 1, (200, 200))).astype(np.float32)
    reference = (100.0 + rng.normal(0, 1, (200, 200))).astype(np.float32)
    positions = [(50, 50), (140, 60), (90, 150)]
    for x0, y0 in positions:
        add_star(science, x0, y0, sigma=3.0, flux=peak_flux(400.0, 3.0))
        add_star(reference, x0, y0, sigma=1.5, flux=peak_flux(400.0, 1.5))

    sel = delta.select_stamps(science, reference, stamp_radius=12)
    assert len(sel["x"]) == len(positions)
    assert sel["median_fwhm_science"] > sel["median_fwhm_reference"]
    assert sel["direction"] == "reference"  # sharper reference gets convolved


def test_select_stamps_direction_science_when_reference_broader():
    rng = np.random.default_rng(5)
    science = (100.0 + rng.normal(0, 1, (200, 200))).astype(np.float32)
    reference = (100.0 + rng.normal(0, 1, (200, 200))).astype(np.float32)
    for x0, y0 in [(60, 60), (130, 70), (100, 150)]:
        add_star(science, x0, y0, sigma=1.5, flux=peak_flux(400.0, 1.5))
        add_star(reference, x0, y0, sigma=3.0, flux=peak_flux(400.0, 3.0))

    sel = delta.select_stamps(science, reference, stamp_radius=12)
    assert sel["direction"] == "science"


def test_select_stamps_from_catalog():
    img_sci = np.full((200, 200), 100.0, dtype=np.float32)
    img_ref = np.full((200, 200), 100.0, dtype=np.float32)
    positions = [(70, 70), (130, 110)]
    for x0, y0 in positions:
        add_star(img_sci, x0, y0, sigma=3.0, flux=peak_flux(500.0, 3.0))
        add_star(img_ref, x0, y0, sigma=1.5, flux=peak_flux(500.0, 1.5))

    cx = np.array([p[0] for p in positions], dtype=np.int32)
    cy = np.array([p[1] for p in positions], dtype=np.int32)
    sel = delta.select_stamps(img_sci, img_ref, catalog_x=cx, catalog_y=cy, stamp_radius=16)
    assert len(sel["x"]) == len(positions)
    assert sel["direction"] == "reference"
    assert sel["median_fwhm_science"] > sel["median_fwhm_reference"]
