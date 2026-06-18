"""CLI smoke tests: the Typer app drives the pipeline and writes FITS products.

Reuses the synthetic star-field pair from ``test_pipeline`` via a small local
generator so the CLI is exercised end-to-end through real FITS round-trips.
"""

from __future__ import annotations

import delta
import numpy as np
import pytest

typer_testing = pytest.importorskip("typer.testing")
from delta.cli import app  # noqa: E402

runner = typer_testing.CliRunner()


def _stars(h, w, positions, total, sigma):
    img = np.zeros((h, w), np.float64)
    ys, xs = np.mgrid[0:h, 0:w]
    amp = total / (2 * np.pi * sigma**2)
    for x, y in positions:
        img += amp * np.exp(-((xs - x) ** 2 + (ys - y) ** 2) / (2 * sigma**2))
    return img


@pytest.fixture
def fits_pair(tmp_path):
    """Write a science/reference FITS pair (science is the broader seeing)."""
    h = w = 200
    rng = np.random.default_rng(0)
    grid = [(x, y) for x in (40, 80, 120, 160) for y in (40, 80, 120, 160)]
    ref = _stars(h, w, grid, 12000.0, 1.6) + 100 + rng.normal(0, 3, (h, w))
    sci = (
        _stars(h, w, grid, 12000.0, 2.4)
        + _stars(h, w, [(60, 140)], 6000.0, 2.4)
        + 100
        + rng.normal(0, 3, (h, w))
    )
    sci_p, ref_p = tmp_path / "sci.fits", tmp_path / "ref.fits"
    delta.write_fits(str(sci_p), sci.astype(np.float32))
    delta.write_fits(str(ref_p), ref.astype(np.float32))
    return sci_p, ref_p


def test_version():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert delta.__version__ in result.stdout


def test_info(fits_pair):
    sci, _ = fits_pair
    result = runner.invoke(app, ["info", str(sci)])
    assert result.exit_code == 0, result.exception


def test_subtract_writes_products(fits_pair, tmp_path):
    sci, ref = fits_pair
    out = tmp_path / "diff.fits"
    sol = tmp_path / "sol.npz"
    result = runner.invoke(
        app,
        [
            "subtract",
            str(sci),
            str(ref),
            "-o",
            str(out),
            "--gain",
            "1.5",
            "--read-noise",
            "4",
            "--n-knots",
            "4",
            "--stamp-radius",
            "12",
            "--score",
            "--save-solution",
            str(sol),
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert out.exists() and sol.exists()

    layers = delta.read_fits(str(out))
    assert layers["data"].shape == (200, 200)
    # The transient should survive subtraction as a positive peak.
    diff = layers["data"]
    assert diff[140 - 8 : 140 + 9, 60 - 8 : 60 + 9].max() > 5 * np.std(diff)

    loaded = delta.KernelSolution.load(str(sol))
    assert loaded.direction == "reference"


def test_apply_reuses_saved_solution(fits_pair, tmp_path):
    sci, ref = fits_pair
    sol = tmp_path / "sol.npz"
    # Fit once, saving the solution.
    fit = runner.invoke(
        app,
        [
            "subtract",
            str(sci),
            str(ref),
            "-o",
            str(tmp_path / "d1.fits"),
            "--gain",
            "1.5",
            "--n-knots",
            "4",
            "--stamp-radius",
            "12",
            "--save-solution",
            str(sol),
        ],
    )
    assert fit.exit_code == 0, fit.stdout

    # Apply the saved solution without re-fitting.
    out = tmp_path / "d2.fits"
    result = runner.invoke(
        app,
        ["apply", str(sol), str(sci), str(ref), "-o", str(out), "--gain", "1.5"],
    )
    assert result.exit_code == 0, result.stdout
    assert out.exists()

    diff = delta.read_fits(str(out))["data"]
    assert diff.shape == (200, 200)
    # The transient survives as a positive peak.
    assert diff[140 - 8 : 140 + 9, 60 - 8 : 60 + 9].max() > 5 * np.std(diff)


def test_apply_missing_solution(fits_pair, tmp_path):
    sci, ref = fits_pair
    result = runner.invoke(
        app,
        ["apply", str(tmp_path / "nope.npz"), str(sci), str(ref), "-o", str(tmp_path / "d.fits")],
    )
    assert result.exit_code == 2


def test_subtract_refuses_overwrite(fits_pair, tmp_path):
    sci, ref = fits_pair
    out = tmp_path / "diff.fits"
    out.write_bytes(b"")  # pre-existing
    result = runner.invoke(app, ["subtract", str(sci), str(ref), "-o", str(out)])
    assert result.exit_code == 2


def test_subtract_missing_input(fits_pair, tmp_path):
    _, ref = fits_pair
    result = runner.invoke(
        app, ["subtract", str(tmp_path / "nope.fits"), str(ref), "-o", str(tmp_path / "d.fits")]
    )
    assert result.exit_code == 2
