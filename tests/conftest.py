"""Shared test fixtures, including the JPG visual-preview helper.

The `preview` fixture writes a side-by-side JPG (science | reference | difference
| ...) under tests/artifacts/ so end-to-end results can be eyeballed. It is a
no-op if Pillow is not installed, so it never blocks the suite.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

ARTIFACTS = Path(__file__).parent / "artifacts"


def _gray(arr, vmin, vmax):
    a = np.clip((np.asarray(arr, dtype=np.float64) - vmin) / (vmax - vmin + 1e-12), 0, 1)
    return (a * 255).astype(np.uint8)


def _panel(label, arr):
    """Grayscale a panel: difference/score symmetric about 0, else 1-99 pct."""
    arr = np.asarray(arr, dtype=np.float64)
    if label in ("difference", "score"):
        m = float(np.percentile(np.abs(arr), 99)) or 1.0
        return _gray(arr, -m, m)
    lo, hi = np.percentile(arr, [1, 99])
    return _gray(arr, lo, hi if hi > lo else lo + 1.0)


@pytest.fixture(autouse=True)
def _silence_delta_logging():
    """Restore delta's silent-by-default logging after every test.

    The CLI tests call ``configure_logging``, which binds a loguru sink to the
    (captured) ``sys.stderr`` and enables the ``delta`` namespace globally. Left
    in place, that stale sink leaks into later library-only tests and raises
    "I/O operation on closed file" once pytest tears the captured stream down.
    Re-asserting the library default (no sinks, namespace disabled) isolates it.
    """
    from loguru import logger

    yield
    logger.remove()
    logger.disable("delta")


@pytest.fixture
def preview():
    """Return save(name, **panels) writing tests/artifacts/<name>.jpg (or no-op)."""

    def _save(name, **panels):
        try:
            from PIL import Image, ImageDraw
        except ImportError:
            return None

        grays = [(label, _panel(label, arr)) for label, arr in panels.items()]
        pad, label_h = 4, 14
        height = max(g.shape[0] for _, g in grays)
        total_w = sum(g.shape[1] for _, g in grays) + pad * (len(grays) + 1)
        canvas = Image.new("L", (total_w, height + label_h + pad), color=32)
        draw = ImageDraw.Draw(canvas)
        x = pad
        for label, g in grays:
            canvas.paste(Image.fromarray(g), (x, label_h))
            draw.text((x, 2), label, fill=255)
            x += g.shape[1] + pad

        ARTIFACTS.mkdir(exist_ok=True)
        path = ARTIFACTS / f"{name}.jpg"
        canvas.convert("RGB").save(path, "JPEG", quality=85)
        return path

    return _save
