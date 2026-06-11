"""delta — high-performance astronomical difference imaging.

A modern, statistically-principled reformulation of the Alard & Lupton (1998)
PSF-matching algorithm. See ``docs/SPEC.md`` for the design.
"""

from __future__ import annotations

from ._core import __version__, read_fits, weighted_mean, write_fits

__all__ = ["__version__", "read_fits", "weighted_mean", "write_fits"]
