"""delta — high-performance astronomical difference imaging.

A modern, statistically-principled reformulation of the Alard & Lupton (1998)
PSF-matching algorithm. See ``docs/SPEC.md`` for the design.
"""

from __future__ import annotations

from ._core import __version__, weighted_mean

__all__ = ["__version__", "weighted_mean"]
