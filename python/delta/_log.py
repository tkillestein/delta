"""Logging plumbing for the ``delta`` package, built on `loguru`.

Library code logs through loguru's global ``logger``. Following loguru's guidance
for libraries we ``logger.disable("delta")`` at import, so importing ``delta`` is
silent until a host opts in. The CLI (or any application) calls
:func:`configure_logging` to add a sink, set the level, and ``enable`` the
``delta`` namespace. This keeps the "library is quiet, the application decides"
contract while giving us loguru's nicer formatting/colour out of the box.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager

from loguru import logger

# Stay silent until a host enables the "delta" namespace (loguru filters by the
# module __name__ of the call site, all of which live under "delta").
logger.disable("delta")

# Console sink format: timestamp, padded level, message — colourised by loguru.
_CONSOLE_FORMAT = (
    "<green>{time:HH:mm:ss}</green> <level>{level: <7}</level> <level>{message}</level>"
)

__all__ = ["configure_logging", "log_timing", "logger"]


def configure_logging(verbosity: int = 0, *, colorize: bool | None = None) -> None:
    """Configure loguru for application/CLI use and enable ``delta`` logging.

    ``verbosity`` maps to a level: ``<0`` -> WARNING (quiet), ``0`` -> INFO,
    ``>=1`` -> DEBUG. Removes loguru's default handler and installs a single
    ``stderr`` sink so output never collides with FITS products written to
    stdout. Idempotent: re-calling replaces the sink rather than stacking. Pass
    ``colorize`` to force ANSI colour on/off (default: auto-detect the tty).
    """
    level = "WARNING" if verbosity < 0 else "INFO" if verbosity == 0 else "DEBUG"
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format=_CONSOLE_FORMAT,
        colorize=colorize,
        backtrace=False,
        diagnose=False,
    )
    logger.enable("delta")


@contextmanager
def log_timing(label: str, *, level: str = "INFO") -> Iterator[None]:
    """Log ``label`` with its wall-clock duration when the block exits.

    Records are attributed to ``delta._log`` (this module). We deliberately do
    not rewrite the source frame via ``logger.opt(depth=...)``: inside a
    ``@contextmanager`` the frame just above is contextlib's machinery, which
    falls outside the ``delta`` namespace and would slip past ``disable``. The
    console format does not show the call site, so attribution is moot.
    """
    start = time.perf_counter()
    logger.debug("{} ...", label)
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logger.log(level, "{} done in {:.3f}s", label, elapsed)
