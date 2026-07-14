"""Runtime/environment identity for FITS header provenance (SPEC §8).

Separate from :meth:`KernelSolution.header_cards` (the *fit* provenance) and
:meth:`Subtractor.config_cards` (the *configuration* provenance) — this module
covers the *run* provenance: what code, on what machine, by whom, when.
"""

from __future__ import annotations

import getpass
import platform
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from . import __version__


def _git_commit() -> str:
    """Short commit hash of the running source checkout, or "unknown".

    Best-effort: most installs (wheels, site-packages) aren't a git checkout,
    so failures here are the common case, not an error.
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return out.stdout.strip() if out.returncode == 0 else "unknown"


def _username() -> str:
    try:
        return getpass.getuser()
    except OSError:
        return "unknown"


def environment_cards() -> dict[str, object]:
    """Software/host identity cards: version, commit, host, user, platform, UTC timestamp."""
    return {
        "DLTVERS": __version__,
        "DLTGITC": _git_commit(),
        "DLTPYVER": platform.python_version(),
        "DLTHOST": socket.gethostname(),
        "DLTUSER": _username(),
        "DLTPLAT": platform.platform(),
        "DLTDATE": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
    }
