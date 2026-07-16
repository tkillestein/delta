"""Runtime/environment identity for FITS header provenance (SPEC §8).

Separate from :meth:`KernelSolution.header_cards` (the *fit* provenance) and
:meth:`Subtractor.config_cards` (the *configuration* provenance) — this module
covers the *run* provenance: what code, on what machine, by whom, when.
"""

from __future__ import annotations

import getpass
import platform
import re
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from . import __version__


def _is_delta_checkout(toplevel: Path) -> bool:
    """Whether `toplevel` is delta's own repo root, not an unrelated one.

    A venv created inside a user's own git repo puts site-packages inside
    that repo's working tree, so `git rev-parse --show-toplevel` run from
    the installed package directory finds the *user's* repo, not delta's.
    Guard against that by checking the discovered root actually looks like
    delta's checkout (its pyproject.toml declares `name = "delta"`) rather
    than trusting whatever repo happens to contain us.
    """
    pyproject = toplevel / "pyproject.toml"
    if not pyproject.is_file():
        return False
    try:
        text = pyproject.read_text()
    except OSError:
        return False
    return re.search(r'(?m)^name\s*=\s*"delta"\s*$', text) is not None


def _git_commit() -> str:
    """Short commit hash of the running source checkout, or "unknown".

    Best-effort: most installs (wheels, site-packages) aren't a git checkout,
    so failures here are the common case, not an error. Also "unknown" if the
    nearest git repo isn't delta's own (see `_is_delta_checkout`) — recording
    an unrelated project's commit as delta's would be silently wrong
    provenance, worse than "unknown".
    """
    cwd = Path(__file__).resolve().parent
    try:
        top = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=2,
        )
        if top.returncode != 0:
            return "unknown"
        if not _is_delta_checkout(Path(top.stdout.strip())):
            return "unknown"
        out = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=cwd,
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
