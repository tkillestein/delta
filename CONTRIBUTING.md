# Contributing to delta

Thanks for your interest in contributing! `delta` is an astronomical difference-imaging
engine: a C++20 core (`delta::` namespace) with a thin nanobind Python layer.
`docs/SPEC.md` is the authoritative design document — section numbers (e.g. "SPEC §3.6")
are referenced throughout the code.

## Development setup

The toolchain is [uv](https://docs.astral.sh/uv/) + [ruff](https://docs.astral.sh/ruff/)
+ [ty](https://github.com/astral-sh/ty). The C++ extension is built by scikit-build-core.

System dependencies (via `pkg-config`): a C++20 compiler, CMake ≥ 3.18, Eigen ≥ 3.4, and
CFITSIO. The FFT is vendored (header-only PocketFFT in `extern/`), so no system FFT
library is needed. OpenMP is used if found.

```sh
uv sync                       # create venv, build the C++ core, install dev deps
uv run pytest                 # run the full test suite
uv run pytest tests/test_fit.py::test_name   # run a single test
uv run ruff check             # lint
uv run ruff format            # format
uv run ty check               # type-check
```

> **Rebuilding the C++ core:** after editing any C++ source/header, recompile with
> `uv sync --reinstall-package delta`. A plain `uv sync` / `uv run pytest` does **not**
> rebuild on source changes (uv keys reinstall on the package version, which is
> unchanged) and will silently run the stale binary. Python-only changes need no rebuild.

Enable the pre-commit hook once per clone (it runs ruff fix/format + `ty check` on staged
Python and gates the commit):

```sh
git config core.hooksPath .githooks
```

## Pull requests

- Branch off `main`; keep changes focused.
- Add or update tests in `tests/` (they mirror the modules); `uv run pytest` must pass.
- Run `uv run ruff check`, `uv run ruff format`, and `uv run ty check` before pushing.
- CI builds, lints, type-checks, and tests on Linux and macOS.
- For changes that touch the algorithm or its claims, reference the relevant SPEC section
  and, where appropriate, the literature (see the References in the README).
- Note user-facing changes in `CHANGELOG.md` under "Unreleased".

## Reporting issues

Please include the `delta` version, platform/compiler, and a minimal reproducer
(ideally a small synthetic array) when filing a bug.
