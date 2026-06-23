"""``delta`` command-line interface.

A thin Typer application over the :mod:`delta` Python API. It reads FITS inputs,
runs the difference-imaging pipeline, and writes a multi-extension FITS product —
with a Rich-rendered result summary and structured logging whose verbosity is
controlled by ``-v`` / ``-q``.

Install the CLI extra to pull in Typer + Rich::

    uv sync --extra cli      # or: pip install 'delta[cli]'

then run ``delta --help``. The console-script entry point is declared in
``pyproject.toml`` (``[project.scripts] delta = "delta.cli:run"``).

Options use :data:`typing.Annotated` so the Typer metadata lives in the type, not
the default — the modern Typer idiom (no function calls in argument defaults).
"""

from __future__ import annotations

import sys
import time
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import numpy as np

try:
    import typer
    from rich.console import Console
    from rich.table import Table
except ImportError as exc:  # pragma: no cover - exercised via the entry point
    raise SystemExit(
        "the delta CLI needs Typer and Rich; install them with: "
        "uv sync --extra cli  (or  pip install 'delta[cli]')"
    ) from exc

from . import __version__
from ._log import configure_logging, logger
from .pipeline import Subtractor
from .solution import KernelSolution

if TYPE_CHECKING:
    from ._core import FitsLayers

_console = Console(stderr=True)

app = typer.Typer(
    name="delta",
    help="Astronomical difference imaging — a modern Alard & Lupton PSF-matching engine.",
    add_completion=True,
    no_args_is_help=True,
    rich_markup_mode="rich",
    context_settings={"help_option_names": ["-h", "--help"]},
)


class Direction(str, Enum):
    """Which image to convolve onto the other (``auto`` = pick the sharper)."""

    auto = "auto"
    science = "science"
    reference = "reference"


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"delta {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            help="Show the delta version and exit.",
            callback=_version_callback,
            is_eager=True,
        ),
    ] = False,
) -> None:
    """delta — high-performance astronomical difference imaging."""


def _read_layer(path: Path, what: str) -> FitsLayers:
    """Read a FITS file into {data, variance, mask}, failing loudly on error."""
    from . import _core

    if not path.exists():
        _console.print(f"[red]error:[/] {what} file not found: {path}")
        raise typer.Exit(code=2)
    logger.debug("reading {} from {}", what, path)
    try:
        return _core.read_fits(str(path))
    except Exception as exc:  # noqa: BLE001 - surface any CFITSIO failure cleanly
        _console.print(f"[red]error:[/] failed to read {what} ({path}): {exc}")
        raise typer.Exit(code=2) from exc


def _input_cards(
    *,
    science: Path,
    reference: Path,
    science_var: Path | None = None,
    reference_var: Path | None = None,
    catalog: Path | None = None,
    gain: float | None = None,
    read_noise: float = 0.0,
    direction: Direction | None = None,
    solution: Path | None = None,
) -> dict[str, object]:
    """Header cards for the inputs the pipeline itself has no way to know about:
    file paths, the noise-model knobs, and the exact command line invoked."""
    cards: dict[str, object] = {
        "DLTSCI": str(science),
        "DLTREF": str(reference),
        "DLTRN": read_noise,
        "DLTCMD": " ".join(sys.argv),
    }
    if direction is not None:
        cards["DLTDREQ"] = direction.value
    if science_var is not None:
        cards["DLTSVARF"] = str(science_var)
    if reference_var is not None:
        cards["DLTRVARF"] = str(reference_var)
    if catalog is not None:
        cards["DLTCAT"] = str(catalog)
    if gain is not None:
        cards["DLTGAIN"] = gain
    if solution is not None:
        cards["DLTSOLF"] = str(solution)
    return cards


def _load_catalog(path: Path) -> np.ndarray:
    """Load an (N, 2) integer x/y stamp catalog from a whitespace text file."""
    cat = np.loadtxt(path, ndmin=2)
    if cat.ndim != 2 or cat.shape[1] < 2:
        raise typer.BadParameter(f"catalog {path} must have at least two columns (x y)")
    return np.ascontiguousarray(cat[:, :2], dtype=np.int32)


def _summary_table(result, elapsed: float) -> Table:
    """Build a Rich table summarising the fit + difference products."""
    sol = result.solution
    table = Table(title="delta subtraction summary", title_style="bold cyan", show_header=False)
    table.add_column("key", style="cyan", no_wrap=True)
    table.add_column("value")
    h, w = sol.shape
    table.add_row("frame", f"{w} x {h}")
    table.add_row("convolved", sol.direction)
    table.add_row("beta / n_max", f"{sol.beta:.3f} / {sol.n_max}")
    table.add_row("lambda (GCV)", f"{sol.lam:.3g}  (GCV {sol.gcv:.4g})")
    table.add_row("effective dof", f"{sol.effective_dof:.1f}")
    table.add_row("RSS", f"{sol.rss:.4g}")
    table.add_row("difference rms", f"{float(np.std(result.difference)):.4g}")
    table.add_row("variance", "yes" if result.variance is not None else "no")
    if result.variance_scale is not None:
        table.add_row("variance rescale", f"{result.variance_scale:.4f} (chi2 -> 1)")
    table.add_row("score", "yes" if result.score is not None else "no")
    table.add_row("wall time", f"{elapsed:.3f}s")
    return table


@app.command()
def subtract(
    science: Annotated[Path, typer.Argument(help="Science image FITS path.")],
    reference: Annotated[Path, typer.Argument(help="Reference (template) image FITS path.")],
    output: Annotated[
        Path,
        typer.Option("-o", "--output", help="Output multi-extension FITS path for the difference."),
    ],
    # -- noise model -------------------------------------------------------
    science_var: Annotated[
        Path | None, typer.Option("--science-var", help="FITS variance map for science.")
    ] = None,
    reference_var: Annotated[
        Path | None, typer.Option("--reference-var", help="FITS variance map for reference.")
    ] = None,
    gain: Annotated[
        float | None,
        typer.Option("--gain", help="Detector gain (e-/ADU); synthesises variance if no map."),
    ] = None,
    read_noise: Annotated[
        float, typer.Option("--read-noise", help="Read noise (ADU) for synthesised variance.")
    ] = 0.0,
    # -- kernel / spatial model -------------------------------------------
    n_max: Annotated[int, typer.Option("--n-max", min=0, help="Gauss-Hermite basis order.")] = 6,
    beta: Annotated[
        float | None,
        typer.Option("--beta", help="Kernel basis scale (px). Auto from FWHMs if omitted."),
    ] = None,
    n_knots: Annotated[
        int, typer.Option("--n-knots", min=2, help="Thin-plate knots per axis.")
    ] = 5,
    stamp_radius: Annotated[
        int, typer.Option("--stamp-radius", min=1, help="Stamp half-size (px).")
    ] = 15,
    threshold_sigma: Annotated[
        float, typer.Option("--threshold-sigma", help="Detection threshold for stamp selection.")
    ] = 5.0,
    max_stamps: Annotated[
        int, typer.Option("--max-stamps", min=1, help="Cap on stamps used.")
    ] = 200,
    saturation: Annotated[
        float, typer.Option("--saturation", help="Saturation level; reject stamps above (0=off).")
    ] = 0.0,
    catalog: Annotated[
        Path | None,
        typer.Option("--catalog", help="Text file of 'x y' stamp centres (overrides detection)."),
    ] = None,
    direction: Annotated[
        Direction,
        typer.Option("--direction", help="Which image to convolve (auto = the sharper one)."),
    ] = Direction.auto,
    # -- post-processing ---------------------------------------------------
    decorrelate: Annotated[
        bool, typer.Option("--decorrelate/--no-decorrelate", help="ZOGY-style noise decorrelation.")
    ] = False,
    score: Annotated[
        bool, typer.Option("--score/--no-score", help="Emit a matched-filter S/N score image.")
    ] = False,
    block: Annotated[
        int, typer.Option("--block", min=32, help="FFT block size for decorrelation.")
    ] = 256,
    rescale_variance: Annotated[
        bool,
        typer.Option(
            "--rescale-variance/--no-rescale-variance",
            help="Rescale variance by a scalar to force the diff-image reduced chi2 to 1.",
        ),
    ] = False,
    # -- output / feedback -------------------------------------------------
    save_solution: Annotated[
        Path | None,
        typer.Option("--save-solution", help="Also write the kernel solution to a .npz file."),
    ] = None,
    overwrite: Annotated[
        bool, typer.Option("--overwrite", help="Overwrite the output if it exists.")
    ] = False,
    compress: Annotated[
        bool,
        typer.Option(
            "--compress/--no-compress",
            help="Tile-compress (RICE) the output layers; difference moves to an extension.",
        ),
    ] = True,
    verbose: Annotated[
        int, typer.Option("-v", "--verbose", count=True, help="Increase log verbosity (-v=debug).")
    ] = 0,
    quiet: Annotated[bool, typer.Option("-q", "--quiet", help="Only warnings and errors.")] = False,
) -> None:
    """Difference two FITS images: fit the matching kernel, subtract, and write products."""
    configure_logging(-1 if quiet else verbose)

    if output.exists() and not overwrite:
        _console.print(f"[red]error:[/] output exists (use --overwrite): {output}")
        raise typer.Exit(code=2)

    sci = _read_layer(science, "science")
    ref = _read_layer(reference, "reference")
    svar = _read_layer(science_var, "science variance")["data"] if science_var else sci["variance"]
    rvar = (
        _read_layer(reference_var, "reference variance")["data"]
        if reference_var
        else ref["variance"]
    )
    cat = _load_catalog(catalog) if catalog else None

    sub = Subtractor(
        n_max=n_max,
        beta=beta,
        n_knots=n_knots,
        stamp_radius=stamp_radius,
        threshold_sigma=threshold_sigma,
        max_stamps=max_stamps,
        saturation=saturation,
        decorrelate=decorrelate,
        score=score,
        block=block,
        rescale_variance=rescale_variance,
    )

    start = time.perf_counter()
    try:
        result = sub.subtract(
            sci["data"],
            ref["data"],
            science_var=svar,
            reference_var=rvar,
            science_mask=sci["mask"],
            reference_mask=ref["mask"],
            gain=gain,
            read_noise=read_noise,
            catalog=cat,
            direction=None if direction is Direction.auto else direction.value,
        )
    except ValueError as exc:
        _console.print(f"[red]error:[/] {exc}")
        raise typer.Exit(code=1) from exc
    elapsed = time.perf_counter() - start

    cards = _input_cards(
        science=science,
        reference=reference,
        science_var=science_var,
        reference_var=reference_var,
        catalog=catalog,
        gain=gain,
        read_noise=read_noise,
        direction=direction,
    )
    _write_output(result, output, overwrite, compress=compress, extra_cards=cards)
    if save_solution is not None:
        result.solution.save(str(save_solution))
        logger.info("wrote solution to {}", save_solution)

    if not quiet:
        _console.print(_summary_table(result, elapsed))
    _console.print(f"[green]wrote[/] {output}")


def _write_output(
    result,
    output: Path,
    overwrite: bool,
    *,
    compress: bool = True,
    extra_cards: dict[str, object] | None = None,
) -> None:
    """Write the difference products, preferring astropy (provenance + score)."""
    try:
        result.write(str(output), overwrite=overwrite, compress=compress, extra_cards=extra_cards)
        logger.debug("wrote multi-extension FITS via astropy (compress={})", compress)
    except ImportError:
        from . import _core

        if result.score is not None:
            logger.warning("astropy not installed; score image and provenance header dropped")
        _core.write_fits(str(output), result.difference, result.variance, result.mask)


@app.command()
def apply(
    solution: Annotated[
        Path, typer.Argument(help="Saved kernel solution (.npz) from a previous fit.")
    ],
    science: Annotated[Path, typer.Argument(help="Science image FITS path.")],
    reference: Annotated[Path, typer.Argument(help="Reference (template) image FITS path.")],
    output: Annotated[
        Path,
        typer.Option("-o", "--output", help="Output multi-extension FITS path for the difference."),
    ],
    # -- noise model -------------------------------------------------------
    science_var: Annotated[
        Path | None, typer.Option("--science-var", help="FITS variance map for science.")
    ] = None,
    reference_var: Annotated[
        Path | None, typer.Option("--reference-var", help="FITS variance map for reference.")
    ] = None,
    gain: Annotated[
        float | None,
        typer.Option("--gain", help="Detector gain (e-/ADU); synthesises variance if no map."),
    ] = None,
    read_noise: Annotated[
        float, typer.Option("--read-noise", help="Read noise (ADU) for synthesised variance.")
    ] = 0.0,
    # -- output model (kernel/spatial knobs come from the solution) --------
    saturation: Annotated[
        float, typer.Option("--saturation", help="Output saturation mask level (0=off).")
    ] = 0.0,
    stamp_radius: Annotated[
        int, typer.Option("--stamp-radius", min=1, help="Stamp half-size (px) for the score PSF.")
    ] = 15,
    # -- post-processing ---------------------------------------------------
    decorrelate: Annotated[
        bool, typer.Option("--decorrelate/--no-decorrelate", help="ZOGY-style noise decorrelation.")
    ] = False,
    score: Annotated[
        bool, typer.Option("--score/--no-score", help="Emit a matched-filter S/N score image.")
    ] = False,
    block: Annotated[
        int, typer.Option("--block", min=32, help="FFT block size for decorrelation.")
    ] = 256,
    rescale_variance: Annotated[
        bool,
        typer.Option(
            "--rescale-variance/--no-rescale-variance",
            help="Rescale variance by a scalar to force the diff-image reduced chi2 to 1.",
        ),
    ] = False,
    # -- output / feedback -------------------------------------------------
    overwrite: Annotated[
        bool, typer.Option("--overwrite", help="Overwrite the output if it exists.")
    ] = False,
    compress: Annotated[
        bool,
        typer.Option(
            "--compress/--no-compress",
            help="Tile-compress (RICE) the output layers; difference moves to an extension.",
        ),
    ] = True,
    verbose: Annotated[
        int, typer.Option("-v", "--verbose", count=True, help="Increase log verbosity (-v=debug).")
    ] = 0,
    quiet: Annotated[bool, typer.Option("-q", "--quiet", help="Only warnings and errors.")] = False,
) -> None:
    """Apply a saved kernel solution to a new pair and subtract — no re-fitting."""
    configure_logging(-1 if quiet else verbose)

    if output.exists() and not overwrite:
        _console.print(f"[red]error:[/] output exists (use --overwrite): {output}")
        raise typer.Exit(code=2)
    if not solution.exists():
        _console.print(f"[red]error:[/] solution file not found: {solution}")
        raise typer.Exit(code=2)
    try:
        sol = KernelSolution.load(str(solution))
    except Exception as exc:  # noqa: BLE001 - surface any load failure cleanly
        _console.print(f"[red]error:[/] failed to load solution ({solution}): {exc}")
        raise typer.Exit(code=2) from exc

    sci = _read_layer(science, "science")
    ref = _read_layer(reference, "reference")
    svar = _read_layer(science_var, "science variance")["data"] if science_var else sci["variance"]
    rvar = (
        _read_layer(reference_var, "reference variance")["data"]
        if reference_var
        else ref["variance"]
    )

    sub = Subtractor(
        stamp_radius=stamp_radius,
        saturation=saturation,
        decorrelate=decorrelate,
        score=score,
        block=block,
        rescale_variance=rescale_variance,
    )

    start = time.perf_counter()
    try:
        result = sub.apply(
            sol,
            sci["data"],
            ref["data"],
            science_var=svar,
            reference_var=rvar,
            science_mask=sci["mask"],
            reference_mask=ref["mask"],
            gain=gain,
            read_noise=read_noise,
        )
    except ValueError as exc:
        _console.print(f"[red]error:[/] {exc}")
        raise typer.Exit(code=1) from exc
    elapsed = time.perf_counter() - start

    cards = _input_cards(
        science=science,
        reference=reference,
        science_var=science_var,
        reference_var=reference_var,
        gain=gain,
        read_noise=read_noise,
        solution=solution,
    )
    _write_output(result, output, overwrite, compress=compress, extra_cards=cards)

    if not quiet:
        _console.print(_summary_table(result, elapsed))
    _console.print(f"[green]wrote[/] {output}")


@app.command()
def info(
    path: Annotated[Path, typer.Argument(help="FITS file to inspect.")],
    verbose: Annotated[
        int, typer.Option("-v", "--verbose", count=True, help="Increase verbosity.")
    ] = 0,
) -> None:
    """Report the data/variance/mask layers and basic statistics of a FITS file."""
    configure_logging(verbose)
    layers = _read_layer(path, "image")
    data = layers["data"]

    table = Table(title=str(path), title_style="bold cyan", show_header=True)
    table.add_column("layer", style="cyan")
    table.add_column("shape")
    table.add_column("dtype")
    table.add_column("min / median / max", justify="right")
    for name in ("data", "variance", "mask"):
        arr = layers.get(name)
        if arr is None:
            table.add_row(name, "-", "-", "absent")
            continue
        a = np.asarray(arr)
        finite = a[np.isfinite(a)]
        stats = (
            f"{finite.min():.4g} / {np.median(finite):.4g} / {finite.max():.4g}"
            if finite.size
            else "n/a"
        )
        table.add_row(name, " x ".join(map(str, a.shape)), str(a.dtype), stats)
    _console.print(table)
    mask = layers.get("mask")
    if mask is not None:
        n_bad = int(np.count_nonzero(mask))
        _console.print(f"masked pixels: {n_bad} ({100.0 * n_bad / data.size:.2f}%)")


def run() -> None:
    """Entry point wrapper (keeps tracebacks tidy on Ctrl-C)."""
    try:
        app()
    except KeyboardInterrupt:  # pragma: no cover
        _console.print("[yellow]interrupted[/]")
        sys.exit(130)


if __name__ == "__main__":  # pragma: no cover
    run()
