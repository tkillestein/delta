# Command line

`delta` ships a standalone command-line interface (a [Typer](https://typer.tiangolo.com)
app with [Rich](https://rich.readthedocs.io) output). It reads FITS inputs, runs the
difference-imaging pipeline, and writes a multi-extension FITS product with provenance
header cards.

Install the `cli` extra to pull in Typer and Rich:

```sh
uv sync --extra cli            # or: pip install 'delta[cli]'
```

The console-script entry point is `delta`. Progress and per-stage timings are logged to
stderr via [loguru](https://github.com/Delgan/loguru); control verbosity with `-v`
(debug) / `-q` (warnings only). Run `delta --help` for the top-level help and
`delta <command> --help` for a full option list.

```sh
delta --version          # print the version and exit
delta --help             # top-level help
```

## `delta subtract`

Difference two FITS images: fit the matching kernel, subtract, and write products.

```sh
delta subtract SCIENCE REFERENCE -o OUTPUT [OPTIONS]
```

```sh
delta subtract science.fits reference.fits -o diff.fits \
    --gain 1.5 --read-noise 4 --decorrelate --score -v
```

The output is a multi-extension FITS (difference + variance/mask, plus score when
`--score`). With `--compress` (the default) the layers are RICE tile-compressed and the
difference is moved to an extension.

**Arguments**

| argument | meaning |
|---|---|
| `SCIENCE` | science image FITS path |
| `REFERENCE` | reference (template) image FITS path |

**Options**

*Noise model*

| option | meaning | default |
|---|---|---|
| `--science-var PATH` | FITS variance map for science | — |
| `--reference-var PATH` | FITS variance map for reference | — |
| `--gain FLOAT` | detector gain (e⁻/ADU); synthesises variance if no map | — |
| `--read-noise FLOAT` | read noise (ADU) for synthesised variance | `0.0` |

*Kernel / spatial model*

| option | meaning | default |
|---|---|---|
| `--n-max INT` | Gauss-Hermite basis order | `6` |
| `--beta FLOAT` | kernel basis scale (px); auto from FWHMs if omitted | — |
| `--n-knots INT` | thin-plate knots per axis | `5` |
| `--stamp-radius INT` | stamp half-size (px) | `15` |
| `--threshold-sigma FLOAT` | detection threshold for stamp selection | `5.0` |
| `--max-stamps INT` | cap on stamps used | `200` |
| `--saturation FLOAT` | saturation level; reject stamps above (`0` = off) | `0.0` |
| `--catalog PATH` | text file of `x y` stamp centres (overrides detection) | — |
| `--direction [auto\|science\|reference]` | which image to convolve (`auto` = the sharper) | `auto` |

*Post-processing*

| option | meaning | default |
|---|---|---|
| `--decorrelate / --no-decorrelate` | ZOGY-style noise decorrelation | off |
| `--score / --no-score` | emit a matched-filter S/N score image | off |
| `--block INT` | FFT block size for decorrelation | `256` |

*Output / feedback*

| option | meaning | default |
|---|---|---|
| `-o, --output PATH` | output multi-extension FITS path (required) | — |
| `--save-solution PATH` | also write the kernel solution to a `.npz` file | — |
| `--overwrite` | overwrite the output if it exists | off |
| `--compress / --no-compress` | RICE tile-compress the output layers | on |
| `-v, --verbose` | increase log verbosity (`-v` = debug) | `0` |
| `-q, --quiet` | only warnings and errors | off |

!!! note
    The CLI exposes the common knobs; advanced `Subtractor` options
    (`bright_mask`, `lambda_grid`, `clip_sigma`, `clip_iterations`, `min_stamps`,
    `cv_folds`, `spatial_scale`, `radius`) are available only through the
    [Python API][delta.Subtractor].

## `delta apply`

Apply a saved kernel solution (`.npz`, from `delta subtract --save-solution` or
[`KernelSolution.save`][delta.KernelSolution]) to a new science/reference pair and
subtract — without re-fitting. The convolution direction, basis and coefficients all
come from the solution; the inputs must match its frame shape.

```sh
delta apply SOLUTION SCIENCE REFERENCE -o OUTPUT [OPTIONS]
```

```sh
delta subtract sci1.fits ref.fits -o diff1.fits --save-solution kernel.npz --gain 1.5
delta apply kernel.npz sci2.fits ref.fits -o diff2.fits --gain 1.5
```

**Arguments**

| argument | meaning |
|---|---|
| `SOLUTION` | saved kernel solution (`.npz`) from a previous fit |
| `SCIENCE` | science image FITS path |
| `REFERENCE` | reference (template) image FITS path |

**Options**

The noise-model (`--science-var`, `--reference-var`, `--gain`, `--read-noise`),
post-processing (`--decorrelate`, `--score`, `--block`), and output
(`-o/--output`, `--overwrite`, `--compress`, `-v`, `-q`) options match
[`delta subtract`](#delta-subtract). The kernel/spatial model is fixed by the
solution, so only `--saturation` (output mask level) and `--stamp-radius` (score
PSF half-size) remain from that group.

## Provenance

Every FITS product written by `delta subtract` / `delta apply` (and
[`DiffResult.write`][delta.DiffResult.write] in the Python API) carries a full
provenance header — see [Provenance headers](usage.md#provenance-headers) for the
complete card reference.

## `delta info`

Report the data/variance/mask layers and basic statistics of a FITS file.

```sh
delta info PATH [-v]
```

```sh
delta info science.fits        # inspect data/variance/mask layers
```

Prints a table of each layer's shape, dtype, and min/median/max, plus the masked-pixel
fraction when a mask is present.
