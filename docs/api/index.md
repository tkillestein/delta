# API reference

The public API is re-exported from the top-level `delta` package. There are two layers.

## High-level pipeline

The headline path most users want — coerce inputs, fit a spatially-varying kernel,
subtract, and (optionally) decorrelate and score:

```
delta.subtract(...)  ──▶  DiffResult  ──▶  KernelSolution
        │                     │
   Subtractor            .write("diff.fits")
   (reusable config)     .difference / .variance / .mask / .score
```

- **[Pipeline](pipeline.md)** — [`subtract`][delta.subtract] (one-shot wrapper),
  [`Subtractor`][delta.Subtractor] (reusable configuration), and
  [`DiffResult`][delta.DiffResult] (the products of a run).
- **[Kernel solution](solution.md)** — [`KernelSolution`][delta.KernelSolution], the
  serializable fit result (basis params, knots, coefficients, diagnostics) with
  `.save` / `.load` and FITS provenance.
- **[Inputs](inputs.md)** — [`as_layers`][delta.as_layers] and
  [`synth_variance`][delta.synth_variance] for input coercion and noise synthesis.
- **[Validation](validation.md)** — synthetic source injection/recovery helpers.

## Low-level core

The C++ core (`delta._core`) is also re-exported for advanced use — basis evaluation,
convolution, the GLS solver, thin-plate splines, detection, decorrelation, and FITS
I/O. These are the building blocks the high-level pipeline orchestrates; most users
will not need them directly. See **[Core (low-level)](core.md)**.
