# Third-party software

`delta` is distributed under the MIT License (see `LICENSE`). It bundles and/or links
against the following third-party components, each under its own license. None is
copyleft; in particular `delta` has **no GPL dependency**.

## Vendored (bundled in this repository)

### PocketFFT — `extern/pocketfft/pocketfft_hdronly.h`

Header-only FFT used by the noise-decorrelation stage (`src/noise.cpp`). BSD-3-Clause.
Upstream: <https://github.com/mreineck/pocketfft> (branch `cpp`), pinned at commit
`862d90670232e3389e4fd29f820948ef7a048c92`. See `extern/pocketfft/README.md` for the
exact provenance (commit, SHA-256) and the update procedure. The full license is
retained at the top of the vendored header and is reproduced here as required for
binary redistribution:

```
Copyright (C) 2010-2024 Max-Planck-Society
Copyright (C) 2019-2020 Peter Bell
(odd-sized DCT-IV: Copyright (C) 2003, 2007-14 Matteo Frigo and the Massachusetts
Institute of Technology; prev_good_size: Copyright (C) 2024 Tan Ping Liang, Peter Bell;
good_size overflow safeguards: Copyright (C) 2024 Cris Luengo)
Authors: Martin Reinecke, Peter Bell
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice, this
  list of conditions and the following disclaimer in the documentation and/or
  other materials provided with the distribution.
* Neither the name of the copyright holder nor the names of its contributors may
  be used to endorse or promote products derived from this software without
  specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
... (see the vendored header for the full disclaimer)
```

## Build / link dependencies (not bundled; installed separately)

| Component | Role | License |
|-----------|------|---------|
| [nanobind](https://github.com/wjakob/nanobind) | C++/Python bindings (`src/bindings.cpp`) | BSD-3-Clause |
| [Eigen](https://eigen.tuxfamily.org) (≥ 3.4) | dense linear algebra for the GLS solve | MPL-2.0 |
| [CFITSIO](https://heasarc.gsfc.nasa.gov/fitsio/) | FITS I/O | permissive (NASA/HEASARC) |
| [NumPy](https://numpy.org) | array interop | BSD-3-Clause |
| [loguru](https://github.com/Delgan/loguru) | logging | MIT |
| [Typer](https://typer.tiangolo.com) / [Rich](https://github.com/Textualize/rich) (CLI extra) | command-line interface | MIT |

## Acknowledgement

`delta` is benchmarked against and inspired by **HOTPANTS** (Becker 2015, ascl:1504.004),
an external tool that is neither bundled nor linked. See the References section of the
README for the scientific lineage.
