# Vendored: PocketFFT

`pocketfft_hdronly.h` is vendored (not a submodule) so it ships in the source
distribution and a fresh `git clone` builds with no extra steps. This is the same
approach NumPy and SciPy take with PocketFFT.

## Provenance

- **Upstream:** <https://github.com/mreineck/pocketfft> (branch `cpp`)
- **File:** `pocketfft_hdronly.h`
- **Pinned commit:** `862d90670232e3389e4fd29f820948ef7a048c92`
- **SHA-256:** `2e88a5b0addff3863a58258cc5bd92469a81c88cc4ec70cfc135e43f0cff8c14`
- **License:** BSD-3-Clause (retained in the header; see `../../THIRD_PARTY.md`)

## Updating

PocketFFT is a single header, so updating is a re-copy + re-pin:

```sh
COMMIT=<new-commit-sha>
curl -sSL -o extern/pocketfft/pocketfft_hdronly.h \
  "https://raw.githubusercontent.com/mreineck/pocketfft/$COMMIT/pocketfft_hdronly.h"
sha256sum extern/pocketfft/pocketfft_hdronly.h    # update this README + THIRD_PARTY.md
uv sync --reinstall-package delta && uv run pytest tests/test_noise.py
```

Only update deliberately (e.g. to pick up an upstream fix); pinning to a reviewed
commit keeps the build reproducible.
