"""HOTPANTS head-to-head test (SPEC §13.4).

Skipped when the `hotpants` binary is not on PATH. When present, it asserts that
delta's residual RMS in a source-free region is competitive with HOTPANTS.
"""

import delta
import numpy as np
import pytest
from delta import validation

from benchmarks import compare_hotpants


@pytest.mark.skipif(not compare_hotpants.hotpants_available(), reason="hotpants binary not on PATH")
def test_delta_competitive_with_hotpants():
    h = w = 256
    rng = np.random.default_rng(0)
    grid = [(x, y) for x in (50, 110, 170, 220) for y in (50, 110, 170, 220)]
    ref = np.full((h, w), 120.0)
    sci = np.full((h, w), 120.0)
    for x, y in grid:
        ref += validation.gaussian_psf((h, w), x, y, 12000.0, 1.6)
        sci += validation.gaussian_psf((h, w), x, y, 12000.0, 2.4)
    var = np.full((h, w), 9.0, np.float32)
    ref_n = (ref + rng.normal(0, 3.0, (h, w))).astype(np.float32)
    sci_n = (sci + rng.normal(0, 3.0, (h, w))).astype(np.float32)

    region = (slice(75, 105), slice(75, 105))
    res = delta.subtract(
        sci_n, ref_n, science_var=var, reference_var=var, n_knots=4, stamp_radius=12
    )
    delta_rms = compare_hotpants.residual_rms(res.difference, region)
    hp_rms = compare_hotpants.residual_rms(compare_hotpants.run_hotpants(sci_n, ref_n), region)
    # Within 50% of HOTPANTS' residual (typically better).
    assert delta_rms < 1.5 * hp_rms
