"""Input coercion: turn ndarrays / astropy objects into (data, variance, mask).

astropy is an optional dependency; nothing here imports it at module load. We
duck-type ``CCDData`` / FITS HDUs by their ``.data`` / ``.uncertainty`` / ``.mask``
attributes so the pipeline works with or without astropy installed.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Layers = tuple[NDArray[np.float32], NDArray[np.float32] | None, NDArray[np.uint8] | None]


def _split(obj) -> Layers:
    """Pull (data, variance, mask) out of an ndarray or astropy-like object."""
    data = getattr(obj, "data", None)
    if data is None:
        # Plain array-like.
        return np.asarray(obj), None, None

    variance = None
    unc = getattr(obj, "uncertainty", None)
    if unc is not None and getattr(unc, "array", None) is not None:
        arr = np.asarray(unc.array, dtype=np.float64)
        # StdDevUncertainty stores sigma; VarianceUncertainty stores variance.
        utype = getattr(unc, "uncertainty_type", "std")
        variance = arr**2 if utype == "std" else arr

    mask = getattr(obj, "mask", None)
    if mask is np.ma.nomask:
        # np.ma.MaskedArray with no masked values reports `mask` as the
        # scalar `nomask` sentinel rather than a per-pixel array; treat it
        # the same as "no mask supplied" instead of letting the scalar flow
        # into the shape check below.
        mask = None
    return np.asarray(data), variance, mask


def as_layers(
    obj,
    variance: NDArray | None = None,
    mask: NDArray | None = None,
) -> Layers:
    """Coerce an image input to contiguous (data float32, variance, mask uint8).

    Explicit ``variance`` / ``mask`` arguments override anything carried by the
    object. A boolean mask is interpreted as True == bad (-> kMaskBad).
    """
    data, v, m = _split(obj)
    if variance is not None:
        v = variance
    if mask is not None:
        m = mask

    data = np.ascontiguousarray(data, dtype=np.float32)
    if data.ndim != 2:
        raise ValueError(f"expected a 2-D image, got shape {data.shape}")

    if v is not None:
        v = np.ascontiguousarray(v, dtype=np.float32)
        if v.shape != data.shape:
            raise ValueError("variance shape does not match data")
    if m is not None:
        m = np.asarray(m)
        if m.dtype == bool:
            m = m.astype(np.uint8)  # True (bad) -> 1 == kMaskBad
        m = np.ascontiguousarray(m, dtype=np.uint8)
        if m.shape != data.shape:
            raise ValueError("mask shape does not match data")
    return data, v, m


def synth_variance(
    data: NDArray[np.float32], gain: float, read_noise: float = 0.0
) -> NDArray[np.float32]:
    """Synthesize a per-pixel variance map from gain and read noise (SPEC §3.6).

    In ADU, Var = max(data, 0) / gain + read_noise**2 (read_noise in ADU): the
    Poisson term from the source/sky plus the read-noise floor.
    """
    if gain <= 0.0:
        raise ValueError("gain must be > 0")
    var = np.clip(data, 0.0, None) / gain + float(read_noise) ** 2
    return np.ascontiguousarray(var, dtype=np.float32)
