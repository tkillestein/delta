"""The fitted kernel/background solution — serializable for QA and reuse."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from . import _core


@dataclass
class KernelSolution:
    """A spatially-varying matching-kernel + differential-background solution.

    Holds the Gauss-Hermite basis parameters, the thin-plate knots, the fitted
    coefficient vector ``theta`` (ordered [c_0 | ... | c_{nc-1} | b]) and GCV
    diagnostics. ``direction`` is which image was convolved ("reference" or
    "science"); ``shape`` is the full-frame (height, width).
    """

    beta: float
    n_max: int
    radius: int
    knots: NDArray[np.float64]
    theta: NDArray[np.float64]
    n_components: int
    n_spatial: int
    lam: float
    gcv: float
    effective_dof: float
    rss: float
    component_sums: NDArray[np.float64]
    direction: str
    shape: tuple[int, int]
    # Goodness-of-fit / stamp-clipping diagnostics (SPEC §3.3, §13).
    reduced_chi2: float = float("nan")
    n_stamps_used: int = 0
    n_stamps_rejected: int = 0
    stamp_x: NDArray[np.int32] | None = None
    stamp_y: NDArray[np.int32] | None = None
    stamp_chi2: NDArray[np.float64] | None = None
    stamp_accepted: NDArray[np.uint8] | None = None

    def photometric_scale(self) -> NDArray[np.float32]:
        """Per-pixel photometric scale field sum_n a_n(x,y) S_n, shape (H, W)."""
        h, w = self.shape
        return _core.photometric_scale(self.knots, self.theta, self.component_sums, h, w)

    def photometric_scale_at(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        """Photometric scale at (m, 2) pixel coordinates."""
        return _core.photometric_scale_at(
            self.knots, self.theta, self.component_sums, np.asarray(points, np.float64)
        )

    def save(self, path: str) -> None:
        """Serialize to a .npz file. Optional per-stamp diagnostics are stored as
        empty arrays when absent."""
        empty_i = np.empty(0, np.int32)

        def _arr(a: NDArray | None, like: NDArray) -> NDArray:
            return like if a is None else a

        np.savez(
            path,
            beta=self.beta,
            n_max=self.n_max,
            radius=self.radius,
            knots=self.knots,
            theta=self.theta,
            n_components=self.n_components,
            n_spatial=self.n_spatial,
            lam=self.lam,
            gcv=self.gcv,
            effective_dof=self.effective_dof,
            rss=self.rss,
            component_sums=self.component_sums,
            direction=self.direction,
            shape=np.asarray(self.shape),
            reduced_chi2=self.reduced_chi2,
            n_stamps_used=self.n_stamps_used,
            n_stamps_rejected=self.n_stamps_rejected,
            stamp_x=_arr(self.stamp_x, empty_i),
            stamp_y=_arr(self.stamp_y, empty_i),
            stamp_chi2=_arr(self.stamp_chi2, np.empty(0, np.float64)),
            stamp_accepted=_arr(self.stamp_accepted, np.empty(0, np.uint8)),
        )

    @classmethod
    def load(cls, path: str) -> KernelSolution:
        """Load a solution saved with :meth:`save`."""
        d = np.load(path, allow_pickle=False)
        return cls(
            beta=float(d["beta"]),
            n_max=int(d["n_max"]),
            radius=int(d["radius"]),
            knots=d["knots"],
            theta=d["theta"],
            n_components=int(d["n_components"]),
            n_spatial=int(d["n_spatial"]),
            lam=float(d["lam"]),
            gcv=float(d["gcv"]),
            effective_dof=float(d["effective_dof"]),
            rss=float(d["rss"]),
            component_sums=d["component_sums"],
            direction=str(d["direction"]),
            shape=(int(d["shape"][0]), int(d["shape"][1])),
            reduced_chi2=float(d["reduced_chi2"]) if "reduced_chi2" in d else float("nan"),
            n_stamps_used=int(d["n_stamps_used"]) if "n_stamps_used" in d else 0,
            n_stamps_rejected=int(d["n_stamps_rejected"]) if "n_stamps_rejected" in d else 0,
            stamp_x=d.get("stamp_x"),
            stamp_y=d.get("stamp_y"),
            stamp_chi2=d.get("stamp_chi2"),
            stamp_accepted=d.get("stamp_accepted"),
        )

    def header_cards(self) -> dict[str, object]:
        """Provenance key/value pairs for FITS headers (SPEC §8)."""
        return {
            "DLTBETA": self.beta,
            "DLTNMAX": self.n_max,
            "DLTNKNOT": int(self.knots.shape[0]),
            "DLTLAM": self.lam,
            "DLTGCV": self.gcv,
            "DLTEDOF": self.effective_dof,
            "DLTCHI2": self.reduced_chi2,
            "DLTNSTU": self.n_stamps_used,
            "DLTNSTR": self.n_stamps_rejected,
            "DLTCONV": self.direction,
        }
