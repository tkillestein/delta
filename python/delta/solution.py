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
        """Serialize to a .npz file."""
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
            "DLTCONV": self.direction,
        }
