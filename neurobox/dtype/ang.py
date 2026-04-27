"""
ang.py  —  NBDang
==================
Pairwise inter-marker spherical coordinates.

Port of ``MTADang``.

Data layout: ``(T, N_markers, N_markers, 3)`` — time × marker_i × marker_j ×
spherical component.

The spherical component axis is:
  ``[0]``  azimuth θ  (radians, ``cart2sph`` convention: atan2(y, x))
  ``[1]``  elevation φ  (radians, above/below horizontal plane)
  ``[2]``  radius r   (mm, Euclidean distance)

Only the off-diagonal entries are meaningful: ``ang[:, i, j, :]`` gives the
spherical coordinates of marker *j* **in the frame of marker *i*** (i.e. from
*i* looking toward *j*).  Diagonal entries ``ang[:, i, i, :]`` are set to
``nan``.

Design differences from MTADang
---------------------------------
* MATLAB's ``cart2sph`` returns ``(azimuth, elevation, r)`` (longitude-first);
  we keep the same ``[theta, phi, r]`` order for backward compatibility with
  analysis code that indexes the 4th dimension.
* Uses vectorised numpy broadcasting instead of a double for-loop.
* Lazy: ``data`` is ``None`` until :meth:`create` is called.

Usage
-----
::

    ang = NBDang.from_xyz(xyz)

    # Head azimuth (yaw) relative to spine_lower → head_back
    theta = ang[:, spine_lower_idx, head_back_idx, 0]

    # All elevations relative to head_back
    phi_all = ang[:, head_back_idx, :, 1]
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from neurobox.dtype.data  import NBData
from neurobox.dtype.epoch import NBEpoch
from neurobox.dtype.model import NBModel


class NBDang(NBData):
    """Pairwise inter-marker spherical coordinate time-series.

    Parameters
    ----------
    data:
        Array of shape ``(T, N, N, 3)``.  Pass *None* for lazy
        construction via :meth:`create` / :meth:`from_xyz`.
    model:
        :class:`~neurobox.dtype.model.NBModel` with the marker names.
    samplerate:
        Frame rate of the underlying xyz data (Hz).
    sync, origin:
        Recording window (forwarded to :class:`NBData`).
    """

    def __init__(
        self,
        data: np.ndarray | None = None,
        model: NBModel | None = None,
        samplerate: float = 120.0,
        sync: NBEpoch | None = None,
        origin: float = 0.0,
        path: Path | str | None = None,
        filename: str | None = None,
        name: str = "",
    ) -> None:
        super().__init__(
            path       = path,
            filename   = filename,
            data       = data,
            samplerate = samplerate,
            sync       = sync,
            origin     = origin,
            type_      = "TimeSeries",
            ext        = "ang",
            name       = name,
            label      = "angles",
            key        = "a",
        )
        self.model: NBModel | None = model

    # ------------------------------------------------------------------ #
    # Abstract interface                                                   #
    # ------------------------------------------------------------------ #

    def load(self, *args, **kwargs) -> "NBDang":          # type: ignore[override]
        """Alias for :meth:`create`."""
        return self.create(*args, **kwargs)

    def create(self, xyz: "NBDxyz", **_) -> "NBDang":     # type: ignore[override]
        """Compute pairwise spherical coordinates from an :class:`NBDxyz`.

        Parameters
        ----------
        xyz:
            Loaded position object.  ``xyz.data`` must be ``(T, N, 3)``.

        Returns
        -------
        self, with ``data`` set to ``(T, N, N, 3)`` and ``model`` copied
        from *xyz*.
        """
        if xyz.data is None:
            raise RuntimeError("xyz data is not loaded.")

        d = xyz.data.astype(np.float64)   # (T, N, 3)
        T, N, _ = d.shape

        # Vectorised diff-matrix: diff[t, i, j, dim] = pos[t, j, dim] - pos[t, i, dim]
        # Shape: (T, N, N, 3)
        diff = d[:, np.newaxis, :, :] - d[:, :, np.newaxis, :]   # (T, N, N, 3)
        # diff[:, i, j, :] = pos_j - pos_i  →  vector from i to j

        dx = diff[..., 0]
        dy = diff[..., 1]
        dz = diff[..., 2]

        # Spherical coordinates (MATLAB cart2sph convention)
        theta = np.arctan2(dy, dx)                       # azimuth
        r_xy  = np.sqrt(dx**2 + dy**2)
        phi   = np.arctan2(dz, r_xy)                     # elevation
        r     = np.sqrt(dx**2 + dy**2 + dz**2)           # distance (mm)

        ang = np.stack([theta, phi, r], axis=-1)          # (T, N, N, 3)

        # Diagonal (marker vs itself) → nan
        diag = np.arange(N)
        ang[:, diag, diag, :] = np.nan

        self._data     = ang.astype(np.float32)
        self.model     = xyz.model
        self.samplerate = xyz.samplerate
        self.sync      = xyz.sync
        self.origin    = xyz.origin
        return self

    # ------------------------------------------------------------------ #
    # Convenience constructors                                             #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_xyz(cls, xyz: "NBDxyz", **kwargs) -> "NBDang":
        """Build and return an :class:`NBDang` from *xyz* in one call."""
        obj = cls(**kwargs)
        return obj.create(xyz)

    # ------------------------------------------------------------------ #
    # Named-pair access                                                    #
    # ------------------------------------------------------------------ #

    def between(
        self,
        marker_i: "str | int",
        marker_j: "str | int",
        component: "str | int" = "theta",
    ) -> np.ndarray:
        """Return one spherical component between two named markers.

        Parameters
        ----------
        marker_i, marker_j:
            Marker names (string) or 0-based integer indices.
        component:
            ``'theta'`` / 0 — azimuth (yaw, radians)
            ``'phi'``   / 1 — elevation (pitch, radians)
            ``'r'``     / 2 — Euclidean distance (mm)

        Returns
        -------
        np.ndarray, shape ``(T,)``
        """
        if self._data is None:
            raise RuntimeError("Angular data not computed yet.")

        _COMP = {"theta": 0, "azimuth": 0, "phi": 1, "elevation": 1, "r": 2, "dist": 2}
        ci = _COMP.get(component, component) if isinstance(component, str) else int(component)

        def _idx(m):
            return self.model.index(m) if isinstance(m, str) else int(m)

        return self._data[:, _idx(marker_i), _idx(marker_j), ci]

    def head_direction(
        self,
        from_marker: str = "head_back",
        to_marker:   str = "head_front",
    ) -> np.ndarray:
        """Return the horizontal head-direction angle (azimuth, radians).

        Mirrors the most common use-case in MTA place-field analyses.

        Parameters
        ----------
        from_marker:
            The reference marker (origin of the local frame).
        to_marker:
            The target marker pointing in the head direction.

        Returns
        -------
        np.ndarray, shape ``(T,)``
            Azimuth θ in ``[-π, π]``.
        """
        return self.between(from_marker, to_marker, component="theta")

    # ------------------------------------------------------------------ #
    # Repr                                                                 #
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        shape = self.shape if self._data is not None else "not computed"
        n = len(self.model) if self.model is not None else "?"
        return (
            f"NBDang(shape={shape}, n_markers={n}, "
            f"sr={self.samplerate}Hz)"
        )
