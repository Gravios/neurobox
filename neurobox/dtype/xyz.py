"""
xyz.py  —  NBDxyz
==================
Port of MTADxyz.  3-D position data for motion-captured marker tracking.

Data layout: ``(T, N_markers, 3)``  — time × marker × [x, y, z].

Design differences from MTADxyz
---------------------------------
* Marker names stored as a plain ``list[str]``; no separate MTAModel
  object (model is introduced in a later iteration).
* ``vel``, ``acc``, ``dist``, ``com`` are methods; velocity and
  acceleration are returned as plain ``(T, N_markers)`` numpy arrays
  rather than new MTADxyz objects (simpler for downstream code).
* Lazy loading: ``data`` is loaded on first access if a file path
  was provided but the data wasn't passed at construction time.
* ``sel(markers, dims)`` replaces MATLAB's model-based subsref for
  named-marker indexing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from neurobox.dtype.data import NBData
from neurobox.dtype.epoch import NBEpoch


class NBDxyz(NBData):
    """3-D position time-series for motion-captured markers.

    Parameters
    ----------
    data:
        Array of shape ``(T, N_markers, 3)`` (or ``(T, N_markers, 2)``
        for 2-D tracking).  Pass *None* for lazy loading.
    markers:
        Ordered list of marker name strings.
    samplerate:
        Tracking sample rate in Hz (Vicon ≈ 120 Hz).
    sync:
        NBEpoch defining the valid recording window.
    origin:
        Time in seconds of the first sample relative to the
        electrophysiology recording start.
    path / filename:
        Location of saved data file (for lazy loading).
    """

    def __init__(
        self,
        data: np.ndarray | None = None,
        markers: list[str] | None = None,
        samplerate: float = 120.0,
        sync: NBEpoch | None = None,
        origin: float = 0.0,
        path: Path | str | None = None,
        filename: str | None = None,
        name: str = "",
        label: str = "position",
        key: str = "x",
    ) -> None:
        super().__init__(
            path=path, filename=filename, data=data,
            samplerate=samplerate, sync=sync, origin=origin,
            type_="TimeSeries", ext="pos", name=name, label=label, key=key,
        )
        self.markers: list[str] = markers or []

    # ------------------------------------------------------------------ #
    # Abstract interface                                                   #
    # ------------------------------------------------------------------ #

    def load(self, *args, **kwargs) -> "NBDxyz":
        """Load from a saved .npy / .pkl file (to be implemented)."""
        raise NotImplementedError("NBDxyz.load() not yet implemented.")

    def create(self, *args, **kwargs) -> "NBDxyz":
        """Create position data from a Motive CSV or C3D file."""
        raise NotImplementedError("NBDxyz.create() not yet implemented.")

    # ------------------------------------------------------------------ #
    # Named-marker selection                                               #
    # ------------------------------------------------------------------ #

    def sel(self, markers=None, dims=None) -> np.ndarray:
        """Select markers and/or spatial dimensions.

        Parameters
        ----------
        markers:
            ``None`` (all), list of names, list of int indices, or a
            single string.
        dims:
            ``None`` (all), or list/slice of dimension indices (0=x, 1=y, 2=z).

        Returns
        -------
        np.ndarray, shape ``(T, n_markers_sel, n_dims_sel)``
        """
        d = self.data
        if d is None:
            raise RuntimeError("Data not loaded.")

        if markers is None:
            mi = slice(None)
        elif isinstance(markers, str):
            mi = [self._marker_index(markers)]
        elif isinstance(markers, (list, tuple)):
            mi = [self._marker_index(m) if isinstance(m, str) else int(m)
                  for m in markers]
        else:
            mi = markers

        if dims is None:
            di = slice(None)
        else:
            di = dims

        return d[:, mi, :][:, :, di] if not isinstance(di, slice) else d[:, mi, di]

    def _marker_index(self, name: str) -> int:
        try:
            return self.markers.index(name)
        except ValueError:
            raise KeyError(f"Marker {name!r} not found in {self.markers}") from None

    # ------------------------------------------------------------------ #
    # Kinematics                                                           #
    # ------------------------------------------------------------------ #

    def vel(self, markers=None, dims=None) -> np.ndarray:
        """Instantaneous speed in cm/s for selected markers.

        Returns
        -------
        v : np.ndarray, shape ``(T, n_markers)``
            Speed (norm of velocity vector) at each frame.
            First row is zero-padded.
        """
        xyz = self.sel(markers, dims).astype(np.float64)   # (T, M, D)
        # diff → (T-1, M, D)
        dxyz = np.diff(xyz, axis=0) * self.samplerate / 10.0  # mm/s → cm/s
        speed = np.sqrt((dxyz ** 2).sum(axis=-1))              # (T-1, M)
        return np.vstack([np.zeros((1, speed.shape[1])), speed])

    def acc(self, markers=None, dims=None) -> np.ndarray:
        """Instantaneous acceleration (cm/s²) for selected markers.

        Returns
        -------
        a : np.ndarray, shape ``(T, n_markers)``
        """
        v = self.vel(markers, dims)   # (T, M)
        da = np.diff(v, axis=0) * self.samplerate
        return np.vstack([da, da[-1:]])

    def dist(self, markerA, markerB) -> np.ndarray:
        """Euclidean distance between two markers over time.

        Parameters
        ----------
        markerA, markerB:
            Marker name (str) or index (int).

        Returns
        -------
        d : np.ndarray, shape ``(T,)``
        """
        ia = self._marker_index(markerA) if isinstance(markerA, str) else int(markerA)
        ib = self._marker_index(markerB) if isinstance(markerB, str) else int(markerB)
        d = self.data
        diff = d[:, ia, :] - d[:, ib, :]
        return np.sqrt((diff ** 2).sum(axis=-1))

    def com(self, markers=None, dims=None) -> np.ndarray:
        """Center of mass across selected markers.

        Returns
        -------
        c : np.ndarray, shape ``(T, n_dims)``
        """
        return self.sel(markers, dims).mean(axis=1)

    # ------------------------------------------------------------------ #
    # Repr                                                                 #
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        shape = self.shape if self._data is not None else "not loaded"
        return (f"NBDxyz(shape={shape}, markers={self.markers}, "
                f"sr={self.samplerate}Hz)")
