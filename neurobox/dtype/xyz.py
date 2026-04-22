"""
xyz.py  —  NBDxyz
==================
Port of MTADxyz.  3-D position data for motion-captured marker tracking.

Data layout: ``(T, N_markers, 3)``  — time × marker × [x, y, z].

Loading
-------
``NBDxyz.from_motive_csv(path, model)``
    Load directly from a Motive ``.csv`` export.

``NBDxyz.from_npy(path)``
    Load from a saved ``.npy`` or ``.npz`` array.

``session.load('xyz')``  (in NBSession)
    Lazy-load via the session pipeline.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from neurobox.dtype.data  import NBData
from neurobox.dtype.epoch import NBEpoch
from neurobox.dtype.model import NBModel


# ---------------------------------------------------------------------------
# NBDxyz
# ---------------------------------------------------------------------------

class NBDxyz(NBData):
    """3-D position time-series for motion-captured markers.

    Parameters
    ----------
    data:
        Array of shape ``(T, N_markers, 3)`` (or ``(T, N_markers, 2)``
        for 2-D tracking).  Pass *None* for lazy loading.
    model:
        NBModel describing the markers.  If None and *markers* is given,
        one is constructed automatically.
    markers:
        Convenience alternative to passing a full NBModel.  Ignored
        when *model* is provided.
    samplerate:
        Tracking sample rate in Hz (Vicon ≈ 120 Hz, Optitrack ≈ 120 Hz).
    sync:
        NBEpoch defining the valid recording window.
    origin:
        Time in seconds of the first sample relative to the
        electrophysiology recording start.
    path / filename:
        Location of saved data file (for save/load).
    """

    def __init__(
        self,
        data: np.ndarray | None = None,
        model: NBModel | None = None,
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
        if model is not None:
            self.model: NBModel = model
        elif markers is not None:
            self.model = NBModel(markers=markers)
        else:
            self.model = NBModel()

    # ------------------------------------------------------------------ #
    # Convenience property (MTA compat)                                   #
    # ------------------------------------------------------------------ #

    @property
    def markers(self) -> list[str]:
        return self.model.markers

    # ------------------------------------------------------------------ #
    # Abstract interface                                                   #
    # ------------------------------------------------------------------ #

    def load(self, path: Path | str | None = None) -> "NBDxyz":
        """Load from a saved ``.npy`` or ``.npz`` file.

        Parameters
        ----------
        path:
            Path to the ``.npy`` / ``.npz`` file.  Falls back to
            ``self.fpath`` if not provided.
        """
        src = Path(path) if path is not None else self.fpath
        if src is None:
            raise ValueError("No path specified.")
        if src.suffix == ".npz":
            ds = np.load(src, allow_pickle=True)
            self._data   = ds["data"]
            if "markers" in ds:
                self.model = NBModel(markers=list(ds["markers"]))
            if "samplerate" in ds:
                self.samplerate = float(ds["samplerate"])
        else:
            self._data = np.load(src, allow_pickle=False)
        return self

    def save_npy(self, path: Path | str | None = None) -> None:
        """Save data to a ``.npz`` file with markers and samplerate."""
        dst = Path(path) if path is not None else self.fpath
        if dst is None:
            raise ValueError("No path specified.")
        np.savez_compressed(
            dst,
            data       = self._data,
            markers    = np.array(self.model.markers),
            samplerate = np.float64(self.samplerate),
        )

    def create(self, *args, **kwargs) -> "NBDxyz":
        return self.load(*args, **kwargs)

    # ------------------------------------------------------------------ #
    # Named-marker selection                                               #
    # ------------------------------------------------------------------ #

    def sel(self, markers=None, dims=None) -> np.ndarray:
        """Select markers and/or spatial dimensions.

        Parameters
        ----------
        markers:
            ``None`` (all), marker name, list of names, or list of ints.
        dims:
            ``None`` (all), or list/slice of dimension indices
            (0=x, 1=y, 2=z).

        Returns
        -------
        np.ndarray, shape ``(T, n_markers_sel, n_dims_sel)``
        """
        d = self.data
        if d is None:
            raise RuntimeError("Data not loaded.")

        mi = self.model.resolve(markers)
        out = d[:, mi, :]
        if dims is not None:
            out = out[:, :, dims]
        return out

    # ------------------------------------------------------------------ #
    # Kinematics                                                           #
    # ------------------------------------------------------------------ #

    def vel(self, markers=None, dims=None) -> np.ndarray:
        """Instantaneous speed (cm/s) for selected markers.

        Returns
        -------
        v : np.ndarray, shape ``(T, n_markers)``
            Speed at each frame.  First row zero-padded.
        """
        xyz   = self.sel(markers, dims).astype(np.float64)   # (T, M, D)
        dxyz  = np.diff(xyz, axis=0) * self.samplerate / 10.0  # mm→cm, per sec
        speed = np.sqrt((dxyz ** 2).sum(axis=-1))              # (T-1, M)
        return np.vstack([np.zeros((1, speed.shape[1])), speed])

    def acc(self, markers=None, dims=None) -> np.ndarray:
        """Instantaneous acceleration (cm/s²) for selected markers."""
        v  = self.vel(markers, dims)     # (T, M)
        da = np.diff(v, axis=0) * self.samplerate
        return np.vstack([da, da[-1:]])

    def dist(self, markerA, markerB) -> np.ndarray:
        """Euclidean distance between two markers over time (mm).

        Returns
        -------
        d : np.ndarray, shape ``(T,)``
        """
        ia = (self.model.index(markerA) if isinstance(markerA, str)
              else int(markerA))
        ib = (self.model.index(markerB) if isinstance(markerB, str)
              else int(markerB))
        diff = self.data[:, ia, :] - self.data[:, ib, :]
        return np.sqrt((diff ** 2).sum(axis=-1))

    def com(self, markers=None, dims=None) -> np.ndarray:
        """Center of mass across selected markers.

        Returns
        -------
        c : np.ndarray, shape ``(T, n_dims)``
        """
        return self.sel(markers, dims).mean(axis=1)

    # ------------------------------------------------------------------ #
    # Motive CSV factory                                                   #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_motive_csv(
        cls,
        csv_path: Path | str,
        subject_name: str | None = None,
        samplerate: float | None = None,
        xyz_order: str = "XYZ",
        scale_to_mm: bool = True,
    ) -> "NBDxyz":
        """Load marker position data from a Motive ``.csv`` export.

        The Motive CSV layout (7-line header, then data rows):

        Line 1: metadata  (``Total Frames in Take``, ``Export Frame Rate``, …)
        Line 2: blank
        Line 3: column-type labels (``Frame``, ``Time``, ``Bone``, …)
        Line 4: object names  (rigid body / marker names per column)
        Line 5: element IDs
        Line 6: data types  (``Position``, ``Rotation``, …)
        Line 7: dimension labels  (``X``, ``Y``, ``Z``, ``W``, …)

        Then numeric rows, one per frame.

        Parameters
        ----------
        csv_path:
            Path to the Motive export CSV.
        subject_name:
            If provided, only columns whose model-name matches this
            string are loaded.  Otherwise all ``Bone`` (marker) columns
            are loaded.
        samplerate:
            Override the frame rate read from the header.
        xyz_order:
            Motive exports in ``'XZY'`` by default.  Pass ``'XZY'`` to
            re-order to ``'XYZ'``.  Pass ``'XYZ'`` if already correct.
        scale_to_mm:
            Motive exports in metres.  When True, multiply by 1000.

        Returns
        -------
        NBDxyz
        """
        csv_path = Path(csv_path)
        with open(csv_path, "r") as fh:
            lines = [fh.readline() for _ in range(7)]
        hdr_tokens = lines[0].rstrip("\n").split(",")

        # ── Extract frame rate from header ─────────────────────────────── #
        def _hdr_val(key: str) -> str | None:
            for i, tok in enumerate(hdr_tokens):
                if key.lower() in tok.lower() and i + 1 < len(hdr_tokens):
                    return hdr_tokens[i + 1].strip()
            return None

        if samplerate is None:
            sr_str = _hdr_val("Export Frame Rate")
            samplerate = float(sr_str) if sr_str else 120.0

        n_frames_str = _hdr_val("Total Frames in Take")
        n_exported_str = _hdr_val("Total Exported Frames")
        n_frames   = int(n_frames_str)   if n_frames_str   else None
        n_exported = int(n_exported_str) if n_exported_str else n_frames

        # ── Parse column-header rows ────────────────────────────────────── #
        col_types   = lines[2].rstrip("\n").split(",")  # Frame/Time/Bone/…
        model_names = lines[3].rstrip("\n").split(",")  # per-column object names
        data_types  = lines[5].rstrip("\n").split(",")  # Position/Rotation/…
        dim_labels  = lines[6].rstrip("\n").split(",")  # X/Y/Z/W

        # ── Identify Position columns for selected subject ──────────────── #
        position_cols: list[int] = []   # column indices (0-based)
        marker_order:  list[str] = []   # marker name for each selected column

        # Group consecutive X,Y,Z position triplets by object name
        i = 0
        while i < len(col_types):
            ct = col_types[i].strip()
            dt = data_types[i].strip() if i < len(data_types) else ""
            mn = model_names[i].strip() if i < len(model_names) else ""

            # A Bone position block: three consecutive X,Y,Z columns
            if "Bone" in ct and "Position" in dt:
                if subject_name is None or subject_name.lower() in mn.lower():
                    marker_name = mn.split(":")[-1].strip() if ":" in mn else mn
                    # Expect X, Y (or Z), Z (or Y) in next columns
                    for d in range(3):
                        if i + d < len(col_types):
                            position_cols.append(i + d)
                    marker_order.append(marker_name)
                i += 3
                continue
            i += 1

        if not marker_order:
            raise ValueError(
                f"No marker position columns found in {csv_path.name}.  "
                f"Check subject_name={subject_name!r} and that the CSV "
                f"contains Bone/Position data."
            )

        # ── Load numeric data ───────────────────────────────────────────── #
        # Columns 0 and 1 are Frame and Time; positions start at 2
        # Use numpy's genfromtxt to skip the 7-header lines
        all_data = np.genfromtxt(
            csv_path,
            delimiter=",",
            skip_header=7,
            filling_values=np.nan,
        )
        # Slice to position columns
        pos = all_data[:, position_cols].reshape(-1, len(marker_order), 3)

        # ── Re-order XZY → XYZ if needed ───────────────────────────────── #
        if xyz_order.upper() == "XZY":
            pos = pos[:, :, [0, 2, 1]]

        # ── Scale to mm ────────────────────────────────────────────────── #
        if scale_to_mm:
            pos = pos * 1000.0

        # ── Handle missing frames (interpolate or keep NaN) ─────────────── #
        if n_frames is not None and n_exported is not None and n_frames != n_exported:
            timestamps = all_data[:, 1]  # column 1 = time in seconds
            t_full = np.linspace(timestamps[0], timestamps[-1], n_frames)
            pos_full = np.full((n_frames, len(marker_order), 3), np.nan)
            for mi in range(len(marker_order)):
                for di in range(3):
                    valid = np.isfinite(pos[:, mi, di])
                    if valid.sum() > 1:
                        from scipy.interpolate import interp1d
                        f = interp1d(timestamps[valid], pos[valid, mi, di],
                                     kind="linear", fill_value=np.nan,
                                     bounds_error=False)
                        pos_full[:, mi, di] = f(t_full)
            pos = pos_full

        model = NBModel(markers=marker_order)
        return cls(
            data       = pos,
            model      = model,
            samplerate = samplerate,
            name       = csv_path.stem,
        )

    # ------------------------------------------------------------------ #
    # Repr                                                                 #
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        shape = self.shape if self._data is not None else "not loaded"
        return (f"NBDxyz(shape={shape}, "
                f"markers={self.model.markers}, "
                f"sr={self.samplerate}Hz)")
