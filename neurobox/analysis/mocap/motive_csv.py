"""
neurobox.analysis.mocap.motive_csv
===================================
Parse rigid-body-orientation (RBO) data from OptiTrack Motive's
exported CSV format.

Port of :file:`MTA/utilities/mocap/parse_rbo_from_csv.m`.

The MATLAB original was tightly coupled to the Sirota-lab session
metadata convention (``meta.subjects(s).rb(r).alias`` etc.), reading
the CSV and writing out a per-take ``.mat`` file containing a
multi-subject ``subjects`` struct array.  This port decouples the
parsing from the metadata conventions: you pass a list of rigid-body
aliases (the names Motive used for each rigid body), and you get
back a plain :class:`MotiveTakeResult` with everything as ndarrays
and dicts.  Saving to disk is up to the caller.

Format
------
Motive's CSV layout used by the lab (Format 1.x):

* Row 1 — global metadata: ``"Format Version,1.0,Take Name,foo,Capture
  Frame Rate,250,..."`` plus ``"Total Frames in Take"`` and
  ``"Total Exported Frames"`` keys.
* Row 2 — empty.
* Rows 3-7 — column headers:
    3. ``clabels`` — top-level: ``"Rigid Body"`` / ``"Rigid Body
       Marker"`` / ``"Marker"``
    4. ``modelNames`` — per-rigid-body name (the alias string)
    5. ``elementId`` — internal Motive ID
    6. ``dataType`` — ``Rotation`` / ``Position`` / ``Mean Marker
       Error`` etc.
    7. ``dimId`` — ``X`` / ``Y`` / ``Z`` / ``W``
* Row 8 onward — per-frame data, comma-separated.

Each rigid body contributes 8 columns: ``qw, qx, qy, qz, x, y, z,
mean_error``.

Coordinate convention
---------------------
The MATLAB original applied a symmetric column swap to both the
4-channel rotation block and the 3-channel position block:
``cols [3,4]`` and ``cols [6,7]`` in 1-based indexing are swapped
within each rigid body.

For position this is the standard **Y-up → Z-up** conversion (Motive
emits Y-up; downstream lab code uses Z-up): ``(x, y_up, z) →
(x, z, y_up)``.

For the rotation channels the lab applied the **same** ``[3,4]`` swap
— this is **not** a mathematically rigorous quaternion change-of-basis
for a Y→Z up conversion (which would require additionally negating
one component), but it is what the lab pipeline used in practice.
This port preserves that behaviour verbatim so that downstream
analyses behave identically.

Position is also rescaled from metres → millimetres.

Final per-rigid-body channel order is
``(x_mm, y_mm, z_mm, mean_err, qa, qb, qc, qd)``, matching the
MATLAB reorder ``[5:8, 1:4]``.  Whether the quaternion conventionally
reads ``(qw, qx, qy, qz)`` or ``(qx, qy, qz, qw)`` depends on the
Motive export settings; this port does not attempt to disambiguate
and treats the four rotation channels as opaque scalars in the order
emitted.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass
class MotiveTakeResult:
    """Parsed contents of one Motive CSV take.

    Attributes
    ----------
    rbo_data:
        ``(T, n_bodies, 8)`` array of per-frame rigid-body data.
        Last-axis order: ``(x_mm, y_mm, z_mm, mean_err, qa, qb, qc,
        qd)`` — the four rotation channels are passed through in the
        order emitted by Motive (commonly ``qx, qy, qz, qw`` or
        ``qw, qx, qy, qz`` depending on export settings).
    rbo_aliases:
        List of rigid-body aliases in the order of the second axis of
        ``rbo_data`` (i.e. ``rbo_data[:, k, :]`` belongs to
        ``rbo_aliases[k]``).
    frames:
        ``(T,)`` Motive frame indices.
    timestamps:
        ``(T,)`` per-frame timestamps in seconds.
    samplerate:
        Capture frame rate (Hz).
    n_frames_total:
        ``Total Frames in Take`` from the CSV header — may exceed the
        number of rows actually present (Motive can drop frames).
    n_frames_exported:
        ``Total Exported Frames`` from the header.
    """
    rbo_data:          np.ndarray
    rbo_aliases:       list[str]
    frames:            np.ndarray
    timestamps:        np.ndarray
    samplerate:        float
    n_frames_total:    int
    n_frames_exported: int


# ─────────────────────────────────────────────────────────────────────────── #
# Public API                                                                   #
# ─────────────────────────────────────────────────────────────────────────── #

def parse_rbo_from_csv(
    path:        str | Path,
    aliases:     Sequence[str] | None = None,
    interpolate_dropped_frames: bool = True,
) -> MotiveTakeResult:
    """Parse a Motive CSV file and return rigid-body orientation data.

    Port of :file:`MTA/utilities/mocap/parse_rbo_from_csv.m`.

    Parameters
    ----------
    path:
        Path to the Motive ``.csv`` export.
    aliases:
        Optional list of rigid-body aliases to extract.  ``None``
        (default) auto-detects every distinct rigid-body name in
        the CSV's ``modelNames`` row.  When supplied, the result's
        ``rbo_data`` is reordered to match the requested list and
        an error is raised if any alias is not present.
    interpolate_dropped_frames:
        If True (default), when ``Total Exported Frames`` differs
        from ``Total Frames in Take``, linearly interpolate every
        column to fill the missing frames (matches the MATLAB
        ``interp1`` branch on line 97).  When False, the result has
        ``T == Total Exported Frames`` and the ``frames`` array
        carries the actual Motive frame indices.

    Returns
    -------
    :class:`MotiveTakeResult`
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Motive CSV not found: {path}")

    with path.open("r") as fh:
        # ── Row 1: global metadata key/value pairs ─────────────────── #
        header = fh.readline().rstrip("\n").split(",")
        meta = _parse_kv_header(header)
        n_frames_total    = int(float(meta["Total Frames in Take"]))
        n_frames_exported = int(float(meta["Total Exported Frames"]))
        samplerate        = float(meta["Export Frame Rate"])

        # ── Row 2: blank ───────────────────────────────────────────── #
        fh.readline()

        # ── Rows 3-7: column headers ────────────────────────────────── #
        clabels    = fh.readline().rstrip("\n").split(",")
        model_names = fh.readline().rstrip("\n").split(",")
        _element_id = fh.readline().rstrip("\n").split(",")
        data_type  = fh.readline().rstrip("\n").split(",")
        dim_id     = fh.readline().rstrip("\n").split(",")

        # ── Detect rigid-body column groups ─────────────────────────── #
        rbo_columns = [i for i, lbl in enumerate(clabels) if lbl == "Rigid Body"]
        if not rbo_columns:
            raise ValueError(
                f"parse_rbo_from_csv: no 'Rigid Body' columns found in {path}"
            )
        # Each rigid body contributes 8 columns.  Sanity check.
        if len(rbo_columns) % 8 != 0:
            raise ValueError(
                f"parse_rbo_from_csv: number of 'Rigid Body' columns "
                f"({len(rbo_columns)}) is not a multiple of 8 in {path}"
            )
        n_rbos_in_csv = len(rbo_columns) // 8

        # First column of each rigid-body group gives the alias name
        all_aliases = [model_names[rbo_columns[8 * k]]
                       for k in range(n_rbos_in_csv)]

        # ── Row 8 onward: per-frame data ────────────────────────────── #
        rows = []
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            row = _parse_csv_row(line, len(clabels))
            if row is not None:
                rows.append(row)

    if not rows:
        raise ValueError(f"parse_rbo_from_csv: no data rows in {path}")
    raw = np.asarray(rows, dtype=np.float64)            # (n_rows, n_cols)

    # ── Optional resampling to fill dropped frames ─────────────────── #
    if interpolate_dropped_frames and n_frames_exported != n_frames_total:
        # Interpolate every column to a uniform 1..n_frames_total grid
        target = np.arange(1, n_frames_total + 1, dtype=np.float64)
        src    = raw[:, 0]                              # frame index column
        out = np.empty((target.size, raw.shape[1]), dtype=np.float64)
        for c in range(raw.shape[1]):
            out[:, c] = np.interp(target, src, raw[:, c])
        raw = out

    frames     = raw[:, 0].astype(np.int64)
    timestamps = raw[:, 1].astype(np.float64)
    payload    = raw[:, 2:]                             # (T, 8 * n_rbos_in_csv)

    # ── Reshape to (T, n_rbos, 8) ──────────────────────────────────── #
    # The Motive export packs a rigid body's 8 channels contiguously, then
    # moves to the next.  In MATLAB this was reshape -> permute([1,3,2]).
    rbo = payload.reshape(payload.shape[0], n_rbos_in_csv, 8)

    # ── Reorder XZY → XYZ within each block ────────────────────────── #
    # MATLAB indices [3, 4, 6, 7] (1-based) → [2, 3, 5, 6] (0-based).
    # The swap is (col 2 ↔ col 3) and (col 5 ↔ col 6) within the 8-col block.
    rbo[:, :, [2, 3, 5, 6]] = rbo[:, :, [3, 2, 6, 5]]

    # ── Rescale position columns from m → mm ──────────────────────── #
    # Original block layout after the swap above is
    # (qw, qx, qy, qz, x, y, z, err) — columns 4..7 are position+err.
    # MATLAB scaled columns 5:8 (1-based) which is 4..7 (0-based).
    rbo[:, :, 4:8] *= 1000.0

    # ── Reorder channels to (x, y, z, err, qw, qx, qy, qz) ────────── #
    rbo = rbo[:, :, [4, 5, 6, 7, 0, 1, 2, 3]]

    # ── Subset / reorder by requested aliases ─────────────────────── #
    if aliases is None:
        aliases_out = list(all_aliases)
    else:
        idx = []
        for a in aliases:
            try:
                idx.append(all_aliases.index(a))
            except ValueError:
                raise ValueError(
                    f"parse_rbo_from_csv: rigid body {a!r} not found in CSV. "
                    f"Available: {all_aliases}"
                )
        rbo = rbo[:, idx, :]
        aliases_out = list(aliases)

    return MotiveTakeResult(
        rbo_data          = rbo,
        rbo_aliases       = aliases_out,
        frames            = frames,
        timestamps        = timestamps,
        samplerate        = samplerate,
        n_frames_total    = n_frames_total,
        n_frames_exported = n_frames_exported,
    )


# ─────────────────────────────────────────────────────────────────────────── #
# Internal helpers                                                             #
# ─────────────────────────────────────────────────────────────────────────── #

def _parse_kv_header(tokens: list[str]) -> dict[str, str]:
    """Parse Motive's ``"key,val,key,val,..."`` first-row header."""
    out: dict[str, str] = {}
    for i in range(0, len(tokens) - 1, 2):
        out[tokens[i]] = tokens[i + 1]
    return out


_NUMERIC_RX = re.compile(r"^-?\d+(\.\d+)?([eE][+-]?\d+)?$")


def _parse_csv_row(line: str, n_cols: int) -> list[float] | None:
    """Parse one data row.  Returns None if the row is too short."""
    parts = line.split(",")
    if len(parts) < 3:
        return None
    out = []
    for p in parts[:n_cols]:
        if p == "":
            out.append(np.nan)
        else:
            try:
                out.append(float(p))
            except ValueError:
                out.append(np.nan)
    return out
