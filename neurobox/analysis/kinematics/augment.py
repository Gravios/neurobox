"""
neurobox.analysis.kinematics.augment
=====================================
Augment a tracking :class:`NBDxyz` with the standard derived markers
used throughout MTA pipelines: body / head / all centre of mass, plus
a synthesised ``nose`` marker.

Ported from :file:`MTA/utilities/preproc_xyz.m` (Anton Sirota /
Justin Graboski).  The MATLAB original was 420 lines split between
two responsibilities:

1. **Spline-spine resampling** (`procOpts` switch, lines 1-326).
   Five modes that fit cubic splines through the spine markers and
   resample to equidistant arclength.  These depend on a chain of
   MTA-only helpers (``fet_spline_spine``, ``arclength``,
   ``interparc``, ``NearestNeighbour``) totalling another ~600 lines
   of MATLAB.  Not ported in this round — file an issue if you need
   one of these modes.

2. **Virtual marker injection** (lines 328-419).  Adds ``bcom``,
   ``acom``, ``hcom``, ``nose``, replaces NaNs.  This is the part
   most pipelines actually use — every ``preproc_xyz(Trial)`` call
   with no ``procOpts`` lands here.

This module ports (2).  The function name has been changed from the
opaque ``preproc_xyz`` to :func:`augment_xyz` because the operation
no longer includes the spline-resampling preprocessing.

What gets added
---------------
* ``bcom`` (body centre of mass) — averaged from spine markers
  ``spine_lower``, ``pelvis_root``, ``spine_middle``, ``spine_upper``
  (only when ≥ 3 of those are present).
* ``acom`` (all-body COM) — same formula, including head markers.
* ``hcom`` (head COM) — averaged from ``head_back``, ``head_left``,
  ``head_front``, ``head_right`` (or just front + back if some are
  missing).  Only when ≥ 3 head markers are present.
* ``nose`` — synthesised 40 mm forward of ``hcom`` along the head's
  local x-axis.  Only added when no ``nose`` marker is already
  present and ``hcom`` exists.

Each existing marker is left unchanged.  The ``hcom`` step is skipped
when ``hcom`` already exists (e.g. it was added in a previous call).

Returns a new :class:`NBDxyz`; the original is unmodified.

Examples
--------
>>> from neurobox.analysis.kinematics import augment_xyz
>>> aug = augment_xyz(xyz)
>>> 'bcom' in aug.model.markers
True
>>> 'nose' in aug.model.markers
True
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from neurobox.dtype.xyz import NBDxyz


# Standard marker groups used by the COM / nose construction
_BODY_MARKERS:  tuple[str, ...] = (
    "spine_lower", "pelvis_root", "spine_middle", "spine_upper",
)
_ALL_MARKERS_BODY: tuple[str, ...] = (
    "spine_lower", "pelvis_root", "spine_middle",
    "head_back", "head_front",
)
_HEAD_MARKERS_FULL: tuple[str, ...] = (
    "head_back", "head_left", "head_front", "head_right",
)
_HEAD_MARKERS_FALLBACK: tuple[str, ...] = ("head_back", "head_front")

# Distance from hcom to the synthesised nose marker, mm (matches MATLAB).
_NOSE_FORWARD_MM: float = 40.0


def _present(model_markers: Sequence[str], names: Sequence[str]) -> list[str]:
    """Return the subset of *names* that are actually in *model_markers*."""
    available = set(model_markers)
    return [n for n in names if n in available]


def _safe_normalise(v: np.ndarray) -> np.ndarray:
    """Row-wise unit-vector normalise, leaving zero-norm rows as zeros."""
    norm = np.linalg.norm(v, axis=1, keepdims=True)
    out = np.zeros_like(v)
    np.divide(v, norm, out=out, where=norm > 0)
    return out


def _build_head_orthonormal_basis(
    xyz_data: np.ndarray,
    idx_back: int,
    idx_left: int | None,
    idx_right: int | None,
    idx_hcom: int,
) -> np.ndarray:
    """Return ``(T, 3, 3)`` rotation matrices: head-frame xyz axes per frame.

    Mirrors the MATLAB construction at lines 387-400 of preproc_xyz.m.
    Uses ``head_left`` if available, else ``head_right`` with a
    sign-flip on the cross product (matches the MATLAB ``hsign``
    fallback).
    """
    head_back = xyz_data[:, idx_back, :]              # (T, 3)
    hcom      = xyz_data[:, idx_hcom, :]              # (T, 3)
    if idx_left is not None:
        side = xyz_data[:, idx_left, :]
        sign = +1.0
    elif idx_right is not None:
        side = xyz_data[:, idx_right, :]
        sign = -1.0
    else:
        raise ValueError("Need head_left or head_right to build head frame")

    nz = _safe_normalise(sign * np.cross(head_back - hcom, side - hcom))
    ny = _safe_normalise(np.cross(nz, head_back - hcom))
    nx = _safe_normalise(np.cross(ny, nz))
    return nx, ny, nz


def augment_xyz(
    xyz:               NBDxyz,
    *,
    samplerate:        float | None = None,
    add_bcom:          bool = True,
    add_acom:          bool = True,
    add_hcom:          bool = True,
    add_nose:          bool = True,
    nose_forward_mm:   float = _NOSE_FORWARD_MM,
    fill_nan_with_eps: bool = True,
) -> NBDxyz:
    """Add standard derived markers to an :class:`NBDxyz`.

    Port of the second half of :file:`MTA/utilities/preproc_xyz.m`.
    See module docstring for what's added and what was deferred.

    Parameters
    ----------
    xyz:
        Source position data.  Must contain enough of the spine and
        head markers for the requested COM calculations.  Markers
        missing for a given COM cause that COM to be silently
        skipped (matching MATLAB).
    samplerate:
        Optional resample to this rate after augmentation.  ``None``
        (default) → no resampling.
    add_bcom, add_acom, add_hcom, add_nose:
        Which derived markers to add.  Default all True.
    nose_forward_mm:
        Distance (mm) along the head's local x-axis at which the
        synthetic nose marker sits.  MATLAB default 40.
    fill_nan_with_eps:
        If True (default), replace NaN values in the output with
        :func:`numpy.finfo` epsilon — matches MATLAB's
        ``xyz.data(isnan(xyz.data)) = eps``, which avoids NaN
        propagation in downstream Cartesian arithmetic.

    Returns
    -------
    NBDxyz
        New object; the input is unmodified.

    Notes
    -----
    The MATLAB original adds markers in this order:
    ``bcom`` → ``acom`` → ``hcom`` → ``nose``.  This port preserves
    that order, which matters because the ``nose`` synthesis depends
    on ``hcom`` already being present.

    The MATLAB also tracks coloured "stick" connections used for
    skeleton plotting.  Those carry through into ``NBModel.connections``
    as plain edge tuples (no colours — neurobox's plotting layer
    doesn't use them yet).
    """
    if xyz._data is None:
        raise RuntimeError("XYZ data not loaded.")

    out = xyz
    markers = list(out.model.markers)

    # ── 1. bcom — body centre of mass ────────────────────────────────── #
    if add_bcom and "bcom" not in markers:
        body_present = _present(markers, _BODY_MARKERS)
        if len(body_present) >= 3:
            bcom = out.com(body_present)
            connections = [(m, "bcom") for m in body_present]
            out = out.add_marker("bcom", bcom, connections=connections)
            markers.append("bcom")

    # ── 2. acom — full-body centre of mass ──────────────────────────── #
    if add_acom and "acom" not in markers:
        all_present = _present(markers, _ALL_MARKERS_BODY)
        if len(all_present) >= 3:
            acom = out.com(all_present)
            connections = [(m, "acom") for m in all_present]
            out = out.add_marker("acom", acom, connections=connections)
            markers.append("acom")

    # ── 3. hcom — head centre of mass ───────────────────────────────── #
    if add_hcom and "hcom" not in markers:
        head_full = _present(markers, _HEAD_MARKERS_FULL)
        if len(head_full) >= 3:
            hcom = out.com(head_full)
            connections = [(m, "hcom") for m in head_full]
            out = out.add_marker("hcom", hcom, connections=connections)
            markers.append("hcom")
        else:
            head_fallback = _present(markers, _HEAD_MARKERS_FALLBACK)
            if len(head_fallback) >= 2:
                hcom = out.com(head_fallback)
                connections = [(m, "hcom") for m in head_fallback]
                out = out.add_marker("hcom", hcom, connections=connections)
                markers.append("hcom")

    # ── 4. nose — synthesised 40 mm forward of hcom ─────────────────── #
    if add_nose and "nose" not in markers and "hcom" in markers:
        if "head_back" in markers and (
            "head_left" in markers or "head_right" in markers
        ):
            data = out._data
            idx_back  = out.model.index("head_back")
            idx_hcom  = out.model.index("hcom")
            idx_left  = out.model.index("head_left")  if "head_left"  in markers else None
            idx_right = out.model.index("head_right") if "head_right" in markers else None
            try:
                nx, ny, nz = _build_head_orthonormal_basis(
                    data, idx_back, idx_left, idx_right, idx_hcom
                )
            except ValueError:
                nx = None
            if nx is not None:
                hcom_pos = data[:, idx_hcom, :]
                # MATLAB: nose = hcom + nx * (-40)
                # The negation comes from the sign convention used in the
                # MATLAB head-frame; nx points toward the back of the head.
                nose = hcom_pos + nx * (-float(nose_forward_mm))
                connections = [
                    (m, "nose") for m in
                    _present(markers, _HEAD_MARKERS_FULL)
                ]
                out = out.add_marker("nose", nose, connections=connections)

    # ── 5. NaN → eps ─────────────────────────────────────────────────── #
    if fill_nan_with_eps:
        # add_marker returned a new copy whose _data is a fresh ndarray
        # (concatenated), but the existing-marker block may still alias
        # the input.  Copy on first NaN-fill to be safe.
        if np.isnan(out._data).any():
            data = out._data.copy()
            data[np.isnan(data)] = float(np.finfo(data.dtype).eps)
            import copy as _copy
            new = _copy.copy(out)
            new._data = data
            out = new

    # ── 6. Optional resample ────────────────────────────────────────── #
    if samplerate is not None:
        out = out.resample(float(samplerate))

    return out
