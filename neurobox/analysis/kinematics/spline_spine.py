"""
neurobox.analysis.kinematics.spline_spine
==========================================
Cubic-spline interpolation of the spine markers, parameterised by
arc-length, plus the corresponding ``preproc_xyz`` modes.

Ports of:

* :file:`MTA/features/fet_spline_spine.m` → :func:`spline_spine`
* :file:`MTA/utilities/preproc_xyz.m`     →
  :func:`preproc_xyz_spline_spine_head_eqi` and
  :func:`preproc_xyz_spline_spine_head_eqd`

What it does
------------
The motion-capture system records 4 spine-aligned markers
(``spine_lower``, ``pelvis_root``, ``spine_middle``, ``spine_upper``)
plus the head-of-mass.  The raw inter-marker distances vary across
animals because of body-size differences and slightly different
marker placements.  This was a major confound when training
behaviour classifiers across sessions.

The ``SPLINE_SPINE_*`` family of preprocessing modes fits a cubic
spline through the 5 spine+head markers, then **resamples that
spline at fixed reference distances** so every session ends up with
a "standard" body geometry.  Two flavours:

* ``SPLINE_SPINE_HEAD_EQI`` (equal-index): re-place markers at
  fixed normalised arc-length positions ``[0, 0.25, 0.50, 0.75, 1.0]``.
* ``SPLINE_SPINE_HEAD_EQD`` (equal-distance): re-place markers at
  fixed *absolute* distances along the spine (the lab's standard
  "TRB" geometry).  This is what `fet_bref` uses by default.

Both produce an :class:`NBDxyz` with the same marker names but
inter-session-comparable positions.

Differences from MATLAB
-----------------------
* MATLAB used :func:`cscvn` (cardinal-spline curve through points)
  + :func:`fnplt` to evaluate at 100 sample points.  The Python port
  uses :class:`scipy.interpolate.CubicSpline` parameterised by
  cumulative chord length, then reshapes onto 100 samples.
* MATLAB called :func:`interparc` for arc-length-parameterised
  resampling.  Python uses cumulative-chord-length resampling via
  :func:`numpy.interp` for the same effect.
* MATLAB's disk-cache + lazy-recompute pattern is replaced by the
  caller deciding when to call this function.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from copy import deepcopy

import numpy as np
from scipy.interpolate import CubicSpline

from neurobox.dtype.xyz import NBDxyz


__all__ = [
    "SplineSpineResult",
    "spline_spine",
    "preproc_xyz_spline_spine_head_eqi",
    "preproc_xyz_spline_spine_head_eqd",
]


# Standard spine-marker order used by the lab pipelines
DEFAULT_SPINE_MARKERS = (
    "spine_lower", "pelvis_root", "spine_middle", "spine_upper", "hcom",
)
DEFAULT_N_INTERP = 100


@dataclass
class SplineSpineResult:
    """Output of :func:`spline_spine`.

    Attributes
    ----------
    points : np.ndarray, shape ``(T, n_interp, 3)``
        Spline-interpolated points.  Default ``n_interp = 100``.
    markers : tuple[str, ...]
        Marker names used for the spline anchor points (in order).
    samplerate : float
        Inherited from the input *xyz*.
    """
    points:     np.ndarray
    markers:    tuple[str, ...]
    samplerate: float


# ─────────────────────────────────────────────────────────────────────── #
# Helpers                                                                    #
# ─────────────────────────────────────────────────────────────────────── #

def _frame_finite(frame_xyz: np.ndarray) -> bool:
    """A frame is usable if all marker coordinates are finite and non-zero."""
    return bool(np.isfinite(frame_xyz).all() and (frame_xyz != 0).any())


def _spline_one_frame(anchors: np.ndarray, n_out: int) -> np.ndarray:
    """Cubic spline through ``anchors`` resampled at ``n_out`` chord-uniform points.

    Parameters
    ----------
    anchors:
        ``(M, 3)`` marker positions for one frame.
    n_out:
        Number of output points (default 100 in the MATLAB pipeline).

    Returns
    -------
    np.ndarray
        ``(n_out, 3)`` spline samples.
    """
    # Chord-length parameterisation
    diffs   = np.diff(anchors, axis=0)
    seg_len = np.linalg.norm(diffs, axis=1)
    if not np.all(seg_len > 0):
        # Degenerate — return zeros
        return np.zeros((n_out, 3))
    t = np.concatenate([[0.0], np.cumsum(seg_len)])
    cs = CubicSpline(t, anchors, axis=0, bc_type="natural")
    t_eval = np.linspace(t[0], t[-1], n_out)
    return cs(t_eval)


def _arc_length_resample(
    points:    np.ndarray,
    n_targets: int,
    *,
    target_distances: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Resample a curve at uniform (or specified) arc-length positions.

    Parameters
    ----------
    points:
        ``(N, 3)`` polyline.
    n_targets:
        Number of equally-spaced output points along arc length (used
        when *target_distances* is None).
    target_distances:
        Optional ``(K,)`` absolute arc-length positions to sample at.
        If given, *n_targets* is ignored and the output has shape
        ``(K, 3)``.

    Returns
    -------
    np.ndarray
        ``(n_targets, 3)`` or ``(K, 3)`` resampled points.
    """
    diffs   = np.diff(points, axis=0)
    seg_len = np.linalg.norm(diffs, axis=1)
    cum     = np.concatenate([[0.0], np.cumsum(seg_len)])
    total   = float(cum[-1])
    if total <= 0:
        # All points coincide
        return np.tile(points[0], (n_targets if target_distances is None
                                   else len(target_distances), 1))

    if target_distances is None:
        target_distances = np.linspace(0.0, total, n_targets)

    # Per-axis linear interpolation against cumulative arc-length
    out = np.empty((len(target_distances), 3), dtype=np.float64)
    for d in range(3):
        out[:, d] = np.interp(target_distances, cum, points[:, d])
    return out


# ─────────────────────────────────────────────────────────────────────── #
# Public API                                                                 #
# ─────────────────────────────────────────────────────────────────────── #

def spline_spine(
    xyz:        NBDxyz,
    *,
    markers:    Sequence[str] = DEFAULT_SPINE_MARKERS,
    n_interp:   int = DEFAULT_N_INTERP,
) -> SplineSpineResult:
    """Cubic spline through the spine markers, evaluated at *n_interp* points.

    Port of :file:`MTA/features/fet_spline_spine.m`.

    Parameters
    ----------
    xyz:
        Source :class:`NBDxyz`.  Must contain all *markers*.
    markers:
        Marker names defining the spline anchor points, in order from
        tail to head.  Default
        ``('spine_lower', 'pelvis_root', 'spine_middle',
        'spine_upper', 'hcom')`` matches MATLAB.
    n_interp:
        Number of output points along the spline.  Default 100
        matches MATLAB ``fnplt`` default.

    Returns
    -------
    SplineSpineResult
        ``points`` array of shape ``(T, n_interp, 3)``.  Frames where
        any anchor marker is non-finite or all-zero produce zeros.
    """
    # Validate markers
    missing = [m for m in markers if xyz.model.index(m) is None
               and m not in xyz.model.markers]
    # `xyz.model.index` raises on missing — work around to give a nicer error
    for m in markers:
        if m not in xyz.model.markers:
            raise KeyError(
                f"spline_spine: marker {m!r} not in xyz.model "
                f"(have: {xyz.model.markers!r})"
            )

    indices = [xyz.model.index(m) for m in markers]
    T = xyz.data.shape[0]
    out = np.zeros((T, n_interp, 3), dtype=np.float64)
    for t in range(T):
        anchors = xyz.data[t, indices, :]
        if not _frame_finite(anchors):
            continue
        out[t] = _spline_one_frame(anchors, n_interp)

    return SplineSpineResult(
        points     = out,
        markers    = tuple(markers),
        samplerate = float(xyz.samplerate),
    )


def preproc_xyz_spline_spine_head_eqi(
    xyz:             NBDxyz,
    *,
    target_markers:  Sequence[str] = DEFAULT_SPINE_MARKERS,
    n_interp:        int = DEFAULT_N_INTERP,
) -> NBDxyz:
    """``SPLINE_SPINE_HEAD_EQI`` preprocessing — equal-index spine.

    Port of the ``SPLINE_SPINE_HEAD_EQI`` branch of
    :file:`MTA/utilities/preproc_xyz.m`.

    Re-places the inner spine markers (``pelvis_root``,
    ``spine_middle``, ``spine_upper``) at **fixed normalised arc-length
    positions** along the cubic spline through the original spine +
    head markers.  The endpoints (``spine_lower`` and ``hcom``) are
    preserved.

    Parameters
    ----------
    xyz:
        Source :class:`NBDxyz` with the standard 5 spine+head anchor
        markers.
    target_markers:
        Markers in tail-to-head order.  Default matches MATLAB.
    n_interp:
        Spline evaluation density.  Default 100.

    Returns
    -------
    NBDxyz
        New :class:`NBDxyz` with the inner spine markers replaced by
        their equal-index (normalised arc-length) positions on the
        spline.  Endpoints unchanged.
    """
    n_markers = len(target_markers)
    if n_markers < 3:
        raise ValueError(
            f"need at least 3 anchor markers; got {n_markers}"
        )

    # Build the spline once
    ssr = spline_spine(xyz, markers=target_markers, n_interp=n_interp)

    # Equal-index target indices on the (n_interp,) spline curve.
    # MATLAB's baseInd = 100/(numMarkers-1) gives indices at
    # multiples of (n_interp-1)/(n_markers-1).
    interior_count = n_markers - 2
    fractional_positions = np.linspace(
        0.0, 1.0, n_markers,
    )[1:-1]  # exclude endpoints

    out = NBDxyz(
        data       = xyz.data.copy(),
        model      = deepcopy(xyz.model),
        samplerate = xyz.samplerate,
        name       = xyz.name + "_sehs" if xyz.name else "sehs",
        label      = "sehs",
        key        = "h",
        path       = xyz.path,
    )

    for k, frac in enumerate(fractional_positions):
        marker = target_markers[k + 1]
        midx = out.model.index(marker)
        # Linear-interp into the spline curve at index frac*(n_interp-1)
        target_idx = frac * (n_interp - 1)
        i_lo = int(np.floor(target_idx))
        i_hi = min(i_lo + 1, n_interp - 1)
        w_hi = target_idx - i_lo
        out.data[:, midx, :] = (
            ssr.points[:, i_lo, :] * (1.0 - w_hi)
            + ssr.points[:, i_hi, :] * w_hi
        )

    out.update_hash()
    return out


def preproc_xyz_spline_spine_head_eqd(
    xyz:                 NBDxyz,
    *,
    target_markers:      Sequence[str] = DEFAULT_SPINE_MARKERS,
    n_interp:            int = DEFAULT_N_INTERP,
    reference_session:   Optional[NBDxyz] = None,
) -> NBDxyz:
    """``SPLINE_SPINE_HEAD_EQD`` preprocessing — equal-distance spine.

    Port of the ``SPLINE_SPINE_HEAD_EQD`` branch of
    :file:`MTA/utilities/preproc_xyz.m`.

    Re-places the inner spine markers at **fixed absolute arc-length
    distances** along the cubic spline.  The reference distances are
    derived from the **median segment lengths** of the source
    session's own spine — this normalises within-session jitter but
    leaves between-session geometry alone.

    For cross-session normalisation, pass *reference_session* — the
    target distances will then be derived from that reference.

    Parameters
    ----------
    xyz:
        Source :class:`NBDxyz`.
    target_markers:
        Markers in tail-to-head order.  Default matches MATLAB.
    n_interp:
        Spline evaluation density.  Default 100.
    reference_session:
        Optional reference :class:`NBDxyz` whose inter-marker
        distances define the target geometry.  ``None`` → use the
        source session's own median.

    Returns
    -------
    NBDxyz
        New :class:`NBDxyz` with markers re-placed at the standardised
        arc-length distances.
    """
    n_markers = len(target_markers)
    if n_markers < 3:
        raise ValueError(
            f"need at least 3 anchor markers; got {n_markers}"
        )

    # Build the spline
    ssr = spline_spine(xyz, markers=target_markers, n_interp=n_interp)

    # Reference distances between consecutive markers
    ref = reference_session if reference_session is not None else xyz
    ref_indices = [ref.model.index(m) for m in target_markers]
    ref_pos = ref.data[:, ref_indices, :]
    seg_lens = np.linalg.norm(np.diff(ref_pos, axis=1), axis=2)  # (T, n-1)
    finite_frame = np.isfinite(seg_lens).all(axis=1) & (seg_lens > 0).all(axis=1)
    if not finite_frame.any():
        # Fall back to mean segment length on the source spline itself
        target_lens = np.full(n_markers - 1, 1.0)
    else:
        target_lens = np.median(seg_lens[finite_frame], axis=0)
    target_total = float(target_lens.sum())
    cum_targets  = np.r_[0.0, np.cumsum(target_lens)]   # absolute arc-length

    out = NBDxyz(
        data       = xyz.data.copy(),
        model      = deepcopy(xyz.model),
        samplerate = xyz.samplerate,
        name       = xyz.name + "_seh" if xyz.name else "seh",
        label      = "seh",
        key        = "h",
        path       = xyz.path,
    )

    T = xyz.data.shape[0]
    inner = list(range(1, n_markers - 1))
    for t in range(T):
        if not _frame_finite(xyz.data[t, [out.model.index(m)
                                           for m in target_markers], :]):
            continue
        # Resample the spline at the target absolute distances.  Scale
        # cum_targets to fit into the [0, total_arc_length] of this frame.
        diffs   = np.diff(ssr.points[t], axis=0)
        seg_len = np.linalg.norm(diffs, axis=1)
        cum     = np.r_[0.0, np.cumsum(seg_len)]
        frame_total = float(cum[-1])
        if frame_total <= 0:
            continue
        # Scale target arc-length to this frame's total
        scale = frame_total / target_total if target_total > 0 else 1.0
        scaled_targets = cum_targets * scale
        for k in inner:
            mk = target_markers[k]
            midx = out.model.index(mk)
            target = scaled_targets[k]
            for d in range(3):
                out.data[t, midx, d] = float(np.interp(
                    target, cum, ssr.points[t, :, d],
                ))

    out.update_hash()
    return out
