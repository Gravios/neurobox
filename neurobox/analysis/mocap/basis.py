"""
neurobox.analysis.mocap.basis
==============================
Per-frame orthonormal coordinate frames built from a set of
co-moving rigid-body markers.

The functions here all mirror MTA's :func:`compute_rb_basis_timeseries`
/ :func:`compute_rigidbody_basis_timeseries` (the two MATLAB files are
verbatim duplicates), and the per-triad construction in
:func:`compute_marker_triads`.

Convention
----------
The axes are built as follows, using the first two markers
``m₀, m₁`` of the supplied marker list along with the rigid-body
centre of mass ``c``::

    n_z = − (m₀ − c) × (m₁ − c)
    n_y =   n_z × (m₀ − c)
    n_x =   n_z × n_y

Each axis is unit-normalised.  This matches the MATLAB convention
exactly — note the leading minus sign on ``n_z``.

Returned shape is ``(T, 3, 3)`` where ``basis[:, :, 0]`` is ``n_x``,
``basis[:, :, 1]`` is ``n_y``, ``basis[:, :, 2]`` is ``n_z``.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from neurobox.dtype.xyz import NBDxyz


def _normalise(v: np.ndarray) -> np.ndarray:
    """Row-wise unit-normalise an ``(T, 3)`` vector array.

    Frames with zero norm are returned as zeros (rather than NaN).
    """
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    out = np.zeros_like(v)
    np.divide(v, norm, out=out, where=norm > 0)
    return out


def rigid_body_basis(
    xyz:     NBDxyz,
    markers: Sequence[str] = ("head_back", "head_left", "head_front", "head_right"),
) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame orthonormal basis for the rigid body spanned by *markers*.

    Port of :file:`MTA/utilities/mocap/compute_rb_basis_timeseries.m`
    (which is identical to :file:`compute_rigidbody_basis_timeseries.m`
    — the MATLAB code base contains both versions verbatim).

    Parameters
    ----------
    xyz:
        Source position data.
    markers:
        At least two marker names defining the rigid body.  The first
        two are used to build the basis axes; the rest contribute to
        the centre-of-mass calculation.  Default is the standard
        4-head-marker rigid body.

    Returns
    -------
    basis : np.ndarray, shape ``(T, 3, 3)``
        ``basis[:, :, 0]`` is the local x-axis, ``[:, :, 1]`` y-axis,
        ``[:, :, 2]`` z-axis.  Each axis is a unit 3-vector per frame.
    com : np.ndarray, shape ``(T, 3)``
        Centre of mass of the rigid body.

    Raises
    ------
    ValueError
        If fewer than two markers are supplied or any marker is
        missing from ``xyz.model``.
    """
    if xyz._data is None:
        raise RuntimeError("xyz data is not loaded.")
    if len(markers) < 2:
        raise ValueError("rigid_body_basis: need at least 2 markers.")

    com = xyz.com(list(markers))                        # (T, 3)

    idx0 = xyz.model.index(markers[0])
    idx1 = xyz.model.index(markers[1])
    v0 = xyz.data[:, idx0, :] - com                     # (T, 3)
    v1 = xyz.data[:, idx1, :] - com                     # (T, 3)

    nz = _normalise(-np.cross(v0, v1))
    ny = _normalise( np.cross(nz, v0))
    nx = _normalise( np.cross(nz, ny))

    basis = np.stack([nx, ny, nz], axis=-1)             # (T, 3, 3)
    return basis, com


# Backwards-compatible aliases mirroring the two MATLAB names
compute_rb_basis_timeseries        = rigid_body_basis
compute_rigidbody_basis_timeseries = rigid_body_basis


def intermarker_distances(
    xyz:     NBDxyz,
    markers: Sequence[str] | None = None,
) -> np.ndarray:
    """Pairwise marker distances over time.

    Port of :file:`MTA/utilities/mocap/compute_rb_intermarker_distances.m`.

    Parameters
    ----------
    xyz:
        Source position data.
    markers:
        Optional subset of markers.  ``None`` (default) uses every
        marker in ``xyz.model``.

    Returns
    -------
    distances : np.ndarray, shape ``(T, n_pairs)``
        ``n_pairs = N(N-1)/2`` Euclidean distances between every pair
        of markers, ordered by the standard ``(i, j)`` lexicographic
        upper-triangle traversal: (0,1), (0,2), ..., (0,N-1),
        (1,2), ..., (N-2, N-1).  Same ordering as MATLAB's nested
        ``for markerA``/``for markerB`` loops.

    Notes
    -----
    For 3-D xyz data this returns Euclidean distances; for 2-D xyz it
    returns 2-D distances.
    """
    if xyz._data is None:
        raise RuntimeError("xyz data is not loaded.")
    if markers is None:
        markers = xyz.model.markers
    idx = xyz.model.resolve(list(markers))
    sub = xyz.data[:, idx, :]                            # (T, N, n_dims)

    T, N, _ = sub.shape
    n_pairs = N * (N - 1) // 2
    out = np.empty((T, n_pairs), dtype=sub.dtype)
    p = 0
    for a in range(N):
        for b in range(a + 1, N):
            out[:, p] = np.linalg.norm(sub[:, a, :] - sub[:, b, :], axis=1)
            p += 1
    return out


def marker_triads(
    xyz:     NBDxyz,
    markers: Sequence[str] | None = None,
) -> "MarkerTriadResult":
    """For every triplet of markers, compute the local triad basis + diagnostics.

    Port of :file:`MTA/utilities/mocap/compute_marker_triads.m`.

    For each ``(i, j, k)`` triple chosen from the marker set (with the
    middle marker ``j`` treated as the local origin), this returns:

    * ``coor[t, n, :, :]`` — a ``(3, 3)`` local frame whose rows are
      ``(p_k − p_j)``, ``-((p_k − p_j) × (p_i − p_j))``,
      ``(p_k − p_j) × second_axis``, evaluated per frame.  Note: not
      orthonormalised — these are the raw spans matching MATLAB.
    * ``com[t, n, :]`` — centre of mass of the triple.
    * ``imd[t, n, :]`` — the two intermarker distances ``|p_k − p_j|``
      and ``|p_i − p_j|``.
    * ``imo[t, n]`` — the angle (radians) between the two arms,
      computed via ``atan2(|cross|, dot)``.

    The ``nck`` attribute holds the index triples in
    ``itertools.combinations`` order.

    Parameters
    ----------
    xyz:
        Source position data.  3-D or 2-D both supported.
    markers:
        Subset of markers to enumerate triples over.  Default is the
        full model.

    Returns
    -------
    MarkerTriadResult
    """
    from itertools import combinations
    from dataclasses import dataclass

    if xyz._data is None:
        raise RuntimeError("xyz data is not loaded.")
    if markers is None:
        markers = xyz.model.markers
    idx = xyz.model.resolve(list(markers))
    sub = xyz.data[:, idx, :]                            # (T, N, D)

    T, N, D = sub.shape
    triples = list(combinations(range(N), 3))
    n_tri = len(triples)

    coor = np.full((T, n_tri, D, D), np.nan, dtype=np.float64)
    com  = np.full((T, n_tri, D),    np.nan, dtype=np.float64)
    imd  = np.full((T, n_tri, 2),    np.nan, dtype=np.float64)
    imo  = np.full((T, n_tri),       np.nan, dtype=np.float64)

    for n, (i, j, k) in enumerate(triples):
        # MATLAB:
        #   oriSpan = xyz(:, [k, i], :) - xyz(:, j, :)   →  (T, 2, D)
        # we follow the same pattern but keep the two arms separate.
        arm_k = sub[:, k, :] - sub[:, j, :]              # (T, D)
        arm_i = sub[:, i, :] - sub[:, j, :]              # (T, D)

        # Distance to k and to i (matches MATLAB imd[:, n, :])
        imd[:, n, 0] = np.linalg.norm(arm_k, axis=1)
        imd[:, n, 1] = np.linalg.norm(arm_i, axis=1)

        # Angle between the two arms, via atan2(|cross|, dot)
        cross = np.cross(arm_k, arm_i) if D == 3 else (arm_k[:, 0] * arm_i[:, 1] - arm_k[:, 1] * arm_i[:, 0])
        cross_mag = (np.linalg.norm(cross, axis=1) if D == 3 else np.abs(cross))
        dot = np.einsum("ti,ti->t", arm_k, arm_i)
        imo[:, n] = np.arctan2(cross_mag, dot)

        # Local frame: row 0 = arm_k; row 1 = -(arm_k × arm_i); row 2 = arm_k × row1.
        if D == 3:
            second = -np.cross(arm_k, arm_i)
            third  =  np.cross(arm_k, second)
            coor[:, n, 0, :] = arm_k
            coor[:, n, 1, :] = second
            coor[:, n, 2, :] = third
        else:
            # 2-D: only the first arm makes sense as a basis row.
            coor[:, n, 0, :] = arm_k
            coor[:, n, 1, :] = arm_i

        com[:, n, :] = sub[:, [i, j, k], :].mean(axis=1)

    @dataclass
    class _Result:
        nck:  np.ndarray
        com:  np.ndarray
        coor: np.ndarray
        imd:  np.ndarray
        imo:  np.ndarray

    return _Result(
        nck  = np.asarray(triples, dtype=np.int32),
        com  = com,
        coor = coor,
        imd  = imd,
        imo  = imo,
    )


# Public reference type for `marker_triads` callers
from dataclasses import dataclass


@dataclass
class MarkerTriadResult:
    """Output of :func:`marker_triads` (re-exported for type-checkers)."""
    nck:  np.ndarray
    com:  np.ndarray
    coor: np.ndarray
    imd:  np.ndarray
    imo:  np.ndarray


# ─────────────────────────────────────────────────────────────────── #
# Marker-difference utilities                                          #
# ─────────────────────────────────────────────────────────────────── #

def marker_diff_matrix(xyz: NBDxyz) -> np.ndarray:
    """Pairwise per-frame marker-difference tensor.

    Port of :file:`MTA/utilities/markerDiffMatrix.m`.

    For every pair of markers (i, j) and every spatial dim d, returns
    ``xyz[t, j, d] - xyz[t, i, d]`` — i.e. the position of marker j
    expressed in marker i's frame of reference, minus marker i's
    own position.

    Returns
    -------
    diff : np.ndarray, shape ``(T, n_markers, n_markers, n_dims)``
        ``diff[t, i, j, :]`` is ``xyz[t, j] - xyz[t, i]``.
    """
    if xyz._data is None:
        raise RuntimeError("xyz data is not loaded.")
    data = xyz.data
    # diff[t, i, j, d] = data[t, j, d] - data[t, i, d]
    return data[:, None, :, :] - data[:, :, None, :]


def inter_marker_distance(xyz: NBDxyz) -> np.ndarray:
    """Per-frame Euclidean distance between every pair of markers.

    Port of :file:`MTA/utilities/transforms/interMarkerDistance.m`.

    Returns
    -------
    dist : np.ndarray, shape ``(T, n_markers, n_markers)``
        ``dist[t, i, j]`` is ``|xyz[t, j] - xyz[t, i]|``.

    Notes
    -----
    Unlike :func:`intermarker_distances` (which returns the
    upper-triangular ``(T, N(N-1)/2)`` form), this returns the full
    symmetric ``(T, N, N)`` distance tensor.  Use this when you need
    to look up by marker-pair indices; use ``intermarker_distances``
    when you want a flat per-pair series.
    """
    diff = marker_diff_matrix(xyz)
    return np.sqrt(np.sum(diff * diff, axis=-1))


def inter_marker_angles(xyz: NBDxyz) -> np.ndarray:
    """Per-frame triple-marker angle tensor.

    Port of :file:`MTA/utilities/transforms/interMarkerAngles.m`.

    For every ordered triple ``(i, j, k)`` returns the angle at vertex
    *i* between the rays ``i → j`` and ``i → k`` (in radians, using
    ``atan2`` for full quadrant resolution).

    Returns
    -------
    angles : np.ndarray, shape ``(T, n_markers, n_markers, n_markers)``
        ``angles[t, i, j, k]`` is the angle at vertex *i*.

    Notes
    -----
    Memory: this is ``T × N³`` — a single 8-marker session at 250 Hz
    over 1 hour is ~290 MB.  For sparse use-cases call
    :func:`marker_triads` (round 11) instead.
    """
    diff = marker_diff_matrix(xyz)                  # (T, N, N, 3)
    # diff[t, i, j] is vector i → j; we want angles at vertex i between
    # vectors (i → j) and (i → k), so we need to broadcast:
    #   v1 = diff[t, i, j]
    #   v2 = diff[t, i, k]
    # angles[t, i, j, k] = atan2(|v1 × v2|, v1 · v2)
    v1 = diff[:, :, :, None, :]                     # (T, N, N, 1, 3)
    v2 = diff[:, :, None, :, :]                     # (T, N, 1, N, 3)
    cross = np.cross(v1, v2)                        # (T, N, N, N, 3)
    cross_mag = np.linalg.norm(cross, axis=-1)
    dot = np.sum(v1 * v2, axis=-1)
    return np.arctan2(cross_mag, dot)


def inter_marker_orientation(xyz: NBDxyz) -> np.ndarray:
    """Per-frame 4-marker orientation tensor.

    Port of :file:`MTA/utilities/transforms/interMarkerOrientation.m`.

    For every ordered 4-tuple ``(i, j, k, l)`` returns the angle
    between the normal of the (i,j,k) plane and the vector ``i → l``.
    Useful for measuring whether marker *l* lies above or below the
    plane defined by the other three.

    Returns
    -------
    angles : np.ndarray, shape ``(T, n_markers, n_markers, n_markers, n_markers)``

    Memory warning
    --------------
    This is ``T × N⁴``.  For an 8-marker session at 1 hour @ 250 Hz
    that's already 1.2 GB; for any reasonable rigid body do **not**
    call this on the full session.  Slice xyz with a state mask first
    or compute only the specific 4-tuples you need by hand.
    """
    diff = marker_diff_matrix(xyz)                  # (T, N, N, 3)
    # cross[t, i, j, k] = (i → j) × (i → k)
    v1 = diff[:, :, :, None, :]                     # (T, N, N, 1, 3)
    v2 = diff[:, :, None, :, :]                     # (T, N, 1, N, 3)
    cross = np.cross(v1, v2)                        # (T, N, N, N, 3)
    # Reshape to broadcast against (i → l)
    cross_e = cross[:, :, :, :, None, :]            # (T, N, N, N, 1, 3)
    v3      = diff[:, :, None, None, :, :]          # (T, N, 1, 1, N, 3)
    inner_cross = np.cross(cross_e, v3)             # (T, N, N, N, N, 3)
    inner_mag   = np.linalg.norm(inner_cross, axis=-1)
    inner_dot   = np.sum(cross_e * v3, axis=-1)
    return np.arctan2(inner_mag, inner_dot)
