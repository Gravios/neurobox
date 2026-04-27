"""
transform_origin.py  —  Egocentric coordinate-frame transformation
====================================================================

Port of ``MTASession.transform_origin`` and its utilities
(``rotZAxis``, ``rotYAxis``, ``detectRoll``).

Transforms marker-difference vectors from the lab frame into the
animal's egocentric frame by applying three sequential rotations:

1. **Yaw removal** (rotZAxis) — rotate around the Z-axis so that the
   head-front → head-back orientation vector lies along the +X axis.
2. **Pitch removal** (rotYAxis) — rotate around the Y-axis so that
   the orientation vector is horizontal (z = 0).
3. **Roll removal** (detectRoll / rotXAxis) — rotate around the X-axis
   to level the lateral markers.

The function returns a :class:`TransformResult` struct with the
continuous yaw (``direction``), pitch, and roll angle time-series,
plus the transformed coordinates of any additional marker vectors
specified in ``vector_markers``.

Usage
-----
::

    from neurobox.analysis.transform_origin import transform_origin

    result = transform_origin(
        xyz,
        origin_marker      = "head_back",
        orientation_marker = "head_front",
        vector_markers     = ["head_left", "head_right"],
    )

    # Continuous head direction angle (yaw, radians)
    yaw   = result.direction   # (T,)

    # Pitch and roll angles
    pitch = result.pitch       # (T,)
    roll  = result.roll        # (T,)

    # Transformed coordinates of the lateral markers
    coords = result.trans_coords   # (T, n_vector_markers, 3)

Notes
-----
The MATLAB implementation loop-expands the rotation matrices via
``repmat`` + ``permute`` + ``shiftdim``; the numpy port achieves the
same via batched matrix multiplication using ``np.einsum``.

Coordinate convention: positions in mm, angles in radians,
time as the first axis.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class TransformResult:
    """Results from :func:`transform_origin`.

    Attributes
    ----------
    direction : np.ndarray, shape (T,)
        Continuous yaw angle of the orientation vector (radians).
        Positive = counter-clockwise from +X axis.
    pitch : np.ndarray, shape (T,)
        Pitch angle (elevation above horizontal, radians).
        Positive = nose up.
    roll : np.ndarray, shape (T,)
        Roll angle (rotation around the naso-occipital axis, radians).
        Only populated when *vector_markers* contains at least one
        lateral marker pair; otherwise all zeros.
    ori_vector : np.ndarray, shape (T, 3)
        Fully de-rotated orientation vector (should lie along +X after
        yaw + pitch removal).
    trans_coords : np.ndarray | None, shape (T, N_vecs, 3)
        Egocentric coordinates of the *vector_markers* after all three
        rotations.  ``None`` when no vector markers were requested.
    """
    direction:    np.ndarray
    pitch:        np.ndarray
    roll:         np.ndarray
    ori_vector:   np.ndarray
    trans_coords: "np.ndarray | None" = None


# ---------------------------------------------------------------------------
# Internal rotation helpers
# ---------------------------------------------------------------------------

def _rot_z_axis(vec: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Remove yaw from a (T, 3) vector by rotating around Z.

    Ports ``rotZAxis.m``.

    Returns
    -------
    rz : (T, 3)   — rotated vector
    r_mat : (T, 3, 3) — rotation matrices (one per frame)
    theta : (T,)  — yaw angles in radians
    """
    T = vec.shape[0]
    r_mat = np.zeros((T, 3, 3))

    xy_len = np.sqrt(vec[:, 0] ** 2 + vec[:, 1] ** 2)
    # Guard against zero-length projection
    safe = xy_len > 0
    theta = np.where(safe, np.arccos(np.clip(np.abs(vec[:, 0]) / np.where(safe, xy_len, 1.0), -1.0, 1.0)), 0.0)

    # Quadrant correction: if x < 0 the angle wraps around pi
    theta = np.where(vec[:, 0] < 0, np.abs(theta - np.pi), theta)

    # Clockwise flag: positive y → counter-clockwise rotation
    cw = vec[:, 1] >= 0   # (T,)

    c, s = np.cos(theta), np.sin(theta)
    z = np.zeros(T)
    o = np.ones(T)

    # CW frames: standard Z-rotation
    # CW (y >= 0): rotate clockwise around Z to align with +x
    r_mat[cw] = np.stack([
        c[cw],   s[cw], z[cw],
       -s[cw],   c[cw], z[cw],
        z[cw],   z[cw], o[cw],
    ], axis=1).reshape(-1, 3, 3)

    # CCW (y < 0): rotate counter-clockwise
    ccw = ~cw
    r_mat[ccw] = np.stack([
        c[ccw],  -s[ccw], z[ccw],
        s[ccw],   c[ccw], z[ccw],
        z[ccw],   z[ccw], o[ccw],
    ], axis=1).reshape(-1, 3, 3)

    # Apply: (T,3,3) @ (T,3,1) → (T,3)
    rz = np.einsum("tij,tj->ti", r_mat, vec)

    # Sign-encode direction: negative y → negative theta
    theta = np.where(vec[:, 1] < 0, -theta, theta)

    return rz, r_mat, theta


def _rot_y_axis(vec: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Remove pitch from a (T, 3) yaw-corrected vector by rotating around Y.

    Ports ``rotYAxis.m``.

    Returns
    -------
    ry : (T, 3)
    r_mat : (T, 3, 3)
    theta : (T,)   pitch angles in radians
    """
    T = vec.shape[0]
    r_mat = np.zeros((T, 3, 3))

    xz_len = np.sqrt(vec[:, 0] ** 2 + vec[:, 2] ** 2)
    safe = xz_len > 0
    theta = np.where(
        safe,
        np.abs(
            np.arccos(np.clip(
                np.abs(vec[:, 2]) / np.where(safe, xz_len, 1.0), -1.0, 1.0
            )) - np.pi / 2
        ),
        0.0,
    )

    cw = vec[:, 2] > 0   # positive z → clockwise rotation around Y
    c, s = np.cos(theta), np.sin(theta)
    z = np.zeros(T)
    o = np.ones(T)

    # CW (z > 0): rotate to bring z-component to zero
    r_mat[cw] = np.stack([
        c[cw],  z[cw],  s[cw],
        z[cw],  o[cw],  z[cw],
       -s[cw],  z[cw],  c[cw],
    ], axis=1).reshape(-1, 3, 3)

    ccw = ~cw
    r_mat[ccw] = np.stack([
        c[ccw],  z[ccw], -s[ccw],
        z[ccw],  o[ccw],  z[ccw],
        s[ccw],  z[ccw],  c[ccw],
    ], axis=1).reshape(-1, 3, 3)

    ry = np.einsum("tij,tj->ti", r_mat, vec)
    theta = np.where(vec[:, 2] < 0, -theta, theta)

    return ry, r_mat, theta


def _detect_roll(vec_set: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Remove roll from a (T, N_vecs, 3) array of transformed vectors.

    Ports ``detectRoll.m``.

    Returns
    -------
    coords : (T, N_vecs, 3) — de-rolled coordinates
    roll   : (T,)           — roll angles from the *first* vector
    """
    T, N, _ = vec_set.shape
    coords    = np.zeros_like(vec_set)
    roll_all  = np.zeros(T)

    for i in range(N):
        v = vec_set[:, i, :]            # (T, 3)
        yz_len = np.sqrt(v[:, 1] ** 2 + v[:, 2] ** 2)
        safe   = yz_len > 0
        roll   = np.where(
            safe,
            np.arccos(np.clip(np.abs(v[:, 1]) / np.where(safe, yz_len, 1.0), -1.0, 1.0)),
            0.0,
        )
        roll = np.where(v[:, 2] > 0, -roll, roll)
        # MTA quirk: first vector sign is flipped
        if i == 0:
            roll    = -roll
            roll_all = roll

        c, s = np.cos(roll), np.sin(roll)
        z    = np.zeros(T)
        o    = np.ones(T)

        r_mat = np.stack([
            o,  z,   z,
            z,  c,  -s,
            z,  s,   c,
        ], axis=1).reshape(T, 3, 3)

        coords[:, i, :] = np.einsum("tij,tj->ti", r_mat, v)

    return coords, roll_all


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def transform_origin(
    xyz,
    origin_marker:      str       = "head_back",
    orientation_marker: str       = "head_front",
    vector_markers:     "list[str] | None" = None,
    low_pass_hz:        "float | None"     = 50.0,
    low_pass_order:     int                = 3,
) -> TransformResult:
    """Transform marker trajectories into the animal's egocentric frame.

    Port of ``MTASession.transform_origin``.

    The egocentric frame is defined by the vector from *origin_marker*
    toward *orientation_marker* (typically head_back → head_front):

    * **+X** points in the direction of heading.
    * **Z** is the world vertical (unchanged by yaw removal).
    * After pitch removal the orientation vector is horizontal.

    Parameters
    ----------
    xyz:
        Loaded :class:`~neurobox.dtype.xyz.NBDxyz` object.
    origin_marker:
        Marker at the reference origin (default ``'head_back'``).
    orientation_marker:
        Marker defining the forward direction (default ``'head_front'``).
    vector_markers:
        Additional markers to transform into the egocentric frame.
        Typically the lateral head markers — ``['head_left', 'head_right']``
        — used for roll removal.  ``None`` → no additional transforms.
    low_pass_hz:
        Low-pass filter cut-off in Hz applied to *xyz* before computing
        angles.  ``None`` → no filtering.  Default 50 Hz.
    low_pass_order:
        Butterworth filter order (default 3).

    Returns
    -------
    :class:`TransformResult`

    Examples
    --------
    ::

        from neurobox.analysis.transform_origin import transform_origin

        result = transform_origin(
            trial.xyz,
            vector_markers=["head_left", "head_right"],
        )

        # Head direction (yaw) and pitch time-series
        yaw   = result.direction   # radians, (T,)
        pitch = result.pitch       # radians, (T,)
        roll  = result.roll        # radians, (T,)
    """
    if xyz.data is None:
        raise RuntimeError("xyz data is not loaded.")

    if vector_markers is None:
        vector_markers = []

    # ── Optionally low-pass filter a working copy ─────────────────────── #
    if low_pass_hz is not None:
        work = xyz.copy()
        work.filter("butter", cutoff=low_pass_hz, order=low_pass_order,
                    btype="low")
    else:
        work = xyz

    # ── Select the required markers ───────────────────────────────────── #
    all_markers = [origin_marker, orientation_marker] + list(vector_markers)
    sub = work.subset(all_markers)                   # (T, M, 3)
    d   = sub.data.astype(np.float64)                # (T, M, 3)
    T, M, _ = d.shape

    # Indices into the sub-array
    idx_origin = 0
    idx_ori    = 1
    idx_vecs   = list(range(2, M))

    # ── Difference matrix: vec[i→j] = pos_j - pos_i ───────────────────── #
    # We only need origin→orientation and origin→vector_markers
    ori_vec = d[:, idx_ori,  :] - d[:, idx_origin, :]   # (T, 3)

    # ── Stage 1: yaw removal ─────────────────────────────────────────── #
    rz, rz_mat, direction = _rot_z_axis(ori_vec)

    # ── Stage 2: pitch removal ────────────────────────────────────────── #
    ry, ry_mat, pitch = _rot_y_axis(rz)

    # ── Stage 3: transform additional markers + roll removal ─────────── #
    trans_coords = None
    roll         = np.zeros(T)

    if idx_vecs:
        vec_set = np.zeros((T, len(idx_vecs), 3))
        for col, vi in enumerate(idx_vecs):
            v = d[:, vi, :] - d[:, idx_origin, :]
            # Apply yaw rotation
            v_rz = np.einsum("tij,tj->ti", rz_mat, v)
            # Apply pitch rotation
            v_ry = np.einsum("tij,tj->ti", ry_mat, v_rz)
            vec_set[:, col, :] = v_ry

        trans_coords, roll = _detect_roll(vec_set)

    return TransformResult(
        direction    = direction,
        pitch        = pitch,
        roll         = roll,
        ori_vector   = ry,
        trans_coords = trans_coords,
    )
