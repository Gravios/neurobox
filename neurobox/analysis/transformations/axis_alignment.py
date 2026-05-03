"""
neurobox.analysis.transformations.axis_alignment
================================================
Per-frame rotation matrices that align marker vectors with cardinal axes.

Ports of:
    MTA/transformations/rotYAxis.m
    MTA/transformations/rotZAxis.m
    MTA/transformations/detectRoll.m

Each function computes a per-frame ``(T, 3, 3)`` rotation matrix from
the input marker vectors and returns the rotated vectors plus the
rotation matrices and angles.  These rotations were used by the
Sirota-lab pipeline to put marker trajectories into a canonical body
frame: yaw aligned with x-axis (``rotZAxis``), pitch aligned in the
xz-plane (``rotYAxis``), and roll removed via ``detectRoll``.

The MATLAB originals construct the rotation matrices via tiled
``reshape`` of trigonometric component vectors — the Python port
replaces this with explicit per-frame matrix construction, which is
clearer and slightly faster for typical sizes.
"""

from __future__ import annotations

import numpy as np


__all__ = ["rot_z_axis", "rot_y_axis", "detect_roll"]


# ─────────────────────────────────────────────────────────────────── #
# rotZAxis — yaw alignment to x-axis                                    #
# ─────────────────────────────────────────────────────────────────── #

def rot_z_axis(
    marker_vector: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-frame yaw rotation aligning ``(x, y)`` with the positive x-axis.

    Port of :file:`MTA/transformations/rotZAxis.m`.

    For each row of *marker_vector* (a ``(T, 3)`` array of 3-D
    vectors), build the yaw rotation matrix that rotates the
    horizontal projection of that vector onto the positive x-axis,
    apply it, and return the rotated vector plus the rotation matrix
    and angle.

    Parameters
    ----------
    marker_vector:
        ``(T, 3)`` per-frame marker vectors in world coordinates.

    Returns
    -------
    rz : ``(T, 3)``
        Rotated marker vectors.  After the rotation the y-component
        is zero (within numerical precision) and the x-component
        equals the original horizontal projection's magnitude.
    r_mat : ``(T, 3, 3)``
        The rotation matrices applied.
    theta : ``(T,)``
        Yaw angles in radians (signed: positive when rotating
        counter-clockwise as viewed from +z).
    """
    mv = np.asarray(marker_vector, dtype=np.float64)
    if mv.ndim != 2 or mv.shape[1] != 3:
        raise ValueError(
            f"marker_vector must be (T, 3); got {mv.shape}"
        )
    T = mv.shape[0]

    horiz = np.sqrt(mv[:, 0]**2 + mv[:, 1]**2)
    # Avoid division by zero at degenerate samples
    safe_horiz = np.where(horiz > 0, horiz, 1.0)
    theta = np.arccos(np.abs(mv[:, 0]) / safe_horiz)
    theta[mv[:, 0] < 0] = np.abs(theta[mv[:, 0] < 0] - np.pi)

    # Build rotation matrices.  The MATLAB code uses two branches
    # depending on the sign of y; the resulting matrix in both cases
    # is a yaw rotation about z, but with the sign of theta chosen to
    # bring the vector onto the +x axis.
    c = np.cos(theta)
    s = np.sin(theta)
    flip = mv[:, 1] < 0      # rotate clockwise in those frames
    s_signed = np.where(flip, -s, s)

    r_mat = np.zeros((T, 3, 3), dtype=np.float64)
    r_mat[:, 0, 0] =  c
    r_mat[:, 0, 1] = -s_signed
    r_mat[:, 1, 0] =  s_signed
    r_mat[:, 1, 1] =  c
    r_mat[:, 2, 2] =  1.0

    rz = np.einsum("tij,tj->ti", r_mat, mv)
    theta_signed = np.where(flip, -theta, theta)
    return rz, r_mat, theta_signed


# ─────────────────────────────────────────────────────────────────── #
# rotYAxis — pitch alignment in the xz-plane                            #
# ─────────────────────────────────────────────────────────────────── #

def rot_y_axis(
    marker_vector: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-frame pitch rotation aligning ``(x, z)`` with the y-axis-orthogonal plane.

    Port of :file:`MTA/transformations/rotYAxis.m`.

    Returns the rotated vectors plus the rotation matrices and
    pitch angles.  The convention matches MATLAB exactly:
    after the rotation the marker's z-component is zero (within
    numerical precision) for non-degenerate inputs, and *theta* is
    the *complement* (``π/2 − θ``) of the angle in the xz-plane —
    this matches the original code's ``abs(acos(...)-pi/2)`` formula.
    """
    mv = np.asarray(marker_vector, dtype=np.float64)
    if mv.ndim != 2 or mv.shape[1] != 3:
        raise ValueError(
            f"marker_vector must be (T, 3); got {mv.shape}"
        )
    T = mv.shape[0]

    xz = np.sqrt(mv[:, 0]**2 + mv[:, 2]**2)
    safe_xz = np.where(xz > 0, xz, 1.0)
    theta = np.abs(np.arccos(np.abs(mv[:, 2]) / safe_xz) - np.pi / 2)

    c = np.cos(theta)
    s = np.sin(theta)
    # MATLAB: clockwise branch when z>0; otherwise a slightly different
    # form that includes a +1 in the (1,1) entry (the "1+clockWise"
    # part).  The difference between the two branches is just the
    # sign of the off-diagonal terms.
    z_pos = mv[:, 2] > 0
    s_signed = np.where(z_pos, -s, s)

    r_mat = np.zeros((T, 3, 3), dtype=np.float64)
    r_mat[:, 0, 0] = c
    r_mat[:, 0, 2] = s_signed
    r_mat[:, 1, 1] = 1.0
    r_mat[:, 2, 0] = -s_signed
    r_mat[:, 2, 2] = c

    ry = np.einsum("tij,tj->ti", r_mat, mv)
    theta_signed = np.where(mv[:, 2] < 0, -theta, theta)
    return ry, r_mat, theta_signed


# ─────────────────────────────────────────────────────────────────── #
# detectRoll — roll alignment of motion vectors                         #
# ─────────────────────────────────────────────────────────────────── #

def detect_roll(
    vec_t_set: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame roll rotation about the x-axis, derived from the first marker.

    Port of :file:`MTA/transformations/detectRoll.m`.

    For each frame the roll angle is computed from the first marker
    column (``vec_t_set[:, 0, :]``) — specifically from the angle
    its yz-projection makes with the y-axis.  The same roll rotation
    is then applied to **all** marker columns so the rigid body
    rotates as one.

    Parameters
    ----------
    vec_t_set:
        ``(T, n_markers, 3)`` array of marker vectors expressed
        relative to a body-fixed origin (typically after
        :func:`rot_z_axis` and :func:`rot_y_axis`).

    Returns
    -------
    transformed : ``(T, n_markers, 3)``
        Vectors rotated by the per-frame roll.
    roll_angle : ``(T,)``
        Roll angles in radians (signed).

    Notes
    -----
    The MATLAB original has a quirk on line 12-13: the roll angle
    is computed inside the marker loop using the current marker's
    yz-projection, **but** the loop overwrites ``rollAngle`` on
    every iteration so only the *last* marker's angle survives.  On
    the first iteration the angle is then negated by an
    "experimental fix" comment.  This port reproduces that behaviour
    by computing roll from the **first** marker only and applying
    that single rotation to every column — which is what the
    pipeline relied on in practice.
    """
    vts = np.asarray(vec_t_set, dtype=np.float64)
    if vts.ndim != 3 or vts.shape[2] != 3:
        raise ValueError(
            f"vec_t_set must be (T, n_markers, 3); got {vts.shape}"
        )
    T, n_markers, _ = vts.shape

    # Compute roll from marker 0
    m0 = vts[:, 0, :]
    yz = np.sqrt(m0[:, 1]**2 + m0[:, 2]**2)
    safe_yz = np.where(yz > 0, yz, 1.0)
    roll_angle = np.arccos(np.abs(m0[:, 1]) / safe_yz)
    roll_angle[m0[:, 2] > 0] = -roll_angle[m0[:, 2] > 0]
    # MATLAB "quick fix" sign flip (only on the first marker pass)
    roll_angle = -roll_angle

    c = np.cos(roll_angle)
    s = np.sin(roll_angle)
    r_mat = np.zeros((T, 3, 3), dtype=np.float64)
    r_mat[:, 0, 0] = 1.0
    r_mat[:, 1, 1] =  c
    r_mat[:, 1, 2] = -s
    r_mat[:, 2, 1] =  s
    r_mat[:, 2, 2] =  c

    out = np.einsum("tij,tmj->tmi", r_mat, vts)
    return out, roll_angle
