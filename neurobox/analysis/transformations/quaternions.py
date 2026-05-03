"""
neurobox.analysis.transformations.quaternions
==============================================
Quaternion ↔ rotation-matrix and Euler-angle conversions.

Ports of:
    MTA/utilities/transforms/quat2rotm.m
    MTA/utilities/transforms/quaternion2rad.m

Quaternion convention
---------------------
Both functions operate on quaternions in the ``(w, x, y, z)`` order
(scalar-first), matching the MATLAB code's column convention where
``q(:,1)`` is the scalar part.  The MATLAB code uses 1-based
indexing so ``q(:,1)..q(:,4)`` corresponds to ``(w, x, y, z)`` in
this port.

OptiTrack Motive emits quaternions in ``(x, y, z, w)`` order
(see :mod:`neurobox.analysis.mocap.motive_csv`).  Reorder them
before calling these functions::

    motive = parse_rbo_from_csv(...)        # last 4 channels: qa qb qc qd
    qxyzw  = motive.rbo_data[:, :, 4:8]     # (T, n_bodies, 4) = (qx,qy,qz,qw)
    qwxyz  = qxyzw[..., [3, 0, 1, 2]]
"""

from __future__ import annotations

import numpy as np


__all__ = ["quat2rotm", "quaternion2rad"]


def quat2rotm(quat: np.ndarray) -> np.ndarray:
    """Convert a batch of quaternions ``(w, x, y, z)`` to rotation matrices.

    Port of :file:`MTA/utilities/transforms/quat2rotm.m`.

    Parameters
    ----------
    quat:
        ``(N, 4)`` array of quaternions in scalar-first ``(w, x, y, z)``
        order.

    Returns
    -------
    rotm : ``(N, 3, 3)``
        Per-row rotation matrices.

    Notes
    -----
    The MATLAB original is the second (un-commented) variant in
    quat2rotm.m — the comment block at the top contains an
    alternative formula with sign differences which is not used.
    This port matches the live (un-commented) formula exactly.
    """
    q = np.asarray(quat, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != 4:
        raise ValueError(f"quat must be (N, 4); got {q.shape}")
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    rotm = np.empty((q.shape[0], 3, 3), dtype=np.float64)
    rotm[:, 0, 0] = 1 - 2 * (y * y + z * z)
    rotm[:, 0, 1] = 2 * (x * y - z * w)
    rotm[:, 0, 2] = 2 * (x * z + y * w)
    rotm[:, 1, 0] = 2 * (x * y + z * w)
    rotm[:, 1, 1] = 1 - 2 * (x * x + z * z)
    rotm[:, 1, 2] = 2 * (y * z - x * w)
    rotm[:, 2, 0] = 2 * (x * z - y * w)
    rotm[:, 2, 1] = 2 * (y * z + x * w)
    rotm[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return rotm


def quaternion2rad(quat: np.ndarray) -> np.ndarray:
    """Convert quaternions ``(w, x, y, z)`` to Euler ``(yaw, pitch, roll)`` (radians).

    Port of :file:`MTA/utilities/transforms/quaternion2rad.m`.

    Parameters
    ----------
    quat:
        ``(N, 4)`` array, scalar-first.

    Returns
    -------
    eAng : ``(N, 3)``
        Columns are ``[yaw, pitch, roll]`` in radians.  Pitch is
        clamped to ``π/2`` where the asin argument exceeds 1
        (matches MATLAB).

    Notes
    -----
    The exact convention is the MATLAB intrinsic ZYX Tait-Bryan
    decomposition used in the lab pipeline.  Other quaternion
    libraries (``scipy.spatial.transform.Rotation``) use different
    axis conventions; round-tripping via Rotation will not give bit-
    identical results.
    """
    q = np.asarray(quat, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != 4:
        raise ValueError(f"quat must be (N, 4); got {q.shape}")
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    # MATLAB: q(:,1)=w, q(:,2)=x, q(:,3)=y, q(:,4)=z
    yaw   = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    pitch = np.arcsin (2 * (w * y - z * x))
    roll  = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))

    out = np.column_stack([yaw, pitch, roll])
    # MATLAB clamp: where 2*(w*y - z*x) > 1, pitch -> π/2
    out[2 * (w * y - z * x) > 1, 1] = np.pi / 2
    return out
