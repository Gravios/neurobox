"""
neurobox.analysis.mocap.rotations
==================================
Axis-angle (Rodrigues) rotations applied per time-frame to batches of
3-D points.

Three convenience layers, mirroring the MTA originals:

* :func:`rotate_points_around_vectors`
  Pure tensor primitive — given ``(T, 3)`` points, ``(T, 3)`` rotation
  axes (unit vectors), and a scalar angle, return ``(T, 3)`` rotated
  points.  Port of :file:`MTA/utilities/mocap/rotate_points_around_vectors.m`.

* :func:`rotate_point_around_vector`
  Convenience: rotate one named marker about a normal computed from
  the cross-product of two reference markers (relative to ``hcom``).
  Port of :file:`MTA/utilities/mocap/rotate_point_around_vector.m`.

* :func:`rotate_marker_around_vector`
  Like ``rotate_point_around_vector`` but with an explicit origin
  marker (rather than hardcoded ``hcom``).  Port of
  :file:`MTA/utilities/mocap/rotate_marker_around_vector.m`.

The MATLAB cross-product-matrix construction
(``head_cpm = reshape(head_norm(:, k)', ...) .* j``) is replaced by an
explicit, readable build of the per-frame ``[n]×`` matrix.

Important angle-unit difference
-------------------------------
MATLAB's ``rotate_point_around_vector`` takes an angle in **degrees**;
``rotate_marker_around_vector`` takes an angle in **radians**.  This
port aligns on **radians** for both — ``rotate_point_around_vector``
takes ``angle_deg`` as a separate keyword for backward call-site
compatibility but uses ``angle`` (radians) by default.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from neurobox.dtype.xyz import NBDxyz


# ─────────────────────────────────────────────────────────────────────────── #
# Pure tensor primitive                                                        #
# ─────────────────────────────────────────────────────────────────────────── #

def _cross_product_matrices(axes: np.ndarray) -> np.ndarray:
    """Build per-frame ``[n]×`` cross-product matrices.

    Given ``(T, 3)`` axis vectors, return ``(T, 3, 3)`` skew-symmetric
    matrices such that ``([n]× v) == cross(n, v)`` for every frame.
    """
    nx, ny, nz = axes[:, 0], axes[:, 1], axes[:, 2]
    Z = np.zeros_like(nx)
    K = np.stack([
        np.stack([ Z, -nz,  ny], axis=-1),
        np.stack([ nz,  Z, -nx], axis=-1),
        np.stack([-ny, nx,   Z], axis=-1),
    ], axis=-2)                                            # (T, 3, 3)
    return K


def rotate_points_around_vectors(
    points: np.ndarray,
    axes:   np.ndarray,
    angle:  float,
) -> np.ndarray:
    """Rotate each of ``T`` points around its corresponding axis.

    Port of :file:`MTA/utilities/mocap/rotate_points_around_vectors.m`.

    Parameters
    ----------
    points:
        ``(T, 3)`` points to rotate.  Must be expressed in a frame
        whose origin is on the axis of rotation (typically already
        translated relative to that origin).
    axes:
        ``(T, 3)`` rotation axes.  Should be unit vectors;
        non-unit-norm axes will give a non-isometric transform.  The
        MATLAB original silently assumes unit norm — this port keeps
        that contract.
    angle:
        Rotation angle in **radians**.  Same scalar applied across
        all frames.  (For per-frame angles, call this in a loop —
        the MATLAB original did not support varying angles either.)

    Returns
    -------
    rotated : np.ndarray, shape ``(T, 3)``
    """
    points = np.asarray(points, dtype=np.float64)
    axes   = np.asarray(axes,   dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must be (T, 3); got {points.shape}")
    if axes.shape != points.shape:
        raise ValueError(
            f"axes shape {axes.shape} must match points shape {points.shape}"
        )

    T = points.shape[0]
    c = np.cos(angle)
    s = np.sin(angle)

    # Rodrigues:  R = cos·I + sin·[n]× + (1-cos)·n nᵀ
    K       = _cross_product_matrices(axes)             # (T, 3, 3)
    n_outer = np.einsum("ti,tj->tij", axes, axes)        # (T, 3, 3)
    eye     = np.eye(3)[None, :, :]                      # (1, 3, 3)
    R       = c * eye + s * K + (1.0 - c) * n_outer       # (T, 3, 3)

    return np.einsum("tij,tj->ti", R, points)


# ─────────────────────────────────────────────────────────────────────────── #
# Marker-based wrappers                                                        #
# ─────────────────────────────────────────────────────────────────────────── #

def _normal_from_two_vectors(
    xyz_data: np.ndarray,
    idx_origin: int,
    idx_a: int,
    idx_b: int,
) -> np.ndarray:
    """Build a per-frame normal vector via cross product.

    Returns the unit cross-product of ``(a - origin)`` and ``(b - origin)``.
    Frames where the cross product is zero (collinear) get a zero normal
    — consistent with the MATLAB ``multiprod(..., 1/sqrt(...))`` pattern,
    which silently produced ``inf`` in those cases.  Here we keep zeros
    so downstream rotations become identity at those frames.
    """
    a = xyz_data[:, idx_a, :] - xyz_data[:, idx_origin, :]
    b = xyz_data[:, idx_b, :] - xyz_data[:, idx_origin, :]
    n = np.cross(a, b)
    norm = np.linalg.norm(n, axis=1, keepdims=True)
    out = np.zeros_like(n)
    np.divide(n, norm, out=out, where=norm > 0)
    return out


def rotate_point_around_vector(
    xyz:           NBDxyz,
    marker:        str = "hbx",
    angle:         float | None = None,
    angle_deg:     float | None = None,
    ref_markers:   tuple[str, str] = ("hbx", "hrx"),
    origin_marker: str = "hcom",
) -> np.ndarray:
    """Rotate one named marker about a normal computed from two reference markers.

    Port of :file:`MTA/utilities/mocap/rotate_point_around_vector.m`.

    The rotation axis is the unit cross-product of
    ``ref_markers[0] - origin_marker`` and ``ref_markers[1] - origin_marker``.
    The rotation is applied to ``marker - origin_marker``, then the
    origin is added back.

    Parameters
    ----------
    xyz:
        Source position data.  Must contain *marker*, *origin_marker*,
        and both *ref_markers*.
    marker:
        Name of the marker to rotate.
    angle:
        Rotation angle in **radians**.  Either *angle* or *angle_deg*
        must be provided.
    angle_deg:
        Alternative spelling, in degrees — matches the MATLAB
        signature where the angle was in degrees.  Converted to
        radians internally.  ``angle_deg=45`` matches the MATLAB
        default.
    ref_markers:
        Two markers whose positions (relative to *origin_marker*)
        define the rotation axis via cross product.
    origin_marker:
        Marker treated as the rotation centre.  Default ``'hcom'``.

    Returns
    -------
    new_positions : np.ndarray, shape ``(T, 3)``
        New world-space positions for *marker* after rotation.
        Frames where the source xyz fails the
        :func:`finite_nonzero_mask` check pass through unchanged.
    """
    if angle is None and angle_deg is None:
        raise ValueError("Specify either angle (rad) or angle_deg (deg).")
    if angle is None:
        angle = float(np.deg2rad(angle_deg))

    if xyz._data is None:
        raise RuntimeError("xyz data is not loaded.")
    for n in (marker, origin_marker, *ref_markers):
        if n not in xyz.model.markers:
            raise ValueError(
                f"rotate_point_around_vector: xyz has no {n!r} marker."
            )

    data = xyz._data
    idx_marker = xyz.model.index(marker)
    idx_origin = xyz.model.index(origin_marker)
    idx_a = xyz.model.index(ref_markers[0])
    idx_b = xyz.model.index(ref_markers[1])

    origin_pos = data[:, idx_origin, :]                # (T, 3)
    marker_pos = data[:, idx_marker, :]                # (T, 3)
    pts        = marker_pos - origin_pos
    normals    = _normal_from_two_vectors(data, idx_origin, idx_a, idx_b)

    # Validity mask matching MATLAB nniz: rotate only where xyz is valid.
    from neurobox.analysis.kinematics.helpers import finite_nonzero_mask
    valid = finite_nonzero_mask(data.reshape(data.shape[0], -1))

    out = marker_pos.copy()
    if valid.any():
        rotated = rotate_points_around_vectors(pts[valid], normals[valid], angle)
        out[valid, :] = rotated + origin_pos[valid, :]
    return out


def rotate_marker_around_vector(
    xyz:           NBDxyz,
    marker:        str = "head_right",
    angle:         float = np.pi,
    origin_marker: str = "hcom",
    ref_markers:   tuple[str, str] = ("head_back", "head_right"),
) -> np.ndarray:
    """Like :func:`rotate_point_around_vector`, with explicit origin.

    Port of :file:`MTA/utilities/mocap/rotate_marker_around_vector.m`.

    The MATLAB original takes the angle in **radians** (the other
    function takes degrees) — this port matches.  ``angle=pi``
    (180°) matches the MATLAB default.

    Parameters and return value are otherwise identical to
    :func:`rotate_point_around_vector`.
    """
    return rotate_point_around_vector(
        xyz,
        marker        = marker,
        angle         = float(angle),
        ref_markers   = ref_markers,
        origin_marker = origin_marker,
    )
