"""
neurobox.analysis.kinematics.features.body_frame
=================================================
Velocity projections in body-fixed reference frames.

Each feature in this module takes the centred-difference velocity
of a marker trajectory and projects it onto a 2×2 rotation frame
defined by a marker pair (e.g. ``hcom → nose`` for the head frame,
``bcom → spine_upper`` for the body frame).  An optional rotation
``theta`` is composed onto the frame to apply a per-subject yaw
calibration (matches MATLAB's ``Trial.meta.correction.headYaw``).

Algorithm (shared by all three features)
----------------------------------------
1. Filter xyz with a Butterworth lowpass.
2. Build a (T, 2, 2) rotation matrix per frame::

       u = (xyz[:, marker_b, :2] - xyz[:, marker_a, :2]) / |·|
       Rᵢ = [u, R₉₀ u] · R(θ)

   where ``R₉₀ = [[0, -1], [1, 0]]`` rotates 90° CCW so the second
   column is the lateral (perpendicular) axis.

3. Take the centred difference of the *projection* marker's
   trajectory: ``v[t] = pos[t+1] - pos[t-1]``.

4. Project ``v`` onto the rotation frame to get the longitudinal
   and lateral velocity components.

The MATLAB scaling factor ``samplerate / 10`` (or ``samplerate / 20``
for ``fet_bref_BXY``) is preserved verbatim.  This is the
2-sample centred-difference normalisation; the divide-by-10 is
empirical (Sirota lab convention) — it converts the (mm / 2 samples)
quantity into "approximate cm/s at 250 Hz tracking".
"""

from __future__ import annotations

import numpy as np

from neurobox.analysis.lfp.filtering import butter_filter
from neurobox.dtype.fet import NBDfet
from neurobox.dtype.xyz import NBDxyz
from ..helpers import finite_nonzero_mask


# ─────────────────────────────────────────────────────────────────────────── #
# Shared helpers                                                               #
# ─────────────────────────────────────────────────────────────────────────── #

def _build_marker_pair_rotation(
    xyz_xy:    np.ndarray,
    idx_from:  int,
    idx_to:    int,
    theta:     float = 0.0,
) -> np.ndarray:
    """Return ``(T, 2, 2)`` rotation frames built from a marker pair.

    Mirrors :file:`MTA/utilities/transforms/transform_vector_to_rotation_matrix.m`.

    Parameters
    ----------
    xyz_xy:
        ``(T, n_markers, 2)`` xy-only position array.
    idx_from, idx_to:
        Column indices into ``xyz_xy`` defining the frame's primary
        axis (``to - from``).
    theta:
        Additional yaw rotation in radians.

    Returns
    -------
    rotmat : np.ndarray, shape ``(T, 2, 2)``
        Per-frame orthonormal basis.  Rows are time, first column
        of axis-0 is the longitudinal unit vector, second column
        the lateral.
    """
    vec = xyz_xy[:, idx_to, :] - xyz_xy[:, idx_from, :]   # (T, 2)
    norm = np.linalg.norm(vec, axis=1, keepdims=True)
    u_long = np.divide(vec, norm, out=np.zeros_like(vec), where=norm > 0)
    # 90° CCW rotation: [x, y] -> [-y, x]
    u_lat = np.column_stack([-u_long[:, 1], u_long[:, 0]])
    R = np.stack([u_long, u_lat], axis=2)                  # (T, 2, 2)

    if theta != 0.0:
        c, s = np.cos(theta), np.sin(theta)
        rot = np.array([[c, -s], [s, c]])                  # (2, 2)
        # Rotate each per-frame R: equivalent to MATLAB's
        # multiprod(R, [[c,-s],[s,c]], [2,3], [1,2])
        R = np.einsum("tij,jk->tik", R, rot)
    return R


def _centred_difference(traj: np.ndarray) -> np.ndarray:
    """Two-sample centred difference: ``traj[t+1] - traj[t-1]``."""
    out = np.roll(traj, -1, axis=0) - np.roll(traj, +1, axis=0)
    out[0]  = 0.0
    out[-1] = 0.0
    return out


def _project_velocity(
    xyz:           NBDxyz,
    *,
    frame_from:    str,
    frame_to:      str,
    proj_marker:   str,
    samplerate:    float | None,
    filter_cutoff: float,
    filter_order:  int,
    theta:         float,
    scale:         float,
    columns:       list[str],
    label:         str,
    name:          str,
    titles:        list[str],
    descriptions:  list[str],
    key:           str,
) -> NBDfet:
    """Common backbone for fet_href_HXY / fet_bref_BXY / fet_hvfl."""
    needed = {frame_from, frame_to, proj_marker}
    missing = [m for m in needed if m not in xyz.model.markers]
    if missing:
        raise ValueError(
            f"{label}: xyz is missing required marker(s): {missing}"
        )

    fs = float(xyz.samplerate if samplerate is None else samplerate)

    # Resample if needed
    work = xyz if (samplerate is None or samplerate == xyz.samplerate) \
           else xyz.resample(fs)

    # Zero out invalid samples (matches MATLAB nniz mask)
    raw = work.data.copy()
    bad = ~finite_nonzero_mask(raw.reshape(raw.shape[0], -1))
    raw[bad, :, :] = 0.0

    # Lowpass filter the xy data (MATLAB applies the filter to xyz.data)
    raw_xy = raw[:, :, :2].copy()
    flat = raw_xy.reshape(raw_xy.shape[0], -1)
    flat_filt = butter_filter(
        flat,
        cutoff     = filter_cutoff,
        samplerate = fs,
        order      = filter_order,
        btype      = "low",
    )
    xy = flat_filt.reshape(raw_xy.shape)

    # Rotation matrix from the marker pair
    idx_from = work.model.index(frame_from)
    idx_to   = work.model.index(frame_to)
    R = _build_marker_pair_rotation(xy, idx_from, idx_to, theta=theta)  # (T, 2, 2)

    # Centred difference of the projection marker
    idx_proj = work.model.index(proj_marker)
    v = _centred_difference(xy[:, idx_proj, :])     # (T, 2)

    # Project onto frame: out[t, k] = sum_i v[t, i] * R[t, i, k]
    proj = np.einsum("ti,tik->tk", v, R)            # (T, 2)
    proj = proj * scale * fs

    proj[bad, :] = 0.0

    return NBDfet(
        data       = proj,
        columns    = columns,
        samplerate = fs,
        label      = label,
        name       = name,
        titles     = titles,
        descriptions = descriptions,
        key        = key,
    )


# ─────────────────────────────────────────────────────────────────────────── #
# fet_href_HXY                                                                 #
# ─────────────────────────────────────────────────────────────────────────── #

def fet_href_HXY(
    xyz:           NBDxyz,
    samplerate:    float | None = None,
    filter_cutoff: float = 4.0,
    filter_order:  int = 4,
    theta:         float = 0.0,
) -> NBDfet:
    """Head-COM velocity projected onto the head-fixed frame.

    Port of :file:`MTA/features/fet_href_HXY.m`.

    The head frame's longitudinal axis is the unit vector
    ``hcom → nose``; the lateral axis is its 90° CCW rotation.  The
    centred-difference velocity of the ``hcom`` trajectory is then
    projected onto these axes.

    Parameters
    ----------
    xyz:
        Augmented xyz; needs ``hcom`` and ``nose``.
    samplerate:
        Optional resample target before computation.
    filter_cutoff:
        Lowpass Butterworth cutoff (Hz).  Default 4.
    filter_order:
        Filter order (passed to :func:`scipy.signal.butter`).  Default 4.
    theta:
        Per-subject yaw calibration in radians.  Pass
        ``Trial.meta.correction.headYaw`` from the relevant
        :class:`SubjectInfo`.

    Returns
    -------
    NBDfet
        Two columns: ``speed_AP`` (anteroposterior) and ``speed_LAT``
        (lateral), both scaled to approximate cm/s.
    """
    return _project_velocity(
        xyz,
        frame_from   = "hcom",
        frame_to     = "nose",
        proj_marker  = "hcom",
        samplerate   = samplerate,
        filter_cutoff= filter_cutoff,
        filter_order = filter_order,
        theta        = theta,
        scale        = 1.0 / 10.0,        # MATLAB: ".*xyz.sampleRate/10"
        columns      = ["speed_AP", "speed_LAT"],
        label        = "fet_href_HXY",
        name         = "Head-frame head velocity",
        titles       = ["Speed AP (cm/s)", "Speed LAT (cm/s)"],
        descriptions = [
            "Anteroposterior head velocity in the head frame",
            "Lateral head velocity in the head frame",
        ],
        key          = "h",
    )


# ─────────────────────────────────────────────────────────────────────────── #
# fet_bref_BXY                                                                 #
# ─────────────────────────────────────────────────────────────────────────── #

def fet_bref_BXY(
    xyz:           NBDxyz,
    samplerate:    float | None = None,
    filter_cutoff: float = 4.0,
    filter_order:  int = 4,
    theta:         float = 0.0,
) -> NBDfet:
    """Body-COM velocity projected onto the body-fixed frame.

    Port of :file:`MTA/features/fet_bref_BXY.m`.

    The body frame's longitudinal axis is the unit vector
    ``bcom → spine_upper``; lateral axis is the 90° CCW rotation.
    The centred-difference velocity of the ``bcom`` trajectory is
    projected onto these axes.

    The MATLAB scale factor is half that of fet_href_HXY
    (``samplerate / 20`` instead of ``/10``) — preserved verbatim.

    Parameters
    ----------
    xyz:
        Augmented xyz; needs ``bcom`` and ``spine_upper``.
    samplerate, filter_cutoff, filter_order, theta:
        See :func:`fet_href_HXY`.

    Returns
    -------
    NBDfet
        Two columns: ``speed_AP``, ``speed_LAT``.
    """
    return _project_velocity(
        xyz,
        frame_from   = "bcom",
        frame_to     = "spine_upper",
        proj_marker  = "bcom",
        samplerate   = samplerate,
        filter_cutoff= filter_cutoff,
        filter_order = filter_order,
        theta        = theta,
        scale        = 1.0 / 20.0,        # MATLAB: ".*(xyz.sampleRate/10)*0.5"
        columns      = ["speed_AP", "speed_LAT"],
        label        = "fet_bref_BXY",
        name         = "Body-frame body velocity",
        titles       = ["Speed AP (cm/s)", "Speed LAT (cm/s)"],
        descriptions = [
            "Anteroposterior body velocity in the body frame",
            "Lateral body velocity in the body frame",
        ],
        key          = "b",
    )


# ─────────────────────────────────────────────────────────────────────────── #
# fet_hvfl                                                                     #
# ─────────────────────────────────────────────────────────────────────────── #

def fet_hvfl(
    xyz:           NBDxyz,
    samplerate:    float | None = None,
    filter_cutoff: float = 4.0,
    filter_order:  int = 4,
    theta:         float = 0.0,
    vector:        tuple[str, str] = ("hcom", "nose"),
    proj_marker:   str = "hcom",
) -> NBDfet:
    """Generalised marker velocity projected onto a configurable head frame.

    Port of :file:`MTA/features/fet_hvfl.m`.  More flexible than
    :func:`fet_href_HXY`: the rotation-defining marker pair and the
    projection marker are independent inputs.

    Parameters
    ----------
    xyz:
        Augmented xyz.
    samplerate, filter_cutoff, filter_order, theta:
        See :func:`fet_href_HXY`.
    vector:
        ``(from_marker, to_marker)`` pair defining the head-frame's
        longitudinal axis.  Default ``('hcom', 'nose')``.
    proj_marker:
        Marker whose velocity is projected onto the frame.
        Default ``'hcom'``.

    Returns
    -------
    NBDfet
        Two columns: ``speed_AP``, ``speed_LAT``.

    Notes
    -----
    The MATLAB scaling for fet_hvfl is ``samplerate / 2 / 10``
    instead of fet_href_HXY's ``samplerate / 10`` — different by a
    factor of 2.  This is preserved verbatim; downstream code
    treats the two outputs as equivalent up to that factor.
    """
    return _project_velocity(
        xyz,
        frame_from   = vector[0],
        frame_to     = vector[1],
        proj_marker  = proj_marker,
        samplerate   = samplerate,
        filter_cutoff= filter_cutoff,
        filter_order = filter_order,
        theta        = theta,
        scale        = 1.0 / 20.0,        # MATLAB: ".*(xyz.sampleRate/2)/10"
        columns      = ["speed_AP", "speed_LAT"],
        label        = "fet_hvfl",
        name         = "Head-frame velocity",
        titles       = ["Speed AP (cm/s)", "Speed LAT (cm/s)"],
        descriptions = [
            "Anteroposterior velocity in the configured head frame",
            "Lateral velocity in the configured head frame",
        ],
        key          = "h",
    )
