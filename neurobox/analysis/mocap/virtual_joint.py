"""
neurobox.analysis.mocap.virtual_joint
======================================
Estimate the position of a virtual joint (e.g. the neck) by finding
the offset within the head's rigid-body frame at which the head's
translational motion is minimised.

Port of :file:`MTA/utilities/mocap/infer_virtual_joint_from_rigidbody_kinematics.m`.

Method
------
The intuition: a head rigid body rotates around an actual neck joint
that lies inside (not at) the rigid body's centre of mass.  When you
compute the centre of mass's *velocity* (in body-fixed coordinates),
that velocity has a contribution from the neck's translation plus a
contribution from rotation about the neck.  If you instead compute
the velocity at a candidate point ``c + R · offset`` (where ``c`` is
the COM and ``offset`` is in body-fixed coordinates), the rotational
contribution shifts.  At the actual joint, the rotational
contribution is minimum — the joint moves only with the body's
translation, not with rotation.

The function sweeps each axis independently from −100 mm to +100 mm
(in body-fixed coordinates), computes the squared sum of clipped
body-frame velocities for each shift, and picks the minimum.

Once the per-axis shifts are found, every rigid-body marker is
shifted by the same body-fixed offset, and a new ``hcom`` virtual
marker is added at the inferred joint position.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from neurobox.dtype.xyz import NBDxyz

from .basis import rigid_body_basis


def _body_frame_velocity(
    basis:       np.ndarray,        # (T, 3, 3)
    positions:   np.ndarray,        # (T, 3)
    samplerate:  float,
    speed_clip:  float = 1000.0,
    scale:       float | None = None,
) -> np.ndarray:
    """Project the centred-difference velocity of *positions* onto *basis*.

    The MATLAB scaling factor was ``samplerate / 10 / 2`` for the
    initial speed-magnitude computation and ``samplerate / 1`` for
    the gradient sweep — we keep that distinction via the *scale*
    parameter.  The returned velocity is clipped at ``±speed_clip``
    in each component.

    Returns ``(T, 3)`` body-fixed velocities.
    """
    if scale is None:
        scale = samplerate / 10.0 / 2.0
    fwd = np.roll(positions, -1, axis=0)
    bwd = np.roll(positions,  1, axis=0)
    vel = (fwd - bwd)                                   # (T, 3)
    vel[0]  = 0.0
    vel[-1] = 0.0
    # Project into body frame: out[t, k] = sum_i basis[t, i, k] * vel[t, i]
    body_vel = np.einsum("tik,ti->tk", basis, vel) * scale
    # Clip
    np.clip(body_vel, -speed_clip, speed_clip, out=body_vel)
    return body_vel


def infer_virtual_joint(
    xyz:                   NBDxyz,
    rigid_body_markers:    Sequence[str] = (
        "head_back", "head_left", "head_front", "head_right",
    ),
    auxiliary_markers:     Sequence[str] = ("head_top",),
    samplerate:            float = 40.0,
    speed_min_cm_s:        float = 5.0,
    speed_max_cm_s:        float = 100.0,
    height_max_mm:         float = 150.0,
    shift_range_mm:        tuple[float, float] = (-100.0, 100.0),
    shift_step_mm:         float = 1.0,
) -> tuple[NBDxyz, np.ndarray]:
    """Estimate a virtual joint inside a rigid body by minimising motion.

    Port of :file:`MTA/utilities/mocap/infer_virtual_joint_from_rigidbody_kinematics.m`.

    Parameters
    ----------
    xyz:
        Source position data, must contain all of *rigid_body_markers*.
        Auxiliary markers are shifted along with the rigid body if
        present, but missing auxiliary markers are silently skipped.
    rigid_body_markers:
        Markers that move together as a rigid body.  Default is the
        standard 4-head-marker set.
    auxiliary_markers:
        Additional markers to translate by the same shift (e.g.
        ``head_top``).
    samplerate:
        Working samplerate for the optimisation, in Hz.  Default 40 Hz
        matches the MATLAB original.  Only the shift estimate uses
        this; the returned xyz keeps the original samplerate.
    speed_min_cm_s, speed_max_cm_s:
        Body-frame speed window (cm/s) used to mask "appropriate"
        samples for the optimisation.  Defaults 5 and 100 mirror the
        MATLAB original.
    height_max_mm:
        Maximum head COM z-coordinate (mm) for samples in the
        optimisation window.  Default 150.
    shift_range_mm:
        ``(min, max)`` shift range in mm to sweep over each axis.
        Default ``(-100, 100)``.
    shift_step_mm:
        Step size in mm for the shift sweep.  Default 1.

    Returns
    -------
    xyz_corrected : NBDxyz
        New :class:`NBDxyz` with all rigid-body and auxiliary markers
        shifted by the inferred body-fixed offset, and a new ``hcom``
        marker placed at the virtual joint position.
    offset : np.ndarray, shape ``(3,)``
        The inferred body-fixed offset (mm).  ``offset[i]`` is the
        shift along the ``i``-th axis of the per-frame rigid-body
        basis.
    """
    if xyz._data is None:
        raise RuntimeError("xyz data is not loaded.")
    for m in rigid_body_markers:
        if m not in xyz.model.markers:
            raise ValueError(
                f"infer_virtual_joint: xyz has no {m!r} marker."
            )

    # ── Phase 1: estimate the offset on the resampled data ─────────── #
    xyz_low = xyz.resample(float(samplerate)) if samplerate != xyz.samplerate else xyz

    basis_low, com_low = rigid_body_basis(xyz_low, list(rigid_body_markers))

    # Initial body-frame speed magnitude — used to mask "appropriate" samples
    body_vel0 = _body_frame_velocity(
        basis_low, com_low,
        samplerate=samplerate,
        scale=samplerate / 10.0 / 2.0,
    )
    speed_mag = np.linalg.norm(body_vel0, axis=1)
    appropriate = (
        (speed_mag > speed_min_cm_s) &
        (speed_mag < speed_max_cm_s) &
        (com_low[:, 2] < height_max_mm)
    )
    if not appropriate.any():
        raise RuntimeError(
            "infer_virtual_joint: no samples in the speed/height window — "
            "loosen the constraints or check input data."
        )

    shifts = np.arange(
        shift_range_mm[0], shift_range_mm[1] + 0.5 * shift_step_mm,
        shift_step_mm,
    )
    n_dims = xyz.data.shape[2]
    offset = np.zeros(n_dims, dtype=np.float64)

    for dim in range(n_dims):
        unit = np.zeros(n_dims, dtype=np.float64)
        unit[dim] = 1.0
        # Shift in body-fixed coords: body_offset = unit * shift, world offset
        # is basis @ body_offset.
        world_shift_per_unit = np.einsum("tik,k->ti", basis_low, unit)  # (T, 3)

        gradient = np.empty(shifts.size, dtype=np.float64)
        for si, shift in enumerate(shifts):
            scom = com_low + world_shift_per_unit * shift
            scom_vel = (np.roll(scom, -1, axis=0) - np.roll(scom, 1, axis=0))
            scom_vel[0]  = 0.0
            scom_vel[-1] = 0.0
            # Body-frame velocity of shifted point
            body_vel = np.einsum(
                "tik,ti->tk", basis_low[appropriate], scom_vel[appropriate]
            )
            np.clip(body_vel, -100.0, 100.0, out=body_vel)
            gradient[si] = np.log10(np.nansum(body_vel ** 2) + np.finfo(float).tiny)

        offset[dim] = float(shifts[int(np.argmin(gradient))])

    # ── Phase 2: apply the offset to the original-rate data ────────── #
    basis_full, com_full = rigid_body_basis(xyz, list(rigid_body_markers))
    world_shift = np.einsum("tik,k->ti", basis_full, offset)  # (T, 3)

    new = xyz
    # Add a new hcom marker at the virtual joint position
    if "hcom" in new.model.markers:
        new = new.add_marker("hcom", com_full + world_shift, overwrite=True)
    else:
        new = new.add_marker(
            "hcom", com_full + world_shift,
            connections=[(m, "hcom") for m in rigid_body_markers],
        )

    # Shift all rigid-body markers
    data = new._data.copy()
    for m in rigid_body_markers:
        idx = new.model.index(m)
        data[:, idx, :] = data[:, idx, :] + world_shift

    # Shift auxiliary markers if present
    for m in auxiliary_markers:
        if m in new.model.markers:
            idx = new.model.index(m)
            data[:, idx, :] = data[:, idx, :] + world_shift

    import copy as _copy
    out = _copy.copy(new)
    out._data = data
    return out, offset
