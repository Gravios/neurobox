"""
neurobox.analysis.kinematics.features.pitch
============================================
Head and body pitch features — angles in the spherical-coordinate
elevation channel (radians) between body / head landmarks.
"""

from __future__ import annotations

import numpy as np

from neurobox.dtype.ang import NBDang
from neurobox.dtype.fet import NBDfet
from neurobox.dtype.xyz import NBDxyz
from ..helpers import finite_nonzero_mask


def _circ_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Signed circular difference a - b, wrapped to [-π, π]."""
    return np.angle(np.exp(1j * (a - b)))


# ─────────────────────────────────────────────────────────────────────────── #
# fet_head_pitch                                                               #
# ─────────────────────────────────────────────────────────────────────────── #

def fet_head_pitch(xyz: NBDxyz, samplerate: float | None = None) -> NBDfet:
    """Head pitch — elevation of head_back → head_front vector.

    Port of :file:`MTA/features/fet_head_pitch.m`.

    Returns one column ``head_pitch`` (radians).  Positive = nose
    above the back of the head.
    """
    for needed in ("head_back", "head_front"):
        if needed not in xyz.model.markers:
            raise ValueError(
                f"fet_head_pitch: xyz has no {needed!r} marker."
            )
    ang = NBDang.from_xyz(xyz)
    pitch = ang.between("head_back", "head_front", component="phi")  # (T,)
    data = pitch.reshape(-1, 1).astype(np.float64)

    mask = finite_nonzero_mask(xyz.data.reshape(xyz.data.shape[0], -1))
    data[~mask, :] = 0.0

    fet = NBDfet(
        data       = data,
        columns    = ["head_pitch"],
        samplerate = float(xyz.samplerate),
        label      = "fet_head_pitch",
        name       = "Head pitch",
        titles     = ["Pitch HBHF"],
        descriptions = ["head pitch (head_back → head_front) relative to xy plane"],
        key        = "h",
    )
    if samplerate is not None and samplerate != xyz.samplerate:
        fet = fet.resample(float(samplerate))
    return fet


# ─────────────────────────────────────────────────────────────────────────── #
# fet_HB_pitch                                                                 #
# ─────────────────────────────────────────────────────────────────────────── #

def fet_HB_pitch(
    xyz:         NBDxyz,
    samplerate:  float | None = None,
    nose_marker: str | None = None,
) -> NBDfet:
    """Three sequential head/body pitches: pelvis→spine_upper, spine_upper→hcom, hcom→nose.

    Port of :file:`MTA/features/fet_HB_pitch.m`.

    Parameters
    ----------
    xyz:
        Augmented xyz; needs ``pelvis_root``, ``spine_upper``,
        ``hcom``, and a nose marker (``nose`` or ``head_nose``).
    nose_marker:
        Override the nose-marker name; default tries ``nose`` then
        ``head_nose``.

    Returns
    -------
    NBDfet
        Three columns: ``pitch_BPBU``, ``pitch_BUHC``, ``pitch_HBHF``
        (named after the MATLAB convention: BP=body proximal,
        BU=body upper, HC=head center, HB=head back, HF=head front).
    """
    needed_base = ("pelvis_root", "spine_upper", "hcom")
    for n in needed_base:
        if n not in xyz.model.markers:
            raise ValueError(f"fet_HB_pitch: xyz has no {n!r} marker.")

    if nose_marker is None:
        for cand in ("nose", "head_nose"):
            if cand in xyz.model.markers:
                nose_marker = cand
                break
    if nose_marker is None or nose_marker not in xyz.model.markers:
        raise ValueError(
            "fet_HB_pitch: xyz has no nose marker (tried 'nose', 'head_nose'). "
            "Run augment_xyz first."
        )

    ang = NBDang.from_xyz(xyz)
    p1 = ang.between("pelvis_root", "spine_upper", component="phi")
    p2 = ang.between("spine_upper", "hcom",        component="phi")
    p3 = ang.between("hcom",        nose_marker,   component="phi")
    data = np.column_stack([p1, p2, p3]).astype(np.float64)

    mask = finite_nonzero_mask(xyz.data.reshape(xyz.data.shape[0], -1))
    data[~mask, :] = 0.0

    fet = NBDfet(
        data       = data,
        columns    = ["pitch_BPBU", "pitch_BUHC", "pitch_HCHN"],
        samplerate = float(xyz.samplerate),
        label      = "fet_HB_pitch",
        name       = "Head + body pitch",
        titles     = ["Pitch BPBU", "Pitch BUHC", "Pitch HCHN"],
        descriptions = [
            "pelvis_root → spine_upper pitch",
            "spine_upper → hcom pitch",
            f"hcom → {nose_marker} pitch",
        ],
        key        = "b",
    )
    if samplerate is not None and samplerate != xyz.samplerate:
        fet = fet.resample(float(samplerate))
    return fet


# ─────────────────────────────────────────────────────────────────────────── #
# fet_HB_pitchB                                                                #
# ─────────────────────────────────────────────────────────────────────────── #

def fet_HB_pitchB(
    xyz:        NBDxyz,
    samplerate: float | None = None,
) -> NBDfet:
    """Head/body pitch pair: ``_circ_dist(HCHN, BPBU)`` and ``BPBU``.

    Port of :file:`MTA/features/fet_HB_pitchB.m`.  This is the
    "compact" form of fet_HB_pitch used by fet_xyhb / fet_hzp.
    """
    pch = fet_HB_pitch(xyz)  # (T, 3) — pitch_BPBU, pitch_BUHC, pitch_HCHN
    p_BPBU = pch.sel("pitch_BPBU")
    p_HCHN = pch.sel("pitch_HCHN")

    head_minus_body = _circ_dist(p_HCHN, p_BPBU)
    data = np.column_stack([head_minus_body, p_BPBU]).astype(np.float64)
    mask = finite_nonzero_mask(xyz.data.reshape(xyz.data.shape[0], -1))
    data[~mask, :] = 0.0

    fet = NBDfet(
        data       = data,
        columns    = ["head_minus_body_pitch", "body_pitch"],
        samplerate = float(xyz.samplerate),
        label      = "fet_HB_pitchB",
        name       = "Head body pitch (compact)",
        titles     = ["Pitch HCHN-BPBU", "Pitch BPBU"],
        descriptions = [
            "circular distance between head and body pitches",
            "body pitch (pelvis_root → spine_upper)",
        ],
        key        = "b",
    )
    if samplerate is not None and samplerate != xyz.samplerate:
        fet = fet.resample(float(samplerate))
    return fet


# ─────────────────────────────────────────────────────────────────────────── #
# fet_xyhb                                                                     #
# ─────────────────────────────────────────────────────────────────────────── #

def fet_xyhb(xyz: NBDxyz, samplerate: float | None = None) -> NBDfet:
    """Head xy + head/body pitch pair.

    Port of :file:`MTA/features/fet_xyhb.m`.  Concatenation of
    ``fet_xy`` and ``fet_HB_pitchB``.
    """
    if "hcom" not in xyz.model.markers:
        raise ValueError("fet_xyhb: xyz has no 'hcom' marker.")

    pchB = fet_HB_pitchB(xyz)
    hcom = xyz.sel(["hcom"], dims=[0, 1])[:, 0, :]    # (T, 2)
    data = np.column_stack([hcom, pchB.data]).astype(np.float64)

    mask = finite_nonzero_mask(xyz.data.reshape(xyz.data.shape[0], -1))
    data[~mask, :] = 0.0

    fet = NBDfet(
        data       = data,
        columns    = ["x", "y", "head_minus_body_pitch", "body_pitch"],
        samplerate = float(xyz.samplerate),
        label      = "fet_xyhb",
        name       = "Head xy + head/body pitch",
        titles     = [
            "position x (mm)", "Position y (mm)",
            "Pitch HCHN-BPBU", "Pitch BPBU",
        ],
        descriptions = [
            "Position along the X axis",
            "Position along the Y axis",
            "head pitch relative to body pitch",
            "body pitch (pelvis_root → spine_upper)",
        ],
        key        = "x",
    )
    if samplerate is not None and samplerate != xyz.samplerate:
        fet = fet.resample(float(samplerate))
    return fet


# ─────────────────────────────────────────────────────────────────────────── #
# fet_hzp                                                                      #
# ─────────────────────────────────────────────────────────────────────────── #

def fet_hzp(xyz: NBDxyz, samplerate: float | None = None) -> NBDfet:
    """Head height + circular head/body pitch difference.

    Port of :file:`MTA/features/fet_hzp.m`.

    Returns
    -------
    NBDfet
        Two columns: ``head_minus_body_pitch`` (rad), ``head_height`` (mm).
    """
    if "hcom" not in xyz.model.markers:
        raise ValueError("fet_hzp: xyz has no 'hcom' marker.")

    pch = fet_HB_pitch(xyz)
    p_BPBU = pch.sel("pitch_BPBU")
    p_HCHN = pch.sel("pitch_HCHN")
    head_minus_body = _circ_dist(p_HCHN, p_BPBU)
    head_height = xyz.sel(["hcom"], dims=[2])[:, 0, 0]    # (T,)
    data = np.column_stack([head_minus_body, head_height]).astype(np.float64)

    mask = finite_nonzero_mask(xyz.data.reshape(xyz.data.shape[0], -1))
    data[~mask, :] = 0.0

    fet = NBDfet(
        data       = data,
        columns    = ["head_minus_body_pitch", "head_height"],
        samplerate = float(xyz.samplerate),
        label      = "fet_hzp",
        name       = "Head height and body pitch",
        titles     = ["Pitch HCHN-BPBU", "Head Height"],
        descriptions = [
            "circular distance between head and body pitches",
            "z-coordinate of head center",
        ],
        key        = "h",
    )
    if samplerate is not None and samplerate != xyz.samplerate:
        fet = fet.resample(float(samplerate))
    return fet
