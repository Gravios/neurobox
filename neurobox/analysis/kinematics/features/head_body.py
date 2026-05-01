"""
neurobox.analysis.kinematics.features.head_body
================================================
Head-body azimuth angle and its angular velocity.
"""

from __future__ import annotations

import numpy as np

from neurobox.analysis.lfp.filtering import butter_filter
from neurobox.dtype.fet import NBDfet
from neurobox.dtype.xyz import NBDxyz
from ..helpers import finite_nonzero_mask


def _circ_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Signed circular difference a - b, wrapped to [-π, π]."""
    return np.angle(np.exp(1j * (a - b)))


def fet_hba(
    xyz:                 NBDxyz,
    samplerate:          float | None = None,
    head_body_correction: float = 0.0,
    offset:              float = 0.0,
) -> NBDfet:
    """Head-body azimuth angle (radians) — port of :file:`fet_hba.m`.

    The head-body angle is defined as the **circular distance**
    between the body's xy heading (``bcom → spine_upper``) and the
    head's xy heading (``hcom → nose``), with a per-subject
    ``head_body_correction`` offset and an optional global ``offset``.

    The MATLAB sign convention is preserved: the result is negated
    so that positive values correspond to "head turned to the left
    relative to the body" (right-handed convention).

    Parameters
    ----------
    xyz:
        Augmented xyz; must have ``bcom``, ``spine_upper``, ``hcom``,
        ``nose``.
    samplerate:
        Optional resampling target.
    head_body_correction:
        Per-subject calibration offset applied before negation.
        Mirrors MATLAB's ``Trial.meta.correction.headBody``.
        Default 0; pass the value from the relevant
        :class:`neurobox.config.SubjectInfo`.
    offset:
        Constant offset added at the end (matches MATLAB's
        ``offset`` argument).

    Returns
    -------
    NBDfet
        One column ``head_body_angle`` (rad).
    """
    needed = ("bcom", "spine_upper", "hcom", "nose")
    for n in needed:
        if n not in xyz.model.markers:
            raise ValueError(f"fet_hba: xyz has no {n!r} marker.")

    body_vec = (
        xyz.sel(["spine_upper"], dims=[0, 1])[:, 0, :] -
        xyz.sel(["bcom"],        dims=[0, 1])[:, 0, :]
    )
    head_vec = (
        xyz.sel(["nose"], dims=[0, 1])[:, 0, :] -
        xyz.sel(["hcom"], dims=[0, 1])[:, 0, :]
    )
    body_az = np.arctan2(body_vec[:, 1], body_vec[:, 0])
    head_az = np.arctan2(head_vec[:, 1], head_vec[:, 0])
    hba = -(_circ_dist(head_az, body_az) + float(head_body_correction))
    data = (hba + float(offset)).reshape(-1, 1).astype(np.float64)

    mask = finite_nonzero_mask(xyz.data.reshape(xyz.data.shape[0], -1))
    data[~mask, :] = 0.0

    fet = NBDfet(
        data       = data,
        columns    = ["head_body_angle"],
        samplerate = float(xyz.samplerate),
        label      = "fet_hba",
        name       = "Head body angle",
        titles     = ["Head Body Angle (rad)"],
        descriptions = ["head xy direction relative to body xy direction"],
        key        = "a",
    )
    if samplerate is not None and samplerate != xyz.samplerate:
        fet = fet.resample(float(samplerate))
    return fet


def fet_hbav(
    xyz:                 NBDxyz,
    samplerate:          float | None = None,
    head_body_correction: float = 0.0,
    offset:              float = 0.0,
    low_pass_cutoff:     float = 2.4,
    filter_order:        int = 4,
) -> NBDfet:
    """Head-body angular velocity — :file:`fet_hbav.m`.

    Computes :func:`fet_hba`, low-pass filters it, then takes the
    centred finite difference and scales by ``samplerate / 2``
    (matches MATLAB's ``(circshift(-1) - circshift(+1)) * fs/2``
    discrete-derivative pattern).

    Returns
    -------
    NBDfet
        One column ``head_body_ang_vel`` (rad/s).
    """
    hba = fet_hba(xyz, samplerate=samplerate,
                  head_body_correction=head_body_correction, offset=offset)
    fs = hba.samplerate

    filtered = butter_filter(
        hba.data,
        cutoff     = low_pass_cutoff,
        samplerate = fs,
        order      = filter_order,
        btype      = "low",
    )
    # Centred difference: (x[t+1] - x[t-1]) * fs/2
    deriv = (np.roll(filtered, -1, axis=0) - np.roll(filtered, +1, axis=0)) * (fs / 2.0)
    deriv[0, :]  = 0.0
    deriv[-1, :] = 0.0

    mask = finite_nonzero_mask(xyz.data.reshape(xyz.data.shape[0], -1))
    if mask.shape[0] == deriv.shape[0]:
        deriv[~mask, :] = 0.0

    return NBDfet(
        data       = deriv,
        columns    = ["head_body_ang_vel"],
        samplerate = fs,
        label      = "fet_hbav",
        name       = "Head body angular velocity",
        titles     = ["Head Body Angular Velocity (rad/s)"],
        descriptions = [
            "centred derivative of low-pass-filtered head-body angle",
        ],
        key        = "a",
    )
