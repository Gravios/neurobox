"""
neurobox.analysis.kinematics.features.basic
============================================
Position-only features derived from an augmented xyz tracking object.
"""

from __future__ import annotations

import numpy as np

from neurobox.dtype.xyz import NBDxyz
from neurobox.dtype.fet import NBDfet
from ..helpers import finite_nonzero_mask


def fet_xy(xyz: NBDxyz, samplerate: float | None = None) -> NBDfet:
    """Head COM xy position (mm).

    Port of :file:`MTA/features/fet_xy.m`.

    Parameters
    ----------
    xyz:
        Augmented :class:`NBDxyz`; must contain an ``hcom`` marker.
        Typically the output of :func:`augment_xyz`.
    samplerate:
        Optional resampling target.  ``None`` (default) keeps the
        input rate.

    Returns
    -------
    NBDfet
        Two columns: ``x``, ``y`` (head COM xy in mm).  Invalid rows
        (NaN, inf, or all-zero in the source xyz) are zeroed —
        matches MATLAB's ``fet.data(~nniz(xyz),:) = 0`` convention.
    """
    if "hcom" not in xyz.model.markers:
        raise ValueError(
            "fet_xy: xyz has no 'hcom' marker; pass through augment_xyz first."
        )
    hcom = xyz.sel(["hcom"], dims=[0, 1])           # (T, 1, 2)
    data = hcom[:, 0, :].copy()                     # (T, 2)

    # Zero out invalid samples
    mask = finite_nonzero_mask(xyz.data.reshape(xyz.data.shape[0], -1))
    data[~mask] = 0.0

    fet = NBDfet(
        data       = data,
        columns    = ["x", "y"],
        samplerate = float(xyz.samplerate),
        label      = "fet_xy",
        name       = "Head xy position",
        titles     = ["position x (mm)", "position y (mm)"],
        descriptions = [
            "Position along the X axis",
            "Position along the Y axis",
        ],
        key        = "p",
    )
    if samplerate is not None and samplerate != xyz.samplerate:
        fet = fet.resample(float(samplerate))
    return fet


def fet_dxy(xyz: NBDxyz, samplerate: float | None = None) -> NBDfet:
    """Head yaw plus head COM xy position.

    Port of :file:`MTA/features/fet_dxy.m`.  Yaw is computed as the
    azimuth of the ``head_front - hcom`` vector — i.e. the
    head-direction angle.

    Notes
    -----
    The MATLAB original uses ``atan2(hxy(:,1), hxy(:,2))`` instead of
    ``atan2(dy, dx)``.  This is a non-standard angle convention
    where the angle is measured **from the y axis clockwise toward
    x** rather than the standard "from x axis counter-clockwise".
    This port preserves that convention for downstream compatibility.
    """
    for needed in ("hcom", "head_front"):
        if needed not in xyz.model.markers:
            raise ValueError(
                f"fet_dxy: xyz has no {needed!r} marker; "
                "pass through augment_xyz first."
            )
    hcom = xyz.sel(["hcom"],       dims=[0, 1])[:, 0, :]   # (T, 2)
    head_front = xyz.sel(["head_front"], dims=[0, 1])[:, 0, :]
    hxy = head_front - hcom
    norm = np.linalg.norm(hxy, axis=1, keepdims=True)
    hxy_unit = np.divide(
        hxy, norm,
        out=np.zeros_like(hxy),
        where=norm > 0,
    )
    # MATLAB convention: atan2(dx, dy) — see notes
    yaw = np.arctan2(hxy_unit[:, 0], hxy_unit[:, 1])

    data = np.column_stack([yaw, hcom])              # (T, 3)
    mask = finite_nonzero_mask(xyz.data.reshape(xyz.data.shape[0], -1))
    data[~mask] = 0.0

    fet = NBDfet(
        data       = data,
        columns    = ["head_yaw", "x", "y"],
        samplerate = float(xyz.samplerate),
        label      = "fet_dxy",
        name       = "Head direction + xy",
        titles     = [
            "Head Yaw (rad)",
            "position x (mm)",
            "Position y (mm)",
        ],
        descriptions = [
            "Head direction (azimuth of head_front - hcom)",
            "Position along the X axis",
            "Position along the Y axis",
        ],
        key        = "d",
    )
    if samplerate is not None and samplerate != xyz.samplerate:
        fet = fet.resample(float(samplerate))
    return fet
