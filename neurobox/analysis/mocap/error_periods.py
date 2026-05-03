"""
neurobox.analysis.mocap.error_periods
======================================
Detect periods where mocap data violates rigid-body constraints
(typically marker swaps, dropouts, or tracking jitter).

Port of :file:`MTA/utilities/mocap/FindErrorPeriods.m`.

Algorithm
---------
For a 4-marker rigid body (``head_back``, ``head_left``,
``head_front``, ``head_right`` by default), the function builds **four**
egocentric transforms:

* origin = back, orientation = front,  vectors = (left,  right)
* origin = front, orientation = back,  vectors = (right, left)
* origin = right, orientation = left,  vectors = (back,  front)
* origin = left,  orientation = right, vectors = (front, back)

Each transform projects the lateral pair into the corresponding
egocentric frame.  When the rigid body is intact, those projections
are stationary (the markers don't move relative to each other).
When a marker is swapped, dropped, or jitters, the second component
of the egocentric pair (typically the lateral offset) becomes
unstable.

The function takes the first lateral component of two of those
projections, computes the per-frame deviation from the mean, and
flags samples where the deviation exceeds 1 standard deviation as
errors.  Contiguous error samples form periods.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from neurobox.analysis.transform_origin import transform_origin
from neurobox.analysis.lfp.oscillations import thresh_cross
from neurobox.dtype.xyz import NBDxyz


def find_error_periods(
    xyz:     NBDxyz,
    markers: Sequence[str] = ("head_back", "head_left", "head_front", "head_right"),
    threshold_z: float = 1.0,
) -> tuple[np.ndarray, list, np.ndarray]:
    """Detect rigid-body-violation periods.

    Port of :file:`MTA/utilities/mocap/FindErrorPeriods.m`.

    Parameters
    ----------
    xyz:
        Source position data.  Must contain all four *markers*.
    markers:
        Four-marker rigid body to validate.  Default
        ``(head_back, head_left, head_front, head_right)`` matches the
        MATLAB original.
    threshold_z:
        Number of standard deviations above the mean deviation at
        which a sample is flagged as an error.  Default 1.0 matches
        the MATLAB original — note this is unusually permissive
        (1σ flags ~16% of samples even with no actual errors).

    Returns
    -------
    error_periods : np.ndarray, shape ``(N, 2)``
        Sample-index intervals where the rigid body fails the test.
    transforms : list of :class:`TransformResult`
        The four egocentric transforms (kept for diagnostic plotting).
    error_signal : np.ndarray, shape ``(T,)``
        The summed-deviation signal that was thresholded.
    """
    if xyz._data is None:
        raise RuntimeError("xyz data is not loaded.")
    if len(markers) != 4:
        raise ValueError(
            f"find_error_periods: expected 4 markers; got {len(markers)}"
        )

    m0, m1, m2, m3 = markers
    transforms = [
        transform_origin(xyz, m0, m2, [m1, m3]),     # back → front
        transform_origin(xyz, m2, m0, [m3, m1]),     # front → back
        transform_origin(xyz, m3, m1, [m0, m2]),     # right → left
        transform_origin(xyz, m1, m3, [m2, m0]),     # left → right
    ]

    # MATLAB:  efet = [hfcl{1}.transVec(:,1,2), hfcl{2}.transVec(:,1,2)]
    # That's the 2nd component (Y in egocentric coords) of the FIRST
    # vector marker, from the first two transforms.
    efet = np.column_stack([
        transforms[0].trans_coords[:, 0, 1],
        transforms[1].trans_coords[:, 0, 1],
    ])

    # Per-column mean (ignoring NaN), summed |deviation|
    means = np.nanmean(efet, axis=0)
    deviation = np.nansum(np.abs(efet - means), axis=1)   # (T,)

    valid = np.isfinite(deviation)
    if not valid.any():
        # Nothing to threshold → no error periods
        return np.zeros((0, 2), dtype=np.int64), transforms, deviation

    err_mean = float(np.mean(deviation[valid]))
    err_std  = float(np.std(deviation[valid]))
    if err_std == 0:
        return np.zeros((0, 2), dtype=np.int64), transforms, deviation

    z = np.abs((deviation - err_mean) / err_std)
    error_state = (z > threshold_z).astype(np.float64)
    error_state[~valid] = 0.0

    error_periods = thresh_cross(error_state, threshold=0.5).astype(np.int64)

    return error_periods, transforms, deviation
