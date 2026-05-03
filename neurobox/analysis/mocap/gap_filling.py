"""
neurobox.analysis.mocap.gap_filling
====================================
Fill short NaN gaps in mocap position data via PCHIP interpolation.

Port of :file:`MTA/utilities/mocap/mocap_fill_gaps.m`.

The MATLAB original had a typo (``gapPer`` used on line 3 before the
variable was defined; it should reference the input ``gapPeriods``).
This port fixes that bug and otherwise preserves the algorithm.

Algorithm
---------
1. Filter the input gap periods by length, keeping only those with
   length ≤ ``min_gap_length`` samples.
2. Build a per-sample indicator marking those short gaps; smooth it
   with a moving-average box of width ``min_gap_length`` (this
   merges close-together gaps into single fillable regions).
3. Threshold-cross to recover the merged regions.
4. For each region, only fill it if the distance to the previous and
   next regions is at least ``2 × gap_length`` — i.e. there's enough
   surrounding good data on both sides to anchor the interpolation.
5. Within each fillable region, interpolate every column of the
   ``(T, n_markers, n_dims)`` array via PCHIP (cubic monotone
   Hermite) using only the valid (non-NaN-non-zero) samples within
   ``2 × gap_length`` of the gap.
"""

from __future__ import annotations

import copy as _copy

import numpy as np
from scipy.interpolate import PchipInterpolator

from neurobox.analysis.kinematics.helpers import finite_nonzero_mask
from neurobox.analysis.lfp.oscillations import thresh_cross
from neurobox.dtype.xyz import NBDxyz


def fill_gaps(
    xyz:            NBDxyz,
    gap_periods:    np.ndarray | None = None,
    min_gap_length: int = 5,
) -> NBDxyz:
    """Fill short NaN gaps in xyz data via PCHIP.

    Port of :file:`MTA/utilities/mocap/mocap_fill_gaps.m` (with bug fixes).

    Parameters
    ----------
    xyz:
        Source position data.  Gaps must be marked by NaN, Inf, or
        all-zero rows (matching :func:`finite_nonzero_mask`).
    gap_periods:
        Optional ``(N, 2)`` array of explicit gap start / end sample
        indices.  When ``None`` (default), gaps are auto-detected
        from the validity mask of *xyz*.
    min_gap_length:
        Maximum gap length (in samples) eligible for filling.  Gaps
        longer than this are left as-is — the MATLAB original took
        this same parameter and used it both as a length threshold
        and as the smoothing kernel width.  Default 5.

    Returns
    -------
    NBDxyz
        New object with gaps filled where possible.  The original
        is unmodified.
    """
    if xyz._data is None:
        raise RuntimeError("xyz data is not loaded.")

    data = xyz._data.copy()
    T = data.shape[0]

    # Auto-detect gap periods if not supplied
    if gap_periods is None:
        valid = finite_nonzero_mask(data.reshape(T, -1))
        bad   = ~valid
        # Find runs of consecutive bad samples
        edges  = np.diff(np.concatenate([[False], bad, [False]]).astype(np.int8))
        starts = np.where(edges == 1)[0]
        stops  = np.where(edges == -1)[0]              # exclusive end
        if starts.size == 0:
            return _copy.copy(xyz)
        # Convert to inclusive (start, end) sample indices, matching MATLAB's
        # ThreshCross convention.
        gap_periods = np.column_stack([starts, stops - 1])
    gap_periods = np.asarray(gap_periods, dtype=np.int64)
    if gap_periods.size == 0:
        return _copy.copy(xyz)

    gap_lengths = gap_periods[:, 1] - gap_periods[:, 0] + 1

    # Build a smoothed indicator over short gaps
    short = gap_lengths <= min_gap_length
    if not short.any():
        return _copy.copy(xyz)

    indicator = np.zeros(T, dtype=np.float64)
    for s, e in gap_periods[short]:
        indicator[s:e + 1] = 1.0

    # Convolve with same-size box of width min_gap_length
    if min_gap_length > 1:
        kernel = np.ones(min_gap_length, dtype=np.float64)
        smoothed = np.convolve(indicator, kernel, mode="same")
    else:
        smoothed = indicator

    # Threshold-cross to get merged regions
    merged = thresh_cross(smoothed, threshold=0.1)
    if merged.shape[0] == 0:
        return _copy.copy(xyz)

    # MATLAB: gapPer = bsxfun(@plus, gapPer, [1, 0]) — shifts starts by +1
    merged = merged.astype(np.int64).copy()
    merged[:, 0] += 1

    # Distance from each region's start to the previous region's end
    # (or to sample 0 / T-1 at the boundaries) — matches the
    # MATLAB ``gapDstBA`` calculation.
    starts_padded = np.concatenate([merged[:, 0], [T]])
    stops_padded  = np.concatenate([[0],          merged[:, 1] + 1])
    gap_dst_ba    = starts_padded - stops_padded     # (n_merged + 1,)

    region_lengths = merged[:, 1] - merged[:, 0]

    # Only fill regions where both neighbours are at least 2x gap-length away
    fillable = (
        (gap_dst_ba[:-1] > 2 * region_lengths) &
        (gap_dst_ba[ 1:] > 2 * region_lengths)
    )

    n_markers, n_dims = data.shape[1], data.shape[2]

    for gid in np.where(fillable)[0]:
        gap_len    = int(region_lengths[gid])
        half_seg   = int(round(gap_len * (1 + 2)))      # MATLAB sum([1, 2] * gapLen)
        seg_start  = merged[gid, 0] - half_seg
        seg_stop   = merged[gid, 1] + half_seg
        seg_inds   = np.arange(seg_start, seg_stop + 1)
        # Clip to bounds
        seg_inds   = seg_inds[(seg_inds >= 0) & (seg_inds < T)]
        if seg_inds.size < 4:
            continue   # PCHIP needs at least 2 points per axis

        # Identify NaN / valid samples within the segment using marker 0 dim 0
        # (matches MATLAB's ``seg = obj(segInd, 1, 1)``)
        seg = data[seg_inds, 0, 0]
        nan_mask  = ~np.isfinite(seg) | (seg == 0)
        good_mask = ~nan_mask

        if good_mask.sum() < 2 or nan_mask.sum() == 0:
            continue

        good_inds = seg_inds[good_mask]
        nan_inds  = seg_inds[nan_mask]

        # Interpolate every column independently
        for m in range(n_markers):
            for d in range(n_dims):
                col = data[good_inds, m, d]
                # Skip if any of the anchor samples are NaN/Inf for this column
                if not np.all(np.isfinite(col)):
                    continue
                pchip = PchipInterpolator(good_inds.astype(np.float64), col,
                                           extrapolate=False)
                new_vals = pchip(nan_inds.astype(np.float64))
                # PchipInterpolator returns NaN outside support; only assign
                # where it's finite
                ok = np.isfinite(new_vals)
                if ok.any():
                    data[nan_inds[ok], m, d] = new_vals[ok]

    new = _copy.copy(xyz)
    new._data = data
    return new
