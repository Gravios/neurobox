"""
xyz_fill_gaps.py
================
Fill zero / NaN gaps in assembled motion-capture position data.

After sync, the xyz array has zero-padded regions representing either:
  (a) gaps between recording blocks (inter-trial zeroes)
  (b) dropout frames within a block (tracking lost)

This module provides functions to interpolate across both types of gap
using pchip (monotone cubic) or linear interpolation — a Python port of
the MTA ``mocap_fill_gaps`` / ``fill_gaps_rigidbody`` logic, without the
interactive/GUI components.

Functions
---------
fill_gaps(xyz_arr, samplerate, max_gap_sec, method)
    Detect zero-rows in *xyz_arr* and fill them by interpolating from
    the nearest valid neighbours per marker per dimension.

detect_gaps(xyz_arr)
    Return a boolean mask and gap-period array for diagnostic use.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Gap detection
# ---------------------------------------------------------------------------

def detect_gaps(
    xyz_arr: np.ndarray,
    sentinel: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Identify gap (missing / sentinel-filled) frames in *xyz_arr*.

    A frame is considered a gap if all markers have magnitude ≤ the
    sentinel value across all dimensions.  In practice:

    * MTA fills intra-block dropouts with 0 or ``eps``
    * Our pipeline fills zero-values with ``np.finfo(float32).eps``
    * True zero is never a valid tracker position

    Parameters
    ----------
    xyz_arr:
        Array of shape ``(T, N_markers, 3)``.
    sentinel:
        Threshold below which a value is treated as missing.
        Defaults to ``2 * np.finfo(np.float32).eps``.

    Returns
    -------
    gap_mask : np.ndarray, shape (T,), bool
        True where a frame is missing.
    gap_periods : np.ndarray, shape (K, 2), int
        Start/stop sample indices ``[i_start, i_stop)`` of each gap run.
    """
    if sentinel is None:
        sentinel = 2.0 * float(np.finfo(np.float32).eps)

    # A frame is a gap if every value in it is ≤ sentinel
    gap_mask = (np.abs(xyz_arr).max(axis=(1, 2)) <= sentinel)

    # Convert boolean mask to [start, stop) periods
    edges      = np.diff(gap_mask.astype(np.int8),
                         prepend=0, append=0)
    starts = np.where(edges ==  1)[0]
    stops  = np.where(edges == -1)[0]
    gap_periods = np.column_stack([starts, stops]) if len(starts) else \
                  np.empty((0, 2), dtype=np.int64)

    return gap_mask, gap_periods


# ---------------------------------------------------------------------------
# Gap filling
# ---------------------------------------------------------------------------

def fill_gaps(
    xyz_arr:       np.ndarray,
    samplerate:    float,
    max_gap_sec:   float = 0.5,
    method:        str   = "pchip",
    inplace:       bool  = False,
) -> np.ndarray:
    """Fill zero / sentinel gaps in a motion-capture position array.

    Port of MTA ``mocap_fill_gaps``.

    Gaps longer than *max_gap_sec* are left as zero — they are likely
    true inter-block padding, not tracking dropouts.

    Parameters
    ----------
    xyz_arr:
        Array of shape ``(T, N_markers, 3)``.
    samplerate:
        Frame rate in Hz, used to convert *max_gap_sec* to samples.
    max_gap_sec:
        Maximum gap duration to fill (default 0.5 s).  Longer gaps
        are left untouched.
    method:
        Interpolation method: ``'pchip'`` (default, monotone cubic,
        preserves local extrema) or ``'linear'``.
    inplace:
        If True modify *xyz_arr* in place; otherwise return a copy.

    Returns
    -------
    np.ndarray, shape ``(T, N_markers, 3)``
        Array with short gaps interpolated.

    Examples
    --------
    >>> xyz_clean = fill_gaps(session.xyz.data, session.xyz.samplerate,
    ...                       max_gap_sec=0.3)
    >>> session.xyz.data = xyz_clean
    """
    from scipy.interpolate import PchipInterpolator, interp1d

    arr = xyz_arr if inplace else xyz_arr.copy().astype(np.float64)
    max_gap_samp = int(round(max_gap_sec * samplerate))

    gap_mask, gap_periods = detect_gaps(arr)

    if gap_periods.shape[0] == 0:
        return arr

    T, N_markers, N_dims = arr.shape

    for gap_start, gap_stop in gap_periods:
        gap_len = gap_stop - gap_start
        if gap_len > max_gap_samp:
            continue                   # too long — leave as zero

        # Need at least one valid sample on each side
        if gap_start == 0 or gap_stop >= T:
            continue

        # Find the nearest valid sample before and after the gap
        left_end  = gap_start - 1
        right_beg = gap_stop

        # Gather a small neighbourhood for interpolation context
        ctx_half = max(gap_len * 3, int(round(0.3 * samplerate)))
        i0 = max(0,   left_end  - ctx_half)
        i1 = min(T-1, right_beg + ctx_half)

        ctx_range  = np.arange(i0, i1 + 1)
        valid_mask = ~gap_mask[i0:i1 + 1]

        if valid_mask.sum() < 4:
            continue   # not enough neighbours for cubic interp

        t_valid = ctx_range[valid_mask]
        t_fill  = np.arange(gap_start, gap_stop)

        for mi in range(N_markers):
            for di in range(N_dims):
                y_valid = arr[t_valid, mi, di]

                try:
                    if method == "pchip":
                        f = PchipInterpolator(t_valid, y_valid,
                                              extrapolate=False)
                        y_fill = f(t_fill)
                    else:
                        f = interp1d(t_valid, y_valid, kind="linear",
                                     bounds_error=False,
                                     fill_value=(y_valid[0], y_valid[-1]))
                        y_fill = f(t_fill)
                except Exception:
                    continue

                # Only write back where interpolation gave finite values
                finite = np.isfinite(y_fill)
                arr[t_fill[finite], mi, di] = y_fill[finite]

    return arr


# ---------------------------------------------------------------------------
# Convenience: fill gaps on an NBDxyz object in-place
# ---------------------------------------------------------------------------

def fill_xyz_gaps(
    xyz_obj,
    max_gap_sec: float = 0.5,
    method:      str   = "pchip",
) -> None:
    """Fill gaps in an NBDxyz object's data array in-place.

    Parameters
    ----------
    xyz_obj:
        ``NBDxyz`` instance.  Must have ``.data`` (ndarray) and
        ``.samplerate`` (float) attributes.
    max_gap_sec:
        Maximum gap to fill (default 0.5 s).
    method:
        ``'pchip'`` or ``'linear'``.

    Examples
    --------
    >>> from neurobox.io import fill_xyz_gaps
    >>> fill_xyz_gaps(session.xyz, max_gap_sec=0.3)
    """
    if xyz_obj.data is None:
        raise RuntimeError("NBDxyz.data is not loaded.")
    before = int((np.abs(xyz_obj.data).max(axis=(1,2))
                  <= 2 * float(np.finfo(np.float32).eps)).sum())
    xyz_obj.data = fill_gaps(
        xyz_obj.data,
        samplerate  = xyz_obj.samplerate,
        max_gap_sec = max_gap_sec,
        method      = method,
        inplace     = True,
    )
    after = int((np.abs(xyz_obj.data).max(axis=(1,2))
                 <= 2 * float(np.finfo(np.float32).eps)).sum())
    print(f"fill_xyz_gaps: {before} → {after} gap frames remaining "
          f"(max_gap={max_gap_sec} s)")
