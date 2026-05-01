"""
neurobox.analysis.kinematics.helpers
=====================================
Small utilities shared by the kinematic-feature implementations.

Contents
--------
:func:`finite_nonzero_mask`
    Port of MTA's ``nniz`` — boolean mask of samples that are
    finite (no NaN, no inf) and non-zero across all extra dimensions.
:func:`zscore_with_mask`
    Port of MTA's ``nunity`` — column-wise z-score that ignores
    invalid (non-finite, zero) samples when estimating mean/std.

For the inter-marker angle table (MTADang.create), use the
:class:`neurobox.dtype.NBDang` class directly:

>>> ang = NBDang.from_xyz(xyz)
>>> azimuth = ang.between('head_back', 'head_front')[:, 0]
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


# ─────────────────────────────────────────────────────────────────────────── #
# nniz — finite-and-non-zero mask                                              #
# ─────────────────────────────────────────────────────────────────────────── #

def finite_nonzero_mask(arr: np.ndarray) -> np.ndarray:
    """Boolean mask of samples that are finite AND non-zero.

    Mirrors MTA's :file:`utilities/nniz.m`.  A sample is "valid" iff
    it has no NaN, no Inf, and no exact zero across every dimension
    after the first.

    Parameters
    ----------
    arr:
        Array of shape ``(T, ...)``.  Any number of trailing dims OK.

    Returns
    -------
    mask : np.ndarray of bool, shape ``(T,)``
        True where the row passes all three filters.

    Examples
    --------
    >>> a = np.array([[1, 2], [np.nan, 3], [0, 4], [5, 0]])
    >>> finite_nonzero_mask(a)
    array([ True, False, False, False])
    """
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return np.isfinite(arr) & (arr != 0)
    # All-finite AND all-non-zero across every trailing dim.
    flat = arr.reshape(arr.shape[0], -1)
    valid = np.isfinite(flat) & (flat != 0)
    return valid.all(axis=1)


# ─────────────────────────────────────────────────────────────────────────── #
# nunity — masked z-score                                                      #
# ─────────────────────────────────────────────────────────────────────────── #

def zscore_with_mask(
    arr:           np.ndarray,
    mean:          np.ndarray | None = None,
    std:           np.ndarray | None = None,
    fill_value:    float | None = np.nan,
    drop_outlier_pct: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Column-wise z-score that ignores invalid samples.

    Mirrors MTA's :file:`utilities/transforms/nunity.m`.  Computes
    mean and std over the rows that pass :func:`finite_nonzero_mask`,
    optionally trimming outliers, then subtracts and divides per
    column.  Invalid rows are filled with ``fill_value`` (default
    NaN; MATLAB used NaN as well).

    Parameters
    ----------
    arr:
        ``(T, n_cols)`` data array (extra trailing dims OK).
    mean, std:
        Pre-computed normalisation statistics.  If both are provided,
        they are used directly; otherwise computed from valid rows.
    fill_value:
        Value to write into invalid rows of the output.  Default NaN.
    drop_outlier_pct:
        If given as ``(lo, hi)``, percentile boundaries — values
        outside this range are excluded from the mean/std estimate
        (same column-wise as MATLAB's ``drpOutPrctile``).

    Returns
    -------
    U : np.ndarray
        Same shape as *arr*.  Invalid rows are *fill_value*.
    mean_used, std_used : np.ndarray
        Statistics used for normalisation; useful when applying the
        same transform to a held-out signal.
    """
    arr = np.asarray(arr, dtype=np.float64)
    mask = finite_nonzero_mask(arr)
    valid = arr[mask]

    if drop_outlier_pct is not None and valid.size:
        lo, hi = drop_outlier_pct
        # Per-column percentile boundaries
        bounds = np.percentile(valid, [lo, hi], axis=0)
        within = (valid > bounds[0]) & (valid < bounds[1])
        # Column-wise mean/std using only the inside-bounds samples
        if mean is None:
            mean = np.array([
                valid[within[:, c], c].mean() if within[:, c].any() else np.nan
                for c in range(valid.shape[-1] if valid.ndim > 1 else 1)
            ])
        if std is None:
            std = np.array([
                valid[within[:, c], c].std()  if within[:, c].any() else np.nan
                for c in range(valid.shape[-1] if valid.ndim > 1 else 1)
            ])
    else:
        if mean is None:
            mean = valid.mean(axis=0) if valid.size else np.zeros(arr.shape[1:])
        if std is None:
            std = valid.std(axis=0)  if valid.size else np.ones(arr.shape[1:])

    out = np.full_like(arr, fill_value)
    safe_std = np.where(std == 0, 1.0, std)
    out[mask] = (arr[mask] - mean) / safe_std
    return out, np.asarray(mean), np.asarray(std)
