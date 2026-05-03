"""
neurobox.analysis.feature_dynamics.core
========================================
Numerical kernels for time-lagged feature relationships.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


__all__ = [
    "TimeLaggedResult",
    "time_lagged_mutual_information",
    "time_lagged_cross_correlation",
]


@dataclass
class TimeLaggedResult:
    """Output of :func:`time_lagged_mutual_information` /
    :func:`time_lagged_cross_correlation`.

    Attributes
    ----------
    values : np.ndarray, shape ``(n_lags, n_features, n_features)``
        Per-lag pairwise statistic — MI in bits or correlation
        coefficient in ``[-1, 1]``.
    lags : np.ndarray, shape ``(n_lags,)``
        Time shifts in samples.  Positive lag = column-2 shifted
        backwards (column-1 *leads* column-2).
    samplerate : float
        Sampling rate of the input features.
    measure : str
        ``'mutual_information'`` or ``'cross_correlation'``.
    """
    values:     np.ndarray
    lags:       np.ndarray
    samplerate: float
    measure:    str


# ─────────────────────────────────────────────────────────────────────── #
# Helpers                                                                    #
# ─────────────────────────────────────────────────────────────────────── #

def _validate_inputs(
    features:   np.ndarray,
    ind_mask:   np.ndarray,
    lags:       np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = np.asarray(features, dtype=np.float64)
    if features.ndim == 1:
        features = features[:, None]
    if features.ndim != 2:
        raise ValueError(
            f"features must be 1-D or 2-D, got shape {features.shape}"
        )
    ind_mask = np.asarray(ind_mask)

    # Convert int-indices to a boolean mask first (they don't have to
    # match features.shape[0] in length; they just need to all be < T).
    if ind_mask.dtype != bool:
        if ind_mask.ndim == 1 and np.issubdtype(ind_mask.dtype, np.integer):
            mask = np.zeros(features.shape[0], dtype=bool)
            ind_mask_int = ind_mask.astype(np.int64)
            if ind_mask_int.size > 0:
                if (ind_mask_int < 0).any() \
                        or (ind_mask_int >= features.shape[0]).any():
                    raise ValueError(
                        "integer ind_mask values must be in "
                        f"[0, {features.shape[0]})"
                    )
                mask[ind_mask_int] = True
            ind_mask = mask
        else:
            ind_mask = ind_mask.astype(bool)

    if ind_mask.shape != (features.shape[0],):
        raise ValueError(
            f"ind_mask shape {ind_mask.shape} must match "
            f"features.shape[0]={features.shape[0]}"
        )

    lags = np.asarray(lags, dtype=np.int64).ravel()
    return features, ind_mask, lags


def _shift_columnwise(arr: np.ndarray, shift: int) -> np.ndarray:
    """Shift *arr* by ``shift`` samples along axis 0 (circular).

    Mirrors MATLAB's ``circshift(sv, -diff(sbound([s,s+1])))``.
    """
    return np.roll(arr, -shift, axis=0)


def _hist2_normalised(
    x:         np.ndarray,
    y:         np.ndarray,
    edges_x:   np.ndarray,
    edges_y:   np.ndarray,
    n_total:   int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Joint and marginal probability mass functions over (*x*, *y*).

    Returns
    -------
    pxy : np.ndarray, shape ``(len(edges_x)-1, len(edges_y)-1)``
        Joint pmf.
    px : np.ndarray, shape ``(len(edges_x)-1,)``
    py : np.ndarray, shape ``(len(edges_y)-1,)``

    All three are normalised so that they sum to ``count / n_total``
    (so are ≤ 1).  Bins outside the edge ranges are dropped — same as
    MATLAB's ``hist2`` / ``histc`` semantics.
    """
    finite = np.isfinite(x) & np.isfinite(y)
    counts2d, _, _ = np.histogram2d(
        x[finite], y[finite], bins=[edges_x, edges_y],
    )
    countsx, _ = np.histogram(x[finite], bins=edges_x)
    countsy, _ = np.histogram(y[finite], bins=edges_y)
    pxy = counts2d / max(n_total, 1)
    px  = countsx  / max(n_total, 1)
    py  = countsy  / max(n_total, 1)
    return pxy, px, py


def _segments_around(
    arr:    np.ndarray,        # (T, n_features)
    centres: np.ndarray,       # (n_events,) integer indices
    seg_left:  int,
    seg_right: int,
) -> np.ndarray:
    """Pull windows ``[c-seg_left : c+seg_right]`` for every centre.

    Mirrors MATLAB's ``GetSegs(v(:,m), ind-30, 60, nan)``.

    Parameters
    ----------
    arr:
        ``(T, n_features)`` source.
    centres:
        Event-anchor sample indices.
    seg_left, seg_right:
        Samples taken before and after each centre.

    Returns
    -------
    np.ndarray, shape ``(seg_left + seg_right, n_events, n_features)``
        Windows.  Out-of-bounds segments are filled with NaN.
    """
    T, n_features = arr.shape
    seg_len = seg_left + seg_right
    out = np.full((seg_len, len(centres), n_features), np.nan,
                   dtype=np.float64)
    rel = np.arange(-seg_left, seg_right)
    for i, c in enumerate(centres):
        idxs = c + rel
        valid = (idxs >= 0) & (idxs < T)
        out[valid, i, :] = arr[idxs[valid], :]
    return out


# ─────────────────────────────────────────────────────────────────────── #
# Public API                                                                 #
# ─────────────────────────────────────────────────────────────────────── #

def time_lagged_mutual_information(
    features:   np.ndarray,
    ind_mask:   np.ndarray,
    *,
    lags:       Sequence[int] = range(-240, 241),
    edges:      Optional[np.ndarray] = None,
    samplerate: float = 1.0,
) -> TimeLaggedResult:
    """Time-lagged pairwise mutual information.

    Port of :file:`MTA/analysis/compute_time_lagged_mutual_information.m`
    (the inner numerical kernel from line 117 onwards).

    For each lag *s*, every other column is circularly shifted by *s*,
    then for every (i, j) feature pair the joint and marginal
    histograms over the masked samples are computed and combined into
    Shannon MI in bits::

        I(X; Y) = sum_{x, y} p(x, y) log2( p(x, y) / (p(x) p(y)) )

    Parameters
    ----------
    features:
        ``(T, n_features)`` feature matrix.  Caller is responsible for
        any normalisation / mapping.
    ind_mask:
        Boolean mask of length ``T``, OR a 1-D integer array of
        sample indices.  MI is computed only over the True samples
        (or selected indices).
    lags:
        Sequence of integer sample shifts.  Default ``-240..240``
        (≈ ±2 s at 120 Hz, matching MATLAB).
    edges:
        Histogram bin edges shared across both axes.  Default
        ``np.linspace(-1, 1, 64)`` matches MATLAB.
    samplerate:
        Forwarded to the result; used for downstream conversion of
        *lags* to seconds.

    Returns
    -------
    TimeLaggedResult
        ``values`` shape ``(n_lags, n_features, n_features)``.

    Notes
    -----
    The MATLAB original walks the sbound list as differences (``sv =
    circshift(sv, -diff(sbound))``).  This port shifts each column
    independently from the original — the result is identical but
    avoids accumulated floating-point error from chained shifts.
    """
    features, ind_mask, lags_arr = _validate_inputs(
        features, ind_mask, np.asarray(list(lags)),
    )
    if edges is None:
        edges = np.linspace(-1.0, 1.0, 64)
    edges = np.asarray(edges, dtype=np.float64)

    T, n_features = features.shape
    n_lags = len(lags_arr)
    n_total = int(ind_mask.sum())

    out = np.zeros((n_lags, n_features, n_features), dtype=np.float64)

    if n_total == 0:
        return TimeLaggedResult(out, lags_arr, samplerate,
                                  "mutual_information")

    # Pre-extract masked rows of features (fixed)
    fx = features[ind_mask, :]                # (n_total, n_features)

    for li, lag in enumerate(lags_arr):
        # Shift every column by `-lag` so that "lag" represents
        # "feature j evaluated `lag` samples after feature i"
        shifted = np.roll(features, -int(lag), axis=0)
        fy = shifted[ind_mask, :]             # (n_total, n_features)

        for i in range(n_features):
            for j in range(n_features):
                pxy, px, py = _hist2_normalised(
                    fx[:, i], fy[:, j], edges, edges, n_total,
                )
                # MI in bits, ignore zero / log(0) entries
                with np.errstate(divide="ignore", invalid="ignore"):
                    pmf_outer = np.outer(px, py)
                    ratio = np.where(
                        (pxy > 0) & (pmf_outer > 0),
                        pxy / pmf_outer, 1.0,
                    )
                    mi_terms = np.where(pxy > 0, pxy * np.log2(ratio), 0.0)
                out[li, i, j] = float(np.nansum(mi_terms))

    return TimeLaggedResult(
        values     = out,
        lags       = lags_arr,
        samplerate = samplerate,
        measure    = "mutual_information",
    )


def time_lagged_cross_correlation(
    features:    np.ndarray,
    centres:     np.ndarray,
    *,
    lags:        Sequence[int] = range(-240, 241),
    seg_left:    int = 30,
    seg_right:   int = 30,
    samplerate:  float = 1.0,
) -> TimeLaggedResult:
    """Time-lagged segment-anchored Pearson correlation.

    Port of :file:`MTA/analysis/compute_cross_correlation.m`
    (inner kernel from line 124 onwards).

    For each lag *s*, every column is circularly shifted by *s*; for
    every (i, j) feature pair, ``seg_left + seg_right``-sample windows
    are extracted around each centre index from feature *i* and the
    shifted feature *j*; the **mean of the per-segment Pearson
    correlations** is the output statistic.

    Parameters
    ----------
    features:
        ``(T, n_features)``.
    centres:
        ``(n_events,)`` integer sample indices specifying the centre
        of each correlation window.  Typically state midpoints
        (e.g. mid-walk samples).
    lags:
        Integer sample shifts.  Default ``-240..240``.
    seg_left, seg_right:
        Samples taken before / after each centre.  Default 30/30
        (60 samples ≈ 0.5 s at 120 Hz, matching MATLAB).
    samplerate:
        Forwarded to result.

    Returns
    -------
    TimeLaggedResult
        ``values`` shape ``(n_lags, n_features, n_features)``.
    """
    features = np.asarray(features, dtype=np.float64)
    if features.ndim == 1:
        features = features[:, None]
    centres = np.asarray(centres, dtype=np.int64).ravel()
    lags_arr = np.asarray(list(lags), dtype=np.int64).ravel()
    T, n_features = features.shape
    n_lags = len(lags_arr)

    out = np.zeros((n_lags, n_features, n_features), dtype=np.float64)
    if centres.size == 0:
        return TimeLaggedResult(out, lags_arr, samplerate,
                                  "cross_correlation")

    # Pre-extract un-shifted segments for every column
    base_segs = _segments_around(features, centres, seg_left, seg_right)

    for li, lag in enumerate(lags_arr):
        shifted = np.roll(features, -int(lag), axis=0)
        shifted_segs = _segments_around(shifted, centres, seg_left, seg_right)

        for i in range(n_features):
            xseg = base_segs[:, :, i]             # (seg_len, n_events)
            xmean = np.nanmean(xseg, axis=0)
            xstd  = np.nanstd (xseg, axis=0)
            for j in range(n_features):
                yseg = shifted_segs[:, :, j]
                ymean = np.nanmean(yseg, axis=0)
                ystd  = np.nanstd (yseg, axis=0)

                # Pearson r per event window, then average
                num = (xseg - xmean) * (yseg - ymean)
                den = xstd * ystd
                with np.errstate(divide="ignore", invalid="ignore"):
                    per_seg = np.where(den > 0, num / den, np.nan)
                # MATLAB averages across both segment-time and events
                out[li, i, j] = float(np.nanmean(per_seg))

    return TimeLaggedResult(
        values     = out,
        lags       = lags_arr,
        samplerate = samplerate,
        measure    = "cross_correlation",
    )
