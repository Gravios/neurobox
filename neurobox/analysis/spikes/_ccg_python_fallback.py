"""
neurobox.analysis.spikes._ccg_python_fallback
==============================================
Pure-Python / numpy reference implementation of the CCG kernel.

Used as a fallback when the compiled Cython extension
(``_ccg_engine``) is not available — for example, when running in an
environment without a C compiler, or when the source distribution was
installed without building the extension.

Roughly 5-15× slower than the Cython version on typical session sizes
(10⁵ spikes) but algorithmically identical, so unit tests pass against
both implementations.

Implementation
--------------
Vectorised double-sided sweep using :func:`numpy.searchsorted` to find
the time-window bounds for each centre spike, then an iterated histogram
fill with :func:`numpy.add.at` (the unbuffered scatter-add).  The outer
Python loop is over centres rather than pairs, so for ``N`` spikes with
average ``k`` neighbours in the window, runtime is ``O(N k)`` — same as
the C kernel.
"""

from __future__ import annotations

import numpy as np


def _validate_inputs(
    times: np.ndarray,
    clu: np.ndarray,
    bin_size: float,
    half_bins: int,
    n_groups: int,
) -> tuple[np.ndarray, np.ndarray]:
    if times.shape[0] != clu.shape[0]:
        raise ValueError(
            f"times and clu must have same length: {times.shape[0]} vs {clu.shape[0]}"
        )
    if half_bins < 0:
        raise ValueError(f"half_bins must be ≥ 0, got {half_bins}")
    if bin_size <= 0:
        raise ValueError(f"bin_size must be > 0, got {bin_size}")
    if n_groups <= 0:
        raise ValueError(f"n_groups must be > 0, got {n_groups}")
    return (np.ascontiguousarray(times, dtype=np.float64),
            np.ascontiguousarray(clu, dtype=np.int64))


def compute_ccg_counts(
    times: np.ndarray,
    clu: np.ndarray,
    bin_size: float,
    half_bins: int,
    n_groups: int,
) -> np.ndarray:
    """Counts-only fallback.  See :mod:`_ccg_engine` for semantics."""
    times, clu = _validate_inputs(times, clu, bin_size, half_bins, n_groups)
    n_spikes = times.shape[0]
    n_bins = 1 + 2 * half_bins
    furthest_edge = bin_size * (half_bins + 0.5)
    counts = np.zeros((n_bins, n_groups, n_groups), dtype=np.int64)

    if n_spikes < 2:
        return counts

    # For each centre, find the contiguous time-window using searchsorted.
    # Backward window: [t1 - furthest_edge, t1)  → strict (matches dt > furthest_edge break)
    # Forward window:  (t1, t1 + furthest_edge) → strict (matches dt >= furthest_edge break)
    # The C asymmetry: backward uses '>' (so dt == furthest_edge is INCLUDED),
    # forward uses '>=' (so dt == furthest_edge is EXCLUDED).
    for centre in range(n_spikes):
        m1 = clu[centre]
        if m1 < 0 or m1 >= n_groups:
            continue
        t1 = times[centre]

        # Backward: indices [lo, centre) where t1 - times[i] <= furthest_edge.
        lo = np.searchsorted(times, t1 - furthest_edge, side="left")
        # Forward: indices (centre, hi) where times[i] - t1 < furthest_edge.
        hi = np.searchsorted(times, t1 + furthest_edge, side="left")

        if lo == centre and hi == centre + 1:
            continue

        # Combine both halves
        if lo < centre:
            t2_back = times[lo:centre]
            m2_back = clu[lo:centre]
            valid_back = (m2_back >= 0) & (m2_back < n_groups)
            if valid_back.any():
                bins_back = (half_bins + np.floor(0.5 + (t2_back - t1) / bin_size)).astype(np.int64)
                in_range = valid_back & (bins_back >= 0) & (bins_back < n_bins)
                np.add.at(counts, (bins_back[in_range], m1, m2_back[in_range]), 1)

        if hi > centre + 1:
            t2_fwd = times[centre + 1:hi]
            m2_fwd = clu[centre + 1:hi]
            valid_fwd = (m2_fwd >= 0) & (m2_fwd < n_groups)
            if valid_fwd.any():
                bins_fwd = (half_bins + np.floor(0.5 + (t2_fwd - t1) / bin_size)).astype(np.int64)
                in_range = valid_fwd & (bins_fwd >= 0) & (bins_fwd < n_bins)
                np.add.at(counts, (bins_fwd[in_range], m1, m2_fwd[in_range]), 1)

    return counts


def compute_ccg_counts_with_pairs(
    times: np.ndarray,
    clu: np.ndarray,
    bin_size: float,
    half_bins: int,
    n_groups: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Counts + pairs fallback.

    Returns
    -------
    counts : ndarray, shape (n_bins, n_groups, n_groups), int64
    pairs : ndarray, shape (n_pairs, 2), int64
    """
    times, clu = _validate_inputs(times, clu, bin_size, half_bins, n_groups)
    n_spikes = times.shape[0]
    n_bins = 1 + 2 * half_bins
    furthest_edge = bin_size * (half_bins + 0.5)
    counts = np.zeros((n_bins, n_groups, n_groups), dtype=np.int64)

    if n_spikes < 2:
        return counts, np.empty((0, 2), dtype=np.int64)

    pair_chunks: list[np.ndarray] = []

    for centre in range(n_spikes):
        m1 = clu[centre]
        if m1 < 0 or m1 >= n_groups:
            continue
        t1 = times[centre]
        lo = np.searchsorted(times, t1 - furthest_edge, side="left")
        hi = np.searchsorted(times, t1 + furthest_edge, side="left")

        if lo < centre:
            others = np.arange(lo, centre)
            t2 = times[others]
            m2 = clu[others]
            valid = (m2 >= 0) & (m2 < n_groups)
            if valid.any():
                bins = (half_bins + np.floor(0.5 + (t2 - t1) / bin_size)).astype(np.int64)
                in_range = valid & (bins >= 0) & (bins < n_bins)
                if in_range.any():
                    np.add.at(counts, (bins[in_range], m1, m2[in_range]), 1)
                    chunk = np.empty((in_range.sum(), 2), dtype=np.int64)
                    chunk[:, 0] = centre
                    chunk[:, 1] = others[in_range]
                    pair_chunks.append(chunk)

        if hi > centre + 1:
            others = np.arange(centre + 1, hi)
            t2 = times[others]
            m2 = clu[others]
            valid = (m2 >= 0) & (m2 < n_groups)
            if valid.any():
                bins = (half_bins + np.floor(0.5 + (t2 - t1) / bin_size)).astype(np.int64)
                in_range = valid & (bins >= 0) & (bins < n_bins)
                if in_range.any():
                    np.add.at(counts, (bins[in_range], m1, m2[in_range]), 1)
                    chunk = np.empty((in_range.sum(), 2), dtype=np.int64)
                    chunk[:, 0] = centre
                    chunk[:, 1] = others[in_range]
                    pair_chunks.append(chunk)

    if pair_chunks:
        pairs = np.concatenate(pair_chunks, axis=0)
    else:
        pairs = np.empty((0, 2), dtype=np.int64)
    return counts, pairs
