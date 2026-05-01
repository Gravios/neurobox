# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""
neurobox.analysis.spikes._ccg_engine
=====================================
Cython kernel for cross-correlogram counting.

Direct port of :file:`labbox/TF/CCGHeart.c` (Ken Harris).  The algorithm is a
double-sided sweep: for every centre spike, walk backwards and forwards
through the (time-sorted) spike list as long as the time difference is
within ``half_bins * bin_size``, accumulating bin counts in a 3-D
histogram.

Public entry points
-------------------
:func:`compute_ccg_counts`
    Counts only.  Output: ``(n_bins, n_groups, n_groups)`` ``int64``.
:func:`compute_ccg_counts_with_pairs`
    Counts plus pair indices.  Two-pass implementation (counts first,
    then fills a pre-sized ``(n_pairs, 2)`` ``int64`` array) so we avoid
    the 800 MB up-front allocation in the original C version.

Both expect:
* ``times``    — ``float64`` array, **sorted ascending**.
* ``clu``      — ``int64`` array of cluster indices in ``[0, n_groups)``.
* ``bin_size`` — float, in the same time units as ``times``.
* ``half_bins`` — int, total bins = ``1 + 2*half_bins``.
* ``n_groups`` — int, must satisfy ``clu.max() < n_groups``.

Indexing convention
-------------------
Output index for one count: ``[bin, mark1, mark2]`` where ``mark1`` is the
**centre** spike's group and ``mark2`` the **other** spike's group.
``bin = half_bins`` is lag 0.

This is **0-indexed** clusters, deliberately differing from the labbox
1-indexed convention (which existed only as a guard against missing
group labels — Python is happy with ``-1`` for that).
"""

import numpy as np
cimport numpy as cnp
from libc.math cimport fabs, floor

cnp.import_array()


# ─────────────────────────────────────────────────────────────────────────── #
# Counts only                                                                  #
# ─────────────────────────────────────────────────────────────────────────── #

def compute_ccg_counts(
    cnp.ndarray[double, ndim=1, mode="c"] times not None,
    cnp.ndarray[long, ndim=1, mode="c"] clu not None,
    double bin_size,
    long half_bins,
    long n_groups,
):
    """Compute the bin-count histogram of pairs.

    Returns
    -------
    counts : ndarray, shape (n_bins, n_groups, n_groups), dtype int64
        ``counts[b, m1, m2]`` is the number of (centre, other) spike pairs
        with centre group ``m1``, other group ``m2``, and lag falling in
        bin ``b``.  ``b = half_bins`` corresponds to lag 0.
    """
    if times.shape[0] != clu.shape[0]:
        raise ValueError(
            f"times and clu must have same length: "
            f"{times.shape[0]} vs {clu.shape[0]}"
        )
    if half_bins < 0:
        raise ValueError(f"half_bins must be ≥ 0, got {half_bins}")
    if bin_size <= 0:
        raise ValueError(f"bin_size must be > 0, got {bin_size}")
    if n_groups <= 0:
        raise ValueError(f"n_groups must be > 0, got {n_groups}")

    cdef long n_spikes = times.shape[0]
    cdef long n_bins = 1 + 2 * half_bins
    cdef double furthest_edge = bin_size * (half_bins + 0.5)

    cdef cnp.ndarray[long, ndim=3, mode="c"] counts = np.zeros(
        (n_bins, n_groups, n_groups), dtype=np.int64
    )

    # Local pointers / typed views for speed
    cdef double *times_ptr = &times[0]
    cdef long *clu_ptr = &clu[0]
    cdef long [:, :, ::1] counts_view = counts

    cdef long centre, second, mark1, mark2, bin_idx
    cdef double t1, t2, dt

    for centre in range(n_spikes):
        mark1 = clu_ptr[centre]
        # Skip spikes whose group is out of range (matches labbox's
        # treatment of clu==0 as "ignore", but in 0-indexed terms).
        if mark1 < 0 or mark1 >= n_groups:
            continue
        t1 = times_ptr[centre]

        # Walk backwards
        second = centre - 1
        while second >= 0:
            t2 = times_ptr[second]
            dt = t1 - t2  # > 0 here
            if dt > furthest_edge:
                break
            mark2 = clu_ptr[second]
            if 0 <= mark2 < n_groups:
                # bin = half_bins + round((t2 - t1) / bin_size)
                # Using floor(0.5 + x) to match the C version exactly.
                bin_idx = half_bins + <long>floor(0.5 + (t2 - t1) / bin_size)
                if 0 <= bin_idx < n_bins:
                    counts_view[bin_idx, mark1, mark2] += 1
            second -= 1

        # Walk forwards
        second = centre + 1
        while second < n_spikes:
            t2 = times_ptr[second]
            dt = t2 - t1  # > 0 here
            # Note: the C code uses >= for the forward direction and > for
            # the backward direction (asymmetric).  We replicate that
            # exactly to be byte-for-byte compatible with labbox CCG.
            if dt >= furthest_edge:
                break
            mark2 = clu_ptr[second]
            if 0 <= mark2 < n_groups:
                bin_idx = half_bins + <long>floor(0.5 + (t2 - t1) / bin_size)
                if 0 <= bin_idx < n_bins:
                    counts_view[bin_idx, mark1, mark2] += 1
            second += 1

    return counts


# ─────────────────────────────────────────────────────────────────────────── #
# Counts with pair indices                                                     #
# ─────────────────────────────────────────────────────────────────────────── #

def compute_ccg_counts_with_pairs(
    cnp.ndarray[double, ndim=1, mode="c"] times not None,
    cnp.ndarray[long, ndim=1, mode="c"] clu not None,
    double bin_size,
    long half_bins,
    long n_groups,
):
    """Compute counts AND the spike-index pairs that contributed.

    Two-pass implementation: pass 1 counts pairs, pass 2 fills.  This
    avoids the 100M-pair upfront allocation in the original C version.

    Returns
    -------
    counts : ndarray, shape (n_bins, n_groups, n_groups), dtype int64
        Same as :func:`compute_ccg_counts`.
    pairs : ndarray, shape (n_pairs, 2), dtype int64
        Each row is ``[centre_index, other_index]`` into the original
        ``times`` / ``clu`` arrays.
    """
    if times.shape[0] != clu.shape[0]:
        raise ValueError(
            f"times and clu must have same length: "
            f"{times.shape[0]} vs {clu.shape[0]}"
        )
    if half_bins < 0:
        raise ValueError(f"half_bins must be ≥ 0, got {half_bins}")
    if bin_size <= 0:
        raise ValueError(f"bin_size must be > 0, got {bin_size}")
    if n_groups <= 0:
        raise ValueError(f"n_groups must be > 0, got {n_groups}")

    cdef long n_spikes = times.shape[0]
    cdef long n_bins = 1 + 2 * half_bins
    cdef double furthest_edge = bin_size * (half_bins + 0.5)
    cdef long centre, second, mark1, mark2, bin_idx
    cdef double t1, t2, dt
    cdef long n_pairs = 0

    cdef double *times_ptr = &times[0]
    cdef long *clu_ptr = &clu[0]

    # ── Pass 1: count pairs ──────────────────────────────────────────────── #
    for centre in range(n_spikes):
        mark1 = clu_ptr[centre]
        if mark1 < 0 or mark1 >= n_groups:
            continue
        t1 = times_ptr[centre]
        second = centre - 1
        while second >= 0:
            t2 = times_ptr[second]
            if t1 - t2 > furthest_edge:
                break
            mark2 = clu_ptr[second]
            if 0 <= mark2 < n_groups:
                bin_idx = half_bins + <long>floor(0.5 + (t2 - t1) / bin_size)
                if 0 <= bin_idx < n_bins:
                    n_pairs += 1
            second -= 1
        second = centre + 1
        while second < n_spikes:
            t2 = times_ptr[second]
            if t2 - t1 >= furthest_edge:
                break
            mark2 = clu_ptr[second]
            if 0 <= mark2 < n_groups:
                bin_idx = half_bins + <long>floor(0.5 + (t2 - t1) / bin_size)
                if 0 <= bin_idx < n_bins:
                    n_pairs += 1
            second += 1

    # ── Pass 2: allocate and fill ────────────────────────────────────────── #
    cdef cnp.ndarray[long, ndim=3, mode="c"] counts = np.zeros(
        (n_bins, n_groups, n_groups), dtype=np.int64
    )
    cdef cnp.ndarray[long, ndim=2, mode="c"] pairs = np.empty(
        (n_pairs, 2), dtype=np.int64
    )
    cdef long [:, :, ::1] counts_view = counts
    cdef long [:, ::1] pairs_view = pairs
    cdef long pair_idx = 0

    for centre in range(n_spikes):
        mark1 = clu_ptr[centre]
        if mark1 < 0 or mark1 >= n_groups:
            continue
        t1 = times_ptr[centre]
        second = centre - 1
        while second >= 0:
            t2 = times_ptr[second]
            if t1 - t2 > furthest_edge:
                break
            mark2 = clu_ptr[second]
            if 0 <= mark2 < n_groups:
                bin_idx = half_bins + <long>floor(0.5 + (t2 - t1) / bin_size)
                if 0 <= bin_idx < n_bins:
                    counts_view[bin_idx, mark1, mark2] += 1
                    pairs_view[pair_idx, 0] = centre
                    pairs_view[pair_idx, 1] = second
                    pair_idx += 1
            second -= 1
        second = centre + 1
        while second < n_spikes:
            t2 = times_ptr[second]
            if t2 - t1 >= furthest_edge:
                break
            mark2 = clu_ptr[second]
            if 0 <= mark2 < n_groups:
                bin_idx = half_bins + <long>floor(0.5 + (t2 - t1) / bin_size)
                if 0 <= bin_idx < n_bins:
                    counts_view[bin_idx, mark1, mark2] += 1
                    pairs_view[pair_idx, 0] = centre
                    pairs_view[pair_idx, 1] = second
                    pair_idx += 1
            second += 1

    return counts, pairs
