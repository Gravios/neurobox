"""
neurobox.analysis.lfp.ranges
=============================
Epoch / range arithmetic — union, intersection, difference, complement.

Port of three labbox / MTA utilities:

==============================  =============================================
labbox / MTA                     neurobox
==============================  =============================================
MTA/utilities/JoinRanges.m       :func:`join_ranges`
MTA/utilities/IntersectRanges.m  :func:`intersect_ranges`
MTA/utilities/SubstractRanges.m  :func:`subtract_ranges`
(De Morgan inversion)            :func:`complement_ranges`
==============================  =============================================

The MATLAB versions rely on De Morgan inversion via ``IntersectRanges``
(``A ∪ B = (Aᶜ ∩ Bᶜ)ᶜ``).  This port uses a direct sweep-line algorithm
which is both clearer and more numerically robust at infinity.

Conventions
-----------
* Input ranges are ``(N, 2)`` ``[start, stop]`` arrays.  Empty
  ``(0, 2)`` arrays are accepted everywhere and produce the
  mathematically correct identity (e.g. ``A ∪ ∅ = A``).
* Outputs are **canonical**: sorted by start, non-overlapping, and
  with no zero-length intervals.  Adjacent ranges that touch at a
  single point (``r1.stop == r2.start``) are merged for ``join_ranges``
  but kept separate for ``intersect_ranges``.
* All operations work in time units consistent with the inputs (samples
  or seconds — the functions don't care which).

Why not :class:`NBEpoch`?
-------------------------
These primitives operate on plain numpy arrays so they can be called
from arbitrary code paths.  :class:`NBEpoch` adds samplerate, sync,
and label tracking — which is great for end-user code but unwanted
when you just want to compute ``A ∪ B`` inside an analysis function.
A future ``NBEpoch.union(other)`` / ``.intersection()`` / ``.difference()``
can wrap these primitives.
"""

from __future__ import annotations

import numpy as np


# ─────────────────────────────────────────────────────────────────────────── #
# Internal: canonicalise ranges                                                #
# ─────────────────────────────────────────────────────────────────────────── #

def _canonicalise(ranges: np.ndarray) -> np.ndarray:
    """Sort, merge overlaps, and drop zero-length intervals."""
    arr = np.asarray(ranges, dtype=np.float64)
    if arr.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    if arr.ndim == 1 and arr.size == 2:
        arr = arr.reshape(1, 2)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"ranges must be (N, 2); got shape {arr.shape}")
    if np.any(arr[:, 1] < arr[:, 0]):
        raise ValueError("each range must satisfy start ≤ stop")

    # Sort by start, then by stop
    order = np.lexsort((arr[:, 1], arr[:, 0]))
    arr = arr[order]

    # Merge overlapping / touching ranges
    out_starts = [arr[0, 0]]
    out_stops  = [arr[0, 1]]
    for s, e in arr[1:]:
        if s <= out_stops[-1]:                 # overlap or touch → merge
            out_stops[-1] = max(out_stops[-1], e)
        else:
            out_starts.append(s)
            out_stops.append(e)

    out = np.column_stack([out_starts, out_stops])
    # Drop zero-length intervals (start == stop after merging)
    out = out[out[:, 1] > out[:, 0]]
    return out


# ─────────────────────────────────────────────────────────────────────────── #
# join_ranges                                                                  #
# ─────────────────────────────────────────────────────────────────────────── #

def join_ranges(*range_arrays: np.ndarray) -> np.ndarray:
    """Union of one or more range sets.

    Port of :file:`MTA/utilities/JoinRanges.m`, generalised to any
    number of inputs (the MATLAB version takes exactly two and emulates
    the union via ``A ∪ B = (Aᶜ ∩ Bᶜ)ᶜ``).

    Parameters
    ----------
    *range_arrays:
        One or more ``(N, 2)`` arrays of ``[start, stop]`` ranges.

    Returns
    -------
    union : ndarray of shape ``(M, 2)``, dtype ``float64``
        Canonicalised union — sorted, non-overlapping, no zero-length
        intervals.

    Examples
    --------
    >>> A = np.array([[0, 5], [10, 15]])
    >>> B = np.array([[3, 7], [20, 25]])
    >>> join_ranges(A, B)
    array([[ 0.,  7.],
           [10., 15.],
           [20., 25.]])
    """
    if len(range_arrays) == 0:
        return np.empty((0, 2), dtype=np.float64)
    pieces = [r for r in range_arrays if np.asarray(r).size > 0]
    if not pieces:
        return np.empty((0, 2), dtype=np.float64)
    stacked = np.vstack([np.asarray(r, dtype=np.float64).reshape(-1, 2)
                         for r in pieces])
    return _canonicalise(stacked)


# ─────────────────────────────────────────────────────────────────────────── #
# intersect_ranges                                                             #
# ─────────────────────────────────────────────────────────────────────────── #

def intersect_ranges(R1: np.ndarray, R2: np.ndarray) -> np.ndarray:
    """Intersection of two range sets.

    Port of :file:`MTA/utilities/IntersectRanges.m` — used internally by
    the MATLAB ``JoinRanges`` and ``SubstractRanges``.

    Parameters
    ----------
    R1, R2:
        ``(N, 2)`` and ``(M, 2)`` arrays of ``[start, stop]`` ranges.
        Need not be canonicalised on input.

    Returns
    -------
    intersection : ndarray of shape ``(K, 2)``
        Sorted, non-overlapping intervals where points lie in **both**
        ``R1`` and ``R2``.

    Examples
    --------
    >>> A = np.array([[0, 10], [20, 30]])
    >>> B = np.array([[5, 25]])
    >>> intersect_ranges(A, B)
    array([[ 5., 10.],
           [20., 25.]])
    """
    A = _canonicalise(R1)
    B = _canonicalise(R2)
    if A.shape[0] == 0 or B.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float64)

    # Two-pointer sweep over canonical inputs
    out_starts: list[float] = []
    out_stops:  list[float] = []
    i = j = 0
    while i < A.shape[0] and j < B.shape[0]:
        a0, a1 = A[i]
        b0, b1 = B[j]
        lo, hi = max(a0, b0), min(a1, b1)
        if lo < hi:
            out_starts.append(lo)
            out_stops.append(hi)
        # Advance the pointer whose range ends first
        if a1 < b1:
            i += 1
        else:
            j += 1
    if not out_starts:
        return np.empty((0, 2), dtype=np.float64)
    return np.column_stack([out_starts, out_stops])


# ─────────────────────────────────────────────────────────────────────────── #
# subtract_ranges                                                              #
# ─────────────────────────────────────────────────────────────────────────── #

def subtract_ranges(R1: np.ndarray, R2: np.ndarray) -> np.ndarray:
    """Set difference: time in ``R1`` but not in ``R2``.

    Port of :file:`MTA/utilities/SubstractRanges.m` (sic).

    The MATLAB implementation is ``IntersectRanges(R1, complement(R2))``.
    We compute the same result with a direct sweep that is robust when
    ``R2`` is empty (the MATLAB version accesses ``R2(1, 1)`` and crashes
    on empty input).

    Parameters
    ----------
    R1, R2:
        ``(N, 2)`` and ``(M, 2)`` arrays of ``[start, stop]`` ranges.

    Returns
    -------
    difference : ndarray of shape ``(K, 2)``
        Sorted, non-overlapping intervals.

    Examples
    --------
    >>> A = np.array([[0, 10]])
    >>> B = np.array([[3, 5], [7, 8]])
    >>> subtract_ranges(A, B)
    array([[ 0.,  3.],
           [ 5.,  7.],
           [ 8., 10.]])
    """
    A = _canonicalise(R1)
    B = _canonicalise(R2)
    if A.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float64)
    if B.shape[0] == 0:
        return A.copy()

    out_starts: list[float] = []
    out_stops:  list[float] = []
    j = 0
    for a0, a1 in A:
        cur = a0
        # Advance j until we find the first B-range that could overlap
        while j < B.shape[0] and B[j, 1] <= cur:
            j += 1
        # Carve out each overlap of this A-range with subsequent B-ranges
        k = j
        while k < B.shape[0] and B[k, 0] < a1:
            b0, b1 = B[k]
            if cur < b0:
                out_starts.append(cur)
                out_stops.append(min(b0, a1))
            cur = max(cur, b1)
            if cur >= a1:
                break
            k += 1
        if cur < a1:
            out_starts.append(cur)
            out_stops.append(a1)
    if not out_starts:
        return np.empty((0, 2), dtype=np.float64)
    return np.column_stack([out_starts, out_stops])


# ─────────────────────────────────────────────────────────────────────────── #
# complement_ranges                                                            #
# ─────────────────────────────────────────────────────────────────────────── #

def complement_ranges(
    ranges: np.ndarray,
    extent: tuple[float, float] | None = None,
) -> np.ndarray:
    """Complement: gaps between ranges, optionally bounded by ``extent``.

    Equivalent to the MATLAB ``NotR = [[-inf, r(1,1)]; [r(1:end-1,2),
    r(2:end,1)]; [r(end,2), inf]]`` idiom used inside ``JoinRanges`` /
    ``SubstractRanges``.  Useful as a building block when you need
    "everything that's not in this set of epochs".

    Parameters
    ----------
    ranges:
        ``(N, 2)`` array of ``[start, stop]`` ranges.
    extent:
        Optional ``(t_min, t_max)`` tuple bounding the universe.  If
        ``None`` (default), the complement extends to ``±inf`` on each
        side.

    Returns
    -------
    complement : ndarray of shape ``(M, 2)``
        Sorted, non-overlapping intervals covering the complement of
        ``ranges`` within ``extent`` (or ``(-inf, +inf)``).

    Examples
    --------
    >>> A = np.array([[5, 10], [20, 30]])
    >>> complement_ranges(A, extent=(0, 40))
    array([[ 0.,  5.],
           [10., 20.],
           [30., 40.]])
    """
    R = _canonicalise(ranges)
    t_lo = -np.inf if extent is None else float(extent[0])
    t_hi =  np.inf if extent is None else float(extent[1])
    if t_hi < t_lo:
        raise ValueError(f"extent[1] must be ≥ extent[0]; got {extent!r}")
    if R.shape[0] == 0:
        return np.array([[t_lo, t_hi]], dtype=np.float64) if t_hi > t_lo \
            else np.empty((0, 2), dtype=np.float64)

    out_starts: list[float] = []
    out_stops:  list[float] = []
    cur = t_lo
    for r0, r1 in R:
        if r0 > cur:
            out_starts.append(cur)
            out_stops.append(min(r0, t_hi))
        cur = max(cur, r1)
        if cur >= t_hi:
            break
    if cur < t_hi:
        out_starts.append(cur)
        out_stops.append(t_hi)
    if not out_starts:
        return np.empty((0, 2), dtype=np.float64)
    out = np.column_stack([out_starts, out_stops])
    return out[out[:, 1] > out[:, 0]]
