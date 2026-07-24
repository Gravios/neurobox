# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""
neurobox.analysis.lfp._within_ranges_engine
============================================
Cython kernel for :func:`neurobox.analysis.lfp.within_ranges` matrix mode.

Algorithm
---------
Sweep-line with active counter.  Given the merged time-ordered stream of
range starts, range stops, and query points:

* Maintain a small ``int[n_labels]`` array ``active`` whose entry ``k``
  is the number of currently-open ranges with label ``k``.
* Walk the merged stream in time order:

  - On a *start* event for label ``k``: ``active[k] += 1``.
  - On a *stop*  event for label ``k``: ``active[k] -= 1``.
  - On a *point* (query) event at index ``i``: snapshot
    ``out[i, k] = (active[k] > 0)`` for all ``k``.

Tie-breaking: at a tied time ``t``, *starts* are processed before
*points* before *stops*.  This realises the labbox inclusive-on-both-
ends semantics — a query at ``x = start_k`` or ``x = stop_k`` lies in
range ``k``.

Complexity
----------
* Time:   ``O(n_events + n_points × n_labels)``
* Memory: ``O(n_points × n_labels)`` for the output (unavoidable —
  the caller asked for a per-(point, label) boolean) plus
  ``O(n_labels)`` for the active counter.

Compared with the per-label Python loop in
:mod:`._within_ranges_python_fallback`, this is constant-factor
improvement (no per-iteration Python overhead, no per-iteration numpy
allocation) and tightens for large ``n_labels``.

Public entry point
------------------
:func:`within_ranges_matrix_engine`
    Inputs are the *merged* and *sorted* event arrays — the caller does
    the sort in numpy then hands the kernel three small arrays.  This
    keeps the .pyx narrow and lets us reuse numpy's stable sort.

Integer typing
--------------
All integer buffers and scalars use ``cnp.int64_t`` rather than bare C
``long``.  The two coincide on LP64 (Linux, macOS) but ``long`` is only
32 bits under Windows' LLP64 model, which would make the buffer
declarations disagree with the ``np.int64`` arrays the Python callers
build and raise ``ValueError: Buffer dtype mismatch`` at runtime.
Fixed-width types keep the declared buffer type and the allocated array
dtype in lockstep on every platform.  Do not "simplify" these back to
``long``.
"""

import numpy as np
cimport numpy as cnp
from libc.stdint cimport uint8_t

cnp.import_array()


# Event-kind codes.  Defined as module-level Python ints so the
# fallback module can use the same constants.
KIND_START = 0
KIND_POINT = 1
KIND_STOP  = 2


def within_ranges_matrix_engine(
    cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] event_kind not None,
    cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] event_label not None,
    cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] event_point_idx not None,
    cnp.int64_t n_points,
    cnp.int64_t n_labels,
):
    """Run the sweep over a pre-merged and sorted event stream.

    Parameters
    ----------
    event_kind:
        ``int64`` array of length ``n_events`` with values 0 (start),
        1 (point), or 2 (stop).  Must be sorted by event time, with
        the start/point/stop tie-breaking already applied.
    event_label:
        ``int64`` array of length ``n_events``.  For start/stop
        events: the 0-indexed label.  For point events: ignored
        (typically -1).
    event_point_idx:
        ``int64`` array of length ``n_events``.  For point events:
        the 0-indexed position in the original (un-sorted) input.
        For start/stop events: ignored (typically -1).
    n_points:
        Number of query points (= number of point events).
    n_labels:
        Number of distinct labels.

    Returns
    -------
    out : ndarray
        ``(n_points, n_labels)`` ``uint8`` array; ``1`` where the
        point lies in any range with that label, ``0`` otherwise.
    """
    cdef cnp.int64_t n_events = event_kind.shape[0]
    cdef cnp.ndarray[uint8_t, ndim=2, mode="c"] out = (
        np.zeros((n_points, n_labels), dtype=np.uint8)
    )
    cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] active = (
        np.zeros(n_labels, dtype=np.int64)
    )
    cdef cnp.int64_t i, k, kind, lab, pidx

    for i in range(n_events):
        kind = event_kind[i]
        if kind == 0:                                # start
            lab = event_label[i]
            active[lab] += 1
        elif kind == 2:                              # stop
            lab = event_label[i]
            active[lab] -= 1
        else:                                        # point (kind == 1)
            pidx = event_point_idx[i]
            for k in range(n_labels):
                out[pidx, k] = 1 if active[k] > 0 else 0
    return out
