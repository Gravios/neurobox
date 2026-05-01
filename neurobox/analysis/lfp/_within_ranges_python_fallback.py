"""
neurobox.analysis.lfp._within_ranges_python_fallback
=====================================================
Pure-numpy fallback for the matrix-mode of :func:`within_ranges`.

Used when the compiled Cython kernel
(:mod:`._within_ranges_engine`) is not available — for example in
environments without a C compiler, or when the source distribution
was installed without building the extension.

Algorithm
---------
Per-label binary-search approach (the same algorithm that lived in the
single ``within_ranges`` function before round 4).  For each label
``k``, sort the starts and stops belonging to label ``k`` and use
:func:`numpy.searchsorted` to count, for each query point, the number
of ``label-k`` ranges that have *started* but not yet *stopped*.

Complexity
----------
* Time:   ``O(n_labels × N log(R/n_labels))``
* Memory: ``O(n_points × n_labels)`` for the output

Slower than the Cython sweep at large ``n_labels`` (linear in
``n_labels`` with a numpy-bound constant factor), but algorithmically
correct.

Note on signature
-----------------
This fallback takes the **original** unsorted arrays
(``x``, ``starts``, ``stops``, ``range_label``).  The Cython kernel
takes a **pre-merged sorted event stream** because that representation
is cheaper to consume in C.  The dispatcher in
:mod:`neurobox.analysis.lfp.oscillations` therefore does different prep
work depending on which engine is loaded.
"""

from __future__ import annotations

import numpy as np


def within_ranges_matrix_engine_python(
    x:           np.ndarray,
    starts:      np.ndarray,
    stops:       np.ndarray,
    range_label: np.ndarray,
    n_labels:    int,
) -> np.ndarray:
    """Per-label searchsorted implementation.

    Parameters
    ----------
    x:
        1-D query points (any order).
    starts:
        ``(R,)`` range start times.
    stops:
        ``(R,)`` range stop times.
    range_label:
        ``(R,)`` int64 label for each range, in ``[0, n_labels)``
        (**0-indexed**, unlike the public 1-indexed API).
    n_labels:
        Number of distinct labels.

    Returns
    -------
    out : ndarray
        ``(len(x), n_labels)`` ``uint8`` array; ``1`` where the point
        lies in any range with that label, ``0`` otherwise.
    """
    out = np.zeros((x.size, n_labels), dtype=np.uint8)
    for k in range(n_labels):
        sel = range_label == k
        if not sel.any():
            continue
        s_sorted = np.sort(starts[sel])
        e_sorted = np.sort(stops[sel])
        inside = (np.searchsorted(s_sorted, x, side="right")
                  > np.searchsorted(e_sorted, x, side="left"))
        out[:, k] = inside.astype(np.uint8)
    return out
