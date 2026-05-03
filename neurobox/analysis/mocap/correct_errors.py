"""
neurobox.analysis.mocap.correct_errors
=======================================
Re-assign rigid-body markers to fix swap errors detected by
:func:`find_error_periods`.

Port of :file:`MTA/utilities/mocap/CorrectPointErrors.m`.

The MATLAB original used a custom 5-D tensor reshape that had a known
bug (see the `%% TODO - Prevent trajectory assignment to multiple
markers` comment in the source).  This port re-derives the intent —
"for each frame in an error period, pick the permutation of source
markers that minimises the inter-frame trajectory cost" — and solves
it cleanly with the Hungarian algorithm
(:func:`scipy.optimize.linear_sum_assignment`).

Algorithm
---------
For each error period ``[s, e]``:

1. Take the immediately-preceding "good" frame at index ``s−1`` as
   the anchor.
2. For each frame ``t = s, s+1, …, e``:

   a. Build a cost matrix ``C[i, j] = ||x_t[j] − x_{t−1}[i]||²``,
      where ``x_t[j]`` is the position of source marker ``j`` at
      frame ``t`` and ``x_{t−1}[i]`` is the position of *destination*
      marker ``i`` at the previous frame.

   b. Solve the assignment problem: find the permutation ``π`` that
      minimises ``Σᵢ C[i, π(i)]``.  This gives, for each destination
      marker, which source marker should fill its slot.

   c. Apply the permutation to ``x_t``.

3. The corrected segment replaces the original error period in the
   xyz array.

Because the Hungarian algorithm enforces a one-to-one assignment,
no two destination markers can receive the same source marker —
fixing the MATLAB original's known bug.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from neurobox.dtype.xyz import NBDxyz

from .error_periods import find_error_periods


def correct_point_errors(
    xyz:           NBDxyz,
    markers:       Sequence[str] = (
        "head_back", "head_left", "head_front", "head_right",
    ),
    error_periods: np.ndarray | None = None,
    threshold_z:   float = 1.0,
) -> NBDxyz:
    """Re-assign rigid-body markers within detected error periods.

    Port of :file:`MTA/utilities/mocap/CorrectPointErrors.m`.

    Parameters
    ----------
    xyz:
        Source position data.  Must contain all *markers*.
    markers:
        Rigid-body marker set.  Default 4-head-marker set.
    error_periods:
        Optional ``(N, 2)`` integer-sample-index error periods.
        ``None`` (default) auto-detects via
        :func:`find_error_periods`.
    threshold_z:
        Only used when *error_periods* is ``None``.  See
        :func:`find_error_periods`.

    Returns
    -------
    NBDxyz
        New :class:`NBDxyz` with the rigid-body markers re-permuted
        within each error period to minimise inter-frame jumps.  The
        original is unmodified.
    """
    if xyz._data is None:
        raise RuntimeError("xyz data is not loaded.")
    for m in markers:
        if m not in xyz.model.markers:
            raise ValueError(
                f"correct_point_errors: xyz has no {m!r} marker."
            )

    if error_periods is None:
        error_periods, _, _ = find_error_periods(
            xyz, markers=markers, threshold_z=threshold_z,
        )
    error_periods = np.asarray(error_periods, dtype=np.int64)
    if error_periods.size == 0:
        import copy as _copy
        return _copy.copy(xyz)

    data = xyz._data.copy()
    T = data.shape[0]
    rb_idx = np.array([xyz.model.index(m) for m in markers], dtype=np.int64)

    for s, e in error_periods:
        s = int(max(0, s))
        e = int(min(T - 1, e))
        if e <= s:
            continue
        if s == 0:
            # No anchor frame available — skip
            continue

        # Anchor: previous frame
        prev = data[s - 1, rb_idx, :]                  # (n_rb, 3)

        for t in range(s, e + 1):
            curr = data[t, rb_idx, :]                  # (n_rb, 3) candidates
            # Cost: dest i picks source j → ||curr[j] - prev[i]||²
            diff = curr[None, :, :] - prev[:, None, :] # (n_rb, n_rb, 3)
            cost = np.einsum("ijk,ijk->ij", diff, diff)
            # Hungarian assignment: row i → col col_ind[i]
            row_ind, col_ind = linear_sum_assignment(cost)
            # row_ind is just 0..n_rb-1 in order; col_ind tells us which
            # source marker should fill each destination slot.
            data[t, rb_idx, :] = curr[col_ind, :]
            prev = data[t, rb_idx, :]

    import copy as _copy
    out = _copy.copy(xyz)
    out._data = data
    return out
