"""
neurobox.analysis.decoding.state_matrix
========================================
Convert a state collection to a per-sample one-hot/labelled matrix.

Port of :file:`MTA/utilities/stc/stc2mat.m`.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from neurobox.dtype.stc import NBStateCollection


def stc2mat(
    stc:        NBStateCollection,
    n_samples:  int,
    samplerate: float,
    states:     Sequence[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Build a state-membership matrix from an :class:`NBStateCollection`.

    Port of :file:`MTA/utilities/stc/stc2mat.m`.  The MATLAB original
    encoded each state with its column index ``g`` (so column ``g``
    contained zeros and the integer ``g`` where state ``g`` was
    active); the same convention is preserved here.  This makes
    multi-state queries via ``smat.any(axis=1)`` or
    ``smat[:, [i, j]].any(axis=1)`` natural.

    Parameters
    ----------
    stc:
        State collection.
    n_samples:
        Length of the desired output (typically the matching
        :class:`NBDxyz` data length).
    samplerate:
        Sample rate of the output, in Hz.
    states:
        Optional list of state labels (or single-character keys) to
        encode.  ``None`` → use ``stc.list_states()``.

    Returns
    -------
    smat : np.ndarray, shape ``(n_samples, n_states)``
        Integer matrix.  ``smat[t, i] = i+1`` if state ``i`` is
        active at sample ``t``, else 0.
    state_names : list[str]
        Resolved state labels in column order.
    """
    if states is None:
        states = stc.list_states()
    state_names = list(states)

    smat = np.zeros((n_samples, len(state_names)), dtype=np.int32)
    for i, name in enumerate(state_names):
        try:
            ep = stc[name] if hasattr(stc, "__getitem__") else stc.get_state(name)
        except (KeyError, ValueError):
            # Missing state — column stays zeros, matches MATLAB's
            # silent-skip behaviour on unknown keys.
            continue
        # Convert the epoch to a per-sample boolean mask
        ep_resampled = ep.resample(samplerate)
        mask = ep_resampled.to_mask(n_samples)
        smat[mask, i] = i + 1

    return smat, state_names
