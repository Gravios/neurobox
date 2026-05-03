"""
neurobox.analysis.feature_selection.hmi
=========================================
Hierarchical mutual-information feature selection.

Port of :file:`MTA/analysis/old/select_features_hmi.m`.

Algorithm
---------
Given a state collection, a feature matrix, and a list of behavioural
states, this picks an *order* of states to discriminate and a *set
of feature columns* per state, by greedy MI gain:

1. Compute MI between each feature and the full state vector
   (row 0 of :func:`mutual_information_states_features`), and the
   MI with each state held out (rows 1..n).
2. For each state, compute Δ_MI(state, feature) =
   row_0[feature] − row_state[feature].  This is the MI lost when
   that state is excluded — high values = the feature is informative
   *about that state in particular*.
3. The state with the largest sum of positive Δ_MI is the best
   "binary state" to discriminate first.  Its feature set is all
   features with Δ_MI > 0.2 bits.
4. Remove that state and recurse on the remaining ones.
5. When only 2 states remain, switch to a d′ (Cohen's d) criterion
   for the final binary discriminator.

This produces a tree of binary classifiers, where each node uses a
small subset of features.  In the lab pipeline, neural networks are
trained at each level of this tree.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from neurobox.analysis.classifiers.stc_utils import (
    mutual_information_states_features,
)
from neurobox.dtype.stc import NBStateCollection


__all__ = [
    "HierarchicalMIResult",
    "select_features_hmi",
]


@dataclass
class HierarchicalMIResult:
    """Output of :func:`select_features_hmi`.

    Attributes
    ----------
    state_order : tuple[str, ...]
        States in the order they should be peeled off.  Length =
        number of input states.  The last 2 are paired together at
        the leaf.
    feature_indices : list[np.ndarray]
        ``feature_indices[k]`` = column indices selected for
        discriminating ``state_order[k]`` from all later states.
        Same length as *state_order*.
    mi_per_level : list[np.ndarray]
        ``mi_per_level[k]`` = the
        :func:`mutual_information_states_features` table at level *k*
        (before removing ``state_order[k]``).  Useful for
        diagnostic plots.
    """
    state_order:      tuple[str, ...]
    feature_indices:  list[np.ndarray]
    mi_per_level:     list[np.ndarray]


def select_features_hmi(
    stc:                NBStateCollection,
    feature_data:       np.ndarray,
    feature_samplerate: float,
    *,
    states:             Sequence[str] = (
        "rear", "walk", "turn", "pause", "groom", "sit",
    ),
    mi_threshold_bits:  float = 0.20,
    dprime_threshold:   float = 2.0,
    n_bins:             int = 128,
) -> HierarchicalMIResult:
    """Hierarchical MI-gain feature selection over states.

    Port of :file:`MTA/analysis/old/select_features_hmi.m`.

    Parameters
    ----------
    stc:
        State collection containing all of *states*.
    feature_data:
        ``(T, n_features)`` feature matrix.
    feature_samplerate:
        Hz of *feature_data*.  Must match *stc*'s sample rate after
        resampling internally.
    states:
        Ordered behavioural states to peel off.  Default matches
        MATLAB.
    mi_threshold_bits:
        Δ_MI threshold (bits) for selecting features at each level.
        Default 0.20 matches MATLAB.
    dprime_threshold:
        |d′| threshold for the leaf-level binary discriminator.
        Default 2.0 matches MATLAB.
    n_bins:
        Histogram bin count for MI estimation.  Default 128.

    Returns
    -------
    HierarchicalMIResult
    """
    feature_data = np.asarray(feature_data, dtype=np.float64)
    if feature_data.ndim != 2:
        raise ValueError(
            f"feature_data must be (T, n_features); got {feature_data.shape}"
        )

    remaining_states = list(states)
    state_order: list[str] = []
    feature_indices: list[np.ndarray] = []
    mi_per_level: list[np.ndarray] = []
    feature_ranges = None

    while len(remaining_states) > 2:
        # Compute MI matrix at this level
        mi_table, feature_ranges = mutual_information_states_features(
            stc, feature_data, feature_samplerate,
            states=remaining_states, n_bins=n_bins,
            feature_ranges=feature_ranges,
        )
        mi_per_level.append(mi_table)

        # Sum of positive Δ_MI per state
        sum_pos_dm = np.zeros(len(remaining_states))
        for s_idx in range(len(remaining_states)):
            dm = mi_table[0, :] - mi_table[s_idx + 1, :]
            sum_pos_dm[s_idx] = float(dm[dm > 0].sum())

        best_state_idx = int(np.argmax(sum_pos_dm))
        dm = mi_table[0, :] - mi_table[best_state_idx + 1, :]
        selected = np.flatnonzero(dm > mi_threshold_bits)

        state_order.append(remaining_states[best_state_idx])
        feature_indices.append(selected.astype(np.int64))
        del remaining_states[best_state_idx]

    # Leaf level: 2 states remaining → use d′
    s1, s2 = remaining_states
    from neurobox.analysis.classifiers.stc_utils import stc2mat
    T = feature_data.shape[0]
    smat, _ = stc2mat(stc, T, feature_samplerate, states=[s1, s2])
    mask1 = smat[:, 0] != 0
    mask2 = smat[:, 1] != 0
    if mask1.any() and mask2.any():
        with np.errstate(invalid="ignore", divide="ignore"):
            mu1 = np.nanmean(feature_data[mask1, :], axis=0)
            mu2 = np.nanmean(feature_data[mask2, :], axis=0)
            v1  = np.nanvar (feature_data[mask1, :], axis=0)
            v2  = np.nanvar (feature_data[mask2, :], axis=0)
            denom = 0.5 * (v1 + v2)
            dprime = (mu1 - mu2) / np.where(denom > 0, denom, 1.0)
        leaf_features = np.flatnonzero(np.abs(dprime) > dprime_threshold)
    else:
        leaf_features = np.array([], dtype=np.int64)
    feature_indices.append(leaf_features.astype(np.int64))
    state_order.extend(remaining_states)

    return HierarchicalMIResult(
        state_order      = tuple(state_order),
        feature_indices  = feature_indices,
        mi_per_level     = mi_per_level,
    )
