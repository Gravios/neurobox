"""
neurobox.analysis.classifiers.stc_utils
========================================
Editing and comparison utilities for behavioural-state collections.

Ports of:

* ``MTA/utilities/mat2stc.m``                                  → :func:`mat_to_stc`
* ``MTA/utilities/confusion_matrix.m``                         → :func:`confusion_matrix`
* ``MTA/utilities/cmp_stcs.m``                                 → :func:`compare_stcs`
* ``MTA/utilities/reassign_state_by_duration.m``               → :func:`reassign_state_by_duration`
* ``MTA/utilities/calculate_MI_states_vs_features.m``          → :func:`mutual_information_states_features`
* ``MTA/utilities/swap_state_vector_ids.m``                    → :func:`swap_state_vector_ids`
* ``MTA/classifiers/utilities/reduce_stc_to_loc.m``            → :func:`reduce_stc_to_loc`
* ``MTA/classifiers/utilities/reassign_low_duration_state_to_neighboring_states.m``
                                                               → :func:`reassign_short_periods`
* ``MTA/classifiers/utilities/reassign_period_to_neighboring_states.m``
                                                               → internal helper

These functions all operate on the ``(T, n_states)`` integer state-
membership matrix produced by
:func:`neurobox.analysis.decoding.stc2mat`, or on
:class:`NBStateCollection` directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from neurobox.analysis.decoding.state_matrix import stc2mat
from neurobox.dtype.epoch import NBEpoch
from neurobox.dtype.stc import NBStateCollection


# ─────────────────────────────────────────────────────────────────────── #
# stc2mat ↔ mat2stc                                                         #
# ─────────────────────────────────────────────────────────────────────── #

def mat_to_stc(
    state_matrix:   np.ndarray,
    state_names:    Sequence[str],
    samplerate:     float,
    state_keys:     Sequence[str] | None = None,
    stc:            NBStateCollection | None = None,
    overwrite:      bool = True,
) -> NBStateCollection:
    """Inverse of :func:`stc2mat` — build (or update) an :class:`NBStateCollection`.

    Port of :file:`MTA/utilities/mat2stc.m`.

    Parameters
    ----------
    state_matrix:
        ``(T, n_states)`` integer / boolean / float matrix.  Non-zero
        entries in column *i* mark sample *t* as belonging to state
        ``state_names[i]``.  This is the layout produced by
        :func:`stc2mat` or by argmax over a softmax output.
    state_names:
        Per-column state labels.
    samplerate:
        Sampling rate of *state_matrix* in Hz; the resulting epoch
        periods are returned in seconds.
    state_keys:
        Optional single-character keys for each state.  Default
        ``state_names[i][0]``.
    stc:
        Optional existing collection to extend.  ``None`` → make a
        new empty one.
    overwrite:
        If True (default) and a state with the same label exists, it
        is replaced; if False, raises.

    Returns
    -------
    NBStateCollection
    """
    state_matrix = np.asarray(state_matrix)
    if state_matrix.ndim != 2:
        raise ValueError(
            f"state_matrix must be (T, n_states); got {state_matrix.shape}"
        )
    T, n_states = state_matrix.shape
    if len(state_names) != n_states:
        raise ValueError(
            f"len(state_names)={len(state_names)} doesn't match column count "
            f"{n_states}"
        )
    if state_keys is None:
        state_keys = [n[0] for n in state_names]
    if len(state_keys) != n_states:
        raise ValueError(
            f"len(state_keys)={len(state_keys)} doesn't match column count "
            f"{n_states}"
        )

    if stc is None:
        stc = NBStateCollection()

    for i, (label, key) in enumerate(zip(state_names, state_keys)):
        col = state_matrix[:, i] != 0
        # Find runs of True samples
        edges  = np.diff(np.concatenate([[False], col, [False]]).astype(np.int8))
        starts = np.where(edges == +1)[0]
        stops  = np.where(edges == -1)[0]                 # exclusive end
        if starts.size == 0:
            periods = np.zeros((0, 2), dtype=np.float64)
        else:
            # Exclusive-end convention to match NBEpoch.to_mask /
            # to_periods: a run [start_idx, stop_idx) → seconds
            # (start_idx/sr, stop_idx/sr).
            periods = np.column_stack([
                starts.astype(np.float64) / samplerate,
                stops.astype(np.float64) / samplerate,
            ])

        ep = NBEpoch(
            data       = periods,
            samplerate = 1.0,
            label      = label,
            key        = key,
        )
        if stc.has_state(label):
            if not overwrite:
                raise ValueError(
                    f"State {label!r} already in stc; pass overwrite=True"
                )
            del stc._states[label]
            for k_, lbl_ in list(stc._keys.items()):
                if lbl_ == label:
                    del stc._keys[k_]
        stc.add_state(ep, label=label, key=key)
    return stc


# ─────────────────────────────────────────────────────────────────────── #
# Confusion matrix + label statistics                                       #
# ─────────────────────────────────────────────────────────────────────── #

@dataclass
class LabelComparisonStats:
    """Output of :func:`compare_stcs`.

    Attributes
    ----------
    confusion_matrix:
        ``(n_states, n_states)`` matrix.  ``confusion_matrix[i, j]``
        is the count (or duration in seconds, see *as_seconds*) of
        samples labelled state *i* in *stc1* and state *j* in *stc2*.
    precision:
        ``(n_states,)`` per-state precision.  In this convention
        ``precision[i] = TP[i] / row_sum[i]`` — what fraction of
        samples *stc1* called state *i* were also called *i* by *stc2*.
    sensitivity:
        ``(n_states,)`` per-state sensitivity (recall).
        ``sensitivity[i] = TP[i] / col_sum[i]``.
    accuracy:
        Scalar overall accuracy = trace / total.
    state_names:
        Labels in the order of the matrix indices.
    """
    confusion_matrix: np.ndarray
    precision:        np.ndarray
    sensitivity:      np.ndarray
    accuracy:         float
    state_names:      list[str]


def confusion_matrix(
    stc1:        NBStateCollection,
    stc2:        NBStateCollection,
    n_samples:   int,
    samplerate:  float,
    states:      Sequence[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Confusion matrix between two state collections.

    Port of :file:`MTA/utilities/confusion_matrix.m`.

    Parameters
    ----------
    stc1, stc2:
        Two state collections covering the same time window.
    n_samples:
        Output length used by both :func:`stc2mat` calls.
    samplerate:
        Hz of the resulting per-sample integer matrices.
    states:
        Optional state labels to compare.  ``None`` → intersection of
        both collections' labels (matches MATLAB).

    Returns
    -------
    confmat : np.ndarray, shape ``(n_states, n_states)``
        ``confmat[i, j]`` = count of samples labelled state *i* in
        *stc1* and state *j* in *stc2*.
    labels : list of str
    """
    if states is None:
        s1 = set(stc1.list_states())
        s2 = set(stc2.list_states())
        states = sorted(s1 & s2)
        if not states:
            raise ValueError(
                "confusion_matrix: stc1 and stc2 share no state labels."
            )
    states = list(states)

    smat1, names1 = stc2mat(stc1, n_samples, samplerate, states=states)
    smat2, names2 = stc2mat(stc2, n_samples, samplerate, states=states)
    if names1 != names2:
        raise RuntimeError(
            f"State name ordering mismatch: {names1} vs {names2}"
        )

    n_states = len(states)
    confmat = np.zeros((n_states, n_states), dtype=np.int64)

    # Direct pairwise overlap count.  Unlike MATLAB's port which
    # collapsed each stc to a single integer code per sample (and
    # therefore silently broke when states overlapped at a sample),
    # this counts ``smat1[:, i] != 0 AND smat2[:, j] != 0`` directly,
    # so overlapping/multi-label states are handled correctly.
    masks1 = (smat1 != 0)
    masks2 = (smat2 != 0)
    for i in range(n_states):
        for j in range(n_states):
            confmat[i, j] = int(np.sum(masks1[:, i] & masks2[:, j]))
    return confmat, list(states)


def compare_stcs(
    stc1:        NBStateCollection,
    stc2:        NBStateCollection,
    n_samples:   int,
    samplerate:  float,
    states:      Sequence[str] | None = None,
    as_seconds:  bool = True,
) -> LabelComparisonStats:
    """Confusion matrix + precision/sensitivity/accuracy between two stcs.

    Port of :file:`MTA/utilities/cmp_stcs.m`.

    Parameters
    ----------
    stc1, stc2:
        State collections to compare.
    n_samples:
        Length of the per-sample matrices used internally.
    samplerate:
        Hz of those matrices.
    states:
        Optional explicit state list; ``None`` → intersection.
    as_seconds:
        If True (default), the confusion-matrix entries are in
        seconds (counts ÷ samplerate).  If False, raw counts.

    Returns
    -------
    LabelComparisonStats
    """
    cm, labels = confusion_matrix(stc1, stc2, n_samples, samplerate, states)
    cm_f = cm.astype(np.float64)
    if as_seconds:
        cm_f = cm_f / samplerate

    diag = np.diag(cm).astype(np.float64)
    row_sum = cm.sum(axis=1).astype(np.float64)
    col_sum = cm.sum(axis=0).astype(np.float64)
    total   = float(cm.sum())

    with np.errstate(divide="ignore", invalid="ignore"):
        precision   = np.where(row_sum > 0, diag / row_sum, np.nan)
        sensitivity = np.where(col_sum > 0, diag / col_sum, np.nan)
    accuracy = (diag.sum() / total) if total > 0 else float("nan")

    return LabelComparisonStats(
        confusion_matrix = cm_f,
        precision        = precision,
        sensitivity      = sensitivity,
        accuracy         = accuracy,
        state_names      = list(labels),
    )


# ─────────────────────────────────────────────────────────────────────── #
# State-vector editing                                                       #
# ─────────────────────────────────────────────────────────────────────── #

def swap_state_vector_ids(
    state_vector: np.ndarray,
    ind1:         int,
    ind2:         int,
) -> np.ndarray:
    """Swap two integer state codes in a per-sample state-code vector.

    Port of :file:`MTA/utilities/swap_state_vector_ids.m`.

    Operates **in place** on a copy of the input (not the caller's
    array); returns the modified copy for convenience.
    """
    out = np.asarray(state_vector).copy()
    a = out == ind1
    b = out == ind2
    out[a] = ind2
    out[b] = ind1
    return out


def reassign_short_periods(
    state_matrix:        np.ndarray,
    target_state_col:    int,
    duration_samples:    int,
) -> np.ndarray:
    """Reassign short periods of a target state to its neighbours.

    Combined port of:

    * :file:`MTA/classifiers/utilities/reassign_low_duration_state_to_neighboring_states.m`
    * :file:`MTA/classifiers/utilities/reassign_period_to_neighboring_states.m`

    For each contiguous run of *target_state_col* shorter than
    *duration_samples*, split the run at its midpoint and copy the
    state codes from the immediately preceding sample to the first
    half and the following sample to the second half.

    Parameters
    ----------
    state_matrix:
        ``(T, n_states)`` boolean / integer matrix produced by
        :func:`stc2mat`.  The matrix is modified out-of-place.
    target_state_col:
        Column index of the state whose short periods are to be
        reassigned.
    duration_samples:
        Periods strictly shorter than this are reassigned.

    Returns
    -------
    np.ndarray
        Modified ``(T, n_states)`` matrix.
    """
    sm = np.asarray(state_matrix).copy()
    if sm.ndim != 2:
        raise ValueError(
            f"state_matrix must be (T, n_states); got {sm.shape}"
        )
    col = sm[:, target_state_col] != 0
    edges  = np.diff(np.concatenate([[False], col, [False]]).astype(np.int8))
    starts = np.where(edges == +1)[0]
    stops  = np.where(edges == -1)[0] - 1     # inclusive end
    for s, e in zip(starts, stops):
        dur = e - s + 1
        if dur >= duration_samples:
            continue
        if s - 1 < 0 or e + 1 >= sm.shape[0]:
            continue                          # no neighbour available
        start_vec = sm[s - 1, :].copy()
        stop_vec  = sm[e + 1, :].copy()
        midpoint  = (s + e) // 2
        sm[s:midpoint + 1, :]  = start_vec[None, :]
        sm[midpoint + 1:e + 1, :] = stop_vec[None, :]
    return sm


def reassign_state_by_duration(
    stc:                 NBStateCollection,
    target_label:        str,
    duration_threshold_s: float,
    default_state:       str | None = None,
    rule:                str = "shorter_than",
) -> NBStateCollection:
    """Reassign periods of *target_label* whose duration matches *rule*.

    Port of :file:`MTA/utilities/reassign_state_by_duration.m`.

    The MATLAB original took a logical function (``@lt`` / ``@gt``)
    parameter; this port exposes that as the *rule* string.

    Parameters
    ----------
    stc:
        State collection to edit.  **Not** modified — the function
        returns a new copy.
    target_label:
        Label of the state whose periods will be examined.
    duration_threshold_s:
        Threshold in seconds.
    default_state:
        Optional fallback state label.  If given, every period
        matching *rule* is relabelled as this state.  If ``None``,
        the period is left in place but the original is removed
        (matching MATLAB behaviour where short periods of "rare"
        states get pruned).
    rule:
        ``'shorter_than'`` (default — keep periods *not* shorter
        than threshold), ``'longer_than'``, or ``'equal'``.

    Returns
    -------
    NBStateCollection
        New collection with the reassigned periods applied.
    """
    if rule not in ("shorter_than", "longer_than", "equal"):
        raise ValueError(
            f"rule must be 'shorter_than'/'longer_than'/'equal'; got {rule!r}"
        )
    import copy as _copy
    new_stc = _copy.deepcopy(stc)
    if not new_stc.has_state(target_label):
        return new_stc

    target = new_stc.get_state(target_label)
    periods = np.asarray(target.data, dtype=np.float64)
    durations = periods[:, 1] - periods[:, 0]

    if rule == "shorter_than":
        mask = durations < duration_threshold_s
    elif rule == "longer_than":
        mask = durations > duration_threshold_s
    else:
        mask = np.isclose(durations, duration_threshold_s)

    matched = periods[mask]
    kept    = periods[~mask]

    # Replace target with kept-only periods
    target.data = kept

    # Optionally, push matched periods into the default state
    if default_state is not None and matched.size > 0:
        if not new_stc.has_state(default_state):
            new_ep = NBEpoch(
                data=matched, samplerate=1.0,
                label=default_state, key=default_state[0],
            )
            new_stc.add_state(new_ep, label=default_state,
                              key=default_state[0])
        else:
            ds = new_stc.get_state(default_state)
            ds.data = np.vstack([ds.data, matched])

    return new_stc


def reduce_stc_to_loc(
    stc:        NBStateCollection,
    walk_label: str = "walk",
    turn_label: str = "turn",
    new_label:  str = "loc",
    new_key:    str = "x",
) -> NBStateCollection:
    """Replace 'walk' and 'turn' states with a unified 'loc' state.

    Port of :file:`MTA/classifiers/utilities/reduce_stc_to_loc.m`.

    Returns a new collection with *walk* and *turn* dropped and a
    new state *new_label* added containing their union.
    """
    import copy as _copy
    new_stc = _copy.deepcopy(stc)
    if not (new_stc.has_state(walk_label) and new_stc.has_state(turn_label)):
        return new_stc

    walk_ep = new_stc.get_state(walk_label)
    turn_ep = new_stc.get_state(turn_label)
    union = walk_ep | turn_ep

    # Remove the old "loc" state if it exists, then drop walk + turn
    if new_stc.has_state(new_label):
        del new_stc._states[new_label]
        for k_, lbl_ in list(new_stc._keys.items()):
            if lbl_ == new_label:
                del new_stc._keys[k_]

    union_ep = NBEpoch(
        data=union.data, samplerate=union.samplerate,
        label=new_label, key=new_key,
    )
    new_stc.add_state(union_ep, label=new_label, key=new_key)
    return new_stc


# ─────────────────────────────────────────────────────────────────────── #
# Mutual information                                                        #
# ─────────────────────────────────────────────────────────────────────── #

def mutual_information_states_features(
    stc:                NBStateCollection,
    feature_data:       np.ndarray,
    feature_samplerate: float,
    states:             Sequence[str],
    n_bins:             int = 128,
    feature_ranges:     Sequence[tuple[float, float]] | None = None,
) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """Mutual information between state labels and each feature column.

    Port of :file:`MTA/utilities/calculate_MI_states_vs_features.m`.

    Returns a ``(n_holdouts + 1, n_features)`` MI table.  Row 0 is
    MI computed using all *states*; row *k+1* is MI computed with
    state *k* held out.  The hold-out rows let you see which states
    each feature is most informative about.

    Parameters
    ----------
    stc:
        State collection.
    feature_data:
        ``(T, n_features)`` feature matrix.
    feature_samplerate:
        Hz of *feature_data*.
    states:
        Ordered state labels.
    n_bins:
        Histogram bins per feature.  Default 128.
    feature_ranges:
        Optional per-feature ``(low, high)`` ranges (in feature
        units).  Default None → use the 1st/99th percentiles of
        valid samples (matches MATLAB).

    Returns
    -------
    mi : np.ndarray, shape ``(len(states) + 1, n_features)``
    feature_ranges : list of (low, high) tuples
    """
    feature_data = np.asarray(feature_data, dtype=np.float64)
    if feature_data.ndim != 2:
        raise ValueError(
            f"feature_data must be (T, n_features); got {feature_data.shape}"
        )
    T, n_feat = feature_data.shape

    smat, _ = stc2mat(stc, T, feature_samplerate, states=list(states))
    # Per-sample integer state codes (1..n_states, 0 = unlabelled)
    code = np.zeros(T, dtype=np.int32)
    for i in range(len(states)):
        code[smat[:, i] != 0] = i + 1

    n_states = len(states)
    out = np.zeros((n_states + 1, n_feat), dtype=np.float64)

    if feature_ranges is None:
        # Use 1st/99th percentile over rows where any state is active
        valid = (code > 0) & np.isfinite(feature_data).all(axis=1)
        feature_ranges = []
        for f in range(n_feat):
            lo, hi = np.percentile(feature_data[valid, f], [1, 99])
            feature_ranges.append((float(lo), float(hi)))
    feature_ranges = list(feature_ranges)

    for s in range(n_states + 1):
        if s == 0:
            keep_states = list(range(1, n_states + 1))
        else:
            keep_states = [k for k in range(1, n_states + 1) if k != s]
        valid = np.isin(code, keep_states) & np.isfinite(feature_data).all(axis=1)
        n_valid = int(valid.sum())
        if n_valid < 2:
            out[s, :] = 0.0
            continue
        for f in range(n_feat):
            edx = np.linspace(feature_ranges[f][0], feature_ranges[f][1],
                              n_bins + 1)
            edy = np.arange(0.5, len(keep_states) + 1.5)
            fx = feature_data[valid, f]
            fy = code[valid]
            # Map fy from "raw label" to compact 1..len(keep_states)
            mapping = {lab: i + 1 for i, lab in enumerate(keep_states)}
            fy_mapped = np.fromiter(
                (mapping[v] for v in fy), dtype=np.int32, count=fy.size,
            )

            counts, _, _ = np.histogram2d(fx, fy_mapped, bins=[edx, edy])
            pxy = counts / n_valid
            px  = pxy.sum(axis=1, keepdims=True)
            py  = pxy.sum(axis=0, keepdims=True)
            with np.errstate(divide="ignore", invalid="ignore"):
                logterm = np.log2(np.where(pxy > 0, pxy / (px * py), 1.0))
            out[s, f] = float(np.nansum(pxy * logterm))

    return out, feature_ranges
