"""
state_decoding.py
=================
Structured decoding of per-frame behaviour-state probabilities into
label sequences: transition-matrix fitting, Viterbi decoding, and
minimum-duration constraints.

Motivation
----------
The MATLAB pipeline turned classifier outputs into state periods by
per-frame ``argmax`` followed by ``ThreshCross``, which yields ragged
boundaries and spurious micro-bouts; ``optimize_stc_transition.m``
(222 LoC, stage 3 of ``label_behavior``) then existed largely to
repair them.  Decoding the label *sequence* jointly — maximising
emission scores **plus** log transition probabilities, under
per-state minimum-duration floors — addresses the cause instead of
patching the symptom, and works with **any** classifier backend
(including the MATLAB-parity ``patternnet``), since it only consumes
the ``(T, n_states)`` probability matrix.

Minimum durations are enforced exactly via the standard
expanded-state construction: a state with floor ``d`` becomes a chain
of ``d`` sub-states, where sub-states ``1..d-1`` transition only to
the next sub-state and only sub-state ``d`` may self-loop or leave.
Interior bouts therefore last at least ``d`` frames; bouts truncated
by the start or end of the recording are exempt (the recording may
begin or end mid-bout), which is why decoding may still emit a short
segment touching either edge.

Nothing here is a port — MATLAB had no equivalent.  It is the
principled replacement recommended by the ``label_behavior`` audit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from neurobox.dtype import NBStateCollection
from neurobox.analysis.decoding.state_matrix import stc2mat


__all__ = [
    "TransitionModel",
    "fit_transition_matrix",
    "labels_from_stc",
    "viterbi_decode",
    "decode_labels",
]


_LOG_EPS = -1e30      # effective -inf that survives addition


# ─────────────────────────────────────────────────────────────────────── #
# Label rasterisation                                                     #
# ─────────────────────────────────────────────────────────────────────── #

def labels_from_stc(
    stc:        NBStateCollection,
    states:     Sequence[str],
    n_samples:  int,
    samplerate: float,
) -> np.ndarray:
    """Rasterise *stc* into per-frame integer labels.

    Returns ``(n_samples,)`` int32 with values in ``[0, n_states)`` or
    ``-1`` where no listed state is active.  Where states overlap
    (possible in hand labels), the **earlier** entry of *states* wins,
    matching the column-priority behaviour of MATLAB's
    ``[~, ix] = max(stc2mat(...), [], 2)``.
    """
    smat, _ = stc2mat(stc, n_samples, samplerate, states=list(states))
    active = smat > 0
    labels = np.full(n_samples, -1, dtype=np.int32)
    # first active column wins → iterate in reverse so earlier overwrites
    for i in range(len(states) - 1, -1, -1):
        labels[active[:, i]] = i
    return labels


# ─────────────────────────────────────────────────────────────────────── #
# Transition model                                                        #
# ─────────────────────────────────────────────────────────────────────── #

@dataclass(frozen=True)
class TransitionModel:
    """Log transition matrix + per-state minimum durations.

    Attributes
    ----------
    log_transitions:
        ``(S, S)``; entry ``[i, j]`` is ``log P(state_j at t+1 |
        state_i at t)``.  Rows sum to one in probability space.
    min_durations:
        ``(S,)`` integer frame floors (``1`` = unconstrained).
    states:
        Label order defining the axes.
    samplerate:
        Hz the durations are expressed at, recorded for provenance.
    """
    log_transitions: np.ndarray
    min_durations:   np.ndarray
    states:          tuple[str, ...]
    samplerate:      float

    def __post_init__(self):
        S = len(self.states)
        lt = np.asarray(self.log_transitions, dtype=np.float64)
        md = np.asarray(self.min_durations, dtype=np.int64)
        if lt.shape != (S, S):
            raise ValueError(
                f"log_transitions must be ({S}, {S}), got {lt.shape}")
        if md.shape != (S,):
            raise ValueError(
                f"min_durations must be ({S},), got {md.shape}")
        if (md < 1).any():
            raise ValueError("min_durations must be >= 1 frame")
        object.__setattr__(self, "log_transitions", lt)
        object.__setattr__(self, "min_durations", md)


def fit_transition_matrix(
    label_sequences: Sequence[np.ndarray],
    n_states:        int,
    *,
    states:          Sequence[str] | None = None,
    samplerate:      float = 1.0,
    min_duration_s:  float | Sequence[float] = 0.0,
    pseudocount:     float = 1.0,
) -> TransitionModel:
    """Fit a :class:`TransitionModel` from hand-labelled sequences.

    Parameters
    ----------
    label_sequences:
        One int array per session, values in ``[0, n_states)`` or
        ``-1`` for unlabelled frames.  Transitions into, out of, or
        across ``-1`` frames are not counted.
    n_states:
        Number of states (columns of the classifier output).
    states:
        Optional labels for provenance; default ``("s0", "s1", ...)``.
    samplerate:
        Hz of the label sequences — needed to convert
        *min_duration_s* to frames.
    min_duration_s:
        Scalar or per-state minimum bout duration in **seconds**.
        ``0`` disables the floor for that state.
    pseudocount:
        Laplace smoothing added to every transition count, so states
        absent from the training labels still have finite log
        probabilities.

    Returns
    -------
    TransitionModel
    """
    if states is None:
        states = tuple(f"s{i}" for i in range(n_states))
    if len(states) != n_states:
        raise ValueError(
            f"states has {len(states)} entries but n_states={n_states}")

    counts = np.full((n_states, n_states), float(pseudocount))
    for seq in label_sequences:
        seq = np.asarray(seq)
        a, b = seq[:-1], seq[1:]
        ok = (a >= 0) & (b >= 0)
        np.add.at(counts, (a[ok], b[ok]), 1.0)

    row_sums = counts.sum(axis=1, keepdims=True)
    probs = np.divide(counts, row_sums,
                      out=np.full_like(counts, 1.0 / n_states),
                      where=row_sums > 0)
    # Rows with zero observations (possible at pseudocount=0) fall back
    # to uniform rather than NaN — note np.maximum would NOT clean a
    # NaN up, since maximum propagates NaN.
    with np.errstate(divide="ignore"):
        log_t = np.log(probs)
    log_t = np.maximum(log_t, _LOG_EPS)

    md_s = np.broadcast_to(
        np.asarray(min_duration_s, dtype=np.float64), (n_states,))
    min_durations = np.maximum(
        np.round(md_s * samplerate).astype(np.int64), 1)

    return TransitionModel(
        log_transitions = log_t,
        min_durations   = min_durations,
        states          = tuple(states),
        samplerate      = float(samplerate),
    )


# ─────────────────────────────────────────────────────────────────────── #
# Viterbi with minimum-duration floors                                    #
# ─────────────────────────────────────────────────────────────────────── #

def _expand_states(model: TransitionModel) -> tuple[np.ndarray, np.ndarray,
                                                    np.ndarray]:
    """Build the expanded-state transition matrix.

    A state with floor ``d`` becomes sub-states ``0..d-1``; only the
    final sub-state may self-loop (``log a_ss``) or leave
    (``log a_ss'`` into the *first* sub-state of ``s'``); sub-state
    ``k < d-1`` deterministically advances to ``k+1``.

    Returns
    -------
    (expanded_log_T, owner, entry):
        ``expanded_log_T`` — ``(E, E)`` log transitions;
        ``owner`` — ``(E,)`` original state index per sub-state;
        ``entry`` — ``(S,)`` index of each state's first sub-state.
    """
    d = model.min_durations
    S = len(model.states)
    E = int(d.sum())
    owner = np.empty(E, dtype=np.int64)
    entry = np.empty(S, dtype=np.int64)
    pos = 0
    for s in range(S):
        entry[s] = pos
        owner[pos:pos + d[s]] = s
        pos += d[s]

    expT = np.full((E, E), _LOG_EPS)
    for s in range(S):
        first, last = entry[s], entry[s] + d[s] - 1
        # Forced advances carry the self-transition cost log a_ss:
        # the duration floor changes the FEASIBLE SET, not the
        # objective.  With cost 0 here, constrained paths would score
        # differently from the same label sequence in the original
        # chain, and the decoder would no longer maximise the standard
        # HMM path score subject to the constraint.
        for k in range(first, last):
            expT[k, k + 1] = model.log_transitions[s, s]
        expT[last, last] = model.log_transitions[s, s]     # self-loop
        for s2 in range(S):
            if s2 != s:
                expT[last, entry[s2]] = model.log_transitions[s, s2]
    return expT, owner, entry


def viterbi_decode(
    log_probs: np.ndarray,
    model:     TransitionModel,
) -> np.ndarray:
    """Maximum-a-posteriori label path under *model*.

    Parameters
    ----------
    log_probs:
        ``(T, S)`` log emission scores — typically
        ``np.log(probs + eps)`` of a classifier's softmax output.
        Rows may contain ``-inf`` (impossible states); a row of all
        ``-inf`` is treated as uninformative.
    model:
        :class:`TransitionModel`.  Minimum durations are enforced for
        interior bouts; bouts clipped by the sequence start or end are
        exempt (see module docstring).

    Returns
    -------
    ``(T,)`` int32 state indices.
    """
    log_probs = np.asarray(log_probs, dtype=np.float64)
    T, S = log_probs.shape
    if S != len(model.states):
        raise ValueError(
            f"log_probs has {S} states but model has {len(model.states)}")
    if T == 0:
        return np.zeros(0, dtype=np.int32)

    emit = np.where(np.isfinite(log_probs), log_probs, _LOG_EPS)
    # An all -inf row carries no information — flatten it to zeros so it
    # neither traps nor forbids every path.
    dead = ~np.isfinite(log_probs).any(axis=1)
    emit[dead, :] = 0.0

    expT, owner, entry = _expand_states(model)
    E = expT.shape[0]

    # Start anywhere inside a chain (recording may begin mid-bout).
    dp = emit[0, owner].copy()
    back = np.zeros((T, E), dtype=np.int32)

    for t in range(1, T):
        # score[i, j] = dp[i] + expT[i, j]
        score = dp[:, None] + expT
        back[t] = np.argmax(score, axis=0)
        dp = score[back[t], np.arange(E)] + emit[t, owner]

    # End anywhere inside a chain (recording may end mid-bout).
    e = int(np.argmax(dp))
    path = np.empty(T, dtype=np.int32)
    for t in range(T - 1, -1, -1):
        path[t] = owner[e]
        e = back[t, e]
    return path


def decode_labels(
    probs:   np.ndarray,
    *,
    method:  str = "argmax",
    model:   TransitionModel | None = None,
    eps:     float = 1e-12,
) -> np.ndarray:
    """Uniform entry point over the decoding strategies.

    Parameters
    ----------
    probs:
        ``(T, S)`` probabilities (not logs).
    method:
        ``"argmax"`` — per-frame, the MATLAB behaviour;
        ``"viterbi"`` — structured decoding, requires *model*.
    model:
        :class:`TransitionModel`, mandatory for ``"viterbi"``.
    eps:
        Floor added before the log for Viterbi.
    """
    probs = np.asarray(probs, dtype=np.float64)
    if method == "argmax":
        return np.argmax(probs, axis=1).astype(np.int32)
    if method == "viterbi":
        if model is None:
            raise ValueError('method="viterbi" requires a TransitionModel')
        return viterbi_decode(np.log(probs + eps), model)
    raise ValueError(f"unknown decode method {method!r}")
