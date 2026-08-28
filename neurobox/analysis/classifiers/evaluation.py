"""
evaluation.py
=============
Evaluation harness for behaviour labelling: frame metrics, segmental
metrics, boundary error, bout statistics, and leave-one-session-out
splitting.

Why these metrics
-----------------
The MATLAB pipeline reported a per-frame confusion matrix, which is
nearly blind to the failure mode that motivated
``optimize_stc_transition.m``: ragged boundaries and spurious
micro-bouts.  Two labelings with identical frame accuracy can differ
wildly in bout structure.  This module therefore adds the segmental
measures standard in the action-segmentation literature:

* **Segmental F1 @ IoU τ** (Lea et al., 2017): predicted segments are
  matched greedily to same-class ground-truth segments requiring
  intersection-over-union ≥ τ, each truth segment matched at most
  once; F1 over the resulting TP/FP/FN.  Reported at several τ.
* **Boundary MAE**: mean absolute distance (seconds) from each true
  transition to the nearest predicted transition — the quantity
  stage 3 tried to optimise by hand.
* **Bout statistics**: per-state bout counts and duration
  distributions, since these are what downstream science consumes.

Frames labelled ``-1`` (unlabelled) in the ground truth are excluded
from frame metrics and break segments in the segmental ones.

Leave-one-session-out is the only honest split at this data scale:
random frame splits leak temporal autocorrelation and inflate every
number.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Sequence

import numpy as np


__all__ = [
    "segments_from_labels",
    "frame_scores",
    "segmental_f1",
    "boundary_mae",
    "bout_statistics",
    "loso_splits",
    "LabelingScore",
    "evaluate_labeling",
]


# ─────────────────────────────────────────────────────────────────────── #
# Segments                                                                #
# ─────────────────────────────────────────────────────────────────────── #

def segments_from_labels(labels: np.ndarray) -> list[tuple[int, int, int]]:
    """Run-length encode integer labels.

    Returns ``[(state, start, stop), ...]`` with half-open frame
    intervals ``[start, stop)``, in temporal order.  Runs of ``-1``
    (unlabelled) are omitted.
    """
    labels = np.asarray(labels)
    if labels.size == 0:
        return []
    change = np.flatnonzero(np.diff(labels)) + 1
    starts = np.concatenate([[0], change])
    stops = np.concatenate([change, [labels.size]])
    return [(int(labels[a]), int(a), int(b))
            for a, b in zip(starts, stops) if labels[a] >= 0]


# ─────────────────────────────────────────────────────────────────────── #
# Frame metrics                                                           #
# ─────────────────────────────────────────────────────────────────────── #

def frame_scores(
    y_true:   np.ndarray,
    y_pred:   np.ndarray,
    n_states: int,
) -> tuple[float, float, np.ndarray]:
    """Frame accuracy, macro-F1 and per-state F1.

    Frames where ``y_true == -1`` are excluded.  A state absent from
    both truth and prediction contributes F1 = NaN and is skipped by
    the macro average, so unused states don't dilute the score.

    Returns
    -------
    (accuracy, macro_f1, per_state_f1):
        ``per_state_f1`` has shape ``(n_states,)`` with NaN for
        states absent from both.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"shape mismatch: {y_true.shape} vs {y_pred.shape}")
    m = y_true >= 0
    yt, yp = y_true[m], y_pred[m]
    if yt.size == 0:
        return float("nan"), float("nan"), np.full(n_states, np.nan)

    acc = float((yt == yp).mean())
    f1 = np.full(n_states, np.nan)
    for s in range(n_states):
        tp = int(((yt == s) & (yp == s)).sum())
        fp = int(((yt != s) & (yp == s)).sum())
        fn = int(((yt == s) & (yp != s)).sum())
        if tp + fp + fn == 0:
            continue
        f1[s] = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    macro = float(np.nanmean(f1)) if np.isfinite(f1).any() else float("nan")
    return acc, macro, f1


# ─────────────────────────────────────────────────────────────────────── #
# Segmental F1                                                            #
# ─────────────────────────────────────────────────────────────────────── #

def segmental_f1(
    y_true:   np.ndarray,
    y_pred:   np.ndarray,
    *,
    iou:      float = 0.5,
) -> float:
    """Segmental F1 at overlap threshold *iou* (Lea et al., 2017).

    Each predicted segment is matched (greedily, in temporal order) to
    an unmatched ground-truth segment of the same class with
    intersection-over-union ≥ *iou*; matched pairs are TP, unmatched
    predictions FP, unmatched truths FN.

    Returns NaN when both labelings contain no segments.
    """
    seg_t = segments_from_labels(y_true)
    seg_p = segments_from_labels(y_pred)
    if not seg_t and not seg_p:
        return float("nan")
    if not seg_t or not seg_p:
        return 0.0

    used = np.zeros(len(seg_t), dtype=bool)
    tp = 0
    for cls_p, a_p, b_p in seg_p:
        best_j, best_iou = -1, 0.0
        for j, (cls_t, a_t, b_t) in enumerate(seg_t):
            if used[j] or cls_t != cls_p:
                continue
            inter = max(0, min(b_p, b_t) - max(a_p, a_t))
            union = (b_p - a_p) + (b_t - a_t) - inter
            r = inter / union if union else 0.0
            if r > best_iou:
                best_iou, best_j = r, j
        if best_j >= 0 and best_iou >= iou:
            used[best_j] = True
            tp += 1
    fp = len(seg_p) - tp
    fn = len(seg_t) - tp
    denom = 2 * tp + fp + fn
    return 2 * tp / denom if denom else float("nan")


# ─────────────────────────────────────────────────────────────────────── #
# Boundary error                                                          #
# ─────────────────────────────────────────────────────────────────────── #

def boundary_mae(
    y_true:     np.ndarray,
    y_pred:     np.ndarray,
    samplerate: float,
) -> float:
    """Mean |Δt| (seconds) from each true boundary to its nearest
    predicted boundary.

    A boundary is any frame where the label changes (transitions into
    or out of ``-1`` included, since a bout edge against unlabelled
    data is still an edge).  Returns NaN if the truth has no
    boundaries; ``inf`` if the truth has boundaries but the prediction
    has none.
    """
    bt = np.flatnonzero(np.diff(np.asarray(y_true))) + 1
    bp = np.flatnonzero(np.diff(np.asarray(y_pred))) + 1
    if bt.size == 0:
        return float("nan")
    if bp.size == 0:
        return float("inf")
    d = np.abs(bt[:, None] - bp[None, :]).min(axis=1)
    return float(d.mean() / samplerate)


# ─────────────────────────────────────────────────────────────────────── #
# Bout statistics                                                         #
# ─────────────────────────────────────────────────────────────────────── #

def bout_statistics(
    labels:     np.ndarray,
    n_states:   int,
    samplerate: float,
) -> dict[int, dict[str, float]]:
    """Per-state bout count and duration summary.

    Returns ``{state: {"n_bouts", "total_s", "mean_s", "median_s"}}``
    for every state in ``range(n_states)`` (zeros/NaN when absent).
    """
    segs = segments_from_labels(labels)
    out: dict[int, dict[str, float]] = {}
    for s in range(n_states):
        durs = np.array([(b - a) for c, a, b in segs if c == s], float)
        durs /= samplerate
        out[s] = {
            "n_bouts":  float(durs.size),
            "total_s":  float(durs.sum()),
            "mean_s":   float(durs.mean()) if durs.size else float("nan"),
            "median_s": float(np.median(durs)) if durs.size else float("nan"),
        }
    return out


# ─────────────────────────────────────────────────────────────────────── #
# LOSO                                                                    #
# ─────────────────────────────────────────────────────────────────────── #

def loso_splits(n_sessions: int) -> Iterator[tuple[list[int], int]]:
    """Leave-one-session-out index splits.

    Yields ``(train_indices, test_index)`` once per session.
    """
    if n_sessions < 2:
        raise ValueError(
            f"leave-one-session-out needs >= 2 sessions, got {n_sessions}")
    for k in range(n_sessions):
        yield [i for i in range(n_sessions) if i != k], k


# ─────────────────────────────────────────────────────────────────────── #
# One-call bundle                                                         #
# ─────────────────────────────────────────────────────────────────────── #

@dataclass
class LabelingScore:
    """All metrics for one (truth, prediction) pair."""
    accuracy:      float
    macro_f1:      float
    per_state_f1:  np.ndarray
    segmental:     dict[float, float]          # iou → F1
    boundary_mae_s: float
    n_bouts_true:  int
    n_bouts_pred:  int

    def summary(self, states: Sequence[str] | None = None) -> str:
        lines = [
            f"frame accuracy : {self.accuracy:6.3f}",
            f"macro F1       : {self.macro_f1:6.3f}",
        ]
        for tau, v in sorted(self.segmental.items()):
            lines.append(f"segF1@{int(tau*100):<3d}     : {v:6.3f}")
        lines.append(f"boundary MAE   : {self.boundary_mae_s:6.3f} s")
        lines.append(
            f"bouts          : {self.n_bouts_pred} predicted "
            f"vs {self.n_bouts_true} true")
        if states is not None:
            per = ", ".join(
                f"{s}={v:.2f}" for s, v in zip(states, self.per_state_f1)
                if np.isfinite(v))
            lines.append(f"per-state F1   : {per}")
        return "\n".join(lines)


def evaluate_labeling(
    y_true:     np.ndarray,
    y_pred:     np.ndarray,
    n_states:   int,
    samplerate: float,
    *,
    iou_levels: Sequence[float] = (0.10, 0.25, 0.50),
) -> LabelingScore:
    """Compute the full metric bundle for one session."""
    acc, macro, per_state = frame_scores(y_true, y_pred, n_states)
    return LabelingScore(
        accuracy       = acc,
        macro_f1       = macro,
        per_state_f1   = per_state,
        segmental      = {t: segmental_f1(y_true, y_pred, iou=t)
                          for t in iou_levels},
        boundary_mae_s = boundary_mae(y_true, y_pred, samplerate),
        n_bouts_true   = len(segments_from_labels(y_true)),
        n_bouts_pred   = len(segments_from_labels(y_pred)),
    )
