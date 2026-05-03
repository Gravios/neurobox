"""
neurobox.analysis.feature_selection.core
==========================================
Mutual-information-based feature ranking.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


__all__ = [
    "MIFeatureRanking",
    "pairwise_mutual_information_ranking",
]


@dataclass
class MIFeatureRanking:
    """Output of :func:`pairwise_mutual_information_ranking`.

    Attributes
    ----------
    mi : np.ndarray, shape ``(n_features, n_classes)``
        Mutual information in bits between each feature and each
        binary "is-state-X" label.
    ranked_indices : np.ndarray, shape ``(n_features, n_classes)``
        Per-state, the feature column indices sorted by descending
        MI.  ``ranked_indices[0, k]`` is the feature most informative
        about class *k*.
    class_labels : tuple[str, ...]
        Names of the classes corresponding to the columns of
        ``mi`` and ``ranked_indices``.
    """
    mi:              np.ndarray
    ranked_indices:  np.ndarray
    class_labels:    tuple[str, ...]


# ─────────────────────────────────────────────────────────────────────── #
# Helpers                                                                    #
# ─────────────────────────────────────────────────────────────────────── #

def _mi_continuous_vs_binary(
    feature:    np.ndarray,
    label:      np.ndarray,
    n_bins:     int = 64,
) -> float:
    """Mutual information between a continuous feature and a binary label.

    Estimator:

    1. Bin *feature* into *n_bins* equal-width bins (over the data
       range).
    2. Build the 2-D joint histogram with the two label classes.
    3. Compute Shannon MI in bits.

    Mirrors the MATLAB ``hist2`` + ``log2`` chain in
    :file:`req20160310_5_genfigs.m`.
    """
    feature = np.asarray(feature, dtype=np.float64)
    label   = np.asarray(label).astype(int)
    finite = np.isfinite(feature)
    if not finite.any():
        return 0.0
    fv = feature[finite]
    lv = label[finite]

    edges = np.linspace(fv.min(), fv.max() + np.finfo(float).eps,
                         n_bins + 1)
    bin_idx = np.clip(np.digitize(fv, edges) - 1, 0, n_bins - 1)

    n = lv.size
    classes = np.unique(lv)
    if classes.size < 2:
        return 0.0

    pxy = np.zeros((n_bins, classes.size), dtype=np.float64)
    for ci, cls in enumerate(classes):
        sel = lv == cls
        counts = np.bincount(bin_idx[sel], minlength=n_bins)
        pxy[:, ci] = counts / n

    px = pxy.sum(axis=1, keepdims=True)             # marginal over label
    py = pxy.sum(axis=0, keepdims=True)             # marginal over feature
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where((pxy > 0) & (px > 0) & (py > 0),
                          pxy / (px * py), 1.0)
        terms = np.where(pxy > 0, pxy * np.log2(ratio), 0.0)
    return float(np.nansum(terms))


# ─────────────────────────────────────────────────────────────────────── #
# Public API                                                                 #
# ─────────────────────────────────────────────────────────────────────── #

def pairwise_mutual_information_ranking(
    features:        np.ndarray,
    labels:          np.ndarray,
    *,
    class_labels:    Optional[Sequence[str]] = None,
    n_bins:          int = 64,
) -> MIFeatureRanking:
    """Rank features by mutual information with each behavioural state.

    Port of the inner kernel of
    :file:`MTA/analysis/req/req20160310_5_genfigs.m` (the ``mibi``
    block; lines depending on MATLAB release).

    For each combination of (feature column, behavioural state),
    computes Shannon MI in bits between the (continuous) feature and
    the binary "is-this-state" label, then sorts the feature columns
    by descending MI per state.

    The MATLAB original computed this for hand-labelled epochs of
    {walk, rear, turn, pause, groom, sit} against the 30-column
    `fet_bref` set, then plotted the ranking as a heatmap to choose
    the most informative subset for downstream classifier training.

    Parameters
    ----------
    features:
        ``(T, n_features)`` feature matrix.
    labels:
        ``(T,)`` integer labels (one column per *T*-length vector) or
        ``(T, n_classes)`` boolean / 0-1 mask matrix (one column per
        class).  The 1-D form is converted to a column-stack of
        per-class indicators.
    class_labels:
        Names of the classes.  Required if *labels* is 2-D and you
        want them in the result.  Default ``("class_0", "class_1",
        ...)``.
    n_bins:
        Number of feature-axis histogram bins.  Default 64 matches
        MATLAB.

    Returns
    -------
    MIFeatureRanking
        ``mi[f, c]`` = MI of feature *f* with class *c* in bits.
        ``ranked_indices[:, c]`` = feature indices sorted by
        descending ``mi[:, c]``.
    """
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2:
        raise ValueError(
            f"features must be 2-D (T, n_features); got shape {features.shape}"
        )
    labels = np.asarray(labels)
    T, n_features = features.shape

    # Convert to binary one-hot (T, n_classes)
    if labels.ndim == 1:
        if labels.shape[0] != T:
            raise ValueError(
                f"labels length {labels.shape[0]} ≠ features T={T}"
            )
        classes = np.unique(labels)
        binary = np.zeros((T, classes.size), dtype=np.int8)
        for ci, cls in enumerate(classes):
            binary[:, ci] = (labels == cls).astype(np.int8)
        if class_labels is None:
            class_labels = tuple(str(c) for c in classes)
    elif labels.ndim == 2:
        if labels.shape[0] != T:
            raise ValueError(
                f"labels rows {labels.shape[0]} ≠ features T={T}"
            )
        binary = labels.astype(np.int8)
        if class_labels is None:
            class_labels = tuple(f"class_{i}" for i in range(binary.shape[1]))
    else:
        raise ValueError(
            f"labels must be 1-D or 2-D; got shape {labels.shape}"
        )
    if len(class_labels) != binary.shape[1]:
        raise ValueError(
            f"class_labels length {len(class_labels)} ≠ "
            f"n_classes {binary.shape[1]}"
        )

    n_classes = binary.shape[1]
    mi = np.zeros((n_features, n_classes), dtype=np.float64)
    for c in range(n_classes):
        cls_label = binary[:, c]
        if cls_label.sum() == 0 or cls_label.sum() == T:
            # Empty or full class — no information
            continue
        for f in range(n_features):
            mi[f, c] = _mi_continuous_vs_binary(
                features[:, f], cls_label, n_bins=n_bins,
            )

    ranked = np.argsort(-mi, axis=0)              # descending
    return MIFeatureRanking(
        mi              = mi,
        ranked_indices  = ranked,
        class_labels    = tuple(class_labels),
    )
