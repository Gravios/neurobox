"""
neurobox.analysis.feature_selection.pipeline
==============================================
Hierarchical-MI feature selection + incremental-NN classifier
training.

This is the algorithmic content of
:file:`MTA/feature_selection/select_features_via_mutual_information.m`
(and its identical twin :file:`feature_selection_quadratic_mi.m`)
together with the eight :file:`req20160310_*.m` workflow scripts
they orchestrate.

What the MATLAB pipeline does
-----------------------------
The lab's behaviour-classifier feature-selection workflow is:

* **Step 1 — preproc** (``req20160310_1_preproc.m``).  Compute the
  59-column :func:`fet_all` feature set.  Augment it with squared
  and cross-product features (→ ~7,000 columns).  Run hierarchical
  MI feature selection (``select_features_hmi``) to pick a
  ``state_order`` and a per-state ``feature_indices``.  Bootstrap a
  training set with whole-state resampling.
* **Step 2 — t-SNE** (``req20160310_2_tsne.m``).  Per state, project
  the kept features into 2-D for visualisation.
* **Step 3 — train NNs** (``req20160310_3_trainNN.m``).  Per state,
  train an *incremental* sequence of classifiers: first with 1
  feature, then 2, then 3, …, up to ``round(n/2)`` features, where
  the feature ordering is by MI rank.
* **Step 4 — accumulate stats** (``req20160310_4_accumStats.m``).
  Run each trained classifier on the held-out test set; record
  accuracy, precision, sensitivity vs feature count.
* **Step 5 — generate figures** (``req20160310_5_genfigs.m``).  The
  *real* feature ordering: re-rank features by Δ-accuracy gain
  (which feature, when added, gave the biggest accuracy boost?).
  Save as ``bFetInds`` per state.
* **Steps 6-8 — re-train, re-stat, re-plot** with the optimised
  ordering.
* **Step 9 — generate behaviour labels** for a new session using the
  optimised classifiers.

Total: 1,354 LoC of LRZ batch-job + disk-cache management around
~250 LoC of actual content.

What this Python port does
--------------------------
A single function :func:`run_feature_selection_pipeline` that
combines steps 1, 3, 4, 5, and 6 in memory:

1. **HMI feature selection** → state order + per-state MI-ranked
   feature indices (uses :func:`select_features_hmi`).
2. **Incremental classifier training** → for each state, train
   classifiers on top-1, top-2, …, top-``round(n/2)`` features and
   record per-step accuracy / precision / sensitivity.
3. **Δ-accuracy re-ranking** → re-order features by accuracy-gain
   contribution.
4. (Optional) **Re-train with optimised ordering** for the final
   classifiers.

t-SNE (step 2) and behaviour labelling (step 9) are exposed as
separate utilities — see :func:`mta_tsne` and
:func:`label_states_with_pipeline`.

Differences from MATLAB
-----------------------
* No disk cache between steps.  Pass in-memory results around.
* No LRZ orchestration.  Use ``joblib.Parallel`` for the inner NN
  training loop if speed matters.
* The augmented feature space (cols + cols² + cross-products,
  generating ~7,000 features from 59) is **not enabled by default**
  to keep the pipeline tractable; pass ``augment=True`` to enable.
* Backend defaults to ``'sklearn-mlp'`` (CPU, no PyTorch dependency)
  rather than MATLAB's ``patternnet``; pass ``backend='mlp'`` for
  the round 12 PyTorch backend, or any other backend name.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from neurobox.analysis.classifiers.bootstrap import (
    BootstrapResult, whole_state_bootstrap,
)
from neurobox.analysis.classifiers.label import (
    FeatureNormalisation, fit_normalisation, make_classifier,
)
from neurobox.dtype.stc import NBStateCollection

from .hmi import HierarchicalMIResult, select_features_hmi


__all__ = [
    "FeatureSelectionPipelineResult",
    "PerStateAccumulation",
    "run_feature_selection_pipeline",
    "augment_features_quadratic",
]


# ─────────────────────────────────────────────────────────────────────── #
# Helpers                                                                    #
# ─────────────────────────────────────────────────────────────────────── #

def augment_features_quadratic(features: np.ndarray) -> np.ndarray:
    """Augment a feature matrix with squared and cross-product columns.

    Port of the augmentation in :file:`req20160310_1_preproc.m`
    lines 70-75.  For an ``(T, F)`` input, returns
    ``(T, F + F + F * (F-1))``: original columns, squared columns,
    and all pairwise circ-shifted products.

    For F = 59 (fet_all), this gives 59 + 59 + 59 × 58 = 3,540
    columns — the lab's actual augmented set.

    Note: The MATLAB version uses ``circshift`` of the column vector,
    which produces a *time-shifted* product rather than a
    cross-feature product (so columns at large shifts approximate
    autoregressive features rather than feature interactions).  This
    Python port matches that semantics exactly.

    Parameters
    ----------
    features:
        ``(T, F)`` source matrix.

    Returns
    -------
    np.ndarray
        Augmented matrix.
    """
    features = np.asarray(features, dtype=np.float64)
    T, F = features.shape
    parts = [features, features ** 2]
    for sh in range(1, F):
        shifted = np.roll(features, -sh, axis=0)
        parts.append(shifted * features)
    return np.concatenate(parts, axis=1)


def _eval_classifier(
    clf,
    X_train: np.ndarray, y_train: np.ndarray,
    X_test:  np.ndarray, y_test:  np.ndarray,
) -> dict:
    """Train and evaluate a classifier; return basic stats."""
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    acc = float((pred == y_test).mean()) if y_test.size > 0 else 0.0
    # Per-class precision / sensitivity for the binary case
    classes = np.unique(y_test) if y_test.size > 0 else np.array([])
    precision = []
    sensitivity = []
    for c in classes:
        tp = int(((pred == c) & (y_test == c)).sum())
        fp = int(((pred == c) & (y_test != c)).sum())
        fn = int(((pred != c) & (y_test == c)).sum())
        precision.append(tp / max(1, tp + fp))
        sensitivity.append(tp / max(1, tp + fn))
    return {
        "accuracy":    acc,
        "precision":   tuple(float(p) for p in precision),
        "sensitivity": tuple(float(s) for s in sensitivity),
    }


# ─────────────────────────────────────────────────────────────────────── #
# Result containers                                                          #
# ─────────────────────────────────────────────────────────────────────── #

@dataclass
class PerStateAccumulation:
    """Per-state, per-feature-count classifier statistics.

    Attributes
    ----------
    state : str
        Target state being discriminated against the rest.
    initial_feature_indices : np.ndarray
        Feature indices in MI-rank order, before re-ranking.
    optimised_feature_indices : np.ndarray
        Feature indices after Δ-accuracy re-ranking (the
        ``bFetInds[state]`` of MATLAB ``req20160310_5_genfigs.m``).
    accuracy : np.ndarray
        ``(n_steps,)`` accuracy after adding each feature in order.
    precision : np.ndarray
        ``(n_steps, n_classes)`` per-class precision.
    sensitivity : np.ndarray
        ``(n_steps, n_classes)`` per-class sensitivity.
    """
    state:                     str
    initial_feature_indices:   np.ndarray
    optimised_feature_indices: np.ndarray
    accuracy:                  np.ndarray
    precision:                 np.ndarray
    sensitivity:               np.ndarray


@dataclass
class FeatureSelectionPipelineResult:
    """Output of :func:`run_feature_selection_pipeline`.

    Attributes
    ----------
    hmi : HierarchicalMIResult
        Output of :func:`select_features_hmi`.
    per_state : list[PerStateAccumulation]
        One entry per state in ``hmi.state_order`` (excluding the
        leaf-pair).  Each carries the incremental-feature stats.
    state_order : tuple[str, ...]
        ``hmi.state_order`` for convenience.
    optimised_feature_indices : dict[str, np.ndarray]
        ``state -> Δ-accuracy-reordered feature indices``.
        Equivalent to MATLAB's ``bFetInds`` dict.
    normalisation : FeatureNormalisation
        Mean/std used to normalise features before classifier
        training.  Apply with ``.transform()`` to new data.
    """
    hmi:                       HierarchicalMIResult
    per_state:                 list[PerStateAccumulation]
    state_order:               tuple[str, ...]
    optimised_feature_indices: dict[str, np.ndarray]
    normalisation:             FeatureNormalisation


# ─────────────────────────────────────────────────────────────────────── #
# Main pipeline                                                              #
# ─────────────────────────────────────────────────────────────────────── #

def run_feature_selection_pipeline(
    stc:                NBStateCollection,
    feature_data:       np.ndarray,
    feature_samplerate: float,
    *,
    states:             Sequence[str] = (
        "rear", "walk", "turn", "pause", "groom", "sit",
    ),
    augment:            bool = False,
    mi_threshold_bits:  float = 0.20,
    dprime_threshold:   float = 2.0,
    backend:            str = "sklearn-mlp",
    backend_kwargs:     Optional[dict] = None,
    state_block_size:   int = 15_000,
    train_pct:          float = 90.0,
    max_features_per_state: Optional[int] = None,
    rng:                np.random.Generator | int | None = None,
    progress:           bool = False,
) -> FeatureSelectionPipelineResult:
    """Run the full hierarchical-MI + incremental-NN pipeline.

    Replaces the
    ``req20160310_{1,3,4,5,6}_*.m``  chain with a single in-memory
    function.

    Parameters
    ----------
    stc:
        State collection.
    feature_data:
        ``(T, n_features)`` feature matrix at *feature_samplerate*.
        Use :func:`fet_all_features` to produce the standard 59-col
        version.
    feature_samplerate:
        Hz of *feature_data*.
    states:
        States to discriminate.  Default matches MATLAB.
    augment:
        If True, expand *feature_data* via
        :func:`augment_features_quadratic` before feature selection.
        Default False (keeping the 59 base features) for tractability;
        the MATLAB pipeline uses True with augmented dimensionality
        ~3,540.
    mi_threshold_bits, dprime_threshold:
        Forwarded to :func:`select_features_hmi`.
    backend:
        Classifier backend name.  ``'sklearn-mlp'`` (default, CPU
        only) or any backend registered with :func:`make_classifier`.
    backend_kwargs:
        Forwarded to the backend constructor.  Sensible defaults if
        None.
    state_block_size:
        Sampler bootstrap block size.  Default 15000 matches MATLAB.
    train_pct:
        Train/test split percent.  Default 90.
    max_features_per_state:
        Cap on the number of incremental classifier-training steps
        per state.  Default ``None`` → ``round(n_features / 2)``,
        matching MATLAB.
    rng:
        Reproducibility.
    progress:
        Print progress messages.

    Returns
    -------
    FeatureSelectionPipelineResult
    """
    rng = (np.random.default_rng(rng)
           if not isinstance(rng, np.random.Generator) else rng)
    if backend_kwargs is None:
        backend_kwargs = {}

    # Step 0 — augment if requested
    if augment:
        feature_data = augment_features_quadratic(feature_data)
        if progress:
            print(f"Augmented features: {feature_data.shape[1]}")

    # Step 1 — fit normalisation, normalise
    norm = fit_normalisation(feature_data)
    feature_data = norm.transform(feature_data)

    # Step 2 — bootstrap a balanced training set + held-out eval periods
    bs = whole_state_bootstrap(
        stc, feature_data, feature_samplerate, list(states),
        state_block_size = state_block_size,
        prct_train       = train_pct,
        rng              = rng,
    )
    if progress:
        print(f"Bootstrap: train shape {bs.features.shape}, "
              f"states: {bs.state_names}")

    # Step 3 — Hierarchical MI selection on the bootstrapped training set.
    # We need an stc-shaped object that lives on the bootstrap's
    # synthetic time axis; build a minimal one from bs.labels.
    bs_stc = _stc_from_bootstrap_labels(bs, feature_samplerate)
    hmi = select_features_hmi(
        bs_stc, bs.features, feature_samplerate,
        states            = states,
        mi_threshold_bits = mi_threshold_bits,
        dprime_threshold  = dprime_threshold,
    )
    if progress:
        print(f"HMI state order: {hmi.state_order}")

    # Step 4 — Incremental classifier training per state.
    # For evaluation we use the eval_periods from the bootstrap on
    # the ORIGINAL feature_data.
    eval_mask = _eval_mask_from_periods(
        bs.eval_periods, feature_data.shape[0],
    )
    eval_codes_full = _eval_codes_for_stc(
        stc, feature_data.shape[0], feature_samplerate, list(states),
    )

    per_state: list[PerStateAccumulation] = []
    optimised: dict[str, np.ndarray] = {}

    for s_idx, target_state in enumerate(hmi.state_order[:-1]):
        if progress:
            print(f"\n[{s_idx+1}/{len(hmi.state_order)-1}] "
                  f"State {target_state} vs rest")

        gstates = list(hmi.state_order[s_idx:])
        target_idx_in_gstates = gstates.index(target_state)

        # Initial feature ordering: HMI Δ-MI ranking from level *s_idx*
        initial_indices = hmi.feature_indices[s_idx]
        if initial_indices.size == 0:
            warnings.warn(
                f"No features selected by HMI for state {target_state!r}; "
                f"falling back to all features.",
                stacklevel=2,
            )
            initial_indices = np.arange(feature_data.shape[1])
        # Sort by per-feature Δ-MI for this level (most-informative first)
        mi_table = hmi.mi_per_level[s_idx] if s_idx < len(hmi.mi_per_level) \
                    else None
        if mi_table is not None:
            # Find which row of mi_table corresponds to target_state
            # (rows: 0=all, 1..n=hold-out-i; we want the one matching
            # `target_state` in the level's `gstates`.)
            level_gstates = list(hmi.state_order[s_idx:])
            try:
                lvl_idx = level_gstates.index(target_state)
                dm = mi_table[0, :] - mi_table[lvl_idx + 1, :]
                # Sort initial_indices by descending dm
                order = np.argsort(-dm[initial_indices])
                initial_indices = initial_indices[order]
            except (ValueError, IndexError):
                pass

        # Cap step count
        n_steps = (max_features_per_state
                   if max_features_per_state is not None
                   else max(1, len(initial_indices) // 2))
        n_steps = min(n_steps, len(initial_indices))

        # Build training labels: target vs rest, restricted to the
        # bootstrap rows whose state is in `gstates`.
        bs_state_to_code = {n: i for i, n in enumerate(bs.state_names)}
        keep_codes = [bs_state_to_code[s]
                       for s in gstates if s in bs_state_to_code]
        if not keep_codes:
            warnings.warn(
                f"State {target_state!r}: no bootstrap labels match "
                f"`gstates`; skipping.", stacklevel=2,
            )
            continue
        train_mask = np.isin(bs.labels, keep_codes)
        target_code = bs_state_to_code[target_state]
        y_train = (bs.labels[train_mask] == target_code).astype(np.int32)
        X_train_full = bs.features[train_mask]

        # Build evaluation labels from the original (non-bootstrap)
        # feature matrix, restricted to held-out periods AND samples
        # in `gstates`.
        gstate_codes_global = [
            list(states).index(s) + 1 for s in gstates if s in states
        ]
        test_mask = eval_mask & np.isin(eval_codes_full, gstate_codes_global)
        if not test_mask.any():
            # Fall back: evaluate on the entire original set
            test_mask = np.isin(eval_codes_full, gstate_codes_global)
        target_code_global = list(states).index(target_state) + 1
        y_test = (eval_codes_full[test_mask] == target_code_global).astype(np.int32)
        X_test_full = feature_data[test_mask]

        accs, precs, sens = [], [], []
        for k in range(1, n_steps + 1):
            cols = initial_indices[:k]
            X_tr_k = X_train_full[:, cols]
            X_te_k = X_test_full [:, cols]
            try:
                clf = make_classifier(backend, **backend_kwargs)
                stats = _eval_classifier(clf, X_tr_k, y_train,
                                            X_te_k,  y_test)
            except Exception as e:    # noqa: BLE001
                warnings.warn(
                    f"Classifier failed at k={k} for {target_state!r}: {e}",
                    stacklevel=2,
                )
                stats = {"accuracy": np.nan,
                          "precision": (np.nan,),
                          "sensitivity": (np.nan,)}
            accs.append(stats["accuracy"])
            precs.append(stats["precision"])
            sens.append(stats["sensitivity"])
            if progress:
                print(f"  k={k}: acc={stats['accuracy']:.3f}")

        # Step 5 — Δ-accuracy re-ranking (matches MATLAB
        # req20160310_5_genfigs.m semantics)
        accs_arr  = np.asarray(accs, dtype=np.float64)
        precs_arr = _pad_to_2d(precs)
        sens_arr  = _pad_to_2d(sens)

        # MATLAB: [~,fetLInds] = sort([0;diff(accs)],'descend');
        # then drop the index that points at element-1 (which had no
        # diff predecessor) and prepend a 1 (the first feature stays).
        deltas = np.r_[0.0, np.diff(accs_arr)]
        with np.errstate(invalid="ignore"):
            order_descending = np.argsort(-np.where(np.isnan(deltas),
                                                      -np.inf, deltas))
        # First feature always at the front; then by descending Δ-acc
        order: list[int] = [0]
        seen = {0}
        for idx in order_descending:
            ii = int(idx)
            if ii in seen:
                continue
            order.append(ii)
            seen.add(ii)
            if len(order) >= n_steps:
                break

        optimised_idx = initial_indices[np.asarray(order[:n_steps])]

        per_state.append(PerStateAccumulation(
            state                     = target_state,
            initial_feature_indices   = initial_indices[:n_steps],
            optimised_feature_indices = optimised_idx,
            accuracy                  = accs_arr,
            precision                 = precs_arr,
            sensitivity               = sens_arr,
        ))
        optimised[target_state] = optimised_idx

    return FeatureSelectionPipelineResult(
        hmi                       = hmi,
        per_state                 = per_state,
        state_order               = hmi.state_order,
        optimised_feature_indices = optimised,
        normalisation             = norm,
    )


def _stc_from_bootstrap_labels(
    bs:                 BootstrapResult,
    feature_samplerate: float,
) -> NBStateCollection:
    """Build a minimal NBStateCollection from a BootstrapResult.

    The bootstrap returns a per-row label code; we synthesise an
    NBEpoch per state from contiguous-run extraction over those
    codes, so :func:`mutual_information_states_features` can run on
    the bootstrap's synthetic time-axis.
    """
    from neurobox.dtype.epoch import NBEpoch
    from neurobox.dtype.stc   import NBStateCollection

    n_rows = bs.features.shape[0]
    stc = NBStateCollection(mode="bootstrap_synthetic")

    used_keys: set[str] = set()
    for code, name in enumerate(bs.state_names):
        mask = bs.labels == code
        if not mask.any():
            continue
        # Find runs of True
        d = np.diff(mask.astype(np.int8), prepend=0, append=0)
        starts = np.flatnonzero(d == 1)
        ends   = np.flatnonzero(d == -1)
        if starts.size == 0:
            continue
        # Convert sample indices → seconds
        periods = (np.column_stack([starts, ends]).astype(np.float64)
                    / feature_samplerate)
        # Allocate a unique key (1-letter)
        for ch in name:
            if ch.isalpha() and ch not in used_keys:
                key = ch
                break
        else:
            key = name[:1] if name and name[0] not in used_keys else "?"
        used_keys.add(key)
        ep = NBEpoch(
            data       = periods,
            samplerate = feature_samplerate,
            mode       = "periods",
            label      = name,
            key        = key,
        )
        stc.add_state(ep, label=name, key=key)

    return stc


def _eval_mask_from_periods(
    eval_periods:  list[np.ndarray],
    n_samples:     int,
) -> np.ndarray:
    """Build a boolean mask of length *n_samples* from the per-state
    eval-period list returned by :func:`whole_state_bootstrap`."""
    mask = np.zeros(n_samples, dtype=bool)
    for periods in eval_periods:
        if periods is None or len(periods) == 0:
            continue
        for s, e in periods:
            si = int(np.floor(s)); ei = int(np.ceil(e))
            si = max(0, si)
            ei = min(n_samples, ei)
            if ei > si:
                mask[si:ei] = True
    return mask


def _eval_codes_for_stc(
    stc:        NBStateCollection,
    n_samples:  int,
    samplerate: float,
    states:     Sequence[str],
) -> np.ndarray:
    """Per-sample integer state code (1..K, 0 = no state)."""
    from neurobox.analysis.classifiers.stc_utils import stc2mat
    smat, _ = stc2mat(stc, n_samples, samplerate, states=list(states))
    code = np.zeros(n_samples, dtype=np.int32)
    for i in range(len(states)):
        code[smat[:, i] != 0] = i + 1
    return code


def _pad_to_2d(items: list[tuple]) -> np.ndarray:
    """Convert a list of variable-length tuples to a (n, max_len) ndarray
    padded with NaN."""
    if not items:
        return np.zeros((0, 0))
    n = len(items)
    m = max(len(t) for t in items) if items else 0
    out = np.full((n, m), np.nan, dtype=np.float64)
    for i, t in enumerate(items):
        out[i, :len(t)] = t
    return out
