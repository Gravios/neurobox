"""
neurobox.analysis.classifiers.label
=====================================
Orchestrate end-to-end behavioural-state labelling using one of the
:mod:`neurobox.analysis.classifiers` backends.

Port of :file:`MTA/classifiers/bhv_nn.m` and
:file:`bhv_nn_multi_session_patternnet.m` (the multi-session
ensemble variant).

What this module does
---------------------
The MATLAB ``bhv_nn_multi_session_patternnet`` does six things:

1. Concatenate features across multiple training sessions.
2. (optional) Map all sessions onto a single reference session's
   feature space and z-score them with a shared mean/std.
3. For each of *n_iter* iterations:
   a. Whole-state-bootstrap-noisy-trim resample the training matrix.
   b. Train a fresh patternnet on that bootstrap.
   c. Predict on every labelling session, accumulate softmax outputs.
4. Average softmax outputs across iterations.
5. Take argmax to get integer state labels.
6. Median-smooth the integer labels, threshold-cross into period
   intervals, write back to an :class:`NBStateCollection`.

This port factors that out into:

* :func:`label_states` — the full pipeline, defaulting to
  ``backend='patternnet'`` with the same ensemble defaults as MATLAB.
* :func:`train_classifier_ensemble` — just step 3 (returns a list of
  fitted :class:`Classifier` objects).
* :func:`predict_with_ensemble` — just step 3-4 (averaged softmax).
* :func:`smooth_labels_to_state_collection` — just step 5-6.

Map-to-reference is **deferred**: the ``map_to_reference_session`` step
(MATLAB's manifold-based per-feature shift correction) was deferred
in round 10 and is not yet ported.  Pass ``map_to_reference=False``
to skip it (the default), or rely on a pre-aligned feature matrix.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.signal import medfilt

from neurobox.analysis.lfp.oscillations import thresh_cross
from neurobox.dtype.epoch import NBEpoch
from neurobox.dtype.fet import NBDfet
from neurobox.dtype.stc import NBStateCollection

from . import _registry
from .base import Classifier
from .bootstrap import whole_state_bootstrap


# ──────────────────────────────────────────────────────────────────── #
# Lazy backend importers                                                #
# ──────────────────────────────────────────────────────────────────── #

def _import_torch_backends() -> None:
    from . import torch_backends  # noqa: F401


def _import_sklearn_backends() -> None:
    from . import sklearn_backends  # noqa: F401


# Register the lazy importers (so users can construct a backend by
# string name without having to manually import the defining module).
for _b in ("patternnet", "mlp", "cnn", "lstm"):
    _registry.register_lazy(_b, "neurobox.analysis.classifiers.torch_backends",
                             _import_torch_backends)
for _b in ("sklearn-mlp", "rf", "gbm"):
    _registry.register_lazy(_b, "neurobox.analysis.classifiers.sklearn_backends",
                             _import_sklearn_backends)


def make_classifier(backend: str, **kwargs) -> Classifier:
    """Construct a :class:`Classifier` by backend name.

    Parameters
    ----------
    backend:
        One of ``'patternnet'``, ``'mlp'``, ``'cnn'``, ``'lstm'``,
        ``'sklearn-mlp'``, ``'rf'``, ``'gbm'``.
    **kwargs:
        Forwarded to the backend's constructor.

    Returns
    -------
    Classifier
    """
    cls = _registry.get_classifier_class(backend)
    return cls(**kwargs)


# ──────────────────────────────────────────────────────────────────── #
# Normalisation                                                          #
# ──────────────────────────────────────────────────────────────────── #

@dataclass
class FeatureNormalisation:
    """Mean/std + optional clip for z-score normalisation.

    Mirrors the role of MATLAB's ``mapminmax``/``nunity`` parameters.
    """
    mean: np.ndarray
    std:  np.ndarray
    clip: float | None = None

    def transform(self, X: np.ndarray) -> np.ndarray:
        out = (X - self.mean) / np.where(self.std == 0, 1.0, self.std)
        if self.clip is not None:
            np.clip(out, -self.clip, self.clip, out=out)
        return out

    def to_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist(),
                "clip": self.clip}

    @classmethod
    def from_dict(cls, d: dict) -> "FeatureNormalisation":
        return cls(mean=np.asarray(d["mean"]), std=np.asarray(d["std"]),
                   clip=d.get("clip"))


def fit_normalisation(
    feature_data: np.ndarray,
    clip: float | None = None,
) -> FeatureNormalisation:
    """Fit a :class:`FeatureNormalisation` from finite/non-zero rows of *feature_data*.

    Mirrors MATLAB's ``nunity`` mean/std estimate which excluded
    NaN, Inf and all-zero rows.
    """
    finite = np.isfinite(feature_data).all(axis=1) & (feature_data != 0).any(axis=1)
    valid  = feature_data[finite] if finite.any() else feature_data
    return FeatureNormalisation(
        mean = np.nanmean(valid, axis=0),
        std  = np.nanstd(valid, axis=0),
        clip = clip,
    )


# ──────────────────────────────────────────────────────────────────── #
# Ensemble training                                                      #
# ──────────────────────────────────────────────────────────────────── #

@dataclass
class TrainedEnsemble:
    """A trained ensemble of classifiers + the data-prep pipeline.

    Attributes
    ----------
    classifiers:
        List of fitted :class:`Classifier` objects (one per iteration).
    backend:
        Backend identifier (same for every classifier).
    state_names:
        State labels in the order they appear as integer codes.
    normalisation:
        Optional :class:`FeatureNormalisation` to apply before
        prediction; ``None`` means features are used as-is.
    """
    classifiers:   list[Classifier]
    backend:       str
    state_names:   list[str]
    normalisation: FeatureNormalisation | None = None

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Average softmax over the ensemble.

        Returns ``(T, n_states)``.
        """
        if self.normalisation is not None:
            X = self.normalisation.transform(X)
        accum = None
        for clf in self.classifiers:
            p = clf.predict_proba(X)
            accum = p if accum is None else (accum + p)
        return accum / len(self.classifiers)

    def save(self, dirpath: Path | str) -> None:
        """Save the ensemble to a directory.

        Layout::

            <dirpath>/
                ensemble.json
                iter_0/                (Classifier.save layout)
                iter_1/
                ...
        """
        dirpath = Path(dirpath)
        dirpath.mkdir(parents=True, exist_ok=True)
        meta = {
            "backend":       self.backend,
            "state_names":   self.state_names,
            "n_iter":        len(self.classifiers),
            "normalisation": self.normalisation.to_dict()
                              if self.normalisation is not None else None,
        }
        (dirpath / "ensemble.json").write_text(json.dumps(meta, indent=2))
        for i, clf in enumerate(self.classifiers):
            clf.save(dirpath / f"iter_{i}")

    @classmethod
    def load(cls, dirpath: Path | str) -> "TrainedEnsemble":
        dirpath = Path(dirpath)
        meta = json.loads((dirpath / "ensemble.json").read_text())
        clfs = [Classifier.load(dirpath / f"iter_{i}")
                for i in range(meta["n_iter"])]
        norm = (FeatureNormalisation.from_dict(meta["normalisation"])
                if meta.get("normalisation") else None)
        return cls(
            classifiers   = clfs,
            backend       = meta["backend"],
            state_names   = list(meta["state_names"]),
            normalisation = norm,
        )


def train_classifier_ensemble(
    sessions:    Sequence[tuple[NBStateCollection, np.ndarray, float]],
    states:      Sequence[str],
    *,
    backend:     str = "patternnet",
    n_iter:      int = 10,
    classifier_kwargs: dict | None = None,
    bootstrap_kwargs:  dict | None = None,
    normalise:   bool = True,
    norm_clip:   float | None = 5.0,
    rng:         np.random.Generator | None = None,
    verbose:     bool = False,
) -> TrainedEnsemble:
    """Train an ensemble of classifiers on multi-session features.

    Parameters
    ----------
    sessions:
        Iterable of ``(stc, feature_data, feature_samplerate)`` tuples,
        one per training session.  ``stc`` must contain labelled
        periods for every state in *states*; ``feature_data`` is the
        ``(T, n_features)`` matrix at *feature_samplerate* Hz.
    states:
        State labels to train on, in the order of the integer codes.
    backend:
        Classifier backend identifier (see :func:`make_classifier`).
        Default ``'patternnet'``.
    n_iter:
        Number of bootstrap iterations.  Default 10 matches MATLAB's
        ``bhv_nn_multi_session_patternnet``.
    classifier_kwargs:
        Extra keyword arguments forwarded to the backend constructor
        on each iteration.
    bootstrap_kwargs:
        Extra keyword arguments forwarded to
        :func:`whole_state_bootstrap`.
    normalise:
        If True (default), z-score features with a shared mean/std
        estimated from the concatenated training matrix before
        bootstrapping.  Mirrors MATLAB's ``nunity`` step.
    norm_clip:
        Optional symmetric clip (in σ) applied after z-scoring.
        Default 5.  Pass None to disable.
    rng:
        Optional random generator for reproducibility.
    verbose:
        Print per-iteration progress.

    Returns
    -------
    TrainedEnsemble
    """
    if rng is None:
        rng = np.random.default_rng()
    classifier_kwargs = dict(classifier_kwargs or {})
    bootstrap_kwargs  = dict(bootstrap_kwargs or {})

    # ── Estimate shared normalisation across all sessions ────────── #
    norm = None
    if normalise:
        all_X = np.concatenate(
            [s[1] for s in sessions if s[1].shape[0] > 0],
            axis=0,
        )
        norm = fit_normalisation(all_X, clip=norm_clip)
        # Apply it to a working copy of each session's features.
        sessions_z = [
            (stc, norm.transform(X), fs) for stc, X, fs in sessions
        ]
    else:
        sessions_z = list(sessions)

    # ── Train n_iter classifiers ─────────────────────────────────── #
    clfs: list[Classifier] = []
    for it in range(n_iter):
        if verbose:
            print(f"[ensemble] iter {it + 1}/{n_iter}: bootstrapping ...")
        # Bootstrap one matrix per session, then concatenate.
        Xb_blocks: list[np.ndarray] = []
        yb_blocks: list[np.ndarray] = []
        for stc, X, fs in sessions_z:
            res = whole_state_bootstrap(
                stc, X, fs, states=states, rng=rng, **bootstrap_kwargs,
            )
            Xb_blocks.append(res.features)
            yb_blocks.append(res.labels)
        Xb = np.concatenate(Xb_blocks, axis=0)
        yb = np.concatenate(yb_blocks, axis=0)

        if verbose:
            print(f"           training {backend} on {Xb.shape[0]} samples")
        clf = make_classifier(backend, **classifier_kwargs)
        clf.fit(Xb, yb)
        clfs.append(clf)

    return TrainedEnsemble(
        classifiers   = clfs,
        backend       = backend,
        state_names   = list(states),
        normalisation = norm,
    )


def predict_with_ensemble(
    ensemble:    TrainedEnsemble,
    feature_data: np.ndarray,
) -> np.ndarray:
    """Convenience: ``ensemble.predict_proba(feature_data)``."""
    return ensemble.predict_proba(feature_data)


# ──────────────────────────────────────────────────────────────────── #
# Decision smoothing + state writeout                                    #
# ──────────────────────────────────────────────────────────────────── #

def smooth_labels_to_state_collection(
    label_probs:        np.ndarray,
    state_names:        Sequence[str],
    feature_samplerate: float,
    *,
    smoothing_window_s: float = 0.2,
    valid_mask:         np.ndarray | None = None,
    state_keys:         Sequence[str] | None = None,
    stc:                NBStateCollection | None = None,
    overwrite:          bool = True,
) -> NBStateCollection:
    """Convert per-sample softmax outputs into state-collection periods.

    Mirrors the post-processing tail of :file:`bhv_nn.m`:

    1. Argmax → integer state code per sample.
    2. Mark invalid samples (``valid_mask=False``) as code -1.
    3. Median-filter the integer labels with a window of
       ``smoothing_window_s`` (matches MATLAB's "200 ms state minimum").
    4. For each state, threshold-cross into ``(start, stop)`` sample
       index intervals.
    5. Add states to *stc* (creating one if not given).

    Parameters
    ----------
    label_probs:
        ``(T, n_states)`` softmax / probability output (typically from
        :meth:`TrainedEnsemble.predict_proba`).
    state_names:
        Ordered state labels (column index → label).
    feature_samplerate:
        Hz at which *label_probs* was computed.  Used to
        time-stamp the resulting state periods (in seconds).
    smoothing_window_s:
        Median-filter window in seconds.  Default 0.2 matches MATLAB.
    valid_mask:
        Optional ``(T,)`` boolean mask; samples where False are
        marked unlabelled (no state assigned).
    state_keys:
        Optional single-character state keys for the
        :class:`NBStateCollection`.  Default ``None`` →
        ``state_names[i][0]``.
    stc:
        Existing state collection to append to.  ``None`` (default) →
        construct a new empty one.
    overwrite:
        If True (default) and an existing state with the same label
        is present, it is replaced; if False, raises.

    Returns
    -------
    NBStateCollection
        Always with one state per *state_names* entry, even if the
        period list for that state is empty.
    """
    if state_keys is None:
        state_keys = [n[0] for n in state_names]
    if len(state_keys) != len(state_names):
        raise ValueError(
            "state_keys must have the same length as state_names"
        )

    label_probs = np.asarray(label_probs)
    T, n_states = label_probs.shape

    # Argmax → integer labels in [0, n_states-1]
    labels = np.argmax(label_probs, axis=1).astype(np.int32)

    # Invalid samples: code -1
    if valid_mask is not None:
        valid_mask = np.asarray(valid_mask, dtype=bool)
        if valid_mask.shape[0] != T:
            raise ValueError(
                f"valid_mask length {valid_mask.shape[0]} does not match "
                f"label_probs length {T}"
            )
        labels[~valid_mask] = -1

    # Median filter — odd-length window in samples
    win = int(round(smoothing_window_s * feature_samplerate))
    if win % 2 == 0:
        win += 1
    if win >= 3 and T >= win:
        # medfilt on int array with -1 sentinels: do +2 shift so all values
        # are non-negative before the median, then shift back
        shifted = labels + 2
        smoothed = medfilt(shifted, kernel_size=win) - 2
        labels = smoothed.astype(np.int32)

    # State writeout
    if stc is None:
        stc = NBStateCollection()

    for i, (label, key) in enumerate(zip(state_names, state_keys)):
        active = (labels == i).astype(np.float64)
        # thresh_cross returns sample-index intervals where active>0.5
        ivs = thresh_cross(active, threshold=0.5)
        if ivs.size:
            # Convert (n,2) sample indices → seconds
            periods = ivs.astype(np.float64) / feature_samplerate
        else:
            periods = np.zeros((0, 2), dtype=np.float64)

        ep = NBEpoch(
            data       = periods,
            samplerate = 1.0,                 # periods are in seconds
            label      = label,
            key        = key,
        )
        # If a state with this label already exists, decide overwrite
        if stc.has_state(label):
            if not overwrite:
                raise ValueError(
                    f"State {label!r} already in stc; pass overwrite=True"
                )
            # Reach into the private dict to avoid double-add
            del stc._states[label]
            for k_, lbl_ in list(stc._keys.items()):
                if lbl_ == label:
                    del stc._keys[k_]
        stc.add_state(ep, label=label, key=key)

    return stc


# ──────────────────────────────────────────────────────────────────── #
# End-to-end one-shot                                                    #
# ──────────────────────────────────────────────────────────────────── #

def label_states(
    train_sessions:   Sequence[tuple[NBStateCollection, np.ndarray, float]],
    target_features:  np.ndarray,
    target_samplerate: float,
    states:           Sequence[str],
    *,
    backend:          str = "patternnet",
    n_iter:           int = 10,
    classifier_kwargs: dict | None = None,
    bootstrap_kwargs:  dict | None = None,
    normalise:        bool = True,
    norm_clip:        float | None = 5.0,
    smoothing_window_s: float = 0.2,
    valid_mask:       np.ndarray | None = None,
    state_keys:       Sequence[str] | None = None,
    target_stc:       NBStateCollection | None = None,
    rng:              np.random.Generator | None = None,
    verbose:          bool = False,
) -> tuple[NBStateCollection, TrainedEnsemble, np.ndarray]:
    """One-shot: train ensemble on training sessions, label one target session.

    Port of :file:`MTA/classifiers/bhv_nn.m` / :file:`bhv_nn_multi_session_patternnet.m`.

    Parameters
    ----------
    train_sessions:
        Iterable of ``(stc, feature_data, feature_samplerate)`` tuples
        for training.  Each session contributes its own bootstrap.
    target_features:
        ``(T_target, n_features)`` feature matrix to label.
    target_samplerate:
        Hz of *target_features*.
    states:
        Ordered state labels.
    backend:
        Classifier backend.  Default ``'patternnet'``.
    n_iter:
        Ensemble size.  Default 10.
    classifier_kwargs, bootstrap_kwargs:
        Passed through to :func:`train_classifier_ensemble`.
    normalise, norm_clip:
        See :func:`train_classifier_ensemble`.
    smoothing_window_s, valid_mask, state_keys, target_stc:
        See :func:`smooth_labels_to_state_collection`.
    rng:
        Optional :class:`numpy.random.Generator`.
    verbose:
        Print progress.

    Returns
    -------
    stc : NBStateCollection
        State collection with one state per *states* entry.
    ensemble : TrainedEnsemble
        Fitted ensemble (save it via ``ensemble.save(...)`` to reuse
        without retraining).
    label_probs : np.ndarray
        ``(T_target, n_states)`` softmax outputs averaged over the
        ensemble.
    """
    ensemble = train_classifier_ensemble(
        train_sessions, states,
        backend           = backend,
        n_iter            = n_iter,
        classifier_kwargs = classifier_kwargs,
        bootstrap_kwargs  = bootstrap_kwargs,
        normalise         = normalise,
        norm_clip         = norm_clip,
        rng               = rng,
        verbose           = verbose,
    )

    label_probs = ensemble.predict_proba(target_features)

    stc = smooth_labels_to_state_collection(
        label_probs,
        state_names        = states,
        feature_samplerate = target_samplerate,
        smoothing_window_s = smoothing_window_s,
        valid_mask         = valid_mask,
        state_keys         = state_keys,
        stc                = target_stc,
    )
    return stc, ensemble, label_probs
