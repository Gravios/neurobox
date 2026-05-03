"""
neurobox.analysis.classifiers.base
====================================
Strategy interface for behavioural-state classifiers.

The MTA pipeline (``bhv_nn``, ``bhv_nn_multi_session_patternnet``,
``bhv_qda``, ``bhv_lda``, etc.) uses MATLAB toolbox-specific model
objects (``patternnet``, ``fitcdiscr``, ``ClassificationSVM``, ...).
This port factors the per-backend model into a thin strategy class
:class:`Classifier` so the surrounding bookkeeping
(multi-session feature concatenation, WSBNT bootstrapping, ensemble
averaging, decision smoothing, state-collection writeout) lives in
one place — :func:`neurobox.analysis.classifiers.label.label_states` —
and the only thing each backend has to provide is a fit / predict /
save / load implementation.

Backends in this port
---------------------
* :class:`PatternNetMLP`  (``backend='patternnet'``)
    The PyTorch faithful reproduction of MATLAB's ``patternnet``:
    a single hidden layer with sigmoid activation, softmax output,
    cross-entropy loss, scaled-conjugate-gradient–like training
    via Adam.  Default backend.
* :class:`MLPClassifierTorch`  (``backend='mlp'``)
    PyTorch multi-layer perceptron with arbitrary depth, ReLU
    activations, and dropout.  Training via Adam + early stopping.
* :class:`Conv1DClassifier`  (``backend='cnn'``)
    PyTorch 1-D convolutional network operating on a context
    window of ±N samples around each prediction.  Useful when
    temporal context distinguishes states (transient turns vs.
    sustained pauses, etc.).
* :class:`BiLSTMClassifier`  (``backend='lstm'``)
    PyTorch bidirectional LSTM for sequence-labelling.  Captures
    state-duration structure implicitly.
* :class:`SklearnMLPClassifier`  (``backend='sklearn-mlp'``)
    scikit-learn's :class:`~sklearn.neural_network.MLPClassifier` —
    a CPU-only fallback for users who don't want a torch dependency.
* :class:`RandomForestClassifierWrapper`  (``backend='rf'``)
    scikit-learn's :class:`~sklearn.ensemble.RandomForestClassifier`.
    Robust baseline; no input normalisation needed; gives feature
    importances.
* :class:`HistGBClassifierWrapper`  (``backend='gbm'``)
    scikit-learn's :class:`~sklearn.ensemble.HistGradientBoostingClassifier`.
    Strongest tabular baseline; trains an order of magnitude faster
    than MLPs and frequently matches or beats them on behavioural
    feature data.

When to use which
-----------------
* Reproducing an existing MATLAB model → ``patternnet``.
* Fresh training, want highest-accuracy default → start with ``gbm``,
  upgrade to ``cnn`` if temporal context helps.
* Have very long uninterrupted state bouts (rears, grooming) →
  ``lstm`` may reduce post-hoc smoothing artefacts.
* Want feature-importance diagnostics → ``rf`` or ``gbm``.
* No torch installed → ``rf``, ``gbm``, or ``sklearn-mlp``.
"""

from __future__ import annotations

import json
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class FitInfo:
    """Per-iteration training diagnostics.

    Attributes
    ----------
    n_train, n_features, n_classes:
        Shape of the training matrix.
    backend:
        Backend identifier string.
    epochs:
        Number of epochs / iterations actually used (early-stopping
        aware).
    train_loss, val_loss:
        Final per-iteration losses if the backend tracks them; ``nan``
        when not applicable (e.g. RF/GBM).
    extra:
        Any additional backend-specific scalars (depth, n_estimators,
        etc.).
    """
    n_train:    int
    n_features: int
    n_classes:  int
    backend:    str
    epochs:     int   = 0
    train_loss: float = float("nan")
    val_loss:   float = float("nan")
    extra:      dict  = field(default_factory=dict)


class Classifier(ABC):
    """Strategy interface for behavioural-state classifiers.

    Concrete subclasses must implement :meth:`fit`,
    :meth:`predict_proba`, :meth:`_save_state` and
    :meth:`_load_state`.  The surrounding pipeline calls
    :meth:`save` / :meth:`load` (defined here) which wrap state
    writeout with shared metadata bookkeeping.

    Subclasses **must** also define a class-level :attr:`backend`
    string used as a registry key.
    """

    backend: str = ""

    def __init__(self) -> None:
        # Filled in by fit():
        self.n_features_in_: int | None  = None
        self.n_classes_:     int | None  = None
        self.fit_info_:      FitInfo | None = None

    # ── Required overrides ─────────────────────────────────────────── #

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> "Classifier":
        """Train on ``(n_samples, n_features)`` X with integer labels y.

        Implementations must set ``self.n_features_in_``,
        ``self.n_classes_`` and ``self.fit_info_`` before returning.
        """
        ...

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return ``(n_samples, n_classes)`` softmax / probability outputs."""
        ...

    @abstractmethod
    def _save_state(self, dirpath: Path) -> None:
        """Write backend-specific state to *dirpath*."""
        ...

    @classmethod
    @abstractmethod
    def _load_state(cls, dirpath: Path) -> "Classifier":
        """Read backend-specific state from *dirpath* and return a fitted classifier."""
        ...

    # ── Convenience ────────────────────────────────────────────────── #

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Argmax labels; convenience over :meth:`predict_proba`."""
        return np.argmax(self.predict_proba(X), axis=1)

    # ── I/O wrappers (shared across backends) ──────────────────────── #

    def save(self, dirpath: Path | str) -> None:
        """Save the fitted model to a directory.

        Layout::

            <dirpath>/
                meta.json
                state.<ext>      (or files written by _save_state)

        The ``meta.json`` carries enough info to dispatch back to
        the right subclass at load time.
        """
        if self.fit_info_ is None:
            raise RuntimeError("Classifier has not been fit yet.")
        dirpath = Path(dirpath)
        dirpath.mkdir(parents=True, exist_ok=True)
        meta = {
            "backend":          self.backend,
            "n_features_in_":   self.n_features_in_,
            "n_classes_":       self.n_classes_,
            "fit_info_": {
                "n_train":    self.fit_info_.n_train,
                "n_features": self.fit_info_.n_features,
                "n_classes":  self.fit_info_.n_classes,
                "backend":    self.fit_info_.backend,
                "epochs":     self.fit_info_.epochs,
                "train_loss": self.fit_info_.train_loss,
                "val_loss":   self.fit_info_.val_loss,
                "extra":      self.fit_info_.extra,
            },
        }
        (dirpath / "meta.json").write_text(json.dumps(meta, indent=2))
        self._save_state(dirpath)

    @classmethod
    def load(cls, dirpath: Path | str) -> "Classifier":
        """Load a model from a directory created by :meth:`save`.

        Dispatches to the correct subclass based on the ``backend``
        field in ``meta.json``.
        """
        dirpath = Path(dirpath)
        meta_path = dirpath / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"No meta.json in {dirpath}")
        meta = json.loads(meta_path.read_text())
        backend = meta["backend"]
        # Lazy import to avoid pulling torch/sklearn at module load
        from . import _registry
        target_cls = _registry.get_classifier_class(backend)
        clf = target_cls._load_state(dirpath)
        clf.n_features_in_ = meta["n_features_in_"]
        clf.n_classes_     = meta["n_classes_"]
        fi = meta["fit_info_"]
        clf.fit_info_ = FitInfo(
            n_train    = fi["n_train"],
            n_features = fi["n_features"],
            n_classes  = fi["n_classes"],
            backend    = fi["backend"],
            epochs     = fi.get("epochs", 0),
            train_loss = fi.get("train_loss", float("nan")),
            val_loss   = fi.get("val_loss",   float("nan")),
            extra      = fi.get("extra", {}),
        )
        return clf
