"""
neurobox.analysis.classifiers.sklearn_backends
================================================
scikit-learn implementations of behavioural-state classifiers.

Provides three backends:

* :class:`SklearnMLPClassifier`  (``backend='sklearn-mlp'``)
* :class:`RandomForestClassifierWrapper`  (``backend='rf'``)
* :class:`HistGBClassifierWrapper`  (``backend='gbm'``)

All three register themselves on import.  Importing this module
pulls in :mod:`sklearn`.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.neural_network import MLPClassifier as _SkMLP

from . import _registry
from .base import Classifier, FitInfo


def _wrap_proba(probs: np.ndarray, classes: np.ndarray, n_classes: int) -> np.ndarray:
    """Re-pad sklearn ``predict_proba`` output if some classes were missing.

    sklearn drops zero-frequency classes from the output of
    ``predict_proba``.  This helper re-pads so the column index lines
    up with the integer label codes ``0..n_classes-1`` regardless of
    which classes actually appeared in the training data.
    """
    if probs.shape[1] == n_classes and np.array_equal(classes, np.arange(n_classes)):
        return probs
    out = np.zeros((probs.shape[0], n_classes), dtype=probs.dtype)
    for j, c in enumerate(classes):
        out[:, int(c)] = probs[:, j]
    return out


# ──────────────────────────────────────────────────────────────────── #
# sklearn MLP                                                           #
# ──────────────────────────────────────────────────────────────────── #

class SklearnMLPClassifier(Classifier):
    """scikit-learn :class:`~sklearn.neural_network.MLPClassifier` wrapper.

    A CPU-only fallback for users without a torch dependency.
    """

    backend = "sklearn-mlp"

    def __init__(
        self,
        hidden_layer_sizes: tuple[int, ...] = (64, 64),
        alpha:              float = 1e-4,
        learning_rate_init: float = 1e-3,
        max_iter:           int   = 200,
        early_stopping:     bool  = True,
        n_iter_no_change:   int   = 20,
        validation_fraction: float = 0.1,
        random_state:       int | None = None,
        verbose:            bool = False,
    ) -> None:
        super().__init__()
        self._kwargs = dict(
            hidden_layer_sizes  = tuple(hidden_layer_sizes),
            alpha               = float(alpha),
            learning_rate_init  = float(learning_rate_init),
            max_iter            = int(max_iter),
            early_stopping      = bool(early_stopping),
            n_iter_no_change    = int(n_iter_no_change),
            validation_fraction = float(validation_fraction),
            random_state        = random_state,
            verbose             = bool(verbose),
        )
        self._clf: _SkMLP | None = None

    def fit(self, X, y, **kwargs):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        n_classes = int(y.max() + 1) if y.size else 0
        self._clf = _SkMLP(**self._kwargs)
        self._clf.fit(X, y)
        self.n_features_in_ = X.shape[1]
        self.n_classes_     = n_classes
        self.fit_info_ = FitInfo(
            n_train=X.shape[0], n_features=X.shape[1], n_classes=n_classes,
            backend=self.backend, epochs=self._clf.n_iter_,
            train_loss=float(self._clf.loss_), val_loss=float("nan"),
            extra={"hidden_layer_sizes": list(self._kwargs["hidden_layer_sizes"])},
        )
        return self

    def predict_proba(self, X):
        if self._clf is None:
            raise RuntimeError("SklearnMLPClassifier not fitted.")
        X = np.asarray(X, dtype=np.float32)
        return _wrap_proba(self._clf.predict_proba(X),
                           self._clf.classes_, int(self.n_classes_))

    def _save_state(self, dirpath):
        with open(dirpath / "model.pkl", "wb") as f:
            pickle.dump({"clf": self._clf, "kwargs": self._kwargs}, f)

    @classmethod
    def _load_state(cls, dirpath):
        with open(dirpath / "model.pkl", "rb") as f:
            b = pickle.load(f)
        clf = cls(**b["kwargs"])
        clf._clf = b["clf"]
        return clf


# ──────────────────────────────────────────────────────────────────── #
# Random forest                                                          #
# ──────────────────────────────────────────────────────────────────── #

class RandomForestClassifierWrapper(Classifier):
    """scikit-learn :class:`~sklearn.ensemble.RandomForestClassifier` wrapper.

    Robust baseline; no input normalisation needed; gives a
    ``feature_importances_`` array via ``self.feature_importances``.
    """

    backend = "rf"

    def __init__(
        self,
        n_estimators:   int = 200,
        max_depth:      int | None = None,
        min_samples_leaf: int = 1,
        n_jobs:         int = -1,
        random_state:   int | None = None,
    ) -> None:
        super().__init__()
        self._kwargs = dict(
            n_estimators     = int(n_estimators),
            max_depth        = max_depth,
            min_samples_leaf = int(min_samples_leaf),
            n_jobs           = int(n_jobs),
            random_state     = random_state,
        )
        self._clf: RandomForestClassifier | None = None

    def fit(self, X, y, **kwargs):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        n_classes = int(y.max() + 1) if y.size else 0
        self._clf = RandomForestClassifier(**self._kwargs)
        self._clf.fit(X, y)
        self.n_features_in_ = X.shape[1]
        self.n_classes_     = n_classes
        self.fit_info_ = FitInfo(
            n_train=X.shape[0], n_features=X.shape[1], n_classes=n_classes,
            backend=self.backend, epochs=0,
            train_loss=float("nan"), val_loss=float("nan"),
            extra={"n_estimators": self._kwargs["n_estimators"],
                   "max_depth":    str(self._kwargs["max_depth"])},
        )
        return self

    def predict_proba(self, X):
        if self._clf is None:
            raise RuntimeError("RandomForestClassifierWrapper not fitted.")
        X = np.asarray(X, dtype=np.float32)
        return _wrap_proba(self._clf.predict_proba(X),
                           self._clf.classes_, int(self.n_classes_))

    @property
    def feature_importances(self) -> np.ndarray:
        if self._clf is None:
            raise RuntimeError("Not fitted.")
        return self._clf.feature_importances_

    def _save_state(self, dirpath):
        with open(dirpath / "model.pkl", "wb") as f:
            pickle.dump({"clf": self._clf, "kwargs": self._kwargs}, f)

    @classmethod
    def _load_state(cls, dirpath):
        with open(dirpath / "model.pkl", "rb") as f:
            b = pickle.load(f)
        clf = cls(**b["kwargs"])
        clf._clf = b["clf"]
        return clf


# ──────────────────────────────────────────────────────────────────── #
# Histogram-based gradient boosting                                      #
# ──────────────────────────────────────────────────────────────────── #

class HistGBClassifierWrapper(Classifier):
    """scikit-learn :class:`~sklearn.ensemble.HistGradientBoostingClassifier` wrapper.

    Strongest tabular-data baseline; trains an order of magnitude
    faster than MLPs and frequently matches or beats them on
    behavioural feature data.
    """

    backend = "gbm"

    def __init__(
        self,
        max_iter:        int   = 200,
        learning_rate:   float = 0.05,
        max_leaf_nodes:  int   = 31,
        min_samples_leaf: int  = 20,
        early_stopping:  bool  = True,
        n_iter_no_change: int  = 10,
        validation_fraction: float = 0.1,
        random_state:    int | None = None,
    ) -> None:
        super().__init__()
        self._kwargs = dict(
            max_iter            = int(max_iter),
            learning_rate       = float(learning_rate),
            max_leaf_nodes      = int(max_leaf_nodes),
            min_samples_leaf    = int(min_samples_leaf),
            early_stopping      = bool(early_stopping),
            n_iter_no_change    = int(n_iter_no_change),
            validation_fraction = float(validation_fraction),
            random_state        = random_state,
        )
        self._clf: HistGradientBoostingClassifier | None = None

    def fit(self, X, y, **kwargs):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        n_classes = int(y.max() + 1) if y.size else 0
        self._clf = HistGradientBoostingClassifier(**self._kwargs)
        self._clf.fit(X, y)
        self.n_features_in_ = X.shape[1]
        self.n_classes_     = n_classes
        self.fit_info_ = FitInfo(
            n_train=X.shape[0], n_features=X.shape[1], n_classes=n_classes,
            backend=self.backend, epochs=int(self._clf.n_iter_),
            train_loss=float("nan"), val_loss=float("nan"),
            extra={"max_iter":      self._kwargs["max_iter"],
                   "learning_rate": self._kwargs["learning_rate"]},
        )
        return self

    def predict_proba(self, X):
        if self._clf is None:
            raise RuntimeError("HistGBClassifierWrapper not fitted.")
        X = np.asarray(X, dtype=np.float32)
        return _wrap_proba(self._clf.predict_proba(X),
                           self._clf.classes_, int(self.n_classes_))

    def _save_state(self, dirpath):
        with open(dirpath / "model.pkl", "wb") as f:
            pickle.dump({"clf": self._clf, "kwargs": self._kwargs}, f)

    @classmethod
    def _load_state(cls, dirpath):
        with open(dirpath / "model.pkl", "rb") as f:
            b = pickle.load(f)
        clf = cls(**b["kwargs"])
        clf._clf = b["clf"]
        return clf


# ──────────────────────────────────────────────────────────────────── #
# Registration                                                          #
# ──────────────────────────────────────────────────────────────────── #

_registry.register("sklearn-mlp", SklearnMLPClassifier)
_registry.register("rf",          RandomForestClassifierWrapper)
_registry.register("gbm",         HistGBClassifierWrapper)
