"""
neurobox.analysis.classifiers
==============================
Behavioural-state classifiers ported from
:file:`MTA/classifiers/`.

What's here
-----------
* :func:`whole_state_bootstrap` — multi-session feature resampler with
  optional period trim + Gaussian noise.  Port of
  :file:`MTA/utilities/resample_whole_state_bootstrap_*.m` (with
  ``stateBlockSize`` bug fix).
* :class:`Classifier` — strategy interface for backends.
* :func:`make_classifier` — construct a classifier by backend name.
* :class:`TrainedEnsemble` — multi-iter ensemble of fitted classifiers
  with shared normalisation pipeline.
* :func:`train_classifier_ensemble` — train one ensemble across
  sessions.
* :func:`smooth_labels_to_state_collection` — argmax + median-smooth
  + threshold-cross softmax outputs into an :class:`NBStateCollection`.
* :func:`label_states` — end-to-end equivalent of MATLAB's
  ``bhv_nn_multi_session_patternnet``.

Backends
--------
PyTorch (requires ``torch``):
* ``patternnet`` — single hidden layer + tanh + softmax.  Faithful
  reproduction of MATLAB's ``patternnet``.  **Default.**
* ``mlp``       — multi-layer perceptron with ReLU + dropout.
* ``cnn``       — 1-D CNN with temporal context window.
* ``lstm``      — bidirectional LSTM for sequence labelling.

scikit-learn (requires ``scikit-learn``):
* ``sklearn-mlp`` — sklearn's MLPClassifier (no torch needed).
* ``rf``          — RandomForestClassifier.
* ``gbm``         — HistGradientBoostingClassifier.

When to pick which backend
--------------------------
* Reproducing an existing MATLAB model → ``patternnet``.
* Fresh training, want strongest tabular default → ``gbm``.
* Temporal context matters (transient turns vs. sustained pauses) →
  ``cnn``.
* Long uninterrupted state bouts (rears, grooming) → ``lstm``.
* Want feature importances → ``rf`` or ``gbm``.
* No torch installed → ``rf``, ``gbm``, or ``sklearn-mlp``.
"""

from .base       import Classifier, FitInfo
from .bootstrap  import whole_state_bootstrap, BootstrapResult
from .label      import (
    make_classifier,
    train_classifier_ensemble,
    predict_with_ensemble,
    smooth_labels_to_state_collection,
    label_states,
    TrainedEnsemble,
    FeatureNormalisation,
    fit_normalisation,
)
from .stc_utils  import (
    mat_to_stc,
    confusion_matrix,
    compare_stcs, LabelComparisonStats,
    swap_state_vector_ids,
    reassign_short_periods,
    reassign_state_by_duration,
    reduce_stc_to_loc,
    mutual_information_states_features,
)
from .session_alignment import (
    BehaviouralManifoldStats,
    behavioural_manifold_stats,
    map_to_reference_session,
)

__all__ = [
    "Classifier",
    "FitInfo",
    "whole_state_bootstrap",
    "BootstrapResult",
    "make_classifier",
    "train_classifier_ensemble",
    "predict_with_ensemble",
    "smooth_labels_to_state_collection",
    "label_states",
    "TrainedEnsemble",
    "FeatureNormalisation",
    "fit_normalisation",
    # stc utilities
    "mat_to_stc",
    "confusion_matrix",
    "compare_stcs", "LabelComparisonStats",
    "swap_state_vector_ids",
    "reassign_short_periods",
    "reassign_state_by_duration",
    "reduce_stc_to_loc",
    "mutual_information_states_features",
    # session alignment
    "BehaviouralManifoldStats",
    "behavioural_manifold_stats",
    "map_to_reference_session",
]
