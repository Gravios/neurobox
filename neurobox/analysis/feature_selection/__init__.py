"""
neurobox.analysis.feature_selection
=====================================
Mutual-information feature selection and the full hierarchical
classifier-training pipeline.

Two layers:

1. **Single-purpose primitives** for ad-hoc feature ranking:

   * :func:`pairwise_mutual_information_ranking` — per-feature MI
     against each binary state label.  Matches the inner kernel of
     :file:`req20160310_5_genfigs.m`.

2. **Full pipeline** that replaces the
   :file:`feature_selection_via_mutual_information.m` orchestrator
   (round 19):

   * :func:`select_features_hmi` — hierarchical Δ-MI feature
     selection.  Port of :file:`select_features_hmi.m`.
   * :func:`mta_tsne` — t-SNE wrapper with state-aware
     sub-sampling.  Port of :file:`mta_tsne.m`.
   * :func:`run_feature_selection_pipeline` — end-to-end:
     hierarchical MI selection → incremental classifier training →
     Δ-accuracy re-ranking.  Replaces ``req20160310_{1,3,4,5,6}_*.m``
     in a single in-memory function.
   * :func:`augment_features_quadratic` — feature-space augmentation
     (cols + cols² + circ-shifted cross-products).  Port of the
     augmentation in :file:`req20160310_1_preproc.m`.

What's still not ported and why
-------------------------------
* :file:`req20160310_8_genOptfigs.m` (537 LoC) — paper-specific
  figure generation.  The numerical content is in this module's
  result dataclasses; rendering them is application code.
* :file:`req20160310_9_bhvLabeling.m` — labels a NEW session using
  pre-trained classifiers; once you have classifiers trained by
  this pipeline, calling them on new data is straightforward
  scikit-learn / round-12 backend code.
* The LRZ ``MatSubmitLRZ`` batch-job submission — replaced by
  ``joblib.Parallel`` if you need parallelism for the inner NN
  training loop.
"""

from __future__ import annotations

from .core import (
    pairwise_mutual_information_ranking,
    MIFeatureRanking,
)
from .hmi import (
    select_features_hmi,
    HierarchicalMIResult,
)
from .tsne import (
    mta_tsne,
    TSNEResult,
)
from .pipeline import (
    run_feature_selection_pipeline,
    augment_features_quadratic,
    FeatureSelectionPipelineResult,
    PerStateAccumulation,
)


__all__ = [
    "pairwise_mutual_information_ranking",
    "MIFeatureRanking",
    "select_features_hmi",
    "HierarchicalMIResult",
    "mta_tsne",
    "TSNEResult",
    "run_feature_selection_pipeline",
    "augment_features_quadratic",
    "FeatureSelectionPipelineResult",
    "PerStateAccumulation",
]
