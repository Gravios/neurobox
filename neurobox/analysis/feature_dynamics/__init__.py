"""
neurobox.analysis.feature_dynamics
====================================
Time-lagged statistical relationships between feature time-series.

Two complementary measures, both computed at user-specified time
shifts and event-anchored:

* :func:`time_lagged_mutual_information` — port of
  :file:`MTA/analysis/compute_time_lagged_mutual_information.m`.
  Histogram-based MI between every pair of feature columns at each
  lag, restricted to event-anchored time-points.
* :func:`time_lagged_cross_correlation` — port of
  :file:`MTA/analysis/compute_cross_correlation.m`.  Pearson
  correlation of segment-anchored windows.

These are typically used to map the "topology" of a behavioural
feature set: which features lead which, on what time scale, during
each behavioural state.  The MATLAB originals were used in
``MjgER2016`` to figure out which subset of `fet_bref` columns to
feed to the behaviour classifier.

Anti-scope
----------
The MATLAB orchestration layer
(:file:`compute_time_lagged_mutual_information.m` lines 31-115)
does five things:

1. Load Trials from a session-list YAML.
2. Compute :func:`fet_bref` per Trial.
3. Map each to a reference session via
   :func:`map_to_reference_session`.
4. Apply pre-computed per-feature normalisation params.
5. Slice to a sub-set of feature columns.

The Python port keeps these as separate, testable steps:

* Loading sessions / fet_bref → caller's responsibility, using
  :mod:`neurobox.analysis.kinematics.body_referenced` and
  :mod:`neurobox.analysis.classifiers`.
* This module receives **already-prepared feature arrays** plus an
  index mask, and just does the inner numerical work.
"""

from __future__ import annotations

from .core import (
    time_lagged_mutual_information,
    time_lagged_cross_correlation,
    TimeLaggedResult,
)


__all__ = [
    "time_lagged_mutual_information",
    "time_lagged_cross_correlation",
    "TimeLaggedResult",
]
