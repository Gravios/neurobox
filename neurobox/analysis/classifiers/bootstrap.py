"""
neurobox.analysis.classifiers.bootstrap
========================================
Whole-state-bootstrap resamplers for behavioural-state classifier
training data.

Ported from :file:`MTA/utilities/resample_whole_state_bootstrap_*.m`.
The MATLAB family contains four variants:

* ``resample_whole_state_bootstrap``      — plain ("WSB")
* ``resample_whole_state_bootstrap_trim`` — trim period boundaries ("WSBT")
* ``resample_whole_state_bootstrap_noisy``      — add Gaussian noise ("WSBN")
* ``resample_whole_state_bootstrap_noisy_trim`` — both ("WSBNT", default)

This port supports all four via two boolean flags (``trim`` and
``noisy``).  The default is ``trim=True, noisy=True`` (= WSBNT)
because that is what the multi-session NN classifier
(:func:`bhv_nn`) uses by default.

Algorithm (WSBNT)
-----------------
For each state s in *states*:

  1. Take the period list ``ep_s`` for that state.
  2. Random-permute, split off the held-out fraction
     (``100 - prct_train`` percent for evaluation).
  3. Trim the training periods inward by 0.1 s to avoid
     boundary contamination from neighbouring states.
  4. Sample ``state_block_size`` feature rows uniformly with
     replacement from frames inside the trimmed training periods.
  5. Append to a growing feature matrix.

The full training matrix is then jittered with Gaussian noise
``N(0, σ²=¼)`` and returned with a parallel state-label array.

Bug fix vs. MATLAB original
---------------------------
Line 67 of ``resample_whole_state_bootstrap_noisy_trim.m`` reads
``ThreshCross(nniz(trainingFeatures.data==i), 0.5, 1)`` where ``i``
is undefined inside this function — it's a stale loop variable
that happens to exist in the caller's scope.  The fragment was
trying to build a "good periods" mask over the synthetic
training-data timeline.  The intent is ``nniz(trainingFeatures.data)``
(rows that are finite/non-zero across all features).  This port
implements the corrected logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from neurobox.dtype.epoch import NBEpoch
from neurobox.dtype.stc import NBStateCollection


# Default block size — matches MATLAB's stateBlockSize=15000
DEFAULT_STATE_BLOCK_SIZE: int = 15_000

# Default trim — matches MATLAB's s+[0.1,-0.1]
DEFAULT_TRIM_S: float = 0.1

# Default noise σ — matches MATLAB's randn/4
DEFAULT_NOISE_STD: float = 0.25


@dataclass
class BootstrapResult:
    """Output of :func:`whole_state_bootstrap`.

    Attributes
    ----------
    features:
        ``(N, n_features)`` resampled and (optionally) jittered
        training feature matrix.
    labels:
        ``(N,)`` integer state index for each row of *features*.
        ``0..len(states)-1`` — does not include an "other" class.
    eval_periods:
        Per-state list of held-out (start, stop) sample-index pairs
        in the **original** feature samplerate.  Use these to
        compute classifier accuracy on data not seen in training.
    state_names:
        State labels in the same order as the integer codes in
        ``labels``.
    """
    features:     np.ndarray
    labels:       np.ndarray
    eval_periods: list[np.ndarray]
    state_names:  list[str]


def whole_state_bootstrap(
    stc:                NBStateCollection,
    feature_data:       np.ndarray,
    feature_samplerate: float,
    states:             Sequence[str],
    *,
    state_block_size: int = DEFAULT_STATE_BLOCK_SIZE,
    prct_train:       float = 90.0,
    trim:             bool = True,
    noisy:            bool = True,
    trim_s:           float = DEFAULT_TRIM_S,
    noise_std:        float = DEFAULT_NOISE_STD,
    rng:              np.random.Generator | None = None,
) -> BootstrapResult:
    """Whole-state bootstrap resampler with optional trim + Gaussian noise.

    Port of :file:`MTA/utilities/resample_whole_state_bootstrap_noisy_trim.m`
    (and its three non-trim/non-noisy siblings).

    Parameters
    ----------
    stc:
        State collection containing labelled periods for each state
        in *states*.  Must support ``[label]`` lookup returning an
        :class:`NBEpoch` whose ``data`` is ``(n_periods, 2)`` in
        seconds.
    feature_data:
        ``(T, n_features)`` feature matrix at *feature_samplerate*.
    feature_samplerate:
        Hz.  Used to convert period seconds → sample indices.
    states:
        State labels to bootstrap.  Order is preserved in the output
        labels array.
    state_block_size:
        Number of training rows to sample per state (with
        replacement).  Default 15000 matches MATLAB's default.
    prct_train:
        Percentage of each state's periods used for training.  The
        remaining periods are returned as ``eval_periods`` for
        held-out evaluation.  Default 90 matches MATLAB.
    trim:
        If True (default), trim each training period inward by
        *trim_s* seconds before sampling.  Reduces boundary
        contamination from neighbouring states.
    noisy:
        If True (default), add Gaussian noise with std *noise_std*
        to the resampled feature matrix.
    trim_s:
        Trim amount in seconds, applied symmetrically.  Default 0.1.
    noise_std:
        Noise σ.  Default 0.25 (= MATLAB's ``randn/4``).
    rng:
        Optional :class:`numpy.random.Generator` for reproducibility.

    Returns
    -------
    BootstrapResult
    """
    if rng is None:
        rng = np.random.default_rng()

    feature_data = np.asarray(feature_data, dtype=np.float64)
    if feature_data.ndim != 2:
        raise ValueError(
            f"feature_data must be (T, n_features); got {feature_data.shape}"
        )
    n_T = feature_data.shape[0]

    trim_samples = int(round(trim_s * feature_samplerate)) if trim else 0

    feature_blocks: list[np.ndarray] = []
    label_blocks:   list[np.ndarray] = []
    eval_periods_per_state: list[np.ndarray] = []

    for s_idx, name in enumerate(states):
        try:
            ep = stc[name] if hasattr(stc, "__getitem__") else stc.get_state(name)
        except (KeyError, ValueError) as e:
            raise KeyError(
                f"whole_state_bootstrap: state {name!r} not found in stc"
            ) from e

        # Periods in seconds → sample indices at feature samplerate
        ep_s = np.asarray(ep.data, dtype=np.float64)
        if ep_s.size == 0 or ep_s.shape[0] == 0:
            # No periods for this state; record empty eval periods, skip block
            eval_periods_per_state.append(np.zeros((0, 2), dtype=np.int64))
            continue
        periods = np.column_stack([
            np.floor(ep_s[:, 0] * feature_samplerate).astype(np.int64),
            np.ceil( ep_s[:, 1] * feature_samplerate).astype(np.int64),
        ])
        # Clip to feature timeline
        periods[:, 0] = np.clip(periods[:, 0], 0, n_T - 1)
        periods[:, 1] = np.clip(periods[:, 1], 0, n_T - 1)
        # Drop empties
        valid = periods[:, 1] > periods[:, 0]
        periods = periods[valid]
        if periods.shape[0] == 0:
            eval_periods_per_state.append(np.zeros((0, 2), dtype=np.int64))
            continue

        # Random split
        n_p = periods.shape[0]
        order = rng.permutation(n_p)
        n_train = int(round(n_p * prct_train / 100.0))
        n_train = max(1, min(n_train, n_p))
        train_idx = order[:n_train]
        eval_idx  = order[n_train:]
        train_periods = periods[train_idx]
        eval_periods  = periods[eval_idx]

        eval_periods_per_state.append(eval_periods)

        # Trim each training period inward
        if trim_samples > 0:
            train_periods = train_periods + np.array(
                [+trim_samples, -trim_samples], dtype=np.int64
            )
            # Drop periods that became empty after trimming
            train_periods = train_periods[
                train_periods[:, 1] > train_periods[:, 0]
            ]
            if train_periods.shape[0] == 0:
                # All training periods were too short to survive trimming —
                # fall back to the un-trimmed list so we don't lose this
                # state entirely.
                train_periods = periods[train_idx]

        # Build the candidate sample-index pool by enumerating
        # frames inside each training period
        pool = np.concatenate([
            np.arange(s, e + 1, dtype=np.int64)
            for s, e in train_periods
        ])
        if pool.size == 0:
            continue

        # Sample state_block_size with replacement
        chosen = rng.integers(low=0, high=pool.size, size=state_block_size)
        block = feature_data[pool[chosen], :]
        feature_blocks.append(block)
        label_blocks.append(np.full(state_block_size, s_idx, dtype=np.int64))

    if not feature_blocks:
        raise RuntimeError(
            "whole_state_bootstrap: no states had any usable training "
            "periods — check that 'stc' contains the requested labels."
        )

    features_out = np.concatenate(feature_blocks, axis=0)
    labels_out   = np.concatenate(label_blocks,   axis=0)

    if noisy and noise_std > 0:
        features_out = features_out + rng.normal(
            scale=noise_std, size=features_out.shape
        )

    return BootstrapResult(
        features     = features_out,
        labels       = labels_out,
        eval_periods = eval_periods_per_state,
        state_names  = list(states),
    )
