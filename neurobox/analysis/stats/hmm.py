"""
neurobox.analysis.stats.hmm
============================
Gaussian Hidden Markov Models for state segmentation.

Port of :file:`labbox/Stats/gausshmm.m` (Anton Sirota / Evgeny Resnik),
which is itself a thin wrapper around the third-party MATLAB ``hmmbox``
toolbox by Iain Murray and the netlab ``kmeans`` initialiser by
Christopher Bishop / Ian Nabney.

This port delegates to :mod:`hmmlearn` (the de-facto-standard Python
HMM library, with a tested C backend for forward-backward and Viterbi).
The wrapper here is responsible for matching the labbox API surface:
k-means initialisation, Baum-Welch fit, Viterbi decode, and the
post-processing step that's ubiquitous in MTA code — finding "the HMM
state corresponding to the max of feature dimension *d*".

Optional dependency
-------------------
:mod:`hmmlearn` is **not** in the base neurobox install.  Install with::

    pip install hmmlearn
    # or
    pip install 'neurobox[hmm]'

(``neurobox[hmm]`` is an optional-dependencies group declared in
``pyproject.toml``.)  Importing :func:`gauss_hmm` without it raises
``ImportError`` with a clear install hint.

Why not reimplement Baum-Welch in Cython?
------------------------------------------
The audit (``CYTHON_CANDIDATES.md``) flagged HMM forward-backward as a
canonical Cython use case.  However, :mod:`hmmlearn` already provides a
production-quality C implementation (cython-wrapped), is widely used in
neuroscience, and supports diagonal/full covariances, custom
initialisations, and inspection of all internal parameters.
Reimplementing it would duplicate maintained code with no scientific or
performance benefit.

API notes
---------
* Inputs are plain ``(T, n_features)`` ndarrays — features are
  whatever you've already extracted from your :class:`NBData`.
* The labbox returns a 1-indexed state vector; we return **0-indexed**
  (Python convention).  Use ``+ 1`` if you need labbox-compatible
  output for downstream MATLAB scripts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ─────────────────────────────────────────────────────────────────────────── #
# Result container                                                             #
# ─────────────────────────────────────────────────────────────────────────── #

@dataclass
class HMMResult:
    """Output of :func:`gauss_hmm`.

    Attributes
    ----------
    states:
        ``(T,)`` int array — Viterbi-decoded state sequence, **0-indexed**.
        The labbox 1-indexed convention is recovered with ``states + 1``.
    means:
        ``(n_states, n_features)`` array of state-conditional Gaussian
        means.
    covars:
        ``(n_states, n_features, n_features)`` array of state-conditional
        Gaussian covariance matrices.
    transmat:
        ``(n_states, n_states)`` row-stochastic transition matrix —
        ``transmat[i, j]`` is ``P(state_{t+1}=j | state_t=i)``.
    startprob:
        ``(n_states,)`` initial-state probability distribution.
    log_likelihood:
        Log-likelihood of the data under the fitted model.
    posteriors:
        ``(T, n_states)`` smoothed state posteriors (forward-backward
        gamma).  ``posteriors[t, k]`` is ``P(state_t = k | data, model)``.
    n_iter:
        Number of EM iterations actually performed before convergence
        (or until ``max_iter`` was hit).
    converged:
        Whether the EM loop reached the convergence threshold.
    model:
        The underlying :class:`hmmlearn.hmm.GaussianHMM` instance, for
        users who need to call additional methods (``score``,
        ``predict_proba``, ``sample``) on it.
    """

    states:          np.ndarray
    means:           np.ndarray
    covars:          np.ndarray
    transmat:        np.ndarray
    startprob:       np.ndarray
    log_likelihood:  float
    posteriors:      np.ndarray
    n_iter:          int
    converged:       bool
    model:           "object"   # hmmlearn.hmm.GaussianHMM, but kept opaque for type-checkers without hmmlearn

    def state_for_max_feature(self, feature_dim: int = 0) -> int:
        """Return the state index whose mean is maximal along ``feature_dim``.

        This is the post-processing step ubiquitous in MTA code:

            [~, rearstate] = max(cat(1, hmm.state.Mu));

        For example, when fitting a 2-state HMM to running speed, the
        "running" state is identified as the one with the higher mean
        velocity::

            result = gauss_hmm(speed[:, None], n_states=2)
            running_state = result.state_for_max_feature(0)
            running_mask = result.states == running_state

        Parameters
        ----------
        feature_dim:
            Index into the feature axis to compare.

        Returns
        -------
        int
            The 0-indexed state with the maximum mean on the chosen
            feature.
        """
        if not (0 <= feature_dim < self.means.shape[1]):
            raise ValueError(
                f"feature_dim={feature_dim} out of range [0, {self.means.shape[1]})"
            )
        return int(np.argmax(self.means[:, feature_dim]))


# ─────────────────────────────────────────────────────────────────────────── #
# Public API                                                                   #
# ─────────────────────────────────────────────────────────────────────────── #

def gauss_hmm(
    features:       np.ndarray,
    n_states:       int = 2,
    *,
    update_obs_model: bool = True,
    max_iter:       int = 30,
    tol:            float = 1e-2,
    covariance_type: str = "full",
    init:           str = "kmeans",
    random_state:   Optional[int | np.random.Generator] = None,
    verbose:        bool = False,
) -> HMMResult:
    """Fit a Gaussian HMM and return the Viterbi-decoded state sequence.

    Port of :file:`labbox/Stats/gausshmm.m` — delegates to
    :class:`hmmlearn.hmm.GaussianHMM` for the actual fit and decode.

    The labbox version does:
      1. k-means cluster the features into ``n_states`` clusters
      2. Initialise an HMM with cluster means/covariances and a uniform
         transition matrix
      3. Train with Baum-Welch (forward-backward + EM) for 30 iterations
      4. Decode with Viterbi
      5. Return ``(state_sequence, hmm_struct, decode_struct)``

    This wrapper does the same thing using maintained Python code.

    Parameters
    ----------
    features:
        ``(T, n_features)`` array of observations.  Must not contain
        NaN — drop or impute beforehand.
    n_states:
        Number of hidden states.  Default 2 (matches labbox default;
        suitable for binary state classification — e.g. theta vs no
        theta, running vs not).
    update_obs_model:
        If ``True`` (default), update the Gaussian observation
        parameters during EM.  If ``False``, keep the k-means
        initialisation fixed and only update the transition matrix —
        useful when you have prior knowledge of the state means.
    max_iter:
        Maximum EM iterations.  Default 30 (labbox default).
    tol:
        Convergence threshold on the log-likelihood improvement per
        iteration.  Default 1e-2.
    covariance_type:
        One of ``'full'`` (default, matches labbox ``hmminit(..., 'full')``),
        ``'diag'``, ``'tied'``, ``'spherical'``.
    init:
        Initialisation strategy.  ``'kmeans'`` (default) — k-means cluster
        the features and use cluster centres as initial state means
        (matches labbox).  ``'random'`` — random initialisation
        (faster but may converge to local optima).
    random_state:
        Seed or :class:`numpy.random.Generator` for the k-means
        initialisation and the EM tie-breaking.  None → non-deterministic.
    verbose:
        If ``True``, print per-iteration log-likelihood.

    Returns
    -------
    :class:`HMMResult`

    Examples
    --------
    Segment running from quiescence with a 2-state HMM on speed::

        import numpy as np
        from neurobox.analysis.stats import gauss_hmm
        # speed: (T,) instantaneous speed in cm/s
        result = gauss_hmm(speed[:, None], n_states=2, random_state=0)
        running_state = result.state_for_max_feature(0)
        running_mask = result.states == running_state
        print(f"Run/quiet means: {result.means.ravel()}")

    Multi-feature behavioural segmentation (the MTA ``bhv_hmm`` use case)::

        # vel: (T, 3) — body velocity, head velocity, head angular velocity
        result = gauss_hmm(vel, n_states=8, max_iter=50, random_state=0)
        # State assignments
        states_per_frame = result.states            # (T,) ints in [0, 8)

    Raises
    ------
    ImportError
        If ``hmmlearn`` is not installed.  Install with
        ``pip install 'neurobox[hmm]'`` or ``pip install hmmlearn``.
    ValueError
        If ``features`` contains NaN, has wrong shape, or
        ``covariance_type`` / ``init`` is unrecognised.
    """
    # ── Optional-dependency check ────────────────────────────────────────── #
    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError as exc:
        raise ImportError(
            "gauss_hmm requires the 'hmmlearn' package, which is not installed. "
            "Install with:\n"
            "    pip install 'neurobox[hmm]'\n"
            "or:\n"
            "    pip install hmmlearn"
        ) from exc

    # ── Validation ────────────────────────────────────────────────────────── #
    features = np.asarray(features, dtype=np.float64)
    if features.ndim == 1:
        features = features[:, np.newaxis]
    if features.ndim != 2:
        raise ValueError(
            f"features must be 1-D or 2-D (T,) / (T, n_features); "
            f"got shape {features.shape}"
        )
    if not np.isfinite(features).all():
        raise ValueError(
            "features contains NaN or Inf — drop or impute before calling gauss_hmm"
        )
    if n_states < 1:
        raise ValueError(f"n_states must be ≥ 1, got {n_states}")
    if covariance_type not in ("full", "diag", "tied", "spherical"):
        raise ValueError(
            f"covariance_type must be one of 'full', 'diag', 'tied', "
            f"'spherical'; got {covariance_type!r}"
        )
    if init not in ("kmeans", "random"):
        raise ValueError(f"init must be 'kmeans' or 'random'; got {init!r}")

    # ── Resolve RNG state for hmmlearn ──────────────────────────────────── #
    if isinstance(random_state, np.random.Generator):
        # hmmlearn expects an int or None or numpy.random.RandomState
        rs = int(random_state.integers(0, 2**31 - 1))
    else:
        rs = random_state

    # ── Build "params" / "init_params" flags from update_obs_model ─────── #
    # hmmlearn lets us choose which parameters to update via two strings:
    #   init_params  : which params to initialise from the data
    #   params       : which params to update during EM
    # Letters: 's' = startprob, 't' = transmat, 'm' = means, 'c' = covars
    #
    # When update_obs_model=False, we still want to *initialise* m/c from
    # k-means, but skip them in the EM update — match labbox's
    # `obsupdate=zeros(...)` semantics.
    init_params = "stmc" if init == "kmeans" else "stmc"  # always init all
    params = "stmc" if update_obs_model else "st"

    # ── Construct and fit ───────────────────────────────────────────────── #
    model = GaussianHMM(
        n_components    = int(n_states),
        covariance_type = covariance_type,
        n_iter          = int(max_iter),
        tol             = float(tol),
        init_params     = init_params,
        params          = params,
        random_state    = rs,
        verbose         = bool(verbose),
    )

    if init == "random":
        # hmmlearn uses k-means by default; force random by setting the
        # initial means to random samples from the data range.
        rng = np.random.default_rng(rs if rs is not None else None)
        model.means_ = rng.uniform(
            features.min(axis=0), features.max(axis=0),
            size=(n_states, features.shape[1]),
        )
        # Skip means initialisation since we just set it
        model.init_params = init_params.replace("m", "")

    model.fit(features)

    # ── Decode + posteriors ─────────────────────────────────────────────── #
    states     = model.predict(features).astype(np.int64)            # Viterbi
    posteriors = model.predict_proba(features)                        # forward-backward
    log_lik    = float(model.score(features))

    # ── Reshape covariance to (n_states, n_features, n_features) regardless ─ #
    # of covariance_type.  Modern hmmlearn (≥ 0.3) returns ``covars_`` already
    # in this canonical shape for all four covariance types — so we just copy.
    # We keep the explicit allocation to ensure dtype and writeability.
    n_features = features.shape[1]
    covars_arr = np.asarray(model.covars_)
    if covars_arr.shape == (n_states, n_features, n_features):
        covars_full = covars_arr.astype(np.float64, copy=True)
    elif covariance_type == "diag" and covars_arr.shape == (n_states, n_features):
        # Older hmmlearn versions
        covars_full = np.zeros((n_states, n_features, n_features), dtype=np.float64)
        for k in range(n_states):
            np.fill_diagonal(covars_full[k], covars_arr[k])
    elif covariance_type == "tied" and covars_arr.shape == (n_features, n_features):
        covars_full = np.broadcast_to(
            covars_arr[None, :, :], (n_states, n_features, n_features)
        ).astype(np.float64, copy=True)
    elif covariance_type == "spherical" and covars_arr.shape == (n_states,):
        covars_full = np.zeros((n_states, n_features, n_features), dtype=np.float64)
        for k in range(n_states):
            np.fill_diagonal(covars_full[k], float(covars_arr[k]))
    else:
        raise RuntimeError(
            f"Unexpected hmmlearn covars_ shape {covars_arr.shape} "
            f"for covariance_type={covariance_type!r}"
        )

    return HMMResult(
        states          = states,
        means           = np.asarray(model.means_, dtype=np.float64),
        covars          = covars_full,
        transmat        = np.asarray(model.transmat_, dtype=np.float64),
        startprob       = np.asarray(model.startprob_, dtype=np.float64),
        log_likelihood  = log_lik,
        posteriors      = np.asarray(posteriors, dtype=np.float64),
        n_iter          = int(getattr(model, "monitor_", None).iter
                              if hasattr(model, "monitor_") else max_iter),
        converged       = bool(model.monitor_.converged
                               if hasattr(model, "monitor_") else False),
        model           = model,
    )
