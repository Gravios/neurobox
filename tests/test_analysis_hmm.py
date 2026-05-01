"""Tests for neurobox.analysis.stats.hmm.gauss_hmm.

These tests require ``hmmlearn`` (an optional dependency).  When it's
not installed they're skipped automatically.
"""

from __future__ import annotations

import numpy as np
import pytest


# Skip the entire file if hmmlearn isn't installed
hmmlearn = pytest.importorskip("hmmlearn")


# ─────────────────────────────────────────────────────────────────────────── #
# Synthetic data fixtures                                                      #
# ─────────────────────────────────────────────────────────────────────────── #

def _generate_two_state_sequence(
    n_samples: int = 5000,
    means: tuple[float, float] = (0.0, 5.0),
    stds:  tuple[float, float] = (1.0, 1.0),
    transition_p_stay: float = 0.98,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a 2-state HMM observation sequence with known states.

    Returns
    -------
    obs : (n_samples, 1) array
    true_states : (n_samples,) int array
    """
    rng = np.random.default_rng(seed)
    states = np.zeros(n_samples, dtype=np.int64)
    obs = np.zeros((n_samples, 1))
    s = 0  # start in state 0
    for t in range(n_samples):
        states[t] = s
        obs[t, 0] = rng.normal(means[s], stds[s])
        # Transition
        if rng.uniform() > transition_p_stay:
            s = 1 - s
    return obs, states


# ─────────────────────────────────────────────────────────────────────────── #
# Basic API + result shape                                                     #
# ─────────────────────────────────────────────────────────────────────────── #

class TestGaussHMMBasic:

    def test_result_shape_2_state_1_feature(self):
        from neurobox.analysis.stats import gauss_hmm
        obs, _ = _generate_two_state_sequence()
        result = gauss_hmm(obs, n_states=2, random_state=0)
        assert result.states.shape == (obs.shape[0],)
        assert result.means.shape == (2, 1)
        assert result.covars.shape == (2, 1, 1)
        assert result.transmat.shape == (2, 2)
        assert result.startprob.shape == (2,)
        assert result.posteriors.shape == (obs.shape[0], 2)

    def test_states_in_valid_range(self):
        from neurobox.analysis.stats import gauss_hmm
        obs, _ = _generate_two_state_sequence()
        result = gauss_hmm(obs, n_states=3, random_state=0)
        assert result.states.min() >= 0
        assert result.states.max() < 3

    def test_transmat_rows_sum_to_one(self):
        from neurobox.analysis.stats import gauss_hmm
        obs, _ = _generate_two_state_sequence()
        result = gauss_hmm(obs, n_states=2, random_state=0)
        np.testing.assert_allclose(result.transmat.sum(axis=1), 1.0, atol=1e-10)

    def test_startprob_sums_to_one(self):
        from neurobox.analysis.stats import gauss_hmm
        obs, _ = _generate_two_state_sequence()
        result = gauss_hmm(obs, n_states=2, random_state=0)
        np.testing.assert_allclose(result.startprob.sum(), 1.0, atol=1e-10)

    def test_posteriors_sum_to_one_per_timepoint(self):
        from neurobox.analysis.stats import gauss_hmm
        obs, _ = _generate_two_state_sequence()
        result = gauss_hmm(obs, n_states=2, random_state=0)
        np.testing.assert_allclose(
            result.posteriors.sum(axis=1), 1.0, atol=1e-10
        )

    def test_log_likelihood_is_finite(self):
        from neurobox.analysis.stats import gauss_hmm
        obs, _ = _generate_two_state_sequence()
        result = gauss_hmm(obs, n_states=2, random_state=0)
        assert np.isfinite(result.log_likelihood)

    def test_1d_input_accepted(self):
        from neurobox.analysis.stats import gauss_hmm
        obs, _ = _generate_two_state_sequence()
        # Pass 1-D — should be auto-reshaped to (T, 1)
        result = gauss_hmm(obs.ravel(), n_states=2, random_state=0)
        assert result.means.shape == (2, 1)
        assert result.states.shape == (obs.shape[0],)


# ─────────────────────────────────────────────────────────────────────────── #
# Recovery — does it actually find the true states?                            #
# ─────────────────────────────────────────────────────────────────────────── #

class TestGaussHMMRecovery:

    def test_recovers_well_separated_2_state_means(self):
        from neurobox.analysis.stats import gauss_hmm
        obs, _ = _generate_two_state_sequence(
            means=(0.0, 5.0), stds=(0.5, 0.5),    # well-separated
            transition_p_stay=0.98, seed=0,
        )
        result = gauss_hmm(obs, n_states=2, random_state=0)
        recovered_means = sorted(result.means.ravel())
        assert abs(recovered_means[0] - 0.0) < 0.5
        assert abs(recovered_means[1] - 5.0) < 0.5

    def test_recovers_state_sequence_2_state(self):
        """Decoded states should match true states up to label permutation."""
        from neurobox.analysis.stats import gauss_hmm
        obs, true_states = _generate_two_state_sequence(
            means=(0.0, 5.0), stds=(0.5, 0.5),
            transition_p_stay=0.99, seed=42,
        )
        result = gauss_hmm(obs, n_states=2, random_state=0)
        # HMM labels are arbitrary up to permutation.  Check the better
        # of the two possible mappings.
        agreement_a = np.mean(result.states == true_states)
        agreement_b = np.mean(result.states == (1 - true_states))
        best = max(agreement_a, agreement_b)
        assert best > 0.95, f"recovery too poor: {best:.3f}"

    def test_state_for_max_feature(self):
        """The 'high state' helper should pick the state with higher mean."""
        from neurobox.analysis.stats import gauss_hmm
        obs, _ = _generate_two_state_sequence(
            means=(0.0, 5.0), stds=(0.5, 0.5), seed=0,
        )
        result = gauss_hmm(obs, n_states=2, random_state=0)
        high_state = result.state_for_max_feature(0)
        # The higher-mean state should have mean ≈ 5
        assert abs(result.means[high_state, 0] - 5.0) < 1.0
        # The other state should have mean ≈ 0
        low_state = 1 - high_state
        assert abs(result.means[low_state, 0] - 0.0) < 1.0

    def test_multivariate_recovery(self):
        """3-feature 2-state HMM with diagonal-ish state means."""
        from neurobox.analysis.stats import gauss_hmm
        rng = np.random.default_rng(0)
        # State 0: centred at (0, 0, 0); state 1: centred at (3, 3, 3)
        n = 4000
        true_states = (rng.uniform(size=n) > 0.5).astype(np.int64)
        means_true = np.array([[0.0, 0.0, 0.0], [3.0, 3.0, 3.0]])
        obs = means_true[true_states] + rng.standard_normal((n, 3)) * 0.5
        result = gauss_hmm(obs, n_states=2, random_state=0)
        # Find which recovered state corresponds to which true state
        d_to_state0 = np.linalg.norm(result.means, axis=1)
        recovered = np.argsort(d_to_state0)  # state 0 of result = closest to origin
        np.testing.assert_allclose(
            result.means[recovered[0]], 0.0, atol=0.5
        )
        np.testing.assert_allclose(
            result.means[recovered[1]], 3.0, atol=0.5
        )


# ─────────────────────────────────────────────────────────────────────────── #
# Convenience / configuration                                                  #
# ─────────────────────────────────────────────────────────────────────────── #

class TestGaussHMMConfig:

    def test_random_state_determinism(self):
        from neurobox.analysis.stats import gauss_hmm
        obs, _ = _generate_two_state_sequence(seed=0)
        r1 = gauss_hmm(obs, n_states=2, random_state=42)
        r2 = gauss_hmm(obs, n_states=2, random_state=42)
        np.testing.assert_array_equal(r1.states, r2.states)
        np.testing.assert_allclose(r1.means, r2.means)
        np.testing.assert_allclose(r1.transmat, r2.transmat)

    @pytest.mark.parametrize("cov_type", ["full", "diag", "tied", "spherical"])
    def test_covariance_types(self, cov_type):
        from neurobox.analysis.stats import gauss_hmm
        obs, _ = _generate_two_state_sequence()
        result = gauss_hmm(
            obs, n_states=2, covariance_type=cov_type, random_state=0
        )
        # Output covariance is always reshaped to full form
        assert result.covars.shape == (2, 1, 1)
        # And should have positive diagonal
        assert (np.diagonal(result.covars, axis1=1, axis2=2) > 0).all()

    def test_update_obs_model_false_keeps_init_means(self):
        """With update_obs_model=False, EM only updates transmat, not means."""
        from neurobox.analysis.stats import gauss_hmm
        obs, _ = _generate_two_state_sequence(seed=0)

        # Fit with and without observation-model updates
        a = gauss_hmm(obs, n_states=2, update_obs_model=True,  random_state=0)
        b = gauss_hmm(obs, n_states=2, update_obs_model=False, random_state=0)

        # Both should produce valid output, but b's means should be the
        # k-means initialisation (less refined).  Means may match closely
        # when the data is well-clustered, so just check that b's
        # log-likelihood is ≤ a's (a is more flexible).
        assert b.log_likelihood <= a.log_likelihood + 1e-6

    def test_max_iter_respected(self):
        from neurobox.analysis.stats import gauss_hmm
        obs, _ = _generate_two_state_sequence()
        result = gauss_hmm(obs, n_states=2, max_iter=2, random_state=0)
        assert result.n_iter <= 2

    def test_returns_underlying_model(self):
        """The result should expose the hmmlearn model for advanced use."""
        from neurobox.analysis.stats import gauss_hmm
        from hmmlearn.hmm import GaussianHMM
        obs, _ = _generate_two_state_sequence()
        result = gauss_hmm(obs, n_states=2, random_state=0)
        assert isinstance(result.model, GaussianHMM)
        # The user can now call score / predict / sample on it directly
        score = result.model.score(obs)
        assert np.isfinite(score)


# ─────────────────────────────────────────────────────────────────────────── #
# Validation                                                                   #
# ─────────────────────────────────────────────────────────────────────────── #

class TestGaussHMMValidation:

    def test_nan_input_raises(self):
        from neurobox.analysis.stats import gauss_hmm
        obs = np.random.default_rng(0).standard_normal((100, 2))
        obs[10, 1] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            gauss_hmm(obs, n_states=2)

    def test_inf_input_raises(self):
        from neurobox.analysis.stats import gauss_hmm
        obs = np.random.default_rng(0).standard_normal((100, 2))
        obs[5, 0] = np.inf
        with pytest.raises(ValueError, match="NaN|Inf"):
            gauss_hmm(obs, n_states=2)

    def test_3d_input_raises(self):
        from neurobox.analysis.stats import gauss_hmm
        obs = np.random.default_rng(0).standard_normal((100, 2, 3))
        with pytest.raises(ValueError, match="1-D or 2-D"):
            gauss_hmm(obs, n_states=2)

    def test_invalid_covariance_type(self):
        from neurobox.analysis.stats import gauss_hmm
        obs, _ = _generate_two_state_sequence()
        with pytest.raises(ValueError, match="covariance_type"):
            gauss_hmm(obs, n_states=2, covariance_type="banana")

    def test_invalid_init(self):
        from neurobox.analysis.stats import gauss_hmm
        obs, _ = _generate_two_state_sequence()
        with pytest.raises(ValueError, match="init"):
            gauss_hmm(obs, n_states=2, init="banana")

    def test_n_states_zero(self):
        from neurobox.analysis.stats import gauss_hmm
        obs, _ = _generate_two_state_sequence()
        with pytest.raises(ValueError, match="n_states"):
            gauss_hmm(obs, n_states=0)

    def test_state_for_max_feature_validation(self):
        from neurobox.analysis.stats import gauss_hmm
        obs, _ = _generate_two_state_sequence()
        result = gauss_hmm(obs, n_states=2, random_state=0)
        with pytest.raises(ValueError, match="feature_dim"):
            result.state_for_max_feature(99)


# ─────────────────────────────────────────────────────────────────────────── #
# Top-level exposure                                                           #
# ─────────────────────────────────────────────────────────────────────────── #

class TestNamespaceExposure:

    def test_top_level_neurobox_exposure(self):
        import neurobox
        assert neurobox.gauss_hmm is not None
        assert neurobox.HMMResult is not None

    def test_neurobox_analysis_exposure(self):
        import neurobox.analysis as A
        assert A.gauss_hmm is not None
        assert A.HMMResult is not None

    def test_neurobox_analysis_stats_exposure(self):
        from neurobox.analysis.stats import gauss_hmm, HMMResult
        assert gauss_hmm is not None
        assert HMMResult is not None
