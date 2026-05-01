"""Tests for neurobox.analysis.stats.circular."""

from __future__ import annotations

import numpy as np
import pytest


# ─────────────────────────────────────────────────────────────────────────── #
# circ_mean / circ_r                                                           #
# ─────────────────────────────────────────────────────────────────────────── #

class TestCircMeanCircR:

    def test_uniform_zero_resultant(self):
        from neurobox.analysis.stats import circ_r
        # A long uniform sample should have r ~ 0.
        rng = np.random.default_rng(0)
        theta = rng.uniform(-np.pi, np.pi, size=20_000)
        r = circ_r(theta)
        assert r < 0.05, f"expected r near 0, got {r}"

    def test_concentrated_high_resultant(self):
        from neurobox.analysis.stats import circ_mean, circ_r
        rng = np.random.default_rng(0)
        theta = rng.vonmises(mu=0.7, kappa=20.0, size=2000)
        mu = circ_mean(theta)
        r  = circ_r(theta)
        assert abs(mu - 0.7) < 0.05
        assert r > 0.9

    def test_known_distribution(self):
        from neurobox.analysis.stats import circ_mean, circ_r
        # 4 angles at multiples of pi/2 → r = 0
        theta = np.array([0, np.pi / 2, np.pi, -np.pi / 2])
        assert abs(circ_r(theta)) < 1e-12

    def test_axis_argument(self):
        from neurobox.analysis.stats import circ_mean
        rng = np.random.default_rng(0)
        theta = rng.vonmises(mu=0.0, kappa=10.0, size=(50, 3))
        mu = circ_mean(theta, axis=0)
        assert mu.shape == (3,)


# ─────────────────────────────────────────────────────────────────────────── #
# bessel_ratio_inverse                                                         #
# ─────────────────────────────────────────────────────────────────────────── #

class TestBesselRatioInverse:

    def test_round_trip(self):
        """bessel_ratio_inverse(I1(k)/I0(k)) should recover k (approximately)."""
        from scipy.special import i0, i1
        from neurobox.analysis.stats import bessel_ratio_inverse

        for k_true in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
            r = i1(k_true) / i0(k_true)
            k_est = bessel_ratio_inverse(np.atleast_1d(r))[0]
            # Fisher's piecewise approximation has ~1% accuracy
            assert abs(k_est - k_true) / k_true < 0.05, \
                f"k_true={k_true}: got {k_est}"

    def test_branch_continuity(self):
        """Approximation should be continuous at branch points 0.53 and 0.85."""
        from neurobox.analysis.stats import bessel_ratio_inverse
        for branch in [0.53, 0.85]:
            below = bessel_ratio_inverse(np.atleast_1d(branch - 1e-6))[0]
            above = bessel_ratio_inverse(np.atleast_1d(branch + 1e-6))[0]
            # Tolerate up to 5% jump (these are known non-smooth)
            assert abs(below - above) / max(below, above) < 0.05


# ─────────────────────────────────────────────────────────────────────────── #
# von_mises_fit / pdf / rvs                                                    #
# ─────────────────────────────────────────────────────────────────────────── #

class TestVonMises:

    def test_fit_recovers_parameters(self):
        from neurobox.analysis.stats import von_mises_fit
        rng = np.random.default_rng(42)
        for mu_true, kappa_true in [(0.0, 5.0), (1.5, 2.0), (-2.0, 10.0)]:
            theta = rng.vonmises(mu=mu_true, kappa=kappa_true, size=5000)
            mu_est, kappa_est = von_mises_fit(theta)
            assert abs(mu_est - mu_true) < 0.1
            # κ recovery is harder; accept 15% tolerance for moderate κ
            assert abs(kappa_est - kappa_true) / kappa_true < 0.15

    def test_pdf_integrates_to_1(self):
        from neurobox.analysis.stats import von_mises_pdf
        from scipy.integrate import quad
        for mu, kappa in [(0.0, 0.5), (1.0, 5.0), (-1.0, 2.0)]:
            integral, _ = quad(lambda t: von_mises_pdf(t, mu, kappa),
                               -np.pi, np.pi)
            assert abs(integral - 1.0) < 1e-6

    def test_rvs_seed_reproducibility(self):
        from neurobox.analysis.stats import von_mises_rvs
        a = von_mises_rvs(0.0, 5.0, size=100, random_state=123)
        b = von_mises_rvs(0.0, 5.0, size=100, random_state=123)
        np.testing.assert_array_equal(a, b)


# ─────────────────────────────────────────────────────────────────────────── #
# PPC                                                                          #
# ─────────────────────────────────────────────────────────────────────────── #

class TestPPC:

    def test_unbiased_under_uniformity(self):
        """Mean PPC over many uniform samples should be ≈ 0."""
        from neurobox.analysis.stats import ppc
        rng = np.random.default_rng(0)
        n_per_sample = 50
        n_samples = 1000
        vals = np.array([
            ppc(rng.uniform(-np.pi, np.pi, size=n_per_sample))
            for _ in range(n_samples)
        ])
        assert abs(vals.mean()) < 0.005, f"PPC mean under H0: {vals.mean()}"

    def test_high_concentration_near_one(self):
        from neurobox.analysis.stats import ppc
        rng = np.random.default_rng(0)
        theta = rng.vonmises(mu=0.0, kappa=50.0, size=200)
        val = ppc(theta)
        assert val > 0.95

    def test_matches_naive_ppc(self):
        """Closed-form PPC should equal the O(n²) double-sum definition."""
        from neurobox.analysis.stats import ppc
        rng = np.random.default_rng(0)
        theta = rng.uniform(-np.pi, np.pi, size=100)

        # Naive O(n²) reference
        n = theta.size
        total = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                total += np.cos(theta[i] - theta[j])
        naive = 2.0 * total / (n * (n - 1))

        fast = ppc(theta)
        assert abs(naive - fast) < 1e-10

    def test_too_short_returns_nan(self):
        from neurobox.analysis.stats import ppc
        assert np.isnan(ppc(np.array([])))
        assert np.isnan(ppc(np.array([0.5])))


# ─────────────────────────────────────────────────────────────────────────── #
# rayleigh_test — single-vector form                                           #
# ─────────────────────────────────────────────────────────────────────────── #

class TestRayleighTestSingle:

    def test_uniform_high_p(self):
        from neurobox.analysis.stats import rayleigh_test
        rng = np.random.default_rng(0)
        theta = rng.uniform(-np.pi, np.pi, size=300)
        result = rayleigh_test(theta)
        # Under H0 (uniformity), p should be uniform on [0, 1].  Single
        # draws can sit anywhere; require only that we don't reject hard.
        assert result.p > 0.05

    def test_concentrated_low_p(self):
        from neurobox.analysis.stats import rayleigh_test
        rng = np.random.default_rng(0)
        theta = rng.vonmises(mu=0.0, kappa=2.0, size=200)
        result = rayleigh_test(theta)
        assert result.p < 1e-3
        assert abs(result.th0) < 0.2  # near 0
        assert result.r > 0.4
        assert result.kappa > 1.0
        assert result.n == 200

    def test_empty_returns_nan(self):
        from neurobox.analysis.stats import rayleigh_test
        result = rayleigh_test(np.array([]))
        assert np.isnan(result.p)
        assert np.isnan(result.th0)
        assert np.isnan(result.r)
        assert result.n == 0

    def test_unbiased_uses_ppc_for_small_n(self):
        """For small n, unbiased path uses PPC instead of |z|."""
        from neurobox.analysis.stats import rayleigh_test, ppc
        rng = np.random.default_rng(0)
        theta = rng.vonmises(mu=0.0, kappa=2.0, size=100)
        biased   = rayleigh_test(theta, unbiased=False)
        unbiased = rayleigh_test(theta, unbiased=True)
        # Unbiased r should equal sqrt(max(PPC, 0))
        expected_r = np.sqrt(max(ppc(theta), 0.0))
        assert abs(unbiased.r - expected_r) < 1e-12
        # Biased r should be the resultant length (different value)
        assert abs(biased.r - unbiased.r) > 1e-6


# ─────────────────────────────────────────────────────────────────────────── #
# rayleigh_test — per-cluster form                                             #
# ─────────────────────────────────────────────────────────────────────────── #

class TestRayleighTestPerCluster:

    def test_per_cluster_shape(self):
        from neurobox.analysis.stats import rayleigh_test
        rng = np.random.default_rng(0)
        theta = rng.vonmises(mu=0.0, kappa=2.0, size=500)
        clu   = rng.integers(1, 6, size=500)  # clusters 1..5
        result = rayleigh_test(theta, clusters=clu, max_clusters=5)
        assert result.p.shape == (5,)
        assert result.th0.shape == (5,)
        assert result.r.shape == (5,)
        assert result.kappa.shape == (5,)
        assert result.n.shape == (5,)
        # All non-empty in this case
        assert np.all(np.isfinite(result.p))

    def test_empty_clusters_are_nan(self):
        from neurobox.analysis.stats import rayleigh_test
        # Only clusters 1 and 3 fire; cluster 2 is empty.
        theta = np.array([0.1, 0.2, 0.3, 0.5, 0.6])
        clu   = np.array([1, 1, 1, 3, 3])
        result = rayleigh_test(theta, clusters=clu, max_clusters=3)
        assert result.n[1] == 0
        assert np.isnan(result.p[1])
        assert np.isnan(result.r[1])
        assert np.isnan(result.th0[1])
        # Non-empty clusters have valid stats
        assert np.isfinite(result.p[0])
        assert np.isfinite(result.p[2])

    def test_per_cluster_matches_loop(self):
        """Vectorised per-cluster output must equal per-cluster looping."""
        from neurobox.analysis.stats import rayleigh_test
        rng = np.random.default_rng(0)
        n_total = 1000
        theta = rng.vonmises(mu=0.0, kappa=1.0, size=n_total)
        clu   = rng.integers(1, 11, size=n_total)
        # Vectorised
        vec = rayleigh_test(theta, clusters=clu, max_clusters=10)
        # Loop reference
        for k in range(1, 11):
            ref = rayleigh_test(theta[clu == k])
            assert abs(vec.p[k - 1] - ref.p) < 1e-10, f"cluster {k}: p mismatch"
            assert abs(vec.r[k - 1] - ref.r) < 1e-10, f"cluster {k}: r mismatch"
            assert abs(vec.th0[k - 1] - ref.th0) < 1e-10, f"cluster {k}: th0"

    def test_dropped_non_positive_labels(self):
        """clusters[i] ≤ 0 should be excluded (matches labbox)."""
        from neurobox.analysis.stats import rayleigh_test
        theta = np.array([0.0, 0.1, 0.2, 0.3])
        clu   = np.array([1, 0, 1, -2])  # only the two with clu==1 count
        result = rayleigh_test(theta, clusters=clu, max_clusters=1)
        assert result.n[0] == 2
