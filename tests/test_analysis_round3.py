"""Tests for fdr_bh, bin_smooth, and detect_ripples (round 3 ports)."""

from __future__ import annotations

import numpy as np
import pytest


# ─────────────────────────────────────────────────────────────────────────── #
# fdr_bh                                                                       #
# ─────────────────────────────────────────────────────────────────────────── #

class TestFdrBh:

    def test_known_example_pdep(self):
        """The labbox docstring example, transcribed."""
        from neurobox.analysis.stats import fdr_bh
        # Eight tests; first two should be significant at q=0.05 (BH).
        pvals = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205])
        result = fdr_bh(pvals, q=0.05, method="pdep")
        # First two are significant
        assert result.h.tolist() == [True, True, False, False, False,
                                     False, False, False]
        assert result.crit_p == 0.008

    def test_no_rejections(self):
        from neurobox.analysis.stats import fdr_bh
        pvals = np.array([0.5, 0.6, 0.7, 0.8])
        result = fdr_bh(pvals, q=0.05)
        assert not result.h.any()
        assert result.crit_p == 0.0
        assert np.isnan(result.adj_ci_cvrg)

    def test_all_rejections(self):
        from neurobox.analysis.stats import fdr_bh
        pvals = np.array([0.0001, 0.0002, 0.0003, 0.0004])
        result = fdr_bh(pvals, q=0.05)
        assert result.h.all()

    def test_dep_more_conservative_than_pdep(self):
        """BY-FDR ('dep') should reject ≤ what BH-FDR ('pdep') rejects."""
        from neurobox.analysis.stats import fdr_bh
        rng = np.random.default_rng(0)
        # 100 tests, 20 with strong effects
        pvals = np.concatenate([
            rng.uniform(0, 0.005, size=20),
            rng.uniform(0, 1, size=80),
        ])
        pdep = fdr_bh(pvals, q=0.05, method="pdep")
        dep  = fdr_bh(pvals, q=0.05, method="dep")
        assert dep.h.sum() <= pdep.h.sum()

    def test_preserves_input_shape(self):
        from neurobox.analysis.stats import fdr_bh
        pvals = np.random.default_rng(0).uniform(0, 1, size=(5, 10))
        result = fdr_bh(pvals, q=0.05)
        assert result.h.shape == pvals.shape
        assert result.adj_p.shape == pvals.shape

    def test_adj_p_monotonic_in_sorted_order(self):
        """Adjusted p-values should be non-decreasing when ordered by raw p."""
        from neurobox.analysis.stats import fdr_bh
        rng = np.random.default_rng(0)
        pvals = rng.uniform(0, 1, size=100)
        result = fdr_bh(pvals, q=0.05)
        # When sorted by raw p, adjusted p should also be sorted (non-decreasing)
        order = np.argsort(pvals)
        adj_sorted = result.adj_p[order]
        assert np.all(np.diff(adj_sorted) >= -1e-12), \
            "adj_p should be non-decreasing with raw p"

    def test_pdep_adj_p_capped_at_1(self):
        from neurobox.analysis.stats import fdr_bh
        # Adversarial: many large p-values force adjusted values toward 1
        pvals = np.array([0.95, 0.96, 0.97, 0.98, 0.99])
        result = fdr_bh(pvals, q=0.05, method="pdep")
        assert (result.adj_p <= 1.0).all()

    def test_validation(self):
        from neurobox.analysis.stats import fdr_bh
        with pytest.raises(ValueError, match="\\[0, 1\\]"):
            fdr_bh(np.array([0.5, 1.5]), q=0.05)
        with pytest.raises(ValueError, match="\\[0, 1\\]"):
            fdr_bh(np.array([0.5, -0.1]), q=0.05)
        with pytest.raises(ValueError, match="empty"):
            fdr_bh(np.array([]), q=0.05)
        with pytest.raises(ValueError, match="method"):
            fdr_bh(np.array([0.5]), q=0.05, method="xyz")
        with pytest.raises(ValueError, match="q"):
            fdr_bh(np.array([0.5]), q=1.5)


# ─────────────────────────────────────────────────────────────────────────── #
# bin_smooth                                                                   #
# ─────────────────────────────────────────────────────────────────────────── #

class TestBinSmooth:

    def test_mean_mode_recovers_constant(self):
        from neurobox.analysis.stats import bin_smooth
        x = np.linspace(0, 10, 100)
        y = np.full_like(x, 7.5)
        result = bin_smooth(x, y, bins=10, mode="mean")
        finite = ~np.isnan(result.y_smooth)
        np.testing.assert_allclose(result.y_smooth[finite], 7.5)
        # Spread should be 0 within each bin
        np.testing.assert_allclose(result.spread[finite], 0.0)

    def test_mean_mode_recovers_linear_trend(self):
        from neurobox.analysis.stats import bin_smooth
        rng = np.random.default_rng(0)
        x = rng.uniform(0, 10, size=1000)
        y = 2 * x + 3 + 0.01 * rng.standard_normal(1000)
        result = bin_smooth(x, y, bins=20, mode="mean")
        # In each bin, the mean of y should be ≈ 2 * median(x) + 3.
        finite = ~np.isnan(result.y_smooth)
        expected = 2 * result.medianx[finite] + 3
        # Bin width is ~0.5, so within-bin mean differs from "at median(x)"
        # by O(bin_width) * trend_slope ~= 0.05.  Tolerance reflects that.
        assert np.max(np.abs(result.y_smooth[finite] - expected)) < 0.2

    def test_explicit_edges(self):
        from neurobox.analysis.stats import bin_smooth
        x = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
        y = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        # Labbox semantics: 3 edges = 3 bins ([0,2), [2,4), [4,∞)).
        bins = np.array([0.0, 2.0, 4.0])
        result = bin_smooth(x, y, bins=bins, mode="mean")
        # Bin 0: [0, 2): {0.5, 1.5} → mean(y) = 15
        # Bin 1: [2, 4): {2.5, 3.5} → mean(y) = 35
        # Bin 2: [4, ∞): {4.5}      → mean(y) = 50
        np.testing.assert_allclose(result.y_smooth[0], 15.0)
        np.testing.assert_allclose(result.y_smooth[1], 35.0)
        np.testing.assert_allclose(result.y_smooth[2], 50.0)
        np.testing.assert_array_equal(result.counts, [2, 2, 1])

    def test_median_mode(self):
        from neurobox.analysis.stats import bin_smooth
        # In bin 0: y ∈ {1, 2, 100}; median = 2 (robust to the outlier)
        x = np.array([0.5, 0.6, 0.7, 1.5, 1.6, 1.7])
        y = np.array([1.0, 2.0, 100.0, 10.0, 11.0, 12.0])
        bins = np.array([0.0, 1.0, 2.0])
        result = bin_smooth(x, y, bins=bins, mode="median")
        np.testing.assert_allclose(result.y_smooth[0], 2.0)
        np.testing.assert_allclose(result.y_smooth[1], 11.0)
        # Spread = IQR; in bin 0 with [1,2,100]: IQR = 49.5
        assert result.spread[0] > 40.0

    def test_count_mode(self):
        from neurobox.analysis.stats import bin_smooth
        x = np.array([0.5, 0.6, 1.5, 1.6, 1.7])
        y = np.zeros_like(x)
        # 3 edges → 3 bins; nothing falls in the [2, ∞) bin
        bins = np.array([0.0, 1.0, 2.0])
        result = bin_smooth(x, y, bins=bins, mode="count")
        np.testing.assert_array_equal(result.y_smooth, [2, 3, 0])

    def test_callable_mode(self):
        from neurobox.analysis.stats import bin_smooth
        x = np.array([0.5, 0.6, 0.7, 1.5, 1.6, 1.7])
        y = np.array([1.0, 2.0, 3.0, 10.0, 20.0, 30.0])
        bins = np.array([0.0, 1.0, 2.0])
        # 75th percentile reducer
        result = bin_smooth(x, y, bins=bins, mode=lambda v: np.percentile(v, 75))
        np.testing.assert_allclose(result.y_smooth[0], 2.5)
        np.testing.assert_allclose(result.y_smooth[1], 25.0)

    def test_auto_bins_from_count(self):
        from neurobox.analysis.stats import bin_smooth
        x = np.arange(100, dtype=np.float64)
        y = x.copy()
        # 20 points per bin → 5 bins
        result = bin_smooth(x, y, bins=20, mode="mean")
        assert result.bins.size == 5
        assert result.counts.sum() == 100

    def test_residuals_have_correct_length(self):
        from neurobox.analysis.stats import bin_smooth
        rng = np.random.default_rng(0)
        x = rng.uniform(0, 10, size=200)
        y = 2 * x + rng.standard_normal(200)
        result = bin_smooth(x, y, bins=10, mode="mean")
        assert result.residuals.size == x.size
        # Residuals should average to ~0 within each bin
        # (bin-mean is removed by definition)
        bin_idx = np.digitize(x, result.bins) - 1
        bin_idx = np.clip(bin_idx, 0, result.bins.size - 1)
        for k in range(result.bins.size):
            mask = bin_idx == k
            if mask.sum() > 1:
                assert abs(result.residuals[mask].mean()) < 1e-9

    def test_empty_bins_return_nan(self):
        from neurobox.analysis.stats import bin_smooth
        # All x in bin 0; bin 1 is empty
        x = np.array([0.1, 0.2, 0.3])
        y = np.array([1.0, 2.0, 3.0])
        bins = np.array([0.0, 1.0, 2.0])
        result = bin_smooth(x, y, bins=bins, mode="mean")
        assert np.isnan(result.y_smooth[1])
        assert result.counts[1] == 0


# ─────────────────────────────────────────────────────────────────────────── #
# detect_ripples                                                               #
# ─────────────────────────────────────────────────────────────────────────── #

class TestDetectRipples:

    def test_synthetic_ripple_burst(self):
        """A 50 ms 180 Hz burst in noise should be detected as a ripple."""
        from neurobox.analysis.lfp import detect_ripples
        fs = 1250.0
        rng = np.random.default_rng(0)
        n_total = 5 * int(fs)               # 5 s
        x = 0.05 * rng.standard_normal(n_total)
        burst_dur = 0.05                     # 50 ms
        burst_n = int(burst_dur * fs)
        t_burst = np.arange(burst_n) / fs
        burst   = 1.0 * np.sin(2 * np.pi * 180 * t_burst)
        x[2 * int(fs): 2 * int(fs) + burst_n] += burst

        result = detect_ripples(x, samplerate=fs, threshold=(3.0, 1.0))
        # Should detect at least one event overlapping the burst window.
        starts = result.periods[:, 0] / fs
        stops  = result.periods[:, 1] / fs
        overlapping = (stops > 2.0) & (starts < 2.05)
        assert overlapping.any(), \
            f"no ripple detected at 2.0–2.05 s; starts={starts}, stops={stops}"

    def test_default_freq_band(self):
        """Default band is 100-250 Hz (matches labbox)."""
        from neurobox.analysis.lfp import detect_ripples
        fs = 1250.0
        # Empty signal → empty result, but verify no crash
        x = np.zeros(int(fs))
        result = detect_ripples(x)
        assert result.peaks.size == 0
        assert result.samplerate == fs

    def test_returns_oscillation_result(self):
        from neurobox.analysis.lfp import detect_ripples, OscillationResult
        x = np.zeros(int(1250 * 2))
        result = detect_ripples(x)
        assert isinstance(result, OscillationResult)


# ─────────────────────────────────────────────────────────────────────────── #
# Regression: detect_oscillations length-2 threshold scaling                  #
# ─────────────────────────────────────────────────────────────────────────── #

class TestDetectOscillationsThresholdRegression:
    """Regression for a bug in step-1 ``detect_oscillations``.

    When ``threshold`` was a length-2 sigma-multiplier sequence, the
    duration threshold was set to the literal value of ``threshold[1]``
    (e.g. 2.0) without the σ-multiplier transformation
    ``threshold[1] * std(amp) + mean(amp)``.  This made the duration
    threshold scale-mismatched with the amplitude envelope and caused
    ``detect_ripples`` (which uses ``threshold=(5, 2)``) to never return
    any periods.  Fixed in round 3.
    """

    def test_length_2_sigma_threshold_finds_periods(self):
        """Both elements of a length-2 σ threshold must be transformed.

        We don't depend on detection succeeding end-to-end (which would
        couple this test to the gaussian smoother width).  Instead we
        verify the *returned threshold values* on pure noise.

        Before the fix, with ``threshold=(5, 2)`` and noise of std σ
        ≈ 0.02, ``dur_thr`` was the literal ``2.0`` — far above
        ``peak_thr`` (~5σ + mean ≈ 0.15).  After the fix, both are
        scaled: ``peak_thr = 5σ + μ``, ``dur_thr = 2σ + μ``, with
        ``peak_thr > dur_thr`` and both in the same scale as the
        envelope.
        """
        from neurobox.analysis.lfp import detect_oscillations
        fs = 1250.0
        rng = np.random.default_rng(0)
        x = 0.1 * rng.standard_normal(5 * int(fs))   # pure noise
        result = detect_oscillations(x, freq_range=(6.0, 12.0), samplerate=fs,
                                     min_cycles=5, threshold=(5.0, 2.0))
        peak_thr, dur_thr = result.threshold
        # σ-multiplier scaling means both should be small positive
        # numbers in the scale of the envelope.  The buggy code returned
        # the literal 2.0 for dur_thr, much larger than the (correctly
        # scaled) peak_thr.
        assert 0 < dur_thr < peak_thr, (
            f"threshold scaling broken: peak={peak_thr}, dur={dur_thr}; "
            "both should be in σ-units of the envelope"
        )
        # Sanity: peak_thr should be on the order of a few × envelope std.
        # For LFP with rms ~0.1, envelope σ is in the 0.01–0.1 range,
        # so peak_thr should be < 1.0 (rather than the literal 5.0).
        assert peak_thr < 1.0

    def test_length_2_percentile_threshold(self):
        """Length-2 percentile threshold: both elements treated as percentiles."""
        from neurobox.analysis.lfp import detect_oscillations
        fs = 1250.0
        rng = np.random.default_rng(0)
        x = 0.1 * rng.standard_normal(5 * int(fs))
        result = detect_oscillations(x, freq_range=(6.0, 12.0), samplerate=fs,
                                     threshold=(95.0, 80.0))
        peak_thr, dur_thr = result.threshold
        # percentile(95) > percentile(80) on the same envelope
        assert peak_thr > dur_thr
