"""Tests for neurobox.analysis.lfp filtering and oscillation primitives."""

from __future__ import annotations

import numpy as np
import pytest


# ─────────────────────────────────────────────────────────────────────────── #
# butter_filter (port of ButFilter)                                            #
# ─────────────────────────────────────────────────────────────────────────── #

class TestButterFilter:

    def test_lowpass_attenuates_high_freq(self):
        from neurobox.analysis.lfp import butter_filter
        fs = 1000.0
        t = np.arange(2 * int(fs)) / fs
        # 5 Hz signal + 200 Hz noise
        x = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 200 * t)
        y = butter_filter(x, cutoff=20.0, samplerate=fs, order=4, btype="lowpass")
        # The 5 Hz component should still be there, the 200 Hz component should be < 1% of original
        # Compare RMS
        rms_in  = np.sqrt(np.mean(x ** 2))
        rms_out = np.sqrt(np.mean(y ** 2))
        # Output RMS should be close to that of the 5 Hz alone (~1/sqrt(2)).
        assert 0.6 < rms_out < 0.8
        assert rms_out < rms_in

    def test_bandpass_passes_in_band_signal(self):
        from neurobox.analysis.lfp import butter_filter
        fs = 1000.0
        t = np.arange(2 * int(fs)) / fs
        x = np.sin(2 * np.pi * 10 * t)  # 10 Hz tone
        y = butter_filter(x, cutoff=[6, 14], samplerate=fs, order=2, btype="bandpass")
        # In-band, should pass through with very little attenuation.
        rms_x = np.sqrt(np.mean(x ** 2))
        rms_y = np.sqrt(np.mean(y ** 2))
        assert rms_y / rms_x > 0.85, f"in-band attenuation too high: {rms_y/rms_x}"

    def test_zero_phase_no_delay(self):
        """filtfilt must be zero-phase: peaks of input and output align."""
        from neurobox.analysis.lfp import butter_filter
        fs = 1000.0
        t = np.arange(2 * int(fs)) / fs
        x = np.sin(2 * np.pi * 5 * t)
        y = butter_filter(x, cutoff=20.0, samplerate=fs, order=4, btype="lowpass")
        # Cross-correlation peak should be at lag 0.
        from scipy.signal import correlate
        c = correlate(y, x, mode="same")
        peak = np.argmax(np.abs(c))
        center = len(c) // 2
        assert abs(peak - center) <= 1, f"phase shift detected: peak at {peak}, expected {center}"

    def test_multichannel_input(self):
        from neurobox.analysis.lfp import butter_filter
        fs = 1000.0
        t = np.arange(int(fs)) / fs
        x = np.column_stack([np.sin(2 * np.pi * 5 * t),
                             np.sin(2 * np.pi * 50 * t),
                             np.sin(2 * np.pi * 200 * t)])
        y = butter_filter(x, cutoff=20.0, samplerate=fs, btype="lowpass")
        assert y.shape == x.shape
        # Channel 0 (5 Hz) survives, channel 2 (200 Hz) is killed.
        rms_per_ch = np.sqrt(np.mean(y ** 2, axis=0))
        assert rms_per_ch[0] > 0.6
        assert rms_per_ch[2] < 0.05

    def test_btype_aliases(self):
        from neurobox.analysis.lfp import butter_filter
        fs = 1000.0
        x = np.random.RandomState(0).randn(int(fs))
        y_low_alias = butter_filter(x, 20.0, fs, btype="low")
        y_low_full  = butter_filter(x, 20.0, fs, btype="lowpass")
        np.testing.assert_array_equal(y_low_alias, y_low_full)


# ─────────────────────────────────────────────────────────────────────────── #
# filter0 (port of Filter0)                                                    #
# ─────────────────────────────────────────────────────────────────────────── #

class TestFilter0:

    def test_odd_length_required_when_shift(self):
        from neurobox.analysis.lfp import filter0
        b = np.array([1.0, 2.0, 3.0, 4.0])  # length 4: even
        with pytest.raises(ValueError, match="odd"):
            filter0(b, np.zeros(100), shift=True)

    def test_no_shift_path_matches_lfilter(self):
        from neurobox.analysis.lfp import filter0
        from scipy.signal import lfilter
        b = np.array([0.5, 0.5])  # 2-tap moving average
        x = np.arange(20.0)
        y = filter0(b, x, shift=False)
        # filter0 with shift=False does lfilter on the (extended) signal,
        # then trims to original length.  The first n_t samples should
        # equal lfilter(b, 1, x_extended)[:n_t] == lfilter(b,1,x).
        expected = lfilter(b, 1.0, x)
        np.testing.assert_allclose(y, expected)

    def test_shift_zero_phase_on_known_signal(self):
        from neurobox.analysis.lfp import filter0
        # 5-tap symmetric FIR (linear phase, group delay = 2 samples).
        b = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
        x = np.zeros(40)
        x[20] = 1.0  # impulse
        y = filter0(b, x, shift=True)
        # After shift correction, the impulse response should be centred
        # near sample 20 (not 22 as a causal lfilter would put it).
        peak = np.argmax(y)
        assert abs(peak - 20) <= 1, f"shift correction failed: peak at {peak}"


# ─────────────────────────────────────────────────────────────────────────── #
# fir_filter (port of FirFilter)                                               #
# ─────────────────────────────────────────────────────────────────────────── #

class TestFirFilter:

    def test_lowpass_returns_coefficients(self):
        from neurobox.analysis.lfp import fir_filter
        fs = 1000.0
        x = np.random.RandomState(0).randn(int(fs))
        y, coeffs = fir_filter(x, cutoff=50.0, samplerate=fs, order=20,
                               btype="lowpass", design="fir1")
        assert y.shape == x.shape
        # numtaps = order + 1, odd
        assert coeffs.size == 21
        assert coeffs.size % 2 == 1

    def test_adaptive_order(self):
        from neurobox.analysis.lfp import fir_filter
        fs = 1000.0
        x = np.zeros(2 * int(fs))
        y, coeffs = fir_filter(x, cutoff=10.0, samplerate=fs, order=0,
                               btype="lowpass")
        # order=0 → adaptive with min_n=15 floor, then forced even.
        # numtaps = order + 1 must be odd.
        assert coeffs.size >= 16
        assert coeffs.size % 2 == 1

    def test_firls_design(self):
        from neurobox.analysis.lfp import fir_filter
        fs = 1000.0
        x = np.random.RandomState(0).randn(int(fs))
        y, coeffs = fir_filter(x, cutoff=[20, 40], samplerate=fs, order=20,
                               btype="bandpass", design="firls")
        assert y.shape == x.shape
        # numtaps = order + 1, odd
        assert coeffs.size == 21
        assert coeffs.size % 2 == 1


# ─────────────────────────────────────────────────────────────────────────── #
# within_ranges (port of WithinRanges)                                         #
# ─────────────────────────────────────────────────────────────────────────── #

class TestWithinRanges:

    def test_matrix_mode_basic(self):
        from neurobox.analysis.lfp import within_ranges
        x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        ranges = np.array([[2, 4], [7, 9]])
        out = within_ranges(x, ranges, mode="matrix")
        assert out.shape == (10, 1)
        # x = 2,3,4 in range[0]; x = 7,8,9 in range[1]; both labels = 1.
        expected = np.array([0, 1, 1, 1, 0, 0, 1, 1, 1, 0]).astype(bool)
        np.testing.assert_array_equal(out[:, 0], expected)

    def test_flat_mode(self):
        from neurobox.analysis.lfp import within_ranges
        x = np.arange(11)
        ranges = np.array([[2, 4], [7, 9]])
        out = within_ranges(x, ranges, mode="flat")
        # Inclusive at both ends: 2..4 and 7..9
        expected = np.array([0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0]).astype(bool)
        np.testing.assert_array_equal(out, expected)

    def test_vector_mode_with_labels(self):
        from neurobox.analysis.lfp import within_ranges
        x = np.arange(11)
        ranges = np.array([[2, 4], [7, 9]])
        labels = np.array([1, 2])
        out = within_ranges(x, ranges, range_label=labels, mode="vector")
        expected = np.array([0, 0, 1, 1, 1, 0, 0, 2, 2, 2, 0])
        np.testing.assert_array_equal(out, expected)

    def test_empty_ranges(self):
        from neurobox.analysis.lfp import within_ranges
        x = np.arange(5)
        ranges = np.empty((0, 2))
        out = within_ranges(x, ranges, mode="flat")
        assert out.shape == (5,)
        assert not out.any()

    def test_overlap_raises_in_vector_mode(self):
        from neurobox.analysis.lfp import within_ranges
        x = np.arange(11)
        ranges = np.array([[2, 5], [4, 8]])
        labels = np.array([1, 2])
        with pytest.raises(ValueError, match="more than one"):
            within_ranges(x, ranges, range_label=labels, mode="vector")


# ─────────────────────────────────────────────────────────────────────────── #
# thresh_cross (port of ThreshCross)                                           #
# ─────────────────────────────────────────────────────────────────────────── #

class TestThreshCross:

    def test_simple_pulses(self):
        from neurobox.analysis.lfp import thresh_cross
        x = np.zeros(100)
        x[10:20] = 1.0
        x[40:55] = 1.0
        x[70:73] = 1.0  # short pulse
        periods = thresh_cross(x, threshold=0.5, min_interval=0)
        # 3 periods, durations 10, 15, 3.
        assert periods.shape[0] == 3
        durs = periods[:, 1] - periods[:, 0]
        np.testing.assert_array_equal(durs, [10, 15, 3])

    def test_min_interval_drops_short(self):
        from neurobox.analysis.lfp import thresh_cross
        x = np.zeros(100)
        x[10:20] = 1.0
        x[40:55] = 1.0
        x[70:73] = 1.0
        periods = thresh_cross(x, threshold=0.5, min_interval=5)
        # min_interval > 5 drops the 3-sample pulse but keeps 10 and 15.
        assert periods.shape[0] == 2
        durs = periods[:, 1] - periods[:, 0]
        np.testing.assert_array_equal(durs, [10, 15])

    def test_starts_above_threshold(self):
        from neurobox.analysis.lfp import thresh_cross
        x = np.zeros(50)
        x[:10] = 1.0
        periods = thresh_cross(x, 0.5)
        assert periods.shape[0] == 1
        assert periods[0, 0] == 0

    def test_ends_above_threshold(self):
        from neurobox.analysis.lfp import thresh_cross
        x = np.zeros(50)
        x[40:] = 1.0
        periods = thresh_cross(x, 0.5)
        assert periods.shape[0] == 1
        assert periods[0, 1] == len(x) - 1

    def test_no_crossings(self):
        from neurobox.analysis.lfp import thresh_cross
        x = np.zeros(50)
        periods = thresh_cross(x, 0.5)
        assert periods.shape == (0, 2)


# ─────────────────────────────────────────────────────────────────────────── #
# local_minima (port of LocalMinima)                                           #
# ─────────────────────────────────────────────────────────────────────────── #

class TestLocalMinima:

    def test_simple_minima(self):
        from neurobox.analysis.lfp import local_minima
        # Three valleys at 5, 15, 25
        x = np.ones(30)
        x[5] = -1.0
        x[15] = -2.0
        x[25] = -3.0
        mins, vals = local_minima(x, not_closer_than=1)
        np.testing.assert_array_equal(mins, [5, 15, 25])
        np.testing.assert_allclose(vals, [-1.0, -2.0, -3.0])

    def test_separation_constraint(self):
        from neurobox.analysis.lfp import local_minima
        # Two close minima — the closer (lower-valued) one wins
        x = np.ones(30)
        x[10] = -1.0
        x[12] = -2.0  # within 5 samples of x[10]
        x[25] = -3.0
        mins, _ = local_minima(x, not_closer_than=5)
        # The pair at 10/12 collapses to keep the lower value (12, -2.0)
        assert 12 in mins
        assert 10 not in mins
        assert 25 in mins

    def test_less_than_threshold(self):
        from neurobox.analysis.lfp import local_minima
        x = np.ones(30)
        x[5]  = -0.5
        x[15] = -2.0
        x[25] = -3.0
        mins, _ = local_minima(x, not_closer_than=1, less_than=-1.0)
        # Only minima below -1 count.
        np.testing.assert_array_equal(mins, [15, 25])

    def test_endpoints_not_counted(self):
        from neurobox.analysis.lfp import local_minima
        # Strictly decreasing then flat → no minimum (endpoints excluded)
        x = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        mins, _ = local_minima(x, not_closer_than=1)
        assert mins.size == 0

    def test_max_results_smallest(self):
        from neurobox.analysis.lfp import local_minima
        x = np.ones(40)
        x[5]  = -1.0
        x[15] = -3.0
        x[25] = -2.0
        x[35] = -4.0
        mins, vals = local_minima(x, not_closer_than=1, max_results=2)
        # Two smallest by value: -4 at idx 35, -3 at idx 15.
        assert set(mins.tolist()) == {15, 35}


# ─────────────────────────────────────────────────────────────────────────── #
# detect_oscillations (port of DetectOscillations)                             #
# ─────────────────────────────────────────────────────────────────────────── #

class TestDetectOscillations:

    def test_synthetic_theta_burst(self):
        """A 2-second 8 Hz burst embedded in white noise should be detected."""
        from neurobox.analysis.lfp import detect_oscillations
        fs = 1250.0
        rng = np.random.RandomState(42)
        n_total = 10 * int(fs)            # 10 seconds
        x = 0.05 * rng.randn(n_total)     # baseline noise
        # Insert a 2-second 8 Hz burst at 4–6 s
        t_burst = np.arange(2 * int(fs)) / fs
        burst   = 1.0 * np.sin(2 * np.pi * 8 * t_burst)
        x[4 * int(fs): 6 * int(fs)] += burst

        result = detect_oscillations(
            x, freq_range=(6.0, 12.0), samplerate=fs,
            min_cycles=5, threshold=90.0,
        )

        # At least one period detected, overlapping the burst window.
        assert result.periods.shape[0] > 0
        starts = result.periods[:, 0] / fs
        stops  = result.periods[:, 1] / fs
        overlapping = ((stops > 4.0) & (starts < 6.0))
        assert overlapping.any(), \
            f"no detected period overlaps 4–6 s burst: starts={starts}, stops={stops}"

    def test_returns_expected_attributes(self):
        from neurobox.analysis.lfp import detect_oscillations, OscillationResult
        fs = 1250.0
        rng = np.random.RandomState(0)
        x = 0.05 * rng.randn(5 * int(fs))
        result = detect_oscillations(x, freq_range=(6.0, 12.0), samplerate=fs)
        assert isinstance(result, OscillationResult)
        # All length-P arrays consistent
        p = result.peaks.size
        assert result.power.size == p
        assert result.z_power.size == p
        assert result.periods.shape == (p, 2)
        assert result.duration_ms.size == p
        # Threshold tuple has two finite numbers
        assert len(result.threshold) == 2
        assert result.samplerate == fs

    def test_freq_range_validation(self):
        from neurobox.analysis.lfp import detect_oscillations
        with pytest.raises(ValueError, match="freq_range"):
            detect_oscillations(np.zeros(1000), freq_range=(20.0, 5.0),
                                samplerate=1250.0)


# ─────────────────────────────────────────────────────────────────────────── #
# within_ranges Cython kernel (round 4)                                        #
# ─────────────────────────────────────────────────────────────────────────── #

class TestWithinRangesEngine:
    """Tests that exercise the Cython sweep-line engine directly and
    verify it agrees with the Python fallback on randomised inputs."""

    def test_engine_matches_fallback_random(self):
        from neurobox.analysis.lfp import within_ranges
        from neurobox.analysis.lfp._within_ranges_python_fallback import (
            within_ranges_matrix_engine_python,
        )
        rng = np.random.RandomState(42)
        for trial in range(10):
            n_pts    = rng.randint(50, 2_000)
            n_labels = rng.randint(1, 30)
            n_per_lab = rng.randint(1, 5)
            x = rng.uniform(0, 1000, size=n_pts)
            edges_raw = rng.uniform(0, 1000, size=(n_labels * n_per_lab, 2))
            edges = np.column_stack([
                np.minimum(edges_raw[:, 0], edges_raw[:, 1]),
                np.maximum(edges_raw[:, 0], edges_raw[:, 1]),
            ])
            labels = np.repeat(np.arange(1, n_labels + 1), n_per_lab)

            out_c = within_ranges(x, edges, range_label=labels, mode="matrix")
            out_p = within_ranges_matrix_engine_python(
                x, edges[:, 0], edges[:, 1], labels - 1, n_labels
            ).astype(bool)
            assert np.array_equal(out_c, out_p), (
                f"trial {trial}: cython kernel disagrees with fallback "
                f"(n_pts={n_pts}, n_labels={n_labels})"
            )

    def test_engine_inclusive_endpoints(self):
        """Labbox semantics: a query exactly at the start or stop of a
        range is *inside* that range.  This is the trickiest case for the
        sweep — tied event times must be processed start → point → stop."""
        from neurobox.analysis.lfp import within_ranges
        # x at 5 (=start), 10 (=stop), and well-inside (7), well-outside (15)
        x = np.array([5.0, 7.0, 10.0, 15.0])
        edges = np.array([[5.0, 10.0]])
        out = within_ranges(x, edges, range_label=np.array([1]), mode="matrix")
        # All four points: True at 5, 7, 10; False at 15.
        np.testing.assert_array_equal(out[:, 0], [True, True, True, False])

    def test_engine_handles_overlapping_same_label(self):
        """Two ranges with the same label — point must be flagged once."""
        from neurobox.analysis.lfp import within_ranges
        x = np.array([3.0, 6.0, 9.0])
        edges = np.array([[2.0, 7.0], [5.0, 10.0]])  # both label 1
        out = within_ranges(x, edges, range_label=np.array([1, 1]), mode="matrix")
        np.testing.assert_array_equal(out[:, 0], [True, True, True])

    def test_engine_dispatch_flag_present(self):
        """The dispatch flag is a public hook for users to check at runtime."""
        from neurobox.analysis.lfp.oscillations import _WR_USING_CYTHON
        assert isinstance(_WR_USING_CYTHON, bool)
