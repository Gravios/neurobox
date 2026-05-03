"""Round-23 — StreamSync + TrialWindow + NBData.restrict_to_window.

Tests the multi-segment-aware sync system that replaces the recursive
MTAData.sync chain.  Multi-segment Vicon recording is the central
test scenario: a stream that paused/resumed multiple times during
one master-clock recording, requiring zero-fill in trial windows
that straddle pauses.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from neurobox.dtype import (
    NBDxyz, NBDfet, NBDlfp, NBModel,
    StreamSync, TrialWindow,
)


# ─────────────────────────────────────────────────────────────────────── #
# StreamSync arithmetic                                                      #
# ─────────────────────────────────────────────────────────────────────── #

class TestStreamSyncBasics:
    def test_construction_normalises(self):
        # 1-D input is reshaped to (1, 2)
        sync = StreamSync(segments=np.array([10.0, 30.0]),
                            samplerate=120.0)
        assert sync.segments.shape == (1, 2)

        # Out-of-order segments are sorted
        sync = StreamSync(segments=np.array([[50, 70], [10, 30]]),
                            samplerate=120.0)
        np.testing.assert_array_equal(sync.segments[0], [10, 30])
        np.testing.assert_array_equal(sync.segments[1], [50, 70])

    def test_rejects_invalid_segments(self):
        with pytest.raises(ValueError, match="strictly stop > start"):
            StreamSync(segments=np.array([[10.0, 5.0]]), samplerate=120.0)
        with pytest.raises(ValueError, match="overlap"):
            StreamSync(segments=np.array([[10, 30], [20, 40]]),
                        samplerate=120.0)

    def test_rejects_non_positive_samplerate(self):
        with pytest.raises(ValueError, match="samplerate"):
            StreamSync(segments=np.array([[0.0, 10.0]]), samplerate=0)

    def test_total_samples_and_cumulative(self):
        sync = StreamSync(
            segments=np.array([[10, 30], [50, 70], [90, 100]]),
            samplerate=120.0,
        )
        assert sync.total_samples == 6000
        np.testing.assert_array_equal(
            sync.cumulative_samples,
            [0, 2400, 4800, 6000],
        )

    def test_master_first_last_span(self):
        sync = StreamSync(
            segments=np.array([[10, 30], [50, 70]]),
            samplerate=120.0,
        )
        assert sync.master_first == 10.0
        assert sync.master_last == 70.0
        assert sync.master_span == 60.0

    def test_gaps(self):
        sync = StreamSync(
            segments=np.array([[10, 30], [50, 70], [90, 100]]),
            samplerate=120.0,
        )
        gaps = sync.gaps
        assert gaps.shape == (2, 2)
        np.testing.assert_array_equal(gaps[0], [30, 50])
        np.testing.assert_array_equal(gaps[1], [70, 90])

    def test_empty_sync(self):
        empty = StreamSync(segments=np.zeros((0, 2)), samplerate=120.0)
        assert empty.is_empty
        assert empty.total_samples == 0
        assert np.isnan(empty.master_first)


class TestStreamSyncConversions:
    def setup_method(self):
        # The canonical example: 3 Vicon takes during one NLX recording
        self.sync = StreamSync(
            segments=np.array([[10, 30], [50, 70], [90, 100]]),
            samplerate=120.0,
        )

    def test_local_to_master_first_segment(self):
        # local 0 → master 10.0 (first sample of segment 0)
        assert self.sync.local_to_master(0) == 10.0
        # local 1200 → master 20.0 (10 s into seg 0)
        assert abs(self.sync.local_to_master(1200) - 20.0) < 1e-6

    def test_local_to_master_segment_boundary(self):
        # local 2400 → master 50.0 (first sample of segment 1, NOT 30)
        assert self.sync.local_to_master(2400) == 50.0

    def test_local_to_master_last_sample(self):
        # local 5999 → master ~99.99
        assert abs(self.sync.local_to_master(5999) - 99.991667) < 1e-3

    def test_local_to_master_out_of_range(self):
        with pytest.raises(IndexError):
            self.sync.local_to_master(-1)
        with pytest.raises(IndexError):
            self.sync.local_to_master(6000)

    def test_master_to_local_in_segment(self):
        assert self.sync.master_to_local(25.0) == 1800
        assert self.sync.master_to_local(60.0) == 3600
        assert self.sync.master_to_local(95.0) == 5400

    def test_master_to_local_in_gap_returns_none(self):
        assert self.sync.master_to_local(40.0) is None  # in gap
        assert self.sync.master_to_local(80.0) is None  # in gap

    def test_master_to_local_outside_total_range(self):
        assert self.sync.master_to_local(5.0) is None    # before any seg
        assert self.sync.master_to_local(105.0) is None  # after all segs

    def test_round_trip(self):
        for t in [10.0, 25.5, 60.0, 99.0]:
            local = self.sync.master_to_local(t)
            assert local is not None
            recovered = self.sync.local_to_master(local)
            assert abs(recovered - t) < 1e-6


class TestStreamSyncWindowOps:
    def setup_method(self):
        self.sync = StreamSync(
            segments=np.array([[10, 30], [50, 70], [90, 100]]),
            samplerate=120.0,
        )

    def test_slice_for_window_straddles_gap(self):
        # Window [20, 60] overlaps seg 0 (20..30) and seg 1 (50..60)
        slices = self.sync.slice_for_window(20.0, 60.0)
        assert slices == [(1200, 2400), (2400, 3600)]

    def test_slice_for_window_inside_one_segment(self):
        slices = self.sync.slice_for_window(55.0, 65.0)
        assert slices == [(3000, 4200)]

    def test_slice_for_window_in_gap_returns_empty(self):
        slices = self.sync.slice_for_window(35.0, 45.0)
        assert slices == []

    def test_restricted_to_window(self):
        sub = self.sync.restricted_to_window(20.0, 60.0)
        np.testing.assert_array_equal(
            sub.segments,
            [[20.0, 30.0], [50.0, 60.0]],
        )

    def test_valid_mask_in_window(self):
        # Window [20, 60] @ 120 Hz = 4800 samples
        # Recorded portions: 1200 samples in seg 0, 1200 in seg 1
        mask = self.sync.valid_mask_in_window(20.0, 60.0)
        assert len(mask) == 4800
        assert mask.sum() == 2400
        # Specifically: True at samples 0..1199 (master 20..30)
        # and 3600..4799 (master 50..60)
        assert mask[0] and mask[1199]
        assert not mask[1200] and not mask[3599]
        assert mask[3600] and mask[4799]


class TestStreamSyncFactories:
    def test_continuous(self):
        sync = StreamSync.continuous(duration_sec=100.0, samplerate=1250.0)
        assert sync.n_segments == 1
        assert sync.master_first == 0.0
        assert sync.master_last == 100.0
        assert sync.total_samples == 125_000

    def test_continuous_with_t_start(self):
        sync = StreamSync.continuous(
            duration_sec=10.0, samplerate=1000.0, t_start=5.0,
        )
        assert sync.master_first == 5.0
        assert sync.master_last == 15.0

    def test_from_ttl_pulses(self):
        starts = np.array([10.0, 50.0, 90.0])
        stops  = np.array([30.0, 70.0, 100.0])
        sync = StreamSync.from_ttl_pulses(starts, stops, samplerate=120.0)
        assert sync.n_segments == 3
        np.testing.assert_array_equal(sync.segments[:, 0], starts)
        np.testing.assert_array_equal(sync.segments[:, 1], stops)

    def test_from_ttl_pulses_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            StreamSync.from_ttl_pulses(
                np.array([1, 2]), np.array([3]),
                samplerate=120.0,
            )


# ─────────────────────────────────────────────────────────────────────── #
# TrialWindow                                                                #
# ─────────────────────────────────────────────────────────────────────── #

class TestTrialWindow:
    def test_basic_construction(self):
        win = TrialWindow(periods=np.array([[20.0, 60.0]]),
                            label="task1", name="task1")
        assert win.t_start == 20.0
        assert win.t_stop == 60.0
        assert win.total_duration == 40.0

    def test_multi_period(self):
        win = TrialWindow(periods=np.array([[10, 20], [40, 50]]))
        # span: 10 to 50 = 40, but total duration sums periods = 20
        assert win.t_start == 10.0
        assert win.t_stop == 50.0
        assert win.total_duration == 20.0

    def test_contains(self):
        win = TrialWindow(periods=np.array([[10, 20], [40, 50]]))
        assert win.contains(15.0)
        assert win.contains(45.0)
        assert not win.contains(30.0)         # in gap
        assert not win.contains(5.0)
        assert not win.contains(60.0)

    def test_rejects_bad_periods(self):
        with pytest.raises(ValueError, match="stop > start"):
            TrialWindow(periods=np.array([[10.0, 5.0]]))

    def test_whole_session_factory(self):
        sync = StreamSync(
            segments=np.array([[10, 30], [50, 70]]),
            samplerate=120.0,
        )
        win = TrialWindow.whole_session(sync, label="all", name="all")
        # Single period from first start to last stop
        assert win.t_start == 10.0
        assert win.t_stop == 70.0
        assert win.label == "all"

    def test_empty_window(self):
        win = TrialWindow(periods=np.zeros((0, 2)))
        assert win.is_empty
        assert win.total_duration == 0.0
        assert not win.contains(5.0)


# ─────────────────────────────────────────────────────────────────────── #
# NBData.restrict_to_window — multi-segment Vicon scenario                   #
# ─────────────────────────────────────────────────────────────────────── #

class TestRestrictToWindow:
    def setup_method(self):
        # Vicon @ 120 Hz, 3 takes; data values = local sample idx
        T = 6000
        data = np.zeros((T, 4, 3))
        for i in range(T):
            data[i, 0, 0] = i      # marker 0, dim x = local idx
        self.xyz = NBDxyz(
            data, model=NBModel(markers=["a", "b", "c", "d"]),
            samplerate=120.0,
            stream_sync=StreamSync(
                segments=np.array([[10, 30], [50, 70], [90, 100]]),
                samplerate=120.0,
            ),
        )

    def test_window_straddling_gap_fill(self):
        """Trial [20, 60] crosses gap [30, 50] → 4800 output samples
        with the gap zero-filled."""
        win = TrialWindow(periods=np.array([[20.0, 60.0]]),
                            label="task1", name="task1")
        sub = self.xyz.restrict_to_window(win)

        assert sub.data.shape == (4800, 4, 3)
        # First 1200 samples come from local 1200..2400 (seg 0 [20,30])
        assert sub.data[0, 0, 0] == 1200
        assert sub.data[1199, 0, 0] == 2399
        # Gap zero-filled
        assert sub.data[1200, 0, 0] == 0
        assert sub.data[3599, 0, 0] == 0
        # Last 1200 samples from local 2400..3600 (seg 1 [50,60])
        assert sub.data[3600, 0, 0] == 2400
        assert sub.data[4799, 0, 0] == 3599
        # New stream_sync covers only the recorded portions
        np.testing.assert_array_equal(
            sub.stream_sync.segments,
            [[20.0, 30.0], [50.0, 60.0]],
        )

    def test_window_straddling_gap_compact(self):
        """fill_gaps=False produces a compact array — gap dropped."""
        win = TrialWindow(periods=np.array([[20.0, 60.0]]))
        sub = self.xyz.restrict_to_window(win, fill_gaps=False)
        assert sub.data.shape == (2400, 4, 3)
        # No gap: seg 0 portion immediately followed by seg 1 portion
        assert sub.data[1199, 0, 0] == 2399
        assert sub.data[1200, 0, 0] == 2400

    def test_window_inside_segment(self):
        """Trial fully inside one segment — no gap fill needed."""
        win = TrialWindow(periods=np.array([[55.0, 65.0]]))
        sub = self.xyz.restrict_to_window(win)
        assert sub.data.shape == (1200, 4, 3)
        # master 55 → local 3000
        assert sub.data[0, 0, 0] == 3000

    def test_window_fully_in_gap_zeros(self):
        """Trial entirely in a gap — all zeros, empty stream_sync."""
        win = TrialWindow(periods=np.array([[35.0, 45.0]]))
        sub = self.xyz.restrict_to_window(win)
        assert sub.data.shape == (1200, 4, 3)
        assert (sub.data == 0).all()
        assert sub.stream_sync.is_empty

    def test_multi_period_window(self):
        """Trial with disjoint periods — concatenated output."""
        win = TrialWindow(periods=np.array([[15, 25], [55, 65]]))
        sub = self.xyz.restrict_to_window(win)
        # Each period: 10 s × 120 Hz = 1200 samples; total 2400
        assert sub.data.shape == (2400, 4, 3)
        # First period: master [15, 25] → local [600, 1800]
        assert sub.data[0, 0, 0] == 600
        assert sub.data[1199, 0, 0] == 1799
        # Second period: master [55, 65] → local [3000, 4200]
        assert sub.data[1200, 0, 0] == 3000
        assert sub.data[2399, 0, 0] == 4199

    def test_no_stream_sync_falls_back_to_continuous(self):
        """Streams without stream_sync are assumed continuous from t=0."""
        T = 1000
        data = np.arange(T)[:, None, None].astype(np.float64) * np.ones((1, 4, 3))
        xyz = NBDxyz(
            data, model=NBModel(markers=["a", "b", "c", "d"]),
            samplerate=10.0,           # 100 s recording
            # NOTE: no stream_sync passed
        )
        win = TrialWindow(periods=np.array([[20.0, 50.0]]))
        sub = xyz.restrict_to_window(win)
        # 30 s × 10 Hz = 300 samples
        assert sub.data.shape == (300, 4, 3)
        # First sample from t=20 → idx 200
        assert sub.data[0, 0, 0] == 200

    def test_unloaded_data_raises(self):
        xyz = NBDxyz(samplerate=120.0)         # no data
        win = TrialWindow(periods=np.array([[0, 1]]))
        with pytest.raises(RuntimeError, match="not loaded"):
            xyz.restrict_to_window(win)

    def test_samplerate_mismatch_raises(self):
        bad_xyz = NBDxyz(
            np.zeros((100, 1, 3)),
            model=NBModel(markers=["a"]),
            samplerate=120.0,
            stream_sync=StreamSync(
                segments=np.array([[0.0, 1.0]]),
                samplerate=99.0,                # wrong!
            ),
        )
        win = TrialWindow(periods=np.array([[0.0, 0.5]]))
        with pytest.raises(ValueError, match="samplerate"):
            bad_xyz.restrict_to_window(win)

    def test_preserves_subclass_fields(self):
        """restrict_to_window returns the same subclass with model intact."""
        win = TrialWindow(periods=np.array([[20.0, 30.0]]))
        sub = self.xyz.restrict_to_window(win)
        assert isinstance(sub, NBDxyz)
        assert sub.model.markers == ["a", "b", "c", "d"]
        assert sub.samplerate == 120.0


# ─────────────────────────────────────────────────────────────────────── #
# Cross-stream rates: LFP @ 1250 + Vicon @ 120 in same trial                 #
# ─────────────────────────────────────────────────────────────────────── #

class TestMixedStreamRates:
    """Verify a single TrialWindow drives correct slicing on streams
    at very different sample rates."""

    def test_lfp_and_xyz_share_trial(self):
        # Master clock = NLX, runs continuously for 100 s
        # LFP @ 1250 Hz across the whole 100 s
        lfp_data = np.arange(125_000)[:, None].astype(np.float64) * np.ones((1, 4))
        lfp = NBDlfp(
            lfp_data, samplerate=1250.0,
            stream_sync=StreamSync.continuous(100.0, 1250.0),
        )
        # Vicon @ 120 Hz, 3 segments
        xyz_data = np.zeros((6000, 4, 3))
        for i in range(6000):
            xyz_data[i, 0, 0] = i
        xyz = NBDxyz(
            xyz_data, model=NBModel(markers=["a", "b", "c", "d"]),
            samplerate=120.0,
            stream_sync=StreamSync(
                segments=np.array([[10, 30], [50, 70], [90, 100]]),
                samplerate=120.0,
            ),
        )

        # One trial window applies to both streams
        win = TrialWindow(periods=np.array([[40.0, 80.0]]),
                            label="task1")
        lfp_sub = lfp.restrict_to_window(win)
        xyz_sub = xyz.restrict_to_window(win)

        # LFP: 40 s × 1250 Hz = 50_000 samples (continuous, no gaps)
        assert lfp_sub.data.shape == (50_000, 4)
        # First sample from master t=40 → local 50_000
        assert lfp_sub.data[0, 0] == 50_000

        # xyz: 40 s × 120 Hz = 4800 samples;
        # gap at [40, 50] (zero-fill, 1200 samples),
        # data at [50, 70] (2400 samples from local 2400..4800),
        # gap at [70, 80] (zero-fill, 1200 samples)
        assert xyz_sub.data.shape == (4800, 4, 3)
        # First 1200 samples should be zero (gap)
        assert (xyz_sub.data[:1200, 0, 0] == 0).all()
        # Next 2400 samples come from local 2400..4800
        assert xyz_sub.data[1200, 0, 0] == 2400
        assert xyz_sub.data[3599, 0, 0] == 4799
        # Last 1200 samples zero-filled (gap)
        assert (xyz_sub.data[3600:, 0, 0] == 0).all()
