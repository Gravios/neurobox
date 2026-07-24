"""Tests for the ``fet_mis`` feature set and the stage-1 heuristic
behaviour labeller.

Both are ports of MATLAB used by ``label_behavior.m``:

* :func:`neurobox.analysis.kinematics.fet_mis.fet_mis`
  ← :file:`MTA/features/fet_mis.m`
* :func:`neurobox.analysis.classifiers.heuristic_labeling.label_with_heuristics`
  ← :file:`MTA/classifiers/label_behavior_with_heuristics.m`
"""
from __future__ import annotations

import numpy as np
import pytest

from neurobox.dtype import NBDxyz, NBModel
from neurobox.analysis.kinematics.augment import augment_xyz
from neurobox.analysis.kinematics.fet_mis import (
    fet_mis, FetMisConfig, FET_MIS_TITLES, FET_MIS_DESCRIPTIONS,
)
from neurobox.analysis.classifiers.heuristic_labeling import (
    label_with_heuristics, HeuristicThresholds, windowed_trajectory_stats,
    _gauss_smooth,
)
from neurobox.analysis.stats.circular import circ_mean


MARKERS = ["spine_lower", "pelvis_root", "spine_middle", "spine_upper",
           "head_back", "head_left", "head_front", "head_right"]


def _synthetic(
    *,
    seed:      int = 1,
    seconds:   float = 60.0,
    fs:        float = 120.0,
    walk_bouts: tuple[tuple[float, float], ...] = ((10., 20.), (35., 45.)),
    rear_bout: tuple[float, float] | None = None,
    sway:      float = 0.0,
) -> NBDxyz:
    """Marker positions with known walk / rear ground truth."""
    rng = np.random.default_rng(seed)
    T = int(seconds * fs)
    t = np.arange(T) / fs

    speed = np.zeros(T)
    for a, b in walk_bouts:
        speed[(t > a) & (t < b)] = 6.0
    path = np.cumsum(np.stack([speed, speed * 0.3], 1), 0)
    base = np.concatenate([path, np.zeros((T, 1))], 1)[:, None, :]
    off = (np.linspace(0, 140, len(MARKERS))[None, :, None]
           * np.array([1., 0., 0.]))
    data = base + off + rng.normal(0, 1.2, size=(T, len(MARKERS), 3))
    data[:, :, 2] += 60.0

    if sway:
        prof = np.linspace(0, 1, len(MARKERS)) ** 2
        s = np.sin(2 * np.pi * 1.6 * t) * (speed > 0)
        data[:, :, 1] += s[:, None] * prof[None, :] * sway

    if rear_bout is not None:
        a, b = rear_bout
        m = (t > a) & (t < b)
        ramp = np.zeros(T)
        ramp[m] = np.hanning(int(m.sum()))
        lift = np.array([0, .15, .45, .75, 1.0, 1.0, 1.15, 1.0])
        data[:, :, 2] += (ramp[:, None] * lift[None, :]) * 150.0
        data[:, :, 0] -= (ramp[:, None] * lift[None, :]) * 60.0

    return augment_xyz(NBDxyz(data, model=NBModel(MARKERS), samplerate=fs))


# ─────────────────────────────────────────────────────────────────────── #
# fet_mis                                                                 #
# ─────────────────────────────────────────────────────────────────────── #

class TestFetMis:
    def test_shape_and_rate(self):
        xyz = _synthetic(seconds=20.0)
        fet = fet_mis(xyz)
        assert fet.data.shape[1] == 18
        assert fet.samplerate == pytest.approx(12.0)
        # 20 s at 12 Hz
        assert abs(fet.data.shape[0] - 240) <= 2

    def test_all_finite(self):
        fet = fet_mis(_synthetic(seconds=20.0))
        assert np.isfinite(fet.data).all()

    def test_titles_and_descriptions_align(self):
        assert len(FET_MIS_TITLES) == 18
        assert len(FET_MIS_DESCRIPTIONS) == 18
        fet = fet_mis(_synthetic(seconds=20.0))
        assert list(fet.titles) == list(FET_MIS_TITLES)

    def test_custom_sample_rate(self):
        xyz = _synthetic(seconds=20.0)
        fet = fet_mis(xyz, config=FetMisConfig(sample_rate=30.0))
        assert fet.samplerate == pytest.approx(30.0)
        assert abs(fet.data.shape[0] - 600) <= 4

    def test_speed_columns_are_log10(self):
        """Columns 11-16 (0-indexed 10-15) are log10 speeds, so a
        faster synthetic must raise them."""
        slow = fet_mis(_synthetic(seconds=30.0, walk_bouts=()))
        fast = fet_mis(_synthetic(seconds=30.0,
                                   walk_bouts=((5., 25.),)))
        assert fast.data[:, 10].mean() > slow.data[:, 10].mean()

    def test_missing_markers_raise(self):
        rng = np.random.default_rng(0)
        bare = ["spine_lower", "pelvis_root"]
        xyz = NBDxyz(rng.normal(size=(600, 2, 3)),
                     model=NBModel(bare), samplerate=120.0)
        with pytest.raises(KeyError, match="requires markers"):
            fet_mis(xyz)

    def test_config_is_hashable(self):
        """Frozen dataclass so it can key a cached_compute cache —
        the replacement for MATLAB's concatenated model string."""
        assert hash(FetMisConfig()) == hash(FetMisConfig())
        assert hash(FetMisConfig()) != hash(FetMisConfig(sample_rate=30.0))


# ─────────────────────────────────────────────────────────────────────── #
# windowed_trajectory_stats                                               #
# ─────────────────────────────────────────────────────────────────────── #

class TestWindowedTrajectoryStats:
    def test_output_shape(self):
        m, v = windowed_trajectory_stats(np.random.randn(640, 3), 64, 8)
        assert m.shape == (80, 3) and v.shape == (80, 3)

    def test_1d_input_promoted(self):
        m, v = windowed_trajectory_stats(np.random.randn(640), 64, 8)
        assert m.shape == (80, 1)

    def test_constant_signal_has_zero_variance(self):
        m, v = windowed_trajectory_stats(np.ones((640, 2)), 64, 8)
        assert np.allclose(v, 0.0)
        assert np.allclose(m, 0.0)      # relative to window start

    def test_linear_ramp_has_positive_mean(self):
        x = np.arange(640, dtype=float)[:, None]
        m, _ = windowed_trajectory_stats(x, 64, 8)
        assert (m > 0).mean() > 0.9

    def test_circular_mode_differs(self):
        rng = np.random.default_rng(0)
        x = rng.uniform(-np.pi, np.pi, (640, 2))
        m_lin, _ = windowed_trajectory_stats(x, 64, 8, circular=False)
        m_circ, _ = windowed_trajectory_stats(x, 64, 8, circular=True)
        assert not np.allclose(m_lin, m_circ)


# ─────────────────────────────────────────────────────────────────────── #
# label_with_heuristics                                                   #
# ─────────────────────────────────────────────────────────────────────── #

class TestLabelWithHeuristics:
    def test_produces_three_states(self):
        stc = label_with_heuristics(_synthetic())
        assert {"gper", "walk", "rear"} <= set(stc.list_states())

    def test_state_keys_match_matlab(self):
        stc = label_with_heuristics(_synthetic())
        assert stc["gper"].key == "a"
        assert stc["walk"].key == "w"
        assert stc["rear"].key == "r"

    def test_detects_known_walk_bouts(self):
        fs = 120.0
        stc = label_with_heuristics(
            _synthetic(walk_bouts=((10., 20.), (35., 45.))))
        per = stc["walk"].data / fs
        assert per.shape[0] == 2
        # within half a window (64 samples ≈ 0.53 s) of ground truth
        assert per[0][0] == pytest.approx(10.0, abs=0.6)
        assert per[0][1] == pytest.approx(20.0, abs=0.6)
        assert per[1][0] == pytest.approx(35.0, abs=0.6)
        assert per[1][1] == pytest.approx(45.0, abs=0.6)

    def test_no_walk_when_stationary(self):
        stc = label_with_heuristics(_synthetic(walk_bouts=()))
        assert stc["walk"].data.shape[0] == 0

    def test_detects_rear(self):
        fs = 120.0
        stc = label_with_heuristics(
            _synthetic(walk_bouts=(), rear_bout=(25., 28.)))
        per = stc["rear"].data / fs
        assert per.shape[0] == 1
        # the 1 Hz low-pass narrows the detected span inside the bout
        assert 25.0 <= per[0][0] <= 28.0
        assert 25.0 <= per[0][1] <= 28.5

    def test_gper_spans_recording(self):
        fs = 120.0
        stc = label_with_heuristics(_synthetic(seconds=60.0))
        g = stc["gper"].data
        assert g.shape == (1, 2)
        assert g[0][1] / fs == pytest.approx(60.0, abs=1.0)

    def test_returns_without_side_effects(self, tmp_path):
        """Unlike MATLAB this writes nothing — the caller decides."""
        before = set(tmp_path.iterdir())
        label_with_heuristics(_synthetic(seconds=20.0))
        assert set(tmp_path.iterdir()) == before

    def test_missing_markers_raise(self):
        rng = np.random.default_rng(0)
        xyz = NBDxyz(rng.normal(size=(600, 2, 3)),
                     model=NBModel(["spine_lower", "pelvis_root"]),
                     samplerate=120.0)
        with pytest.raises(KeyError, match="requires markers"):
            label_with_heuristics(xyz)

    def test_too_short_raises(self):
        rng = np.random.default_rng(0)
        data = rng.normal(size=(50, len(MARKERS), 3))
        xyz = augment_xyz(
            NBDxyz(data, model=NBModel(MARKERS), samplerate=120.0))
        with pytest.raises(ValueError, match="too short"):
            label_with_heuristics(xyz)

    def test_raising_walk_feature_threshold_reduces_detections(self):
        xyz = _synthetic()
        loose = label_with_heuristics(xyz)
        strict = label_with_heuristics(
            xyz, thresholds=HeuristicThresholds(walk_feature=99.0))
        assert strict["walk"].data.shape[0] <= loose["walk"].data.shape[0]
        assert strict["walk"].data.shape[0] == 0

    def test_mode_tag_propagates(self):
        stc = label_with_heuristics(_synthetic(seconds=20.0), mode="custom")
        assert stc.mode == "custom"


# ─────────────────────────────────────────────────────────────────────── #
# The bfet defect                                                         #
# ─────────────────────────────────────────────────────────────────────── #

class TestBfetDefect:
    """MATLAB line 138 builds ``btraj`` from ``hfet`` rather than
    ``bfet``, so the intended 5-column spine-chain signal is discarded
    and ``bf`` duplicates ``hf``.  See the module docstring."""

    def test_default_reproduces_the_defect(self):
        assert HeuristicThresholds().reproduce_bfet_defect is True

    def test_the_two_sources_give_different_traces(self):
        """Directly exercise the branch: the 5-column and 2-column
        inputs produce measurably different smoothed traces, so the
        flag is not cosmetic."""
        rng = np.random.default_rng(0)
        bfet = np.cumsum(rng.normal(0, 0.05, (64 * 20, 5)), axis=0)
        hfet = bfet[:, [2, 4]]

        def trace(x):
            m, v = windowed_trajectory_stats(x, 64, 8, circular=True)
            return _gauss_smooth(
                circ_mean(m, axis=1) * np.sqrt((v ** 2).sum(axis=1)), 21)

        assert not np.allclose(trace(bfet), trace(hfet))

    def test_flag_runs_both_ways(self):
        """Both settings must produce a valid StateCollection; whether
        the labels differ is data-dependent, since the defect only
        bites when ``bf`` crosses the ``body_turn`` gate."""
        xyz = _synthetic(sway=22.0)
        for flag in (True, False):
            stc = label_with_heuristics(
                xyz,
                thresholds=HeuristicThresholds(reproduce_bfet_defect=flag))
            assert {"gper", "walk", "rear"} <= set(stc.list_states())


# ─────────────────────────────────────────────────────────────────────── #
# Thresholds container                                                    #
# ─────────────────────────────────────────────────────────────────────── #

class TestHeuristicThresholds:
    def test_matlab_defaults(self):
        """Transcribed verbatim from the MATLAB source; if these
        change, historical labels change."""
        th = HeuristicThresholds()
        assert th.window == 64 and th.overlap == 8
        assert th.fine_window == 16 and th.fine_overlap == 2
        assert th.walk_feature == pytest.approx(1.6)
        assert th.walk_distance_1 == pytest.approx(1.85)
        assert th.walk_distance_2 == pytest.approx(1.60)
        assert th.walk_distance_3 == pytest.approx(1.80)
        assert th.walk_angle == pytest.approx(0.90)
        assert th.body_turn == pytest.approx(0.02)
        assert th.min_duration == 32
        assert th.merge_gap == 16
        assert th.traj_dot_norm == pytest.approx(10.0)
        assert th.fine_walk_feature == pytest.approx(1.49)
        assert th.rear_feature == pytest.approx(45.0)
        assert th.rear_height == pytest.approx(170.0)

    def test_is_hashable(self):
        assert hash(HeuristicThresholds()) == hash(HeuristicThresholds())
        assert (hash(HeuristicThresholds())
                != hash(HeuristicThresholds(walk_feature=2.0)))
