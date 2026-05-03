"""Round-18 tests — feature dynamics, KNN place fields, domain plots,
   and feature-selection MI ranking.
"""

from __future__ import annotations

import numpy as np
import pytest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402

from neurobox.dtype import (             # noqa: E402
    NBSpk, NBDxyz, NBModel, NBEpoch,
)


# ─────────────────────────────────────────────────────────────────────── #
# Feature dynamics                                                           #
# ─────────────────────────────────────────────────────────────────────── #

class TestTimeLaggedMI:
    def setup_method(self):
        self.rng = np.random.default_rng(0)

    def test_known_lag_recovered(self):
        from neurobox.analysis import time_lagged_mutual_information
        T = 5000
        base = self.rng.standard_normal(T) * 0.3
        col0 = base.copy()
        col1 = np.roll(base, +20)        # col0 leads col1 by 20
        col2 = self.rng.standard_normal(T)
        fet = np.column_stack([col0, col1, col2])
        mask = np.ones(T, dtype=bool)
        result = time_lagged_mutual_information(
            fet, mask, lags=np.arange(-30, 31),
            edges=np.linspace(-2, 2, 32),
        )
        # Peak MI between col0 (i=0) and col1 (j=1) should be at lag=+20
        mi_01 = result.values[:, 0, 1]
        peak_lag = result.lags[mi_01.argmax()]
        assert abs(peak_lag - 20) <= 2

    def test_self_pair_is_high(self):
        from neurobox.analysis import time_lagged_mutual_information
        T = 2000
        col = self.rng.standard_normal(T)
        fet = col[:, None]
        mask = np.ones(T, dtype=bool)
        result = time_lagged_mutual_information(
            fet, mask, lags=[0],
            edges=np.linspace(-3, 3, 24),
        )
        # I(X; X) at zero lag should be near max entropy
        assert result.values[0, 0, 0] > 1.0  # at least 1 bit

    def test_unrelated_features_low_mi(self):
        from neurobox.analysis import time_lagged_mutual_information
        T = 2000
        col0 = self.rng.standard_normal(T)
        col1 = self.rng.standard_normal(T)
        fet = np.column_stack([col0, col1])
        mask = np.ones(T, dtype=bool)
        result = time_lagged_mutual_information(
            fet, mask, lags=[0],
            edges=np.linspace(-3, 3, 16),
        )
        # I(X; Y) between independent gaussians should be small
        assert result.values[0, 0, 1] < 0.2

    def test_int_indices_accepted(self):
        from neurobox.analysis import time_lagged_mutual_information
        T = 1000
        fet = self.rng.standard_normal((T, 2))
        # Pass int indices instead of bool mask
        idx = np.arange(T // 2)
        result = time_lagged_mutual_information(
            fet, idx, lags=[0],
        )
        assert result.values.shape == (1, 2, 2)

    def test_invalid_inputs(self):
        from neurobox.analysis import time_lagged_mutual_information
        with pytest.raises(ValueError):
            time_lagged_mutual_information(
                np.zeros((10, 2, 2)), np.ones(10, bool), lags=[0],
            )


class TestTimeLaggedXC:
    def setup_method(self):
        self.rng = np.random.default_rng(1)

    def test_known_lag_recovered(self):
        from neurobox.analysis import time_lagged_cross_correlation
        T = 5000
        base = self.rng.standard_normal(T)
        col0 = base.copy()
        col1 = np.roll(base, +15)
        fet = np.column_stack([col0, col1])
        centres = self.rng.choice(np.arange(50, T - 50), size=200,
                                    replace=False)
        result = time_lagged_cross_correlation(
            fet, centres, lags=np.arange(-25, 26),
            seg_left=10, seg_right=10,
        )
        peak_lag = result.lags[result.values[:, 0, 1].argmax()]
        assert abs(peak_lag - 15) <= 2

    def test_zero_centres(self):
        from neurobox.analysis import time_lagged_cross_correlation
        result = time_lagged_cross_correlation(
            np.random.standard_normal((100, 2)),
            np.array([], dtype=np.int64),
            lags=[0],
        )
        assert result.values.shape == (1, 2, 2)
        assert (result.values == 0).all()


# ─────────────────────────────────────────────────────────────────────── #
# KNN place fields                                                           #
# ─────────────────────────────────────────────────────────────────────── #

def _make_knn_test_session(seed: int = 7,
                            target_xy: tuple[float, float] = (50.0, 100.0)):
    """Synthetic session: animal does a random walk over a 200×200 box
    and a single unit fires at *target_xy*.

    Uses a wide-step random walk with **wraparound** (not clipping) so
    coverage is approximately uniform over the box rather than
    biased toward the starting position.
    """
    rng = np.random.default_rng(seed)
    fs = 30.0
    T  = 12000              # 6.6 minutes — gives reasonable coverage
    pos = np.zeros((T, 2))
    pos[0] = [100, 100]
    step = rng.standard_normal((T, 2)) * 12
    for i in range(1, T):
        cand = pos[i-1] + step[i]
        # Wrap around to keep distribution uniform
        cand = np.mod(cand, 200)
        pos[i] = cand

    # Unit fires when within 25 mm of target (with wrap-around distance)
    target = np.asarray(target_xy)
    dx = np.abs(pos[:, 0] - target[0])
    dy = np.abs(pos[:, 1] - target[1])
    dx = np.minimum(dx, 200 - dx)
    dy = np.minimum(dy, 200 - dy)
    d2 = dx ** 2 + dy ** 2
    rate = 30.0 * np.exp(-d2 / (2 * 25.0 ** 2))
    n_per_frame = rng.poisson(rate / fs)
    res_sr = 30000.0
    res_list, clu_list = [], []
    for i in range(T):
        if n_per_frame[i] > 0:
            # Spike times in SECONDS (NBSpk.res convention)
            res_list.extend(rng.uniform(i, i + 1, n_per_frame[i]) / fs)
            clu_list.extend([2] * n_per_frame[i])
    res = np.asarray(res_list, dtype=np.float64)
    clu = np.asarray(clu_list, dtype=np.int32)
    sort_idx = np.argsort(res)
    spk = NBSpk(res=res[sort_idx], clu=clu[sort_idx], samplerate=res_sr)

    xyz_data = np.zeros((T, 1, 3))
    xyz_data[:, 0, :2] = pos
    xyz_data[:, 0, 2] = 50.0
    xyz = NBDxyz(xyz_data, model=NBModel(markers=["spine_lower"]),
                  samplerate=fs, name="knn_test")
    return spk, xyz, target


class TestKnnPlaceField:
    def test_basic_shape(self):
        from neurobox.analysis import knn_place_field
        spk, xyz, _ = _make_knn_test_session()
        pf = knn_place_field(
            spk, xyz, units=[2],
            bin_size=10.0, boundary=[[0, 200], [0, 200]],
            samplerate=10.0, n_neighbors=20, dist_threshold=50.0,
        )
        assert pf.rate_map.shape == (20, 20, 1, 1)
        assert len(pf.bin_centres) == 2
        assert pf.bin_centres[0].size == 20

    def test_peak_near_target(self):
        from neurobox.analysis import knn_place_field
        spk, xyz, target = _make_knn_test_session(target_xy=(100.0, 100.0))
        pf = knn_place_field(
            spk, xyz, units=[2],
            bin_size=10.0, boundary=[[0, 200], [0, 200]],
            samplerate=10.0, n_neighbors=20, dist_threshold=50.0,
        )
        rm = pf.rate_map[..., 0, 0]
        peak_idx = np.unravel_index(np.nanargmax(rm), rm.shape)
        peak_xy = np.array([
            pf.bin_centres[0][peak_idx[0]],
            pf.bin_centres[1][peak_idx[1]],
        ])
        # With wrap-around random walk and uniform coverage, peak
        # should land within ~40 mm of the true firing target.
        assert np.linalg.norm(peak_xy - target) < 40.0

    def test_bootstrap_iters(self):
        from neurobox.analysis import knn_place_field
        spk, xyz, _ = _make_knn_test_session()
        pf = knn_place_field(
            spk, xyz, units=[2],
            bin_size=10.0, boundary=[[0, 200], [0, 200]],
            samplerate=10.0, n_neighbors=20, dist_threshold=50.0,
            n_iter=3, block_size_seconds=2.0, rng=42,
        )
        assert pf.rate_map.shape == (20, 20, 1, 3)
        # Iter 0 = deterministic; iters 1+ = block-shuffled
        # The shuffled maps should NOT all be identical to iter 0
        diffs = []
        for it in range(1, 3):
            d = np.nanmean(np.abs(pf.rate_map[..., 0, 0]
                                    - pf.rate_map[..., 0, it]))
            diffs.append(d)
        # At least one shuffled iter should differ from iter 0
        assert max(diffs) > 1e-3

    def test_with_state_mask(self):
        from neurobox.analysis import knn_place_field
        spk, xyz, _ = _make_knn_test_session()
        # State that includes only the first half
        state = NBEpoch(
            data=np.array([[0, 3000]], dtype=np.float64),
            samplerate=30.0, mode="periods", label="run",
        )
        pf = knn_place_field(
            spk, xyz, units=[2],
            bin_size=20.0, boundary=[[0, 200], [0, 200]],
            samplerate=10.0, n_neighbors=10, dist_threshold=50.0,
            state=state,
        )
        assert pf.rate_map.shape == (10, 10, 1, 1)


class TestComputePfstatsBs:
    def test_aggregates_per_state(self):
        from neurobox.analysis import knn_place_field, compute_pfstats_bs
        spk, xyz, _ = _make_knn_test_session(target_xy=(50.0, 50.0))
        pf_walk = knn_place_field(
            spk, xyz, units=[2],
            bin_size=20.0, boundary=[[0, 200], [0, 200]],
            samplerate=10.0, n_neighbors=10, dist_threshold=50.0,
            n_iter=3, rng=42,
        )
        pf_rear = knn_place_field(
            spk, xyz, units=[2],
            bin_size=20.0, boundary=[[0, 200], [0, 200]],
            samplerate=10.0, n_neighbors=10, dist_threshold=50.0,
            n_iter=3, rng=43,
        )
        result = compute_pfstats_bs(
            {"walk": pf_walk, "rear": pf_rear}, units=[2],
            max_n_patches=2, threshold_pct=70.0,
        )
        assert result.peak_patch_area.shape == (2, 3, 1)
        assert result.peak_patch_com.shape  == (2, 3, 1, 2)
        assert result.peak_patch_rate.shape == (2, 3, 1)
        assert result.states == ("walk", "rear")
        # At least one state should have a finite COM
        assert np.isfinite(result.peak_patch_com).any()


# ─────────────────────────────────────────────────────────────────────── #
# Domain plots                                                               #
# ─────────────────────────────────────────────────────────────────────── #

def _synthetic_xyz(T: int = 1200, fs: float = 30.0) -> NBDxyz:
    t = np.arange(T) / fs
    markers = ["spine_lower", "pelvis_root", "spine_middle", "spine_upper",
               "head_back", "head_left", "head_front", "head_right"]
    data = np.zeros((T, len(markers), 3))
    heading = t * 0.3
    body_x = 100 + 80 * np.cos(heading)
    body_y = 100 + 80 * np.sin(heading)
    offsets = {"spine_lower":(-30,0,50),"pelvis_root":(-15,0,50),
               "spine_middle":(0,0,50),"spine_upper":(15,0,50),
               "head_back":(25,0,60),"head_left":(30,5,60),
               "head_front":(35,0,60),"head_right":(30,-5,60)}
    cy_, sy_ = np.cos(heading), np.sin(heading)
    for i, m in enumerate(markers):
        ox, oy, oz = offsets[m]
        data[:, i, 0] = body_x + ox * cy_ - oy * sy_
        data[:, i, 1] = body_y + ox * sy_ + oy * cy_
        data[:, i, 2] = oz + 10 * np.sin(heading * 2)
    model = NBModel(markers=markers, connections=[
        ["spine_lower","pelvis_root"], ["pelvis_root","spine_middle"],
        ["spine_middle","spine_upper"], ["spine_upper","head_back"],
        ["head_back","head_left"], ["head_back","head_right"],
        ["head_back","head_front"],
    ])
    return NBDxyz(data, model=model, samplerate=fs, name="dp_test")


class TestDomainPlots:
    def teardown_method(self):
        plt.close("all")

    def test_plot_xy_with_state(self):
        from neurobox.viz import plot_xy_with_state
        xyz = _synthetic_xyz()
        ep = NBEpoch(np.array([[100, 200]], dtype=np.float64),
                      samplerate=30.0, mode="periods", label="rear")
        fig, ax = plt.subplots()
        plot_xy_with_state(xyz, state=ep, ax=ax)
        assert len(ax.lines) >= 1

    def test_plot_xy_no_state(self):
        from neurobox.viz import plot_xy_with_state
        xyz = _synthetic_xyz()
        fig, ax = plt.subplots()
        plot_xy_with_state(xyz, ax=ax)
        assert len(ax.lines) == 1

    def test_plot_z(self):
        from neurobox.viz import plot_z
        xyz = _synthetic_xyz()
        fig, ax = plt.subplots()
        plot_z(xyz, ax=ax)
        assert len(ax.lines) == 1

    def test_plot_xy_velocity(self):
        from neurobox.viz import plot_xy_velocity
        xyz = _synthetic_xyz()
        fig, ax = plt.subplots()
        plot_xy_velocity(xyz, ax=ax)
        assert len(ax.lines) == 1

    def test_plot_rhm_spectrogram(self):
        from neurobox.viz import plot_rhm_spectrogram
        T, fs = 5000, 30.0
        sig = np.sin(2 * np.pi * 8 * np.arange(T) / fs)
        fig, ax = plt.subplots()
        plot_rhm_spectrogram(sig, fs, ax=ax)
        assert len(ax.images) == 1

    def test_plot_skeleton_3d(self):
        from neurobox.viz import plot_skeleton
        xyz = _synthetic_xyz()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        result = plot_skeleton(xyz, frame_index=500, ax=ax,
                                trajectory_period=(-30, 30))
        # 7 connections, 8 markers, 6 trajectory markers
        assert len(result["sticks"]) == 7
        assert len(result["markers"]) == 8
        assert len(result["trajectory"]) == 6

    def test_plot_skeleton_requires_3d_axes(self):
        from neurobox.viz import plot_skeleton
        xyz = _synthetic_xyz()
        fig, ax = plt.subplots()       # 2D axes
        with pytest.raises(ValueError, match="3-D"):
            plot_skeleton(xyz, frame_index=10, ax=ax)

    def test_plot_skeleton_line_alias(self):
        from neurobox.viz import plot_skeleton_line
        xyz = _synthetic_xyz()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        result = plot_skeleton_line(xyz, frame_index=100, ax=ax)
        assert "sticks" in result

    def test_plot_colored_curve(self):
        from neurobox.viz import plot_colored_curve
        fig, ax = plt.subplots()
        xs = np.linspace(0, 10, 50)
        plot_colored_curve(xs, np.sin(xs), xs, ax=ax)
        assert len(ax.collections) == 1

    def test_plot_colored_curve_3d(self):
        from neurobox.viz import plot_colored_curve_3d
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        xs = np.linspace(0, 5, 30)
        plot_colored_curve_3d(xs, np.sin(xs), np.cos(xs), xs, ax=ax)
        # Line3DCollection is added via add_collection3d


# ─────────────────────────────────────────────────────────────────────── #
# Feature selection                                                          #
# ─────────────────────────────────────────────────────────────────────── #

class TestPairwiseMIRanking:
    def setup_method(self):
        self.rng = np.random.default_rng(11)

    def test_ranks_correctly(self):
        from neurobox.analysis import pairwise_mutual_information_ranking
        T = 2000
        states = self.rng.choice(["A", "B", "C"], size=T,
                                   p=[0.4, 0.3, 0.3])
        fet = self.rng.standard_normal((T, 4))
        fet[states == "A", 0] += 3.0           # col 0 ~ class A
        fet[states == "B", 1] += 3.0           # col 1 ~ class B
        result = pairwise_mutual_information_ranking(
            fet, states, class_labels=("A", "B", "C"),
        )
        assert result.mi.shape == (4, 3)
        assert result.ranked_indices[0, 0] == 0
        assert result.ranked_indices[0, 1] == 1

    def test_two_d_label_matrix(self):
        from neurobox.analysis import pairwise_mutual_information_ranking
        T = 1000
        fet = self.rng.standard_normal((T, 3))
        labels = np.zeros((T, 2), dtype=bool)
        labels[:500, 0] = True       # first half is class 0
        labels[500:, 1] = True       # second half is class 1
        fet[:500, 0] += 5.0          # col 0 separates them
        result = pairwise_mutual_information_ranking(
            fet, labels, class_labels=("first", "second"),
        )
        # Both classes should have col 0 ranked first since it perfectly
        # separates them
        assert result.ranked_indices[0, 0] == 0
        assert result.ranked_indices[0, 1] == 0

    def test_no_signal_low_mi(self):
        from neurobox.analysis import pairwise_mutual_information_ranking
        T = 1000
        fet = self.rng.standard_normal((T, 3))
        states = self.rng.choice(["A", "B"], size=T)
        result = pairwise_mutual_information_ranking(fet, states)
        # All MI values should be below 0.1 bits
        assert (result.mi < 0.1).all()

    def test_validates_inputs(self):
        from neurobox.analysis import pairwise_mutual_information_ranking
        with pytest.raises(ValueError, match="2-D"):
            pairwise_mutual_information_ranking(
                np.zeros(100), np.zeros(100, dtype=int),
            )
        with pytest.raises(ValueError, match="length"):
            pairwise_mutual_information_ranking(
                np.zeros((100, 3)), np.zeros(50, dtype=int),
            )
        with pytest.raises(ValueError):
            pairwise_mutual_information_ranking(
                np.zeros((100, 3)), np.zeros((100, 3)),
                class_labels=("A", "B"),  # length mismatch
            )
