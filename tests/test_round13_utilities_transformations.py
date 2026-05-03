"""Tests for round-13 — utilities + transformations ports."""

from __future__ import annotations

import numpy as np
import pytest

from neurobox.analysis.transformations import (
    BinAxis,
    bin_statistic_2d, bin_statistic_2d_circ, bin_statistic_3d, bin_statistic,
    rot_z_axis, rot_y_axis, detect_roll,
    thetarc_phase,
    quat2rotm, quaternion2rad,
    make_uniform_distr, shilbert, my_theta_phase,
    decompose_xy_motion_wrt_body, BodyMotionSVDModel,
)
from neurobox.analysis.lfp import rect_filter, gauss_window
from neurobox.analysis.mocap import (
    marker_diff_matrix, inter_marker_distance,
    inter_marker_angles, inter_marker_orientation,
)
from neurobox.io import data_hash


# ─────────────────────────────────────────────────────────────────────── #
# bin_statistics                                                            #
# ─────────────────────────────────────────────────────────────────────── #

class TestBinStatistic2D:
    def test_count_and_mean(self):
        rng = np.random.default_rng(0)
        x = rng.uniform(0, 10, 1000)
        y = rng.uniform(0, 10, 1000)
        v = x + y
        edges = np.linspace(0, 10, 6)
        s = bin_statistic_2d((x, edges), (y, edges), v)
        assert s.count.shape == (5, 5)
        assert s.count.sum() == 1000
        # Mean in bin (0,0): values where x<2, y<2 → mean ~ 2.0
        assert 1.5 < s.mean[0, 0] < 2.5

    def test_size_mismatch(self):
        edges = np.linspace(0, 10, 6)
        with pytest.raises(ValueError, match="size mismatch"):
            bin_statistic_2d((np.zeros(10), edges),
                              (np.zeros(20), edges),
                              np.zeros(10))

    def test_arbitrary_reducer(self):
        rng = np.random.default_rng(0)
        x = rng.uniform(0, 10, 200)
        y = rng.uniform(0, 10, 200)
        v = np.ones(200)
        edges = np.linspace(0, 10, 5)
        s = bin_statistic((x, edges), (y, edges), v, np.sum)
        # Sum of 1's per bin == count per bin
        assert np.allclose(s.data, s.count)


class TestBinStatistic2DCirc:
    def test_circ_mean(self):
        rng = np.random.default_rng(0)
        N = 500
        x = rng.uniform(0, 10, N)
        y = rng.uniform(0, 10, N)
        # All angles in bin 0,0 = 1.0 (constant)
        ang = np.full(N, 1.0)
        edges = np.linspace(0, 10, 4)
        s = bin_statistic_2d_circ((x, edges), (y, edges), ang)
        # Mean angle should be ~1.0 wherever count > 0
        valid = s.count > 0
        np.testing.assert_allclose(s.mean[valid], 1.0, atol=1e-6)


class TestBinStatistic3D:
    def test_count_total(self):
        rng = np.random.default_rng(0)
        N = 1000
        x = rng.uniform(0, 10, N)
        y = rng.uniform(0, 10, N)
        z = rng.uniform(0, 10, N)
        v = np.ones(N)
        edges = np.linspace(0, 10, 4)
        s = bin_statistic_3d((x, edges), (y, edges), (z, edges), v)
        assert s.count.shape == (3, 3, 3)
        assert s.count.sum() == N


# ─────────────────────────────────────────────────────────────────────── #
# axis_alignment                                                            #
# ─────────────────────────────────────────────────────────────────────── #

class TestAxisAlignment:
    def test_rot_z_preserves_norm(self):
        rng = np.random.default_rng(0)
        mv = rng.standard_normal((50, 3))
        rz, rmat, theta = rot_z_axis(mv)
        np.testing.assert_allclose(
            np.linalg.norm(rz,  axis=1),
            np.linalg.norm(mv, axis=1),
            atol=1e-10,
        )

    def test_rot_y_preserves_norm(self):
        rng = np.random.default_rng(0)
        mv = rng.standard_normal((50, 3))
        ry, rmat, theta = rot_y_axis(mv)
        np.testing.assert_allclose(
            np.linalg.norm(ry, axis=1),
            np.linalg.norm(mv, axis=1),
            atol=1e-10,
        )

    def test_rot_z_shape_validation(self):
        with pytest.raises(ValueError, match="must be \\(T, 3\\)"):
            rot_z_axis(np.zeros((10, 2)))

    def test_detect_roll_recovers_known_angle(self):
        T = 5
        n_markers = 4
        roll_in = np.array([0.0, 0.5, -0.3, np.pi / 4, np.pi / 2])
        body = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 1]],
                        dtype=float)
        mv = np.zeros((T, n_markers, 3))
        for t in range(T):
            c, s = np.cos(roll_in[t]), np.sin(roll_in[t])
            R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
            mv[t] = body @ R.T
        out, ang = detect_roll(mv)
        np.testing.assert_allclose(ang, roll_in, atol=1e-12)

    def test_detect_roll_shape_validation(self):
        with pytest.raises(ValueError, match="must be \\(T, n_markers, 3\\)"):
            detect_roll(np.zeros((10, 4, 2)))


# ─────────────────────────────────────────────────────────────────────── #
# Quaternion conversions                                                    #
# ─────────────────────────────────────────────────────────────────────── #

class TestQuaternions:
    def test_identity_quaternion(self):
        q = np.array([[1.0, 0.0, 0.0, 0.0]])
        rotm = quat2rotm(q)
        np.testing.assert_allclose(rotm[0], np.eye(3), atol=1e-12)

    def test_90deg_z_rotation(self):
        # Quaternion for 90° rotation about z: (cos(45°), 0, 0, sin(45°))
        c, s = np.cos(np.pi / 4), np.sin(np.pi / 4)
        q = np.array([[c, 0.0, 0.0, s]])
        rotm = quat2rotm(q)
        expected = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        np.testing.assert_allclose(rotm[0], expected, atol=1e-12)

    def test_quaternion2rad_identity(self):
        q = np.array([[1.0, 0.0, 0.0, 0.0]])
        eAng = quaternion2rad(q)
        np.testing.assert_allclose(eAng[0], [0.0, 0.0, 0.0], atol=1e-12)

    def test_shape_validation(self):
        with pytest.raises(ValueError, match="must be \\(N, 4\\)"):
            quat2rotm(np.zeros((10, 3)))


# ─────────────────────────────────────────────────────────────────────── #
# thetarc_phase                                                              #
# ─────────────────────────────────────────────────────────────────────── #

class TestThetarcPhase:
    def test_two_channel_required(self):
        with pytest.raises(ValueError, match="\\(T, 2\\)"):
            thetarc_phase(np.zeros((100, 3)), samplerate=1250.0)

    def test_runs_on_synthetic(self):
        fs = 1250.0
        t = np.arange(int(2 * fs)) / fs
        ch1 = np.cos(2 * np.pi * 8 * t)
        ch2 = np.cos(2 * np.pi * 8 * t + 0.5)
        lfp = np.column_stack([ch1, ch2])
        phase = thetarc_phase(lfp, samplerate=fs)
        assert phase.shape == (lfp.shape[0],)
        assert phase.min() >= 0.0
        assert phase.max() < 2 * np.pi


# ─────────────────────────────────────────────────────────────────────── #
# misc transforms                                                            #
# ─────────────────────────────────────────────────────────────────────── #

class TestMakeUniformDistr:
    def test_uniform_input_unchanged(self):
        # Sorted uniform on [0, 1] should map to itself approximately
        x = np.linspace(0, 1, 100)
        out, _ = make_uniform_distr(x, a=0, b=1)
        np.testing.assert_allclose(out, x, atol=1e-2)

    def test_range_scaling(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        out, _ = make_uniform_distr(x, a=10, b=20)
        # Output is 5 evenly-spaced rank-fractions in [10, 20]
        assert out.min() >= 10.0
        assert out.max() <= 20.0


class TestShilbert:
    def test_real_input(self):
        x = np.cos(2 * np.pi * np.arange(100) / 25.0)
        h = shilbert(x)
        assert np.iscomplexobj(h)
        assert h.shape == x.shape


class TestMyThetaPhase:
    def test_runs_on_synthetic(self):
        fs = 1250.0
        t = np.arange(int(2 * fs)) / fs
        eeg = np.cos(2 * np.pi * 7 * t)
        phase, amp, tot, eegf = my_theta_phase(eeg, samplerate=fs)
        assert phase.shape == eeg.shape
        assert amp.shape == eeg.shape
        assert tot.shape == eeg.shape
        assert eegf.shape == eeg.shape


# ─────────────────────────────────────────────────────────────────────── #
# rect_filter and gauss_window                                              #
# ─────────────────────────────────────────────────────────────────────── #

class TestRectFilter:
    def test_constant_input_unchanged(self):
        x = np.ones(50)
        y = rect_filter(x, order=5, num_applications=3)
        np.testing.assert_allclose(y, 1.0, atol=1e-10)

    def test_linear_smooths_step(self):
        x = np.zeros(50); x[20:] = 10.0
        y = rect_filter(x, order=3, num_applications=1)
        # Values around the step are intermediate, end values still 0/10
        assert y[0]  < 1.0
        assert y[-1] > 9.0

    def test_validates_order(self):
        with pytest.raises(ValueError, match="order"):
            rect_filter(np.zeros(10), order=0)

    def test_validates_data_type(self):
        with pytest.raises(ValueError, match="data_type"):
            rect_filter(np.zeros(10), data_type="bogus")


class TestGaussWindow:
    def test_unit_area(self):
        w = gauss_window(0.1, 250.0)
        np.testing.assert_allclose(w.sum(), 1.0, atol=1e-12)

    def test_odd_length(self):
        w = gauss_window(0.1, 250.0)
        assert len(w) % 2 == 1


# ─────────────────────────────────────────────────────────────────────── #
# marker-diff helpers                                                        #
# ─────────────────────────────────────────────────────────────────────── #

class TestMarkerDiffHelpers:
    def _build_xyz(self):
        from neurobox.dtype import NBDxyz, NBModel
        T = 50
        markers = ["m0", "m1", "m2", "m3"]
        data = np.zeros((T, 4, 3))
        # Fixed positions
        data[:, 0] = [-10, 0, 60]
        data[:, 1] = [  0, 10, 60]
        data[:, 2] = [ 10, 0, 60]
        data[:, 3] = [  0, -10, 60]
        return NBDxyz(data, model=NBModel(markers=markers), samplerate=120.0)

    def test_marker_diff_matrix_shape(self):
        xyz = self._build_xyz()
        d = marker_diff_matrix(xyz)
        assert d.shape == (50, 4, 4, 3)
        # diff[t, i, i] = 0
        np.testing.assert_array_equal(d[:, 0, 0, :], 0.0)

    def test_inter_marker_distance_symmetric(self):
        xyz = self._build_xyz()
        d = inter_marker_distance(xyz)
        assert d.shape == (50, 4, 4)
        # Symmetric and zero on diagonal
        np.testing.assert_allclose(d, np.transpose(d, (0, 2, 1)),
                                    atol=1e-10)
        np.testing.assert_array_equal(np.diagonal(d, axis1=1, axis2=2), 0)

    def test_inter_marker_angles_diagonal_undefined(self):
        # angle at vertex i with j=i and k=i is undefined; should not crash
        xyz = self._build_xyz()
        a = inter_marker_angles(xyz)
        assert a.shape == (50, 4, 4, 4)


# ─────────────────────────────────────────────────────────────────────── #
# decompose_xy_motion_wrt_body                                              #
# ─────────────────────────────────────────────────────────────────────── #

class TestDecomposeXyMotion:
    def _build_aug_xyz(self, T=400, fs=120.0):
        from neurobox.dtype import NBDxyz, NBModel
        from neurobox.analysis.kinematics import augment_xyz
        markers = ["spine_lower", "pelvis_root", "spine_middle", "spine_upper",
                   "head_back", "head_left", "head_front", "head_right"]
        t = np.arange(T) / fs
        heading = t * 0.3
        body_x = 200 * np.cos(t * 0.3)
        body_y = 200 * np.sin(t * 0.3)
        offsets = {
            "spine_lower": (-30, 0, 50),
            "pelvis_root": (-15, 0, 50),
            "spine_middle": (0, 0, 50),
            "spine_upper": (15, 0, 50),
            "head_back": (25, 0, 60),
            "head_left": (30, 5, 60),
            "head_front": (35, 0, 60),
            "head_right": (30, -5, 60),
        }
        data = np.zeros((T, 8, 3))
        for i, m in enumerate(markers):
            ox, oy, oz = offsets[m]
            cy, sy = np.cos(heading), np.sin(heading)
            data[:, i, 0] = body_x + ox * cy - oy * sy
            data[:, i, 1] = body_y + ox * sy + oy * cy
            data[:, i, 2] = oz
        xyz = NBDxyz(data, model=NBModel(markers=markers), samplerate=fs)
        return augment_xyz(xyz)

    def test_compute_then_run_match(self):
        aug = self._build_aug_xyz()
        mask = np.zeros(aug.data.shape[0], dtype=bool)
        mask[80:320] = True
        proj1, model = decompose_xy_motion_wrt_body(
            aug, mode="COMPUTE", train_mask=mask,
            new_samplerate=None, window_length_s=0.2,
        )
        proj2, _ = decompose_xy_motion_wrt_body(
            aug, mode="RUN", model=model,
            new_samplerate=None, window_length_s=0.2,
        )
        np.testing.assert_allclose(proj1, proj2, atol=1e-10)

    def test_run_mode_requires_model(self):
        aug = self._build_aug_xyz()
        with pytest.raises(ValueError, match="requires a fitted model"):
            decompose_xy_motion_wrt_body(aug, mode="RUN")

    def test_compute_mode_requires_train_mask(self):
        aug = self._build_aug_xyz()
        with pytest.raises(ValueError, match="train_mask"):
            decompose_xy_motion_wrt_body(
                aug, mode="COMPUTE", new_samplerate=None,
                window_length_s=0.2,
            )


# ─────────────────────────────────────────────────────────────────────── #
# stc_utils — ported from MTA classifiers/utilities + utilities/             #
# ─────────────────────────────────────────────────────────────────────── #

# These tests exercise the lazy-loaded classifier path, so we need to
# import torch/sklearn-gated dependencies via the analysis namespace.

@pytest.fixture
def two_stcs():
    """Two state collections sharing 'walk' and 'rear' labels."""
    from neurobox.dtype import NBStateCollection, NBEpoch
    stc1 = NBStateCollection()
    stc1.add_state(NBEpoch(
        data=np.array([[0.0, 1.0], [3.0, 4.0]], dtype=float),
        samplerate=1.0, label="walk", key="w",
    ))
    stc1.add_state(NBEpoch(
        data=np.array([[1.0, 2.0]], dtype=float),
        samplerate=1.0, label="rear", key="r",
    ))
    stc2 = NBStateCollection()
    stc2.add_state(NBEpoch(
        data=np.array([[0.5, 1.5], [3.5, 4.5]], dtype=float),
        samplerate=1.0, label="walk", key="w",
    ))
    stc2.add_state(NBEpoch(
        data=np.array([[1.0, 2.0]], dtype=float),
        samplerate=1.0, label="rear", key="r",
    ))
    return stc1, stc2


class TestStcUtils:
    def test_mat_to_stc_roundtrip(self):
        from neurobox.analysis.classifiers import mat_to_stc
        from neurobox.analysis.decoding import stc2mat
        T = 100
        smat = np.zeros((T, 2), dtype=int)
        smat[10:30, 0] = 1
        smat[40:60, 1] = 2
        stc_new = mat_to_stc(smat, ["walk", "rear"], samplerate=10.0)
        # Re-build matrix from stc and check shape + boolean equality
        re_mat, _ = stc2mat(stc_new, T, 10.0, states=["walk", "rear"])
        np.testing.assert_array_equal(
            (re_mat != 0), (smat != 0),
        )

    def test_confusion_matrix_basic(self, two_stcs):
        from neurobox.analysis.classifiers import confusion_matrix
        stc1, stc2 = two_stcs
        cm, labels = confusion_matrix(stc1, stc2, n_samples=500,
                                       samplerate=100.0)
        assert cm.shape == (2, 2)
        assert sorted(labels) == ["rear", "walk"]
        # The 'rear' diagonal entry should equal the count where both
        # collections agree on rear (1.0-2.0s = 100 samples)
        rear_idx = labels.index("rear")
        assert cm[rear_idx, rear_idx] == 100

    def test_compare_stcs(self, two_stcs):
        from neurobox.analysis.classifiers import compare_stcs
        stc1, stc2 = two_stcs
        stats = compare_stcs(stc1, stc2, n_samples=500, samplerate=100.0)
        assert stats.confusion_matrix.shape == (2, 2)
        assert 0.0 <= stats.accuracy <= 1.0

    def test_swap_state_vector_ids(self):
        from neurobox.analysis.classifiers import swap_state_vector_ids
        v = np.array([1, 2, 3, 1, 2, 3])
        out = swap_state_vector_ids(v, 1, 3)
        np.testing.assert_array_equal(out, [3, 2, 1, 3, 2, 1])
        # Original unchanged
        np.testing.assert_array_equal(v, [1, 2, 3, 1, 2, 3])

    def test_reassign_short_periods(self):
        from neurobox.analysis.classifiers import reassign_short_periods
        T = 30
        smat = np.zeros((T, 3), dtype=int)
        smat[:, 0] = 1                # state 0 active everywhere
        smat[10:14, 0] = 0
        smat[10:14, 1] = 1            # 4-sample run of state 1
        # Threshold 5 → reassign
        out = reassign_short_periods(smat, target_state_col=1,
                                      duration_samples=5)
        # State 1 should be empty after reassignment
        assert (out[:, 1] != 0).sum() == 0

    def test_reduce_stc_to_loc(self):
        from neurobox.dtype import NBStateCollection, NBEpoch
        from neurobox.analysis.classifiers import reduce_stc_to_loc
        stc = NBStateCollection()
        stc.add_state(NBEpoch(data=np.array([[0.0, 1.0]]),
                              samplerate=1.0, label="walk", key="w"))
        stc.add_state(NBEpoch(data=np.array([[2.0, 3.0]]),
                              samplerate=1.0, label="turn", key="n"))
        out = reduce_stc_to_loc(stc)
        assert out.has_state("loc")

    def test_mutual_information_runs(self):
        from neurobox.dtype import NBStateCollection, NBEpoch
        from neurobox.analysis.classifiers import (
            mutual_information_states_features,
        )
        stc = NBStateCollection()
        stc.add_state(NBEpoch(data=np.array([[0.0, 5.0]]),
                              samplerate=1.0, label="walk", key="w"))
        stc.add_state(NBEpoch(data=np.array([[5.0, 10.0]]),
                              samplerate=1.0, label="rear", key="r"))
        T = 1000
        # Construct features: column 0 informative (different mean per state),
        # column 1 noise.
        rng = np.random.default_rng(0)
        features = rng.standard_normal((T, 2))
        features[:T // 2, 0] += 5
        mi, ranges = mutual_information_states_features(
            stc, features, feature_samplerate=100.0,
            states=["walk", "rear"], n_bins=32,
        )
        assert mi.shape == (3, 2)              # row 0 + 2 hold-outs
        # Informative feature has higher MI than noise feature
        assert mi[0, 0] > mi[0, 1]


# ─────────────────────────────────────────────────────────────────────── #
# data_hash                                                                  #
# ─────────────────────────────────────────────────────────────────────── #

class TestDataHash:
    def test_deterministic(self):
        a = data_hash({"a": 1, "b": [1, 2, 3]})
        b = data_hash({"b": [1, 2, 3], "a": 1})    # different insert order
        assert a == b

    def test_different_for_different_data(self):
        a = data_hash({"a": 1})
        b = data_hash({"a": 2})
        assert a != b

    def test_md5_matches_matlab_length(self):
        h = data_hash({"x": 1}, algorithm="md5")
        assert len(h) == 32

    def test_numpy_array(self):
        a = data_hash(np.zeros(10))
        b = data_hash(np.zeros(10))
        assert a == b
        c = data_hash(np.ones(10))
        assert a != c
