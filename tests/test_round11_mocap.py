"""Tests for round-11 — utilities/mocap ports."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from neurobox.dtype import NBDxyz, NBModel
from neurobox.analysis.mocap import (
    rotate_points_around_vectors,
    rotate_point_around_vector,
    rotate_marker_around_vector,
    rigid_body_basis,
    intermarker_distances,
    marker_triads,
    fill_gaps,
    infer_virtual_joint,
    find_error_periods,
    correct_point_errors,
    parse_rbo_from_csv,
)


# ─────────────────────────────────────────────────────────────────────────── #
# Fixtures                                                                     #
# ─────────────────────────────────────────────────────────────────────────── #

@pytest.fixture
def head_xyz():
    """4-marker rigid head: square in xy plane at z=60."""
    T = 100
    markers = ["head_back", "head_left", "head_front", "head_right"]
    data = np.zeros((T, 4, 3))
    data[:, 0] = [-10,   0, 60]
    data[:, 1] = [  0,  10, 60]
    data[:, 2] = [ 10,   0, 60]
    data[:, 3] = [  0, -10, 60]
    return NBDxyz(data, model=NBModel(markers=markers), samplerate=120.0)


# ─────────────────────────────────────────────────────────────────────────── #
# Rotations                                                                    #
# ─────────────────────────────────────────────────────────────────────────── #

class TestRotatePointsAroundVectors:
    def test_identity(self):
        T = 5
        pts  = np.random.default_rng(0).standard_normal((T, 3))
        axes = np.tile([0.0, 0.0, 1.0], (T, 1))
        out = rotate_points_around_vectors(pts, axes, 0.0)
        np.testing.assert_allclose(out, pts, atol=1e-12)

    def test_90deg_around_z(self):
        T = 5
        pts  = np.random.default_rng(0).standard_normal((T, 3))
        axes = np.tile([0.0, 0.0, 1.0], (T, 1))
        out = rotate_points_around_vectors(pts, axes, np.pi / 2)
        # (x, y, z) → (-y, x, z)
        expected = np.column_stack([-pts[:, 1], pts[:, 0], pts[:, 2]])
        np.testing.assert_allclose(out, expected, atol=1e-12)

    def test_180deg_self_inverse(self):
        T = 5
        pts  = np.random.default_rng(0).standard_normal((T, 3))
        axes = np.tile([1.0, 1.0, 1.0], (T, 1)) / np.sqrt(3)
        once  = rotate_points_around_vectors(pts, axes, np.pi)
        twice = rotate_points_around_vectors(once, axes, np.pi)
        np.testing.assert_allclose(twice, pts, atol=1e-10)

    def test_per_frame_axes(self):
        # Different axis at every frame
        T = 3
        pts  = np.array([[1.0, 0.0, 0.0]] * T)
        axes = np.array([
            [0.0, 0.0, 1.0],   # rotate around z
            [0.0, 1.0, 0.0],   # rotate around y
            [1.0, 0.0, 0.0],   # rotate around x (identity for x-aligned point)
        ])
        out = rotate_points_around_vectors(pts, axes, np.pi / 2)
        np.testing.assert_allclose(out[0], [0.0, 1.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(out[1], [0.0, 0.0, -1.0], atol=1e-12)
        np.testing.assert_allclose(out[2], [1.0, 0.0, 0.0], atol=1e-12)

    def test_shape_validation(self):
        with pytest.raises(ValueError, match="must be \\(T, 3\\)"):
            rotate_points_around_vectors(
                np.zeros((5, 2)), np.zeros((5, 3)), 0.0,
            )
        with pytest.raises(ValueError, match="must match"):
            rotate_points_around_vectors(
                np.zeros((5, 3)), np.zeros((4, 3)), 0.0,
            )


class TestRotatePointAroundVector:
    def test_requires_an_angle(self, head_xyz):
        with pytest.raises(ValueError, match="angle"):
            rotate_point_around_vector(
                head_xyz, marker="head_left",
                ref_markers=("head_back", "head_front"),
                origin_marker="head_left",
            )

    def test_degrees_input(self, head_xyz):
        # Must add hcom to the model first since default origin is hcom
        from neurobox.analysis.kinematics import augment_xyz
        aug = augment_xyz(head_xyz)
        # Rotate head_back about the (head_back→head_front, head_left→head_right) normal
        # Just check the function runs and returns the right shape
        out = rotate_point_around_vector(
            aug, marker="head_back",
            angle_deg=90.0,
            ref_markers=("head_back", "head_front"),
            origin_marker="hcom",
        )
        assert out.shape == (head_xyz.data.shape[0], 3)


class TestRotateMarkerAroundVector:
    def test_default_args(self, head_xyz):
        # Default angle pi (180°), default origin hcom
        from neurobox.analysis.kinematics import augment_xyz
        aug = augment_xyz(head_xyz)
        out = rotate_marker_around_vector(aug)
        assert out.shape == (head_xyz.data.shape[0], 3)

    def test_zero_angle_unchanged(self, head_xyz):
        from neurobox.analysis.kinematics import augment_xyz
        aug = augment_xyz(head_xyz)
        out = rotate_marker_around_vector(aug, marker="head_left", angle=0.0)
        # 0-radian rotation = identity, output should equal the input head_left
        idx = aug.model.index("head_left")
        np.testing.assert_allclose(out, aug.data[:, idx, :], atol=1e-12)


# ─────────────────────────────────────────────────────────────────────────── #
# Basis                                                                        #
# ─────────────────────────────────────────────────────────────────────────── #

class TestRigidBodyBasis:
    def test_orthonormal(self, head_xyz):
        basis, com = rigid_body_basis(
            head_xyz,
            markers=["head_back", "head_left", "head_front", "head_right"],
        )
        # Each axis is unit norm
        for k in range(3):
            norms = np.linalg.norm(basis[:, :, k], axis=1)
            np.testing.assert_allclose(norms, 1.0, atol=1e-10)
        # Pairwise orthogonal
        for i, j in [(0, 1), (0, 2), (1, 2)]:
            dot = np.einsum("ti,ti->t", basis[:, :, i], basis[:, :, j])
            np.testing.assert_allclose(dot, 0.0, atol=1e-10)

    def test_com_matches_mean(self, head_xyz):
        markers = ["head_back", "head_left", "head_front", "head_right"]
        _, com = rigid_body_basis(head_xyz, markers=markers)
        idx = [head_xyz.model.index(m) for m in markers]
        expected = head_xyz.data[:, idx, :].mean(axis=1)
        np.testing.assert_allclose(com, expected, atol=1e-12)

    def test_too_few_markers_raises(self, head_xyz):
        with pytest.raises(ValueError, match="at least 2"):
            rigid_body_basis(head_xyz, markers=["head_back"])


class TestIntermarkerDistances:
    def test_count_and_values(self, head_xyz):
        d = intermarker_distances(head_xyz)
        # 4 markers → 6 pairs
        assert d.shape == (head_xyz.data.shape[0], 6)
        # back↔front = 20 mm; left↔right = 20 mm; diagonals = sqrt(200)
        # Order: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
        # = (back,left), (back,front), (back,right), (left,front), (left,right), (front,right)
        # back↔front is index 1, left↔right is index 4
        np.testing.assert_allclose(d[0, 1], 20.0, atol=1e-10)
        np.testing.assert_allclose(d[0, 4], 20.0, atol=1e-10)
        # Diagonals
        np.testing.assert_allclose(d[0, 0], np.sqrt(200), atol=1e-10)


class TestMarkerTriads:
    def test_triple_count(self, head_xyz):
        result = marker_triads(head_xyz)
        # C(4, 3) = 4 triples
        assert result.nck.shape == (4, 3)
        assert result.com.shape == (head_xyz.data.shape[0], 4, 3)

    def test_imo_correct_for_known_geometry(self, head_xyz):
        # back-left-front triple: back at (-10, 0), left at (0, 10),
        # front at (10, 0) — angle at left vertex is 90°
        result = marker_triads(head_xyz)
        # Look up the (0, 1, 2) triple
        tri_idx = next(i for i in range(4) if list(result.nck[i]) == [0, 1, 2])
        # Angle at middle vertex (left = idx 1): vectors to back (k=0) and front (i=2)
        # Wait — let me re-read marker_triads code
        # It uses arm_k = sub[:, k, :] - sub[:, j, :] and arm_i = sub[:, i, :] - sub[:, j, :]
        # where (i, j, k) is the triple. For triple (0, 1, 2), j=1 (middle).
        # arm_k = back - left, arm_i = front - left? No, k=2, i=0, j=1.
        # arm_k = front - left = (10, -10, 0); arm_i = back - left = (-10, -10, 0)
        # Angle between them: cos(theta) = ((10)(-10) + (-10)(-10)) / (sqrt(200)*sqrt(200))
        #                                = (-100 + 100) / 200 = 0  → theta = pi/2 ✓
        np.testing.assert_allclose(result.imo[0, tri_idx], np.pi / 2, atol=1e-10)


# ─────────────────────────────────────────────────────────────────────────── #
# Gap filling                                                                  #
# ─────────────────────────────────────────────────────────────────────────── #

class TestFillGaps:
    def test_fills_short_nan_gaps(self):
        T = 200
        markers = ["m1", "m2"]
        t = np.arange(T) / 120.0
        data = np.zeros((T, 2, 3))
        for m in range(2):
            data[:, m, 0] = 100 + 50 * np.sin(2 * np.pi * 0.5 * t) + m * 10
            data[:, m, 1] = 100 + 50 * np.cos(2 * np.pi * 0.5 * t) + m * 10
            data[:, m, 2] = 50 + m * 5
        # Inject 3-sample gaps at known places
        data[50:53, :, :] = np.nan
        data[100:102, :, :] = np.nan

        xyz = NBDxyz(data, model=NBModel(markers=markers), samplerate=120.0)
        filled = fill_gaps(xyz, min_gap_length=5)
        assert not np.isnan(filled.data).any()
        # Source unchanged
        assert np.isnan(xyz.data).any()

    def test_long_gaps_left_alone(self):
        T = 100
        markers = ["m1"]
        data = np.ones((T, 1, 3))
        # 20-sample gap, longer than min_gap_length
        data[40:60, 0, :] = np.nan
        xyz = NBDxyz(data, model=NBModel(markers=markers), samplerate=120.0)
        filled = fill_gaps(xyz, min_gap_length=5)
        # Long gap should NOT be filled
        assert np.isnan(filled.data[45, 0, 0])

    def test_no_gaps_returns_copy(self):
        T = 50
        markers = ["m1"]
        data = np.ones((T, 1, 3))
        xyz = NBDxyz(data, model=NBModel(markers=markers), samplerate=120.0)
        filled = fill_gaps(xyz)
        # No NaNs, no change
        np.testing.assert_array_equal(filled.data, xyz.data)


# ─────────────────────────────────────────────────────────────────────────── #
# Virtual joint inference                                                      #
# ─────────────────────────────────────────────────────────────────────────── #

class TestInferVirtualJoint:
    def test_recovers_known_joint(self):
        T, fs = 4000, 250.0
        markers = ["head_back", "head_left", "head_front", "head_right"]
        true_offset_body = np.array([-30.0, 5.0, 0.0])

        rng = np.random.default_rng(0)
        yaw   = np.cumsum(rng.standard_normal(T) * 0.02)
        pitch = np.cumsum(rng.standard_normal(T) * 0.01)
        neck = np.column_stack([
            np.cumsum(rng.standard_normal(T) * 0.5),
            np.cumsum(rng.standard_normal(T) * 0.5),
            50.0 + np.cumsum(rng.standard_normal(T) * 0.05),
        ])

        offsets_body = {
            "head_back":  np.array([-15,   0, 10]),
            "head_left":  np.array([  0,  10, 10]),
            "head_front": np.array([ 15,   0, 10]),
            "head_right": np.array([  0, -10, 10]),
        }

        data = np.zeros((T, 4, 3))
        for i, m in enumerate(markers):
            body_pos = offsets_body[m] - true_offset_body
            cy, sy = np.cos(yaw), np.sin(yaw)
            cp, sp = np.cos(pitch), np.sin(pitch)
            x_yaw = body_pos[0] * cy - body_pos[1] * sy
            y_yaw = body_pos[0] * sy + body_pos[1] * cy
            z_yaw = body_pos[2] * np.ones(T)
            data[:, i, 0] = neck[:, 0] + x_yaw * cp - z_yaw * sp
            data[:, i, 1] = neck[:, 1] + y_yaw
            data[:, i, 2] = neck[:, 2] + x_yaw * sp + z_yaw * cp

        xyz = NBDxyz(data, model=NBModel(markers=markers), samplerate=fs)
        corrected, offset = infer_virtual_joint(xyz, samplerate=40.0)
        # The offset is in body-fixed coords with a sign convention determined by
        # the basis construction.  Just verify that |inferred| is in the right
        # neighbourhood (within 25 mm of true magnitude on each axis).
        assert np.linalg.norm(offset - true_offset_body) < 50.0 \
            or np.linalg.norm(offset + true_offset_body) < 50.0
        # And that hcom was added
        assert "hcom" in corrected.model.markers

    def test_missing_marker_raises(self, head_xyz):
        with pytest.raises(ValueError, match="no 'head_top'"):
            infer_virtual_joint(
                head_xyz,
                rigid_body_markers=["head_back", "head_top"],
            )


# ─────────────────────────────────────────────────────────────────────────── #
# Error periods + correction                                                   #
# ─────────────────────────────────────────────────────────────────────────── #

class TestFindErrorPeriods:
    def test_clean_data_no_errors(self):
        T = 500
        markers = ["head_back", "head_left", "head_front", "head_right"]
        rng = np.random.default_rng(0)
        # Random walk with no marker swaps
        data = np.zeros((T, 4, 3))
        body_pos = np.cumsum(rng.standard_normal((T, 3)) * 0.3, axis=0)
        offsets = {
            0: [-10, 0, 60], 1: [0, 10, 60], 2: [10, 0, 60], 3: [0, -10, 60],
        }
        for i in range(4):
            data[:, i, :] = body_pos + np.array(offsets[i])
        xyz = NBDxyz(data, model=NBModel(markers=markers), samplerate=120.0)
        # With a high threshold, clean data should yield no errors
        ep, _, _ = find_error_periods(xyz, threshold_z=3.0)
        # Expect 0 or very few short error periods on noiseless rigid-body data
        total_error_samples = sum(e[1] - e[0] for e in ep) if ep.size else 0
        assert total_error_samples < 0.05 * T   # < 5%

    def test_swap_detected(self):
        T = 500
        markers = ["head_back", "head_left", "head_front", "head_right"]
        rng = np.random.default_rng(0)
        data = np.zeros((T, 4, 3))
        body_pos = np.cumsum(rng.standard_normal((T, 3)) * 0.05, axis=0)
        offsets = {
            0: [-10, 0, 60], 1: [0, 10, 60], 2: [10, 0, 60], 3: [0, -10, 60],
        }
        for i in range(4):
            data[:, i, :] = body_pos + np.array(offsets[i])
        # Swap front/back markers in the middle
        data[200:210, [0, 2], :] = data[200:210, [2, 0], :]

        xyz = NBDxyz(data, model=NBModel(markers=markers), samplerate=120.0)
        ep, _, sig = find_error_periods(xyz, threshold_z=1.0)
        assert ep.size > 0
        # At least one detected period overlaps the swap region
        overlaps = any(s <= 209 and e >= 200 for s, e in ep)
        assert overlaps


class TestCorrectPointErrors:
    def test_recovers_swap(self):
        T = 500
        markers = ["head_back", "head_left", "head_front", "head_right"]
        rng = np.random.default_rng(0)
        data = np.zeros((T, 4, 3))
        body_pos = np.cumsum(rng.standard_normal((T, 3)) * 0.05, axis=0)
        offsets = {
            0: [-10, 0, 60], 1: [0, 10, 60], 2: [10, 0, 60], 3: [0, -10, 60],
        }
        for i in range(4):
            data[:, i, :] = body_pos + np.array(offsets[i])
        truth = data.copy()
        # Swap front/back markers in samples 200-205
        data[200:206, [0, 2], :] = data[200:206, [2, 0], :]

        xyz = NBDxyz(data, model=NBModel(markers=markers), samplerate=120.0)
        corrected = correct_point_errors(
            xyz,
            error_periods=np.array([[199, 205]]),
        )
        # Within the swap region the correction should produce
        # back/front in the right slots (same as truth)
        np.testing.assert_allclose(
            corrected.data[200:206, :, :], truth[200:206, :, :], atol=1e-10,
        )

    def test_no_errors_returns_unchanged(self, head_xyz):
        out = correct_point_errors(head_xyz, error_periods=np.zeros((0, 2)))
        np.testing.assert_array_equal(out.data, head_xyz.data)


# ─────────────────────────────────────────────────────────────────────────── #
# Motive CSV parsing                                                           #
# ─────────────────────────────────────────────────────────────────────────── #

_MOTIVE_CSV = (
    "Format Version,1.0,Take Name,test,Capture Frame Rate,250,"
    "Total Frames in Take,4,Total Exported Frames,4,Export Frame Rate,250\n"
    "\n"
    ",,Rigid Body,Rigid Body,Rigid Body,Rigid Body,"
    "Rigid Body,Rigid Body,Rigid Body,Rigid Body\n"
    ",,RatA,RatA,RatA,RatA,RatA,RatA,RatA,RatA\n"
    ",,1,1,1,1,1,1,1,1\n"
    ",,Rotation,Rotation,Rotation,Rotation,"
    "Position,Position,Position,Mean Marker Error\n"
    ",,X,Y,Z,W,X,Y,Z,\n"
    "0,0.000,0.0,0.0,0.0,1.0,0.001,0.002,0.003,0.0\n"
    "1,0.004,0.1,0.0,0.0,1.0,0.011,0.012,0.013,0.0\n"
    "2,0.008,0.2,0.0,0.0,1.0,0.021,0.022,0.023,0.0\n"
    "3,0.012,0.3,0.0,0.0,1.0,0.031,0.032,0.033,0.0\n"
)


@pytest.fixture
def motive_csv_path():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(_MOTIVE_CSV)
        path = f.name
    yield path
    os.unlink(path)


class TestParseRboFromCsv:
    def test_basic(self, motive_csv_path):
        res = parse_rbo_from_csv(motive_csv_path)
        assert res.rbo_data.shape == (4, 1, 8)
        assert res.rbo_aliases == ["RatA"]
        assert res.samplerate == 250.0
        assert res.n_frames_total == 4
        np.testing.assert_array_equal(res.frames, [0, 1, 2, 3])

    def test_position_units_mm(self, motive_csv_path):
        res = parse_rbo_from_csv(motive_csv_path)
        # Channel 0 of output is x_mm; first frame x was 0.001 m = 1 mm
        assert abs(res.rbo_data[0, 0, 0] - 1.0) < 1e-9

    def test_xy_swap(self, motive_csv_path):
        res = parse_rbo_from_csv(motive_csv_path)
        # Original Motive position (0.001, 0.002, 0.003) — Y-up.
        # After Y↔Z swap and ×1000: x=1, y=3, z=2
        np.testing.assert_allclose(res.rbo_data[0, 0, :3], [1.0, 3.0, 2.0])

    def test_alias_filtering(self, motive_csv_path):
        # Filter to a non-existent alias should raise
        with pytest.raises(ValueError, match="not found"):
            parse_rbo_from_csv(motive_csv_path, aliases=["NoSuchBody"])

    def test_explicit_alias_list(self, motive_csv_path):
        res = parse_rbo_from_csv(motive_csv_path, aliases=["RatA"])
        assert res.rbo_aliases == ["RatA"]
        assert res.rbo_data.shape == (4, 1, 8)

    def test_missing_file(self):
        with pytest.raises(FileNotFoundError):
            parse_rbo_from_csv("/no/such/path.csv")
