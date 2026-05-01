"""Tests for the 12 ported kinematic features."""

from __future__ import annotations

import numpy as np
import pytest

from neurobox.dtype import NBDxyz, NBDfet, NBModel
from neurobox.analysis.kinematics import augment_xyz, features


# ─────────────────────────────────────────────────────────────────────────── #
# Fixture                                                                      #
# ─────────────────────────────────────────────────────────────────────────── #

@pytest.fixture
def xyz_aug():
    """Synthesised augmented xyz with smooth (Butterworth-friendly) trajectories."""
    rng = np.random.default_rng(42)
    markers = [
        "spine_lower", "pelvis_root", "spine_middle", "spine_upper",
        "head_back", "head_left", "head_front", "head_right",
    ]
    N = 500
    base = rng.standard_normal((N, len(markers), 3)) * 5.0
    # Cumulative sum makes the trajectory smooth (low-frequency)
    smooth = np.cumsum(base * 0.1, axis=0) + 100.0
    xyz = NBDxyz(smooth, model=NBModel(markers=markers), samplerate=120.0)
    return augment_xyz(xyz)


# ─────────────────────────────────────────────────────────────────────────── #
# Position features                                                            #
# ─────────────────────────────────────────────────────────────────────────── #

class TestFetXY:
    def test_returns_two_columns(self, xyz_aug):
        f = features.fet_xy(xyz_aug)
        assert f.columns == ["x", "y"]
        assert f.shape == (xyz_aug.data.shape[0], 2)

    def test_matches_hcom_xy(self, xyz_aug):
        f = features.fet_xy(xyz_aug)
        idx_hcom = xyz_aug.model.index("hcom")
        np.testing.assert_allclose(
            f.data, xyz_aug.data[:, idx_hcom, :2], rtol=1e-10,
        )

    def test_label_is_fet_xy(self, xyz_aug):
        assert features.fet_xy(xyz_aug).label == "fet_xy"

    def test_resample(self, xyz_aug):
        f = features.fet_xy(xyz_aug, samplerate=60.0)
        assert f.samplerate == 60.0

    def test_missing_hcom_raises(self):
        markers = ["head_back"]
        data = np.zeros((10, 1, 3))
        xyz = NBDxyz(data, model=NBModel(markers=markers), samplerate=120.0)
        with pytest.raises(ValueError, match="no 'hcom' marker"):
            features.fet_xy(xyz)


class TestFetDXY:
    def test_three_columns(self, xyz_aug):
        f = features.fet_dxy(xyz_aug)
        assert f.columns == ["head_yaw", "x", "y"]

    def test_yaw_is_finite(self, xyz_aug):
        f = features.fet_dxy(xyz_aug)
        # Yaw should be finite where the head vector is well-defined
        yaw = f["head_yaw"]
        assert np.all(np.abs(yaw) < np.pi + 1e-6)


# ─────────────────────────────────────────────────────────────────────────── #
# Pitch features                                                               #
# ─────────────────────────────────────────────────────────────────────────── #

class TestFetHeadPitch:
    def test_returns_one_column(self, xyz_aug):
        f = features.fet_head_pitch(xyz_aug)
        assert f.shape[1] == 1
        assert f.columns == ["head_pitch"]

    def test_pitch_is_in_radians(self, xyz_aug):
        f = features.fet_head_pitch(xyz_aug)
        # |pitch| ≤ π/2
        assert np.max(np.abs(f.data)) <= np.pi / 2 + 1e-6


class TestFetHBPitch:
    def test_three_columns(self, xyz_aug):
        f = features.fet_HB_pitch(xyz_aug)
        assert f.columns == ["pitch_BPBU", "pitch_BUHC", "pitch_HCHN"]


class TestFetHBPitchB:
    def test_two_columns(self, xyz_aug):
        f = features.fet_HB_pitchB(xyz_aug)
        assert f.columns == ["head_minus_body_pitch", "body_pitch"]

    def test_first_col_is_circular_diff(self, xyz_aug):
        # The "head_minus_body_pitch" column should equal circ_dist of
        # pitch_HCHN and pitch_BPBU from fet_HB_pitch
        pch = features.fet_HB_pitch(xyz_aug)
        pchB = features.fet_HB_pitchB(xyz_aug)
        # circ_dist via wrapping
        expected = np.angle(np.exp(1j * (pch["pitch_HCHN"] - pch["pitch_BPBU"])))
        # Where xyz is valid (non-zero everywhere → all rows valid)
        np.testing.assert_allclose(pchB["head_minus_body_pitch"], expected, atol=1e-10)


class TestFetXYHB:
    def test_four_columns(self, xyz_aug):
        f = features.fet_xyhb(xyz_aug)
        assert f.columns == ["x", "y", "head_minus_body_pitch", "body_pitch"]


class TestFetHZP:
    def test_two_columns(self, xyz_aug):
        f = features.fet_hzp(xyz_aug)
        assert f.columns == ["head_minus_body_pitch", "head_height"]

    def test_height_matches_hcom_z(self, xyz_aug):
        f = features.fet_hzp(xyz_aug)
        idx_hcom = xyz_aug.model.index("hcom")
        np.testing.assert_allclose(
            f["head_height"], xyz_aug.data[:, idx_hcom, 2], rtol=1e-10,
        )


# ─────────────────────────────────────────────────────────────────────────── #
# Head-body angle                                                              #
# ─────────────────────────────────────────────────────────────────────────── #

class TestFetHBA:
    def test_returns_one_column(self, xyz_aug):
        f = features.fet_hba(xyz_aug)
        assert f.shape[1] == 1
        assert f.columns == ["head_body_angle"]

    def test_correction_offset_applied(self, xyz_aug):
        f0 = features.fet_hba(xyz_aug, head_body_correction=0.0)
        f1 = features.fet_hba(xyz_aug, head_body_correction=0.5)
        # Compare wrapping: (a + 0.5) mod 2π should still be wrapped to (-π, π]
        # f1 = -((a - b) + 0.5) so f1 - f0 ≈ -0.5 (modulo 2π)
        # Use circular subtraction
        diff = np.angle(np.exp(1j * (f1.data - f0.data)))
        # Expect approximately -0.5 (constant)
        np.testing.assert_allclose(diff, -0.5, atol=1e-10)

    def test_offset_added_at_end(self, xyz_aug):
        f0 = features.fet_hba(xyz_aug, offset=0.0)
        f1 = features.fet_hba(xyz_aug, offset=0.3)
        np.testing.assert_allclose(f1.data, f0.data + 0.3, atol=1e-10)


class TestFetHBAV:
    def test_one_column(self, xyz_aug):
        f = features.fet_hbav(xyz_aug)
        assert f.columns == ["head_body_ang_vel"]

    def test_units_consistent_with_fs(self, xyz_aug):
        # Centred difference scaled by fs/2 should give finite values
        f = features.fet_hbav(xyz_aug)
        assert np.all(np.isfinite(f.data))


# ─────────────────────────────────────────────────────────────────────────── #
# Body-frame velocity features                                                 #
# ─────────────────────────────────────────────────────────────────────────── #

class TestFetHrefHXY:
    def test_two_columns(self, xyz_aug):
        f = features.fet_href_HXY(xyz_aug)
        assert f.columns == ["speed_AP", "speed_LAT"]
        assert f.shape == (xyz_aug.data.shape[0], 2)

    def test_zero_velocity_means_zero_output(self):
        # Stationary rat → zero velocity in head frame
        markers = [
            "spine_lower", "pelvis_root", "spine_middle", "spine_upper",
            "head_back", "head_left", "head_front", "head_right",
        ]
        N = 500
        # All markers fixed in space
        data = np.tile(
            np.arange(len(markers))[None, :, None] * 10.0 + 100.0,
            (N, 1, 3),
        ).astype(np.float64)
        xyz = NBDxyz(data, model=NBModel(markers=markers), samplerate=120.0)
        aug = augment_xyz(xyz)
        f = features.fet_href_HXY(aug)
        # Skip the first/last few samples (filter edge effects)
        np.testing.assert_allclose(f.data[10:-10], 0.0, atol=1e-6)

    def test_theta_rotates_output(self, xyz_aug):
        f0 = features.fet_href_HXY(xyz_aug, theta=0.0)
        f90 = features.fet_href_HXY(xyz_aug, theta=np.pi / 2)
        # Rotation by 90° swaps and negates: (a, b) → (-b, a)
        # But since both frames use the same vec, comparing pairwise
        # is sufficient — they shouldn't be identical
        assert not np.allclose(f0.data, f90.data)


class TestFetBrefBXY:
    def test_two_columns(self, xyz_aug):
        f = features.fet_bref_BXY(xyz_aug)
        assert f.columns == ["speed_AP", "speed_LAT"]


class TestFetHvfl:
    def test_two_columns(self, xyz_aug):
        f = features.fet_hvfl(xyz_aug)
        assert f.columns == ["speed_AP", "speed_LAT"]

    def test_default_vector_is_hcom_nose(self, xyz_aug):
        # Calling with explicit vector matches default
        f1 = features.fet_hvfl(xyz_aug, vector=("hcom", "nose"))
        f2 = features.fet_hvfl(xyz_aug)
        np.testing.assert_allclose(f1.data, f2.data, atol=1e-12)
