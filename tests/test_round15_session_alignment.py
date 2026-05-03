"""Tests for round-15 — map_to_reference_session and the
behavioural-manifold offset-correction algorithm."""

from __future__ import annotations

import numpy as np
import pytest

from neurobox.dtype import NBDxyz, NBModel


def _build_alignment_session(name, feat_offset):
    """Helper: build a synthetic session with a known feature offset."""
    from neurobox.analysis.kinematics import augment_xyz
    rng = np.random.default_rng(hash(name) % 2**32)
    T, fs = 6000, 120.0
    t = np.arange(T) / fs
    body_x = 200 * np.cos(t * 0.3)
    body_y = 200 * np.sin(t * 0.3)
    heading = t * 0.3 + np.pi / 2

    markers = ["spine_lower", "spine_middle", "spine_upper",
               "head_back", "head_left", "head_front", "head_right"]
    offsets = {
        "spine_lower":  (-30,  0, 50),
        "spine_middle": (  0,  0, 50),
        "spine_upper":  ( 15,  0, 50),
        "head_back":    ( 25,  0, 60),
        "head_left":    ( 30,  5, 60),
        "head_front":   ( 35,  0, 60),
        "head_right":   ( 30, -5, 60),
    }
    data = np.zeros((T, len(markers), 3))
    for i, m in enumerate(markers):
        ox, oy, oz = offsets[m]
        cy, sy = np.cos(heading), np.sin(heading)
        data[:, i, 0] = body_x + ox * cy - oy * sy
        data[:, i, 1] = body_y + ox * sy + oy * cy
        data[:, i, 2] = oz
    xyz = NBDxyz(data, model=NBModel(markers=markers), samplerate=fs, name=name)
    aug = augment_xyz(xyz)
    bcom_idx = aug.model.index("bcom")
    hcom_idx = aug.model.index("hcom")
    feat = np.column_stack([
        aug.data[:, bcom_idx, 2] + feat_offset,
        aug.data[:, hcom_idx, 2],
    ])
    active = np.ones(T, dtype=bool)
    return aug, feat, active


class TestSessionAlignment:
    def test_linear_offset_recovery(self):
        from neurobox.analysis.classifiers import map_to_reference_session
        ref_xyz, ref_feat, ref_active = _build_alignment_session("ref", 0.0)
        tgt_xyz, tgt_feat, tgt_active = _build_alignment_session("tgt", 12.5)

        aligned, offsets = map_to_reference_session(
            tgt_feat, tgt_xyz, tgt_active,
            ref_feat, ref_xyz, ref_active,
            minimum_occupancy_seconds=0.01,
            return_offsets=True,
        )
        # Offset on column 0 should be ~12.5
        assert abs(offsets[0] - 12.5) < 0.5
        # Offset on column 1 should be ~0
        assert abs(offsets[1]) < 0.5
        # After alignment, target mean should match reference
        np.testing.assert_allclose(
            aligned.mean(axis=0),
            ref_feat.mean(axis=0),
            atol=0.5,
        )

    def test_circular_offset_recovery(self):
        from neurobox.analysis.classifiers import map_to_reference_session
        ref_xyz, _, ref_active = _build_alignment_session("ref", 0.0)
        tgt_xyz, _, tgt_active = _build_alignment_session("tgt", 0.0)
        # Constant circular feature offset by 0.7 rad
        T = ref_xyz.data.shape[0]
        ref_feat = np.zeros((T, 1))
        tgt_feat = np.full((T, 1), 0.7)

        aligned, offsets = map_to_reference_session(
            tgt_feat, tgt_xyz, tgt_active,
            ref_feat, ref_xyz, ref_active,
            circular_columns=[0],
            linear_columns=[],
            minimum_occupancy_seconds=0.01,
            return_offsets=True,
        )
        assert abs(offsets[0] - 0.7) < 0.05
        assert abs(aligned.mean()) < 0.05

    def test_returns_only_aligned_when_offsets_false(self):
        from neurobox.analysis.classifiers import map_to_reference_session
        ref_xyz, ref_feat, ref_active = _build_alignment_session("ref", 0.0)
        tgt_xyz, tgt_feat, tgt_active = _build_alignment_session("tgt", 5.0)
        out = map_to_reference_session(
            tgt_feat, tgt_xyz, tgt_active,
            ref_feat, ref_xyz, ref_active,
            minimum_occupancy_seconds=0.01,
        )
        # Single-return form
        assert isinstance(out, np.ndarray)
        assert out.shape == tgt_feat.shape

    def test_zero_samples_preserved(self):
        """Zero-valued samples in input should remain zero (matches MATLAB)."""
        from neurobox.analysis.classifiers import map_to_reference_session
        ref_xyz, ref_feat, ref_active = _build_alignment_session("ref", 0.0)
        tgt_xyz, tgt_feat, tgt_active = _build_alignment_session("tgt", 5.0)
        # Zero out some target samples
        tgt_feat = tgt_feat.copy()
        tgt_feat[:100, 0] = 0.0
        out = map_to_reference_session(
            tgt_feat, tgt_xyz, tgt_active,
            ref_feat, ref_xyz, ref_active,
            minimum_occupancy_seconds=0.01,
        )
        np.testing.assert_array_equal(out[:100, 0], 0.0)

    def test_validates_column_overlap(self):
        from neurobox.analysis.classifiers import map_to_reference_session
        ref_xyz, ref_feat, ref_active = _build_alignment_session("ref", 0.0)
        tgt_xyz, tgt_feat, tgt_active = _build_alignment_session("tgt", 5.0)
        with pytest.raises(ValueError, match="both circular and linear"):
            map_to_reference_session(
                tgt_feat, tgt_xyz, tgt_active,
                ref_feat, ref_xyz, ref_active,
                circular_columns=[0],
                linear_columns=[0],
                minimum_occupancy_seconds=0.01,
            )

    def test_behavioural_manifold_stats_shape(self):
        from neurobox.analysis.classifiers import behavioural_manifold_stats
        xyz, feat, active = _build_alignment_session("a", 0.0)
        stats = behavioural_manifold_stats(
            feat, xyz, active_mask=active,
        )
        assert stats.mean.shape == (20, 20, 20, 2)
        assert stats.std.shape  == (20, 20, 20, 2)
        assert stats.count.shape == (20, 20, 20)
        # Must be at least one occupied bin for non-empty feature
        assert stats.count.sum() > 0
