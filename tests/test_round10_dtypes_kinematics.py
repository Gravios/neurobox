"""Tests for round-10 additions to NBDxyz / NBDfet / kinematics helpers."""

from __future__ import annotations

import numpy as np
import pytest

from neurobox.dtype import NBDxyz, NBDfet, NBModel
from neurobox.analysis.kinematics import (
    augment_xyz, finite_nonzero_mask, zscore_with_mask,
)


# ─────────────────────────────────────────────────────────────────────────── #
# NBDxyz.add_marker                                                            #
# ─────────────────────────────────────────────────────────────────────────── #

def _basic_xyz(n_t: int = 50, seed: int = 0) -> NBDxyz:
    rng = np.random.default_rng(seed)
    markers = ["spine_lower", "head_front", "head_back"]
    data = rng.standard_normal((n_t, len(markers), 3)) * 10.0
    return NBDxyz(
        data, model=NBModel(markers=markers, connections=[["head_front", "head_back"]]),
        samplerate=120.0,
    )


class TestAddMarker:
    def test_append_simple(self):
        xyz = _basic_xyz()
        new_data = np.full((50, 3), 7.0)
        out = xyz.add_marker("test_marker", new_data)
        assert "test_marker" in out.model.markers
        assert out.data.shape == (50, 4, 3)
        assert np.allclose(out.data[:, 3, :], 7.0)

    def test_original_unmodified(self):
        xyz = _basic_xyz()
        new_data = np.zeros((50, 3))
        out = xyz.add_marker("v", new_data)
        assert "v" not in xyz.model.markers
        assert xyz.data.shape == (50, 3, 3)
        assert out is not xyz

    def test_connections_added(self):
        xyz = _basic_xyz()
        out = xyz.add_marker(
            "tip", np.zeros((50, 3)),
            connections=[("tip", "head_front")],
        )
        assert ["tip", "head_front"] in out.model.connections

    def test_connection_to_unknown_marker_raises(self):
        xyz = _basic_xyz()
        with pytest.raises(ValueError, match="references a marker"):
            xyz.add_marker(
                "tip", np.zeros((50, 3)),
                connections=[("tip", "no_such_marker")],
            )

    def test_overwrite_replaces_values(self):
        xyz = _basic_xyz()
        first = xyz.add_marker("v", np.zeros((50, 3)))
        second = first.add_marker("v", np.ones((50, 3)), overwrite=True)
        assert np.allclose(second.data[:, first.model.index("v"), :], 1.0)
        # Marker count unchanged
        assert len(second.model.markers) == len(first.model.markers)

    def test_duplicate_without_overwrite_raises(self):
        xyz = _basic_xyz()
        first = xyz.add_marker("v", np.zeros((50, 3)))
        with pytest.raises(ValueError, match="already exists"):
            first.add_marker("v", np.ones((50, 3)))

    def test_wrong_data_shape_raises(self):
        xyz = _basic_xyz()
        with pytest.raises(ValueError, match="must have shape"):
            xyz.add_marker("v", np.zeros((40, 3)))      # wrong T

    def test_3d_data_with_singleton_axis(self):
        """add_marker accepts (T, 1, n_dims) with squeeze on the marker axis."""
        xyz = _basic_xyz()
        data_3d = np.full((50, 1, 3), 4.0)
        out = xyz.add_marker("v", data_3d)
        assert np.allclose(out.data[:, out.model.index("v"), :], 4.0)


# ─────────────────────────────────────────────────────────────────────────── #
# NBDfet                                                                       #
# ─────────────────────────────────────────────────────────────────────────── #

class TestNBDfet:
    def test_basic_construction(self):
        f = NBDfet(
            data=np.zeros((100, 3)),
            columns=["a", "b", "c"],
            samplerate=120.0,
            label="test",
        )
        assert f.shape == (100, 3)
        assert f.columns == ["a", "b", "c"]
        assert f.samplerate == 120.0
        assert f.label == "test"
        assert f.n_features == 3

    def test_default_columns_are_indices(self):
        f = NBDfet(data=np.zeros((10, 4)))
        assert f.columns == ["0", "1", "2", "3"]

    def test_index_lookup(self):
        f = NBDfet(data=np.zeros((10, 3)), columns=["x", "y", "z"])
        assert f.index("y") == 1
        assert f.indices(["z", "x"]) == [2, 0]

    def test_index_missing_raises(self):
        f = NBDfet(data=np.zeros((10, 3)), columns=["x", "y", "z"])
        with pytest.raises(KeyError, match="not in NBDfet"):
            f.index("w")

    def test_sel_single_column(self):
        data = np.arange(30, dtype=np.float64).reshape(10, 3)
        f = NBDfet(data=data, columns=["x", "y", "z"])
        out = f.sel("y")
        assert out.shape == (10,)
        np.testing.assert_array_equal(out, data[:, 1])

    def test_sel_multiple_columns(self):
        data = np.arange(30, dtype=np.float64).reshape(10, 3)
        f = NBDfet(data=data, columns=["x", "y", "z"])
        out = f.sel(["z", "x"])
        np.testing.assert_array_equal(out[:, 0], data[:, 2])
        np.testing.assert_array_equal(out[:, 1], data[:, 0])

    def test_getitem_str(self):
        f = NBDfet(data=np.eye(3), columns=["x", "y", "z"])
        np.testing.assert_array_equal(f["y"], [0.0, 1.0, 0.0])

    def test_getitem_list(self):
        data = np.arange(30, dtype=np.float64).reshape(10, 3)
        f = NBDfet(data=data, columns=["x", "y", "z"])
        out = f[["x", "y"]]
        np.testing.assert_array_equal(out[:, 0], data[:, 0])

    def test_columns_count_mismatch_raises(self):
        with pytest.raises(ValueError, match="columns has"):
            NBDfet(data=np.zeros((10, 3)), columns=["x", "y"])

    def test_titles_count_mismatch_raises(self):
        with pytest.raises(ValueError, match="titles has"):
            NBDfet(
                data=np.zeros((10, 3)), columns=["x", "y", "z"],
                titles=["X", "Y"],
            )


# ─────────────────────────────────────────────────────────────────────────── #
# Helpers — finite_nonzero_mask + zscore_with_mask                             #
# ─────────────────────────────────────────────────────────────────────────── #

class TestFiniteNonzeroMask:
    def test_basic_1d(self):
        a = np.array([1.0, 0.0, np.nan, 5.0, np.inf, 3.0])
        np.testing.assert_array_equal(
            finite_nonzero_mask(a),
            [True, False, False, True, False, True],
        )

    def test_2d_requires_all_columns_valid(self):
        a = np.array([
            [1.0, 2.0],
            [3.0, 0.0],     # zero in column 1 → invalid
            [np.nan, 5.0],  # nan → invalid
            [7.0, 8.0],
        ])
        np.testing.assert_array_equal(
            finite_nonzero_mask(a),
            [True, False, False, True],
        )

    def test_3d(self):
        a = np.zeros((4, 2, 3))
        a[0] = 1.0
        a[2, 1, 1] = np.nan
        np.testing.assert_array_equal(
            finite_nonzero_mask(a),
            [True, False, False, False],
        )


class TestZscoreWithMask:
    def test_basic_normalisation(self):
        # Column with mean=3, std=sqrt(2)
        a = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        z, mean, std = zscore_with_mask(a)
        np.testing.assert_allclose(mean, [3.0])
        np.testing.assert_allclose(std, [np.sqrt(2)])
        np.testing.assert_allclose(z[0, 0], -np.sqrt(2))
        np.testing.assert_allclose(z[-1, 0], +np.sqrt(2))

    def test_invalid_rows_get_filled(self):
        a = np.array([[1.0], [np.nan], [3.0], [5.0]])
        z, mean, std = zscore_with_mask(a, fill_value=-99.0)
        assert z[1, 0] == -99.0
        # Other rows are normalised based on valid samples only
        valid_mean = np.mean([1.0, 3.0, 5.0])
        assert abs(mean.item() - valid_mean) < 1e-9

    def test_provided_stats_used(self):
        a = np.array([[1.0], [2.0], [3.0], [4.0]])
        z, mean, std = zscore_with_mask(
            a, mean=np.array([0.0]), std=np.array([1.0]),
        )
        # No subtraction, no division → output equals input
        np.testing.assert_allclose(z[:, 0], a[:, 0])

    def test_zero_std_does_not_divide_by_zero(self):
        a = np.full((10, 1), 5.0)
        z, mean, std = zscore_with_mask(a)
        # Should not raise; std=0 → safe
        assert np.all(np.isfinite(z))


# ─────────────────────────────────────────────────────────────────────────── #
# augment_xyz                                                                  #
# ─────────────────────────────────────────────────────────────────────────── #

class TestAugmentXyz:
    def _full_marker_xyz(self, T=100):
        rng = np.random.default_rng(42)
        markers = [
            "spine_lower", "pelvis_root", "spine_middle", "spine_upper",
            "head_back", "head_left", "head_front", "head_right",
        ]
        data = rng.standard_normal((T, len(markers), 3)) * 50.0
        return NBDxyz(data, model=NBModel(markers=markers), samplerate=120.0)

    def test_adds_all_four_virtual_markers(self):
        xyz = self._full_marker_xyz()
        aug = augment_xyz(xyz)
        for m in ("bcom", "acom", "hcom", "nose"):
            assert m in aug.model.markers

    def test_hcom_is_mean_of_head_markers(self):
        xyz = self._full_marker_xyz()
        aug = augment_xyz(xyz)
        head_idx = [aug.model.index(m) for m in
                    ("head_back", "head_left", "head_front", "head_right")]
        expected = aug.data[:, head_idx, :].mean(axis=1)
        actual = aug.data[:, aug.model.index("hcom"), :]
        np.testing.assert_allclose(actual, expected, rtol=1e-10)

    def test_nose_offset_from_hcom(self):
        xyz = self._full_marker_xyz()
        aug = augment_xyz(xyz, nose_forward_mm=40.0)
        nose = aug.data[:, aug.model.index("nose"), :]
        hcom = aug.data[:, aug.model.index("hcom"), :]
        offset = nose - hcom
        # Default 40 mm offset along the head's local x-axis
        offset_norm = np.linalg.norm(offset, axis=1)
        # Where the head frame is well-defined, the offset should be ~40
        good = offset_norm > 1e-6
        np.testing.assert_allclose(offset_norm[good], 40.0, atol=1e-6)

    def test_skips_when_marker_already_exists(self):
        """If hcom is already there, the virtual-marker step is skipped."""
        xyz = self._full_marker_xyz()
        # Pre-populate hcom with a recognizable value
        xyz_with_hcom = xyz.add_marker("hcom", np.full((100, 3), 999.0))
        aug = augment_xyz(xyz_with_hcom)
        # Should still be 999 — augment_xyz didn't overwrite it
        actual = aug.data[:, aug.model.index("hcom"), :]
        np.testing.assert_allclose(actual, 999.0)

    def test_partial_markers_still_work(self):
        """If only head markers are present, body COMs are skipped."""
        markers = ["head_back", "head_left", "head_front", "head_right"]
        rng = np.random.default_rng(0)
        data = rng.standard_normal((50, 4, 3)) * 50.0
        xyz = NBDxyz(data, model=NBModel(markers=markers), samplerate=120.0)
        aug = augment_xyz(xyz)
        # Body COMs require >= 3 spine markers — should be absent
        assert "bcom" not in aug.model.markers
        # Head COM should be present
        assert "hcom" in aug.model.markers

    def test_nan_replaced_with_eps(self):
        xyz = self._full_marker_xyz()
        # Inject some NaNs
        data = xyz.data.copy()
        data[5, 0, 0] = np.nan
        # Build a fresh xyz with the modified data
        xyz_nan = NBDxyz(data, model=xyz.model, samplerate=120.0)
        aug = augment_xyz(xyz_nan, fill_nan_with_eps=True)
        assert not np.any(np.isnan(aug.data))

    def test_resample(self):
        xyz = self._full_marker_xyz(T=240)
        aug = augment_xyz(xyz, samplerate=60.0)
        assert aug.samplerate == 60.0
        # Approx 60 / 120 * 240 = 120 samples
        assert abs(aug.data.shape[0] - 120) <= 2
