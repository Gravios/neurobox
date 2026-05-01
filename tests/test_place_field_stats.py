"""Tests for neurobox.analysis.spatial.place_field_stats."""

from __future__ import annotations

import numpy as np
import pytest

from neurobox.analysis.spatial import (
    place_field, place_field_stats,
    PlaceFieldResult, Patch, UnitStats,
)


# ─────────────────────────────────────────────────────────────────────────── #
# Synthetic PlaceFieldResult builder                                           #
# ─────────────────────────────────────────────────────────────────────────── #

def _gaussian_field(
    n_x:           int,
    n_y:           int,
    centres:       list[tuple[float, float, float, float]],
    extent:        tuple[float, float] = (-1.0, 1.0),
) -> np.ndarray:
    """Build a (n_x, n_y) rate map as a sum of Gaussians.

    Each centre is ``(cx, cy, sigma, peak)``.
    """
    edges = np.linspace(extent[0], extent[1], n_x + 1)
    centres_x = 0.5 * (edges[:-1] + edges[1:])
    edges_y = np.linspace(extent[0], extent[1], n_y + 1)
    centres_y = 0.5 * (edges_y[:-1] + edges_y[1:])
    xs, ys = np.meshgrid(centres_x, centres_y, indexing='ij')
    out = np.zeros_like(xs, dtype=np.float64)
    for cx, cy, sigma, peak in centres:
        out += peak * np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma ** 2))
    return out


def _make_pf_result(
    n_x: int = 20, n_y: int = 20,
    centres: list = None,
    n_iter: int = 1,
    iter_noise: float = 0.0,
    rng: np.random.Generator = None,
) -> PlaceFieldResult:
    """Wrap a synthetic rate map in a PlaceFieldResult."""
    if centres is None:
        centres = [(0.3, -0.2, 0.15, 30.0)]
    edges_x = np.linspace(-1.0, 1.0, n_x + 1)
    edges_y = np.linspace(-1.0, 1.0, n_y + 1)
    bin_centres = [
        0.5 * (edges_x[:-1] + edges_x[1:]),
        0.5 * (edges_y[:-1] + edges_y[1:]),
    ]
    base = _gaussian_field(n_x, n_y, centres)
    if rng is None:
        rng = np.random.default_rng(0)

    rate_map = np.zeros((n_x, n_y, 1, n_iter))
    for it in range(n_iter):
        noisy = base + rng.standard_normal(base.shape) * iter_noise
        # Mimic occupancy mask: NaN out cells with noise-only signal
        noisy[noisy < 1.0] = np.nan
        rate_map[..., 0, it] = noisy

    return PlaceFieldResult(
        rate_map        = rate_map,
        occupancy       = np.ones((n_x, n_y)),
        spike_count     = np.zeros_like(rate_map),
        occupancy_mask  = ~np.isnan(rate_map[..., 0, 0]),
        bin_edges       = [edges_x, edges_y],
        bin_centres     = bin_centres,
        spatial_info    = np.full((1, n_iter), 1.0),
        sparsity        = np.full((1, n_iter), 0.5),
        mean_rate       = np.full((1, n_iter), 5.0),
        unit_ids        = np.array([1]),
        n_spikes        = np.array([100]),
        samplerate      = 120.0,
        skaggs_correct  = True,
    )


# ─────────────────────────────────────────────────────────────────────────── #
# Basic patch detection                                                        #
# ─────────────────────────────────────────────────────────────────────────── #

class TestPatchDetection:

    def test_single_field_recovered(self):
        pf = _make_pf_result(centres=[(0.3, -0.2, 0.15, 30.0)])
        stats = place_field_stats(pf, max_n_patches=2)
        assert len(stats) == 1
        u = stats[0]
        assert u.unit_id == 1
        assert u.n_patches >= 1
        # Largest patch should sit near the field centre (within 1 bin)
        com = u.patches[0].center_of_mass
        assert abs(com[0] - 0.3) < 0.15, f"COM x = {com[0]}, expected ≈ 0.3"
        assert abs(com[1] - (-0.2)) < 0.15, f"COM y = {com[1]}, expected ≈ -0.2"

    def test_two_fields_sorted_by_area(self):
        # Big field + small field — use 70% percentile so both are above thr
        centres = [(0.3, -0.2, 0.20, 30.0), (-0.4, 0.4, 0.10, 25.0)]
        pf = _make_pf_result(centres=centres)
        stats = place_field_stats(pf, max_n_patches=3, threshold_pct=70.0)
        u = stats[0]
        assert u.n_patches == 2
        # Sorted by area descending
        assert u.patches[0].area >= u.patches[1].area

    def test_max_n_patches_truncates(self):
        # 3 fields, ask for at most 2
        centres = [
            (0.3, -0.2, 0.15, 30.0),
            (-0.4, 0.4, 0.10, 25.0),
            (0.0, 0.0, 0.08, 20.0),
        ]
        pf = _make_pf_result(centres=centres)
        stats = place_field_stats(pf, max_n_patches=2)
        assert stats[0].n_patches == 2

    def test_no_patches_when_threshold_above_peak(self):
        pf = _make_pf_result(centres=[(0.0, 0.0, 0.15, 30.0)])
        # Threshold > peak rate
        stats = place_field_stats(pf, threshold_method="absolute",
                                   threshold_value=100.0)
        assert stats[0].n_patches == 0
        assert np.isnan(stats[0].rate_threshold) or stats[0].rate_threshold == 100.0

    def test_all_nan_rate_map(self):
        pf = _make_pf_result()
        pf.rate_map[:] = np.nan
        stats = place_field_stats(pf)
        assert stats[0].n_patches == 0
        assert np.isnan(stats[0].peak_rate)


# ─────────────────────────────────────────────────────────────────────────── #
# Threshold methods                                                            #
# ─────────────────────────────────────────────────────────────────────────── #

class TestThresholds:

    def test_percentile_default(self):
        pf = _make_pf_result()
        stats = place_field_stats(pf)
        assert stats[0].threshold_method == "percentile"
        # 90th percentile of finite bins
        finite = pf.rate_map[..., 0, 0]
        finite = finite[np.isfinite(finite)]
        expected = np.percentile(finite, 90)
        assert abs(stats[0].rate_threshold - expected) < 1e-9

    def test_halfmax(self):
        pf = _make_pf_result(centres=[(0.0, 0.0, 0.15, 30.0)])
        stats = place_field_stats(pf, threshold_method="halfmax")
        peak = float(np.nanmax(pf.rate_map[..., 0, 0]))
        assert abs(stats[0].rate_threshold - 0.5 * peak) < 1e-9

    def test_absolute(self):
        pf = _make_pf_result()
        stats = place_field_stats(pf, threshold_method="absolute",
                                   threshold_value=10.0)
        assert stats[0].rate_threshold == 10.0

    def test_absolute_requires_value(self):
        pf = _make_pf_result()
        with pytest.raises(ValueError, match="absolute threshold requires"):
            place_field_stats(pf, threshold_method="absolute")

    def test_unknown_method(self):
        pf = _make_pf_result()
        with pytest.raises(ValueError, match="Unknown threshold method"):
            place_field_stats(pf, threshold_method="banana")

    def test_custom_percentile(self):
        pf = _make_pf_result()
        stats = place_field_stats(pf, threshold_pct=50.0)
        finite = pf.rate_map[..., 0, 0]
        finite = finite[np.isfinite(finite)]
        expected = np.percentile(finite, 50)
        assert abs(stats[0].rate_threshold - expected) < 1e-9


# ─────────────────────────────────────────────────────────────────────────── #
# Bootstrap modes                                                              #
# ─────────────────────────────────────────────────────────────────────────── #

class TestBootstrap:

    def test_fixed_patches_mode_iter_arrays_present(self):
        pf = _make_pf_result(n_iter=5, iter_noise=2.0)
        stats = place_field_stats(pf, mode="fixed_patches")
        u = stats[0]
        assert u.n_patches >= 1
        p = u.patches[0]
        assert p.peak_rate_iter is not None
        assert p.peak_rate_iter.shape == (5,)
        assert p.mean_rate_iter is not None
        assert p.com_iter.shape == (5, 2)

    def test_fixed_patches_area_constant(self):
        """In fixed_patches mode, the patch *mask* is fixed, so the
        per-iter area is constant across iters (modulo NaN-masking)."""
        pf = _make_pf_result(n_iter=4, iter_noise=0.5)
        stats = place_field_stats(pf, mode="fixed_patches")
        p = stats[0].patches[0]
        # The number of finite pixels within the fixed mask may differ
        # iter-to-iter, but should be stable for low noise
        finite_areas = p.area_iter[np.isfinite(p.area_iter)]
        if finite_areas.size > 1:
            cv = np.std(finite_areas) / np.mean(finite_areas)
            assert cv < 0.5, f"area unstable across iters: cv={cv:.2f}"

    def test_per_iter_mode_can_have_varying_patches(self):
        """In per_iter mode, peak rate may vary substantially because
        each iter is re-segmented independently."""
        pf = _make_pf_result(n_iter=10, iter_noise=3.0)
        stats = place_field_stats(pf, mode="per_iter")
        p = stats[0].patches[0]
        assert p.peak_rate_iter is not None
        assert p.peak_rate_iter.shape == (10,)

    def test_n_iter_1_no_iter_arrays(self):
        """Without bootstrap (n_iter=1), per-iter arrays stay None."""
        pf = _make_pf_result(n_iter=1)
        stats = place_field_stats(pf)
        p = stats[0].patches[0]
        assert p.peak_rate_iter is None
        assert p.mean_rate_iter is None
        assert p.com_iter is None

    def test_fixed_vs_per_iter_iter0_agreement(self):
        """Iter 0 statistics should be identical between modes (same map,
        same segmentation)."""
        pf = _make_pf_result(n_iter=3, iter_noise=1.0)
        a = place_field_stats(pf, mode="fixed_patches")[0].patches[0]
        b = place_field_stats(pf, mode="per_iter")[0].patches[0]
        np.testing.assert_allclose(a.peak_rate_iter[0], b.peak_rate_iter[0])
        np.testing.assert_allclose(a.area_iter[0],      b.area_iter[0])

    def test_invalid_mode(self):
        pf = _make_pf_result()
        with pytest.raises(ValueError, match="mode must be"):
            place_field_stats(pf, mode="weird")


# ─────────────────────────────────────────────────────────────────────────── #
# Unit selection                                                               #
# ─────────────────────────────────────────────────────────────────────────── #

class TestUnitSelection:

    def _multi_unit(self):
        # Two units, each with one field
        n_x, n_y = 20, 20
        edges = np.linspace(-1, 1, n_x + 1)
        bc = [0.5 * (edges[:-1] + edges[1:])] * 2
        rate_map = np.full((n_x, n_y, 2, 1), np.nan)
        rate_map[..., 0, 0] = _gaussian_field(n_x, n_y, [(0.3, -0.2, 0.15, 30.0)])
        rate_map[..., 1, 0] = _gaussian_field(n_x, n_y, [(-0.5, 0.5, 0.15, 25.0)])
        rate_map[rate_map < 1] = np.nan
        return PlaceFieldResult(
            rate_map = rate_map,
            occupancy = np.ones((n_x, n_y)),
            spike_count = np.zeros_like(rate_map),
            occupancy_mask = ~np.isnan(rate_map[..., 0, 0]),
            bin_edges = [edges, edges],
            bin_centres = bc,
            spatial_info = np.full((2, 1), 1.0),
            sparsity = np.full((2, 1), 0.5),
            mean_rate = np.full((2, 1), 5.0),
            unit_ids = np.array([1, 7]),
            n_spikes = np.array([100, 80]),
            samplerate = 120.0,
            skaggs_correct = True,
        )

    def test_default_returns_all_units(self):
        pf = self._multi_unit()
        stats = place_field_stats(pf)
        assert len(stats) == 2
        assert [s.unit_id for s in stats] == [1, 7]

    def test_scalar_unit_id(self):
        pf = self._multi_unit()
        stats = place_field_stats(pf, units=7)
        assert len(stats) == 1
        assert stats[0].unit_id == 7

    def test_list_of_unit_ids(self):
        pf = self._multi_unit()
        stats = place_field_stats(pf, units=[7])
        assert len(stats) == 1
        assert stats[0].unit_id == 7

    def test_invalid_unit_id_raises(self):
        pf = self._multi_unit()
        with pytest.raises(ValueError, match="not in PlaceFieldResult.unit_ids"):
            place_field_stats(pf, units=99)


# ─────────────────────────────────────────────────────────────────────────── #
# Integration with real place_field()                                          #
# ─────────────────────────────────────────────────────────────────────────── #

class TestIntegration:

    def test_round_trip_with_place_field(self):
        """End-to-end: feed real spike data through place_field, then stats."""
        from neurobox.dtype import NBSpk, NBDxyz, NBModel
        rng = np.random.default_rng(0)
        # 60 s at 120 Hz, slow Lissajous trajectory
        fs = 120.0
        n = int(fs * 60)
        t = np.arange(n) / fs
        x = 400 * np.sin(2 * np.pi * 0.07 * t)
        y = 400 * np.sin(2 * np.pi * 0.11 * t)
        xyz = NBDxyz(np.stack([x, y, np.zeros(n)], axis=1)[:, np.newaxis, :],
                     model=NBModel(["head"]), samplerate=fs)
        # Place cell at (200, 0)
        pos = xyz.data[:, 0, :2]
        rate = 25 * np.exp(-((pos - np.array([200, 0])) ** 2).sum(1) / (2 * 80 ** 2))
        n_per_sample = rng.poisson(rate / fs)
        times = []
        for i, k in enumerate(n_per_sample):
            if k > 0:
                times.extend(t[i] + rng.uniform(0, 1/fs, size=k))
        times = np.sort(times).astype(np.float64)
        spk = NBSpk(
            res = times,
            clu = np.full(len(times), 1, dtype=np.int32),
            map_ = np.array([[1, 1]], dtype=np.int64),
            samplerate = 20000.0,
        )
        from neurobox.analysis.spatial import place_field
        pf = place_field(spk, xyz, units=1, bin_size=50,
                         boundary=[(-500, 500), (-500, 500)],
                         smoothing_sigma=1.5)
        stats = place_field_stats(pf)
        assert stats[0].n_patches >= 1
        # The COM of the largest patch should be near (200, 0) within 150 mm
        com = stats[0].patches[0].center_of_mass
        assert abs(com[0] - 200) < 150, f"COM x={com[0]}"
        assert abs(com[1] -   0) < 150, f"COM y={com[1]}"


# ─────────────────────────────────────────────────────────────────────────── #
# Patch / UnitStats containers                                                 #
# ─────────────────────────────────────────────────────────────────────────── #

class TestContainers:

    def test_patch_n_pixels(self):
        pf = _make_pf_result()
        stats = place_field_stats(pf)
        p = stats[0].patches[0]
        assert p.n_pixels == p.pixel_indices.shape[0]
        assert p.n_pixels == p.pixel_coords.shape[0]

    def test_unit_stats_no_patches(self):
        pf = _make_pf_result(centres=[(0.0, 0.0, 0.15, 0.5)])  # tiny peak
        stats = place_field_stats(pf, threshold_method="absolute",
                                   threshold_value=100.0)
        u = stats[0]
        assert u.n_patches == 0
        assert u.patches == []
