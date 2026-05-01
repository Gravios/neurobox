"""Tests for neurobox.analysis.spatial."""

from __future__ import annotations

import numpy as np
import pytest

from neurobox.dtype import NBSpk, NBDxyz, NBModel, NBEpoch


# ─────────────────────────────────────────────────────────────────────────── #
# Synthetic data fixtures                                                      #
# ─────────────────────────────────────────────────────────────────────────── #

@pytest.fixture
def synthetic_trajectory():
    """A 60-second trajectory at 120 Hz that visits the whole maze.

    The rat does a slow Lissajous pattern over a 1×1 m maze, returning
    enough samples to every bin that occupancy is non-zero everywhere.
    """
    fs = 120.0
    duration = 60.0
    n = int(fs * duration)
    t = np.arange(n) / fs
    # Lissajous — covers the maze without too many revisits to one point
    x = 400 * np.sin(2 * np.pi * 0.07 * t)
    y = 400 * np.sin(2 * np.pi * 0.11 * t)
    xyz_data = np.stack([x, y, np.zeros(n)], axis=1)[:, np.newaxis, :]  # (T, 1, 3)
    xyz = NBDxyz(xyz_data, model=NBModel(["head"]), samplerate=fs)
    return xyz, t


@pytest.fixture
def place_cell_spikes(synthetic_trajectory):
    """Synthetic spikes for a place cell with field at (200, 0).

    Unit 1 is the place cell — fires at high rate when (x, y) is near
    (200, 0).  Unit 2 is a uniform-rate control.
    """
    xyz, t = synthetic_trajectory
    pos = xyz.data[:, 0, :2]
    fs = xyz.samplerate
    rng = np.random.default_rng(42)

    # ── Unit 1: place cell at (200, 0), σ=80 mm gaussian field, peak 30 Hz ── #
    centre = np.array([200.0, 0.0])
    field_sigma = 80.0
    peak_rate = 30.0
    distance_sq = ((pos - centre) ** 2).sum(axis=1)
    rate = peak_rate * np.exp(-distance_sq / (2 * field_sigma ** 2))
    # Convert per-sample rate to per-sample probability (rate / fs); draw spikes
    n_per_sample = rng.poisson(rate / fs)
    unit1_times: list[float] = []
    for i, n_spk in enumerate(n_per_sample):
        if n_spk > 0:
            # Place spikes uniformly within the sample window
            unit1_times.extend(t[i] + rng.uniform(0, 1.0 / fs, size=n_spk))
    unit1_times.sort()

    # ── Unit 2: uniform 5 Hz Poisson — control with no place preference ── #
    unit2_rate = 5.0
    n_per_sample2 = rng.poisson(unit2_rate / fs, size=len(t))
    unit2_times: list[float] = []
    for i, n_spk in enumerate(n_per_sample2):
        if n_spk > 0:
            unit2_times.extend(t[i] + rng.uniform(0, 1.0 / fs, size=n_spk))
    unit2_times.sort()

    res = np.concatenate([unit1_times, unit2_times])
    clu = np.concatenate([
        np.full(len(unit1_times), 1, dtype=np.int32),
        np.full(len(unit2_times), 2, dtype=np.int32),
    ])
    order = np.argsort(res)
    return NBSpk(
        res        = res[order].astype(np.float64),
        clu        = clu[order],
        map_       = np.array([[1, 1], [2, 1]], dtype=np.int64),
        samplerate = 20000.0,
    )


# ─────────────────────────────────────────────────────────────────────────── #
# occupancy_map                                                                #
# ─────────────────────────────────────────────────────────────────────────── #

class TestOccupancyMap:

    def test_basic_shape_and_units(self, synthetic_trajectory):
        from neurobox.analysis.spatial import occupancy_map
        xyz, _ = synthetic_trajectory
        result = occupancy_map(
            xyz, bin_size=50,
            boundary=[(-500, 500), (-500, 500)],
        )
        assert result.occupancy.shape == (20, 20)
        # Total dwell time should equal trajectory duration
        total_occ = np.nansum(result.occupancy_raw)
        expected = xyz.data.shape[0] / xyz.samplerate
        assert abs(total_occ - expected) < 0.5  # < 0.5 s discrepancy

    def test_smoothing_blurs_occupancy(self, synthetic_trajectory):
        from neurobox.analysis.spatial import occupancy_map
        xyz, _ = synthetic_trajectory
        sharp = occupancy_map(xyz, bin_size=50,
                              boundary=[(-500, 500), (-500, 500)])
        smooth = occupancy_map(xyz, bin_size=50,
                               boundary=[(-500, 500), (-500, 500)],
                               smoothing_sigma=2.0)
        # Smooth should have lower variance among occupied bins
        sv = np.nanvar(sharp.occupancy)
        smv = np.nanvar(smooth.occupancy)
        assert smv < sv

    def test_state_mask_shrinks_total_occupancy(self, synthetic_trajectory):
        from neurobox.analysis.spatial import occupancy_map
        xyz, _ = synthetic_trajectory
        # Restrict to first 10 seconds
        ep = NBEpoch(np.array([[0.0, 10.0]]), samplerate=xyz.samplerate)
        result = occupancy_map(xyz, bin_size=50,
                               boundary=[(-500, 500), (-500, 500)],
                               state=ep)
        total = np.nansum(result.occupancy_raw)
        assert abs(total - 10.0) < 0.5

    def test_plain_ndarray_input(self):
        from neurobox.analysis.spatial import occupancy_map
        rng = np.random.default_rng(0)
        traj = rng.uniform(0, 100, size=(1200, 2))  # 10 s at 120 Hz
        result = occupancy_map(
            traj, bin_size=10,
            boundary=[(0, 100), (0, 100)],
            samplerate=120.0,
        )
        assert result.occupancy_raw.shape == (10, 10)
        # Random uniform → most bins should have some dwell
        assert (result.occupancy_raw > 0).sum() > 50

    def test_min_occupancy_masks_low_dwell_bins(self, synthetic_trajectory):
        from neurobox.analysis.spatial import occupancy_map
        xyz, _ = synthetic_trajectory
        # Very high min_occupancy → few bins survive
        result = occupancy_map(xyz, bin_size=50,
                               boundary=[(-500, 500), (-500, 500)],
                               min_occupancy=2.0)
        # Some bins should be masked (NaN)
        assert np.isnan(result.occupancy).any()

    def test_validation(self):
        from neurobox.analysis.spatial import occupancy_map
        with pytest.raises(ValueError, match="samplerate is required"):
            occupancy_map(np.zeros((100, 2)), bin_size=10,
                          boundary=[(0, 100), (0, 100)])
        with pytest.raises(ValueError, match="boundary"):
            occupancy_map(np.zeros((100, 2)), bin_size=10,
                          boundary=[(50, 0), (0, 100)],   # max < min
                          samplerate=120.0)


# ─────────────────────────────────────────────────────────────────────────── #
# place_field — basic correctness                                              #
# ─────────────────────────────────────────────────────────────────────────── #

class TestPlaceFieldBasic:

    def test_shape_and_unit_ids(self, synthetic_trajectory, place_cell_spikes):
        from neurobox.analysis.spatial import place_field
        xyz, _ = synthetic_trajectory
        result = place_field(
            place_cell_spikes, xyz, units=[1, 2],
            bin_size=50,
            boundary=[(-500, 500), (-500, 500)],
        )
        # rate_map shape: (n_bins_x, n_bins_y, n_units, n_iter)
        assert result.rate_map.shape == (20, 20, 2, 1)
        np.testing.assert_array_equal(result.unit_ids, [1, 2])
        assert result.spatial_info.shape == (2, 1)

    def test_place_cell_has_peak_at_field_centre(
        self, synthetic_trajectory, place_cell_spikes
    ):
        from neurobox.analysis.spatial import place_field
        xyz, _ = synthetic_trajectory
        result = place_field(
            place_cell_spikes, xyz, units=1,
            bin_size=50,
            boundary=[(-500, 500), (-500, 500)],
            smoothing_sigma=1.5,
        )
        # Peak of unit 1's rate map should be near (x≈200, y≈0).
        rmap = result.rate_map[:, :, 0, 0]
        # Find arg-max in the smoothed rate map
        valid = np.where(np.isfinite(rmap))
        if valid[0].size == 0:
            pytest.skip("rate map all NaN — test data insufficient")
        peak_idx = np.unravel_index(np.nanargmax(rmap), rmap.shape)
        peak_x = result.bin_centres[0][peak_idx[0]]
        peak_y = result.bin_centres[1][peak_idx[1]]
        # Should be within ~150 mm of the true centre (200, 0)
        assert abs(peak_x - 200) < 150, f"peak at x={peak_x}"
        assert abs(peak_y - 0)   < 150, f"peak at y={peak_y}"

    def test_place_cell_has_higher_spatial_info_than_uniform(
        self, synthetic_trajectory, place_cell_spikes
    ):
        from neurobox.analysis.spatial import place_field
        xyz, _ = synthetic_trajectory
        result = place_field(
            place_cell_spikes, xyz, units=[1, 2],
            bin_size=50,
            boundary=[(-500, 500), (-500, 500)],
            smoothing_sigma=1.5,
        )
        si_place   = result.spatial_info[0, 0]
        si_control = result.spatial_info[1, 0]
        assert np.isfinite(si_place) and np.isfinite(si_control)
        # Place cell should have substantially higher spatial information
        assert si_place > 2 * si_control, \
            f"place cell SI {si_place} not >> control SI {si_control}"

    def test_min_spikes_returns_nan_rate_map(self, synthetic_trajectory):
        """Units below the spike-count threshold get all-NaN rate maps."""
        from neurobox.analysis.spatial import place_field
        xyz, _ = synthetic_trajectory
        # Just 5 spikes, far below min_spikes=11
        spk = NBSpk(
            res        = np.array([0.5, 1.0, 2.0, 3.0, 5.0]),
            clu        = np.array([99, 99, 99, 99, 99], dtype=np.int32),
            map_       = np.array([[99, 1]], dtype=np.int64),
            samplerate = 20000.0,
        )
        result = place_field(
            spk, xyz, units=99,
            bin_size=50,
            boundary=[(-500, 500), (-500, 500)],
        )
        assert np.all(np.isnan(result.rate_map))
        assert result.n_spikes[0] == 5

    def test_state_mask_reduces_occupancy(
        self, synthetic_trajectory, place_cell_spikes
    ):
        from neurobox.analysis.spatial import place_field
        xyz, _ = synthetic_trajectory
        full = place_field(
            place_cell_spikes, xyz, units=1,
            bin_size=50,
            boundary=[(-500, 500), (-500, 500)],
        )
        ep = NBEpoch(np.array([[0.0, 10.0]]), samplerate=xyz.samplerate)
        partial = place_field(
            place_cell_spikes, xyz, units=1,
            bin_size=50,
            boundary=[(-500, 500), (-500, 500)],
            state=ep,
        )
        # Restricted occupancy must be ≤ full occupancy
        full_total    = np.nansum(full.occupancy)
        partial_total = np.nansum(partial.occupancy)
        assert partial_total < full_total

    def test_skaggs_correct_vs_labbox(self, synthetic_trajectory, place_cell_spikes):
        """Both modes return finite values; they generally differ."""
        from neurobox.analysis.spatial import place_field
        xyz, _ = synthetic_trajectory
        a = place_field(place_cell_spikes, xyz, units=1, bin_size=50,
                        boundary=[(-500, 500), (-500, 500)],
                        smoothing_sigma=1.5, skaggs_correct=True)
        b = place_field(place_cell_spikes, xyz, units=1, bin_size=50,
                        boundary=[(-500, 500), (-500, 500)],
                        smoothing_sigma=1.5, skaggs_correct=False)
        assert np.isfinite(a.spatial_info[0, 0])
        assert np.isfinite(b.spatial_info[0, 0])
        # Rate maps are identical regardless of skaggs_correct
        np.testing.assert_array_equal(a.rate_map, b.rate_map)
        # Spatial info / sparsity numerically differ between conventions
        assert a.spatial_info[0, 0] != b.spatial_info[0, 0]


# ─────────────────────────────────────────────────────────────────────────── #
# place_field — bootstrap / halfsample / shuffle                               #
# ─────────────────────────────────────────────────────────────────────────── #

class TestPlaceFieldBootstrap:

    def test_n_iter_dimension(self, synthetic_trajectory, place_cell_spikes):
        from neurobox.analysis.spatial import place_field
        xyz, _ = synthetic_trajectory
        result = place_field(
            place_cell_spikes, xyz, units=1,
            bin_size=50, boundary=[(-500, 500), (-500, 500)],
            n_iter=5, bootstrap_fraction=1.0, rng=0,
        )
        assert result.rate_map.shape == (20, 20, 1, 5)
        assert result.spatial_info.shape == (1, 5)

    def test_bootstrap_introduces_variance(self, synthetic_trajectory, place_cell_spikes):
        from neurobox.analysis.spatial import place_field
        xyz, _ = synthetic_trajectory
        result = place_field(
            place_cell_spikes, xyz, units=1,
            bin_size=50, boundary=[(-500, 500), (-500, 500)],
            n_iter=10, bootstrap_fraction=1.0, rng=0,
        )
        # Spatial info should vary across iterations (positive variance)
        si_iters = result.spatial_info[0]
        assert np.nanvar(si_iters) > 0

    def test_pos_shuffle_destroys_spatial_information(
        self, synthetic_trajectory, place_cell_spikes
    ):
        """Shuffling position destroys the place-cell field signature."""
        from neurobox.analysis.spatial import place_field
        xyz, _ = synthetic_trajectory
        result_real = place_field(
            place_cell_spikes, xyz, units=1,
            bin_size=50, boundary=[(-500, 500), (-500, 500)],
            smoothing_sigma=1.5,
        )
        result_shuf = place_field(
            place_cell_spikes, xyz, units=1,
            bin_size=50, boundary=[(-500, 500), (-500, 500)],
            smoothing_sigma=1.5,
            n_iter=20, pos_shuffle=True, rng=0,
        )
        si_real = result_real.spatial_info[0, 0]
        si_shuf = np.nanmean(result_shuf.spatial_info[0])
        # Shuffled SI distribution mean should be substantially below real
        assert si_real > 2 * si_shuf, \
            f"shuffle didn't destroy SI: real={si_real}, shuf={si_shuf}"

    def test_halfsample_requires_even_n_iter(
        self, synthetic_trajectory, place_cell_spikes
    ):
        from neurobox.analysis.spatial import place_field
        xyz, _ = synthetic_trajectory
        with pytest.raises(ValueError, match="halfsample.*even"):
            place_field(
                place_cell_spikes, xyz, units=1,
                bin_size=50, boundary=[(-500, 500), (-500, 500)],
                n_iter=3, halfsample=True,
            )

    def test_rng_determinism(self, synthetic_trajectory, place_cell_spikes):
        from neurobox.analysis.spatial import place_field
        xyz, _ = synthetic_trajectory
        r1 = place_field(place_cell_spikes, xyz, units=1,
                         bin_size=50, boundary=[(-500, 500), (-500, 500)],
                         n_iter=5, bootstrap_fraction=1.0, rng=42)
        r2 = place_field(place_cell_spikes, xyz, units=1,
                         bin_size=50, boundary=[(-500, 500), (-500, 500)],
                         n_iter=5, bootstrap_fraction=1.0, rng=42)
        np.testing.assert_array_equal(r1.rate_map, r2.rate_map)
        np.testing.assert_array_equal(r1.spatial_info, r2.spatial_info)


# ─────────────────────────────────────────────────────────────────────────── #
# Edge cases                                                                   #
# ─────────────────────────────────────────────────────────────────────────── #

class TestPlaceFieldEdgeCases:

    def test_empty_state_returns_nan(self, synthetic_trajectory, place_cell_spikes):
        from neurobox.analysis.spatial import place_field
        xyz, _ = synthetic_trajectory
        # An epoch outside the recording → empty mask
        ep = NBEpoch(np.array([[1000.0, 1010.0]]), samplerate=xyz.samplerate)
        result = place_field(
            place_cell_spikes, xyz, units=1,
            bin_size=50, boundary=[(-500, 500), (-500, 500)],
            state=ep,
        )
        assert np.all(np.isnan(result.rate_map))
        assert result.n_spikes[0] == 0

    def test_unit_with_no_spikes(self, synthetic_trajectory):
        from neurobox.analysis.spatial import place_field
        xyz, _ = synthetic_trajectory
        spk = NBSpk(
            res        = np.array([], dtype=np.float64),
            clu        = np.array([], dtype=np.int32),
            map_       = np.array([[42, 1]], dtype=np.int64),
            samplerate = 20000.0,
        )
        result = place_field(
            spk, xyz, units=42,
            bin_size=50, boundary=[(-500, 500), (-500, 500)],
        )
        assert result.n_spikes[0] == 0
        assert np.all(np.isnan(result.rate_map))

    def test_all_units_when_units_is_none(
        self, synthetic_trajectory, place_cell_spikes
    ):
        from neurobox.analysis.spatial import place_field
        xyz, _ = synthetic_trajectory
        result = place_field(
            place_cell_spikes, xyz, units=None,
            bin_size=50, boundary=[(-500, 500), (-500, 500)],
        )
        # Both unit 1 and 2 should be present
        assert set(result.unit_ids.tolist()) == {1, 2}

    def test_3d_place_field(self, synthetic_trajectory):
        """Smoke test that 3-D rate maps work."""
        from neurobox.analysis.spatial import place_field
        xyz, t = synthetic_trajectory
        # Add a varying z component
        z = 100 * np.sin(2 * np.pi * 0.13 * t)
        new_data = xyz.data.copy()
        new_data[:, 0, 2] = z
        xyz3 = NBDxyz(new_data, model=NBModel(["head"]), samplerate=xyz.samplerate)

        # Synthesize a unit firing in 3-D field at (200, 0, 0)
        rng = np.random.default_rng(0)
        pos = xyz3.data[:, 0, :]
        rate = 30 * np.exp(-((pos - np.array([200, 0, 0])) ** 2).sum(1) / (2 * 100 ** 2))
        n_per = rng.poisson(rate / xyz3.samplerate)
        times = np.concatenate([[t[i]] * int(n) for i, n in enumerate(n_per) if n > 0])
        spk = NBSpk(
            res        = np.sort(times).astype(np.float64),
            clu        = np.full(len(times), 1, dtype=np.int32),
            map_       = np.array([[1, 1]], dtype=np.int64),
            samplerate = 20000.0,
        )
        result = place_field(
            spk, xyz3, units=1,
            bin_size=[50, 50, 50],
            boundary=[(-500, 500), (-500, 500), (-200, 200)],
        )
        # 20 × 20 × 8 × 1 unit × 1 iter
        assert result.rate_map.shape == (20, 20, 8, 1, 1)

    def test_validation(self, synthetic_trajectory, place_cell_spikes):
        from neurobox.analysis.spatial import place_field
        xyz, _ = synthetic_trajectory
        with pytest.raises(ValueError, match="n_iter"):
            place_field(place_cell_spikes, xyz, units=1, bin_size=50,
                        boundary=[(-500, 500), (-500, 500)], n_iter=0)
        with pytest.raises(ValueError, match="bootstrap_fraction"):
            place_field(place_cell_spikes, xyz, units=1, bin_size=50,
                        boundary=[(-500, 500), (-500, 500)],
                        bootstrap_fraction=-0.5)
