"""Tests for round-14 — cached_compute, CircStat ports, directional zones,
egocentric ratemaps, and the NBData/NBEpoch/NBSpk hash mechanism."""

from __future__ import annotations

import pickle
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from neurobox.io import data_hash, cached_compute, cache_path_for
from neurobox.analysis.stats import (
    circ_dist, circ_dist2, circ_var, circ_std, circ_median,
    circ_kappa, circ_moment, circ_skewness, circ_kurtosis,
    circ_axial, circ_ang2rad, circ_rad2ang,
)
from neurobox.dtype import NBDxyz, NBModel, NBEpoch, NBSpk


# ─────────────────────────────────────────────────────────────────────── #
# cached_compute                                                            #
# ─────────────────────────────────────────────────────────────────────── #

class TestCachedCompute:
    def setup_method(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="nbcc_test_"))

    def teardown_method(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_cache_hit_skips_recompute(self):
        calls = []

        @cached_compute(cache_dir=lambda x: self.tmp,
                         hash_args=["x"], prefix="fn")
        def fn(x, *, scale=10.0):
            calls.append(x)
            return x * scale

        assert fn(5) == 50.0
        assert fn(5) == 50.0      # cached, doesn't increment calls
        assert calls == [5]

    def test_overwrite_forces_recompute(self):
        calls = []

        @cached_compute(cache_dir=lambda x: self.tmp,
                         hash_args=["x"], prefix="fn")
        def fn(x):
            calls.append(x)
            return x * 2

        fn(3)
        fn(3, overwrite=True)
        assert calls == [3, 3]

    def test_hash_args_selectivity(self):
        """Args not in hash_args don't change the cache key."""
        @cached_compute(cache_dir=lambda x: self.tmp,
                         hash_args=["x"], prefix="fn")
        def fn(x, *, scale=10.0):
            return x * scale

        a = fn(7, scale=10.0)
        b = fn(7, scale=999.0)        # cached → 70.0, NOT 6993.0
        assert a == b == 70.0

    def test_cache_false_bypasses(self):
        calls = []

        @cached_compute(cache_dir=lambda x: self.tmp,
                         hash_args=["x"], prefix="fn")
        def fn(x):
            calls.append(x)
            return x * 2

        fn(2, cache=False)
        fn(2, cache=False)
        # Both calls bypass cache
        assert calls == [2, 2]


# ─────────────────────────────────────────────────────────────────────── #
# CircStat ports                                                            #
# ─────────────────────────────────────────────────────────────────────── #

class TestCircStat:
    def test_circ_dist_principal_value(self):
        # Difference between identical → 0
        assert circ_dist(1.0, 1.0) == pytest.approx(0.0)
        # 90° apart
        np.testing.assert_allclose(circ_dist(np.pi / 2, 0.0), np.pi / 2)
        # Wrap: 0 vs 2π → 0 (or near 0)
        assert abs(circ_dist(0.0, 2 * np.pi)) < 1e-10

    def test_circ_dist2_pairwise(self):
        a = np.array([0.0, np.pi / 2])
        b = np.array([0.0, np.pi])
        result = circ_dist2(a, b)
        assert result.shape == (2, 2)
        # Diagonal entries: a[0]-b[0]=0, a[1]-b[1]=-π/2
        np.testing.assert_allclose(result[0, 0], 0.0, atol=1e-10)
        np.testing.assert_allclose(result[1, 1], -np.pi / 2, atol=1e-10)

    def test_circ_var_concentrated_vs_uniform(self):
        concentrated = np.array([0.1, 0.0, -0.1, 0.05])
        uniform = np.linspace(-np.pi, np.pi, 100, endpoint=False)
        assert circ_var(concentrated) < 0.01
        assert circ_var(uniform) > 0.99

    def test_circ_std_outputs(self):
        theta = np.array([0.1, 0.0, -0.1, 0.05])
        s, s0 = circ_std(theta)
        # Both should be small for concentrated data
        assert s < 0.2
        assert s0 < 0.2

    def test_circ_median_known(self):
        # All angles near 0 → median near 0
        theta = np.array([0.1, 0.2, 0.0, -0.05])
        m = circ_median(theta)
        assert abs(m) < 0.5

    def test_circ_kappa_concentrated(self):
        theta = np.array([0.1, 0.0, -0.1, 0.05, 0.02, 0.08])
        k = circ_kappa(theta)
        assert k > 5    # concentrated → high κ

    def test_circ_kappa_uniform(self):
        uniform = np.linspace(-np.pi, np.pi, 100, endpoint=False)
        k = circ_kappa(uniform)
        assert abs(k) < 1e-6   # uniform → κ = 0

    def test_circ_moment_first_order(self):
        theta = np.array([0.1, 0.05, -0.05, -0.1])
        mp, rho, mu = circ_moment(theta, p=1)
        # Centroid mean is real and ~1
        assert rho > 0.99
        assert abs(mu) < 0.1

    def test_circ_skewness_kurtosis_runs(self):
        theta = np.array([0.1, 0.05, -0.05, -0.1])
        b, b0 = circ_skewness(theta)
        k, k0 = circ_kurtosis(theta)
        assert np.isfinite([b, k]).all()

    def test_circ_axial_p2(self):
        """Axial transform: angles π apart map to same value."""
        a = circ_axial(np.array([0.1, np.pi + 0.1]), p=2)
        # Both should be ~0.2 mod 2π
        np.testing.assert_allclose(a[0], a[1] % (2 * np.pi), atol=1e-10)

    def test_circ_deg_rad_roundtrip(self):
        deg = np.array([0.0, 90.0, 180.0, 270.0])
        rad = circ_ang2rad(deg)
        deg_back = circ_rad2ang(rad)
        np.testing.assert_allclose(deg, deg_back, atol=1e-10)


# ─────────────────────────────────────────────────────────────────────── #
# Directional zone scores                                                   #
# ─────────────────────────────────────────────────────────────────────── #

@pytest.fixture
def synthetic_session():
    """Build a minimal synthetic recording for zone-score testing."""
    from neurobox.analysis.kinematics import augment_xyz
    from neurobox.analysis.spatial.place_fields import place_field

    rng = np.random.default_rng(0)
    T, fs = 6000, 120.0
    t = np.arange(T) / fs
    body_x = 200 * np.cos(t * 0.3)
    body_y = 200 * np.sin(t * 0.3)
    heading = t * 0.3 + np.pi / 2

    markers = ["spine_lower", "spine_middle", "spine_upper",
               "head_back", "head_left", "head_front", "head_right"]
    offsets = {
        "spine_lower":  (-30, 0, 50),
        "spine_middle": (0, 0, 50),
        "spine_upper":  (15, 0, 50),
        "head_back":    (25, 0, 60),
        "head_left":    (30, 5, 60),
        "head_front":   (35, 0, 60),
        "head_right":   (30, -5, 60),
    }
    data = np.zeros((T, len(markers), 3))
    for i, m in enumerate(markers):
        ox, oy, oz = offsets[m]
        cy, sy = np.cos(heading), np.sin(heading)
        data[:, i, 0] = body_x + ox * cy - oy * sy
        data[:, i, 1] = body_y + ox * sy + oy * cy
        data[:, i, 2] = oz
    xyz = NBDxyz(data, model=NBModel(markers=markers), samplerate=fs)
    aug = augment_xyz(xyz)

    # Spikes preferentially when rat near (200, 0)
    hidx = aug.model.index("hcom")
    hcom_xy = aug.data[:, hidx, :2]
    sd = np.exp(-((hcom_xy[:, 0] - 200)**2 + hcom_xy[:, 1]**2) / 5000)
    res = [ti for ti in range(T) if rng.random() < sd[ti] * 0.8]
    res = np.asarray(res, dtype=np.int64)
    clu = np.full(res.size, 1, dtype=np.int32)
    map_ = pd.DataFrame({"global_id": [1], "shank": [1], "cluster": [1]})
    spk = NBSpk(res=res, clu=clu, map_=map_, samplerate=fs)

    pf = place_field(
        spk, hcom_xy, units=[1],
        bin_size=30, boundary=[(-300, 300), (-300, 300)],
        n_iter=1, samplerate=fs, min_spikes=5, min_occupancy=0.0,
    )
    return aug, spk, pf


class TestDirectionalZones:
    def test_drz_in_unit_range(self, synthetic_session):
        from neurobox.analysis.placefields import compute_drz
        aug, _, pf = synthetic_session
        score, c = compute_drz(pf, aug, [1])
        assert score.shape == (aug.data.shape[0], 1)
        assert np.isfinite(score).all()
        # DRZ is bounded by ±1 (rate/peak ratio)
        assert np.abs(score).max() <= 1.0

    def test_ddz_signed_distance(self, synthetic_session):
        from neurobox.analysis.placefields import compute_ddz
        aug, _, pf = synthetic_session
        score, c = compute_ddz(pf, aug, [1])
        # Magnitude should be on the order of arena half-width
        assert np.abs(score).max() > 100

    def test_ghz_gaussian_normalised(self, synthetic_session):
        from neurobox.analysis.placefields import compute_ghz
        aug, _, pf = synthetic_session
        score, c = compute_ghz(pf, aug, [1], sigma=150.0)
        assert np.abs(score).max() <= 1.0001    # tiny float slack

    def test_gdz_alias(self, synthetic_session):
        from neurobox.analysis.placefields import compute_gdz, compute_ghz
        aug, _, pf = synthetic_session
        s1, _ = compute_gdz(pf, aug, [1])
        s2, _ = compute_ghz(pf, aug, [1])
        np.testing.assert_array_equal(s1, s2)

    def test_hdz_runs(self, synthetic_session):
        from neurobox.analysis.placefields import compute_hdz
        aug, _, pf = synthetic_session
        score, c = compute_hdz(pf, aug, [1])
        assert score.shape == (aug.data.shape[0], 1)

    def test_hrz_runs(self, synthetic_session):
        from neurobox.analysis.placefields import compute_hrz
        aug, _, pf = synthetic_session
        score, c = compute_hrz(pf, aug, [1])
        assert np.abs(score).max() <= 1.0

    def test_hpv_tpv_alias(self, synthetic_session):
        from neurobox.analysis.placefields import compute_hpv, compute_tpv
        aug, _, pf = synthetic_session
        s1, _ = compute_hpv(pf, aug, [1])
        s2, _ = compute_tpv(pf, aug, [1])
        np.testing.assert_array_equal(s1, s2)

    def test_field_centres_from_result(self, synthetic_session):
        from neurobox.analysis.placefields import field_centres_from_result
        _, _, pf = synthetic_session
        c = field_centres_from_result(pf, [1])
        assert c.shape == (1, 2)
        # Field centre should be finite and within the boundary
        assert np.isfinite(c[0]).all()
        assert -300 < c[0, 0] < 300
        assert -300 < c[0, 1] < 300


# ─────────────────────────────────────────────────────────────────────── #
# Egocentric ratemaps                                                       #
# ─────────────────────────────────────────────────────────────────────── #

class TestEgocentric:
    def test_egocentric_position_shape(self, synthetic_session):
        from neurobox.analysis.placefields import egocentric_position
        aug, _, _ = synthetic_session
        ego = egocentric_position(aug, np.array([100.0, 50.0]))
        assert ego.shape == (aug.data.shape[0], 2)
        assert np.isfinite(ego).all()

    def test_compute_ego_ratemap_returns_dict(self, synthetic_session):
        from neurobox.analysis.placefields import compute_ego_ratemap
        aug, spk, pf = synthetic_session
        result = compute_ego_ratemap(
            spk, aug, units=[1], pft=pf,
            bin_size=40, boundary=[(-200, 200), (-200, 200)],
            min_occupancy=0.0, min_spikes=5,
        )
        assert isinstance(result, dict)
        assert 1 in result
        from neurobox.analysis.spatial.place_fields import PlaceFieldResult
        assert isinstance(result[1], PlaceFieldResult)

    def test_compute_ego_ratemap_conditioned_grid_shape(self, synthetic_session):
        from neurobox.analysis.placefields import compute_ego_ratemap_conditioned
        aug, spk, pf = synthetic_session
        T = aug.data.shape[0]
        # Two conditioning features with 2 bins each → 4 grid cells
        rng = np.random.default_rng(0)
        feat_a = rng.uniform(0, 1, T)
        feat_b = rng.uniform(0, 1, T)
        grid = compute_ego_ratemap_conditioned(
            spk, aug, units=[1], pft=pf,
            conditioning_features={"a": feat_a, "b": feat_b},
            conditioning_bins={"a": [0.0, 0.5, 1.0], "b": [0.0, 0.5, 1.0]},
            bin_size=40, boundary=[(-200, 200), (-200, 200)],
            min_occupancy=0.0, min_spikes=2,
        )
        # Grid keys are (i, j) tuples
        expected_keys = {(0, 0), (0, 1), (1, 0), (1, 1)}
        assert set(grid.keys()) == expected_keys


# ─────────────────────────────────────────────────────────────────────── #
# Hash mechanism — NBData / NBEpoch / NBSpk                                 #
# ─────────────────────────────────────────────────────────────────────── #

class TestHashMechanism:
    def test_nbdxyz_construction_sets_hash(self):
        data = np.random.standard_normal((100, 4, 3))
        xyz = NBDxyz(data, model=NBModel(markers=["a", "b", "c", "d"]),
                      samplerate=120.0, name="t", label="xyz", key="x")
        assert isinstance(xyz.hash, str)
        assert len(xyz.hash) == 40    # SHA-1

    def test_nbdxyz_filter_changes_hash(self):
        data = np.random.standard_normal((100, 4, 3))
        xyz = NBDxyz(data, model=NBModel(markers=["a", "b", "c", "d"]),
                      samplerate=120.0)
        h0 = xyz.hash
        xyz.filter(mode="butter", cutoff=10.0, btype="low")
        assert xyz.hash != h0

    def test_nbdxyz_resample_changes_hash(self):
        data = np.random.standard_normal((100, 4, 3))
        xyz = NBDxyz(data, model=NBModel(markers=["a", "b", "c", "d"]),
                      samplerate=120.0)
        h0 = xyz.hash
        xyz.resample(60.0)
        assert xyz.hash != h0

    def test_consecutive_filters_chain(self):
        """Two filter calls produce different hashes (chaining)."""
        data = np.random.standard_normal((100, 4, 3))
        xyz = NBDxyz(data, model=NBModel(markers=["a", "b", "c", "d"]),
                      samplerate=120.0)
        xyz.filter(mode="butter", cutoff=10.0, btype="low")
        h1 = xyz.hash
        xyz.filter(mode="butter", cutoff=10.0, btype="low")
        h2 = xyz.hash
        assert h1 != h2

    def test_identical_constructs_same_hash(self):
        data = np.random.standard_normal((100, 4, 3))
        kw = dict(model=NBModel(markers=["a", "b", "c", "d"]),
                  samplerate=120.0, name="t", label="xyz", key="x")
        xyz1 = NBDxyz(data,        **kw)
        xyz2 = NBDxyz(data.copy(), **kw)
        assert xyz1.hash == xyz2.hash

    def test_different_metadata_different_hash(self):
        data = np.random.standard_normal((100, 4, 3))
        xyz1 = NBDxyz(data, model=NBModel(markers=["a", "b", "c", "d"]),
                       samplerate=120.0, name="A")
        xyz2 = NBDxyz(data, model=NBModel(markers=["a", "b", "c", "d"]),
                       samplerate=120.0, name="B")
        assert xyz1.hash != xyz2.hash

    def test_nbepoch_hash(self):
        e1 = NBEpoch(data=np.array([[1.0, 2.0]]), label="walk", key="w",
                     samplerate=1.0)
        e2 = NBEpoch(data=np.array([[1.0, 2.0]]), label="walk", key="w",
                     samplerate=1.0)
        assert e1.hash == e2.hash
        e3 = NBEpoch(data=np.array([[1.0, 2.0]]), label="rear", key="r",
                     samplerate=1.0)
        assert e1.hash != e3.hash

    def test_nbepoch_setop_changes_hash(self):
        e1 = NBEpoch(data=np.array([[1.0, 2.0]]), label="walk", key="w",
                     samplerate=1.0)
        e2 = NBEpoch(data=np.array([[3.0, 4.0]]), label="rear", key="r",
                     samplerate=1.0)
        union = e1 | e2
        assert union.hash != e1.hash
        assert union.hash != e2.hash

    def test_nbspk_hash(self):
        res = np.array([10, 20, 30], dtype=np.int64)
        clu = np.array([1, 1, 2], dtype=np.int32)
        map_ = np.array([[1, 1], [2, 1]], dtype=np.int64)
        spk1 = NBSpk(res=res, clu=clu, map_=map_, samplerate=20000.0)
        spk2 = NBSpk(res=res.copy(), clu=clu.copy(), map_=map_.copy(),
                     samplerate=20000.0)
        assert spk1.hash == spk2.hash

    def test_nbspk_different_spikes_different_hash(self):
        map_ = np.array([[1, 1], [2, 1]], dtype=np.int64)
        spk1 = NBSpk(res=np.array([10, 20]), clu=np.array([1, 2]),
                     map_=map_, samplerate=20000.0)
        spk2 = NBSpk(res=np.array([15, 25]), clu=np.array([1, 2]),
                     map_=map_, samplerate=20000.0)
        assert spk1.hash != spk2.hash


# ─────────────────────────────────────────────────────────────────────── #
# Hash + cached_compute integration                                         #
# ─────────────────────────────────────────────────────────────────────── #

class TestHashCachedComputeIntegration:
    def setup_method(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="nbcc_int_"))

    def teardown_method(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_xyz_hash_invalidates_cache(self):
        """Two xyz objects with different hashes get separate cache entries."""
        from neurobox.io import cached_compute

        @cached_compute(
            cache_dir = lambda x: self.tmp,
            hash_args = lambda x: x.hash,
            prefix    = "xyz_summary",
        )
        def summarise(xyz):
            return float(np.mean(xyz.data))

        data1 = np.full((100, 4, 3), 1.0)
        data2 = np.full((100, 4, 3), 2.0)
        kw = dict(model=NBModel(markers=["a", "b", "c", "d"]),
                  samplerate=120.0)
        # Distinct names → distinct hashes (matches MATLAB convention
        # where filename uniquely identifies the data; in-memory
        # objects without a backing file should set `name` instead).
        x1 = NBDxyz(data1, name="session_A", **kw)
        x2 = NBDxyz(data2, name="session_B", **kw)
        assert x1.hash != x2.hash
        assert summarise(x1) == 1.0
        assert summarise(x2) == 2.0
