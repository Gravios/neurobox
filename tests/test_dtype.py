"""Tests for neurobox.dtype core types."""

import numpy as np
import pytest
import tempfile
from pathlib import Path


# ── NBEpoch ─────────────────────────────────────────────────────────────── #

@pytest.fixture
def walk():
    from neurobox.dtype import NBEpoch
    return NBEpoch(np.array([[1.0, 3.0], [5.0, 8.0]]),
                   samplerate=100., label="walk", key="w")


@pytest.fixture
def rear():
    from neurobox.dtype import NBEpoch
    return NBEpoch(np.array([[2.0, 6.0]]),
                   samplerate=100., label="rear", key="r")


def test_epoch_duration(walk):
    assert abs(walk.duration - 5.0) < 1e-9


def test_epoch_n_periods(walk):
    assert walk.n_periods == 2


def test_epoch_intersection(walk, rear):
    inter = walk & rear
    assert inter.n_periods == 2
    np.testing.assert_allclose(inter.data[0], [2.0, 3.0])
    np.testing.assert_allclose(inter.data[1], [5.0, 6.0])


def test_epoch_union(walk, rear):
    union = walk | rear
    assert union.n_periods == 1
    np.testing.assert_allclose(union.data[0, 0], 1.0)
    np.testing.assert_allclose(union.data[0, 1], 8.0)


def test_epoch_difference(walk, rear):
    diff = walk - rear
    assert diff.n_periods == 2


def test_epoch_to_mask(walk):
    mask = walk.to_mask(1000)
    assert mask[100:300].all()
    assert mask[500:800].all()
    assert not mask[0:100].any()
    assert not mask[300:500].any()


def test_epoch_from_logical():
    from neurobox.dtype import NBEpoch
    mask = np.zeros(100, dtype=bool)
    mask[20:40] = True
    mask[60:80] = True
    ep = NBEpoch.from_logical(mask, samplerate=1.0)
    assert ep.n_periods == 2


def test_epoch_fillgaps():
    from neurobox.dtype import NBEpoch
    ep = NBEpoch(np.array([[1.0, 2.0], [2.05, 3.0]]), samplerate=100.)
    filled = ep.fillgaps(0.1)
    assert filled.n_periods == 1


def test_epoch_resample(walk):
    ep50 = walk.resample(50.)
    assert ep50.samplerate == 50.
    assert ep50.n_periods == walk.n_periods


def test_epoch_copy_is_independent(walk):
    cp = walk.copy()
    cp.data[0, 0] = 999.0
    assert walk.data[0, 0] != 999.0


def test_epoch_static_intersect(walk, rear):
    from neurobox.dtype import NBEpoch
    result = NBEpoch.intersect([walk, rear])
    assert result.n_periods == 2


def test_epoch_static_union(walk, rear):
    from neurobox.dtype import NBEpoch
    result = NBEpoch.union([walk, rear])
    assert result.n_periods == 1


# ── select_periods ───────────────────────────────────────────────────────── #

def test_select_periods():
    from neurobox.dtype import select_periods
    data = np.arange(100, dtype=float)
    out = select_periods(data, np.array([[0.1, 0.3], [0.5, 0.7]]),
                         samplerate=100.)
    assert len(out) == 40


# ── NBModel ─────────────────────────────────────────────────────────────── #

def test_model_index():
    from neurobox.dtype import NBModel
    m = NBModel(["head_front", "head_back", "spine_upper"])
    assert m.index("spine_upper") == 2


def test_model_resolve_none():
    from neurobox.dtype import NBModel
    m = NBModel(["a", "b", "c"])
    assert m.resolve(None) == [0, 1, 2]


def test_model_subset():
    from neurobox.dtype import NBModel
    m = NBModel(["a", "b", "c"])
    sub = m.subset(["a", "c"])
    assert sub.markers == ["a", "c"]


def test_model_default_rat():
    from neurobox.dtype import NBModel
    rat = NBModel.default_rat()
    assert len(rat) == 15
    assert "head_front" in rat


def test_model_missing_key_raises():
    from neurobox.dtype import NBModel
    m = NBModel(["a", "b"])
    with pytest.raises(KeyError):
        m.index("z")


# ── NBSpk ───────────────────────────────────────────────────────────────── #

@pytest.fixture
def spk():
    from neurobox.dtype import NBSpk
    return NBSpk(
        res  = np.array([0.1, 0.5, 1.0, 2.0, 3.0], dtype=np.float64),
        clu  = np.array([2, 2, 3, 3, 2], dtype=np.int32),
        map_ = np.array([[2, 1], [3, 1]], dtype=np.int64),
        samplerate = 20000.0,
    )


def test_spk_getitem_unit(spk):
    assert len(spk[2]) == 3
    assert len(spk[3]) == 2


def test_spk_getitem_multi_unit(spk):
    assert len(spk[[2, 3]]) == 5


def test_spk_getitem_with_epoch(spk):
    from neurobox.dtype import NBEpoch
    ep = NBEpoch(np.array([[0.0, 1.5]]), samplerate=1.)
    assert len(spk[2, ep]) == 2    # 0.1 and 0.5


def test_spk_restrict(spk):
    from neurobox.dtype import NBEpoch
    ep = NBEpoch(np.array([[0.0, 1.5]]), samplerate=1.)
    r = spk.restrict(ep)
    assert len(r) == 3


def test_spk_by_unit(spk):
    units = spk.by_unit()
    assert set(units.keys()) == {2, 3}
    assert len(units[2]) == 3


def test_spk_shank_for_unit(spk):
    assert spk.shank_for_unit(2) == 1
    assert spk.shank_for_unit(99) is None


# ── NBDxyz ───────────────────────────────────────────────────────────────── #

@pytest.fixture
def xyz():
    from neurobox.dtype import NBDxyz, NBModel
    data = np.random.default_rng(0).standard_normal((200, 3, 3))
    return NBDxyz(data,
                  model=NBModel(["head", "body", "tail"]),
                  samplerate=120.)


def test_xyz_vel_shape(xyz):
    assert xyz.vel().shape == (200, 3)


def test_xyz_vel_zero_first_row(xyz):
    v = xyz.vel()
    np.testing.assert_array_equal(v[0], 0.0)


def test_xyz_acc_shape(xyz):
    assert xyz.acc().shape == (200, 3)


def test_xyz_dist_shape(xyz):
    assert xyz.dist("head", "body").shape == (200,)


def test_xyz_com_shape(xyz):
    assert xyz.com(["head", "body"]).shape == (200, 3)


def test_xyz_sel_by_name(xyz):
    assert xyz.sel("head").shape == (200, 1, 3)


def test_xyz_sel_list(xyz):
    assert xyz.sel(["head", "body"]).shape == (200, 2, 3)


def test_xyz_save_load(xyz, tmp_path):
    f = tmp_path / "pos.npz"
    xyz.save_npy(f)
    from neurobox.dtype import NBDxyz
    xyz2 = NBDxyz()
    xyz2.load(f)
    assert xyz2.data.shape == xyz.data.shape
    assert xyz2.model.markers == xyz.model.markers


# ── NBDlfp ───────────────────────────────────────────────────────────────── #

@pytest.fixture
def lfp():
    from neurobox.dtype import NBDlfp
    data = np.random.default_rng(0).standard_normal((6250, 8)).astype(np.float32)
    return NBDlfp(data, channels=list(range(8)), samplerate=1250.)


def test_lfp_csd_shape(lfp):
    csd = lfp.csd(channel_interval=1, channel_pitch_um=50.)
    assert csd.shape == (6250, 6)


def test_lfp_filter_butter(lfp):
    lfp.filter("butter", cutoff=[6, 12], btype="band")
    assert lfp.data is not None
    assert lfp.data.shape == (6250, 8)


def test_lfp_filter_gauss(lfp):
    lfp.filter("gauss", sigma_sec=0.05)
    assert lfp.data.shape == (6250, 8)


def test_lfp_getitem_period(lfp):
    from neurobox.dtype import NBEpoch
    ep = NBEpoch(np.array([[0.0, 1.0]]), samplerate=1.)
    chunk = lfp[ep.data]
    assert chunk.shape == (1250, 8)


# ── NBStateCollection ────────────────────────────────────────────────────── #

@pytest.fixture
def stc():
    from neurobox.dtype import NBStateCollection, NBEpoch
    s = NBStateCollection()
    s.add_state(NBEpoch(np.array([[1., 3.], [5., 7.]]), label="walk", key="w"))
    s.add_state(NBEpoch(np.array([[2., 6.]]),            label="rear", key="r"))
    return s


def test_stc_getitem_name(stc):
    ep = stc["walk"]
    assert ep.label == "walk"


def test_stc_getitem_key(stc):
    ep = stc["w"]
    assert ep.label == "walk"


def test_stc_query_intersection(stc):
    inter = stc["walk&rear"]
    assert inter.n_periods == 2


def test_stc_query_union(stc):
    union = stc["walk|rear"]
    assert union.n_periods == 1


def test_stc_query_difference(stc):
    diff = stc["walk-rear"]
    assert diff.n_periods > 0


def test_stc_intersect_method(stc):
    result = stc.intersect("walk", "rear")
    assert result.n_periods == 2


def test_stc_union_method(stc):
    result = stc.union("walk", "rear")
    assert result.n_periods == 1


def test_stc_filter_min_duration(stc):
    # walk periods: [1,3]=2s and [5,7]=2s — neither exceeds 2.5s
    ep = stc.filter("walk", min_duration_sec=2.5)
    assert ep.n_periods == 0

def test_stc_filter_min_duration_keeps_long(stc):
    # threshold below 2s — both periods survive
    ep = stc.filter("walk", min_duration_sec=1.5)
    assert ep.n_periods == 2


def test_stc_get_transitions(stc):
    trans = stc.get_transitions("walk", "rear", window_sec=0.5)
    assert trans.ndim == 2 and trans.shape[1] == 2


def test_stc_persistence(stc, tmp_path):
    stc.path = tmp_path
    stc.filename = "test.stc.pkl"
    stc.save(overwrite=True)
    from neurobox.dtype import NBStateCollection
    loaded = NBStateCollection.load_file(tmp_path / "test.stc.pkl")
    assert loaded.has_state("walk")
    assert loaded.has_state("rear")


def test_stc_missing_state_raises(stc):
    with pytest.raises(KeyError):
        stc["grooming"]


# ── NBSessionPaths ───────────────────────────────────────────────────────── #

def test_paths_session_name():
    from neurobox.dtype import NBSessionPaths, parse_session_name
    parts = parse_session_name("sirotaA-jg-05-20120316")
    assert parts == {
        "sourceId":  "sirotaA",
        "userId":    "jg",
        "subjectId": "05",
        "date":      "20120316",
    }


def test_paths_hierarchy():
    from neurobox.dtype import NBSessionPaths
    p = NBSessionPaths("sirotaA-jg-05-20120316", Path("/data"), "B01", maze="cof")
    assert p.spath         == Path("/data/project/B01/sirotaA-jg-05-20120316")
    assert p.processed_ephys == Path(
        "/data/processed/ephys/sirotaA/sirotaA-jg"
        "/sirotaA-jg-05/sirotaA-jg-05-20120316"
    )
    assert p.processed_mocap == Path(
        "/data/processed/mocap/sirotaA/sirotaA-jg"
        "/sirotaA-jg-05/sirotaA-jg-05-20120316/cof"
    )


def test_paths_file_names():
    from neurobox.dtype import NBSessionPaths
    p = NBSessionPaths("sirotaA-jg-05-20120316", Path("/data"), "B01")
    assert p.yaml_file.name  == "sirotaA-jg-05-20120316.yaml"
    assert p.res_file(2).name == "sirotaA-jg-05-20120316.res.2"
    assert p.clu_file(1).name == "sirotaA-jg-05-20120316.clu.1"


def test_paths_bad_name_raises():
    from neurobox.dtype.paths import parse_session_name
    with pytest.raises(ValueError):
        parse_session_name("not-a-valid-name")


# ── Fixes verification tests ────────────────────────────────────────────── #

class TestSubsrefFixes:
    """Verify correctness of the ported subsref / resample fixes."""

    # ── __getitem__ accepts NBEpoch directly ──────────────────────────── #

    def test_getitem_nbepoch_direct(self):
        """obj[epoch] should work without .data — mirrors MTA subsref."""
        from neurobox.dtype import NBDlfp, NBEpoch
        data = np.arange(200, dtype=np.float32).reshape(100, 2)
        lfp  = NBDlfp(data, channels=[0, 1], samplerate=100.)
        ep   = NBEpoch(np.array([[0.1, 0.3]]), samplerate=1.)
        result = lfp[ep]
        assert result.shape == (20, 2)

    def test_getitem_nbepoch_resamples_to_data_sr(self):
        """Epoch at different samplerate should be resampled first."""
        from neurobox.dtype import NBDlfp, NBEpoch
        data = np.ones((1250, 4), dtype=np.float32)
        lfp  = NBDlfp(data, channels=list(range(4)), samplerate=1250.)
        # Epoch at 1 Hz — [0,1] seconds → 1250 samples at lfp samplerate
        ep   = NBEpoch(np.array([[0.0, 1.0]]), samplerate=1.)
        result = lfp[ep]
        assert result.shape[0] == 1250

    def test_getitem_float_array_period_select(self):
        """(N,2) float64 array still triggers period selection."""
        from neurobox.dtype import NBDlfp
        data = np.arange(100, dtype=np.float64).reshape(100, 1)
        lfp  = NBDlfp(data, channels=[0], samplerate=1.)
        periods = np.array([[10., 20.]], dtype=np.float64)
        result  = lfp[periods]
        assert result.shape == (10, 1)

    def test_getitem_int_array_normal_index(self):
        """Integer (N,2) array should NOT trigger period selection."""
        from neurobox.dtype import NBDlfp
        data = np.arange(200, dtype=np.float64).reshape(100, 2)
        lfp  = NBDlfp(data, channels=[0, 1], samplerate=1.)
        # Select rows 5 and 10 via integer array — NOT period-select
        idx = np.array([5, 10])
        result = lfp[idx]
        assert result.shape == (2, 2)   # two rows, two channels

    def test_getitem_int_ndim2_not_period_select(self):
        """(N,2) integer array must not be mistaken for float period array."""
        from neurobox.dtype import NBDlfp
        data = np.arange(200, dtype=np.float64).reshape(100, 2)
        lfp  = NBDlfp(data, channels=[0, 1], samplerate=1.)
        # Integer dtype: shape (2,2) but NOT float64 → should NOT period-select
        idx = np.array([[5, 0], [10, 1]], dtype=np.int64)
        result = lfp[idx]
        # numpy returns shape (2, 2, 2) for fancy indexing on (100,2) with (2,2)
        assert result.shape[0] == 2 and result.dtype == np.float64

    # ── resample with NBData target ────────────────────────────────────── #

    def test_resample_to_ndata_target(self):
        """resample(other_obj) should match length and samplerate."""
        from neurobox.dtype import NBDlfp
        src = NBDlfp(np.random.randn(1250, 4).astype(np.float32),
                     channels=list(range(4)), samplerate=1250.)
        tgt = NBDlfp(np.zeros((100, 4), dtype=np.float32),
                     channels=list(range(4)), samplerate=100.)
        src.resample(tgt)
        assert src.samplerate == 100.
        assert src.n_samples  == 100

    def test_resample_antialias_downsampling(self):
        """Downsampling with spline should attenuate above Nyquist."""
        from neurobox.dtype import NBDlfp
        sr   = 1000.
        t    = np.arange(int(sr * 2)) / sr
        # Signal: 5 Hz (below new Nyquist of 50 Hz) + 200 Hz (above)
        sig  = np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 200 * t)
        lfp  = NBDlfp(sig[:, None].astype(np.float64), samplerate=sr)
        lfp.resample(100., method="spline")
        # After resampling to 100 Hz, the 200 Hz component should be gone
        # Check by looking at variance — should be close to the 5 Hz component alone
        assert lfp.samplerate == 100.
        # Rough check: output variance should be dominated by 5 Hz (amplitude ~1)
        assert lfp._data.var() < 1.5   # pure 5Hz sin has variance 0.5

    def test_resample_poly_correct_length(self):
        """Poly resample should give the expected output length."""
        from neurobox.dtype import NBDlfp
        lfp = NBDlfp(np.zeros((1250, 2), dtype=np.float32),
                     channels=[0, 1], samplerate=1250.)
        lfp.resample(100., method="poly")
        assert lfp.samplerate == 100.
        assert lfp.n_samples  == 100   # 1250/1250*100 = 100

    # ── NBEpoch mask resample nearest-neighbour ────────────────────────── #

    def test_epoch_mask_resample_no_shift(self):
        """Mask resampling should not introduce a systematic offset."""
        from neurobox.dtype import NBEpoch
        # Create a mask with a 1-second pulse at t=1..2
        sr   = 100.
        mask = np.zeros(300, dtype=bool)
        mask[100:200] = True              # samples 100..199 = 1.0..2.0 s
        ep = NBEpoch(mask.astype(np.float64), samplerate=sr, mode="mask")
        ep_p = ep.to_periods()
        np.testing.assert_allclose(ep_p.data[0], [1.0, 2.0], atol=0.02)

        # Resample to 50 Hz — should still map to same seconds
        ep50 = ep.resample(50.)
        p50  = ep50.to_periods()
        np.testing.assert_allclose(p50.data[0, 0], 1.0, atol=0.04)
        np.testing.assert_allclose(p50.data[0, 1], 2.0, atol=0.04)

    # ── _within_periods performance / correctness ─────────────────────── #

    def test_within_periods_sorted_correctness(self):
        """searchsorted-based _within_periods must match brute-force."""
        from neurobox.dtype.spikes import _within_periods
        rng   = np.random.default_rng(0)
        times = np.sort(rng.uniform(0, 100, size=1000))
        perds = np.array([[10., 20.], [40., 50.], [70., 80.]])

        mask = _within_periods(times, perds)
        # Brute-force reference
        ref  = np.zeros(len(times), dtype=bool)
        for s, e in perds:
            ref |= (times >= s) & (times < e)
        np.testing.assert_array_equal(mask, ref)

    def test_within_periods_edge_cases(self):
        """Spike exactly at period boundary follows half-open [s, e) convention."""
        from neurobox.dtype.spikes import _within_periods
        times = np.array([5.0, 10.0, 15.0, 20.0])
        perds = np.array([[5.0, 10.0]])   # [5, 10): includes 5, excludes 10
        mask  = _within_periods(times, perds)
        assert mask[0] == True    # 5.0 included
        assert mask[1] == False   # 10.0 excluded (half-open)

    def test_within_periods_empty_periods(self):
        from neurobox.dtype.spikes import _within_periods
        times = np.array([1., 2., 3.])
        mask  = _within_periods(times, np.empty((0, 2)))
        assert not mask.any()

    # ── select_periods accepts NBEpoch ────────────────────────────────── #

    def test_select_periods_accepts_epoch(self):
        from neurobox.dtype.epoch import select_periods, NBEpoch
        data = np.arange(100., dtype=np.float64)
        ep   = NBEpoch(np.array([[0.1, 0.3]]), samplerate=1.)
        out  = select_periods(data, ep, samplerate=100.)
        assert len(out) == 20


# ═══════════════════════════════════════════════════════════════════════════ #
# Tests for new MTA-parity methods                                           #
# ═══════════════════════════════════════════════════════════════════════════ #

class TestNBDataNewMethods:
    def _make_lfp(self):
        from neurobox.dtype.lfp import NBDlfp
        import numpy as np
        lfp = NBDlfp(samplerate=1250.)
        lfp._data = np.random.default_rng(0).standard_normal((1250, 4)).astype(np.float32)
        return lfp

    def test_copy_is_independent(self):
        lfp  = self._make_lfp()
        copy = lfp.copy()
        copy._data[0, 0] = 99999.0
        assert lfp._data[0, 0] != 99999.0

    def test_clear_frees_data(self):
        lfp = self._make_lfp()
        lfp.clear()
        assert lfp._data is None
        assert lfp.isempty()

    def test_update_path(self):
        lfp = self._make_lfp()
        lfp.update_path("/tmp/new_path")
        from pathlib import Path
        assert lfp.path == Path("/tmp/new_path")

    def test_update_filename(self):
        lfp = self._make_lfp()
        lfp.update_filename("session.lfp")
        assert lfp.filename == "session.lfp"

    def test_phase_shape(self):
        lfp = self._make_lfp()
        phs = lfp.phase(freq_range=(6, 12))
        assert phs._data.shape == lfp._data.shape

    def test_phase_range(self):
        import numpy as np
        lfp = self._make_lfp()
        phs = lfp.phase(freq_range=(6, 12))
        assert np.all(np.abs(phs._data) <= np.pi + 1e-4)


class TestNBEpochNewMethods:
    def test_cast_periods_to_mask(self):
        from neurobox.dtype.epoch import NBEpoch
        import numpy as np
        ep   = NBEpoch(np.array([[1.0, 3.0]]), samplerate=10.0)
        mask = ep.cast("mask", samplerate=10.0)
        assert mask.mode == "mask"
        assert len(mask.data) == 30           # 3 s * 10 Hz
        assert mask.data[10:30].all()         # samples 10–29 are inside period

    def test_cast_mask_to_periods(self):
        import numpy as np
        from neurobox.dtype.epoch import NBEpoch
        mask = np.zeros(100, dtype=bool)
        mask[20:40] = True
        ep   = NBEpoch(mask, samplerate=10.0, mode="mask")
        per  = ep.cast("periods")
        assert per.mode == "periods"
        assert per.data.shape[1] == 2

    def test_cast_noop(self):
        import numpy as np
        from neurobox.dtype.epoch import NBEpoch
        ep  = NBEpoch(np.array([[0.0, 1.0]]), samplerate=1.0)
        ep2 = ep.cast("periods")
        assert ep2.mode == "periods"

    def test_save_and_load(self, tmp_path):
        import numpy as np
        from pathlib import Path
        from neurobox.dtype.epoch import NBEpoch
        ep   = NBEpoch(np.array([[0.0, 2.0], [3.0, 5.0]]), samplerate=1.0)
        path = tmp_path / "test.pkl"
        ep.save(path)
        loaded = NBEpoch.load_file(path)
        import numpy.testing as npt
        npt.assert_array_equal(loaded.data, ep.data)


class TestNBSpkNewMethods:
    def _make_spk(self):
        import numpy as np
        from neurobox.dtype.spikes import NBSpk
        res = np.array([0.1, 0.2, 0.5], dtype=np.float64)
        clu = np.array([2, 2, 3], dtype=np.int32)
        map_ = np.array([[2, 1, 2], [3, 1, 3]], dtype=np.int64)
        return NBSpk(res, clu, map_)

    def test_clear_frees_arrays(self):
        spk = self._make_spk()
        spk.clear()
        assert len(spk.res) == 0
        assert len(spk.clu) == 0

    def test_copy_is_independent(self):
        import numpy as np
        spk  = self._make_spk()
        copy = spk.copy()
        copy.res[0] = 99.0
        assert spk.res[0] != 99.0

    def test_save_and_load_unit_set(self, tmp_path):
        import numpy as np
        from types import SimpleNamespace
        spk = self._make_spk()
        session = SimpleNamespace(
            spath=tmp_path,
            filebase="test.cof.all",
        )
        ids = np.array([2], dtype=np.int32)
        spk.save_unit_set(session, "good", ids)
        loaded = spk.load_unit_set(session, "good")
        np.testing.assert_array_equal(loaded, ids)

    def test_load_unit_set_missing_raises(self, tmp_path):
        import numpy as np
        from types import SimpleNamespace
        import pytest
        spk = self._make_spk()
        session = SimpleNamespace(spath=tmp_path, filebase="test.cof.all")
        with pytest.raises(FileNotFoundError):
            spk.load_unit_set(session, "nonexistent")


class TestNBDxyzNewMethods:
    def _make_xyz(self):
        import numpy as np
        from neurobox.dtype.xyz import NBDxyz
        from neurobox.dtype.model import NBModel
        data  = np.zeros((100, 3, 3), dtype=np.float64)
        data[:, 0, 2] = np.linspace(0, 50, 100)   # head_back: rising z
        data[:, 1, 2] = 10.0                        # head_front: flat
        data[:, 2, 2] = 5.0                         # spine_back: flat
        model = NBModel(["head_back", "head_front", "spine_back"])
        xyz = NBDxyz(samplerate=120.0, name="test")
        xyz._data = data
        xyz.model = model
        return xyz

    def test_subset_reduces_markers(self):
        xyz = self._make_xyz()
        sub = xyz.subset(["head_back", "head_front"])
        assert sub._data.shape == (100, 2, 3)
        assert sub.model.markers == ["head_back", "head_front"]

    def test_subset_data_matches(self):
        import numpy as np
        xyz = self._make_xyz()
        sub = xyz.subset(["head_back"])
        np.testing.assert_array_equal(sub._data[:, 0, 2], xyz._data[:, 0, 2])

    def test_get_pose_index_threshold(self):
        import numpy as np
        xyz = self._make_xyz()
        idx = xyz.get_pose_index("head_back", threshold=25.0)
        # z goes 0..50 over 100 frames, so > 25 starts around frame 51
        assert len(idx) > 0
        assert np.all(xyz._data[idx, 0, 2] > 25.0)

    def test_get_pose_index_no_results(self):
        xyz = self._make_xyz()
        idx = xyz.get_pose_index("head_front", threshold=100.0)
        assert len(idx) == 0


class TestNBSessionNewMethods:
    def test_update_paths_rebuilds_spath(self, tmp_path):
        from neurobox.dtype.session import NBSession
        from neurobox.config.config import configure_project
        configure_project("TEST_UP", data_root=str(tmp_path), overwrite=True)
        session = NBSession.__new__(NBSession)
        session.name      = "src-jg-01-20230101"
        session.maze      = "cof"
        session.filebase  = "src-jg-01-20230101.cof.all"
        session.samplerate = 20000.0
        session.par = session.spk = session.lfp = session.xyz = session.stc = None
        session.sync = session.nq = session.meta = None
        from neurobox.dtype.paths import NBSessionPaths
        session.paths = NBSessionPaths("src-jg-01-20230101", tmp_path, "TEST_UP")
        session.spath = session.paths.spath

        session.update_paths(data_root=str(tmp_path), project_id="TEST_UP")
        assert session.spath == tmp_path / "project" / "TEST_UP" / "src-jg-01-20230101"

    def test_list_trial_names_empty(self, tmp_path):
        from neurobox.dtype.session import NBSession
        session = NBSession.__new__(NBSession)
        session.name  = "src-jg-01-20230101"
        session.spath = tmp_path
        assert session.list_trial_names() == []

    def test_list_trial_names_finds_trials(self, tmp_path):
        from neurobox.dtype.session import NBSession
        # Create fake .trl.pkl files
        (tmp_path / "src-jg-01-20230101.cof.all.trl.pkl").touch()
        (tmp_path / "src-jg-01-20230101.cof.run1.trl.pkl").touch()
        session = NBSession.__new__(NBSession)
        session.name  = "src-jg-01-20230101"
        session.spath = tmp_path
        names = session.list_trial_names()
        assert "all"  in names
        assert "run1" in names


class TestNBSessionLoadSpkPeriods:
    """Tests for NBSession.load('spk', restrict=..., periods=...)."""

    def _make_session_with_spk(self, tmp_path):
        """Build a minimal NBSession with a pre-loaded NBSpk and a sync epoch."""
        import numpy as np
        from neurobox.dtype.session import NBSession
        from neurobox.dtype.spikes import NBSpk
        from neurobox.dtype.epoch import NBEpoch

        # 100 spikes spread across 0–60 s, 5 units
        rng = np.random.default_rng(7)
        res = np.sort(rng.uniform(0, 60, 100))
        clu = rng.integers(2, 7, size=100).astype(np.int32)   # unit IDs 2–6
        map_ = np.column_stack([
            np.arange(2, 7),
            np.ones(5, dtype=np.int64),
            np.arange(2, 7),
        ]).astype(np.int64)

        session = NBSession.__new__(NBSession)
        session.name       = "test-jg-01-20230101"
        session.maze       = "cof"
        session.trial      = "all"
        session.filebase   = "test-jg-01-20230101.cof.all"
        session.spath      = tmp_path
        session.samplerate = 20000.0
        session.par        = None
        session.lfp        = None
        session.xyz        = None
        session.stc        = None
        session.nq         = {}
        session.meta       = {}
        # sync covers only 10–50 s
        session.sync = NBEpoch(
            np.array([[10.0, 50.0]]), samplerate=1.0, mode="periods"
        )
        # Write fake .res.1 / .clu.1 files so NBSpk.load can find them
        base = tmp_path / "test-jg-01-20230101"
        import struct
        samplerate = 20000.0
        with open(str(base) + ".res.1", "wb") as f:
            for t in res:
                f.write(struct.pack("<q", int(t * samplerate)))
        n_clu = len(np.unique(clu))
        with open(str(base) + ".clu.1", "wb") as f:
            f.write(struct.pack("<i", n_clu + 1))   # header: n_clusters + 1
            for c in clu:
                f.write(struct.pack("<i", int(c)))

        # Also build a minimal par yaml
        import yaml
        par_data = {
            "acquisitionSystem": {
                "nChannels": 8, "nBits": 16,
                "samplingRate": 20000, "voltageRange": 20,
                "amplification": 1000, "offset": 0,
            },
            "fieldPotentials": {"lfpSamplingRate": 1250},
            "spikeDetection": {"channelGroups": [
                {"channels": [0,1,2,3,4,5,6,7],
                 "nSamples": 32, "peakSampleIndex": 16, "nFeatures": 3}
            ]},
        }
        with open(str(base) + ".yaml", "w") as f:
            yaml.dump(par_data, f)

        # Store original arrays for assertions
        session._test_res = res
        session._test_clu = clu
        # Minimal paths object so load("spk") fallback search works
        from neurobox.dtype.paths import NBSessionPaths
        from neurobox.config.config import configure_project
        configure_project("TESTSPK", data_root=str(tmp_path), overwrite=True)
        session.paths = NBSessionPaths(
            session_name = "test-jg-01-20230101",
            data_root    = tmp_path,
            project_id   = "TESTSPK",
        )
        return session

    def test_load_spk_no_sync_returns_all(self, tmp_path):
        """restrict=False ignores self.sync and returns every spike."""
        session = self._make_session_with_spk(tmp_path)
        spk = session.load("spk", restrict=False)
        assert len(spk.res) == len(session._test_res)

    def test_load_spk_auto_restricts_to_sync(self, tmp_path):
        """Default restrict=True applies self.sync automatically."""
        import numpy as np
        session = self._make_session_with_spk(tmp_path)
        spk = session.load("spk")   # restrict=True is the default
        # All returned spikes must fall inside [10, 50)
        assert len(spk.res) > 0
        assert np.all(spk.res >= 10.0)
        assert np.all(spk.res <  50.0)
        # Fewer spikes than the full recording
        assert len(spk.res) < len(session._test_res)

    def test_load_spk_explicit_periods_array(self, tmp_path):
        """Explicit periods= ndarray overrides self.sync."""
        import numpy as np
        session = self._make_session_with_spk(tmp_path)
        # Ask for a narrow window that does not overlap self.sync [10,50]
        spk = session.load("spk", periods=np.array([[0.0, 5.0]]))
        assert len(spk.res) > 0
        assert np.all(spk.res < 5.0)

    def test_load_spk_explicit_periods_epoch(self, tmp_path):
        """Explicit periods= NBEpoch overrides self.sync."""
        import numpy as np
        from neurobox.dtype.epoch import NBEpoch
        session = self._make_session_with_spk(tmp_path)
        ep = NBEpoch(np.array([[55.0, 60.0]]), samplerate=1.0)
        spk = session.load("spk", periods=ep)
        assert np.all(spk.res >= 55.0)

    def test_load_spk_no_sync_set_returns_all(self, tmp_path):
        """When self.sync is None, restrict=True still returns everything."""
        session = self._make_session_with_spk(tmp_path)
        session.sync = None
        spk = session.load("spk")
        assert len(spk.res) == len(session._test_res)

    def test_load_spk_other_kwargs_forwarded(self, tmp_path):
        """Extra kwargs (e.g. include_noise) are forwarded to NBSpk.load."""
        session = self._make_session_with_spk(tmp_path)
        # include_noise=True keeps cluster 0/1 — shouldn't crash with valid data
        spk = session.load("spk", restrict=False, include_noise=True)
        assert spk is not None


# ═══════════════════════════════════════════════════════════════════════════ #
# NBDang tests                                                               #
# ═══════════════════════════════════════════════════════════════════════════ #

class TestNBDang:
    def _make_xyz(self):
        """Simple 3-marker xyz: head_back, head_front, spine_lower."""
        import numpy as np
        from neurobox.dtype.xyz import NBDxyz
        from neurobox.dtype.model import NBModel

        T = 200
        rng = np.random.default_rng(42)
        data = np.zeros((T, 3, 3))
        data[:, 0] = [0.0,  0.0, 10.0]   # head_back  at origin
        data[:, 1] = [10.0, 0.0, 10.0]   # head_front 10mm ahead
        data[:, 2] = [0.0,  0.0,  0.0]   # spine_lower below
        # Add small noise
        data += rng.standard_normal(data.shape) * 0.5

        xyz = NBDxyz(samplerate=120.0)
        xyz._data = data.astype(np.float64)
        xyz.model = NBModel(["head_back", "head_front", "spine_lower"])
        return xyz

    def test_shape(self):
        from neurobox.dtype.ang import NBDang
        xyz = self._make_xyz()
        ang = NBDang.from_xyz(xyz)
        assert ang._data is not None
        assert ang._data.shape == (200, 3, 3, 3)

    def test_diagonal_is_nan(self):
        import numpy as np
        from neurobox.dtype.ang import NBDang
        xyz = self._make_xyz()
        ang = NBDang.from_xyz(xyz)
        for i in range(3):
            assert np.all(np.isnan(ang._data[:, i, i, :]))

    def test_radius_is_positive(self):
        import numpy as np
        from neurobox.dtype.ang import NBDang
        xyz = self._make_xyz()
        ang = NBDang.from_xyz(xyz)
        # Off-diagonal r should be positive
        r = ang._data[:, 0, 1, 2]   # head_back → head_front distance
        assert np.all(r[~np.isnan(r)] > 0)

    def test_radius_matches_dist(self):
        import numpy as np
        from neurobox.dtype.ang import NBDang
        xyz = self._make_xyz()
        ang = NBDang.from_xyz(xyz)
        # ang r should equal xyz.dist()
        r_ang  = ang._data[:, 0, 1, 2].astype(np.float64)
        r_dist = xyz.dist("head_back", "head_front")
        np.testing.assert_allclose(r_ang, r_dist, rtol=1e-4)

    def test_between_api(self):
        from neurobox.dtype.ang import NBDang
        xyz = self._make_xyz()
        ang = NBDang.from_xyz(xyz)
        theta = ang.between("head_back", "head_front", "theta")
        assert theta.shape == (200,)

    def test_head_direction(self):
        import numpy as np
        from neurobox.dtype.ang import NBDang
        xyz = self._make_xyz()
        ang = NBDang.from_xyz(xyz)
        hd = ang.head_direction("head_back", "head_front")
        assert hd.shape == (200,)
        # head_front is ~10mm ahead in X, so azimuth ≈ 0
        assert np.abs(np.nanmean(hd)) < 0.5

    def test_from_xyz_classmethod(self):
        from neurobox.dtype.ang import NBDang
        xyz = self._make_xyz()
        ang = NBDang.from_xyz(xyz)
        assert isinstance(ang, NBDang)
        assert ang.model is not None

    def test_session_load_ang(self, tmp_path):
        """session.load('ang') triggers ang computation."""
        import numpy as np
        from neurobox.dtype.session import NBSession
        from neurobox.dtype.xyz import NBDxyz
        from neurobox.dtype.model import NBModel

        session = NBSession.__new__(NBSession)
        session.name  = "t-jg-01-20230101"
        session.spath = tmp_path
        session.sync  = None
        session.par = session.spk = session.lfp = session.stc = None
        session.ang = session.ufr = session.nq = session.meta = None

        xyz = NBDxyz(samplerate=120.0)
        xyz._data = np.zeros((100, 3, 3))
        xyz._data[:, 0] = [0, 0, 10]
        xyz._data[:, 1] = [10, 0, 10]
        xyz._data[:, 2] = [0, 0, 0]
        xyz.model = NBModel(["head_back", "head_front", "spine_lower"])
        session.xyz = xyz

        from neurobox.dtype.paths import NBSessionPaths
        from neurobox.config.config import configure_project
        configure_project("TANG", data_root=str(tmp_path), overwrite=True)
        session.paths = NBSessionPaths("t-jg-01-20230101", tmp_path, "TANG")

        ang = session.load("ang")
        assert ang._data is not None
        assert ang._data.shape == (100, 3, 3, 3)


# ═══════════════════════════════════════════════════════════════════════════ #
# NBDufr tests                                                               #
# ═══════════════════════════════════════════════════════════════════════════ #

class TestNBDufr:
    def _make_spk(self):
        import numpy as np
        from neurobox.dtype.spikes import NBSpk
        rng = np.random.default_rng(7)
        n_spikes = 500
        res = np.sort(rng.uniform(0, 60, n_spikes))
        # 3 units: 2, 3, 4
        clu = rng.choice([2, 3, 4], size=n_spikes).astype(np.int32)
        map_ = np.array([[2,1,2],[3,1,3],[4,1,4]], dtype=np.int64)
        return NBSpk(res, clu, map_, samplerate=20000.)

    def test_shape(self):
        from neurobox.dtype.ufr import NBDufr
        spk = self._make_spk()
        ufr = NBDufr.compute(spk, samplerate=1250., duration_sec=60.)
        assert ufr._data is not None
        assert ufr._data.shape[0] == 60 * 1250
        assert ufr._data.shape[1] == 3   # 3 units

    def test_unit_ids(self):
        import numpy as np
        from neurobox.dtype.ufr import NBDufr
        spk = self._make_spk()
        ufr = NBDufr.compute(spk, samplerate=1250., duration_sec=60.)
        np.testing.assert_array_equal(np.sort(ufr.unit_ids), [2, 3, 4])

    def test_rates_are_nonneg(self):
        import numpy as np
        from neurobox.dtype.ufr import NBDufr
        spk = self._make_spk()
        ufr = NBDufr.compute(spk, samplerate=1250., duration_sec=60.)
        assert np.all(ufr._data >= 0)

    def test_mean_rate_plausible(self):
        import numpy as np
        from neurobox.dtype.ufr import NBDufr
        spk = self._make_spk()
        ufr = NBDufr.compute(spk, samplerate=1250., duration_sec=60.)
        # Mean rate should be in the ballpark of true rate (not NaN)
        assert np.all(np.isfinite(ufr._data))

    def test_boxcar_mode(self):
        from neurobox.dtype.ufr import NBDufr
        spk = self._make_spk()
        ufr = NBDufr.compute(spk, samplerate=1250., duration_sec=60., mode="boxcar")
        assert ufr._data.shape[1] == 3

    def test_count_mode(self):
        import numpy as np
        from neurobox.dtype.ufr import NBDufr
        spk = self._make_spk()
        ufr = NBDufr.compute(spk, samplerate=1250., duration_sec=60., mode="count")
        # count mode: values are raw spike counts, all non-negative integers
        assert np.all(ufr._data >= 0)
        assert np.all(ufr._data == np.floor(ufr._data))

    def test_subset_units(self):
        import numpy as np
        from neurobox.dtype.ufr import NBDufr
        spk = self._make_spk()
        ufr = NBDufr.compute(spk, samplerate=1250., duration_sec=60., units=[2])
        assert ufr._data.shape[1] == 1
        np.testing.assert_array_equal(ufr.unit_ids, [2])

    def test_rates_for_api(self):
        from neurobox.dtype.ufr import NBDufr
        spk = self._make_spk()
        ufr = NBDufr.compute(spk, samplerate=1250., duration_sec=60.)
        r2 = ufr.rates_for(2)
        assert r2.shape == (60 * 1250,)

    def test_epoch_getitem(self):
        """ufr[epoch] selects time samples via NBData.__getitem__."""
        import numpy as np
        from neurobox.dtype.ufr import NBDufr
        from neurobox.dtype.epoch import NBEpoch
        spk = self._make_spk()
        ufr = NBDufr.compute(spk, samplerate=1250., duration_sec=60.)
        ep = NBEpoch(np.array([[10., 20.]]), samplerate=1.0)
        chunk = ufr[ep]
        assert chunk.shape[0] == 10 * 1250   # 10 s at 1250 Hz
        assert chunk.shape[1] == 3

    def test_empty_spk_returns_empty(self):
        import numpy as np
        from neurobox.dtype.spikes import NBSpk
        from neurobox.dtype.ufr import NBDufr
        spk = NBSpk(samplerate=20000.)
        ufr = NBDufr.compute(spk, samplerate=1250., duration_sec=60.)
        assert ufr._data is None or ufr.n_units == 0


class TestSessionParAutoLoad:
    """par is loaded automatically whenever a session is opened."""

    def _write_yaml(self, path, session_name):
        """Write yaml directly into path/ so load_par(path/session_name) finds it."""
        import yaml
        path.mkdir(parents=True, exist_ok=True)
        par_data = {
            "acquisitionSystem": {
                "nChannels": 8, "nBits": 16,
                "samplingRate": 20000,
            },
            "fieldPotentials": {"lfpSamplingRate": 1250},
            "spikeDetection": {"channelGroups": [
                {"channels": [0,1,2,3,4,5,6,7],
                 "nSamples": 32, "peakSampleIndex": 16, "nFeatures": 3}
            ]},
        }
        with open(path / f"{session_name}.yaml", "w") as f:
            yaml.dump(par_data, f)

    def test_par_loaded_on_fresh_session(self, tmp_path):
        from neurobox.dtype.session import NBSession
        from neurobox.config.config import configure_project
        from neurobox.dtype.paths import NBSessionPaths

        configure_project("TPAR", data_root=str(tmp_path), overwrite=True)
        session_name = "src-jg-01-20230101"

        # Write yaml into the project spath (simulates symlinked session)
        spath = tmp_path / "project" / "TPAR" / session_name
        self._write_yaml(spath, session_name)

        session = NBSession.__new__(NBSession)
        session.name      = session_name
        session.maze      = "cof"
        session.trial     = "all"
        session.filebase  = f"{session_name}.cof.all"
        session.spath     = spath
        session.paths     = NBSessionPaths(session_name, tmp_path, "TPAR")
        session._data_root    = tmp_path
        session._project_id   = "TPAR"
        session.par = session.spk = session.lfp = session.xyz = None
        session.stc = session.ang = session.ufr = None
        session.sync = None
        session.samplerate = None
        session.nq = {}
        session.meta = {}

        session._init_par()
        assert session.par is not None
        assert session.samplerate == 20000.0

    def test_par_loaded_after_ses_pkl_restore(self, tmp_path):
        """par is None in the pickle but auto-loaded from YAML on open."""
        import pickle
        from neurobox.dtype.session import NBSession
        from neurobox.config.config import configure_project
        from neurobox.dtype.paths import NBSessionPaths
        from neurobox.dtype.epoch import NBEpoch
        import numpy as np

        configure_project("TPAR2", data_root=str(tmp_path), overwrite=True)
        session_name = "src-jg-01-20230102"
        spath = tmp_path / "project" / "TPAR2" / session_name
        spath.mkdir(parents=True, exist_ok=True)
        self._write_yaml(spath, session_name)

        # Write a minimal .ses.pkl with par=None (as if saved before YAML existed)
        ses_file = spath / f"{session_name}.cof.all.ses.pkl"
        state = {
            "name":      session_name,
            "maze":      "cof",
            "trial":     "all",
            "filebase":  f"{session_name}.cof.all",
            "spath":     spath,
            "paths":     NBSessionPaths(session_name, tmp_path, "TPAR2"),
            "_data_root":    tmp_path,
            "_project_id":   "TPAR2",
            "par":       None,   # ← not saved
            "samplerate": None,
            "sync":      None,
            "xyz":       None,
            "lfp":       None,
            "spk":       None,
            "stc":       None,
            "ang":       None,
            "ufr":       None,
            "nq":        {},
            "meta":      {},
        }
        with open(ses_file, "wb") as f:
            pickle.dump(state, f)

        # Now open via NBSession — should auto-load par
        session = NBSession.__new__(NBSession)
        session._data_root    = tmp_path
        session._project_id   = "TPAR2"
        session._load_ses_file(ses_file)

        assert session.par is not None, "par should be auto-loaded from YAML"
        assert int(session.par.acquisitionSystem.samplingRate) == 20000

    def test_save_excludes_par(self, tmp_path):
        """save() never writes par into the pickle."""
        import pickle
        from neurobox.dtype.session import NBSession
        from neurobox.config.config import configure_project
        from neurobox.dtype.paths import NBSessionPaths

        configure_project("TPAR3", data_root=str(tmp_path), overwrite=True)
        session_name = "src-jg-01-20230103"
        spath = tmp_path / "project" / "TPAR3" / session_name
        spath.mkdir(parents=True, exist_ok=True)
        self._write_yaml(spath, session_name)

        session = NBSession.__new__(NBSession)
        session.name         = session_name
        session.maze         = "cof"
        session.trial        = "all"
        session.filebase     = f"{session_name}.cof.all"
        session.spath        = spath
        session.paths        = NBSessionPaths(session_name, tmp_path, "TPAR3")
        session._data_root   = tmp_path
        session._project_id  = "TPAR3"
        session.sync = session.xyz = session.lfp = session.spk = None
        session.stc = session.ang = session.ufr = None
        session.nq = {}; session.meta = {}; session.samplerate = None

        # Give it a par so we can check it gets excluded
        session._init_par()
        assert session.par is not None

        session.save()

        ses_file = spath / f"{session_name}.cof.all.ses.pkl"
        with open(ses_file, "rb") as f:
            state = pickle.load(f)

        assert "par" not in state, "par must not be stored in .ses.pkl"


# ═══════════════════════════════════════════════════════════════════════════ #
# transform_origin tests                                                      #
# ═══════════════════════════════════════════════════════════════════════════ #

class TestTransformOrigin:
    def _make_xyz(self, T=300, pitch_amp=0.0, yaw_deg=0.0):
        """Simple xyz with animal moving along +x, optional pitch/yaw."""
        import numpy as np
        from neurobox.dtype.xyz import NBDxyz
        from neurobox.dtype.model import NBModel

        t = np.linspace(0, 5, T)
        data = np.zeros((T, 4, 3))
        yaw_rad = np.radians(yaw_deg)

        # head_back at origin moving along x
        data[:, 0, 0] = t * 5
        # head_front 30 mm ahead in heading direction
        data[:, 1, 0] = t * 5 + 30 * np.cos(yaw_rad)
        data[:, 1, 1] = 30 * np.sin(yaw_rad)
        # head_left/right lateral markers
        data[:, 2, 1] = -15
        data[:, 3, 1] =  15
        # Add pitch oscillation
        data[:, 1, 2] += pitch_amp * np.sin(2 * np.pi * 0.5 * t)

        xyz = NBDxyz(samplerate=120.0)
        xyz._data = data.copy()
        xyz.model = NBModel(["head_back", "head_front", "head_left", "head_right"])
        return xyz

    def test_output_shapes(self):
        import numpy as np
        from neurobox.analysis.transform_origin import transform_origin
        xyz = self._make_xyz()
        r = transform_origin(xyz, low_pass_hz=None)
        T = 300
        assert r.direction.shape   == (T,)
        assert r.pitch.shape       == (T,)
        assert r.roll.shape        == (T,)
        assert r.ori_vector.shape  == (T, 3)

    def test_forward_heading_direction_zero(self):
        """Animal facing +x → mean yaw ≈ 0."""
        import numpy as np
        from neurobox.analysis.transform_origin import transform_origin
        xyz = self._make_xyz(yaw_deg=0.0)
        r = transform_origin(xyz, low_pass_hz=None)
        assert np.abs(np.nanmean(r.direction)) < 0.05  # < ~3 degrees

    def test_90deg_heading(self):
        """Animal facing +y → mean yaw ≈ π/2."""
        import numpy as np
        from neurobox.analysis.transform_origin import transform_origin
        xyz = self._make_xyz(yaw_deg=90.0)
        r = transform_origin(xyz, low_pass_hz=None)
        assert np.abs(np.nanmean(r.direction) - np.pi / 2) < 0.1

    def test_pitch_zero_for_flat_trajectory(self):
        """Flat trajectory (no z motion) → mean pitch ≈ 0."""
        import numpy as np
        from neurobox.analysis.transform_origin import transform_origin
        xyz = self._make_xyz(pitch_amp=0.0)
        r = transform_origin(xyz, low_pass_hz=None)
        assert np.abs(np.nanmean(r.pitch)) < 0.05

    def test_pitch_nonzero_for_nodding(self):
        """Head nodding → non-trivial pitch range."""
        import numpy as np
        from neurobox.analysis.transform_origin import transform_origin
        xyz = self._make_xyz(pitch_amp=15.0)
        r = transform_origin(xyz, low_pass_hz=None)
        assert (r.pitch.max() - r.pitch.min()) > 0.1   # at least ~6 degrees range

    def test_trans_coords_shape(self):
        """With vector_markers the trans_coords array has correct shape."""
        import numpy as np
        from neurobox.analysis.transform_origin import transform_origin
        xyz = self._make_xyz()
        r = transform_origin(
            xyz,
            vector_markers=["head_left", "head_right"],
            low_pass_hz=None,
        )
        assert r.trans_coords is not None
        assert r.trans_coords.shape == (300, 2, 3)

    def test_no_vector_markers_trans_coords_none(self):
        from neurobox.analysis.transform_origin import transform_origin
        xyz = self._make_xyz()
        r = transform_origin(xyz, vector_markers=[], low_pass_hz=None)
        assert r.trans_coords is None

    def test_ori_vector_approximately_x_axis(self):
        """After full de-rotation, y and z components should be near zero."""
        import numpy as np
        from neurobox.analysis.transform_origin import transform_origin
        xyz = self._make_xyz(yaw_deg=45.0, pitch_amp=10.0)
        r = transform_origin(xyz, low_pass_hz=None)
        # After yaw + pitch removal, y and z should be near zero
        # (de-rotated vector lies along +x)
        assert np.abs(np.nanmean(r.ori_vector[:, 1])) < 5.0   # y near zero
        assert np.abs(np.nanmean(r.ori_vector[:, 2])) < 5.0   # z near zero

    def test_import_from_analysis(self):
        from neurobox.analysis import transform_origin, TransformResult
        assert callable(transform_origin)

    def test_import_from_top_level(self):
        import neurobox
        assert hasattr(neurobox, "transform_origin")
        assert hasattr(neurobox, "TransformResult")


# ═══════════════════════════════════════════════════════════════════════════ #
# Spectral analysis tests                                                     #
# ═══════════════════════════════════════════════════════════════════════════ #

class TestSpectralParams:
    def test_defaults(self):
        from neurobox.analysis.lfp.spectral import SpectralParams
        p = SpectralParams()
        assert p.samplerate == 1250.0
        assert p.n_tapers == 5           # 2*3 - 1
        assert p.step == 512

    def test_for_lfp(self):
        from neurobox.analysis.lfp.spectral import SpectralParams
        p = SpectralParams.for_lfp(1250.)
        assert p.freq_range == (1.0, 140.0)
        assert p.win_len == 1024

    def test_for_xyz(self):
        from neurobox.analysis.lfp.spectral import SpectralParams
        p = SpectralParams.for_xyz(120.)
        assert p.samplerate == 120.
        f = p.freq_axis()
        assert f[0] >= 0.1 and f[-1] <= 50.0

    def test_freq_axis_within_range(self):
        import numpy as np
        from neurobox.analysis.lfp.spectral import SpectralParams
        p = SpectralParams.for_lfp(1250.)
        f = p.freq_axis()
        assert f[0] >= p.freq_range[0]
        assert f[-1] <= p.freq_range[1]

    def test_time_resolution(self):
        from neurobox.analysis.lfp.spectral import SpectralParams
        p = SpectralParams(samplerate=1250., win_len=1024, n_overlap=512)
        assert abs(p.time_resolution - 512/1250.) < 1e-9


class TestMultitaperSpectrogram:
    def _make_signal(self, T=15000, sr=1250., seed=0):
        import numpy as np
        rng = np.random.default_rng(seed)
        t = np.arange(T) / sr
        sig = np.column_stack([
            np.sin(2*np.pi*8*t) + rng.standard_normal(T)*0.2,
            rng.standard_normal(T),
        ])
        return sig, sr

    def test_output_shape(self):
        from neurobox.analysis.lfp.spectral import SpectralParams, multitaper_spectrogram
        x, sr = self._make_signal()
        p = SpectralParams.for_lfp(sr)
        r = multitaper_spectrogram(x, p)
        assert r.power.ndim == 3                # (T, F, C)
        assert r.power.shape[2] == 2            # 2 channels
        assert len(r.freqs) == r.power.shape[1]
        assert len(r.times) == r.power.shape[0]

    def test_power_is_nonneg(self):
        import numpy as np
        from neurobox.analysis.lfp.spectral import SpectralParams, multitaper_spectrogram
        x, sr = self._make_signal()
        r = multitaper_spectrogram(x, p := SpectralParams.for_lfp(sr))
        assert np.all(r.power >= 0)

    def test_theta_peak_detected(self):
        import numpy as np
        from neurobox.analysis.lfp.spectral import SpectralParams, multitaper_spectrogram
        x, sr = self._make_signal(T=75000)   # 60 sec
        r = multitaper_spectrogram(x[:, 0], SpectralParams.for_lfp(sr))
        peak_f = r.freqs[np.argmax(r.power.mean(axis=0).squeeze())]
        assert 6.0 <= peak_f <= 10.0, f"Expected theta peak, got {peak_f:.1f} Hz"

    def test_1d_input(self):
        import numpy as np
        from neurobox.analysis.lfp.spectral import SpectralParams, multitaper_spectrogram
        x, sr = self._make_signal()
        r = multitaper_spectrogram(x[:, 0], SpectralParams.for_lfp(sr))
        assert r.power.shape[2] == 1

    def test_freq_range_respected(self):
        import numpy as np
        from neurobox.analysis.lfp.spectral import SpectralParams, multitaper_spectrogram
        x, sr = self._make_signal()
        p = SpectralParams.for_lfp(sr)
        r = multitaper_spectrogram(x, p)
        assert r.freqs[0] >= p.freq_range[0]
        assert r.freqs[-1] <= p.freq_range[1]


class TestMultitaperCoherogram:
    def _make_coherent_signal(self, T=75000, sr=1250., seed=0):
        import numpy as np
        rng = np.random.default_rng(seed)
        t = np.arange(T) / sr
        # ch0, ch1 share theta; ch2 is pure noise
        return np.column_stack([
            np.sin(2*np.pi*8*t) + rng.standard_normal(T)*0.3,
            np.sin(2*np.pi*8*t + np.pi/4) + rng.standard_normal(T)*0.3,
            rng.standard_normal(T),
        ]), sr

    def test_output_shape(self):
        from neurobox.analysis.lfp.spectral import SpectralParams, multitaper_coherogram
        x, sr = self._make_coherent_signal()
        r = multitaper_coherogram(x, SpectralParams.for_lfp(sr))
        T, F, C = r.power.shape
        assert r.coherence.shape == (T, F, C, C)
        assert r.phase.shape     == (T, F, C, C)

    def test_self_coherence_is_one(self):
        import numpy as np
        from neurobox.analysis.lfp.spectral import SpectralParams, multitaper_coherogram
        x, sr = self._make_coherent_signal()
        r = multitaper_coherogram(x, SpectralParams.for_lfp(sr))
        # Diagonal should be 1 (self-coherence)
        for c in range(3):
            assert np.allclose(r.coherence[:, :, c, c], 1.0, atol=1e-6)

    def test_theta_coherence_high(self):
        """Time-averaged theta-band coherence between two theta channels."""
        import numpy as np
        from neurobox.analysis.lfp.spectral import (
            SpectralParams, multitaper_psd
        )
        x, sr = self._make_coherent_signal(T=100000)
        r = multitaper_psd(x, SpectralParams.for_lfp(sr), average=True)
        theta = (r.freqs >= 6) & (r.freqs <= 12)
        coh_01 = r.coherence[theta, 0, 1].mean()
        coh_02 = r.coherence[theta, 0, 2].mean()
        assert coh_01 > 0.7, f"Theta-theta coherence = {coh_01:.3f}"
        assert coh_02 < 0.2, f"Theta-noise coherence = {coh_02:.3f}"

    def test_phase_encodes_delay(self):
        """Phase between ch0 and ch1 (θ₁ = θ₀ + π/4) should be ≈ -45°."""
        import numpy as np
        from neurobox.analysis.lfp.spectral import SpectralParams, multitaper_coherogram
        x, sr = self._make_coherent_signal(T=100000)
        r = multitaper_coherogram(x, SpectralParams.for_lfp(sr))
        theta = (r.freqs >= 7) & (r.freqs <= 9)
        mean_phase = np.degrees(np.angle(
            np.exp(1j * r.phase[:, theta, 0, 1]).mean()
        ))
        assert abs(mean_phase - (-45)) < 10, f"Phase = {mean_phase:.1f} deg"


class TestMultitaperPSD:
    def test_averaged_shape(self):
        import numpy as np
        from neurobox.analysis.lfp.spectral import SpectralParams, multitaper_psd
        rng = np.random.default_rng(0)
        x = rng.standard_normal((10000, 3))
        r = multitaper_psd(x, SpectralParams.for_lfp(1250.))
        assert r.power.shape  == (len(r.freqs), 3)
        assert r.coherence.shape == (len(r.freqs), 3, 3)


class TestWhitenAR:
    def test_output_shape(self):
        import numpy as np
        from neurobox.analysis.lfp.spectral import whiten_ar
        x = np.random.default_rng(0).standard_normal((5000, 4))
        y, a = whiten_ar(x)
        assert y.shape == x.shape
        assert len(a) == 3   # [1, a1, a2]

    def test_whitened_flatter_spectrum(self):
        """Whitening a coloured signal should reduce its spectral tilt."""
        import numpy as np
        from scipy.signal.windows import dpss
        from neurobox.analysis.lfp.spectral import whiten_ar, SpectralParams, multitaper_spectrogram
        rng = np.random.default_rng(42)
        # Low-pass coloured signal
        t  = np.arange(10000)
        x  = np.cumsum(rng.standard_normal(10000))   # integrated noise ∝ 1/f
        y, _ = whiten_ar(x)
        p  = SpectralParams.for_lfp(1250.)
        rx = multitaper_spectrogram(x, p)
        ry = multitaper_spectrogram(y, p)
        # Spectral slope: whitened signal should be flatter
        psd_x = rx.power.mean(0).squeeze()
        psd_y = ry.power.mean(0).squeeze()
        slope_x = np.polyfit(np.log(rx.freqs + 1), np.log(psd_x + 1e-20), 1)[0]
        slope_y = np.polyfit(np.log(ry.freqs + 1), np.log(psd_y + 1e-20), 1)[0]
        assert slope_y > slope_x, "Whitened signal should have flatter spectrum"


class TestFetSpec:
    def test_output_shape(self):
        import numpy as np
        from neurobox.analysis.lfp.spectral import SpectralParams, fet_spec
        rng = np.random.default_rng(0)
        x = rng.standard_normal((10000, 2))
        p = SpectralParams.for_lfp(1250.)
        feat, result = fet_spec(x, 1250., params=p, whiten=False)
        assert feat.ndim == 3
        assert feat.shape[2] == 2

    def test_padded_output_length(self):
        import numpy as np
        from neurobox.analysis.lfp.spectral import SpectralParams, fet_spec
        rng = np.random.default_rng(0)
        x = rng.standard_normal((5000, 1))
        p = SpectralParams.for_lfp(1250.)
        feat, result = fet_spec(x, 1250., params=p, whiten=False, pad_to=100)
        assert feat.shape[0] == 100

    def test_returns_spectrum_result(self):
        import numpy as np
        from neurobox.analysis.lfp.spectral import SpectralParams, fet_spec, SpectrumResult
        x = np.random.default_rng(0).standard_normal((5000,))
        _, result = fet_spec(x, 1250., whiten=False)
        assert isinstance(result, SpectrumResult)

    def test_top_level_import(self):
        import neurobox
        assert hasattr(neurobox, 'SpectralParams')
        assert hasattr(neurobox, 'multitaper_spectrogram')
        assert hasattr(neurobox, 'fet_spec')
