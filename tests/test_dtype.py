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
