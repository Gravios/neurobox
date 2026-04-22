"""Tests for new mocap I/O and gap-filling utilities."""

import tempfile
import types
from pathlib import Path

import numpy as np
import pytest


# ── load_processed_mat ───────────────────────────────────────────────────── #

@pytest.fixture
def mat_file(tmp_path):
    """Write a minimal MTA-format .mat file."""
    try:
        import scipy.io as sio
    except ImportError:
        pytest.skip("scipy not available")

    n_frames, n_markers = 240, 4
    xyzpos  = np.random.randn(n_frames, n_markers, 3).astype(np.float64) * 100
    markers = np.array(
        ["head_front", "head_back", "head_left", "head_right"],
        dtype=object,
    )
    sampleRate = np.float64(120.0)

    f = tmp_path / "session_trial001.mat"
    sio.savemat(str(f), {
        "xyzpos":     xyzpos,
        "markers":    markers,
        "sampleRate": sampleRate,
    })
    return f, xyzpos, list(markers), float(sampleRate)


def test_load_processed_mat_shape(mat_file):
    from neurobox.io import load_processed_mat
    f, orig, _, _ = mat_file
    xyz, mkrs, sr = load_processed_mat(f)
    assert xyz.shape == orig.shape
    assert xyz.dtype == np.float64


def test_load_processed_mat_markers(mat_file):
    from neurobox.io import load_processed_mat
    f, _, orig_mkrs, _ = mat_file
    _, mkrs, _ = load_processed_mat(f)
    assert mkrs == orig_mkrs


def test_load_processed_mat_samplerate(mat_file):
    from neurobox.io import load_processed_mat
    f, _, _, orig_sr = mat_file
    _, _, sr = load_processed_mat(f)
    assert abs(sr - orig_sr) < 0.1


def test_load_processed_mat_missing_file():
    from neurobox.io import load_processed_mat
    with pytest.raises(FileNotFoundError):
        load_processed_mat("/tmp/nonexistent_abc123.mat")


def test_load_processed_mat_missing_xyzpos(tmp_path):
    try:
        import scipy.io as sio
    except ImportError:
        pytest.skip("scipy not available")
    from neurobox.io import load_processed_mat

    f = tmp_path / "bad.mat"
    sio.savemat(str(f), {"foo": np.zeros((10, 3))})
    with pytest.raises(KeyError):
        load_processed_mat(f)


# ── concatenate_processed_mat ────────────────────────────────────────────── #

@pytest.fixture
def mat_dir(tmp_path):
    try:
        import scipy.io as sio
    except ImportError:
        pytest.skip("scipy not available")

    maze_dir = tmp_path / "cof"
    maze_dir.mkdir()
    markers   = np.array(["head_front", "head_back"], dtype=object)
    sampleRate = np.float64(120.0)

    chunks_orig = []
    for i in range(3):
        n = 200 + i * 20
        xyz = np.random.randn(n, 2, 3).astype(np.float64) * 100
        sio.savemat(str(maze_dir / f"session_trial{i+1:03d}.mat"), {
            "xyzpos":     xyz,
            "markers":    markers,
            "sampleRate": sampleRate,
        })
        chunks_orig.append(xyz)

    return maze_dir, chunks_orig


def test_concatenate_count(mat_dir):
    from neurobox.io import concatenate_processed_mat
    d, orig = mat_dir
    chunks, mkrs, sr = concatenate_processed_mat(d)
    assert len(chunks) == 3


def test_concatenate_shapes(mat_dir):
    from neurobox.io import concatenate_processed_mat
    d, orig = mat_dir
    chunks, _, _ = concatenate_processed_mat(d)
    for c, o in zip(chunks, orig):
        assert c.shape == o.shape


def test_concatenate_empty_dir(tmp_path):
    from neurobox.io import concatenate_processed_mat
    with pytest.raises((FileNotFoundError, RuntimeError)):
        concatenate_processed_mat(tmp_path / "empty")


# ── get_event_times / get_ttl_periods ────────────────────────────────────── #

@pytest.fixture
def evt_session(tmp_path):
    """Build a fake session namespace with a .all.evt file."""
    name  = "testA-jg-01-20240101"
    spath = tmp_path / "project" / "TEST" / name
    spath.mkdir(parents=True)

    evt = spath / f"{name}.all.evt"
    evt.write_text(
        "10000.0\t0x0040 TTL Input\n"
        "12000.0\t0x0000 TTL off\n"
        "30000.0\t0x0040 TTL Input\n"
        "32000.0\t0x0000 TTL off\n"
        "40000.0\tStim on\n"
        "42000.0\tStim off\n"
    )
    s = types.SimpleNamespace(name=name, spath=spath, paths=None)
    return s


def test_get_event_times_returns_seconds(evt_session):
    from neurobox.io import get_event_times
    ts = get_event_times(evt_session, "0x0040")
    assert len(ts) == 2
    np.testing.assert_allclose(ts, [10.0, 30.0])


def test_get_event_times_different_label(evt_session):
    from neurobox.io import get_event_times
    ts = get_event_times(evt_session, "Stim on")
    assert len(ts) == 1
    np.testing.assert_allclose(ts, [40.0])


def test_get_event_times_no_match(evt_session):
    from neurobox.io import get_event_times
    ts = get_event_times(evt_session, "XXXX_MISSING")
    assert len(ts) == 0


def test_get_event_times_raw_ms(evt_session):
    from neurobox.io import get_event_times
    ts = get_event_times(evt_session, "0x0040", as_seconds=False)
    np.testing.assert_allclose(ts, [10000.0, 30000.0])


def test_get_ttl_periods(evt_session):
    from neurobox.io import get_ttl_periods
    p = get_ttl_periods(evt_session, "0x0040", "0x0000")
    assert p.shape == (2, 2)
    np.testing.assert_allclose(p[0], [10.0, 12.0])
    np.testing.assert_allclose(p[1], [30.0, 32.0])


def test_get_event_times_from_path(evt_session):
    from neurobox.io import get_event_times
    evt_path = evt_session.spath / f"{evt_session.name}.all.evt"
    ts = get_event_times(str(evt_path), "0x0040")
    assert len(ts) == 2


# ── detect_gaps / fill_gaps ──────────────────────────────────────────────── #

def test_detect_gaps_all_valid():
    from neurobox.io import detect_gaps
    arr = np.random.randn(100, 3, 3).astype(np.float64) * 10
    mask, periods = detect_gaps(arr)
    assert not mask.any()
    assert periods.shape == (0, 2)


def test_detect_gaps_identifies_zeros():
    from neurobox.io import detect_gaps
    arr = np.random.randn(100, 3, 3).astype(np.float64) * 10
    arr[20:30] = 0.0
    mask, periods = detect_gaps(arr)
    assert mask[25]
    assert not mask[10]
    assert periods.shape[0] == 1
    assert periods[0, 0] == 20
    assert periods[0, 1] == 30


def test_detect_gaps_multiple():
    from neurobox.io import detect_gaps
    arr = np.random.randn(200, 2, 3).astype(np.float64) * 10
    arr[10:15]  = 0.0
    arr[100:110] = 0.0
    _, periods = detect_gaps(arr)
    assert periods.shape[0] == 2


def test_fill_gaps_short_gap():
    from neurobox.io import fill_gaps
    arr = np.ones((200, 2, 3), dtype=np.float64) * 5.0
    arr[50:55] = 0.0       # 5-frame gap (< 0.5 s at 120 Hz)

    filled = fill_gaps(arr, samplerate=120.0, max_gap_sec=0.5)
    # Gap should be filled — no longer zero
    assert filled[50:55].max() > 1e-6


def test_fill_gaps_long_gap_untouched():
    from neurobox.io import fill_gaps, detect_gaps
    arr = np.ones((400, 2, 3), dtype=np.float64) * 5.0
    arr[50:150] = 0.0      # 100-frame gap = 0.83 s at 120 Hz

    filled = fill_gaps(arr, samplerate=120.0, max_gap_sec=0.5)
    _, periods = detect_gaps(filled)
    # Gap should still be present (too long to fill)
    assert periods.shape[0] >= 1


def test_fill_gaps_returns_copy_by_default():
    from neurobox.io import fill_gaps
    arr = np.ones((100, 2, 3), dtype=np.float64)
    arr[30:35] = 0.0
    orig = arr.copy()
    filled = fill_gaps(arr, samplerate=120.0, max_gap_sec=0.5)
    np.testing.assert_array_equal(arr, orig)   # original unchanged
    assert filled is not arr


def test_fill_gaps_inplace():
    from neurobox.io import fill_gaps
    arr = np.ones((100, 2, 3), dtype=np.float64) * 3.0
    arr[40:44] = 0.0
    result = fill_gaps(arr, samplerate=120.0, max_gap_sec=0.5, inplace=True)
    assert result is arr


def test_fill_xyz_gaps_nbdxyz():
    from neurobox.io import fill_xyz_gaps
    from neurobox.dtype import NBDxyz, NBModel

    data = np.ones((200, 3, 3), dtype=np.float64) * 4.0
    data[60:65] = 0.0
    xyz = NBDxyz(data=data,
                 model=NBModel(["a", "b", "c"]),
                 samplerate=120.0)

    fill_xyz_gaps(xyz, max_gap_sec=0.5)
    assert xyz.data[62].max() > 1e-6   # gap was filled
