"""Tests for neurobox.io loaders."""

import numpy as np
import pytest
import tempfile
from pathlib import Path


# ── load_par / load_yaml ────────────────────────────────────────────────── #

YAML_TEXT = """\
acquisitionSystem:
  nChannels: 32
  nBits: 16
  samplingRate: 20000
lfpSampleRate: 1250
spikeDetection:
  channelGroups:
    - channels: [0,1,2,3,4,5,6,7]
      nSamples: 32
      peakSampleIndex: 16
      nFeatures: 3
    - channels: [8,9,10,11,12,13,14,15]
      nSamples: 32
      peakSampleIndex: 16
      nFeatures: 3
"""


@pytest.fixture
def yaml_file(tmp_path):
    p = tmp_path / "session.yaml"
    p.write_text(YAML_TEXT)
    return tmp_path / "session"   # base path without extension


def test_load_par_reads_yaml(yaml_file):
    from neurobox.io import load_par
    par = load_par(str(yaml_file))
    assert int(par.acquisitionSystem.nChannels) == 32
    assert float(par.acquisitionSystem.samplingRate) == 20000.0


def test_load_par_missing_raises(tmp_path):
    from neurobox.io import load_par
    with pytest.raises(FileNotFoundError):
        load_par(str(tmp_path / "nonexistent"))


def test_load_yaml_and_get_channel_groups(yaml_file):
    from neurobox.io import load_yaml, get_channel_groups
    par = load_yaml(str(yaml_file) + ".yaml")
    groups = get_channel_groups(par)
    assert groups == [list(range(8)), list(range(8, 16))]


# ── load_clu_res ────────────────────────────────────────────────────────── #

@pytest.fixture
def spike_files(tmp_path):
    base = tmp_path / "session"
    # shank 1: clusters 2,3 (MUA=1 excluded)
    np.array([100, 200, 300, 400, 500], dtype="<i8").tofile(str(base) + ".res.1")
    np.concatenate([
        np.array([2], dtype="<i4"),          # nClusters header
        np.array([2, 2, 3, 3, 1], dtype="<i4"),  # cluster IDs (1=MUA)
    ]).tofile(str(base) + ".clu.1")
    return base


def test_load_clu_res_basic(spike_files):
    from neurobox.io import load_clu_res
    res, clu, smap = load_clu_res(str(spike_files), shank_groups=[1])
    assert len(res) == 4             # MUA excluded
    assert set(np.unique(clu)) == {2, 3}


def test_load_clu_res_sorted(spike_files):
    from neurobox.io import load_clu_res
    res, _, _ = load_clu_res(str(spike_files), shank_groups=[1])
    assert np.all(np.diff(res) >= 0)


def test_load_clu_res_as_seconds(spike_files):
    from neurobox.io import load_clu_res
    res, _, _ = load_clu_res(str(spike_files), shank_groups=[1],
                              sampling_rate=20000.0, as_seconds=True)
    np.testing.assert_allclose(res[0], 100 / 20000.0)


def test_spikes_by_unit(spike_files):
    from neurobox.io import load_clu_res, spikes_by_unit
    res, clu, _ = load_clu_res(str(spike_files), shank_groups=[1])
    units = spikes_by_unit(res, clu)
    assert set(units.keys()) == {2, 3}
    assert len(units[2]) == 2
    assert len(units[3]) == 2


def test_load_clu_res_multi_shank(tmp_path):
    from neurobox.io import load_clu_res
    base = tmp_path / "session"
    for shank, times, ids in [(1, [100, 200, 300], [2, 3, 2]),
                               (2, [400, 500, 600], [2, 2, 3])]:
        np.array(times, dtype="<i8").tofile(f"{base}.res.{shank}")
        np.concatenate([np.array([2], dtype="<i4"),
                        np.array(ids, dtype="<i4")]).tofile(f"{base}.clu.{shank}")
    res, clu, smap = load_clu_res(str(base), shank_groups=[1, 2])
    assert len(res) == 6
    assert len(np.unique(clu)) == 4   # no ID collision across shanks


# ── load_spk ────────────────────────────────────────────────────────────── #

def test_load_spk_shape(tmp_path):
    from neurobox.io import load_spk
    f = tmp_path / "session.spk.1"
    wf = np.random.randint(-1000, 1000, size=(50, 32, 8), dtype=np.int16)
    wf.tofile(f)
    loaded = load_spk(str(f), n_samples=32, n_channels=8)
    assert loaded.shape == (50, 32, 8)


def test_load_spk_uv_conversion(tmp_path):
    from neurobox.io import load_spk
    f = tmp_path / "session.spk.1"
    np.ones((10, 32, 8), dtype=np.int16).tofile(f)
    loaded = load_spk(str(f), n_samples=32, n_channels=8, uv_per_bit=0.195)
    assert loaded.dtype == np.float32
    np.testing.assert_allclose(loaded[0, 0, 0], 0.195, rtol=1e-5)


def test_load_spk_bad_size_raises(tmp_path):
    from neurobox.io import load_spk
    f = tmp_path / "session.spk.1"
    np.ones(17, dtype=np.int16).tofile(f)   # not a multiple of 32*8
    with pytest.raises(ValueError):
        load_spk(str(f), n_samples=32, n_channels=8)


# ── load_binary ─────────────────────────────────────────────────────────── #

@pytest.fixture
def binary_dat(tmp_path):
    from neurobox.dtype import Struct
    raw = np.arange(40, dtype=np.int16).reshape(10, 4)
    f = tmp_path / "session.dat"
    raw.tofile(f)
    par = Struct()
    acq = Struct()
    par.acquisitionSystem = acq
    acq.nChannels = 4
    acq.nBits = 16
    return f, par


def test_load_binary_shape(binary_dat):
    from neurobox.io import load_binary
    f, par = binary_dat
    data = load_binary(str(f), channels=[0, 2], par=par, channel_first=True)
    assert data.shape == (2, 10)


def test_load_binary_values(binary_dat):
    from neurobox.io import load_binary
    f, par = binary_dat
    data = load_binary(str(f), channels=[0], par=par, channel_first=True)
    # channel 0 of row i = i*4
    expected = np.arange(0, 40, 4, dtype=np.int16)
    np.testing.assert_array_equal(data[0], expected)


def test_load_binary_uv(binary_dat):
    from neurobox.io import load_binary
    f, par = binary_dat
    data = load_binary(str(f), channels=[0, 1], par=par,
                       uv_per_bit=1.0, channel_first=False)
    assert data.dtype == np.float32
    assert data.shape == (10, 2)


def test_load_binary_periods(binary_dat):
    from neurobox.io import load_binary
    f, par = binary_dat
    periods = np.array([[2, 5]], dtype=np.int64)
    data = load_binary(str(f), channels=[0], par=par,
                       periods=periods, channel_first=True)
    assert data.shape == (1, 3)


def test_load_binary_bad_channel_raises(binary_dat):
    from neurobox.io import load_binary
    f, par = binary_dat
    with pytest.raises(ValueError):
        load_binary(str(f), channels=[10], par=par)


# ── load_evt ────────────────────────────────────────────────────────────── #

@pytest.fixture
def evt_file(tmp_path):
    f = tmp_path / "session.evt"
    f.write_text(
        "# comment\n"
        "100.0\tStim on\n"
        "200.0\tStim off\n"
        "300.0\tStim on\n"
        "400.0\tStim off\n"
    )
    return f


def test_load_evt_count(evt_file):
    from neurobox.io import load_evt
    ts, labels = load_evt(str(evt_file))
    assert len(ts) == 4
    assert labels[0] == "Stim on"


def test_load_evt_as_seconds(evt_file):
    from neurobox.io import load_evt
    ts, _ = load_evt(str(evt_file), as_seconds=True)
    np.testing.assert_allclose(ts[0], 0.1)


def test_load_evt_pattern(evt_file):
    from neurobox.io import load_evt
    ts, labels = load_evt(str(evt_file), pattern="Stim on")
    assert len(ts) == 2
    assert all("Stim on" in l for l in labels)


def test_evt_to_periods(evt_file):
    from neurobox.io import load_evt, evt_to_periods
    ts, labels = load_evt(str(evt_file))
    periods = evt_to_periods(ts, labels, "Stim on", "Stim off")
    assert periods.shape == (2, 2)
    np.testing.assert_allclose(periods[0], [100.0, 200.0])
