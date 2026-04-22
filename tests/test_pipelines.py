"""Tests for neurobox.pipelines and neurobox.analysis.neuron_quality."""

import csv
import types
import numpy as np
import pytest
from pathlib import Path


# ── Shared fixture ────────────────────────────────────────────────────────── #

@pytest.fixture
def full_session(tmp_path):
    """Build a complete fake session with ephys + mocap + events."""
    from neurobox.dtype.paths import parse_session_name
    from neurobox.config.config import configure_project, link_session
    from neurobox.dtype import NBSession, NBEpoch
    from neurobox.dtype.spikes import NBSpk

    name   = "testA-jg-01-20240101"
    maze   = "cof"
    sr_w   = 20000.0
    lfp_sr = 1250.0
    xyz_sr = 120.0
    n_lfp  = int(60 * lfp_sr)  # 60-second recording

    parts  = parse_session_name(name)
    src_id = parts["sourceId"]
    u_id   = parts["userId"]
    sub_id = parts["subjectId"]
    u, us  = f"{src_id}-{u_id}", f"{src_id}-{u_id}-{sub_id}"

    # ── processed ephys ────────────────────────────────────────────────── #
    eph = tmp_path / "processed" / "ephys" / src_id / u / us / name
    eph.mkdir(parents=True)
    (eph / f"{name}.yaml").write_text(
        f"acquisitionSystem:\n  nChannels: 4\n  nBits: 16\n"
        f"  samplingRate: {sr_w}\nlfpSampleRate: {lfp_sr}\n"
    )
    # LFP: 4-channel, 60 s, sync pulse on ch 0 at 10–12 s and 30–32 s
    lfp_data = np.zeros((n_lfp, 4), dtype=np.int16)
    for t0, t1 in [(10, 12), (30, 32)]:
        lfp_data[int(t0*lfp_sr):int(t1*lfp_sr), 0] = 30000
    lfp_data.tofile(str(eph / f"{name}.lfp"))

    # Spikes
    np.array([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
             dtype="<i8").tofile(str(eph / f"{name}.res.1"))
    np.concatenate([np.array([2], dtype="<i4"),
                    np.array([2,2,3,3,2,3,2,3,2,3], dtype="<i4")]
                   ).tofile(str(eph / f"{name}.clu.1"))

    # Event file: TTL events
    (eph / f"{name}.all.evt").write_text(
        "10000.0\t0x0040 TTL Input\n"
        "12000.0\t0x0000 TTL off\n"
        "30000.0\t0x0040 TTL Input\n"
        "32000.0\t0x0000 TTL off\n"
    )

    # ── processed mocap ────────────────────────────────────────────────── #
    moc = tmp_path / "processed" / "mocap" / src_id / u / us / name / maze
    moc.mkdir(parents=True)

    # Write Motive CSV with full rat marker set
    markers = ["head_front", "head_back", "head_left", "head_right",
               "spine_upper", "spine_middle", "spine_lower", "pelvis_root"]
    for trial_n in range(1, 3):
        n_fr = int(2 * xyz_sr)  # 2-second blocks
        f_out = moc / f"{name}_Trial{trial_n:03d}.csv"
        with open(f_out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Format Version","1.23","","Export Frame Rate",
                        str(xyz_sr),"","Total Frames in Take",str(n_fr),"",
                        "Total Exported Frames",str(n_fr)])
            w.writerow([])
            w.writerow(["Frame","Time"] + [f"Bone"] * (3 * len(markers)))
            model_row = ["", ""]
            for m in markers:
                model_row += [f"Rat:{m}"] * 3
            w.writerow(model_row)
            w.writerow(["",""] + [str(i) for i in range(3 * len(markers))])
            w.writerow(["",""] + ["Position"] * 3 * len(markers))
            w.writerow(["Frame","Time"] + ["X","Y","Z"] * len(markers))
            for fr in range(n_fr):
                # Head z oscillates to create detectable rearing
                row = [str(fr+1), str(round(fr/xyz_sr, 6))]
                for j, m in enumerate(markers):
                    x = float(fr + j)
                    y = float(fr + j + 1)
                    # Head markers get an elevated z for "rearing" frames 30-60
                    z = 50.0 if (m.startswith("head") and 30 <= fr < 60) else 20.0
                    row += [str(x), str(y), str(z)]
                w.writerow(row)

    # ── configure & link ───────────────────────────────────────────────── #
    configure_project("TEST", data_root=tmp_path, dotenv_path=tmp_path / ".env")
    link_session(name, "TEST", data_root=tmp_path, mazes=[maze])

    return tmp_path, name, maze


# ── quick_trial_setup tests ───────────────────────────────────────────────── #

@pytest.fixture
def synced_session(full_session):
    """Return a fully synced NBSession (NLX+Vicon)."""
    from neurobox.dtype import NBSession
    root, name, maze = full_session
    session = NBSession(name, maze=maze, project_id="TEST",
                        data_root=root, overwrite=True)
    session.create(["nlx", "vicon"],
                   ttl_value="0x0040", stop_ttl="0x0000",
                   save_xyz=False)
    return session, root, name, maze


def test_quick_trial_setup_creates_trial(synced_session):
    from neurobox.pipelines import quick_trial_setup
    session, *_ = synced_session
    trial = quick_trial_setup(session, trial_name="all", overwrite=True)
    from neurobox.dtype import NBTrial
    assert isinstance(trial, NBTrial)
    assert trial.trial == "all"


def test_quick_trial_setup_sync_matches_session(synced_session):
    from neurobox.pipelines import quick_trial_setup
    session, *_ = synced_session
    trial = quick_trial_setup(session, overwrite=True)
    assert trial.sync is not None
    # Trial sync should be a subset of session xyz.sync
    sess_t0 = session.xyz.sync._as_periods()[0, 0]
    sess_t1 = session.xyz.sync._as_periods()[-1, 1]
    assert trial.sync._as_periods()[0, 0] >= sess_t0 - 0.1
    assert trial.sync._as_periods()[-1, 1] <= sess_t1 + 0.1


def test_quick_trial_setup_offsets_trim_blocks(synced_session):
    from neurobox.pipelines import quick_trial_setup
    session, *_ = synced_session
    trial_base  = quick_trial_setup(session, trial_name="base", overwrite=True)
    trial_trim  = quick_trial_setup(session, trial_name="trim",
                                    offsets=[1.0, -0.5], overwrite=True)
    # Each block should be shorter by ~1.5 s
    base_dur = float(np.diff(trial_base.sync._as_periods(), axis=1).sum())
    trim_dur = float(np.diff(trial_trim.sync._as_periods(), axis=1).sum())
    n_blocks = trial_base.sync._as_periods().shape[0]
    assert abs(base_dur - trim_dur - n_blocks * 1.5) < 0.5


def test_quick_trial_setup_drop_sync_ind(synced_session):
    from neurobox.pipelines import quick_trial_setup
    session, *_ = synced_session
    n_blocks = session.xyz.sync._as_periods().shape[0]
    if n_blocks < 2:
        pytest.skip("need ≥ 2 sync blocks")
    trial = quick_trial_setup(session, trial_name="drop0",
                              drop_sync_ind=[0], overwrite=True)
    assert trial.sync._as_periods().shape[0] == n_blocks - 1


def test_quick_trial_setup_include_sync_ind(synced_session):
    from neurobox.pipelines import quick_trial_setup
    session, *_ = synced_session
    n_blocks = session.xyz.sync._as_periods().shape[0]
    trial = quick_trial_setup(session, trial_name="inc0",
                              include_sync_ind=[0], overwrite=True)
    assert trial.sync._as_periods().shape[0] == 1


def test_quick_trial_setup_include_overrides_drop(synced_session):
    from neurobox.pipelines import quick_trial_setup
    session, *_ = synced_session
    trial = quick_trial_setup(session, trial_name="inc_over_drop",
                              include_sync_ind=[0],
                              drop_sync_ind=[0, 1],
                              overwrite=True)
    assert trial.sync._as_periods().shape[0] == 1


def test_quick_trial_setup_drop_all_raises(synced_session):
    from neurobox.pipelines import quick_trial_setup
    session, *_ = synced_session
    n = session.xyz.sync._as_periods().shape[0]
    with pytest.raises(RuntimeError, match="All sync blocks"):
        quick_trial_setup(session, trial_name="drop_all",
                          drop_sync_ind=list(range(n)), overwrite=True)


def test_quick_trial_setup_loads_existing(synced_session):
    from neurobox.pipelines import quick_trial_setup
    session, *_ = synced_session
    t1 = quick_trial_setup(session, trial_name="persist", overwrite=True)
    t2 = quick_trial_setup(session, trial_name="persist", overwrite=False)
    # Second load should give same sync as first
    np.testing.assert_allclose(
        t1.sync._as_periods(),
        t2.sync._as_periods(),
        atol=0.01,
    )


# ── batch_trial_setup ─────────────────────────────────────────────────────── #

def test_batch_trial_setup_returns_dict(synced_session):
    from neurobox.pipelines import batch_trial_setup
    from neurobox.dtype import NBTrial
    session, *_ = synced_session
    results = batch_trial_setup([session], trial_name="all", overwrite=True)
    assert session.name in results
    assert isinstance(results[session.name], NBTrial)


# ── neuron_quality ────────────────────────────────────────────────────────── #

@pytest.fixture
def spk_with_waveforms():
    """Synthetic NBSpk with waveforms for quality testing."""
    from neurobox.dtype.spikes import NBSpk

    rng = np.random.default_rng(42)
    sr  = 20000.0
    # 3 units; unit 2 is clean (low ISI), unit 3 is noisy (high ISI)
    t2 = np.sort(rng.uniform(0, 60, 400))   # unit 2: 400 spikes, well-isolated
    t3 = np.sort(rng.uniform(0, 60, 80))    # unit 3: 80 spikes, sparse

    # Introduce ISI violations for unit 3
    t3[:5] = np.arange(5) * 0.0005   # 5 spikes within 2 ms of each other

    res = np.concatenate([t2, t3])
    clu = np.concatenate([np.full(len(t2), 2), np.full(len(t3), 3)]).astype(np.int32)
    order = np.argsort(res)
    res   = res[order]
    clu   = clu[order]
    mp    = np.array([[2, 1], [3, 1]], dtype=np.int64)

    # Fake waveforms: trough-shape for each spike
    n_spk   = len(res)
    n_samp  = 32
    n_chan  = 4
    t_arr   = np.linspace(0, 1, n_samp)
    # Clean gaussian trough shape
    mean_wf = np.zeros((n_samp, n_chan))
    mean_wf[:, 0] = -100 * np.exp(-((t_arr - 0.4) ** 2) / 0.01)   # trough at 40%
    mean_wf[:, 0] += 40  * np.exp(-((t_arr - 0.6) ** 2) / 0.01)   # peak  at 60%
    waveforms = (mean_wf[np.newaxis] + rng.standard_normal((n_spk, n_samp, n_chan)) * 5)
    waveforms = waveforms.astype(np.float32)

    return NBSpk(res, clu, mp, samplerate=sr, spk=waveforms)


def test_nq_returns_all_units(spk_with_waveforms):
    from neurobox.analysis import neuron_quality
    nq = neuron_quality(spk_with_waveforms, duration_sec=60., save=False)
    assert set(nq.keys()) == {2, 3}


def test_nq_isi_low_for_clean_unit(spk_with_waveforms):
    from neurobox.analysis import neuron_quality
    nq = neuron_quality(spk_with_waveforms, duration_sec=60., save=False)
    # Unit 2: random spikes → chance ISI<2ms contamination ~1-3%
    # Unit 3: artificially packed spikes at the start → ISI contamination >> unit 2
    assert nq[2].isi_contamination < nq[3].isi_contamination
    assert nq[3].isi_contamination > 0.04   # 5 spikes within 2 ms of each other


def test_nq_spike_counts(spk_with_waveforms):
    from neurobox.analysis import neuron_quality
    nq = neuron_quality(spk_with_waveforms, duration_sec=60., save=False)
    assert nq[2].n_spikes == 400
    assert nq[3].n_spikes == 80


def test_nq_firing_rate(spk_with_waveforms):
    from neurobox.analysis import neuron_quality
    nq = neuron_quality(spk_with_waveforms, duration_sec=60., save=False)
    assert abs(nq[2].mean_firing_rate - 400/60) < 0.1
    assert nq[3].mean_firing_rate is not None


def test_nq_waveform_metrics_computed(spk_with_waveforms):
    from neurobox.analysis import neuron_quality
    nq = neuron_quality(spk_with_waveforms, duration_sec=60., save=False)
    r  = nq[2]
    assert r.snr is not None  and r.snr > 0
    assert r.spike_width_ms is not None and r.spike_width_ms > 0


def test_nq_is_single_unit_threshold(spk_with_waveforms):
    from neurobox.analysis import neuron_quality
    nq = neuron_quality(spk_with_waveforms, duration_sec=60., save=False)
    # Unit 2: 400 spikes, low ISI → should pass
    assert nq[2].is_single_unit(max_isi_contamination=0.02, min_spikes=100)
    # Unit 3: 80 spikes → should fail (min_spikes=100)
    assert not nq[3].is_single_unit(min_spikes=100)


def test_nq_no_waveforms():
    from neurobox.analysis import neuron_quality
    from neurobox.dtype.spikes import NBSpk
    rng = np.random.default_rng(0)
    t   = np.sort(rng.uniform(0, 60, 300))
    spk = NBSpk(t, np.full(300, 2, dtype=np.int32),
                samplerate=20000., spk=None)
    nq  = neuron_quality(spk, duration_sec=60., save=False)
    assert nq[2].spike_width_ms is None
    assert nq[2].snr is None
    assert nq[2].isi_contamination >= 0


def test_nq_repr():
    from neurobox.analysis import NeuronQualityResult
    r = NeuronQualityResult(unit_id=5, n_spikes=200, isi_contamination=0.01,
                             snr=4.2, spike_width_ms=0.42)
    s = repr(r)
    assert "unit=5" in s
    assert "ISI=0.010" in s


# ── _parse_spec ───────────────────────────────────────────────────────────── #

def test_parse_spec_string():
    from neurobox.pipelines import _parse_spec
    s = _parse_spec("sirotaA-jg-05-20120316")
    assert s["sessionName"] == "sirotaA-jg-05-20120316"


def test_parse_spec_snake_case_aliases():
    from neurobox.pipelines import _parse_spec
    s = _parse_spec({"session_name": "testA-jg-01-20240101",
                     "maze_name": "cof", "ttl_value": "0x0040"})
    assert s["sessionName"] == "testA-jg-01-20240101"
    assert s["mazeName"]    == "cof"
    assert s["ttlValue"]    == "0x0040"


def test_parse_spec_canonical_passthrough():
    from neurobox.pipelines import _parse_spec
    s = _parse_spec({"sessionName": "testA-jg-01-20240101", "mazeName": "nor"})
    assert s["sessionName"] == "testA-jg-01-20240101"
    assert s["mazeName"]    == "nor"
