"""
Tests for neurobox.dtype.sync_pipelines.

These tests exercise the shared helper functions (thresh_cross,
match_chunks_to_windows, build_xyz_array) and the two event-sourcing
paths (TTL events for NLX, pulse channel for OpenEphys) using
fully synthetic data in a temp directory.
"""

import csv
import tempfile
from pathlib import Path

import numpy as np
import pytest

from neurobox.dtype.sync_pipelines import (
    _thresh_cross,
    _find_ttl_windows,
    _match_chunks_to_windows,
    _build_xyz_array,
    _has_paths,
    _find_file,
    dispatch,
)


# ── _thresh_cross ─────────────────────────────────────────────────────────── #

class TestThreshCross:
    def test_two_pulses(self):
        sig = np.zeros(200)
        sig[20:60]   = 2.0
        sig[100:150] = 2.0
        p = _thresh_cross(sig, 1.0)
        assert p.shape == (2, 2)
        assert p[0, 0] == 20  and p[0, 1] == 60
        assert p[1, 0] == 100 and p[1, 1] == 150

    def test_no_crossings(self):
        assert _thresh_cross(np.zeros(100), 1.0).shape == (0, 2)

    def test_starts_high(self):
        # signal starts above threshold — no leading rise, should be ignored
        sig = np.ones(50)
        sig[20:] = 0.0
        p = _thresh_cross(sig, 0.5)
        assert p.shape == (0, 2)

    def test_min_dur_filter(self):
        sig = np.zeros(100)
        sig[10:15] = 2.0   # 5 samples
        sig[40:60] = 2.0   # 20 samples
        p = _thresh_cross(sig, 1.0, min_dur_samples=10)
        assert p.shape == (1, 2)
        assert p[0, 0] == 40

    def test_dtype(self):
        sig = np.zeros(50); sig[10:20] = 2.0
        p = _thresh_cross(sig, 1.0)
        assert p.dtype == np.int64


# ── _find_ttl_windows ─────────────────────────────────────────────────────── #

class TestFindTtlWindows:
    def _make(self, starts, stops):
        n = len(starts) + len(stops)
        ts = np.zeros(n)
        labels = [""] * n
        i = 0
        # Interleave starts and stops by time
        events = (
            [(t, "0x0040") for t in starts] +
            [(t, "0x0000") for t in stops]
        )
        events.sort()
        for j, (t, l) in enumerate(events):
            ts[j]     = t
            labels[j] = l
        return ts, labels

    def test_two_pairs(self):
        ts, lb = self._make([1.0, 5.0], [3.0, 8.0])
        w = _find_ttl_windows(ts, lb, "0x0040", "0x0000")
        assert w.shape == (2, 2)
        np.testing.assert_allclose(w[0], [1.0, 3.0])
        np.testing.assert_allclose(w[1], [5.0, 8.0])

    def test_extra_stop_at_start_ignored(self):
        # A stop before the first start should be dropped
        ts = np.array([0.5, 1.0, 3.0])
        lb = ["0x0000", "0x0040", "0x0000"]
        w = _find_ttl_windows(ts, lb, "0x0040", "0x0000")
        assert w.shape == (1, 2)
        np.testing.assert_allclose(w[0], [1.0, 3.0])

    def test_empty_returns_empty(self):
        w = _find_ttl_windows(np.array([1.0, 2.0]), ["x", "x"])
        assert w.shape == (0, 2)


# ── _match_chunks_to_windows ─────────────────────────────────────────────── #

class TestMatchChunks:
    def _chunk(self, dur_sec, sr=120.0, n_markers=3):
        n = int(round(dur_sec * sr))
        return np.ones((n, n_markers, 3))

    def test_exact_match(self):
        windows = np.array([[1.0, 3.0], [5.0, 8.0]])  # 2s, 3s
        chunks  = [self._chunk(2.0), self._chunk(3.0)]
        mc, mw = _match_chunks_to_windows(chunks, windows, 120.0)
        assert len(mc) == 2
        assert mw.shape == (2, 2)

    def test_one_frame_lead_correction(self):
        # MTA: matched window is shifted back by 1/sr
        windows = np.array([[1.0, 3.0]])
        chunks  = [self._chunk(2.0)]
        _, mw = _match_chunks_to_windows(chunks, windows, 120.0)
        dt = 1.0 / 120.0
        np.testing.assert_allclose(mw[0, 0], 1.0 - dt)
        np.testing.assert_allclose(mw[0, 1], 3.0 - dt)

    def test_no_match_returns_empty(self):
        windows = np.array([[1.0, 3.0]])
        chunks  = [self._chunk(1.5)]   # duration mismatch > 0.2 s
        mc, mw = _match_chunks_to_windows(chunks, windows, 120.0, tolerance_sec=0.2)
        assert mc == []
        assert mw.shape == (0, 2)

    def test_custom_tolerance(self):
        windows = np.array([[1.0, 3.0]])
        chunks  = [self._chunk(2.15)]  # 0.15 s off
        mc, mw = _match_chunks_to_windows(chunks, windows, 120.0, tolerance_sec=0.3)
        assert len(mc) == 1

    def test_extra_chunk_unmatched(self):
        windows = np.array([[1.0, 3.0]])
        chunks  = [self._chunk(2.0), self._chunk(2.0)]
        mc, mw = _match_chunks_to_windows(chunks, windows, 120.0)
        assert len(mc) == 1   # second chunk has no window

    def test_each_window_used_once(self):
        windows = np.array([[1.0, 3.0], [5.0, 7.0]])  # both 2 s
        chunks  = [self._chunk(2.0), self._chunk(2.0)]
        mc, mw = _match_chunks_to_windows(chunks, windows, 120.0)
        assert len(mc) == 2
        # Windows must be different
        assert mw[0, 0] != mw[1, 0]


# ── _build_xyz_array ─────────────────────────────────────────────────────── #

class TestBuildXyzArray:
    def test_single_chunk(self):
        mw     = np.array([[1.0, 3.0]])
        chunk  = np.ones((240, 2, 3))      # 2 s at 120 Hz
        arr    = _build_xyz_array([chunk], mw, 120.0)
        assert arr.shape == (240, 2, 3)
        assert arr[0, 0, 0] > 0            # sentinel replaces zeros

    def test_two_chunks_gap_zeroed(self):
        mw = np.array([[1.0, 3.0], [5.0, 7.0]])
        c1 = np.ones((240, 2, 3)) * 1.0
        c2 = np.ones((240, 2, 3)) * 2.0
        arr = _build_xyz_array([c1, c2], mw, 120.0)
        # Span 1..7 = 6s, total_frames = ceil(6*120) = 720
        assert arr.shape[0] == 720
        # Gap between 3 and 5 s → frames 240..480 should be zero
        assert arr[300, 0, 0] == 0.0
        # c1 region: frames 0..240 → eps-filled (was 1.0, sentinel applied to orig zeros only)
        # c1 values are 1.0 so no sentinel needed
        assert arr[100, 0, 0] == 1.0
        # c2 region
        assert arr[600, 0, 0] == 2.0

    def test_in_chunk_zeros_become_eps(self):
        mw    = np.array([[0.0, 1.0]])
        chunk = np.zeros((120, 1, 3))      # all zeros
        arr   = _build_xyz_array([chunk], mw, 120.0)
        assert (arr != 0.0).all()
        assert (arr == np.finfo(np.float32).eps).all()


# ── Integration: sync_nlx_vicon (synthetic session) ──────────────────────── #

class TestSyncNlxVicon:
    """
    Build a complete minimal fake session directory and call
    sync_nlx_vicon.  Verifies that session fields are populated
    correctly without touching real data.
    """

    @pytest.fixture
    def fake_session(self, tmp_path):
        """Create a minimal fake session wired up with NBSessionPaths."""
        import types
        from neurobox.dtype.paths import NBSessionPaths

        name = "testA-jg-01-20240101"
        sr_wide = 20000.0
        lfp_sr  = 1250.0
        xyz_sr  = 120.0

        # ── YAML par ───────────────────────────────────────────────── #
        eph_dir = (tmp_path / "processed" / "ephys" / "testA"
                   / "testA-jg" / "testA-jg-01" / name)
        eph_dir.mkdir(parents=True)
        (eph_dir / f"{name}.yaml").write_text(
            f"acquisitionSystem:\n"
            f"  nChannels: 4\n"
            f"  nBits: 16\n"
            f"  samplingRate: {sr_wide}\n"
            f"lfpSampleRate: {lfp_sr}\n"
        )

        # ── LFP binary: 60 s, 4 ch, 1250 Hz ──────────────────────── #
        n_lfp = int(60 * lfp_sr)
        lfp_data = np.zeros((n_lfp, 4), dtype=np.int16)
        # Put a 2 s sync pulse on ch 0 between 10–12 s and 30–32 s
        for t_start, t_stop in [(10, 12), (30, 32)]:
            i0 = int(t_start * lfp_sr)
            i1 = int(t_stop  * lfp_sr)
            lfp_data[i0:i1, 0] = 30000
        lfp_data.tofile(str(eph_dir / f"{name}.lfp"))

        # ── .res.1 / .clu.1: 10 fake spikes ───────────────────────── #
        np.array([100, 200, 300, 400, 500,
                  600, 700, 800, 900, 1000], dtype="<i8").tofile(
            str(eph_dir / f"{name}.res.1"))
        np.concatenate([
            np.array([2], dtype="<i4"),
            np.array([2, 2, 3, 3, 2, 3, 2, 3, 2, 3], dtype="<i4"),
        ]).tofile(str(eph_dir / f"{name}.clu.1"))

        # ── .all.evt: TTL events ───────────────────────────────────── #
        evt_lines = (
            "10000.0\t0x0040 TTL Input\n"
            "12000.0\t0x0000 TTL off\n"
            "30000.0\t0x0040 TTL Input\n"
            "32000.0\t0x0000 TTL off\n"
        )
        (eph_dir / f"{name}.all.evt").write_text(evt_lines)

        # ── Motive CSVs in source/mocap ────────────────────────────── #
        src_dir = (tmp_path / "source" / "mocap" / "testA"
                   / "testA-jg" / "testA-jg-01" / name)
        src_dir.mkdir(parents=True)

        def write_csv(path, n_frames, marker_names, sr):
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    "Format Version", "1.23", "",
                    "Export Frame Rate", str(sr), "",
                    "Total Frames in Take", str(n_frames), "",
                    "Total Exported Frames", str(n_frames),
                ])
                w.writerow([])
                col_types = ["Frame", "Time"]
                for _ in marker_names:
                    col_types += ["Bone", "Bone", "Bone"]
                w.writerow(col_types)
                model_row = ["", ""]
                for mn in marker_names:
                    model_row += [f"Rat:{mn}"] * 3
                w.writerow(model_row)
                w.writerow(["", ""] + [str(i) for i in range(len(marker_names) * 3)])
                w.writerow(["", ""] + ["Position"] * len(marker_names) * 3)
                w.writerow(["Frame", "Time"] + ["X", "Y", "Z"] * len(marker_names))
                for i in range(n_frames):
                    row = [str(i + 1), str(round(i / sr, 6))]
                    for j in range(len(marker_names)):
                        row += [str(float(i + j)), str(float(i + j + 1)),
                                str(float(i + j + 2))]
                    w.writerow(row)

        markers = ["head_front", "head_back"]
        # 2 s blocks at 120 Hz
        write_csv(src_dir / f"{name}_Trial001.csv",
                  int(2 * xyz_sr), markers, xyz_sr)
        write_csv(src_dir / f"{name}_Trial002.csv",
                  int(2 * xyz_sr), markers, xyz_sr)

        # ── configure_project + link_session ─────────────────────────#
        from neurobox.config.config import configure_project, link_session
        configure_project("TEST", data_root=tmp_path,
                          dotenv_path=tmp_path / ".env")

        # Need processed mocap dir for link_session
        moc_dir = (tmp_path / "processed" / "mocap" / "testA"
                   / "testA-jg" / "testA-jg-01" / name / "cof")
        moc_dir.mkdir(parents=True)

        link_session(name, "TEST", data_root=tmp_path, mazes=["cof"])

        # ── Build minimal NBSession-like object ───────────────────── #
        paths = NBSessionPaths(name, tmp_path, "TEST", maze="cof")

        # Use a simple namespace instead of full NBSession to avoid .env
        session        = types.SimpleNamespace()
        session.name   = name
        session.maze   = "cof"
        session.trial  = "all"
        session.spath  = paths.spath
        session.paths  = paths
        session.filebase = f"{name}.cof.all"
        session.par    = None
        session.samplerate = None

        return session, tmp_path

    def test_sync_populates_fields(self, fake_session):
        from neurobox.dtype.sync_pipelines import sync_nlx_vicon

        session, _ = fake_session
        sync_nlx_vicon(
            session,
            ttl_value = "0x0040",
            stop_ttl  = "0x0000",
            save_xyz  = False,
        )

        assert session.sync is not None
        assert session.xyz  is not None
        assert session.lfp  is not None
        assert session.spk  is not None
        assert session.stc  is not None

    def test_sync_window_subset_of_recording(self, fake_session):
        from neurobox.dtype.sync_pipelines import sync_nlx_vicon

        session, _ = fake_session
        sync_nlx_vicon(session, ttl_value="0x0040",
                       stop_ttl="0x0000", save_xyz=False)

        t0, t1 = session.sync.data[0]
        assert t0 >= 0.0
        assert t1 <= 60.0      # recording is 60 s
        assert t1 > t0

    def test_sync_xyz_shape(self, fake_session):
        from neurobox.dtype.sync_pipelines import sync_nlx_vicon

        session, _ = fake_session
        sync_nlx_vicon(session, ttl_value="0x0040",
                       stop_ttl="0x0000", save_xyz=False)

        arr = session.xyz.data
        assert arr.ndim == 3
        assert arr.shape[1] == 2    # 2 markers
        assert arr.shape[2] == 3    # x, y, z

    def test_sync_spk_loaded(self, fake_session):
        from neurobox.dtype.sync_pipelines import sync_nlx_vicon

        session, _ = fake_session
        sync_nlx_vicon(session, ttl_value="0x0040",
                       stop_ttl="0x0000", save_xyz=False)

        # All 10 spikes have cluster IDs 2 or 3 (none are MUA=1)
        assert len(session.spk) == 10
        assert session.spk.n_units == 2

    def test_sync_origin_matches_first_window(self, fake_session):
        from neurobox.dtype.sync_pipelines import sync_nlx_vicon

        session, _ = fake_session
        sync_nlx_vicon(session, ttl_value="0x0040",
                       stop_ttl="0x0000", save_xyz=False)

        assert session.xyz.origin == pytest.approx(session.sync.data[0, 0])

    # ── Round 23 — StreamSync / TrialWindow integration ─────────── #

    def test_sync_populates_stream_sync(self, fake_session):
        """sync_nlx_vicon should populate xyz.stream_sync, xyz.recording_windows,
        session.window, lfp.stream_sync alongside the legacy fields."""
        from neurobox.dtype.sync_pipelines import sync_nlx_vicon
        from neurobox.dtype.sync           import StreamSync, TrialWindow

        session, _ = fake_session
        sync_nlx_vicon(session, ttl_value="0x0040",
                       stop_ttl="0x0000", save_xyz=False)

        # session.window is a TrialWindow
        assert isinstance(session.window, TrialWindow)
        assert session.window.name == "all"

        # xyz.stream_sync is a StreamSync (single spanning segment)
        assert isinstance(session.xyz.stream_sync, StreamSync)
        # data is span-aligned; stream_sync covers the whole span
        np.testing.assert_allclose(
            session.xyz.stream_sync.master_first,
            session.window.t_start,
        )
        np.testing.assert_allclose(
            session.xyz.stream_sync.master_last,
            session.window.t_stop,
        )

        # xyz.recording_windows holds the per-block ground truth (2 blocks)
        assert session.xyz.recording_windows is not None
        assert session.xyz.recording_windows.shape == (2, 2)
        # Block 1: master [10, 12]; Block 2: master [30, 32]
        np.testing.assert_allclose(
            session.xyz.recording_windows[0], [10.0, 12.0], atol=0.05,
        )
        np.testing.assert_allclose(
            session.xyz.recording_windows[1], [30.0, 32.0], atol=0.05,
        )

        # lfp.stream_sync is continuous over the full recording
        assert isinstance(session.lfp.stream_sync, StreamSync)
        assert session.lfp.stream_sync.n_segments == 1

    def test_restrict_to_window_handles_session_aligned_data(
        self, fake_session
    ):
        """xyz.restrict_to_window with a multi-segment session-aligned
        array should consult recording_windows and zero-fill the gaps
        appropriately."""
        from neurobox.dtype.sync_pipelines import sync_nlx_vicon
        from neurobox.dtype.sync           import TrialWindow

        session, _ = fake_session
        sync_nlx_vicon(session, ttl_value="0x0040",
                       stop_ttl="0x0000", save_xyz=False)

        xyz_sr = session.xyz.samplerate
        # Trial spanning master [11, 31] — straddles the gap [12, 30]
        win = TrialWindow(periods=np.array([[11.0, 31.0]]), name="task1")
        sub = session.xyz.restrict_to_window(win)

        # Output: 20 s × xyz_sr samples, contiguous
        expected_n = int(round(20.0 * xyz_sr))
        assert sub.data.shape[0] == expected_n
        # New stream_sync reflects only the recorded portions
        # ([11, 12] and [30, 31])
        assert sub.stream_sync.n_segments == 2
        np.testing.assert_allclose(
            sub.stream_sync.segments[0], [11.0, 12.0], atol=0.05,
        )
        np.testing.assert_allclose(
            sub.stream_sync.segments[1], [30.0, 31.0], atol=0.05,
        )

        # The gap-filled middle should be all zeros
        valid = sub.stream_sync.valid_mask_in_window(11.0, 31.0)
        gap_region = ~valid
        assert (sub.data[gap_region] == 0).all() or \
               (np.abs(sub.data[gap_region]).sum() < 1e-3)

    def test_trial_load_xyz_round_trip_through_disk(self, fake_session):
        """The user-facing flow: sync → save → NBTrial(window=…).load('xyz').

        Verifies that recording_windows survives the save_npy/load
        round-trip so the multi-segment-aware restrict_to_window
        works from disk-loaded data, not just freshly-built data.
        """
        from neurobox.dtype                import NBTrial, TrialWindow
        from neurobox.dtype.sync_pipelines import sync_nlx_vicon

        session, mw = fake_session
        sync_nlx_vicon(session, ttl_value="0x0040",
                       stop_ttl="0x0000", save_xyz=True)

        # Build a NBTrial that points at the same project root
        # (the fake_session fixture writes a project dir under "TEST")
        win = TrialWindow(periods=np.array([[11.0, 31.0]]), name="task1")
        # Recover data_root from the session's paths
        data_root = session.paths.data_root
        trial = NBTrial(
            session_name = session.name,
            maze         = "cof",
            trial_name   = "task1",
            project_id   = session.paths.project_id,
            data_root    = data_root,
            window       = win,
        )

        # The actual question — does this work?
        xyz = trial.load("xyz")
        # Output is 20 s × 120 Hz = 2400 samples, contiguous
        assert xyz.data.shape[0] == int(round(20.0 * 120.0))
        # stream_sync reflects only the parts of the trial window
        # where Vicon was actually recording
        assert xyz.stream_sync.n_segments == 2
        # First recorded sub-segment is around [11, 12]
        np.testing.assert_allclose(
            xyz.stream_sync.segments[0], [11.0, 12.0], atol=0.05,
        )
        np.testing.assert_allclose(
            xyz.stream_sync.segments[1], [30.0, 31.0], atol=0.05,
        )


# ── _has_paths / _find_file helpers ─────────────────────────────────────── #

class TestHelpers:
    def test_has_paths_true(self):
        import types
        from neurobox.dtype.paths import NBSessionPaths
        s = types.SimpleNamespace(
            paths=NBSessionPaths("testA-jg-01-20240101", Path("/data"), "B01")
        )
        assert _has_paths(s)

    def test_has_paths_false(self):
        import types
        s = types.SimpleNamespace(paths=None)
        assert not _has_paths(s)

    def test_find_file_spath(self, tmp_path):
        import types
        f = tmp_path / "session.yaml"
        f.write_text("x: 1")
        s = types.SimpleNamespace(spath=tmp_path, paths=None, name="session")
        result = _find_file(s, "session.yaml")
        assert result == f

    def test_find_file_returns_none_if_missing(self, tmp_path):
        import types
        s = types.SimpleNamespace(spath=tmp_path, paths=None, name="session")
        assert _find_file(s, "missing.yaml") is None


# ── sync_ephys_vicon (alias for sync_nlx_vicon) ─────────────────────────── #

class TestSyncEphysVicon:
    """Tests for the generic-ephys entry point.

    ``sync_ephys_vicon`` is currently a thin alias for
    :func:`sync_nlx_vicon` — same code path, different name.  These
    tests verify the alias is wired up correctly and reaches the
    same end state on the same fake-session fixture used by
    :class:`TestSyncNlxVicon`.
    """

    @pytest.fixture
    def fake_session(self, tmp_path):
        """Reuse the TestSyncNlxVicon fixture so we exercise the
        same synthetic data."""
        inst = TestSyncNlxVicon()
        gen = inst.fake_session.__wrapped__(inst, tmp_path)
        if hasattr(gen, "__next__"):
            return next(gen)
        return gen

    def test_runs_without_error(self, fake_session):
        from neurobox.dtype.sync_pipelines import sync_ephys_vicon
        session, _mock_writes = fake_session
        sync_ephys_vicon(
            session, ttl_value="0x0040", stop_ttl="0x0000",
            xyz_samplerate=120.0, save_xyz=False,
        )
        # Same post-conditions as TestSyncNlxVicon.test_populates_session
        assert session.window is not None
        assert not session.window.is_empty
        assert session.xyz is not None
        assert session.xyz.data.size > 0

    def test_matches_sync_nlx_vicon(self, tmp_path):
        """The wrapper should produce byte-identical xyz arrays to
        the underlying ``sync_nlx_vicon`` on the same input."""
        from neurobox.dtype.sync_pipelines import (
            sync_nlx_vicon, sync_ephys_vicon,
        )

        # Build two independent fake-session fixtures (the fixture
        # mutates session state, so we need two separate copies).
        inst1 = TestSyncNlxVicon()
        gen1 = inst1.fake_session.__wrapped__(inst1, tmp_path / "a")
        ses_a, _ = next(gen1) if hasattr(gen1, "__next__") else gen1

        inst2 = TestSyncNlxVicon()
        gen2 = inst2.fake_session.__wrapped__(inst2, tmp_path / "b")
        ses_b, _ = next(gen2) if hasattr(gen2, "__next__") else gen2

        sync_nlx_vicon(ses_a, ttl_value="0x0040", stop_ttl="0x0000",
                        xyz_samplerate=120.0, save_xyz=False)
        sync_ephys_vicon(ses_b, ttl_value="0x0040", stop_ttl="0x0000",
                          xyz_samplerate=120.0, save_xyz=False)

        np.testing.assert_array_equal(ses_a.xyz.data, ses_b.xyz.data)
        assert ses_a.window.t_start == ses_b.window.t_start
        assert ses_a.window.t_stop  == ses_b.window.t_stop


class TestDispatchEphysAlias:
    """Verify the dispatcher routes ``['ephys', 'vicon']`` to the
    sync_ephys_vicon entry, and ``['nlx', 'vicon']`` still goes
    direct to sync_nlx_vicon."""

    def test_dispatch_routes_ephys(self, monkeypatch):
        from neurobox.dtype import sync_pipelines
        called = {}
        def _fake_ephys(session, **kw):
            called["which"] = "ephys"
            called["kw"]    = kw
        def _fake_nlx(session, **kw):
            called["which"] = "nlx"
            called["kw"]    = kw
        monkeypatch.setattr(sync_pipelines, "sync_ephys_vicon", _fake_ephys)
        monkeypatch.setattr(sync_pipelines, "sync_nlx_vicon",   _fake_nlx)
        # Re-build the pipeline map under monkeypatch, to pick up the
        # patched callables.  (The list captured them by reference at
        # module import time.)
        monkeypatch.setattr(sync_pipelines, "_PIPELINE_MAP", [
            (("nlx", "neuralynx"), ("vicon", "optitrack", "motive"), _fake_nlx),
            (("ephys",),           ("vicon", "optitrack", "motive"), _fake_ephys),
        ])

        sync_pipelines.dispatch(object(), ["ephys", "vicon"])
        assert called["which"] == "ephys"

        called.clear()
        sync_pipelines.dispatch(object(), ["nlx", "vicon"])
        assert called["which"] == "nlx"

    def test_dispatch_unknown_lists_ephys_in_error(self):
        from neurobox.dtype.sync_pipelines import dispatch
        with pytest.raises(ValueError, match="ephys"):
            dispatch(object(), ["unknown", "vicon"])


class TestSyncEphysViconReExport:
    """The wrapper should be reachable through the canonical
    ``neurobox.utils.sync`` re-export, mirroring ``sync_nlx_vicon``."""

    def test_importable_from_utils(self):
        from neurobox.utils.sync import sync_ephys_vicon, sync_nlx_vicon
        # Both should be plain callables
        assert callable(sync_ephys_vicon)
        assert callable(sync_nlx_vicon)

    def test_is_in_all(self):
        import neurobox.utils.sync as us
        assert "sync_ephys_vicon" in us.__all__
