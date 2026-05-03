"""Round-20 tests — MTA Browser Qt app.

Skipped if PySide6 isn't installed.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# All these tests need an offscreen platform
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")

pytest.importorskip("PySide6")

from PySide6.QtCore    import Qt, QEvent
from PySide6.QtGui     import QKeyEvent
from PySide6.QtWidgets import QApplication


# Single shared QApplication for the test session
@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication(sys.argv)
    yield app


# ─────────────────────────────────────────────────────────────────────── #
# data_layer                                                                 #
# ─────────────────────────────────────────────────────────────────────── #

class TestDataLayer:
    def _make_project(self, tmp: Path,
                       sessions=("jg05-20120316", "jg05-20120317",
                                 "Ed03-20140624a"),
                       mazes=("cof", "sof"),
                       trials=("task1", "task2")):
        for sname in sessions:
            d = tmp / sname
            d.mkdir()
            for maze in mazes:
                (d / f"{sname}.{maze}.ses.mat").touch()
                for trial in trials:
                    (d / f"{sname}.{maze}.{trial}.trl.mat").touch()
        # An irrelevant directory
        (tmp / "random_dir").mkdir()

    def test_scan_finds_sessions(self):
        from neurobox.gui.mta_browser.data_layer import scan_project
        with tempfile.TemporaryDirectory() as tmp:
            self._make_project(Path(tmp))
            idx = scan_project(tmp)
            assert idx.subjects() == ["Ed03", "jg05"]
            assert idx.dates_for("jg05") == ["20120316", "20120317"]

    def test_mazes_and_trials(self):
        from neurobox.gui.mta_browser.data_layer import scan_project
        with tempfile.TemporaryDirectory() as tmp:
            self._make_project(Path(tmp))
            idx = scan_project(tmp)
            assert idx.mazes_for("jg05", "20120316") == ["cof", "sof"]
            trials = idx.trials_for("jg05", "20120316", "cof")
            # Sorted; 'all' always present
            assert "all" in trials
            assert "task1" in trials
            assert "task2" in trials

    def test_nonexistent_root(self):
        from neurobox.gui.mta_browser.data_layer import scan_project
        idx = scan_project("/this/path/does/not/exist")
        assert idx.subjects() == []

    def test_ignores_non_session_dirs(self):
        from neurobox.gui.mta_browser.data_layer import scan_project
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "junk").mkdir()
            (tmp / "another").mkdir()
            # Only one valid session
            d = tmp / "jg05-20120316"
            d.mkdir()
            (d / "jg05-20120316.cof.ses.mat").touch()
            idx = scan_project(tmp)
            assert idx.subjects() == ["jg05"]


# ─────────────────────────────────────────────────────────────────────── #
# Helper to build a synthetic session                                        #
# ─────────────────────────────────────────────────────────────────────── #

def _make_session(T: int = 600, fs: float = 30.0):
    from neurobox.dtype import (
        NBSession, NBDxyz, NBModel, NBEpoch, NBStateCollection,
    )
    ses = NBSession()
    ses.name      = "synth-20240101"
    ses.samplerate = fs
    ses.maze      = "cof"
    ses.trial     = "all"
    ses.filebase  = "synth-20240101.cof.all"

    t = np.arange(T) / fs
    markers = ["spine_lower", "pelvis_root", "spine_middle", "spine_upper",
               "head_back", "head_front"]
    data = np.zeros((T, len(markers), 3))
    heading = t * 0.3
    for i, _m in enumerate(markers):
        data[:, i, 0] = 100 + 50 * np.cos(heading)
        data[:, i, 1] = 100 + 50 * np.sin(heading)
        data[:, i, 2] = 50 + i * 5
    model = NBModel(markers=markers, connections=[
        ["spine_lower", "pelvis_root"],
        ["pelvis_root", "spine_middle"],
        ["spine_middle", "spine_upper"],
        ["spine_upper", "head_back"],
        ["head_back",   "head_front"],
    ])
    ses.xyz = NBDxyz(data, model=model, samplerate=fs, name="synth")

    stc = NBStateCollection(mode="manual")
    stc.add_state(NBEpoch(
        data=np.array([[1.0, 5.0]]), samplerate=fs,
        mode="periods", label="walk", key="w",
    ))
    ses.stc = stc
    return ses


# ─────────────────────────────────────────────────────────────────────── #
# PlaybackModel                                                              #
# ─────────────────────────────────────────────────────────────────────── #

class TestPlaybackModel:
    def test_from_session_populates_states(self):
        from neurobox.gui.mta_browser.model import PlaybackModel
        ses = _make_session()
        m = PlaybackModel.from_session(ses)
        assert m.n_samples == 600
        assert m.samplerate == 30.0
        assert "walk" in m.states_data
        # Periods 1.0-5.0 s @ 30 Hz = samples 30..150 (120 samples)
        assert int((m.states_data["walk"] > 0).sum()) == 120

    def test_step_paints_state(self):
        from neurobox.gui.mta_browser.model import PlaybackModel
        m = PlaybackModel.from_session(_make_session())
        m.add_state("rear", "r")
        m.set_current_label("rear")
        m.set_idx(200)
        m.step(+10)
        assert m.idx == 210
        # Painted samples 200..210 (11 samples inclusive of both ends)
        assert int((m.states_data["rear"] > 0).sum()) == 11

    def test_step_in_erase_mode(self):
        from neurobox.gui.mta_browser.model import PlaybackModel
        m = PlaybackModel.from_session(_make_session())
        m.set_current_label("walk")
        m.set_idx(50)
        m.set_erase_mode(True)
        before = int((m.states_data["walk"] > 0).sum())
        m.step(+20)
        after = int((m.states_data["walk"] > 0).sum())
        assert after < before    # samples got erased

    def test_listener_fires(self):
        from neurobox.gui.mta_browser.model import PlaybackModel
        m = PlaybackModel.from_session(_make_session())
        events = []
        m.subscribe(lambda e: events.append(e))
        m.set_idx(100)
        m.add_state("rear", "r")
        assert "idx" in events
        assert "states" in events

    def test_add_duplicate_label_raises(self):
        from neurobox.gui.mta_browser.model import PlaybackModel
        m = PlaybackModel.from_session(_make_session())
        with pytest.raises(ValueError, match="already exists"):
            m.add_state("walk", "x")

    def test_add_duplicate_key_raises(self):
        from neurobox.gui.mta_browser.model import PlaybackModel
        m = PlaybackModel.from_session(_make_session())
        with pytest.raises(ValueError, match="already used"):
            m.add_state("foo", "w")

    def test_to_state_collection(self):
        from neurobox.gui.mta_browser.model import PlaybackModel
        m = PlaybackModel.from_session(_make_session())
        stc = m.to_state_collection()
        assert "walk" in stc.list_states()
        # Periods round-trip
        ep = stc.get_state("walk")
        np.testing.assert_allclose(ep.data, [[1.0, 5.0]], atol=0.05)

    def test_rename_state(self):
        from neurobox.gui.mta_browser.model import PlaybackModel
        m = PlaybackModel.from_session(_make_session())
        m.rename_state("walk", "locomotion")
        assert "walk" not in m.states_data
        assert "locomotion" in m.states_data
        assert m.state_keys["locomotion"] == "w"

    def test_remove_state(self):
        from neurobox.gui.mta_browser.model import PlaybackModel
        m = PlaybackModel.from_session(_make_session())
        m.remove_state("walk")
        assert "walk" not in m.states_data


# ─────────────────────────────────────────────────────────────────────── #
# Widgets — at least one round-trip through each                             #
# ─────────────────────────────────────────────────────────────────────── #

class TestWidgets:
    def test_skeleton_viewer_renders(self, qapp):
        from neurobox.gui.mta_browser.model   import PlaybackModel
        from neurobox.gui.mta_browser.widgets import SkeletonViewer3D
        m = PlaybackModel.from_session(_make_session())
        v = SkeletonViewer3D(m)
        v.show()
        qapp.processEvents()
        m.set_idx(100)
        qapp.processEvents()
        assert v.isVisible()
        v.deleteLater()

    def test_state_track_renders(self, qapp):
        from neurobox.gui.mta_browser.model   import PlaybackModel
        from neurobox.gui.mta_browser.widgets import StateTrackView
        m = PlaybackModel.from_session(_make_session())
        v = StateTrackView(m)
        v.show()
        qapp.processEvents()
        m.set_idx(50)
        qapp.processEvents()
        v.deleteLater()

    def test_feature_panel_renders(self, qapp):
        from neurobox.gui.mta_browser.model   import PlaybackModel
        from neurobox.gui.mta_browser.widgets import FeaturePanel
        m = PlaybackModel.from_session(_make_session())
        # Some synthetic features
        T = m.n_samples
        feats = {
            "head_z":     np.sin(np.arange(T) * 0.05),
            "spine_yaw":  np.cos(np.arange(T) * 0.1),
        }
        panel = FeaturePanel(m, features=feats, window_seconds=2.0)
        panel.show()
        qapp.processEvents()
        m.set_idx(100)
        qapp.processEvents()
        # Update features
        panel.set_features({"new_feature": np.ones(T)})
        qapp.processEvents()
        panel.deleteLater()


# ─────────────────────────────────────────────────────────────────────── #
# MTABrowserWindow integration                                                #
# ─────────────────────────────────────────────────────────────────────── #

class TestMtaBrowserWindow:
    def test_launch_with_session(self, qapp):
        from neurobox.gui.mta_browser import MTABrowserWindow
        ses = _make_session()
        win = MTABrowserWindow.launch(session=ses, run=False)
        assert win.isVisible()
        assert win._tabs.count() == 3        # round 21 added "LFP States"
        # Auto-switched to motion-labelling tab
        assert win._tabs.currentIndex() == 1
        assert win._ml_tab.model is not None
        assert win._ml_tab.model.n_samples == 600
        win.close()

    def test_state_editor_lists_states(self, qapp):
        from neurobox.gui.mta_browser import MTABrowserWindow
        ses = _make_session()
        win = MTABrowserWindow.launch(session=ses, run=False)
        editor = win._ml_tab._editor
        assert editor.table.rowCount() == 1
        assert editor.table.item(0, 0).text() == "walk"
        win.close()

    def test_add_state_via_model_updates_editor(self, qapp):
        from neurobox.gui.mta_browser import MTABrowserWindow
        ses = _make_session()
        win = MTABrowserWindow.launch(session=ses, run=False)
        model = win._ml_tab.model
        model.add_state("rear", "r")
        qapp.processEvents()
        editor = win._ml_tab._editor
        assert editor.table.rowCount() == 2
        win.close()

    def test_keyboard_state_toggle(self, qapp):
        from neurobox.gui.mta_browser import MTABrowserWindow
        ses = _make_session()
        win = MTABrowserWindow.launch(session=ses, run=False)
        ml = win._ml_tab
        ev = QKeyEvent(QEvent.KeyPress, Qt.Key_W, Qt.NoModifier, "w")
        ml.keyPressEvent(ev)
        assert ml.model.current_label == "walk"
        win.close()

    def test_keyboard_play_stops_at_end(self, qapp):
        from neurobox.gui.mta_browser import MTABrowserWindow
        ses = _make_session()
        win = MTABrowserWindow.launch(session=ses, run=False)
        ml = win._ml_tab
        ml.model.set_idx(ml.model.n_samples - 5)
        # Set direction forward; tick until the model decides to stop
        ml._set_direction(+1)
        for _ in range(50):
            ml._on_tick()
            if ml._direction == 0:
                break
        assert ml.model.idx == ml.model.n_samples - 1
        assert ml._direction == 0     # auto-stopped
        win.close()

    def test_ctrl_arrow_jumps_50(self, qapp):
        from neurobox.gui.mta_browser import MTABrowserWindow
        ses = _make_session()
        win = MTABrowserWindow.launch(session=ses, run=False)
        ml = win._ml_tab
        ml.model.set_idx(200)
        ev = QKeyEvent(QEvent.KeyPress, Qt.Key_Right, Qt.ControlModifier)
        ml.keyPressEvent(ev)
        assert ml.model.idx == 250
        ev2 = QKeyEvent(QEvent.KeyPress, Qt.Key_Left, Qt.ControlModifier)
        ml.keyPressEvent(ev2)
        assert ml.model.idx == 200
        win.close()

    def test_save_stc_round_trip(self, qapp):
        from neurobox.gui.mta_browser import MTABrowserWindow
        ses = _make_session()
        win = MTABrowserWindow.launch(session=ses, run=False)
        m = win._ml_tab.model
        m.add_state("rear", "r")
        m.set_current_label("rear")
        m.set_idx(100); m.step(20)
        m.update_session_stc()
        assert "walk" in m.session.stc.list_states()
        assert "rear" in m.session.stc.list_states()
        win.close()


# ─────────────────────────────────────────────────────────────────────── #
# Data Management Tab                                                        #
# ─────────────────────────────────────────────────────────────────────── #

class TestDataManagementTab:
    def test_set_root_populates_lists(self, qapp):
        from neurobox.gui.mta_browser.main_window import _DataManagementTab
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            d = tmp / "jg05-20120316"
            d.mkdir()
            (d / "jg05-20120316.cof.ses.mat").touch()
            (d / "jg05-20120316.cof.task1.trl.mat").touch()
            tab = _DataManagementTab()
            tab.set_root(tmp)
            qapp.processEvents()
            assert tab.subject_list.count() == 1
            assert tab.subject_list.item(0).text() == "jg05"
            # Date should auto-select first
            assert tab.date_list.count() == 1
            # Maze should appear
            assert tab.maze_list.count() == 1
            assert tab.maze_list.item(0).text() == "cof"
            # Trials include 'all' + 'task1'
            trials = [tab.trial_list.item(i).text()
                       for i in range(tab.trial_list.count())]
            assert "all" in trials and "task1" in trials
            tab.deleteLater()

    def test_set_root_empty_dir(self, qapp):
        from neurobox.gui.mta_browser.main_window import _DataManagementTab
        with tempfile.TemporaryDirectory() as tmp:
            tab = _DataManagementTab()
            tab.set_root(Path(tmp))
            qapp.processEvents()
            assert tab.subject_list.count() == 0
            assert not tab.load_btn.isEnabled()
            tab.deleteLater()
