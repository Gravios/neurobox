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
# NamingConfig — neurobox naming + mixed-naming projects                     #
# ─────────────────────────────────────────────────────────────────────── #

class TestNamingConfig:
    """The browser's data layer is parameterised by a ``NamingConfig``.

    Two presets are shipped: ``neurobox_naming`` (canonical 4-part
    session names + .pkl files) and ``labbox_mta_naming`` (legacy
    2-letter subject + .mat files).  Both should be discoverable
    out of the box, and a project mixing both should also work.
    """

    def _make_neurobox_session(
        self, root: Path, sname: str,
        mazes=("cof", "sof"),
        trials=("task1", "task2"),
    ):
        d = root / sname
        d.mkdir()
        for maze in mazes:
            (d / f"{sname}.{maze}.ses.pkl").touch()
            for trial in trials:
                (d / f"{sname}.{maze}.{trial}.trl.pkl").touch()

    def test_neurobox_naming_discovered(self):
        from neurobox.gui.mta_browser.data_layer import scan_project
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            self._make_neurobox_session(tmp, "sirotaA-jg-05-20120316")
            self._make_neurobox_session(tmp, "sirotaA-jg-05-20120317")
            idx = scan_project(tmp)
            # subject is the 3rd group of the 4-part name (the
            # subjectId, e.g. "05")
            assert idx.subjects() == ["05"]
            assert sorted(idx.dates_for("05")) == ["20120316", "20120317"]
            assert idx.mazes_for("05", "20120316") == ["cof", "sof"]
            assert idx.trials_for("05", "20120316", "cof") == [
                "all", "task1", "task2",
            ]

    def test_neurobox_session_entry_records_naming(self):
        from neurobox.gui.mta_browser.data_layer import scan_project
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            self._make_neurobox_session(tmp, "sirotaA-jg-05-20120316")
            idx = scan_project(tmp)
            sess = idx.session_for("05", "20120316")
            assert sess is not None
            assert sess.naming == "neurobox"

    def test_labbox_mta_session_entry_records_naming(self):
        from neurobox.gui.mta_browser.data_layer import scan_project
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            d = tmp / "jg05-20120316"
            d.mkdir()
            (d / "jg05-20120316.cof.ses.mat").touch()
            idx = scan_project(tmp)
            sess = idx.session_for("jg05", "20120316")
            assert sess is not None
            assert sess.naming == "labbox-mta"

    def test_mixed_namings_in_one_project(self):
        """A project tree containing both the legacy MATLAB layout
        and the new neurobox layout should be discovered correctly,
        with each session entry tagged with its detected naming."""
        from neurobox.gui.mta_browser.data_layer import scan_project
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            # Legacy session
            old = tmp / "jg05-20120316"
            old.mkdir()
            (old / "jg05-20120316.cof.ses.mat").touch()
            (old / "jg05-20120316.cof.task1.trl.mat").touch()
            # New session
            self._make_neurobox_session(tmp, "sirotaA-jg-05-20120317")

            idx = scan_project(tmp)
            assert sorted(idx.subjects()) == ["05", "jg05"]
            old_sess = idx.session_for("jg05", "20120316")
            new_sess = idx.session_for("05", "20120317")
            assert old_sess.naming == "labbox-mta"
            assert new_sess.naming == "neurobox"
            # Both contribute their own mazes/trials independently
            assert "cof" in old_sess.mazes
            assert old_sess.trials["cof"] == ["all", "task1"]
            assert sorted(new_sess.mazes) == ["cof", "sof"]

    def test_custom_naming_subset(self):
        """Users can pass an explicit ``namings=`` subset to narrow
        scan to a single convention."""
        from neurobox.gui.mta_browser.data_layer import (
            scan_project, neurobox_naming,
        )
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            # Mix both layouts, but ask for neurobox only
            self._make_neurobox_session(tmp, "sirotaA-jg-05-20120316")
            old = tmp / "jg05-20120316"
            old.mkdir()
            (old / "jg05-20120316.cof.ses.mat").touch()
            idx = scan_project(tmp, namings=[neurobox_naming])
            assert idx.subjects() == ["05"]      # legacy filtered out

    def test_namings_precedence_first_match_wins(self):
        """When two configs both could match, the first one in the
        list wins for that directory.  This matters for hypothetical
        configs whose patterns overlap — guarantees deterministic
        scan behaviour."""
        import re
        from neurobox.gui.mta_browser.data_layer import (
            scan_project, NamingConfig, neurobox_naming,
        )
        # A custom config that ALSO matches the neurobox 4-part name
        # but uses a different marker glob (irrelevant for this test)
        custom = NamingConfig(
            name                = "custom",
            session_pattern     = re.compile(
                r"^(?P<subject>.+)-(?P<date>\d{8})$"
            ),
            session_marker_glob = "*.custom.pkl",
            trial_marker_glob   = "*.custom.trl.pkl",
            maze_from_filename  = lambda f: None,
            trial_from_filename = lambda f: None,
        )
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            self._make_neurobox_session(tmp, "sirotaA-jg-05-20120316")
            # Put custom first → it wins, but its glob doesn't match
            # any files, so the resulting entry has no mazes
            idx_custom = scan_project(tmp, namings=[custom, neurobox_naming])
            for sessions in idx_custom.by_subject.values():
                for s in sessions:
                    assert s.naming == "custom"
                    assert s.mazes == []
            # Put neurobox first → mazes are discovered
            idx_neurobox = scan_project(
                tmp, namings=[neurobox_naming, custom],
            )
            sess = idx_neurobox.session_for("05", "20120316")
            assert sess.naming == "neurobox"
            assert sorted(sess.mazes) == ["cof", "sof"]


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


# ─────────────────────────────────────────────────────────────────────── #
# Preferences (round-23 follow-up)                                            #
# ─────────────────────────────────────────────────────────────────────── #

class TestPreferences:
    """Tests for the naming-config preference plumbing.

    All QSettings reads/writes go through an isolated org/app pair
    inside a tmp dir to avoid touching the user's real config.
    """

    def _isolate_settings(self, monkeypatch, tmp_path):
        """Pin QSettings to an in-tree dir so the test doesn't read or
        write the user's real preferences.  Also wipes any leftover
        settings file from previous tests in the same process."""
        from PySide6.QtCore import QSettings, QStandardPaths
        QStandardPaths.setTestModeEnabled(True)
        QSettings.setPath(
            QSettings.IniFormat, QSettings.UserScope, str(tmp_path),
        )
        QSettings.setDefaultFormat(QSettings.IniFormat)
        # Active wipe: clear any keys this test class might persist
        # so each test sees a truly empty settings store.
        s = QSettings("neurobox", "mta_browser")
        s.clear()
        s.sync()

    def test_load_preferences_returns_defaults_when_unset(
        self, qapp, tmp_path, monkeypatch,
    ):
        from neurobox.gui.mta_browser.preferences import (
            load_preferences,
        )
        from neurobox.gui.mta_browser.data_layer import (
            neurobox_naming, labbox_mta_naming,
        )
        self._isolate_settings(monkeypatch, tmp_path)
        prefs = load_preferences()
        names = [c.name for c in prefs.enabled_namings]
        assert names == [neurobox_naming.name, labbox_mta_naming.name]

    def test_save_then_load_round_trip(
        self, qapp, tmp_path, monkeypatch,
    ):
        from neurobox.gui.mta_browser.preferences import (
            Preferences, load_preferences, save_preferences,
        )
        from neurobox.gui.mta_browser.data_layer import (
            labbox_mta_naming, neurobox_naming,
        )
        self._isolate_settings(monkeypatch, tmp_path)
        # Save with reverse order, only labbox-mta enabled
        save_preferences(Preferences(enabled_namings=[labbox_mta_naming]))
        loaded = load_preferences()
        assert [c.name for c in loaded.enabled_namings] == [
            labbox_mta_naming.name,
        ]
        # Now save the reversed-pair ordering — neurobox second
        save_preferences(Preferences(
            enabled_namings=[labbox_mta_naming, neurobox_naming],
        ))
        loaded = load_preferences()
        assert [c.name for c in loaded.enabled_namings] == [
            labbox_mta_naming.name, neurobox_naming.name,
        ]

    def test_dialog_returns_edited_preferences(
        self, qapp, tmp_path, monkeypatch,
    ):
        from PySide6.QtCore import Qt
        from neurobox.gui.mta_browser.preferences import (
            Preferences, PreferencesDialog,
        )
        from neurobox.gui.mta_browser.data_layer import (
            labbox_mta_naming, neurobox_naming,
        )
        self._isolate_settings(monkeypatch, tmp_path)
        # Start with both enabled, neurobox first
        dlg = PreferencesDialog(prefs=Preferences(
            enabled_namings=[neurobox_naming, labbox_mta_naming],
        ))
        # Uncheck the first item (neurobox)
        item0 = dlg._list.item(0)
        assert item0.text() == neurobox_naming.name
        item0.setCheckState(Qt.Unchecked)
        edited = dlg.preferences()
        # Only labbox-mta is enabled now
        assert [c.name for c in edited.enabled_namings] == [
            labbox_mta_naming.name,
        ]
        dlg.deleteLater()

    def test_dialog_unchecking_all_falls_back_to_defaults(
        self, qapp, tmp_path, monkeypatch,
    ):
        from PySide6.QtCore import Qt
        from neurobox.gui.mta_browser.preferences import (
            Preferences, PreferencesDialog,
        )
        from neurobox.gui.mta_browser.data_layer import (
            default_naming_configs, neurobox_naming,
        )
        self._isolate_settings(monkeypatch, tmp_path)
        dlg = PreferencesDialog(prefs=Preferences(
            enabled_namings=[neurobox_naming],
        ))
        # Uncheck every item
        for i in range(dlg._list.count()):
            dlg._list.item(i).setCheckState(Qt.Unchecked)
        edited = dlg.preferences()
        # Empty selection ⇒ silently restore defaults rather than
        # persisting a config that would scan nothing
        default_names = [c.name for c in default_naming_configs()]
        assert [c.name for c in edited.enabled_namings] == default_names
        dlg.deleteLater()

    def test_dialog_move_up_reorders(self, qapp, tmp_path, monkeypatch):
        from neurobox.gui.mta_browser.preferences import (
            Preferences, PreferencesDialog,
        )
        from neurobox.gui.mta_browser.data_layer import (
            labbox_mta_naming, neurobox_naming,
        )
        self._isolate_settings(monkeypatch, tmp_path)
        dlg = PreferencesDialog(prefs=Preferences(
            enabled_namings=[neurobox_naming, labbox_mta_naming],
        ))
        # Select row 1 and move it up
        dlg._list.setCurrentRow(1)
        dlg._move(-1)
        edited = dlg.preferences()
        assert [c.name for c in edited.enabled_namings] == [
            labbox_mta_naming.name, neurobox_naming.name,
        ]
        dlg.deleteLater()

    def test_browser_window_preferences_roundtrip(
        self, qapp, tmp_path, monkeypatch,
    ):
        """Changing the prefs and applying them re-scans the active
        project root with the new naming list."""
        from neurobox.gui.mta_browser           import MTABrowserWindow
        from neurobox.gui.mta_browser.preferences import (
            Preferences,
        )
        from neurobox.gui.mta_browser.data_layer import (
            labbox_mta_naming, neurobox_naming,
        )
        self._isolate_settings(monkeypatch, tmp_path)

        # Build a project containing both kinds of session dirs
        proj = tmp_path / "proj"
        proj.mkdir()
        # Legacy session
        legacy = proj / "jg05-20120316"
        legacy.mkdir()
        (legacy / "jg05-20120316.cof.ses.mat").touch()
        # Neurobox-style session
        new = proj / "sirotaA-jg-05-20120317"
        new.mkdir()
        (new / "sirotaA-jg-05-20120317.cof.ses.pkl").touch()

        win = MTABrowserWindow()
        win._dm_tab.set_root(proj)
        qapp.processEvents()

        # With default prefs both subjects appear (subject = "jg05"
        # for legacy, "05" for neurobox).
        assert sorted(["jg05" if win._dm_tab.subject_list.item(i).text()
                       == "jg05" else win._dm_tab.subject_list.item(i).text()
                       for i in range(win._dm_tab.subject_list.count())]) == [
            "05", "jg05",
        ]

        # Switch to labbox-mta only — neurobox session disappears
        win._apply_preferences(Preferences(
            enabled_namings=[labbox_mta_naming],
        ))
        qapp.processEvents()
        assert [
            win._dm_tab.subject_list.item(i).text()
            for i in range(win._dm_tab.subject_list.count())
        ] == ["jg05"]

        # Switch to neurobox only — legacy session disappears
        win._apply_preferences(Preferences(
            enabled_namings=[neurobox_naming],
        ))
        qapp.processEvents()
        assert [
            win._dm_tab.subject_list.item(i).text()
            for i in range(win._dm_tab.subject_list.count())
        ] == ["05"]

        win.close()
