"""Round-21 tests — LFP States tab integration with MTABrowser.

Skipped if PySide6 isn't installed.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")

pytest.importorskip("PySide6")

from PySide6.QtCore    import Qt, QEvent
from PySide6.QtGui     import QKeyEvent
from PySide6.QtWidgets import QApplication


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication(sys.argv)
    yield app


# ─────────────────────────────────────────────────────────────────────── #
# Helpers                                                                    #
# ─────────────────────────────────────────────────────────────────────── #

def _make_session(theta_period=(10.0, 20.0)):
    """Synthetic session with xyz @ 30 Hz / 50 s and one theta state.

    Pass ``theta_period=None`` for a session with no states.
    """
    from neurobox.dtype import (
        NBSession, NBDxyz, NBModel, NBEpoch, NBStateCollection,
    )
    ses = NBSession()
    ses.name = "synth"; ses.maze = "cof"; ses.trial = "all"
    ses.filebase = "synth"; ses.samplerate = 30.0
    markers = ["spine_lower", "pelvis_root", "spine_middle",
               "spine_upper", "head_back", "head_front"]
    ses.xyz = NBDxyz(
        np.zeros((1500, len(markers), 3)),
        model=NBModel(markers=markers),
        samplerate=30.0, name="t",
    )
    stc = NBStateCollection(mode="manual")
    if theta_period is not None:
        stc.add_state(NBEpoch(
            data=np.array([list(theta_period)]), samplerate=30.0,
            mode="periods", label="theta", key="t",
        ))
    ses.stc = stc
    return ses


def _make_lfp_and_spec():
    """Synthetic 50 s LFP @ 1250 Hz with 8 Hz theta + noise, plus spec."""
    from neurobox.analysis.lfp.spectral import (
        SpectralParams, multitaper_spectrogram,
    )
    sr_lfp = 1250.0
    T = int(50 * sr_lfp)
    rng = np.random.default_rng(0)
    t = np.arange(T) / sr_lfp
    lfp = (np.sin(2 * np.pi * 8 * t) + 0.3 * rng.standard_normal(T))[:, None]
    spec = multitaper_spectrogram(
        lfp,
        SpectralParams(samplerate=sr_lfp, n_fft=1024, win_len=1250,
                        n_overlap=1000, freq_range=(1.0, 50.0)),
    )
    return lfp, sr_lfp, spec


# ─────────────────────────────────────────────────────────────────────── #
# PlaybackModel.from_data                                                    #
# ─────────────────────────────────────────────────────────────────────── #

class TestPlaybackModelFromData:
    def test_from_data_uses_given_rate(self):
        from neurobox.gui.mta_browser.model import PlaybackModel
        ses = _make_session()
        m = PlaybackModel.from_data(ses, n_samples=250, samplerate=5.0)
        assert m.n_samples == 250
        assert m.samplerate == 5.0
        # Same session.stc periods produce different mask lengths
        # at different rates: 1.0-5.0s @ 5 Hz = 5..25 (20 samples)
        assert int((m.states_data["theta"] > 0).sum()) == 50  # 10..20s

    def test_from_data_round_trips_through_stc(self):
        from neurobox.gui.mta_browser.model import PlaybackModel
        ses = _make_session(theta_period=None)   # empty
        m = PlaybackModel.from_data(ses, n_samples=250, samplerate=5.0)
        m.add_state("theta", "t")
        m.set_current_label("theta")
        m.set_idx(50)            # = 10 s
        m.step(15)                 # paint to idx=65 = 13 s
        m.update_session_stc()
        ep = ses.stc.get_state("theta")
        # Period should round-trip in seconds
        assert ep.data.shape[0] == 1
        np.testing.assert_allclose(ep.data[0],
                                      [10.0, 13.0], atol=0.3)


# ─────────────────────────────────────────────────────────────────────── #
# LFP widgets                                                                #
# ─────────────────────────────────────────────────────────────────────── #

class TestLfpWidgets:
    def test_spectrogram_view_with_precomputed(self, qapp):
        from neurobox.gui.mta_browser.model       import PlaybackModel
        from neurobox.gui.mta_browser.lfp_widgets import SpectrogramView
        ses = _make_session()
        _, _, spec = _make_lfp_and_spec()
        n = spec.power.shape[0]
        fr = 1.0 / (spec.times[1] - spec.times[0])
        m = PlaybackModel.from_data(ses, n_samples=n, samplerate=fr)
        v = SpectrogramView(m, spectrogram=spec)
        v.show(); qapp.processEvents()
        assert v.has_data
        assert v.frame_rate_hz is not None
        assert abs(v.frame_rate_hz - fr) < 1e-6
        m.set_idx(100); qapp.processEvents()
        v.deleteLater()

    def test_spectrogram_view_compute_from_lfp(self, qapp):
        from neurobox.gui.mta_browser.model       import PlaybackModel
        from neurobox.gui.mta_browser.lfp_widgets import SpectrogramView
        ses = _make_session()
        lfp, sr_lfp, _ = _make_lfp_and_spec()
        # Use small spec size for speed
        m = PlaybackModel.from_data(ses, n_samples=200, samplerate=5.0)
        v = SpectrogramView(
            m, lfp=lfp, lfp_samplerate=sr_lfp,
            spec_kwargs={"freq_range": (1.0, 30.0),
                          "n_fft": 512, "win_len": 1250,
                          "n_overlap": 1000},
        )
        v.show(); qapp.processEvents()
        assert v.has_data
        v.deleteLater()

    def test_spectrogram_view_no_data(self, qapp):
        from neurobox.gui.mta_browser.model       import PlaybackModel
        from neurobox.gui.mta_browser.lfp_widgets import SpectrogramView
        ses = _make_session()
        m = PlaybackModel.from_data(ses, n_samples=100, samplerate=5.0)
        v = SpectrogramView(m)             # neither spectrogram nor lfp
        v.show(); qapp.processEvents()
        assert not v.has_data
        v.deleteLater()

    def test_lfp_trace_view(self, qapp):
        from neurobox.gui.mta_browser.model       import PlaybackModel
        from neurobox.gui.mta_browser.lfp_widgets import LfpTraceView
        ses = _make_session()
        lfp, sr_lfp, _ = _make_lfp_and_spec()
        # Model at xyz rate, but LFP at LFP rate — widget converts
        m = PlaybackModel.from_session(ses)
        v = LfpTraceView(m, lfp, sr_lfp, window_seconds=0.5)
        v.show(); qapp.processEvents()
        # Scrub and verify no exception
        m.set_idx(900); qapp.processEvents()
        v.deleteLater()

    def test_lfp_trace_with_subset_of_channels(self, qapp):
        from neurobox.gui.mta_browser.model       import PlaybackModel
        from neurobox.gui.mta_browser.lfp_widgets import LfpTraceView
        ses = _make_session()
        T = int(50 * 1250)
        lfp = np.random.randn(T, 8)         # 8 channels
        m = PlaybackModel.from_session(ses)
        v = LfpTraceView(m, lfp, 1250.0,
                            channels=[0, 2, 4],
                            window_seconds=0.5)
        v.show(); qapp.processEvents()
        v.deleteLater()


# ─────────────────────────────────────────────────────────────────────── #
# _LfpStatesTab integration                                                  #
# ─────────────────────────────────────────────────────────────────────── #

class TestLfpStatesTab:
    def test_attach_with_precomputed_spectrogram(self, qapp):
        from neurobox.gui.mta_browser.main_window import _LfpStatesTab
        ses = _make_session()
        _, _, spec = _make_lfp_and_spec()
        tab = _LfpStatesTab()
        tab.attach_lfp_data(ses, spectrogram=spec)
        qapp.processEvents()
        assert tab.model is not None
        # Frame rate should match the spectrogram
        expected_fr = 1.0 / (spec.times[1] - spec.times[0])
        assert abs(tab.model.samplerate - expected_fr) < 1e-6
        # Theta should be populated from session.stc
        assert "theta" in tab.model.states_data
        tab.deleteLater()

    def test_attach_with_raw_lfp_only(self, qapp):
        from neurobox.gui.mta_browser.main_window import _LfpStatesTab
        ses = _make_session()
        lfp, sr_lfp, _ = _make_lfp_and_spec()
        tab = _LfpStatesTab()
        tab.attach_lfp_data(ses, lfp=lfp, lfp_samplerate=sr_lfp,
                              spec_kwargs={"freq_range": (1.0, 30.0)})
        qapp.processEvents()
        assert tab.model is not None
        # Frame rate is determined by the computed spec, not lfp_samplerate
        assert tab.model.samplerate < sr_lfp
        tab.deleteLater()

    def test_attach_requires_data(self, qapp):
        from neurobox.gui.mta_browser.main_window import _LfpStatesTab
        ses = _make_session()
        tab = _LfpStatesTab()
        with pytest.raises(ValueError):
            tab.attach_lfp_data(ses)             # neither arg
        tab.deleteLater()

    def test_paint_states_at_lfp_rate(self, qapp):
        """State labelling at the spec rate round-trips through
        session.stc in seconds."""
        from neurobox.gui.mta_browser.main_window import _LfpStatesTab
        ses = _make_session(theta_period=None)   # start empty
        _, _, spec = _make_lfp_and_spec()
        tab = _LfpStatesTab()
        tab.attach_lfp_data(ses, spectrogram=spec)
        m = tab.model
        m.add_state("theta", "t")
        m.set_current_label("theta")
        m.set_idx(25); m.step(10)            # paint 5..7 s @ 5 Hz spec rate
        tab.commit_to_session()
        ep = ses.stc.get_state("theta")
        # Period should be roughly [5, 7] s
        np.testing.assert_allclose(ep.data[0], [5.0, 7.0], atol=0.5)
        tab.deleteLater()

    def test_keyboard_shortcuts(self, qapp):
        from neurobox.gui.mta_browser.main_window import _LfpStatesTab
        ses = _make_session()
        _, _, spec = _make_lfp_and_spec()
        tab = _LfpStatesTab()
        tab.attach_lfp_data(ses, spectrogram=spec)
        m = tab.model

        # Tap 't' → current_label = theta
        m.set_current_label(None)
        ev = QKeyEvent(QEvent.KeyPress, Qt.Key_T, Qt.NoModifier, "t")
        tab.keyPressEvent(ev)
        assert m.current_label == "theta"
        # Tap 't' again → current_label = None (deselect)
        tab.keyPressEvent(ev)
        assert m.current_label is None
        # Ctrl+→ jump 50
        m.set_idx(20)
        ev2 = QKeyEvent(QEvent.KeyPress, Qt.Key_Right, Qt.ControlModifier)
        tab.keyPressEvent(ev2)
        assert m.idx == 70
        # Delete toggles erase mode
        ev3 = QKeyEvent(QEvent.KeyPress, Qt.Key_Delete, Qt.NoModifier)
        tab.keyPressEvent(ev3)
        assert m.erase_mode
        tab.keyPressEvent(ev3)
        assert not m.erase_mode
        tab.deleteLater()


# ─────────────────────────────────────────────────────────────────────── #
# MTABrowserWindow integration                                                #
# ─────────────────────────────────────────────────────────────────────── #

class TestMtaBrowserWindowLfp:
    def test_three_tabs(self, qapp):
        from neurobox.gui.mta_browser import MTABrowserWindow
        ses = _make_session()
        win = MTABrowserWindow.launch(session=ses, run=False)
        assert win._tabs.count() == 3
        assert win._tabs.tabText(2) == "LFP States"
        win.close()

    def test_session_with_spec_attaches_lfp_tab(self, qapp):
        """When session.spec exists at load time, the LFP tab gets
        auto-attached."""
        from neurobox.gui.mta_browser import MTABrowserWindow
        ses = _make_session()
        _, _, spec = _make_lfp_and_spec()
        ses.spec = spec
        win = MTABrowserWindow.launch(session=ses, run=False)
        assert win._lfp_tab.model is not None
        # Frame rate matches the spec, NOT the xyz rate
        expected_fr = 1.0 / (spec.times[1] - spec.times[0])
        assert abs(win._lfp_tab.model.samplerate - expected_fr) < 1e-6
        # Different from ML tab's rate
        assert win._lfp_tab.model.samplerate != win._ml_tab.model.samplerate
        win.close()

    def test_attach_lfp_after_session(self, qapp):
        """Programmatic API: attach LFP data after the session."""
        from neurobox.gui.mta_browser import MTABrowserWindow
        ses = _make_session()
        win = MTABrowserWindow.launch(session=ses, run=False)
        # No spec on session ⇒ LFP tab shouldn't be auto-populated
        assert win._lfp_tab.model is None
        # Now attach via the public API
        _, _, spec = _make_lfp_and_spec()
        win.attach_lfp(spectrogram=spec)
        qapp.processEvents()
        assert win._lfp_tab.model is not None
        win.close()

    def test_tab_switch_syncs_state_edits(self, qapp):
        """Edits in ML tab become visible in LFP tab on switch, and
        vice versa."""
        from neurobox.gui.mta_browser import MTABrowserWindow
        ses = _make_session()
        _, _, spec = _make_lfp_and_spec()
        ses.spec = spec
        win = MTABrowserWindow.launch(session=ses, run=False)
        ml = win._ml_tab; lt = win._lfp_tab

        # Add 'rear' state in the ML tab and paint
        ml.model.add_state("rear", "r")
        ml.model.set_current_label("rear")
        ml.model.set_idx(900); ml.model.step(60)   # 30..32 s

        # Switch to LFP tab → triggers commit + refresh
        win._tabs.setCurrentWidget(lt)
        qapp.processEvents()
        assert "rear" in lt.model.states_data
        # 2 s @ ~5 Hz ≈ 10 spec frames painted
        rear_count = (lt.model.states_data["rear"] > 0).sum()
        assert 5 < rear_count < 20

        # Paint in LFP tab and switch back to ML tab
        lt.model.add_state("ripple", "i")
        lt.model.set_current_label("ripple")
        lt.model.set_idx(150); lt.model.step(10)   # 30..32 s @ 5 Hz
        win._tabs.setCurrentWidget(ml)
        qapp.processEvents()
        assert "ripple" in ml.model.states_data
        win.close()

    def test_session_with_lfp_only_no_spec(self, qapp):
        """Session has lfp but no precomputed spec — LFP tab still
        attaches by computing the spec on the fly."""
        from neurobox.gui.mta_browser import MTABrowserWindow
        ses = _make_session()
        lfp, sr_lfp, _ = _make_lfp_and_spec()

        class _FakeLfp:
            data = lfp
            samplerate = sr_lfp
        ses.lfp = _FakeLfp()

        win = MTABrowserWindow.launch(session=ses, run=False)
        # The LFP tab attached using raw LFP path
        assert win._lfp_tab.model is not None
        win.close()
