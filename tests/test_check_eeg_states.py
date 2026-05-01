"""Tests for neurobox.gui.check_eeg_states fixes (round 7).

These tests exercise the non-Qt parts of the GUI (period management,
NBEpoch round-trip, auto-segmentation algorithm) by constructing a
window in offscreen mode and calling its methods directly.

Skipped if PySide6 isn't installed.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

# Headless mode must be set before any Qt or matplotlib import
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")

PySide6 = pytest.importorskip("PySide6")
hmmlearn = pytest.importorskip("hmmlearn")


# ─────────────────────────────────────────────────────────────────────────── #
# Shared QApplication                                                          #
# ─────────────────────────────────────────────────────────────────────────── #

@pytest.fixture(scope="session")
def qapp():
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    return app


# ─────────────────────────────────────────────────────────────────────────── #
# Synthetic spectrogram + window factory                                       #
# ─────────────────────────────────────────────────────────────────────────── #

def _make_synthetic_spec(
    duration: float = 60.0,
    fs: float = 1250.0,
    f_lo: float = 1.0,
    f_hi: float = 20.0,
    theta_periods: list = None,
):
    """Build a (F, T, C) log-power spectrogram with theta epochs.

    The trace alternates between low-power and high-power-in-the-theta-band
    according to ``theta_periods``.  Used to drive _auto_segment with a
    well-defined ground truth.
    """
    n_freqs = 32
    win_sec = 1.0
    step    = win_sec * 0.5
    n_steps = int(duration / step)
    freqs = np.linspace(f_lo, f_hi, n_freqs)
    times = np.arange(n_steps) * step + win_sec / 2

    rng = np.random.default_rng(0)
    spec = np.exp(rng.standard_normal((n_freqs, n_steps, 1)) * 0.2 - 1.0)

    # Bump the 5-12 Hz band during theta periods
    if theta_periods is None:
        theta_periods = [(15.0, 25.0), (40.0, 55.0)]
    theta_band = (freqs >= 5) & (freqs <= 12)
    for t0, t1 in theta_periods:
        ti = (times >= t0) & (times <= t1)
        spec[theta_band[:, None] & ti[None, :], 0] *= 4.0

    return spec, freqs, times


def _make_window(qapp, periods=None, state_name="theta"):
    from neurobox.gui.check_eeg_states import CheckEegStatesWindow
    spec, freqs, times = _make_synthetic_spec()
    win = CheckEegStatesWindow(
        file_base   = "/tmp/test_session",
        spec        = spec,
        spec_freqs  = freqs,
        spec_times  = times,
        lfp_sr      = 1250.0,
        n_channels  = 1,
        n_samples   = int(60.0 * 1250.0),
        channels    = [0],
        raw_lfp     = None,
        state_name  = state_name,
        periods     = periods,
    )
    return win


# ─────────────────────────────────────────────────────────────────────────── #
# Period ↔ NBEpoch helper                                                      #
# ─────────────────────────────────────────────────────────────────────────── #

class TestPeriodsAsEpoch:

    def test_empty_periods_returns_empty_epoch(self, qapp):
        win = _make_window(qapp, periods=None)
        ep = win._periods_as_epoch()
        assert ep.isempty()
        assert ep.samplerate == 1250.0
        assert ep.label == "theta"

    def test_drops_nan_tombstones(self, qapp):
        per = np.array([
            [1.0, 5.0],
            [np.nan, np.nan],
            [10.0, 15.0],
        ])
        win = _make_window(qapp, periods=per)
        ep = win._periods_as_epoch()
        assert len(ep) == 2
        np.testing.assert_array_equal(ep.data, [[1, 5], [10, 15]])

    def test_drops_zero_duration(self, qapp):
        per = np.array([
            [1.0, 5.0],
            [3.0, 3.0],   # zero-length
            [10.0, 15.0],
        ])
        win = _make_window(qapp, periods=per)
        ep = win._periods_as_epoch()
        assert len(ep) == 2

    def test_sorts_by_start(self, qapp):
        per = np.array([
            [10.0, 15.0],
            [1.0, 5.0],
            [20.0, 25.0],
        ])
        win = _make_window(qapp, periods=per)
        ep = win._periods_as_epoch()
        np.testing.assert_array_equal(ep.data, [[1, 5], [10, 15], [20, 25]])

    def test_carries_samplerate_and_label(self, qapp):
        per = np.array([[1.0, 5.0]])
        win = _make_window(qapp, periods=per, state_name="REM")
        ep = win._periods_as_epoch()
        assert ep.samplerate == 1250.0
        assert ep.label == "REM"


# ─────────────────────────────────────────────────────────────────────────── #
# Save/load round-trip                                                         #
# ─────────────────────────────────────────────────────────────────────────── #

class TestSaveLoadRoundTrip:

    def test_pkl_roundtrip(self, qapp, tmp_path):
        from neurobox.dtype.epoch import NBEpoch
        per = np.array([[1.0, 5.0], [10.0, 20.0]])
        win = _make_window(qapp, periods=per, state_name="theta")

        # Manually invoke the NBEpoch save (mirrors what _save_periods does)
        epoch = win._periods_as_epoch()
        pkl_path = tmp_path / "test.epoch.pkl"
        epoch.save(pkl_path, overwrite=True)
        assert pkl_path.exists()

        # Round-trip
        loaded = NBEpoch.load_file(pkl_path)
        assert loaded.samplerate == 1250.0
        assert loaded.label == "theta"
        np.testing.assert_array_equal(loaded.data, per)

    def test_sts_format_is_sample_indexed(self, qapp, tmp_path):
        per = np.array([[1.0, 5.0], [10.0, 20.0]])
        win = _make_window(qapp, periods=per, state_name="theta")

        # Replicate the labbox-compat .sts write path
        epoch = win._periods_as_epoch()
        in_samp = np.round(epoch.data * 1250.0).astype(np.int64)
        sts_path = tmp_path / "test.sts.theta"
        np.savetxt(sts_path, in_samp, fmt="%d")

        # Re-load via the .sts text path
        raw = np.loadtxt(sts_path, ndmin=2).astype(np.int64)
        np.testing.assert_array_equal(raw, in_samp)


# ─────────────────────────────────────────────────────────────────────────── #
# _update_periods uses NBEpoch canonicalisation                                #
# ─────────────────────────────────────────────────────────────────────────── #

class TestUpdatePeriods:

    def test_sorts_unsorted_pairs(self, qapp):
        # Boundaries given out of order — _update_periods should sort them
        per = np.array([
            [10.0, 15.0],   # second
            [1.0,  5.0],    # first
        ])
        win = _make_window(qapp, periods=per)
        win._update_periods()
        # Should have sorted by start time
        np.testing.assert_array_equal(win._s.periods, [[1, 5], [10, 15]])

    def test_drops_nans_and_pairs_remaining(self, qapp):
        per = np.array([
            [1.0,    5.0],
            [np.nan, np.nan],
            [10.0,   15.0],
        ])
        win = _make_window(qapp, periods=per)
        win._update_periods()
        assert win._s.periods.shape == (2, 2)
        np.testing.assert_array_equal(win._s.periods, [[1, 5], [10, 15]])

    def test_empty_input_noop(self, qapp):
        win = _make_window(qapp, periods=np.zeros((0, 2)))
        # Should not crash; should not modify
        win._update_periods()
        assert win._s.periods.shape == (0, 2)


# ─────────────────────────────────────────────────────────────────────────── #
# Auto-segmentation uses gauss_hmm                                             #
# ─────────────────────────────────────────────────────────────────────────── #

class TestAutoSegment:

    def test_uses_gauss_hmm(self, qapp, monkeypatch):
        """When called, _auto_segment must use neurobox.analysis.stats.gauss_hmm,
        not the silently-broken old _fit_gaussian_hmm_2state path."""
        from neurobox.analysis.stats import gauss_hmm as real_gauss_hmm

        win = _make_window(qapp)
        # Pre-populate one period spanning the synthetic theta region
        win._s.periods = np.array([[10.0, 60.0]])
        win._s.current_t = 30.0    # cursor inside the period

        # Spy on gauss_hmm to confirm it gets called
        from neurobox.analysis.stats import hmm as hmm_mod
        call_log = []
        original = hmm_mod.gauss_hmm

        def _spy(*args, **kwargs):
            call_log.append((args, kwargs))
            return original(*args, **kwargs)

        monkeypatch.setattr(hmm_mod, "gauss_hmm", _spy)

        # Patch the dialog to auto-accept with reasonable defaults
        from neurobox.gui import check_eeg_states as ces_mod
        from PySide6.QtWidgets import QDialog

        class _StubDialog:
            def __init__(self, parent, t_range, n_channels):
                self._t_range = t_range
            def exec(self):
                return QDialog.DialogCode.Accepted
            def result(self):
                return ([5.0, 12.0],          # frin
                        [(1.0, 5.0), (12.0, 15.0)],  # frout
                        0,                     # ch_idx
                        self._t_range[0],
                        self._t_range[1])

        monkeypatch.setattr(ces_mod, "_AutoSegDialog", _StubDialog)
        monkeypatch.setattr(ces_mod.QMessageBox, "information",
                            lambda *a, **kw: None)
        monkeypatch.setattr(ces_mod.QMessageBox, "warning",
                            lambda *a, **kw: None)

        win._auto_segment()

        assert len(call_log) >= 1, "gauss_hmm was not called"
        # Confirm n_states=2 was passed
        assert call_log[0][1].get("n_states") == 2

    def test_replaces_parent_period(self, qapp, monkeypatch):
        """The cursor's parent period is deleted (NaN'd) and replaced
        with HMM-derived sub-periods."""
        win = _make_window(qapp)
        # Synthetic theta lives in (15-25) and (40-55).  We give a
        # single coarse period spanning them, with the cursor inside.
        win._s.periods = np.array([[10.0, 60.0]])
        win._s.current_t = 30.0

        from neurobox.gui import check_eeg_states as ces_mod
        from PySide6.QtWidgets import QDialog

        class _StubDialog:
            def __init__(self, parent, t_range, n_channels):
                self._t_range = t_range
            def exec(self):
                return QDialog.DialogCode.Accepted
            def result(self):
                return ([5.0, 12.0], [(1.0, 5.0), (12.0, 15.0)], 0,
                        self._t_range[0], self._t_range[1])

        monkeypatch.setattr(ces_mod, "_AutoSegDialog", _StubDialog)
        monkeypatch.setattr(ces_mod.QMessageBox, "information",
                            lambda *a, **kw: None)
        monkeypatch.setattr(ces_mod.QMessageBox, "warning",
                            lambda *a, **kw: None)

        n_before = len(win._s.periods)
        win._auto_segment()

        # Parent period should be NaN'd (tombstone)
        assert np.isnan(win._s.periods[0]).all()
        # New periods appended below
        assert len(win._s.periods) > n_before


# ─────────────────────────────────────────────────────────────────────────── #
# Colour-mode arithmetic — additive, σ-based                                   #
# ─────────────────────────────────────────────────────────────────────────── #

class TestColourMode:

    def test_up_increases_ceiling_by_half_sigma(self, qapp):
        win = _make_window(qapp)
        # Force a known clim and σ
        win._s.clim = (-3.0, 3.0)
        win._spec_dev = 1.0
        win._s.mode = "c"

        class _Ev:
            key = "up"
        win._handle_submode_key("up")
        # Ceiling should increase by 0.5 * σ = 0.5
        np.testing.assert_allclose(win._s.clim[1], 3.5)
        np.testing.assert_allclose(win._s.clim[0], -3.0)   # floor unchanged

    def test_down_decreases_ceiling(self, qapp):
        win = _make_window(qapp)
        win._s.clim = (-3.0, 3.0)
        win._spec_dev = 1.0
        win._s.mode = "c"
        win._handle_submode_key("down")
        np.testing.assert_allclose(win._s.clim[1], 2.5)

    def test_left_decreases_floor(self, qapp):
        win = _make_window(qapp)
        win._s.clim = (-3.0, 3.0)
        win._spec_dev = 2.0
        win._s.mode = "c"
        win._handle_submode_key("left")
        # Floor decreases by 0.5 * 2.0 = 1.0
        np.testing.assert_allclose(win._s.clim[0], -4.0)

    def test_b_key_lowers_floor(self, qapp):
        win = _make_window(qapp)
        win._s.clim = (-3.0, 3.0)
        win._spec_dev = 1.0
        win._s.mode = "c"
        win._handle_submode_key("b")
        # b in c-mode lowers the floor by 0.5 σ
        np.testing.assert_allclose(win._s.clim[0], -3.5)

    def test_c_exits_mode(self, qapp):
        win = _make_window(qapp)
        win._s.clim = (-3.0, 3.0)
        win._spec_dev = 1.0
        win._s.mode = "c"
        win._handle_submode_key("c")
        assert win._s.mode == "t"


# ─────────────────────────────────────────────────────────────────────────── #
# _on_key_press dispatch                                                       #
# ─────────────────────────────────────────────────────────────────────────── #

class TestOnKeyPress:

    def test_mode_entry_keys(self, qapp):
        win = _make_window(qapp)
        for key in ("t", "n", "m", "d", "z", "c", "f", "w"):
            win._s.mode = "t"  # reset

            class _Ev:
                pass
            ev = _Ev()
            ev.key = key
            # Set _spec_dev so c-mode handler doesn't crash on a follow-up
            win._spec_dev = 1.0
            win._on_key_press(ev)
            # Should have entered the named mode
            assert win._s.mode == key, f"key={key} didn't enter mode"

    def test_action_keys_dont_change_mode(self, qapp, monkeypatch):
        win = _make_window(qapp)
        win._s.mode = "t"

        # Stub out _show_help so it doesn't open a modal QMessageBox.exec()
        monkeypatch.setattr(win, "_show_help", lambda: None)

        from neurobox.gui import check_eeg_states as ces_mod
        monkeypatch.setattr(ces_mod.QMessageBox, "information",
                            lambda *a, **kw: None)

        class _Ev:
            pass

        for key in ("u", "h"):
            ev = _Ev()
            ev.key = key
            win._on_key_press(ev)
            assert win._s.mode == "t", f"action key {key} changed mode"

    def test_q_closes(self, qapp, monkeypatch):
        win = _make_window(qapp)
        closed = []
        monkeypatch.setattr(win, "close", lambda: closed.append(True))

        class _Ev:
            key = "q"
        win._on_key_press(_Ev())
        assert closed == [True]


# ─────────────────────────────────────────────────────────────────────────── #
# Reference-point annotation                                                   #
# ─────────────────────────────────────────────────────────────────────────── #

class TestRefPoints:

    def test_draw_ref_annotation_called_when_ref_points_set(self, qapp):
        """When ref_points is set, _draw_cursor calls _draw_ref_annotation
        (which updates the status bar with elapsed time)."""
        win = _make_window(qapp)
        win._s.ref_points = np.array([0.0, 30.0, 60.0])
        win._s.current_t = 35.0

        # Spy on the annotation method
        called = []
        original = win._draw_ref_annotation
        win._draw_ref_annotation = lambda: called.append(True) or original()
        win._draw_cursor()
        assert called, "_draw_ref_annotation was not invoked"
