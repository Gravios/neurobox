"""
neurobox.gui.check_eeg_states
==============================
Interactive LFP spectrogram browser and brain-state period annotator.

Port of ``CheckEegStates.m`` / ``CheckEegStates_aux.m`` (Anton Sirota /
Evgeny Resnik, labbox TF toolkit) to Python using PySide6 ≥ 6.8 and
matplotlib embedded in Qt.

Usage
-----
From a script or IPython::

    from neurobox.gui.check_eeg_states import CheckEegStatesWindow
    from neurobox.dtype.session import NBSession

    session = NBSession("jg05-20120316", maze="cof")
    session.load("lfp", channels=[0, 1, 2, 3])
    session.load("spk")

    app = CheckEegStatesWindow.launch(
        session      = session,
        state_name   = "theta",
        freq_range   = (1, 140),
        channels     = [0, 1, 2, 3],
        window_sec   = 1.0,
    )

Or from the command line::

    python -m neurobox.gui.check_eeg_states \\
        /data/project/B01/jg05-20120316/jg05-20120316 \\
        --state theta --channels 0 1 2 3

Key bindings (match original CheckEegStates)
--------------------------------------------
t / Left-click     — navigate / view LFP trace at cursor
n / Right-click    — add new state period boundary
m / Middle-click   — move nearest period boundary
d                  — delete nearest period boundary
z                  — toggle x-axis zoom mode
c + ↑↓             — adjust spectrogram colour range
f + ↑↓             — zoom spectrogram frequency axis
w + ↑↓             — resize LFP trace window
u                  — update periods (sort, remove NaN, recolour)
s                  — save periods to <filebase>.sts.<state>
l                  — load periods from file
a                  — auto-segment with HMM (theta/delta ratio)
p                  — screenshot PNG
h                  — help
q / Escape         — quit

Period display
--------------
Black vertical lines = period starts.
Magenta vertical lines = period ends.
After pressing ``u`` borders are sorted and paired: blue = start, red = end.
"""

from __future__ import annotations

import os
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("QtAgg")                          # must precede pyplot import
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize

from PySide6.QtCore    import Qt, QTimer, Signal, Slot, QThread, QObject
from PySide6.QtGui     import QKeySequence, QAction, QIcon, QCursor, QColor, QUndoStack, QUndoCommand
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QStatusBar, QToolBar, QFileDialog, QInputDialog,
    QMessageBox, QDialog, QDialogButtonBox, QFormLayout, QLineEdit,
    QLabel, QComboBox, QTableWidget, QTableWidgetItem, QDockWidget,
    QProgressDialog, QHeaderView, QAbstractItemView,
)


# ─────────────────────────────────────────────────────────────────────────── #
# State dataclass                                                              #
# ─────────────────────────────────────────────────────────────────────────── #

@dataclass
class _AppState:
    """All mutable GUI state in one place (replaces MATLAB global struct)."""

    # Session identity
    file_base:   str   = ""
    state_name:  str   = ""
    lfp_sr:      float = 1250.0
    n_channels:  int   = 1
    n_samples:   int   = 0
    channels:    list  = field(default_factory=list)

    # Spectral data
    spec:        Optional[np.ndarray] = None   # (F, T, C) log-power
    spec_freqs:  Optional[np.ndarray] = None   # (F,)
    spec_times:  Optional[np.ndarray] = None   # (T,)

    # Aux data  [(x_arr, y_arr_or_None, data_arr, 'plot'|'imagesc'), ...]
    aux_data:    list = field(default_factory=list)

    # Period annotations  (N, 2) float array in seconds; NaN = deleted
    periods:     Optional[np.ndarray] = None   # shape (N,2) or None

    # Navigation
    current_t:   float = 0.0        # cursor position (seconds)
    x_range:     Optional[tuple] = None   # (t_start, t_end) of spectrogram view
    freq_range:  tuple = (1.0, 20.0)
    window_sec:  float = 1.0

    # Colour limits
    clim:        Optional[tuple] = None

    # Interaction mode
    mode:       str = "t"    # t, n, m, d, z, c, f, w

    # Pending half-period (when adding with keyboard 'n' or right-click)
    pending_start: Optional[float] = None

    # Drag state for line-move
    dragging_period_idx:  int = -1
    dragging_side:        int = -1   # 0=start, 1=end
    ref_points:   Optional[np.ndarray] = None   # sub-recording boundary times


# ─────────────────────────────────────────────────────────────────────────── #
# Main window                                                                 #
# ─────────────────────────────────────────────────────────────────────────── #

class CheckEegStatesWindow(QMainWindow):
    """Qt6 port of CheckEegStates."""

    _HELP_TEXT = """\
Keyboard shortcuts
──────────────────
t           Navigate mode (move cursor, update trace)
n           Add period boundary  (right-click also works)
m           Move nearest boundary  (middle-click also works)
d           Delete nearest boundary
z           Zoom mode  (left-click zooms in, right-click zooms out)
c + ↑/↓    Adjust spectrogram colour ceiling / floor
f + ↑/↓    Zoom frequency axis
w + ↑/↓    Resize LFP trace window
← / →       Pan spectrogram left / right
Space       Jump to next period start
u           Update periods (sort, clean, recolour)
s           Save periods to <filebase>.sts.<state>
l           Load periods from file
a           Auto-segment selected region with HMM
p           Screenshot PNG
h           This help
q / Esc     Quit

Mouse actions
─────────────
Left-click    Navigate / set trace position
Right-click   Add period boundary
Middle-click  Move nearest boundary
"""

    def __init__(
        self,
        file_base:    str,
        spec:         np.ndarray,        # (F, T, C)
        spec_freqs:   np.ndarray,
        spec_times:   np.ndarray,
        lfp_sr:       float,
        n_channels:   int,
        n_samples:    int,
        channels:     list,
        raw_lfp:      Optional[np.ndarray],  # (n_samples, n_channels) or None
        state_name:   str   = "",
        periods:      Optional[np.ndarray] = None,
        freq_range:   tuple = (1.0, 20.0),
        window_sec:   float = 1.0,
        aux_data:     Optional[list] = None,
        parent:       Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)

        # ── State ────────────────────────────────────────────────────────── #
        self._s = _AppState(
            file_base   = file_base,
            state_name  = state_name,
            lfp_sr      = lfp_sr,
            n_channels  = n_channels,
            n_samples   = n_samples,
            channels    = channels,
            spec        = spec,
            spec_freqs  = spec_freqs,
            spec_times  = spec_times,
            periods     = periods if periods is not None else np.zeros((0, 2)),
            freq_range  = freq_range,
            window_sec  = window_sec,
            aux_data    = aux_data or [],
            current_t   = float(spec_times[0]) if spec_times is not None else 0.0,
            x_range     = (float(spec_times[0]), min(float(spec_times[0]) + 120, float(spec_times[-1])))
                          if spec_times is not None else (0, 120),
        )
        self._raw_lfp = raw_lfp          # (n_samples, n_channels) int16 or float
        self._period_lines: list[list] = []   # [plot_panel_idx][period_row] → (Line2D, Line2D)
        self._cursor_lines: list = []         # one per panel
        self._ref_lines:    list = []         # ref-point vlines
        self._cbar_axes:    list = []         # colorbar axes
        self._lfp_path:     Optional[Path] = None   # for lazy disk seek
        self._lfp_n_channels: int = 0
        self._undo_stack = QUndoStack(self)
        self._undo_stack.setUndoLimit(50)

        # ── Window chrome ─────────────────────────────────────────────────── #
        self.setWindowTitle(self._title())
        self.resize(1400, 900)
        self._build_menu()
        self._build_toolbar()
        self._build_figure()
        self._build_statusbar()

        # ── Initial draw ──────────────────────────────────────────────────── #
        self._draw_spectrograms()
        self._draw_aux_panels()
        self._draw_trace()
        self._draw_period_lines()
        self._draw_cursor()
        self._canvas.draw_idle()

        # ── Period dock ───────────────────────────────────────────────────── #
        self._build_period_dock()
        self._refresh_period_table()

        # ── Mode label timer (flash feedback) ─────────────────────────────── #
        self._mode_timer = QTimer(self)
        self._mode_timer.setSingleShot(True)

    # ── Construction helpers ──────────────────────────────────────────────── #

    def _title(self) -> str:
        mode_labels = {
            "t": "navigate", "n": "add boundary", "m": "move boundary",
            "d": "delete boundary", "z": "zoom", "c": "colour adjust",
            "f": "freq zoom", "w": "trace window",
        }
        s = self._s
        name = Path(s.file_base).name or s.file_base
        mode = mode_labels.get(s.mode, s.mode)
        state = f" [{s.state_name}]" if s.state_name else ""
        return f"CheckEegStates — {name}{state}  |  {mode}"

    def _n_panels(self) -> int:
        return self._s.n_channels + len(self._s.aux_data) + 1

    def _build_figure(self) -> None:
        n = self._n_panels()
        heights = [3] * self._s.n_channels + [2] * len(self._s.aux_data) + [2]
        fig = Figure(figsize=(14, max(6, n * 2)), tight_layout=True)
        self._axes = fig.subplots(
            n, 1,
            gridspec_kw={"height_ratios": heights, "hspace": 0.08},
            sharex=False,
        )
        if n == 1:
            self._axes = [self._axes]
        else:
            self._axes = list(self._axes)

        self._ax_spec  = self._axes[:self._s.n_channels]
        self._ax_aux   = self._axes[self._s.n_channels:self._s.n_channels + len(self._s.aux_data)]
        self._ax_trace = self._axes[-1]
        # Reserve small axes for colorbars (created after first pcolormesh)
        self._cbar_axes = []

        self._canvas = FigureCanvasQTAgg(fig)
        self._canvas.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._canvas.setFocus()
        self._canvas.mpl_connect("button_press_event",   self._on_mouse_press)
        self._canvas.mpl_connect("motion_notify_event",  self._on_mouse_move)
        self._canvas.mpl_connect("button_release_event", self._on_mouse_release)
        self._canvas.mpl_connect("key_press_event",      self._on_key_press)
        self._canvas.mpl_connect("scroll_event",         self._on_scroll)

        self._fig = fig

        # Embed canvas
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._canvas)
        self.setCentralWidget(central)

        # Period line store: list of lists: _period_lines[panel][row] = (Line2D|None, Line2D|None)
        self._period_lines = [[] for _ in range(self._n_panels())]
        self._cursor_lines = [None] * self._n_panels()

    def _build_menu(self) -> None:
        mb = self.menuBar()

        # File menu
        file_m = mb.addMenu("&File")
        self._act_save = QAction("&Save periods", self, shortcut="S", triggered=self._save_periods)
        self._act_load = QAction("&Load periods", self, shortcut="L", triggered=self._load_periods)
        self._act_scr  = QAction("Screensh&ot",   self, shortcut="P", triggered=self._screenshot)
        self._act_quit = QAction("&Quit",          self, shortcut="Q", triggered=self.close)
        for a in (self._act_save, self._act_load, self._act_scr, None, self._act_quit):
            if a is None:
                file_m.addSeparator()
            else:
                file_m.addAction(a)

        # Edit menu
        edit_m = mb.addMenu("&Edit")
        self._act_update = QAction("&Update periods", self, shortcut="U", triggered=self._update_periods)
        self._act_erase  = QAction("&Erase all",      self, triggered=self._erase_all)
        self._act_auto   = QAction("&Auto-segment",   self, shortcut="A", triggered=self._auto_segment)
        for a in (self._act_undo, self._act_redo, None,
                  self._act_update, self._act_erase, None, self._act_auto):
            if a is None:
                edit_m.addSeparator()
            else:
                edit_m.addAction(a)

        # Help
        help_m = mb.addMenu("&Help")
        help_m.addAction(QAction("&Keys / Help", self, shortcut="H", triggered=self._show_help))

    def _build_toolbar(self) -> None:
        tb = QToolBar("Mode", self)
        tb.setMovable(False)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, tb)

        def _btn(label, mode, shortcut=""):
            a = QAction(label, self)
            if shortcut:
                a.setShortcut(shortcut)
            a.setCheckable(True)
            a.triggered.connect(lambda checked, m=mode: self._set_mode(m))
            tb.addAction(a)
            return a

        self._tb_actions = {
            "t": _btn("🖱 Navigate", "t", "T"),
            "n": _btn("✏ Add",      "n", "N"),
            "m": _btn("↔ Move",     "m", "M"),
            "d": _btn("✕ Delete",   "d", "D"),
            "z": _btn("🔍 Zoom",    "z", "Z"),
        }
        self._tb_actions["t"].setChecked(True)
        tb.addSeparator()
        tb.addAction(QAction("Update (U)", self, triggered=self._update_periods))
        tb.addAction(QAction("Save (S)",   self, triggered=self._save_periods))
        tb.addAction(QAction("Auto (A)",   self, triggered=self._auto_segment))

    def _build_statusbar(self) -> None:
        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._status_left  = QLabel("")
        self._status_right = QLabel("")
        self._status.addWidget(self._status_left, 1)
        self._status.addPermanentWidget(self._status_right)
        self._update_status()

    def _update_status(self) -> None:
        s = self._s
        n_per = 0 if s.periods is None else np.sum(~np.isnan(s.periods).any(axis=1))
        self._status_left.setText(
            f"t = {s.current_t:.2f} s   mode = {s.mode}   "
            f"periods = {int(n_per)}"
        )
        dur = 0.0
        if s.periods is not None and n_per:
            complete = s.periods[~np.isnan(s.periods).any(axis=1)]
            dur = float(np.sum(complete[:, 1] - complete[:, 0]))
        self._status_right.setText(f"Total [{s.state_name}] = {dur:.1f} s")
        self.setWindowTitle(self._title())

    # ── Drawing ───────────────────────────────────────────────────────────── #

    def _draw_spectrograms(self) -> None:
        s = self._s
        if s.spec is None:
            return

        log_spec = np.log(s.spec + 1e-12)    # (F, T, C)
        flat = log_spec[np.isfinite(log_spec)]
        med = float(np.median(flat))
        dev = float(np.std(flat))
        if np.isnan(dev) or dev == 0:
            dev = abs(med) / 3 or 1.0

        if s.clim is None:
            s.clim = (med - 3 * dev, med + 3 * dev)

        for ch, ax in enumerate(self._ax_spec):
            ax.cla()
            ax.pcolormesh(
                s.spec_times, s.spec_freqs, log_spec[:, :, ch],
                shading="auto", cmap="inferno",
                vmin=s.clim[0], vmax=s.clim[1],
            )
            ax.set_ylim(s.freq_range)
            ax.set_xlim(s.x_range)
            ax.set_ylabel("Hz", fontsize=8)
            if ch == 0:
                ax.set_title(
                    f"Spectrogram ch {s.channels}  [{s.state_name}]",
                    fontsize=9, pad=2,
                )
            ax.tick_params(labelbottom=False, labelsize=7)
        # Add/update colorbars
        if not self._cbar_axes:
            for ax in self._ax_spec:
                div = self._fig.add_axes(
                    [ax.get_position().x1 + 0.002,
                     ax.get_position().y0,
                     0.01,
                     ax.get_position().height]
                )
                self._cbar_axes.append(div)
        self._add_colorbars()

    def _draw_aux_panels(self) -> None:
        s = self._s
        for i, (ax, item) in enumerate(zip(self._ax_aux, s.aux_data)):
            ax.cla()
            x_arr, y_arr, data_arr, disp_func = item
            if disp_func == "plot":
                ax.plot(x_arr, data_arr, lw=0.8)
            elif disp_func == "imagesc":
                d = data_arr
                if len(x_arr) != d.shape[0]:
                    d = d.T
                ax.pcolormesh(x_arr, y_arr, d.T, shading="auto")
                ax.set_aspect("auto")
            ax.set_xlim(s.x_range)
            ax.tick_params(labelbottom=False, labelsize=7)

    def _draw_trace(self) -> None:
        s = self._s
        ax = self._ax_trace

        if self._raw_lfp is None:
            ax.cla()
            ax.set_xlabel("Time (s)", fontsize=8)
            ax.set_title("LFP trace (not loaded)", fontsize=8)
            return

        # Select a window of raw LFP centred at current_t
        half = int(s.window_sec * s.lfp_sr * 1.5)
        i0   = max(0, int(s.current_t * s.lfp_sr) - half)
        i1   = min(s.n_samples, i0 + half * 2)
        _seg = self._load_trace_segment(i0, i1)
        if _seg is None:
            return
        seg  = _seg.astype(np.float32)

        # Z-score each channel for display (equivalent to MATLAB unity())
        # seg loaded via _load_trace_segment for lazy-seek support
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            seg_n = (seg - seg.mean(0)) / (seg.std(0) + 1e-9)

        t_arr = (np.arange(i0, i1) / s.lfp_sr)
        ax.cla()
        offset = 5.0
        for ch in range(seg_n.shape[1]):
            ax.plot(t_arr, seg_n[:, ch] + ch * offset, lw=0.6, color=f"C{ch}")
        ax.set_xlim(t_arr[0], t_arr[-1])
        ax.set_yticks(np.arange(seg_n.shape[1]) * offset)
        ax.set_yticklabels([str(c) for c in s.channels], fontsize=7)
        ax.set_xlabel("Time (s)", fontsize=8)
        ax.set_title(f"LFP  ch {s.channels}", fontsize=8, pad=2)
        ax.tick_params(labelsize=7)

    def _draw_period_lines(self) -> None:
        """Redraw all period boundary lines on every panel."""
        s = self._s
        n_panels = self._n_panels()

        # Clear old lines
        for panel_list in self._period_lines:
            for item in panel_list:
                if item is not None:
                    for ln in item:
                        if ln is not None and ln.axes is not None:
                            ln.remove()
        self._period_lines = [[] for _ in range(n_panels)]

        if s.periods is None or len(s.periods) == 0:
            return

        axes = self._ax_spec + self._ax_aux + [self._ax_trace]
        for pi in range(n_panels):
            ax = axes[pi]
            row_list = []
            for row in range(len(s.periods)):
                t_start, t_end = s.periods[row]
                ln_s = ax.axvline(t_start, color="k",  lw=1.5, ls="-") if not np.isnan(t_start) else None
                ln_e = ax.axvline(t_end,   color="m",  lw=1.5, ls="-") if not np.isnan(t_end)   else None
                row_list.append((ln_s, ln_e))
            self._period_lines[pi] = row_list

    def _draw_cursor(self) -> None:
        """Draw or redraw the cursor line on every panel."""
        axes = self._ax_spec + self._ax_aux + [self._ax_trace]
        t    = self._s.current_t
        for i, ax in enumerate(axes):
            if self._cursor_lines[i] is not None:
                try:
                    self._cursor_lines[i].remove()
                except Exception:
                    pass
            self._cursor_lines[i] = ax.axvline(t, color="k", lw=1.0, ls="--", alpha=0.8)

    def _refresh(self, full: bool = False) -> None:
        """Refresh the canvas.  full=True redraws spectrograms."""
        if full:
            self._draw_spectrograms()
            self._draw_aux_panels()
        self._draw_trace()
        self._draw_period_lines()
        self._draw_cursor()
        # Sync x-limits on spectrogram panels
        for ax in self._ax_spec + self._ax_aux:
            ax.set_xlim(self._s.x_range)
        self._canvas.draw_idle()
        self._update_status()

    # ── Mode management ───────────────────────────────────────────────────── #

    def _set_mode(self, mode: str) -> None:
        self._s.mode = mode
        # Update toolbar toggle state
        for m, act in self._tb_actions.items():
            act.setChecked(m == mode)
        # Update cursor shape
        cursors = {
            "t": Qt.CursorShape.ArrowCursor,
            "n": Qt.CursorShape.CrossCursor,
            "m": Qt.CursorShape.SizeHorCursor,
            "d": Qt.CursorShape.ForbiddenCursor,
            "z": Qt.CursorShape.MagnifyingGlassCursor
                 if hasattr(Qt.CursorShape, "MagnifyingGlassCursor")
                 else Qt.CursorShape.SizeBDiagCursor,
        }
        self._canvas.setCursor(QCursor(cursors.get(mode, Qt.CursorShape.ArrowCursor)))
        self._update_status()

    # ── Keyboard ──────────────────────────────────────────────────────────── #

    def _on_key_press(self, event) -> None:
        key = event.key
        s   = self._s

        # ── Sub-mode keys (require second keypress) ──────────────────────── #
        if s.mode in ("c", "f", "w"):
            self._handle_submode_key(key)
            return
        if s.mode == "z":
            self._handle_zoom_key(key)
            return

        # ── Main mode keys ───────────────────────────────────────────────── #
        if key == "t":          self._set_mode("t")
        elif key == "n":        self._set_mode("n")
        elif key == "m":        self._set_mode("m")
        elif key == "d":        self._set_mode("d")
        elif key == "z":
            self._set_mode("z")
            return
        elif key == "c":
            self._set_mode("c")
            self._status_left.setText("Colour mode: ↑ raise ceiling  ↓ lower ceiling  ← ↑ floor   c to exit")
            return
        elif key == "f":
            self._set_mode("f")
            self._status_left.setText("Freq zoom: ↑ expand  ↓ shrink   f to exit")
            return
        elif key == "w":
            self._set_mode("w")
            self._status_left.setText("Window: ↑ wider  ↓ narrower   w to exit")
            return
        elif key == "u":        self._update_periods()
        elif key == "s":        self._save_periods()
        elif key == "l":        self._load_periods()
        elif key == "a":        self._auto_segment()
        elif key == "p":        self._screenshot()
        elif key == "h":        self._show_help()
        elif key == "b":
            if self._s.clim:
                self._s.clim = (self._s.clim[0] * 1.1, self._s.clim[1])
                self._refresh(full=True)
        elif key == "r":        self._toggle_ref_points()
        elif key in ("q", "escape"):
            self.close()
            return
        elif key == " ":        self._jump_to_next_period()
        elif key == "right":    self._pan(+1)
        elif key == "left":     self._pan(-1)
        elif key == "up":       self._pan_freq(+1)
        elif key == "down":     self._pan_freq(-1)

    def _handle_submode_key(self, key: str) -> None:
        s = self._s
        if s.mode == "c":
            clim = list(s.clim or (-10, 0))
            if   key == "up":    clim[1] *= 1.1
            elif key == "down":  clim[1] /= 1.1
            elif key == "left":  clim[0] *= 1.1
            elif key == "right": clim[0] /= 1.1
            elif key == "b":     clim[0] = clim[0] * 1.1  # floor (MATLAB b)
            elif key == "c":
                self._set_mode("t")
                self._refresh()
                return
            s.clim = tuple(clim)
            self._refresh(full=True)

        elif s.mode == "f":
            lo, hi = s.freq_range
            if   key == "up":   hi = min(s.spec_freqs[-1] if s.spec_freqs is not None else 200, hi * 1.25)
            elif key == "down": hi = max(lo + 2, hi / 1.25)
            elif key == "f":
                self._set_mode("t")
                self._refresh()
                return
            s.freq_range = (lo, hi)
            for ax in self._ax_spec:
                ax.set_ylim(s.freq_range)
            self._canvas.draw_idle()

        elif s.mode == "w":
            if   key == "up":   s.window_sec = min(s.window_sec * 2, 60.0)
            elif key == "down": s.window_sec = max(s.window_sec / 2, 0.05)
            elif key == "w":
                self._set_mode("t")
                self._refresh()
                return
            self._draw_trace()
            self._canvas.draw_idle()

    def _handle_zoom_key(self, key: str) -> None:
        s = self._s
        if key == "z":
            self._set_mode("t")
            self._refresh()
            return
        t0, t1 = s.x_range
        center = s.current_t
        width  = t1 - t0
        t_min  = float(s.spec_times[0])  if s.spec_times is not None else 0
        t_max  = float(s.spec_times[-1]) if s.spec_times is not None else 1e6
        if key == "up":
            width /= 2
        elif key == "down":
            width *= 2
        elif key == "f":
            t0, t1 = t_min, t_max
            for ax in self._ax_spec + self._ax_aux:
                ax.set_xlim(t0, t1)
            s.x_range = (t0, t1)
            self._canvas.draw_idle()
            return
        new_t0 = max(t_min, center - width / 2)
        new_t1 = min(t_max, center + width / 2)
        s.x_range = (new_t0, new_t1)
        for ax in self._ax_spec + self._ax_aux:
            ax.set_xlim(s.x_range)
        self._canvas.draw_idle()

    def _pan(self, direction: int) -> None:
        s = self._s
        t0, t1 = s.x_range
        step = (t1 - t0) * 0.25 * direction
        t_min = float(s.spec_times[0]) if s.spec_times is not None else 0
        t_max = float(s.spec_times[-1]) if s.spec_times is not None else 1e6
        t0 = max(t_min, t0 + step)
        t1 = min(t_max, t1 + step)
        s.x_range = (t0, t1)
        for ax in self._ax_spec + self._ax_aux:
            ax.set_xlim(s.x_range)
        self._refresh()

    def _pan_freq(self, direction: int) -> None:
        s = self._s
        lo, hi = s.freq_range
        step = (hi - lo) * 0.2 * direction
        lo = max(0, lo + step)
        hi = max(lo + 2, hi + step)
        s.freq_range = (lo, hi)
        for ax in self._ax_spec:
            ax.set_ylim(s.freq_range)
        self._canvas.draw_idle()

    def _jump_to_next_period(self) -> None:
        s = self._s
        if s.periods is None or len(s.periods) == 0:
            return
        starts = s.periods[:, 0]
        mask   = starts > s.current_t + 0.01
        if not mask.any():
            return
        s.current_t = float(starts[mask].min())
        t0, t1 = s.x_range
        w = t1 - t0
        s.x_range = (max(0, s.current_t - 3), s.current_t - 3 + w)
        self._refresh()

    # ── Mouse ─────────────────────────────────────────────────────────────── #

    def _on_mouse_press(self, event) -> None:
        if event.inaxes is None:
            return
        x    = event.xdata
        s    = self._s
        btn  = event.button   # 1=left, 2=middle, 3=right

        t_min = float(s.spec_times[0]) if s.spec_times is not None else 0
        t_max = float(s.spec_times[-1]) if s.spec_times is not None else 1e6

        if x is None or not (t_min <= x <= t_max):
            return

        if s.mode == "z" or (s.mode == "t" and btn == 2):
            # zoom with scroll/middle
            return

        # Right-click or 'n' mode: add boundary
        if btn == 3 or s.mode == "n":
            self._add_boundary(x)
            return

        # Middle-click or 'm' mode: start moving nearest boundary
        if btn == 2 or s.mode == "m":
            self._start_move_boundary(x)
            return

        # 'd' mode: delete nearest boundary
        if s.mode == "d":
            self._delete_boundary(x)
            return

        # Zoom with left/right in zoom mode
        if s.mode == "z":
            factor = 0.5 if btn == 1 else 2.0
            t0, t1 = s.x_range
            w = (t1 - t0) * factor
            s.current_t = x
            s.x_range = (max(t_min, x - w / 2), min(t_max, x + w / 2))
            for ax in self._ax_spec + self._ax_aux:
                ax.set_xlim(s.x_range)
            self._refresh()
            return

        # Default (left-click, navigate mode): update trace
        if btn == 1:
            s.current_t = x
            self._draw_trace()
            self._draw_cursor()
            self._canvas.draw_idle()
            self._update_status()

    def _on_mouse_move(self, event) -> None:
        s = self._s
        if event.inaxes is None or event.xdata is None:
            return
        if s.dragging_period_idx >= 0:
            self._drag_boundary(event.xdata)

    def _on_mouse_release(self, event) -> None:
        s = self._s
        if s.dragging_period_idx >= 0:
            self._end_move_boundary(event.xdata)

    def _add_boundary(self, t: float) -> None:
        s = self._s
        _before = s.periods.copy()
        if s.pending_start is None:
            s.pending_start = t
            # Draw a pending (dashed) line
            for ax in self._ax_spec + self._ax_aux + [self._ax_trace]:
                ax.axvline(t, color="k", lw=1.5, ls=":")
            self._canvas.draw_idle()
            self._status_left.setText(f"Start set at {t:.2f} s — click to set end")
        else:
            t_start = min(s.pending_start, t)
            t_end   = max(s.pending_start, t)
            new_row = np.array([[t_start, t_end]])
            s.periods = np.vstack([s.periods, new_row]) if len(s.periods) else new_row
            s.pending_start = None
            self._push_undo("Add period", _before)
            self._refresh_period_table()
            self._refresh()

    def _start_move_boundary(self, t: float) -> None:
        s = self._s
        if s.periods is None or len(s.periods) == 0:
            return
        dist_s = np.abs(s.periods[:, 0] - t)
        dist_e = np.abs(s.periods[:, 1] - t)
        idx_s, idx_e = int(np.nanargmin(dist_s)), int(np.nanargmin(dist_e))
        if dist_s[idx_s] <= dist_e[idx_e]:
            s.dragging_period_idx = idx_s
            s.dragging_side = 0
        else:
            s.dragging_period_idx = idx_e
            s.dragging_side = 1

    def _drag_boundary(self, t: float) -> None:
        s = self._s
        i, side = s.dragging_period_idx, s.dragging_side
        s.periods[i, side] = t
        # Update line positions on all panels
        axes = self._ax_spec + self._ax_aux + [self._ax_trace]
        for pi, ax in enumerate(axes):
            if pi < len(self._period_lines) and i < len(self._period_lines[pi]):
                pair = self._period_lines[pi][i]
                if pair is not None and pair[side] is not None:
                    pair[side].set_xdata([t, t])
        self._canvas.draw_idle()

    def _end_move_boundary(self, t: float) -> None:
        s = self._s
        if t is not None:
            self._drag_boundary(t)
        s.dragging_period_idx = -1
        s.dragging_side = -1
        self._update_status()

    def _delete_boundary(self, t: float) -> None:
        s = self._s
        _before = s.periods.copy()
        if s.periods is None or len(s.periods) == 0:
            return
        dist_s = np.abs(s.periods[:, 0] - t)
        dist_e = np.abs(s.periods[:, 1] - t)
        idx_s, idx_e = int(np.nanargmin(dist_s)), int(np.nanargmin(dist_e))
        if dist_s[idx_s] <= dist_e[idx_e]:
            row, col = idx_s, 0
        else:
            row, col = idx_e, 1
        s.periods[row, col] = np.nan
        self._push_undo("Delete boundary", _before)
        self._refresh_period_table()
        self._refresh()

    # ── Period management ─────────────────────────────────────────────────── #

    def _update_periods(self) -> None:
        """Sort, pair, and recolour all periods.  Mirrors 'update_per' action."""
        s = self._s
        if s.periods is None or len(s.periods) == 0:
            return

        flat = s.periods.ravel()
        valid = flat[~np.isnan(flat)]
        if len(valid) % 2 != 0:
            QMessageBox.warning(self, "Update periods",
                "Odd number of valid boundaries — one border is missing.\n"
                "Check and try again.")
            return

        sorted_vals = np.sort(valid)
        s.periods = sorted_vals.reshape(-1, 2)
        self._refresh()
        # Recolour: start=blue, end=red (MATLAB 'u' behaviour)
        for pi in range(self._n_panels()):
            for row, pair in enumerate(self._period_lines[pi]):
                if pair:
                    if pair[0] is not None: pair[0].set_color("blue")
                    if pair[1] is not None: pair[1].set_color("red")
        self._canvas.draw_idle()
        self._update_status()
        self._refresh_period_table()

    def _erase_all(self) -> None:
        ans = QMessageBox.question(
            self, "Erase all", "Erase all period boundaries?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if ans == QMessageBox.StandardButton.Yes:
            self._s.periods = np.zeros((0, 2))
            self._refresh()

    # ── Auto-segmentation ─────────────────────────────────────────────────── #

    def _auto_segment(self) -> None:
        """HMM-based auto-segmentation on a selected region (port of 'a' action)."""
        s = self._s

        # Ask for parameters
        dlg = _AutoSegDialog(
            parent      = self,
            t_range     = s.x_range,
            n_channels  = s.n_channels,
        )
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        frin, frout, ch_idx, t_start, t_end = dlg.result()

        if s.spec is None or s.spec_times is None:
            QMessageBox.warning(self, "Auto-segment", "No spectrogram loaded.")
            return

        # Compute power ratio within the selected time window
        ti = (s.spec_times >= t_start) & (s.spec_times <= t_end)
        spec_win = s.spec[:, ti, ch_idx]                   # (F, T_win)
        fi_in  = (s.spec_freqs >= frin[0]) & (s.spec_freqs <= frin[1])
        fi_out = np.zeros(len(s.spec_freqs), dtype=bool)
        for f0, f1 in frout:
            fi_out |= (s.spec_freqs >= f0) & (s.spec_freqs <= f1)

        if not fi_in.any() or not fi_out.any():
            QMessageBox.warning(self, "Auto-segment",
                "No frequency bins in pass or stop band.  Adjust frequency ranges.")
            return

        ratio = (np.log(spec_win[fi_in].mean(0) + 1e-12) -
                 np.log(spec_win[fi_out].mean(0) + 1e-12))

        # Two-state Gaussian HMM
        try:
            from neurobox.analysis.lfp.spectral import _fit_gaussian_hmm_2state
            states, _ = _fit_gaussian_hmm_2state(ratio, n_iter=50)
        except ImportError:
            # Fallback: threshold at median
            states = (ratio > np.median(ratio)).astype(int)

        # State with higher mean ratio = target state
        if ratio[states == 0].mean() < ratio[states == 1].mean():
            target_state = 1
        else:
            target_state = 0

        in_target = (states == target_state)
        t_win_arr = s.spec_times[ti]
        dt = float(np.diff(s.spec_times[:2])) if len(s.spec_times) > 1 else 1.0

        # Threshold-crossing detection
        new_periods = _threshold_cross_to_periods(in_target, t_win_arr, dt, min_dur=2.0)

        if len(new_periods) == 0:
            QMessageBox.information(self, "Auto-segment", "No periods detected.")
            return

        s.periods = (
            np.vstack([s.periods, new_periods])
            if len(s.periods) > 0 else new_periods
        )
        QMessageBox.information(self, "Auto-segment",
            f"{len(new_periods)} period(s) detected.")
        self._refresh()

    # ── I/O ───────────────────────────────────────────────────────────────── #

    def _save_periods(self) -> None:
        self._update_periods()
        s = self._s

        if s.periods is None or len(s.periods) == 0:
            QMessageBox.information(self, "Save", "No periods to save.")
            return

        default_name = f"{s.file_base}.sts.{s.state_name}" if s.state_name else f"{s.file_base}.sts"
        fname, _ = QFileDialog.getSaveFileName(
            self, "Save periods", default_name, "All files (*)"
        )
        if not fname:
            return

        # Convert to samples, clip to recording length, remove zero-duration
        in_samp = np.round(s.periods * s.lfp_sr).astype(np.int64)
        in_samp = np.clip(in_samp, 0, s.n_samples)
        in_samp = in_samp[in_samp[:, 1] - in_samp[:, 0] > 0]

        np.savetxt(fname, in_samp, fmt="%d")
        self._status_left.setText(f"Saved {len(in_samp)} period(s) → {Path(fname).name}")

    def _load_periods(self) -> None:
        s   = self._s
        default = f"{s.file_base}.sts.{s.state_name}" if s.state_name else f"{s.file_base}.sts"
        fname, _ = QFileDialog.getOpenFileName(
            self, "Load periods", default, "All files (*)"
        )
        if not fname or not Path(fname).exists():
            return

        raw = np.loadtxt(fname, ndmin=2).astype(np.int64)
        s.periods = raw / s.lfp_sr    # samples → seconds

        # Infer state name from extension
        stem = Path(fname).name
        if "." in stem:
            s.state_name = stem.rsplit(".", 1)[-1]

        dur  = float(np.sum(s.periods[:, 1] - s.periods[:, 0]))
        self._status_left.setText(
            f"Loaded {len(s.periods)} period(s) from {Path(fname).name}  "
            f"| total {dur:.1f} s"
        )
        self._refresh_period_table()
        self._refresh()

    def _screenshot(self) -> None:
        s = self._s
        out = f"{s.file_base}.cesprt.{s.current_t:.1f}.png"
        self._fig.savefig(out, dpi=150, bbox_inches="tight")
        self._status_left.setText(f"Screenshot → {Path(out).name}")

    # ── Dialogs ───────────────────────────────────────────────────────────── #

    def _show_help(self) -> None:
        dlg = QMessageBox(self)
        dlg.setWindowTitle("CheckEegStates — help")
        dlg.setText(self._HELP_TEXT)
        dlg.exec()

    # ── Scroll-wheel zoom ─────────────────────────────────────────────────── #

    def _on_scroll(self, event) -> None:
        """Zoom x-axis around scroll position (new — not in MATLAB original)."""
        if event.inaxes is None or event.xdata is None:
            return
        s = self._s
        t_min = float(s.spec_times[0]) if s.spec_times is not None else 0
        t_max = float(s.spec_times[-1]) if s.spec_times is not None else 1e6
        factor = 0.7 if event.button == "up" else 1.3
        t0, t1 = s.x_range
        half = (t1 - t0) * factor / 2
        s.x_range = (max(t_min, event.xdata - half), min(t_max, event.xdata + half))
        for ax in self._ax_spec + self._ax_aux:
            ax.set_xlim(s.x_range)
        self._canvas.draw_idle()

    # ── Reference points ('r' key) ────────────────────────────────────────── #

    def _toggle_ref_points(self) -> None:
        """Load/clear session reference points from <filebase>.srslen.

        Mirrors the MATLAB 'r' action.  Draws cyan dashed lines at each
        sub-recording boundary and shows elapsed time in the status bar.
        """
        s = self._s
        from pathlib import Path as _P
        if s.ref_points is not None:
            for ln in self._ref_lines:
                try: ln.remove()
                except Exception: pass
            self._ref_lines.clear()
            s.ref_points = None
            self._canvas.draw_idle()
            return

        srslen_path = _P(s.file_base + ".srslen")
        if not srslen_path.exists():
            self._status_left.setText(
                f"No .srslen file found at {srslen_path.name}"
            )
            return

        srslen = np.loadtxt(str(srslen_path)).ravel()
        t_refs = np.concatenate([[0], np.cumsum(srslen[:-1])]) / s.lfp_sr
        s.ref_points = t_refs

        axes = self._ax_spec + self._ax_aux + [self._ax_trace]
        for t_ref in t_refs[1:]:
            for ax in axes:
                ln = ax.axvline(t_ref, color="cyan", lw=1.0, ls="--", alpha=0.7)
                self._ref_lines.append(ln)
        self._canvas.draw_idle()
        self._status_left.setText(
            f"Ref points: {len(t_refs)} sub-recordings"
        )

    def _draw_ref_annotation(self) -> None:
        """Show elapsed time since nearest ref point in status bar."""
        s = self._s
        if s.ref_points is None:
            return
        diffs = s.current_t - s.ref_points
        diffs[diffs < 0] = np.inf
        idx = int(np.argmin(diffs))
        dt = float(diffs[idx])
        suffix = self._status_right.text().split("|")[-1].strip()
        self._status_right.setText(
            f"file {idx + 1}: {dt:.1f} s ({dt / 60:.1f} min)  |  {suffix}"
        )

    # ── Lazy LFP seek ─────────────────────────────────────────────────────── #

    def _load_trace_segment(self, i0: int, i1: int) -> "np.ndarray | None":
        """Read LFP samples i0..i1 from memory or via disk seek."""
        s = self._s
        if self._raw_lfp is not None:
            return self._raw_lfp[i0:i1, :]
        if not hasattr(self, "_lfp_path") or self._lfp_path is None:
            return None
        from neurobox.io import load_binary as _lb
        return _lb(
            str(self._lfp_path),
            channels   = s.channels,
            n_channels = getattr(self, "_lfp_n_channels", max(s.channels) + 1),
            dtype      = "int16",
            periods    = np.array([[i0, i1]], dtype=np.int64),
        )

    # ── Period table dock ─────────────────────────────────────────────────── #

    def _build_period_dock(self) -> None:
        """Dockable table showing all annotated periods with click-to-navigate."""
        dock = QDockWidget("Periods", self)
        dock.setObjectName("period_dock")
        dock.setAllowedAreas(
            Qt.DockWidgetArea.RightDockWidgetArea |
            Qt.DockWidgetArea.BottomDockWidgetArea
        )
        self._period_table = QTableWidget(0, 4)
        self._period_table.setHorizontalHeaderLabels(
            ["Start (s)", "End (s)", "Duration (s)", "Start (samp)"]
        )
        self._period_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self._period_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self._period_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self._period_table.cellDoubleClicked.connect(self._on_period_table_click)
        dock.setWidget(self._period_table)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
        self._period_dock = dock

    def _refresh_period_table(self) -> None:
        """Sync the period table with the current periods array."""
        if not hasattr(self, "_period_table"):
            return
        s = self._s
        tbl = self._period_table
        tbl.setRowCount(0)
        if s.periods is None:
            return
        for row_idx, (t0, t1) in enumerate(s.periods):
            if np.isnan(t0) and np.isnan(t1):
                continue
            tbl.insertRow(tbl.rowCount())
            r = tbl.rowCount() - 1
            t0s = f"{t0:.3f}" if not np.isnan(t0) else "—"
            t1s = f"{t1:.3f}" if not np.isnan(t1) else "—"
            dur = f"{t1 - t0:.3f}" if not (np.isnan(t0) or np.isnan(t1)) else "—"
            samp = str(int(t0 * s.lfp_sr)) if not np.isnan(t0) else "—"
            for col, val in enumerate([t0s, t1s, dur, samp]):
                item = QTableWidgetItem(val)
                item.setData(Qt.ItemDataRole.UserRole, row_idx)
                tbl.setItem(r, col, item)

    def _on_period_table_click(self, row: int, col: int) -> None:
        """Navigate to the period when user double-clicks a table row."""
        item = self._period_table.item(row, 0)
        if item is None:
            return
        try:
            t = float(item.text())
        except ValueError:
            return
        s = self._s
        t0, t1 = s.x_range
        w = t1 - t0
        s.current_t = t
        t_min = float(s.spec_times[0]) if s.spec_times is not None else 0
        t_max = float(s.spec_times[-1]) if s.spec_times is not None else 1e6
        s.x_range = (max(t_min, t - 3), min(t_max, t - 3 + w))
        for ax in self._ax_spec + self._ax_aux:
            ax.set_xlim(s.x_range)
        self._refresh()

    # ── Undo helpers ──────────────────────────────────────────────────────── #

    def _push_undo(self, description: str, before: "np.ndarray") -> None:
        """Snapshot current periods onto the undo stack."""
        cmd = _PeriodUndoCommand(self, description, before, self._s.periods.copy())
        self._undo_stack.push(cmd)

    # ── eegseg.par channel preference file ────────────────────────────────── #

    def _load_channel_prefs(self) -> "list[int] | None":
        """Read 0-based channel indices from <filebase>.eegseg.par."""
        from pathlib import Path as _P
        path = _P(self._s.file_base + ".eegseg.par")
        if not path.exists():
            return None
        return list(np.loadtxt(str(path), dtype=int).ravel())

    def _save_channel_prefs(self, channels: list) -> None:
        """Write channel list to <filebase>.eegseg.par."""
        from pathlib import Path as _P
        np.savetxt(
            str(_P(self._s.file_base + ".eegseg.par")),
            np.array(channels, dtype=int), fmt="%d"
        )

    # ── Launch factory ────────────────────────────────────────────────────── #

    @classmethod
    def launch(
        cls,
        session     = None,
        file_base:  Optional[str] = None,
        state_name: str           = "",
        freq_range: tuple         = (1.0, 140.0),
        channels:   Optional[list] = None,
        window_sec: float          = 1.0,
        aux_data:   Optional[list] = None,
        spec_params = None,
        app:        Optional[QApplication] = None,
    ) -> "CheckEegStatesWindow":
        """High-level entry point.

        Accepts either a :class:`~neurobox.dtype.session.NBSession` or a
        raw *file_base* path.  Computes the spectrogram if not cached,
        then launches the GUI.

        Parameters
        ----------
        session:
            Loaded NBSession.  If provided, LFP, par, and channels are
            extracted automatically.
        file_base:
            Path stem (e.g. ``/data/project/B01/jg05-20120316/jg05-20120316``).
            Required when *session* is None.
        state_name:
            Label of the brain state to annotate (e.g. ``'theta'``).
        freq_range:
            ``(f_low, f_high)`` Hz displayed in the spectrograms.
        channels:
            0-based LFP channel indices.  Defaults to ``[0]``.
        window_sec:
            Width of the LFP trace window in seconds.
        aux_data:
            Extra panels: list of ``(x, y, data, 'plot'|'imagesc')`` tuples.
        spec_params:
            :class:`~neurobox.analysis.lfp.spectral.SpectralParams` instance.
            Defaults to ``SpectralParams.for_lfp(lfp_sr)``.
        app:
            Existing QApplication.  Created if None.
        """
        from neurobox.analysis.lfp.spectral import SpectralParams, multitaper_spectrogram
        from neurobox.io import load_par, load_binary

        owns_app = app is None
        if owns_app:
            app = QApplication.instance() or QApplication(sys.argv)

        # ── Extract data from session or file ─────────────────────────────── #
        if session is not None:
            fb      = str(session.spath / session.name)
            par     = session.par
            lfp_sr  = float(par.fieldPotentials.lfpSamplingRate) if par else 1250.0
            n_ch    = int(par.acquisitionSystem.nChannels) if par else 64
            chans   = channels or [0]
        else:
            if file_base is None:
                raise ValueError("Either 'session' or 'file_base' must be provided.")
            fb      = file_base
            par     = load_par(fb)
            lfp_sr  = float(getattr(getattr(par, "fieldPotentials", None), "lfpSamplingRate", 1250))
            n_ch    = int(getattr(getattr(par, "acquisitionSystem", None), "nChannels", 64))
            chans   = channels or [0]
            session = None

        # ── Load LFP ─────────────────────────────────────────────────────── #
        lfp_file = Path(fb + ".lfp")
        if not lfp_file.exists():
            lfp_file = Path(fb + ".eeg")

        raw_lfp = None
        n_samp  = 0
        if lfp_file.exists():
            print(f"Loading LFP from {lfp_file.name} ...")
            raw_lfp = load_binary(
                str(lfp_file), channels=chans, n_channels=n_ch, dtype="int16"
            )               # (T, len(chans))
            n_samp = raw_lfp.shape[0]

        # ── Compute or load spectrogram ───────────────────────────────────── #
        spec_file = Path(fb + ".eegseg.npz")
        if spec_file.exists():
            print(f"Loading cached spectrogram from {spec_file.name} ...")
            cached = np.load(spec_file)
            spec   = cached["y"]         # (F, T, C)
            freqs  = cached["f"]
            times  = cached["t"]
        elif raw_lfp is not None:
            if spec_params is None:
                spec_params = SpectralParams.for_lfp(lfp_sr)
                spec_params.freq_range = freq_range

            # ── Estimate total windows for the progress dialog ────────── #
            n_total = max(
                1, (raw_lfp.shape[0] - spec_params.win_len) // spec_params.step + 1
            )

            # ── Progress dialog ───────────────────────────────────────── #
            progress_dlg = QProgressDialog(
                "Computing spectrogram…", "Cancel",
                0, n_total,
            )
            progress_dlg.setWindowTitle("neurobox — CheckEegStates")
            progress_dlg.setWindowModality(Qt.WindowModality.WindowModal)
            progress_dlg.setMinimumDuration(500)   # only show if >500 ms
            progress_dlg.setValue(0)

            # ── Worker + thread ───────────────────────────────────────── #
            spec = freqs = times = None
            worker = _SpectrogramWorker(
                raw_lfp    = raw_lfp,
                spec_params = spec_params,
                lfp_sr     = lfp_sr,
                spec_file  = spec_file,
                whiten     = True,
            )
            thread = QThread()
            worker.moveToThread(thread)

            result_holder: dict = {}

            def _on_progress(n_done: int, n_total_: int) -> None:
                progress_dlg.setValue(n_done)
                if progress_dlg.wasCanceled():
                    worker.cancel()

            def _on_finished(result: dict) -> None:
                result_holder.update(result)
                thread.quit()

            def _on_error(msg: str) -> None:
                if msg != "Cancelled":
                    QMessageBox.critical(
                        None, "Spectrogram error",
                        f"Failed to compute spectrogram:\n{msg}"
                    )
                thread.quit()

            worker.progress.connect(_on_progress)
            worker.finished.connect(_on_finished)
            worker.error.connect(_on_error)
            thread.started.connect(worker.run)

            thread.start()

            # Spin the event loop until the thread finishes
            # (allows the progress dialog and Cancel button to work)
            while thread.isRunning():
                app.processEvents()

            progress_dlg.close()
            thread.wait()

            if result_holder:
                spec  = result_holder["spec"]
                freqs = result_holder["freqs"]
                times = result_holder["times"]
                print(f"Saved spectrogram → {spec_file.name}")
            else:
                spec = freqs = times = None

        else:
            spec = freqs = times = None

        # ── Load existing periods ─────────────────────────────────────────── #
        periods = None
        sts_file = Path(fb + f".sts.{state_name}") if state_name else None
        if sts_file and sts_file.exists():
            raw = np.loadtxt(str(sts_file), ndmin=2).astype(np.int64)
            periods = raw / lfp_sr

        # ── Create window ─────────────────────────────────────────────────── #
        win = cls(
            file_base   = fb,
            spec        = spec,
            spec_freqs  = freqs,
            spec_times  = times,
            lfp_sr      = lfp_sr,
            n_channels  = len(chans),
            n_samples   = n_samp,
            channels    = chans,
            raw_lfp     = raw_lfp,
            state_name  = state_name,
            periods     = periods,
            freq_range  = freq_range,
            window_sec  = window_sec,
            aux_data    = aux_data,
        )
        win._lfp_path = lfp_file if lfp_file.exists() else None
        win._lfp_n_channels = n_ch
        win.show()

        if owns_app:
            app.exec()

        return win




# ─────────────────────────────────────────────────────────────────────────── #
# Background spectrogram worker                                               #
# ─────────────────────────────────────────────────────────────────────────── #

class _SpectrogramWorker(QObject):
    """Computes the multi-taper spectrogram on a background QThread.

    Mirrors the 'long computation' pattern from CheckEegStates_aux.m —
    in MATLAB this froze the UI; here it runs on a worker thread with
    live progress updates.

    Signals
    -------
    progress(int, int)    — (n_windows_done, n_windows_total)
    finished(result)      — emits the completed SpectrumResult
    error(str)            — emits an error message on failure
    """

    progress = Signal(int, int)
    finished = Signal(object)
    error    = Signal(str)

    def __init__(
        self,
        raw_lfp:      "np.ndarray",     # (T, C) int16 or float
        spec_params:  "SpectralParams",
        lfp_sr:       float,
        spec_file:    "Path | None",    # where to cache the result
        whiten:       bool = True,
        parent:       "QObject | None" = None,
    ) -> None:
        super().__init__(parent)
        self._raw_lfp   = raw_lfp
        self._params    = spec_params
        self._lfp_sr    = lfp_sr
        self._spec_file = spec_file
        self._whiten    = whiten
        self._cancelled = False

    def cancel(self) -> None:
        """Request early termination (checked between blocks)."""
        self._cancelled = True

    @Slot()
    def run(self) -> None:
        """Entry point — called by QThread.started signal."""
        try:
            from neurobox.analysis.lfp.spectral import (
                multitaper_spectrogram, whiten_ar
            )
            data = self._raw_lfp.astype(np.float64)

            # Whitening (fast — runs synchronously)
            if self._whiten:
                data, _ = whiten_ar(data, ar_order=2, samplerate=self._lfp_sr)

            def _on_progress(n_done: int, n_total: int) -> None:
                if self._cancelled:
                    raise InterruptedError("Cancelled by user")
                self.progress.emit(n_done, n_total)

            result = multitaper_spectrogram(
                data, self._params, progress=_on_progress
            )

            # (T_win, F, C) → (F, T_win, C)
            spec  = np.moveaxis(result.power, 0, 1)
            freqs = result.freqs
            times = result.times

            if self._spec_file is not None and not self._cancelled:
                np.savez(self._spec_file, y=spec, f=freqs, t=times)

            self.finished.emit({"spec": spec, "freqs": freqs, "times": times})

        except InterruptedError:
            self.error.emit("Cancelled")
        except Exception as exc:
            import traceback
            self.error.emit(traceback.format_exc())



# ─────────────────────────────────────────────────────────────────────────── #
# Undo command                                                                 #
# ─────────────────────────────────────────────────────────────────────────── #

class _PeriodUndoCommand(QUndoCommand):
    """Reversible operation on the periods array.

    Stores a before/after snapshot; undo/redo swaps them and redraws.
    """

    def __init__(
        self,
        window,
        description: str,
        before:  "np.ndarray",
        after:   "np.ndarray",
    ) -> None:
        super().__init__(description)
        self._window = window
        self._before = before.copy()
        self._after  = after.copy()

    def undo(self) -> None:
        self._window._s.periods = self._before.copy()
        self._window._refresh()
        self._window._refresh_period_table()

    def redo(self) -> None:
        self._window._s.periods = self._after.copy()
        self._window._refresh()
        self._window._refresh_period_table()



# ─────────────────────────────────────────────────────────────────────────── #
# Auto-segmentation dialog                                                     #
# ─────────────────────────────────────────────────────────────────────────── #

class _AutoSegDialog(QDialog):
    """Parameter dialog for the HMM auto-segmentation tool."""

    def __init__(self, parent, t_range, n_channels):
        super().__init__(parent)
        self.setWindowTitle("Auto-segment — HMM parameters")

        form = QFormLayout(self)

        self._frin  = QLineEdit("5 12")
        self._frout = QLineEdit("1 5 12 15")
        self._ch    = QLineEdit("0")
        self._t0    = QLineEdit(f"{t_range[0]:.2f}")
        self._t1    = QLineEdit(f"{t_range[1]:.2f}")

        form.addRow("Pass-band (Hz, start end):", self._frin)
        form.addRow("Stop-band (Hz, pairs … ):", self._frout)
        form.addRow(f"Channel index (0–{n_channels - 1}):", self._ch)
        form.addRow("Region start (s):", self._t0)
        form.addRow("Region end (s):", self._t1)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        form.addRow(btns)

    def result(self):
        frin  = list(map(float, self._frin.text().split()))
        frout_flat = list(map(float, self._frout.text().split()))
        frout = [(frout_flat[i], frout_flat[i + 1]) for i in range(0, len(frout_flat), 2)]
        ch    = int(self._ch.text())
        t0    = float(self._t0.text())
        t1    = float(self._t1.text())
        return frin, frout, ch, t0, t1


# ─────────────────────────────────────────────────────────────────────────── #
# Utility                                                                      #
# ─────────────────────────────────────────────────────────────────────────── #

def _threshold_cross_to_periods(
    mask: np.ndarray,
    t:    np.ndarray,
    dt:   float,
    min_dur: float = 2.0,
) -> np.ndarray:
    """Convert a boolean mask to (N, 2) period array in the same units as *t*.

    Gaps shorter than ``dt`` samples are bridged; periods shorter than
    *min_dur* seconds (in units of *t*) are dropped.
    """
    # Find rising / falling edges
    padded  = np.concatenate([[False], mask.astype(bool), [False]])
    rising  = np.where(~padded[:-1] & padded[1:])[0]
    falling = np.where(padded[:-1] & ~padded[1:])[0]

    if len(rising) == 0:
        return np.zeros((0, 2))

    periods = np.column_stack([t[rising], t[np.minimum(falling, len(t) - 1)]])
    durations = periods[:, 1] - periods[:, 0]
    return periods[durations >= min_dur]


# ─────────────────────────────────────────────────────────────────────────── #
# Command-line entry point                                                     #
# ─────────────────────────────────────────────────────────────────────────── #

def _cli_main() -> None:
    import argparse

    p = argparse.ArgumentParser(
        description="Interactive LFP spectrogram browser (CheckEegStates port)"
    )
    p.add_argument("file_base", help="Session file stem (no extension)")
    p.add_argument("--state",    default="",    dest="state_name")
    p.add_argument("--channels", nargs="+", type=int, default=[0])
    p.add_argument("--freq",     nargs=2, type=float, default=[1.0, 140.0],
                   metavar=("F_LO", "F_HI"))
    p.add_argument("--window",   type=float, default=1.0, metavar="SEC")
    args = p.parse_args()

    app = QApplication.instance() or QApplication(sys.argv)
    CheckEegStatesWindow.launch(
        file_base   = args.file_base,
        state_name  = args.state_name,
        freq_range  = tuple(args.freq),
        channels    = args.channels,
        window_sec  = args.window,
        app         = app,
    )


if __name__ == "__main__":
    _cli_main()
