"""
neurobox.gui.mta_browser.main_window
======================================
Top-level :class:`MTABrowserWindow` (a :class:`QMainWindow`).

Layout
------
The MATLAB ``MTABrowser.fig`` had a TabBar-like row of "browser
state" buttons across the top (DataManagement / Setup / MotionLabel /
LFP) that swapped a big content panel.  This Python port keeps the
same UX with a :class:`QTabWidget`.

The Motion-Labelling tab uses a :class:`QSplitter` that holds the
3-D viewer on the left, the state editor on the right, and the
timeline + state-track at the bottom.

The application does NOT load a session by default; the Data
Management tab lets you browse a project root and click a row to
open one.  Alternatively pass ``session=`` to
:meth:`MTABrowserWindow.launch` for direct attach.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui  import QAction, QKeyEvent
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QFileDialog, QHBoxLayout, QLabel,
    QListWidget, QListWidgetItem, QMainWindow, QMessageBox,
    QPushButton, QSlider, QSplitter, QStatusBar, QTabWidget,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from neurobox.dtype.session  import NBSession
from neurobox.dtype.stc      import NBStateCollection

from .data_layer import ProjectIndex, scan_project
from .lfp_widgets import LfpTraceView, SpectrogramView
from .model      import PlaybackModel
from .widgets    import FeaturePanel, SkeletonViewer3D, StateTrackView


__all__ = [
    "MTABrowserWindow",
]


# ─────────────────────────────────────────────────────────────────────── #
# Data-Management tab                                                        #
# ─────────────────────────────────────────────────────────────────────── #

class _DataManagementTab(QWidget):
    """Subject / date / maze / trial pickers + load button.

    Emits ``session_loaded`` with the resulting :class:`NBSession` once
    the user clicks "Load Session".
    """

    session_loaded = Signal(object)         # NBSession

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._index: Optional[ProjectIndex] = None
        self._build_ui()

    def _build_ui(self) -> None:
        self.choose_root_btn = QPushButton("Choose project root…")
        self.choose_root_btn.clicked.connect(self._choose_root)
        self.root_label = QLabel("(no project loaded)")
        self.root_label.setStyleSheet("color: gray;")

        self.subject_list = QListWidget()
        self.date_list    = QListWidget()
        self.maze_list    = QListWidget()
        self.trial_list   = QListWidget()
        for w in (self.subject_list, self.date_list,
                  self.maze_list, self.trial_list):
            w.setSelectionMode(QAbstractItemView.SingleSelection)

        self.subject_list.currentItemChanged.connect(self._on_subject)
        self.date_list   .currentItemChanged.connect(self._on_date)
        self.maze_list   .currentItemChanged.connect(self._on_maze)

        self.load_btn = QPushButton("Load Session")
        self.load_btn.setEnabled(False)
        self.load_btn.clicked.connect(self._on_load)

        # Layout: [root row] [4 columns of pickers] [Load]
        layout = QVBoxLayout(self)

        root_row = QHBoxLayout()
        root_row.addWidget(self.choose_root_btn)
        root_row.addWidget(self.root_label, stretch=1)
        layout.addLayout(root_row)

        cols = QHBoxLayout()
        for label, w in [("Subject", self.subject_list),
                          ("Date",    self.date_list),
                          ("Maze",    self.maze_list),
                          ("Trial",   self.trial_list)]:
            colw = QWidget()
            colv = QVBoxLayout(colw)
            colv.setContentsMargins(0, 0, 0, 0)
            colv.addWidget(QLabel(label))
            colv.addWidget(w)
            cols.addWidget(colw)
        layout.addLayout(cols)

        layout.addWidget(self.load_btn)

    # ── Slots ─────────────────────────────────────────────────────── #

    def _choose_root(self) -> None:
        d = QFileDialog.getExistingDirectory(
            self, "Choose project root directory",
        )
        if d:
            self.set_root(Path(d))

    def set_root(self, root: Path) -> None:
        """Programmatic API for tests: load *root* without a dialog."""
        self._index = scan_project(root)
        self.root_label.setText(str(root))
        self.subject_list.clear()
        for s in self._index.subjects():
            self.subject_list.addItem(s)
        if self.subject_list.count() > 0:
            self.subject_list.setCurrentRow(0)

    def _on_subject(self, current, _previous) -> None:
        self.date_list.clear()
        if not current or not self._index:
            return
        subj = current.text()
        for d in self._index.dates_for(subj):
            self.date_list.addItem(d)
        if self.date_list.count() > 0:
            self.date_list.setCurrentRow(0)

    def _on_date(self, current, _previous) -> None:
        self.maze_list.clear()
        if not current or not self._index:
            return
        subj_item = self.subject_list.currentItem()
        if not subj_item:
            return
        for m in self._index.mazes_for(subj_item.text(), current.text()):
            self.maze_list.addItem(m)
        if self.maze_list.count() > 0:
            self.maze_list.setCurrentRow(0)

    def _on_maze(self, current, _previous) -> None:
        self.trial_list.clear()
        self.load_btn.setEnabled(False)
        if not current or not self._index:
            return
        subj_item = self.subject_list.currentItem()
        date_item = self.date_list.currentItem()
        if not (subj_item and date_item):
            return
        for t in self._index.trials_for(
            subj_item.text(), date_item.text(), current.text()
        ):
            self.trial_list.addItem(t)
        if self.trial_list.count() > 0:
            self.trial_list.setCurrentRow(0)
            self.load_btn.setEnabled(True)

    def _on_load(self) -> None:
        sub  = self.subject_list.currentItem().text()
        date = self.date_list.currentItem().text()
        maze = self.maze_list.currentItem().text()
        trial = self.trial_list.currentItem().text() if \
                self.trial_list.currentItem() else "all"
        # Session name is `subject-date`
        session_name = f"{sub}-{date}"
        try:
            session = NBSession(
                session_name = session_name,
                maze         = maze,
                trial        = trial,
            )
            session.load("xyz")
            try:
                session.load("stc")
            except Exception:
                # State collection is optional — start fresh
                session.stc = NBStateCollection(mode="manual")
        except Exception as e:                     # pragma: no cover
            QMessageBox.critical(
                self, "Load failed",
                f"Could not load {session_name} ({maze}/{trial}):\n{e}",
            )
            return
        self.session_loaded.emit(session)


# ─────────────────────────────────────────────────────────────────────── #
# State-editor sub-panel (used inside the Motion-Labelling tab)              #
# ─────────────────────────────────────────────────────────────────────── #

class _StateEditorTable(QWidget):
    """Table of states with add/remove buttons and visibility toggle."""

    def __init__(self, model: PlaybackModel,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.model = model
        self._build_ui()
        model.subscribe(self._on_event)
        self._reload()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(
            ["State", "Key", "Visible", "Frames"]
        )
        self.table.itemChanged.connect(self._on_item_changed)
        self.table.itemSelectionChanged.connect(self._on_selection_changed)
        layout.addWidget(self.table)

        button_row = QHBoxLayout()
        self.add_btn    = QPushButton("Add")
        self.remove_btn = QPushButton("Remove")
        self.save_btn   = QPushButton("Save STC…")
        self.add_btn.clicked.connect(self._on_add_clicked)
        self.remove_btn.clicked.connect(self._on_remove_clicked)
        self.save_btn.clicked.connect(self._on_save_clicked)
        button_row.addWidget(self.add_btn)
        button_row.addWidget(self.remove_btn)
        button_row.addWidget(self.save_btn)
        button_row.addStretch(1)
        layout.addLayout(button_row)

        self._suspend_signals = False

    # ── Slots / model events ──────────────────────────────────────── #

    def _on_event(self, event: str) -> None:
        if event in ("states", "states_data", "current_label",
                       "selected_states", "state_keys"):
            self._reload()

    def _reload(self) -> None:
        self._suspend_signals = True
        self.table.setRowCount(0)
        for label, mask in self.model.states_data.items():
            row = self.table.rowCount()
            self.table.insertRow(row)
            self.table.setItem(row, 0, QTableWidgetItem(label))
            self.table.setItem(row, 1, QTableWidgetItem(
                self.model.state_keys.get(label, "")
            ))
            visible_item = QTableWidgetItem()
            visible_item.setFlags(visible_item.flags()
                                    | Qt.ItemIsUserCheckable)
            visible_item.setCheckState(
                Qt.Checked if self.model.selected_states.get(label, True)
                else Qt.Unchecked
            )
            self.table.setItem(row, 2, visible_item)
            n_active = int((mask > 0).sum())
            n_item = QTableWidgetItem(str(n_active))
            n_item.setFlags(n_item.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(row, 3, n_item)
            if label == self.model.current_label:
                self.table.selectRow(row)
        self._suspend_signals = False

    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        if self._suspend_signals:
            return
        row, col = item.row(), item.column()
        labels = list(self.model.states_data.keys())
        if row >= len(labels):
            return
        label = labels[row]
        try:
            if col == 0:
                new_label = item.text().strip()
                if new_label and new_label != label:
                    self.model.rename_state(label, new_label)
            elif col == 1:
                new_key = item.text().strip()
                if new_key and new_key != self.model.state_keys.get(label):
                    self.model.set_state_key(label, new_key)
            elif col == 2:
                self.model.set_state_selected(
                    label, item.checkState() == Qt.Checked
                )
        except (ValueError, KeyError) as e:
            QMessageBox.warning(self, "Edit rejected", str(e))
            self._reload()                # roll back

    def _on_selection_changed(self) -> None:
        if self._suspend_signals:
            return
        row = self.table.currentRow()
        if row < 0:
            return
        labels = list(self.model.states_data.keys())
        if 0 <= row < len(labels):
            self.model.set_current_label(labels[row])

    def _on_add_clicked(self) -> None:
        # Find a free key
        used = set(self.model.state_keys.values())
        for ch in "abcdefghijklmnopqrstuvwxyz0123456789":
            if ch not in used:
                key = ch; break
        else:
            QMessageBox.warning(self, "No free key",
                                  "All single-character keys are in use.")
            return
        # Find a unique label
        i = 1
        while f"state{i}" in self.model.states_data:
            i += 1
        try:
            self.model.add_state(f"state{i}", key)
        except ValueError as e:                    # pragma: no cover
            QMessageBox.warning(self, "Add failed", str(e))

    def _on_remove_clicked(self) -> None:
        row = self.table.currentRow()
        if row < 0:
            return
        labels = list(self.model.states_data.keys())
        if 0 <= row < len(labels):
            self.model.remove_state(labels[row])

    def _on_save_clicked(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save state collection",
            f"{self.model.session.filebase}.stc.pkl",
            "Pickle (*.pkl);;All files (*)",
        )
        if not path:
            return
        self.model.update_session_stc(mode="manual")
        try:
            import pickle
            with open(path, "wb") as f:
                pickle.dump(self.model.session.stc, f)
            QMessageBox.information(self, "Saved",
                                      f"Saved {len(self.model.states_data)} "
                                      f"states to {path}")
        except Exception as e:                     # pragma: no cover
            QMessageBox.critical(self, "Save failed", str(e))


# ─────────────────────────────────────────────────────────────────────── #
# Motion-Labelling tab                                                       #
# ─────────────────────────────────────────────────────────────────────── #

class _MotionLabellingTab(QWidget):
    """3-D viewer + state editor + state-track + timeline + key handling."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.model: Optional[PlaybackModel] = None
        self._timer = QTimer(self)
        self._timer.setInterval(33)            # ~30 fps
        self._timer.timeout.connect(self._on_tick)
        self._direction = 0                    # +1 = forward, -1 = back, 0 stop
        self._build_ui()

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(2, 2, 2, 2)

        # Splitter: left = 3D viewer, right = state editor
        self._top_split = QSplitter(Qt.Horizontal)
        self._viewer_placeholder = QLabel(
            "Load a session to start labelling.\n"
            "Open the Data Management tab → choose project / subject /\n"
            "date / maze / trial → Load Session.",
        )
        self._viewer_placeholder.setAlignment(Qt.AlignCenter)
        self._viewer_placeholder.setStyleSheet("color: gray;")
        self._top_split.addWidget(self._viewer_placeholder)
        self._editor_placeholder = QLabel("(state editor)")
        self._editor_placeholder.setAlignment(Qt.AlignCenter)
        self._editor_placeholder.setStyleSheet("color: gray;")
        self._top_split.addWidget(self._editor_placeholder)
        self._top_split.setStretchFactor(0, 3)
        self._top_split.setStretchFactor(1, 1)

        outer.addWidget(self._top_split, stretch=3)

        # Bottom: state-track + timeline
        self._track_placeholder = QLabel(
            "(state track will appear after a session is loaded)",
            self,
        )
        self._track_placeholder.setAlignment(Qt.AlignCenter)
        self._track_placeholder.setStyleSheet("color: gray;")
        self._track_placeholder.setMinimumHeight(120)
        outer.addWidget(self._track_placeholder, stretch=1)

        # Timeline controls
        controls = QHBoxLayout()
        self.play_btn  = QPushButton("▶")
        self.stop_btn  = QPushButton("⏸")
        self.back_btn  = QPushButton("◀")
        self.scrubber  = QSlider(Qt.Horizontal)
        self.idx_label = QLabel("frame 0 / 0")
        self.scrubber.setMinimum(0)
        self.scrubber.setMaximum(0)
        self.scrubber.valueChanged.connect(self._on_scrub)
        self.play_btn.clicked.connect(lambda: self._set_direction(+1))
        self.back_btn.clicked.connect(lambda: self._set_direction(-1))
        self.stop_btn.clicked.connect(lambda: self._set_direction(0))
        for w in (self.back_btn, self.play_btn, self.stop_btn):
            w.setMaximumWidth(40)
        controls.addWidget(self.back_btn)
        controls.addWidget(self.play_btn)
        controls.addWidget(self.stop_btn)
        controls.addWidget(self.scrubber, stretch=1)
        controls.addWidget(self.idx_label)
        outer.addLayout(controls)

    # ── Wiring after model becomes available ─────────────────────── #

    def attach_model(self, model: PlaybackModel) -> None:
        """Replace the placeholders with real widgets bound to *model*."""
        # Tear down old children if any
        old = self._top_split.widget(0)
        if old is not None:
            old.deleteLater()
        old = self._top_split.widget(1)
        if old is not None:
            old.deleteLater()

        self.model = model
        viewer = SkeletonViewer3D(model)
        editor = _StateEditorTable(model)
        self._top_split.insertWidget(0, viewer)
        self._top_split.insertWidget(1, editor)
        self._top_split.setStretchFactor(0, 3)
        self._top_split.setStretchFactor(1, 1)
        self._viewer = viewer
        self._editor = editor

        # Replace track placeholder
        old_track = self.layout().itemAt(1).widget()
        if old_track is not None:
            old_track.deleteLater()
        track = StateTrackView(model)
        self.layout().insertWidget(1, track, stretch=1)
        self._track = track

        self.scrubber.setMaximum(max(0, model.n_samples - 1))
        self.scrubber.setValue(0)
        model.subscribe(self._on_model_event)
        self._on_model_event("idx")

    def _on_model_event(self, event: str) -> None:
        if event == "idx" and self.model is not None:
            blocked = self.scrubber.blockSignals(True)
            self.scrubber.setValue(self.model.idx)
            self.scrubber.blockSignals(blocked)
            self.idx_label.setText(
                f"frame {self.model.idx} / {self.model.n_samples}"
            )

    def _on_scrub(self, value: int) -> None:
        if self.model is not None:
            self.model.set_idx(int(value))

    # ── Playback timer ─────────────────────────────────────────────── #

    def _set_direction(self, d: int) -> None:
        self._direction = int(d)
        if self.model is None:
            return
        if d == 0:
            self._timer.stop()
            self.model.set_paused(True)
        else:
            self._timer.start()
            self.model.set_paused(False)

    def _on_tick(self) -> None:
        if self.model is None or self._direction == 0:
            return
        speed = max(1, self.model.play_speed)
        self.model.step(self._direction * speed)
        if self.model.idx == 0 and self._direction < 0:
            self._set_direction(0)
        if self.model.idx >= self.model.n_samples - 1 and \
                self._direction > 0:
            self._set_direction(0)

    # ── Key events ─────────────────────────────────────────────────── #

    def keyPressEvent(self, event: QKeyEvent) -> None:                # noqa: N802
        if self.model is None:
            super().keyPressEvent(event); return
        key = event.key()
        text = event.text()
        mods = event.modifiers()

        # Ctrl + ←/→ jump
        if mods & Qt.ControlModifier and key == Qt.Key_Right:
            self.model.step(+50); return
        if mods & Qt.ControlModifier and key == Qt.Key_Left:
            self.model.step(-50); return

        if key == Qt.Key_Space:
            self._set_direction(0 if self._direction != 0 else +1)
            return
        if key == Qt.Key_Right:
            self._set_direction(+1); return
        if key == Qt.Key_Left:
            self._set_direction(-1); return
        if key == Qt.Key_Up:
            self.model.set_play_speed(self.model.play_speed + 1); return
        if key == Qt.Key_Down:
            self.model.set_play_speed(self.model.play_speed - 1); return

        # 1/2/3 — view from X / Y / Z axis
        if text in ("1", "2", "3") and hasattr(self, "_viewer"):
            ax = self._viewer._ax
            if text == "1":
                ax.view_init(elev=0, azim=0)
            elif text == "2":
                ax.view_init(elev=0, azim=90)
            else:
                ax.view_init(elev=90, azim=0)
            self._viewer._canvas.draw_idle()
            return

        if key in (Qt.Key_Delete, Qt.Key_Backspace):
            self.model.set_erase_mode(not self.model.erase_mode); return

        # State toggle by single character
        if text and len(text) == 1:
            for label, k in self.model.state_keys.items():
                if k == text:
                    if (self.model.current_label == label
                            and not self.model.erase_mode):
                        # Tap same key again → stop labelling
                        self.model.set_current_label(None)
                    else:
                        self.model.set_current_label(label)
                        self.model.set_erase_mode(False)
                    return
        super().keyPressEvent(event)


# ─────────────────────────────────────────────────────────────────────── #
# LFP-States tab                                                              #
# ─────────────────────────────────────────────────────────────────────── #

class _LfpStatesTab(QWidget):
    """LFP-states tab — scrolling spectrogram + LFP trace + state editor.

    Uses the SAME paint-while-scrubbing UX as the Motion-Labelling
    tab, but the playback model is anchored to whichever NB object
    drives this panel:

    * If a precomputed :class:`SpectrumResult` is attached, the model
      runs at the spectrogram's frame rate (typically 5-10 Hz, one
      frame per spectrogram window).
    * If only raw LFP is attached, the spectrogram is computed on
      attach and the model still runs at the resulting frame rate.

    State edits round-trip through ``session.stc``, so labels written
    here will be visible in the Motion-Labelling tab after a
    save / refresh, and vice versa.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.model: Optional[PlaybackModel] = None
        self._timer = QTimer(self)
        self._timer.setInterval(33)              # ~30 fps display rate
        self._timer.timeout.connect(self._on_tick)
        self._direction = 0
        self._spec_view:  Optional[SpectrogramView] = None
        self._trace_view: Optional[LfpTraceView]    = None
        self._editor:     Optional[_StateEditorTable] = None
        self._track:      Optional[StateTrackView]  = None
        self._build_ui()

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(2, 2, 2, 2)

        # Top: spectrogram on the left, state editor on the right
        self._top_split = QSplitter(Qt.Horizontal)
        self._spec_placeholder = QLabel(
            "(spectrogram appears once a session with LFP\n"
            "or precomputed spectrogram is loaded)",
        )
        self._spec_placeholder.setAlignment(Qt.AlignCenter)
        self._spec_placeholder.setStyleSheet("color: gray;")
        self._top_split.addWidget(self._spec_placeholder)

        self._editor_placeholder = QLabel("(state editor)")
        self._editor_placeholder.setAlignment(Qt.AlignCenter)
        self._editor_placeholder.setStyleSheet("color: gray;")
        self._top_split.addWidget(self._editor_placeholder)
        self._top_split.setStretchFactor(0, 3)
        self._top_split.setStretchFactor(1, 1)
        outer.addWidget(self._top_split, stretch=3)

        # Middle: raw LFP trace
        self._trace_placeholder = QLabel(
            "(raw LFP trace appears once attached)",
        )
        self._trace_placeholder.setAlignment(Qt.AlignCenter)
        self._trace_placeholder.setStyleSheet("color: gray;")
        self._trace_placeholder.setMinimumHeight(120)
        outer.addWidget(self._trace_placeholder, stretch=1)

        # Bottom: state-track + scrubber
        self._track_placeholder = QLabel(
            "(state track appears once attached)",
        )
        self._track_placeholder.setAlignment(Qt.AlignCenter)
        self._track_placeholder.setStyleSheet("color: gray;")
        self._track_placeholder.setMinimumHeight(100)
        outer.addWidget(self._track_placeholder, stretch=1)

        controls = QHBoxLayout()
        self.play_btn  = QPushButton("▶")
        self.stop_btn  = QPushButton("⏸")
        self.back_btn  = QPushButton("◀")
        self.scrubber  = QSlider(Qt.Horizontal)
        self.idx_label = QLabel("frame 0 / 0")
        self.scrubber.setMinimum(0); self.scrubber.setMaximum(0)
        self.scrubber.valueChanged.connect(self._on_scrub)
        self.play_btn.clicked.connect(lambda: self._set_direction(+1))
        self.back_btn.clicked.connect(lambda: self._set_direction(-1))
        self.stop_btn.clicked.connect(lambda: self._set_direction(0))
        for w in (self.back_btn, self.play_btn, self.stop_btn):
            w.setMaximumWidth(40)
        controls.addWidget(self.back_btn)
        controls.addWidget(self.play_btn)
        controls.addWidget(self.stop_btn)
        controls.addWidget(self.scrubber, stretch=1)
        controls.addWidget(self.idx_label)
        outer.addLayout(controls)

    # ── Attach API ───────────────────────────────────────────────── #

    def attach_lfp_data(
        self,
        session,
        *,
        spectrogram   = None,        # SpectrumResult | None
        lfp           = None,        # ndarray | None
        lfp_samplerate = None,       # float | None
        spec_kwargs   = None,        # dict | None
    ) -> None:
        """Attach session data and (re)build all child widgets.

        The model's frame rate is determined by the loaded NB object:

        * If *spectrogram* is given (a :class:`SpectrumResult`), the
          model runs at that spectrogram's frame rate (one model
          sample = one spectrogram window).
        * If only *lfp* is given, the spectrogram is computed
          synchronously on attach and the resulting frame rate
          drives the model.

        Parameters
        ----------
        session:
            The :class:`NBSession` whose ``stc`` is the canonical
            state collection.
        spectrogram:
            Pre-computed :class:`SpectrumResult` (recommended path).
        lfp:
            Raw signal — fallback when *spectrogram* is None.
        lfp_samplerate:
            Hz of *lfp*.  Required when *lfp* is provided.
        spec_kwargs:
            Forwarded to :class:`SpectralParams` when computing.
        """
        # Build the spectrogram view first; its frame_rate_hz drives
        # the model rate.
        if spectrogram is None and lfp is None:
            raise ValueError(
                "attach_lfp_data needs either `spectrogram=` or `lfp=`"
            )

        # Tear down old children
        for slot in (0, 1):
            old = self._top_split.widget(slot)
            if old is not None:
                old.deleteLater()
        old_trace = self.layout().itemAt(1).widget()
        if old_trace is not None:
            old_trace.deleteLater()
        old_track = self.layout().itemAt(2).widget()
        if old_track is not None:
            old_track.deleteLater()

        # Construct a temporary model with the right rate so the
        # SpectrogramView can subscribe to it.  We need to know the
        # frame rate BEFORE we can build the model.  Strategy: peek
        # at the spectrogram's times array directly.
        if spectrogram is not None:
            spec_obj = spectrogram
        else:
            # Compute synchronously to determine frame rate
            from neurobox.analysis.lfp.spectral import (
                SpectralParams, multitaper_spectrogram,
            )
            sk = dict(spec_kwargs or {})
            sk.setdefault("samplerate", float(lfp_samplerate))
            sk.setdefault("n_fft",      1024)
            sk.setdefault("win_len",    max(64, int(round(lfp_samplerate))))
            sk.setdefault("n_overlap",  max(0, int(round(lfp_samplerate * 0.8))))
            sk.setdefault("freq_range", (1.0, 100.0))
            params   = SpectralParams(**sk)
            spec_obj = multitaper_spectrogram(
                lfp.astype(np.float64) if lfp.ndim == 2 else lfp[:, None].astype(np.float64),
                params,
            )

        # Frame rate from the spectrogram's times array
        if spec_obj.times.size >= 2 and \
                np.isfinite(spec_obj.times[1] - spec_obj.times[0]) and \
                spec_obj.times[1] != spec_obj.times[0]:
            frame_rate = float(1.0 / (spec_obj.times[1] - spec_obj.times[0]))
        else:
            frame_rate = float(lfp_samplerate or 1.0)
        n_frames = int(spec_obj.power.shape[0]
                        if spec_obj.power.ndim > 2 else 1)

        self.model = PlaybackModel.from_data(
            session    = session,
            n_samples  = n_frames,
            samplerate = frame_rate,
        )

        # Build widgets
        self._spec_view = SpectrogramView(
            self.model, spectrogram=spec_obj, window_seconds=5.0,
        )
        self._editor = _StateEditorTable(self.model)
        self._top_split.insertWidget(0, self._spec_view)
        self._top_split.insertWidget(1, self._editor)
        self._top_split.setStretchFactor(0, 3)
        self._top_split.setStretchFactor(1, 1)

        # LFP trace (only when raw LFP is available)
        if lfp is not None:
            self._trace_view = LfpTraceView(
                self.model, lfp, float(lfp_samplerate),
                window_seconds=1.0,
            )
            self.layout().insertWidget(1, self._trace_view, stretch=1)
        else:
            placeholder = QLabel(
                "(raw LFP trace not available — only the precomputed "
                "spectrogram was provided)",
            )
            placeholder.setAlignment(Qt.AlignCenter)
            placeholder.setStyleSheet("color: gray;")
            placeholder.setMinimumHeight(80)
            self.layout().insertWidget(1, placeholder, stretch=1)

        self._track = StateTrackView(self.model)
        self.layout().insertWidget(2, self._track, stretch=1)

        # Scrubber range
        self.scrubber.setMaximum(max(0, n_frames - 1))
        self.scrubber.setValue(0)
        self.model.subscribe(self._on_model_event)
        self._on_model_event("idx")

    # ── Refresh from session.stc ──────────────────────────────────── #

    def refresh_from_session(self) -> None:
        """Rebuild the model's per-state masks from session.stc.

        Called when this tab becomes visible so it picks up state
        edits made in the Motion-Labelling tab.
        """
        if self.model is not None:
            self.model._populate_states_from_session()
            self.model._emit("states")
            self.model._emit("states_data")

    def commit_to_session(self) -> None:
        """Push this tab's state edits to session.stc so other tabs
        can see them on their next refresh."""
        if self.model is not None:
            self.model.update_session_stc(mode="manual")

    # ── Listener / scrubber wiring (mirrors _MotionLabellingTab) ─── #

    def _on_model_event(self, event: str) -> None:
        if event == "idx" and self.model is not None:
            blocked = self.scrubber.blockSignals(True)
            self.scrubber.setValue(self.model.idx)
            self.scrubber.blockSignals(blocked)
            seconds = self.model.idx / max(self.model.samplerate, 1e-9)
            self.idx_label.setText(
                f"frame {self.model.idx} / {self.model.n_samples}  "
                f"({seconds:.2f} s)"
            )

    def _on_scrub(self, value: int) -> None:
        if self.model is not None:
            self.model.set_idx(int(value))

    def _set_direction(self, d: int) -> None:
        self._direction = int(d)
        if self.model is None:
            return
        if d == 0:
            self._timer.stop()
            self.model.set_paused(True)
        else:
            self._timer.start()
            self.model.set_paused(False)

    def _on_tick(self) -> None:
        if self.model is None or self._direction == 0:
            return
        speed = max(1, self.model.play_speed)
        self.model.step(self._direction * speed)
        if self.model.idx == 0 and self._direction < 0:
            self._set_direction(0)
        if self.model.idx >= self.model.n_samples - 1 and \
                self._direction > 0:
            self._set_direction(0)

    # ── Key events (same as Motion-Labelling, minus the 1/2/3 view) ─ #

    def keyPressEvent(self, event: QKeyEvent) -> None:                # noqa: N802
        if self.model is None:
            super().keyPressEvent(event); return
        key  = event.key()
        text = event.text()
        mods = event.modifiers()

        if mods & Qt.ControlModifier and key == Qt.Key_Right:
            self.model.step(+50); return
        if mods & Qt.ControlModifier and key == Qt.Key_Left:
            self.model.step(-50); return
        if key == Qt.Key_Space:
            self._set_direction(0 if self._direction != 0 else +1); return
        if key == Qt.Key_Right:
            self._set_direction(+1); return
        if key == Qt.Key_Left:
            self._set_direction(-1); return
        if key == Qt.Key_Up:
            self.model.set_play_speed(self.model.play_speed + 1); return
        if key == Qt.Key_Down:
            self.model.set_play_speed(self.model.play_speed - 1); return
        if key in (Qt.Key_Delete, Qt.Key_Backspace):
            self.model.set_erase_mode(not self.model.erase_mode); return

        # State toggle by single character
        if text and len(text) == 1:
            for label, k in self.model.state_keys.items():
                if k == text:
                    if (self.model.current_label == label
                            and not self.model.erase_mode):
                        self.model.set_current_label(None)
                    else:
                        self.model.set_current_label(label)
                        self.model.set_erase_mode(False)
                    return
        super().keyPressEvent(event)


# ─────────────────────────────────────────────────────────────────────── #
# Top-level window                                                            #
# ─────────────────────────────────────────────────────────────────────── #

class MTABrowserWindow(QMainWindow):
    """Top-level browser window."""

    session_loaded = Signal(object)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("MTA Browser (neurobox)")
        self.resize(1200, 800)
        self._tabs = QTabWidget()
        self.setCentralWidget(self._tabs)

        self._dm_tab = _DataManagementTab()
        self._ml_tab = _MotionLabellingTab()
        self._lfp_tab = _LfpStatesTab()
        self._tabs.addTab(self._dm_tab, "Data Management")
        self._tabs.addTab(self._ml_tab, "Motion Labelling")
        self._tabs.addTab(self._lfp_tab, "LFP States")
        self._tabs.currentChanged.connect(self._on_tab_changed)

        self._dm_tab.session_loaded.connect(self._on_session_loaded)

        sb = QStatusBar()
        self.setStatusBar(sb)
        self._status = sb
        self._previous_tab_idx = self._tabs.currentIndex()

    def _on_session_loaded(self, session: NBSession) -> None:
        try:
            model = PlaybackModel.from_session(session)
        except Exception as e:                     # pragma: no cover
            QMessageBox.critical(self, "Setup failed",
                                  f"Could not build playback model:\n{e}")
            return
        self._ml_tab.attach_model(model)

        # Optionally attach LFP / spectrogram to the LFP-states tab.
        # Both are best-effort: we look at session.lfp and an
        # optional precomputed SpectrumResult on the session.
        attached_lfp = False
        try:
            spec = getattr(session, "spec", None)
            lfp_arr = lfp_sr = None
            if getattr(session, "lfp", None) is not None and \
                    session.lfp.data is not None:
                lfp_arr = session.lfp.data
                lfp_sr  = float(session.lfp.samplerate)
            if spec is not None or lfp_arr is not None:
                self._lfp_tab.attach_lfp_data(
                    session,
                    spectrogram    = spec,
                    lfp            = lfp_arr,
                    lfp_samplerate = lfp_sr,
                )
                attached_lfp = True
        except Exception as e:                     # pragma: no cover
            self._status.showMessage(
                f"LFP-states tab not attached: {e}", 5000,
            )

        self._tabs.setCurrentWidget(self._ml_tab)
        msg = (
            f"Loaded {session.name} ({session.maze}/{session.trial}) "
            f"— {model.n_samples} samples @ {model.samplerate} Hz"
        )
        if attached_lfp and self._lfp_tab.model is not None:
            msg += (f"; LFP-states @ "
                    f"{self._lfp_tab.model.samplerate:.2f} Hz")
        self._status.showMessage(msg)
        self.session_loaded.emit(session)

    def _on_tab_changed(self, new_idx: int) -> None:
        """When switching tabs, push the just-edited tab's state edits
        to session.stc, then refresh the newly-shown tab from it.

        This lets the user paint states in either tab and have the
        other tab pick them up on next view, without needing an
        explicit "save" step.
        """
        old_idx = getattr(self, "_previous_tab_idx", new_idx)
        if old_idx == new_idx:
            return

        # Commit edits from the tab we're leaving
        if old_idx == 1 and self._ml_tab.model is not None:
            self._ml_tab.model.update_session_stc(mode="manual")
        elif old_idx == 2 and self._lfp_tab.model is not None:
            self._lfp_tab.commit_to_session()

        # Refresh the tab we're entering
        if new_idx == 1 and self._ml_tab.model is not None:
            self._ml_tab.model._populate_states_from_session()
            self._ml_tab.model._emit("states")
            self._ml_tab.model._emit("states_data")
        elif new_idx == 2 and self._lfp_tab.model is not None:
            self._lfp_tab.refresh_from_session()

        self._previous_tab_idx = new_idx

    # ── Test / programmatic API ────────────────────────────────────── #

    def set_session(self, session: NBSession) -> None:
        """Attach an already-loaded NBSession (skip the picker)."""
        self._on_session_loaded(session)

    def attach_lfp(
        self,
        *,
        spectrogram    = None,
        lfp            = None,
        lfp_samplerate = None,
        spec_kwargs    = None,
    ) -> None:
        """Attach LFP / spectrogram data to the LFP-states tab.

        Useful when the session itself didn't carry a precomputed
        spectrogram or LFP, but the caller has it in memory.
        """
        if self._ml_tab.model is None:
            raise RuntimeError(
                "Load a session first (set_session) before attaching LFP."
            )
        self._lfp_tab.attach_lfp_data(
            self._ml_tab.model.session,
            spectrogram    = spectrogram,
            lfp            = lfp,
            lfp_samplerate = lfp_samplerate,
            spec_kwargs    = spec_kwargs,
        )

    @classmethod
    def launch(
        cls,
        *,
        session: Optional[NBSession] = None,
        project_root: Optional[Path] = None,
        run: bool = True,
    ) -> "MTABrowserWindow":
        """Create and show the window.

        Parameters
        ----------
        session:
            Optional pre-loaded session (skips the data-management tab).
        project_root:
            Optional starting project root for the data-management tab.
        run:
            If True (default), run the Qt event loop until the window
            closes.  Pass ``run=False`` from a unit test that wants
            programmatic control.
        """
        app = QApplication.instance() or QApplication(sys.argv)
        win = cls()
        if project_root is not None:
            win._dm_tab.set_root(Path(project_root))
        if session is not None:
            win.set_session(session)
        win.show()
        if run:                                    # pragma: no cover
            app.exec()
        return win
