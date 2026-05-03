"""
neurobox.gui.mta_browser.widgets
==================================
Reusable Qt widgets for the motion-labelling viewer.

Three classes:

* :class:`SkeletonViewer3D` — animated 3-D rat skeleton, redraws on
  ``model.idx`` change.
* :class:`StateTrackView` — horizontal "piano roll" of state masks
  with a moving centre cursor.
* :class:`FeaturePanel` — multi-row feature time-trace plot, also
  with a centre cursor synced to ``model.idx``.

All three subscribe to a :class:`PlaybackModel` and update on its
events.  None of them mutate the model directly; mutations come from
the main window's keyboard / button handlers.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QSizePolicy, QVBoxLayout, QWidget

from .model import PlaybackModel


__all__ = [
    "SkeletonViewer3D",
    "StateTrackView",
    "FeaturePanel",
]


# ─────────────────────────────────────────────────────────────────────── #
# 3-D skeleton viewer                                                       #
# ─────────────────────────────────────────────────────────────────────── #

class SkeletonViewer3D(QWidget):
    """Animated 3-D skeleton plot synced to a :class:`PlaybackModel`.

    Renders the marker positions at ``model.idx`` plus the model
    connections as sticks.  Re-renders on the ``idx`` event.

    Parameters
    ----------
    model:
        Source of truth for playback head and session.
    parent:
        Qt parent.
    boundary:
        Optional ``[[xlo, xhi], [ylo, yhi], [zlo, zhi]]`` to set
        fixed axes limits.  Default uses the session.maze.boundaries
        if available, else the data bounding box.
    """

    def __init__(
        self,
        model:    PlaybackModel,
        parent:   Optional[QWidget] = None,
        boundary: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__(parent)
        self.model = model
        self._fig    = Figure(figsize=(5, 5), tight_layout=True)
        self._canvas = FigureCanvasQTAgg(self._fig)
        self._ax     = self._fig.add_subplot(111, projection="3d")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._canvas)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Determine axis limits
        xyz = model.session.xyz
        if boundary is None:
            data = xyz.data.reshape(-1, 3)
            data = data[np.isfinite(data).all(axis=1)]
            if len(data) > 0:
                lo = data.min(axis=0); hi = data.max(axis=0)
            else:
                lo = np.array([-100, -100, 0])
                hi = np.array([100, 100, 200])
            boundary = np.column_stack([lo, hi])
        self._boundary = np.asarray(boundary)

        # Stickline / marker plot artists; rebuilt lazily
        self._stick_lines: list = []
        self._marker_dots = None

        # Style
        self._ax.set_xlabel("x")
        self._ax.set_ylabel("y")
        self._ax.set_zlabel("z")
        self._ax.set_xlim(*self._boundary[0])
        self._ax.set_ylim(*self._boundary[1])
        self._ax.set_zlim(*self._boundary[2])
        try:
            self._ax.set_box_aspect((1, 1, 1))
        except Exception:                         # pragma: no cover
            pass

        # Subscribe to model
        model.subscribe(self._on_event)
        self._refresh()

    # ── Listener ───────────────────────────────────────────────────── #

    def _on_event(self, event: str) -> None:
        if event in ("idx",):
            self._refresh()

    # ── Drawing ────────────────────────────────────────────────────── #

    def _refresh(self) -> None:
        xyz = self.model.session.xyz
        idx = self.model.idx
        if idx >= xyz.data.shape[0]:
            return
        positions = xyz.data[idx]               # (n_markers, 3)

        # Update marker dots
        if self._marker_dots is None:
            self._marker_dots = self._ax.scatter(
                positions[:, 0], positions[:, 1], positions[:, 2],
                s=40, c="k", depthshade=False,
            )
        else:
            # 3D scatters lack a clean 'set_offsets' that respects z;
            # use the internal _offsets3d API.
            self._marker_dots._offsets3d = (
                positions[:, 0], positions[:, 1], positions[:, 2],
            )

        # Update sticks
        connections = list(getattr(xyz.model, "connections", []) or [])

        # Build / re-build stick lines lazily
        if len(self._stick_lines) != len(connections):
            for ln in self._stick_lines:
                try:
                    ln.remove()
                except Exception:                 # pragma: no cover
                    pass
            self._stick_lines = []
            for _ in connections:
                ln, = self._ax.plot([], [], [], color="C0", lw=2)
                self._stick_lines.append(ln)

        for i, conn in enumerate(connections):
            try:
                a_name, b_name = conn[0], conn[1]
                a_idx = xyz.model.index(a_name)
                b_idx = xyz.model.index(b_name)
            except (KeyError, ValueError, IndexError):
                continue
            xs = [positions[a_idx, 0], positions[b_idx, 0]]
            ys = [positions[a_idx, 1], positions[b_idx, 1]]
            zs = [positions[a_idx, 2], positions[b_idx, 2]]
            self._stick_lines[i].set_data_3d(xs, ys, zs)

        # Title shows current idx + time
        seconds = idx / self.model.samplerate
        self._ax.set_title(
            f"frame {idx} / {self.model.n_samples}  "
            f"t = {seconds:.2f} s",
            fontsize=10,
        )
        self._canvas.draw_idle()


# ─────────────────────────────────────────────────────────────────────── #
# State track view (piano-roll)                                             #
# ─────────────────────────────────────────────────────────────────────── #

class StateTrackView(QWidget):
    """Horizontal piano-roll of state masks centred on the playback head.

    Each state is drawn as a horizontal lane.  A red vertical line
    marks the current ``model.idx``.  The view shows ``window_size``
    samples of context on either side; states are coloured from
    a fixed palette.
    """

    def __init__(
        self,
        model:        PlaybackModel,
        parent:       Optional[QWidget] = None,
        window_size:  int = 200,
    ) -> None:
        super().__init__(parent)
        self.model       = model
        self.window_size = int(window_size)

        self._fig    = Figure(figsize=(8, 2.0), tight_layout=True)
        self._canvas = FigureCanvasQTAgg(self._fig)
        self._ax     = self._fig.add_subplot(111)
        self._cursor_line = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._canvas)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setMinimumHeight(120)

        model.subscribe(self._on_event)
        self._refresh()

    def _on_event(self, event: str) -> None:
        if event in ("idx", "states", "states_data",
                       "selected_states", "current_label"):
            self._refresh()

    def _refresh(self) -> None:
        ax = self._ax
        ax.clear()
        labels = list(self.model.states_data.keys())
        if not labels:
            ax.set_xlim(-self.window_size, self.window_size)
            ax.set_yticks([])
            self._canvas.draw_idle()
            return

        n = self.model.n_samples
        idx = self.model.idx
        lo = max(0, idx - self.window_size)
        hi = min(n, idx + self.window_size)
        rel = np.arange(lo, hi) - idx

        # Use fixed cmap — round 19's standard palette
        from matplotlib import colormaps
        cmap = colormaps.get_cmap("tab10").resampled(max(1, len(labels)))
        for i, label in enumerate(labels):
            if not self.model.selected_states.get(label, True):
                continue
            mask = self.model.states_data[label][lo:hi]
            y = i + 0.5
            xs = rel[mask > 0]
            ax.scatter(
                xs,
                np.full(len(xs), y),
                marker="s", s=8,
                c=[cmap(i)],
                edgecolors="none",
            )
        # Cursor
        ax.axvline(0, color="r", linestyle="--", linewidth=1.0)
        # Labels with key bindings
        ytl = []
        for label in labels:
            key = self.model.state_keys.get(label, "?")
            tag = " ←" if label == self.model.current_label else ""
            ytl.append(f"{label} ({key}){tag}")
        ax.set_yticks(np.arange(len(labels)) + 0.5)
        ax.set_yticklabels(ytl)
        ax.set_xlim(-self.window_size, self.window_size)
        ax.set_ylim(0, max(1, len(labels)))
        ax.set_xlabel("offset (samples)")
        self._canvas.draw_idle()


# ─────────────────────────────────────────────────────────────────────── #
# Feature panel                                                              #
# ─────────────────────────────────────────────────────────────────────── #

class FeaturePanel(QWidget):
    """Multi-row feature time-trace plot synced to ``model.idx``.

    Parameters
    ----------
    model:
        Source of truth for playback head.
    features:
        ``{name: (T,) array}`` of feature time-series at the same
        samplerate as ``model``.  Pass an empty dict to display
        nothing initially; call :meth:`set_features` later.
    window_seconds:
        How much context to show on either side of the cursor.
    """

    def __init__(
        self,
        model:           PlaybackModel,
        features:        Optional[dict[str, np.ndarray]] = None,
        parent:          Optional[QWidget] = None,
        window_seconds:  float = 5.0,
    ) -> None:
        super().__init__(parent)
        self.model           = model
        self._features       = dict(features or {})
        self.window_seconds  = float(window_seconds)
        self._fig    = Figure(figsize=(8, 4), tight_layout=True)
        self._canvas = FigureCanvasQTAgg(self._fig)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._canvas)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        model.subscribe(self._on_event)
        self._refresh()

    def set_features(self, features: dict[str, np.ndarray]) -> None:
        """Replace the displayed feature set."""
        self._features = dict(features)
        self._refresh()

    def _on_event(self, event: str) -> None:
        if event in ("idx",):
            self._refresh()

    def _refresh(self) -> None:
        self._fig.clear()
        if not self._features:
            self._canvas.draw_idle()
            return

        n = self.model.n_samples
        sr = self.model.samplerate
        idx = self.model.idx
        win_samp = int(self.window_seconds * sr)
        lo = max(0, idx - win_samp)
        hi = min(n, idx + win_samp)
        rel_t = (np.arange(lo, hi) - idx) / sr

        n_feat = len(self._features)
        axes = self._fig.subplots(n_feat, 1, sharex=True)
        if n_feat == 1:
            axes = [axes]
        for ax, (name, sig) in zip(axes, self._features.items()):
            sig = np.asarray(sig).ravel()
            seg = sig[lo:hi] if hi <= len(sig) else sig[lo:]
            tx  = rel_t[:len(seg)]
            ax.plot(tx, seg, color="C0", linewidth=0.8)
            ax.axvline(0, color="r", linestyle="--", linewidth=1.0)
            ax.set_ylabel(name, fontsize=8, rotation=0,
                            ha="right", va="center")
            ax.tick_params(axis="both", labelsize=7)
        axes[-1].set_xlabel("time relative to cursor (s)", fontsize=8)
        self._canvas.draw_idle()
