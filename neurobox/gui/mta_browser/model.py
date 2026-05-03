"""
neurobox.gui.mta_browser.model
================================
Stateful, headless representation of the playback head, the current
state-collection edits, and the visible feature set.

This module deliberately depends only on numpy + neurobox dtypes — no
Qt imports — so it can be unit-tested without a display.

The MATLAB ``MLData`` struct in ``MTABrowser.m`` (set up in
``BSmotionLabeling_Callback`` lines 651-840 and mutated throughout the
file) bundled the playback head, the per-state time-series, the
selected-states mask, the feature display list, and the recording
flag together as anonymous ``setappdata`` keys.  This dataclass
makes that structure explicit and exposes change-notification
callbacks for the Qt layer to subscribe to.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from neurobox.dtype.epoch    import NBEpoch
from neurobox.dtype.session  import NBSession
from neurobox.dtype.stc      import NBStateCollection
from neurobox.dtype.xyz      import NBDxyz


__all__ = [
    "PlaybackModel",
]


@dataclass
class PlaybackModel:
    """In-memory state of the motion-labelling viewer.

    A single instance is created per loaded session and held by the
    Qt window.  All edits funnel through the model's ``set_*`` /
    ``add_*`` methods, which fire the registered listeners.

    Attributes
    ----------
    session : NBSession
        Loaded session (must already have ``xyz`` populated).
    n_samples : int
        ``session.xyz.data.shape[0]``.
    samplerate : float
        ``session.xyz.samplerate``.
    states_data : dict[str, np.ndarray]
        Per-state ``(n_samples,)`` int8 mask: ``1`` where the state
        is active, ``0`` elsewhere.  Initialised from
        ``session.stc`` and synced back on save.
    state_keys : dict[str, str]
        Per-state single-character keyboard shortcut.
    selected_states : dict[str, bool]
        Per-state visibility / selection in the editor table.
    current_label : Optional[str]
        Name of the state being edited; ``None`` = no labelling
        active.
    erase_mode : bool
        If True, the painting action erases instead of adds.
    play_speed : int
        Frames per advance step.  Higher = faster scrubbing.
    paused : bool
    idx : int
        Current playback head, in samples (0..n_samples-1).
    """
    session:          NBSession
    n_samples:        int
    samplerate:       float
    states_data:      dict[str, np.ndarray]      = field(default_factory=dict)
    state_keys:       dict[str, str]             = field(default_factory=dict)
    selected_states:  dict[str, bool]            = field(default_factory=dict)
    current_label:    Optional[str]              = None
    erase_mode:       bool                       = False
    play_speed:       int                        = 5
    paused:           bool                       = True
    idx:              int                        = 0
    listeners:        list[Callable[[str], None]] = field(default_factory=list)

    # ─────────────────────────────────────────────────────────────── #
    # Construction helpers                                              #
    # ─────────────────────────────────────────────────────────────── #

    @classmethod
    def from_session(cls, session: NBSession) -> "PlaybackModel":
        """Build a model from a freshly-loaded session."""
        if session.xyz is None:
            raise RuntimeError(
                "Session must have xyz loaded before building "
                "PlaybackModel."
            )
        n = int(session.xyz.data.shape[0])
        m = cls(
            session    = session,
            n_samples  = n,
            samplerate = float(session.xyz.samplerate),
        )
        m._populate_states_from_session()
        return m

    def _populate_states_from_session(self) -> None:
        """Convert ``session.stc`` periods → per-state binary masks."""
        self.states_data.clear()
        self.state_keys.clear()
        self.selected_states.clear()

        if self.session.stc is None or not self.session.stc.list_states():
            return
        for label in self.session.stc.list_states():
            ep = self.session.stc.get_state(label)
            mask = np.zeros(self.n_samples, dtype=np.int8)
            if ep.mode == "periods":
                periods_samp = (
                    np.asarray(ep.data) * self.samplerate
                ).astype(np.int64)
                for s, e in periods_samp:
                    s = max(0, int(s))
                    e = min(self.n_samples, int(e))
                    if e > s:
                        mask[s:e] = 1
            elif ep.mode == "mask":
                # Already a per-sample mask
                src = np.asarray(ep.data, dtype=np.int8).ravel()
                src_sr = float(getattr(ep, "samplerate", self.samplerate))
                if abs(src_sr - self.samplerate) < 1e-9:
                    n = min(len(src), self.n_samples)
                    mask[:n] = src[:n] != 0
                else:
                    # Naive resampling via repeat / decimation
                    factor = self.samplerate / src_sr
                    n = min(int(len(src) * factor), self.n_samples)
                    src_idx = (np.arange(n) / factor).astype(np.int64)
                    src_idx = np.clip(src_idx, 0, len(src) - 1)
                    mask[:n] = src[src_idx] != 0
            self.states_data[label] = mask
            # Resolve key from epoch.key if set, else first letter
            key = getattr(ep, "key", "") or label[0]
            self.state_keys[label] = str(key)
            self.selected_states[label] = True

    # ─────────────────────────────────────────────────────────────── #
    # Listener machinery                                                #
    # ─────────────────────────────────────────────────────────────── #

    def subscribe(self, listener: Callable[[str], None]) -> None:
        """Register a callback ``fn(event_name)`` for state changes."""
        self.listeners.append(listener)

    def _emit(self, event: str) -> None:
        for l in list(self.listeners):
            try:
                l(event)
            except Exception:                       # pragma: no cover
                # Don't let one buggy listener kill the rest
                import traceback
                traceback.print_exc()

    # ─────────────────────────────────────────────────────────────── #
    # Mutation API                                                      #
    # ─────────────────────────────────────────────────────────────── #

    def set_idx(self, idx: int) -> None:
        idx = int(np.clip(idx, 0, max(0, self.n_samples - 1)))
        if idx != self.idx:
            self.idx = idx
            self._emit("idx")

    def step(self, delta: int) -> None:
        """Advance the playback head by *delta* samples and apply
        labelling/erasing if active."""
        new = int(np.clip(self.idx + delta,
                            0, max(0, self.n_samples - 1)))
        # Paint over the just-traversed range
        if self.current_label and self.current_label in self.states_data:
            lo, hi = sorted([self.idx, new])
            mask = self.states_data[self.current_label]
            if self.erase_mode:
                mask[lo:hi + 1] = 0
            else:
                mask[lo:hi + 1] = 1
            self._emit("states_data")
        self.idx = new
        self._emit("idx")

    def set_play_speed(self, speed: int) -> None:
        speed = int(max(1, min(50, speed)))
        if speed != self.play_speed:
            self.play_speed = speed
            self._emit("play_speed")

    def set_paused(self, paused: bool) -> None:
        if bool(paused) != self.paused:
            self.paused = bool(paused)
            self._emit("paused")

    def set_current_label(self, label: Optional[str]) -> None:
        if label is not None and label not in self.states_data:
            raise KeyError(f"State {label!r} not in model")
        if label != self.current_label:
            self.current_label = label
            self._emit("current_label")

    def set_erase_mode(self, erase: bool) -> None:
        if bool(erase) != self.erase_mode:
            self.erase_mode = bool(erase)
            self._emit("erase_mode")

    def add_state(self, label: str, key: str) -> None:
        """Create a new empty state."""
        if not label:
            raise ValueError("State label must be non-empty")
        if not key or len(key) != 1:
            raise ValueError("State key must be exactly one character")
        if label in self.states_data:
            raise ValueError(f"State {label!r} already exists")
        if key in self.state_keys.values():
            raise ValueError(f"Key {key!r} already used by another state")
        self.states_data[label] = np.zeros(self.n_samples, dtype=np.int8)
        self.state_keys[label] = key
        self.selected_states[label] = True
        self._emit("states")

    def remove_state(self, label: str) -> None:
        if label not in self.states_data:
            raise KeyError(label)
        del self.states_data[label]
        del self.state_keys[label]
        del self.selected_states[label]
        if self.current_label == label:
            self.current_label = None
        self._emit("states")

    def set_state_selected(self, label: str, selected: bool) -> None:
        if label in self.selected_states:
            if self.selected_states[label] != selected:
                self.selected_states[label] = bool(selected)
                self._emit("selected_states")

    def rename_state(self, old: str, new: str) -> None:
        if new in self.states_data:
            raise ValueError(f"State {new!r} already exists")
        self.states_data[new] = self.states_data.pop(old)
        self.state_keys[new]  = self.state_keys.pop(old)
        self.selected_states[new] = self.selected_states.pop(old)
        if self.current_label == old:
            self.current_label = new
        self._emit("states")

    def set_state_key(self, label: str, key: str) -> None:
        if not key or len(key) != 1:
            raise ValueError("State key must be exactly one character")
        if key in self.state_keys.values() and \
                self.state_keys.get(label) != key:
            raise ValueError(f"Key {key!r} already used by another state")
        self.state_keys[label] = key
        self._emit("state_keys")

    # ─────────────────────────────────────────────────────────────── #
    # State export                                                      #
    # ─────────────────────────────────────────────────────────────── #

    def to_state_collection(self, mode: str = "manual"
                             ) -> NBStateCollection:
        """Build a fresh :class:`NBStateCollection` from the current
        masks.  Each state's binary mask is converted into a
        ``(n_periods, 2)`` periods array (in seconds)."""
        stc = NBStateCollection(mode=mode)
        for label, mask in self.states_data.items():
            d = np.diff(mask.astype(np.int8), prepend=0, append=0)
            starts = np.flatnonzero(d == 1)
            ends   = np.flatnonzero(d == -1)
            periods = np.column_stack([starts, ends]).astype(np.float64)
            periods /= self.samplerate
            ep = NBEpoch(
                data       = periods,
                samplerate = self.samplerate,
                mode       = "periods",
                label      = label,
                key        = self.state_keys.get(label, label[:1]),
            )
            stc.add_state(ep, label=label, key=ep.key)
        return stc

    def update_session_stc(self, mode: str = "manual") -> None:
        """Push the current state-edits back to ``session.stc``."""
        self.session.stc = self.to_state_collection(mode=mode)
        self._emit("session.stc")
