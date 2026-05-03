"""
neurobox.gui.mta_browser.lfp_widgets
======================================
LFP trace and scrolling-spectrogram widgets for the LFP-states tab.

Two widgets, both subscribing to a :class:`PlaybackModel` whose
samplerate is set by the loaded NB object (a precomputed
:class:`SpectrumResult` or raw :class:`NBDlfp`):

* :class:`LfpTraceView` — multi-channel raw LFP around the cursor.
  Time-domain conversion handles the case where the LFP samplerate
  differs from the model rate.
* :class:`SpectrogramView` — scrolling spectrogram with a centred
  cursor.  Accepts either a precomputed :class:`SpectrumResult` (no
  recompute — recommended) or raw LFP that will be passed through
  :func:`multitaper_spectrogram` synchronously on attach.

Design principle
----------------
Don't recompute what's already on disk.  When a session has a saved
:class:`SpectrumResult` (or future ``NBDspec`` object), the widget
displays it directly.  Raw-LFP fallback is for the case where the
spectrogram hasn't been precomputed yet.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget,
)

from neurobox.analysis.lfp.spectral import SpectrumResult

from .model import PlaybackModel


__all__ = [
    "LfpTraceView",
    "SpectrogramView",
]


# ─────────────────────────────────────────────────────────────────────── #
# Time conversion helper                                                     #
# ─────────────────────────────────────────────────────────────────────── #

def _model_idx_to_seconds(model: PlaybackModel) -> float:
    if model.samplerate <= 0:
        return 0.0
    return model.idx / model.samplerate


def _seconds_to_sample(seconds: float, sr: float) -> int:
    if sr <= 0:
        return 0
    return int(round(seconds * sr))


# ─────────────────────────────────────────────────────────────────────── #
# Raw LFP trace                                                              #
# ─────────────────────────────────────────────────────────────────────── #

class LfpTraceView(QWidget):
    """Multi-channel raw LFP plotted around the playback cursor.

    Channels are stacked vertically with an automatic y-offset based
    on the global signal std.  A red vertical line marks the current
    cursor position; the displayed window is ``window_seconds`` of
    context on either side.

    Parameters
    ----------
    model:
        Shared :class:`PlaybackModel`.  Cursor position comes from
        ``model.idx / model.samplerate`` (in seconds), allowing the
        LFP data to be at a different samplerate than the model.
    lfp:
        ``(T_lfp, n_channels)`` raw signal.  May be int16 or float.
        1-D arrays are treated as a single channel.
    lfp_samplerate:
        Hz of *lfp*.
    channels:
        Optional column subset.  Default uses all channels.
    window_seconds:
        Half-window of context on each side of the cursor.
    parent:
        Qt parent.
    """

    def __init__(
        self,
        model:           PlaybackModel,
        lfp:             np.ndarray,
        lfp_samplerate:  float,
        *,
        channels:        Optional[Sequence[int]] = None,
        window_seconds:  float = 1.0,
        parent:          Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.model            = model
        self.lfp_samplerate   = float(lfp_samplerate)
        self.window_seconds   = float(window_seconds)

        lfp = np.asarray(lfp)
        if lfp.ndim == 1:
            lfp = lfp[:, None]
        if channels is not None:
            lfp = lfp[:, list(channels)]
        self._lfp = lfp

        # Per-channel offset for stacked display
        std = float(np.nanstd(self._lfp.astype(np.float64).ravel()))
        self._channel_offset = std * 6.0 + 1e-6

        self._fig    = Figure(figsize=(8, 3.5), tight_layout=True)
        self._canvas = FigureCanvasQTAgg(self._fig)
        self._ax     = self._fig.add_subplot(111)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._canvas)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        model.subscribe(self._on_event)
        self._refresh()

    def _on_event(self, event: str) -> None:
        if event in ("idx",):
            self._refresh()

    def _refresh(self) -> None:
        ax = self._ax
        ax.clear()
        n_lfp, n_chan = self._lfp.shape
        win_samp = int(self.window_seconds * self.lfp_samplerate)
        centre   = _seconds_to_sample(_model_idx_to_seconds(self.model),
                                        self.lfp_samplerate)
        lo = max(0, centre - win_samp)
        hi = min(n_lfp, centre + win_samp)
        if hi <= lo:
            self._canvas.draw_idle()
            return

        rel_t = (np.arange(lo, hi) - centre) / self.lfp_samplerate
        for c in range(n_chan):
            seg = self._lfp[lo:hi, c].astype(np.float64)
            seg = seg - seg.mean()
            ax.plot(rel_t, seg + c * self._channel_offset,
                     linewidth=0.6, color=f"C{c % 10}")
            ax.text(
                rel_t[0], c * self._channel_offset, f"ch{c}  ",
                ha="right", va="center", fontsize=7, color=f"C{c % 10}",
            )
        ax.axvline(0, color="r", linestyle="--", linewidth=1.0)
        ax.set_xlim(rel_t[0], rel_t[-1])
        ax.set_xlabel("time relative to cursor (s)", fontsize=8)
        ax.set_yticks([])
        ax.tick_params(axis="x", labelsize=7)
        self._canvas.draw_idle()


# ─────────────────────────────────────────────────────────────────────── #
# Spectrogram                                                                #
# ─────────────────────────────────────────────────────────────────────── #

class SpectrogramView(QWidget):
    """Scrolling spectrogram with a centred cursor synced to the model.

    Accepts either:

    * A pre-computed :class:`SpectrumResult` from
      :func:`neurobox.analysis.lfp.spectral.multitaper_spectrogram`
      (recommended — no recompute).
    * Raw LFP, which will be run through ``multitaper_spectrogram``
      synchronously on attach.

    The widget converts ``model.idx`` to seconds, then locates the
    matching spectrogram window via :attr:`SpectrumResult.times`.
    Power is displayed in log10 with a configurable colormap and
    dB range.

    Parameters
    ----------
    model:
        Shared :class:`PlaybackModel`.  The model's samplerate
        determines how cursor seconds are computed; the spectrogram's
        own ``times`` array determines what gets displayed.
    spectrogram:
        Pre-computed :class:`SpectrumResult`.  Mutually exclusive
        with *lfp*.  If neither is provided, the widget shows an
        empty placeholder.
    lfp:
        Raw ``(T,)`` or ``(T, C)`` signal — only used when
        *spectrogram* is None.  Computation is synchronous.
    lfp_samplerate:
        Hz of *lfp*.  Required when *lfp* is provided.
    spec_kwargs:
        Extra kwargs forwarded to
        :class:`SpectralParams` when computing from raw LFP.
    channel:
        Channel index to display (default 0).  For
        :class:`SpectrumResult.power` shape ``(T, F, C)``, picks
        out the C-axis slice.
    window_seconds:
        Half-window of context shown on each side of the cursor.
    db_range:
        ``(vmin, vmax)`` for log10 power display.  Default
        ``(-2, 4)`` works well for hippocampal LFP after AR-whitening.
    colormap:
        Matplotlib colormap.  Default ``'viridis'``.
    parent:
        Qt parent.
    """

    def __init__(
        self,
        model:           PlaybackModel,
        *,
        spectrogram:     Optional[SpectrumResult] = None,
        lfp:             Optional[np.ndarray]     = None,
        lfp_samplerate:  Optional[float]          = None,
        spec_kwargs:     Optional[dict]           = None,
        channel:         int = 0,
        window_seconds:  float = 5.0,
        db_range:        tuple[float, float] = (-2.0, 4.0),
        colormap:        str = "viridis",
        parent:          Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.model           = model
        self.window_seconds  = float(window_seconds)
        self.db_range        = tuple(db_range)
        self.colormap        = colormap
        self.channel         = int(channel)

        # Compute or unwrap the spectrogram
        if spectrogram is not None:
            self._spec_result: Optional[SpectrumResult] = spectrogram
        elif lfp is not None:
            if lfp_samplerate is None:
                raise ValueError(
                    "lfp_samplerate is required when computing from raw LFP"
                )
            self._spec_result = self._compute_from_lfp(
                lfp, float(lfp_samplerate), spec_kwargs or {},
            )
        else:
            self._spec_result = None

        # Layout
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        self._status = QLabel(self._status_text())
        self._status.setStyleSheet("color: gray;")
        outer.addWidget(self._status)

        self._fig    = Figure(figsize=(8, 3.5), tight_layout=True)
        self._canvas = FigureCanvasQTAgg(self._fig)
        self._ax     = self._fig.add_subplot(111)
        outer.addWidget(self._canvas)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        model.subscribe(self._on_event)
        self._refresh()

    # ── Spectrogram computation (fallback path) ──────────────────── #

    def _compute_from_lfp(
        self,
        lfp:            np.ndarray,
        lfp_samplerate: float,
        spec_kwargs:    dict,
    ) -> SpectrumResult:
        from neurobox.analysis.lfp.spectral import (
            SpectralParams, multitaper_spectrogram,
        )
        if lfp.ndim == 1:
            lfp = lfp[:, None]
        # Sensible defaults for hippocampal LFP at 1250 Hz
        defaults = dict(
            samplerate = lfp_samplerate,
            n_fft      = 1024,
            win_len    = max(64, int(round(lfp_samplerate))),
            n_overlap  = max(0, int(round(lfp_samplerate * 0.8))),
            freq_range = (1.0, 100.0),
        )
        defaults.update(spec_kwargs)
        params = SpectralParams(**defaults)
        return multitaper_spectrogram(lfp.astype(np.float64), params)

    def _status_text(self) -> str:
        if self._spec_result is None:
            return "Spectrogram: not loaded"
        s = self._spec_result
        n_t = s.power.shape[0] if s.power.ndim > 2 else 1
        # Frame rate inferred from times array
        if s.times.size >= 2 and np.isfinite(s.times[1] - s.times[0]):
            fr = 1.0 / (s.times[1] - s.times[0])
            fr_txt = f", frame rate ≈ {fr:.2f} Hz"
        else:
            fr_txt = ""
        return (f"Spectrogram: {n_t} windows × {len(s.freqs)} freqs "
                f"({s.freqs[0]:.1f}-{s.freqs[-1]:.1f} Hz){fr_txt}")

    # ── Listener ─────────────────────────────────────────────────── #

    def _on_event(self, event: str) -> None:
        if event in ("idx",):
            self._refresh()

    # ── Drawing ──────────────────────────────────────────────────── #

    def _refresh(self) -> None:
        ax = self._ax
        ax.clear()
        s = self._spec_result
        if s is None:
            ax.text(0.5, 0.5, "(no spectrogram loaded)",
                     ha="center", va="center", transform=ax.transAxes,
                     color="gray", fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
            self._canvas.draw_idle()
            return

        cursor_sec = _model_idx_to_seconds(self.model)
        lo_sec     = cursor_sec - self.window_seconds
        hi_sec     = cursor_sec + self.window_seconds

        times = s.times
        # power shape: (T_win, F, C) for multi-segment
        if s.power.ndim == 3:
            power = s.power[..., min(self.channel, s.power.shape[-1] - 1)]
        elif s.power.ndim == 2:
            # (F, C) single-segment, or (T, F) single-channel — figure it out
            if s.power.shape[0] == s.freqs.size:
                # (F, C)
                power = s.power[:, min(self.channel,
                                          s.power.shape[1] - 1)][None, :]
            else:
                power = s.power
        else:
            power = s.power.reshape(-1, s.freqs.size)

        # Window indices in spectrogram time-axis
        if times.size == 1 or not np.isfinite(times).any():
            window_mask = np.ones(power.shape[0], dtype=bool)
            time_axis = np.arange(power.shape[0]).astype(np.float64)
        else:
            window_mask = (times >= lo_sec) & (times <= hi_sec)
            if not window_mask.any():
                # Outside the spectrogram range — just paint a placeholder
                ax.text(0.5, 0.5,
                         f"(cursor at {cursor_sec:.1f}s — outside "
                         f"spectrogram range "
                         f"[{times[0]:.1f}, {times[-1]:.1f}]s)",
                         ha="center", va="center", transform=ax.transAxes,
                         color="gray", fontsize=9)
                ax.set_xticks([]); ax.set_yticks([])
                self._canvas.draw_idle()
                return
            time_axis = times[window_mask] - cursor_sec

        seg = power[window_mask, :]              # (T_win, F)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_power = np.where(seg > 0, np.log10(seg), np.nan)

        # imshow expects (rows=freq, cols=time)
        ax.imshow(
            log_power.T,
            origin       = "lower",
            aspect       = "auto",
            extent       = [time_axis[0], time_axis[-1],
                             s.freqs[0], s.freqs[-1]],
            cmap         = self.colormap,
            vmin         = self.db_range[0],
            vmax         = self.db_range[1],
            interpolation = "nearest",
        )
        ax.axvline(0, color="r", linestyle="--", linewidth=1.0)
        ax.set_xlabel("time relative to cursor (s)", fontsize=8)
        ax.set_ylabel("frequency (Hz)", fontsize=8)
        ax.tick_params(axis="both", labelsize=7)
        self._canvas.draw_idle()

    # ── Public helpers ───────────────────────────────────────────── #

    @property
    def has_data(self) -> bool:
        return self._spec_result is not None

    @property
    def frame_rate_hz(self) -> Optional[float]:
        """The native frame rate of the spectrogram in Hz, or None
        if undetermined (e.g. single-segment result)."""
        if self._spec_result is None:
            return None
        t = self._spec_result.times
        if t.size < 2 or not np.isfinite(t[1] - t[0]) or t[1] == t[0]:
            return None
        return float(1.0 / (t[1] - t[0]))
