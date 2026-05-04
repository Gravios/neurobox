"""
data.py  —  NBData
==================
Abstract base class for all neurobox time-series and period data
objects.  Port of MTAData.

Design differences from MTAData
---------------------------------
* ``type_`` used instead of ``type`` (avoids shadowing Python builtin).
* ``dtype`` used instead of ``class`` (same reason).
* ``samplerate`` (lowercase) throughout.
* Period-based indexing via ``__getitem__``; named-feature indexing
  via ``sel(labels, dims)``.
* Hash mechanism (``self.hash`` + ``update_hash()``) ported from
  MATLAB's ``MTAData.update_hash`` — derives a content tag from
  the object's identity fields plus an optional modification hash
  produced by transforms (``filter``, ``resample`` etc.).  Used as
  a cache-invalidation key by :func:`neurobox.io.cached_compute`.
* Concrete subclasses must implement ``data`` as a regular numpy array
  property rather than an abstract MATLAB property.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from neurobox.dtype.epoch import NBEpoch, select_periods


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _trim_or_pad(arr: np.ndarray, target_n: int) -> np.ndarray:
    """Trim or zero-pad *arr* along axis 0 to exactly *target_n* samples.

    Used by ``NBData.resample`` to match output length to a target object
    (equivalent to the tail-correction in MTA's MTAData.resample).
    """
    n = arr.shape[0]
    if n == target_n:
        return arr
    if n > target_n:
        return arr[:target_n]
    # Pad with zeros
    pad = np.zeros((target_n - n,) + arr.shape[1:], dtype=arr.dtype)
    return np.concatenate([arr, pad], axis=0)




class NBData(ABC):
    """Abstract base for neurobox data containers.

    Subclasses
    ----------
    NBDxyz, NBDlfp  (and future NBDang, NBDufr …)

    Parameters
    ----------
    path:
        Directory where the backing ``.pkl`` / binary file is stored.
    filename:
        File name (without path).
    data:
        Underlying numpy array.  Shape conventions:
        TimeSeries → ``(T, *features)``;  TimePeriods → ``(N, 2)``.
    samplerate:
        Samples per second.
    sync:
        NBEpoch defining the recording window.
    origin:
        Time in seconds of the first sample of ``data`` within the
        broader recording.
    type_:
        ``'TimeSeries'`` or ``'TimePeriods'``.
    ext:
        Short file extension string (e.g. ``'pos'``, ``'lfp'``).
    name:
        Subject / session identifier.
    label:
        Human-readable data-type label.
    key:
        Single-character keyboard shorthand.
    """

    def __init__(
        self,
        path: Path | str | None = None,
        filename: str | None = None,
        data: np.ndarray | None = None,
        samplerate: float = 1.0,
        sync: NBEpoch | None = None,
        origin: float = 0.0,
        type_: str = "TimeSeries",
        ext: str = "",
        name: str = "",
        label: str = "",
        key: str = "",
        stream_sync: "StreamSync | None" = None,
    ) -> None:
        self.path: Path | None = Path(path) if path is not None else None
        self.filename: str | None = filename
        self._data: np.ndarray | None = (
            np.asarray(data) if data is not None else None
        )
        self.samplerate: float = float(samplerate)
        self.sync: NBEpoch | None = sync
        self.origin: float = float(origin)
        self.type_: str = type_
        self.ext: str = ext
        self.name: str = name
        self.label: str = label
        self.key: str = key
        # Round-23 multi-segment-aware sync.  Optional — when None,
        # the data is assumed to be continuous from t=0 at this
        # stream's samplerate (the simple single-source case).
        self.stream_sync: "StreamSync | None" = stream_sync
        # Round-23 multi-segment recording windows.  Set when the
        # source array was zero-filled across recording gaps (e.g.
        # by sync_nlx_vicon for Vicon-on / Vicon-off blocks within
        # one master-clock recording).  ``None`` means no gap
        # information available.  Shape ``(n_blocks, 2)`` in
        # master-clock seconds; non-None implies ``stream_sync``
        # is a single spanning segment with zero-filled gaps inside
        # the ``data`` array, and these are the true recording
        # windows to use for valid-sample masking and
        # ``restrict_to_window(fill_gaps=False)``.
        self.recording_windows: "np.ndarray | None" = None
        # Identity hash — computed last so all fields are populated.
        # Mirrors MATLAB MTAData/update_hash.m.
        self.hash: str = ""
        self.update_hash()

    # ------------------------------------------------------------------ #
    # Hash mechanism                                                       #
    # ------------------------------------------------------------------ #

    def update_hash(self, modification_hash: str | None = None) -> None:
        """Recompute ``self.hash`` from identity fields + transform tag.

        Port of :file:`MTA/@MTAData/update_hash.m`.

        The hash combines:

        * Identity: ``filename``, ``name``, ``label``, ``key``, ``ext``,
          ``samplerate``, ``sync``, ``origin``.
        * An optional *modification_hash* tag produced by the calling
          transform (e.g. a ``data_hash({mode, order, freq})`` from
          :meth:`filter`).

        Why the array data is **not** hashed
        -------------------------------------
        Following the MATLAB original, the underlying ``data`` array
        is intentionally excluded from the identity hash.  The MATLAB
        convention assumes ``filename`` uniquely identifies what the
        data is — so two ``MTADxyz`` objects with the same filename
        always hash identically.  In neurobox the same convention
        applies: when working with **in-memory objects without a
        backing file** (e.g. synthetic test data), set ``name=`` to a
        unique session identifier so distinct objects hash distinctly.

        The result is a SHA-1 hex digest (40 chars).  When transforms
        chain, each successive ``update_hash`` includes the previous
        ``self.hash`` automatically because that's part of the object's
        identity at that moment.

        Parameters
        ----------
        modification_hash:
            Hex digest produced by the caller from its transform
            parameters.  ``None`` for fresh-construction hashing.
        """
        # Late import to avoid a top-level neurobox.io ↔ neurobox.dtype
        # circular dependency.
        from neurobox.io.data_hash import data_hash

        # Sync object isn't directly hashable — summarise it.
        sync_repr = None
        if self.sync is not None:
            sync_data = getattr(self.sync, "data", None)
            sync_sr   = getattr(self.sync, "samplerate", None)
            sync_repr = (
                np.asarray(sync_data).tobytes() if sync_data is not None else None,
                float(sync_sr) if sync_sr is not None else None,
            )

        self.hash = data_hash([
            self.filename,
            self.name,
            self.label,
            self.key,
            self.ext,
            self.samplerate,
            sync_repr,
            self.origin,
            modification_hash,
        ])

    # ------------------------------------------------------------------ #
    # Data property (subclasses may override for lazy loading)            #
    # ------------------------------------------------------------------ #

    @property
    def data(self) -> np.ndarray | None:
        return self._data

    @data.setter
    def data(self, value: np.ndarray | None) -> None:
        self._data = None if value is None else np.asarray(value)

    # ------------------------------------------------------------------ #
    # Abstract interface                                                   #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def load(self, *args, **kwargs) -> "NBData":
        """Load data from disk into ``self.data``."""

    @abstractmethod
    def create(self, *args, **kwargs) -> "NBData":
        """Create / compute ``self.data`` from raw sources."""

    # ------------------------------------------------------------------ #
    # Size / shape                                                         #
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        if self._data is None:
            return 0
        return len(self._data)

    def isempty(self) -> bool:
        return self._data is None or self._data.size == 0

    @property
    def shape(self) -> tuple:
        return () if self._data is None else self._data.shape

    @property
    def n_samples(self) -> int:
        """Number of time samples (first axis)."""
        return 0 if self._data is None else self._data.shape[0]

    @property
    def duration(self) -> float:
        """Recording duration in seconds."""
        return self.n_samples / self.samplerate

    # ------------------------------------------------------------------ #
    # Indexing                                                             #
    # ------------------------------------------------------------------ #

    def __getitem__(self, idx):
        """Index into the data array.

        **Period-based selection** (equivalent to MTA's ``SelectPeriods``):
        triggered when the first index is an :class:`NBEpoch` or an
        ``(N, 2)`` *float64* array of ``[start_sec, stop_sec]`` pairs.

        When additional axes are given (e.g. ``lfp[epoch, :2]`` or
        ``xyz[epoch, 'head']``), the non-time axes are applied to the
        underlying array *first* (with ``:`` replacing the time axis),
        and period selection is then applied to that result — exactly
        mirroring MTA's ``subsref``::

            Sa.subs{1} = ':';
            Data = SelectPeriods(builtin('subsref',Data.data,Sa), S.subs{1}, 'c');

        An NBEpoch is resampled to ``self.samplerate`` before conversion
        to period pairs when sample rates differ.

        **Normal numpy indexing** for all other index types.

        Examples
        --------
        >>> lfp[stc['walk']]                  # NBEpoch — all channels
        >>> lfp[stc['walk'], :2]              # first 2 channels only
        >>> xyz[stc['walk'], :, [0, 2]]       # x and z dims of all markers
        >>> lfp[stc['walk'].data]             # (N,2) float64 — same
        >>> lfp[:, 2]                         # channel slice — normal numpy
        """
        if self._data is None:
            raise IndexError("Data has not been loaded yet.")

        from neurobox.dtype.epoch import NBEpoch as _NBEpoch

        first, rest = (idx[0], idx[1:]) if isinstance(idx, tuple) else (idx, ())

        # ── Determine if this is a period-select on the time axis ──── #
        is_epoch = isinstance(first, _NBEpoch)
        is_period_array = (
            isinstance(first, np.ndarray)
            and first.ndim == 2 and first.shape[1] == 2
            and np.issubdtype(first.dtype, np.floating)
        )

        if is_epoch or is_period_array:
            # MTA subsref order:
            #   1. Apply non-time indices with ':' for time axis
            #   2. Period-select the result along axis 0
            if rest:
                base = self._data[(slice(None),) + rest]
            else:
                base = self._data

            if is_epoch:
                ep      = (first if first.samplerate == self.samplerate
                           else first.resample(self.samplerate))
                periods = ep._as_periods()
            else:
                periods = first

            return select_periods(base, periods, self.samplerate)

        # ── Normal numpy indexing ─────────────────────────────────────── #
        return self._data[idx]

    # ------------------------------------------------------------------ #
    # Filtering                                                            #
    # ------------------------------------------------------------------ #

    def filter(self, mode: str = "butter", **kwargs) -> "NBData":
        """Apply a filter to the time-series data in-place.

        Parameters
        ----------
        mode:
            ``'butter'``   — Butterworth (default).
            ``'gauss'``    — Gaussian smoothing (window in seconds).
            ``'rect'``     — Rectangular (boxcar) smoothing.
        **kwargs:
            For ``'butter'``: ``order`` (int, default 3),
            ``cutoff`` (Hz or [lo, hi] for bandpass),
            ``btype`` (``'low'``/``'high'``/``'band'``/``'bandstop'``).
            For ``'gauss'``: ``sigma_sec`` (default 0.05 s).
            For ``'rect'``: ``window_sec`` (default 0.1 s).
        """
        from scipy.signal import butter, sosfiltfilt
        from scipy.ndimage import uniform_filter1d
        from neurobox.io.data_hash import data_hash

        if self._data is None:
            return self

        if self.type_ != "TimeSeries":
            return self

        d = self._data.astype(np.float64)

        if mode == "butter":
            order  = kwargs.get("order", 3)
            cutoff = kwargs.get("cutoff", 4.0)
            btype  = kwargs.get("btype", "low")
            nyq    = self.samplerate / 2.0
            if isinstance(cutoff, (list, tuple, np.ndarray)):
                wn = [c / nyq for c in cutoff]
            else:
                wn = cutoff / nyq
            sos = butter(order, wn, btype=btype, output="sos")
            orig_shape = d.shape
            d = sosfiltfilt(sos, d.reshape(d.shape[0], -1), axis=0)
            self._data = d.reshape(orig_shape)
            mod_hash = data_hash({"mode": "butter", "order": order,
                                  "cutoff": cutoff, "btype": btype})

        elif mode == "gauss":
            sigma_sec = kwargs.get("sigma_sec", 0.05)
            sigma_samp = sigma_sec * self.samplerate
            from scipy.ndimage import gaussian_filter1d
            orig_shape = self._data.shape
            self._data = gaussian_filter1d(
                self._data.astype(np.float64).reshape(orig_shape[0], -1),
                sigma=sigma_samp, axis=0
            ).reshape(orig_shape)
            mod_hash = data_hash({"mode": "gauss", "sigma_sec": sigma_sec})

        elif mode == "rect":
            window_sec = kwargs.get("window_sec", 0.1)
            window_samp = max(1, int(round(window_sec * self.samplerate)))
            orig_shape = self._data.shape
            self._data = uniform_filter1d(
                self._data.astype(np.float64).reshape(orig_shape[0], -1),
                size=window_samp, axis=0
            ).reshape(orig_shape)
            mod_hash = data_hash({"mode": "rect", "window_sec": window_sec})

        else:
            raise ValueError(f"Unknown filter mode: {mode!r}")

        # Each transform bumps the hash.  Including self.hash means
        # subsequent transforms naturally chain — same as MATLAB.
        self.update_hash(data_hash({"prev": self.hash, "filter": mod_hash}))
        return self

    # ------------------------------------------------------------------ #
    # Resampling                                                           #
    # ------------------------------------------------------------------ #

    def resample(self, target, method: str = "poly") -> "NBData":
        """Resample the time series to a new sample rate.

        Mirrors the MTA ``MTAData.resample`` API:

        * When *target* is a **number**, resample to that Hz.
        * When *target* is another **NBData object**, resample to match
          its ``samplerate`` and trim/pad to its ``n_samples`` exactly.

        Anti-aliasing
        -------------
        When downsampling, a Butterworth lowpass at ``new_sr / 2`` is
        applied before interpolation — mirroring the ``ButFilter`` call
        in MTA's ``MTAData.resample``.  ``'poly'`` (default) uses
        ``scipy.signal.resample_poly`` which is inherently anti-aliased
        via a Kaiser-windowed FIR filter.  ``'spline'`` pre-filters
        manually then uses cubic spline interpolation.

        Parameters
        ----------
        target:
            New sample rate in Hz (``float``), **or** another
            ``NBData`` object to resample to match.
        method:
            ``'poly'`` (default) — efficient for integer ratios.
            ``'spline'`` — cubic spline, better for small non-integer
            rate ratios.

        Returns
        -------
        self  (modified in-place)
        """
        if self._data is None or self.type_ != "TimeSeries":
            return self

        # Resolve target samplerate and optional output length
        if isinstance(target, NBData):
            new_samplerate = float(target.samplerate)
            target_n       = target.n_samples if target.n_samples > 0 else None
        else:
            new_samplerate = float(target)
            target_n       = None

        if new_samplerate == self.samplerate:
            if target_n is not None and target_n != self.n_samples:
                self._data = _trim_or_pad(self._data, target_n)
            return self

        n_in       = self._data.shape[0]
        orig_shape = self._data.shape
        d          = self._data.astype(np.float64).reshape(n_in, -1)

        n_out = (target_n if target_n is not None
                 else int(round(n_in / self.samplerate * new_samplerate)))

        if method == "poly":
            from math import gcd
            from scipy.signal import resample_poly
            # ×1000 to handle sub-Hz rates like 119.881 Hz (Vicon)
            scale = 1000
            g  = gcd(int(round(new_samplerate * scale)),
                     int(round(self.samplerate  * scale)))
            up = int(round(new_samplerate * scale)) // g
            dn = int(round(self.samplerate  * scale)) // g
            d  = resample_poly(d, up, dn, axis=0)
        else:
            # Anti-alias: lowpass at new Nyquist before downsampling
            if new_samplerate < self.samplerate:
                from scipy.signal import butter, sosfiltfilt
                nyq = self.samplerate / 2.0
                wn  = min((new_samplerate / 2.0) / nyq, 0.9999)
                sos = butter(3, wn, btype="low", output="sos")
                d   = sosfiltfilt(sos, d, axis=0)
            from scipy.interpolate import interp1d
            t_in  = np.linspace(0, 1, n_in)
            t_out = np.linspace(0, 1, n_out)
            d     = interp1d(t_in, d, kind="cubic", axis=0,
                             fill_value="extrapolate")(t_out)

        d = _trim_or_pad(d, n_out)
        self._data      = d.reshape((n_out,) + orig_shape[1:])
        self.samplerate = new_samplerate

        # Bump hash with resample parameters.
        from neurobox.io.data_hash import data_hash
        self.update_hash(data_hash({
            "prev":             self.hash,
            "resample_to":      new_samplerate,
            "resample_method":  method,
            "resample_target_n": target_n,
        }))
        return self

    # ------------------------------------------------------------------ #
    # Segments (segs)                                                      #
    # ------------------------------------------------------------------ #

    def segs(self, start_times: np.ndarray,
             seg_len_sec: float = 0.5,
             fill_value: float = np.nan) -> np.ndarray:
        """Extract fixed-length snippets starting at *start_times* (s).

        Parameters
        ----------
        start_times:
            1-D array of segment start times in seconds.
        seg_len_sec:
            Length of each segment in seconds.
        fill_value:
            Value used for out-of-range samples.

        Returns
        -------
        segments : np.ndarray, shape ``(n_segs, seg_samp, *features)``
        """
        if self._data is None:
            raise RuntimeError("Data not loaded.")
        seg_samp = int(round(seg_len_sec * self.samplerate))
        n_segs   = len(start_times)
        out_shape = (n_segs, seg_samp) + self._data.shape[1:]
        out = np.full(out_shape, fill_value, dtype=np.float64)
        n_total = self._data.shape[0]
        for j, t in enumerate(start_times):
            i0 = int(round(t * self.samplerate))
            i1 = i0 + seg_samp
            # clip and copy
            src_lo = max(0, i0)
            src_hi = min(n_total, i1)
            dst_lo = src_lo - i0
            dst_hi = dst_lo + (src_hi - src_lo)
            if src_hi > src_lo:
                out[j, dst_lo:dst_hi] = self._data[src_lo:src_hi]
        return out

    # ------------------------------------------------------------------ #
    # File path helper                                                     #
    # ------------------------------------------------------------------ #

    @property
    def fpath(self) -> Path | None:
        if self.path is None or self.filename is None:
            return None
        return self.path / self.filename

    # ------------------------------------------------------------------ #
    # Comparison operators (delegate to data array)                       #
    # ------------------------------------------------------------------ #

    def __eq__(self, other):
        d = self._data
        return d == (other._data if isinstance(other, NBData) else other)

    def __ne__(self, other):
        d = self._data
        return d != (other._data if isinstance(other, NBData) else other)

    def __gt__(self, other):
        d = self._data
        return d > (other._data if isinstance(other, NBData) else other)

    def __ge__(self, other):
        d = self._data
        return d >= (other._data if isinstance(other, NBData) else other)

    def __lt__(self, other):
        d = self._data
        return d < (other._data if isinstance(other, NBData) else other)

    def __le__(self, other):
        d = self._data
        return d <= (other._data if isinstance(other, NBData) else other)


    # ------------------------------------------------------------------ #
    # Trial windowing (round 23 — replaces MTAData.resync)                 #
    # ------------------------------------------------------------------ #

    def restrict_to_window(
        self,
        window:    "TrialWindow",
        *,
        fill_gaps: bool = True,
    ) -> "NBData":
        """Return a new :class:`NBData` restricted to a trial window.

        The returned object's ``data`` covers exactly the trial's
        master-clock window at this stream's samplerate.  Samples
        where this stream was actively recording are copied from
        ``self.data``; gaps inside the window where the stream
        wasn't recording are zero-filled (when ``fill_gaps=True``,
        the default — matching MATLAB ``MTAData.resync`` semantics).

        How "recording windows" are determined
        --------------------------------------
        Three possible representations are supported, in priority
        order:

        1. **``self.recording_windows`` set** (typical for streams
           assembled from per-block mocap files via
           :func:`sync_nlx_vicon` and friends).  ``self.data`` is
           assumed to be a single contiguous array spanning
           ``self.stream_sync``'s span, with zero-filled gaps inside
           where this stream wasn't recording; ``recording_windows``
           identifies the true recording extents in master-clock
           seconds.
        2. **``self.stream_sync`` set, no recording_windows** (the
           explicit multi-segment case where the data array is
           compact — segments concatenated with no gap fill).
        3. **Neither set** — assume continuous recording from
           ``t = 0`` at ``self.samplerate``.

        Parameters
        ----------
        window:
            :class:`TrialWindow` defining the master-clock periods of
            the trial.
        fill_gaps:
            If True (default), output is contiguous from the trial's
            ``t_start`` to ``t_stop`` with zero-fill in stream gaps.
            If False, output is compact (only the recorded portions
            concatenated).

        Returns
        -------
        NBData
            New instance.  The original is unchanged.
        """
        from neurobox.dtype.sync import StreamSync, TrialWindow

        if self._data is None:
            raise RuntimeError(
                "restrict_to_window: data not loaded; call .load() first."
            )

        # Resolve which sync representation describes this object.
        # If recording_windows is set, the data is session-frame-
        # aligned (with zero-fills inside).  We synthesise a
        # logical sync that maps local samples back to master time
        # via the spanning stream_sync, but uses recording_windows
        # for gap-aware slicing.
        if self.recording_windows is not None and self.stream_sync is not None:
            # Span sync (single segment) for local↔master arithmetic.
            span_sync = self.stream_sync
            # Per-block recording windows for valid-sample selection.
            real_segments = np.asarray(self.recording_windows,
                                          dtype=np.float64)
            sync_for_data = StreamSync(
                segments   = real_segments,
                samplerate = self.samplerate,
            )
            # In this case, "local sample" = sample index in the
            # session-frame-aligned data array (relative to span start).
            span_t0 = span_sync.master_first
            sr      = self.samplerate

            def _master_to_local_in_data(t: float) -> int:
                return int(np.round((t - span_t0) * sr))

        elif self.stream_sync is not None:
            # Pure compact multi-segment representation: data length
            # equals stream_sync.total_samples.
            sync_for_data = self.stream_sync
            span_t0       = None    # use sync_for_data.local_to_master
            sr            = self.samplerate
            _master_to_local_in_data = None

        else:
            # No sync at all — assume continuous from t=0
            n  = self._data.shape[0]
            sr = self.samplerate
            sync_for_data = StreamSync.continuous(
                duration_sec = n / sr,
                samplerate   = sr,
            )
            span_t0 = 0.0

            def _master_to_local_in_data(t: float) -> int:
                return int(np.round(t * sr))

        if abs(sync_for_data.samplerate - self.samplerate) > 1e-6:
            raise ValueError(
                f"sync samplerate ({sync_for_data.samplerate}) doesn't "
                f"match data.samplerate ({self.samplerate})"
            )

        per_period_arrays: list[np.ndarray] = []
        new_segments:      list[list[float]] = []

        for p_start, p_stop in window.periods:
            p_start = float(p_start); p_stop = float(p_stop)

            # Find the parts of this period where the stream actually
            # recorded — sync_for_data.slice_for_window returns local
            # indices into a COMPACT view, so we have to translate.
            recorded_intervals: list[tuple[float, float]] = []
            for seg_t0, seg_t1 in sync_for_data.segments:
                ov_t0 = max(seg_t0, p_start)
                ov_t1 = min(seg_t1, p_stop)
                if ov_t0 < ov_t1:
                    recorded_intervals.append((ov_t0, ov_t1))

            if fill_gaps:
                n_out = int(np.round((p_stop - p_start) * sr))
                shape = (n_out,) + self._data.shape[1:]
                buf = np.zeros(shape, dtype=self._data.dtype)
                for ov_t0, ov_t1 in recorded_intervals:
                    out_lo = int(np.round((ov_t0 - p_start) * sr))
                    out_hi = int(np.round((ov_t1 - p_start) * sr))
                    out_lo = max(0, out_lo); out_hi = min(n_out, out_hi)
                    if span_t0 is not None:
                        # data is span-aligned: local idx = (t - span_t0) × sr
                        src_lo = int(np.round((ov_t0 - span_t0) * sr))
                        src_hi = int(np.round((ov_t1 - span_t0) * sr))
                    else:
                        # data is compact: use sync_for_data.master_to_local
                        src_lo = sync_for_data.master_to_local(ov_t0)
                        src_hi_lookup = sync_for_data.master_to_local(
                            ov_t1 - 1.0 / sr
                        )
                        src_hi = (src_hi_lookup + 1
                                  if src_hi_lookup is not None
                                  else src_lo + (out_hi - out_lo))
                    src_lo = max(0, src_lo)
                    src_hi = min(self._data.shape[0], src_hi)
                    if out_hi > out_lo and src_hi > src_lo:
                        copy_len = min(out_hi - out_lo, src_hi - src_lo)
                        buf[out_lo:out_lo + copy_len] = (
                            self._data[src_lo:src_lo + copy_len]
                        )
                per_period_arrays.append(buf)
            else:
                # Compact: just concat the recorded slices
                for ov_t0, ov_t1 in recorded_intervals:
                    if span_t0 is not None:
                        src_lo = int(np.round((ov_t0 - span_t0) * sr))
                        src_hi = int(np.round((ov_t1 - span_t0) * sr))
                    else:
                        src_lo = sync_for_data.master_to_local(ov_t0)
                        src_hi_lookup = sync_for_data.master_to_local(
                            ov_t1 - 1.0 / sr
                        )
                        src_hi = (src_hi_lookup + 1
                                  if src_hi_lookup is not None
                                  else src_lo)
                    src_lo = max(0, src_lo)
                    src_hi = min(self._data.shape[0], src_hi)
                    if src_hi > src_lo:
                        per_period_arrays.append(self._data[src_lo:src_hi])

            # New segments in master-clock seconds
            for ov_t0, ov_t1 in recorded_intervals:
                new_segments.append([ov_t0, ov_t1])

        new_data = (np.concatenate(per_period_arrays, axis=0)
                     if per_period_arrays
                     else np.zeros((0,) + self._data.shape[1:],
                                    dtype=self._data.dtype))
        new_sync = StreamSync(
            segments   = (np.asarray(new_segments, dtype=np.float64)
                            if new_segments else np.zeros((0, 2))),
            samplerate = sr,
        )

        # Build the new NBData via copy so subclass-specific fields
        # (e.g. NBDxyz.model, NBDfet.columns) are preserved.
        new = self.copy()
        new._data       = new_data
        new.stream_sync = new_sync
        # Compact form: clear recording_windows since the new data
        # is no longer span-aligned with internal gaps
        new.recording_windows = None
        new.update_hash()
        return new

    # ------------------------------------------------------------------ #
    # copy / clear / update paths                                         #
    # ------------------------------------------------------------------ #

    def copy(self) -> "NBData":
        """Return a deep copy of this data object."""
        import copy as _copy
        return _copy.deepcopy(self)

    def clear(self) -> "NBData":
        """Free the in-memory data array (keeps metadata intact).

        Mirrors ``MTAData.clear``.  Useful after saving to disk to
        reclaim RAM while retaining path/samplerate/sync metadata so
        the object can be re-loaded later.
        """
        self._data = None
        return self

    def update_path(self, path) -> "NBData":
        """Update the source file directory path.

        Mirrors ``MTAData.updatePath``.
        """
        self.path = Path(path) if path is not None else None
        return self

    def update_filename(self, filename: str) -> "NBData":
        """Update the source filename (base name, no directory).

        Mirrors ``MTAData.updateFilename``.
        """
        self.filename = filename
        return self

    # ------------------------------------------------------------------ #
    # phase  (Hilbert analytic phase of a band-passed signal)             #
    # ------------------------------------------------------------------ #

    def phase(self, freq_range=(6.0, 12.0), order: int = 3) -> "NBData":
        """Compute the instantaneous analytic phase via band-pass + Hilbert.

        Mirrors ``MTAData.phase`` (default theta band 6–12 Hz).

        Parameters
        ----------
        freq_range:
            ``(low_hz, high_hz)`` band-pass frequencies.
        order:
            Butterworth filter order (default 3).

        Returns
        -------
        NBData subclass of the same type whose data contains the
        unwrapped analytic phase in radians.
        """
        import copy as _copy
        from scipy.signal import sosfiltfilt, butter, hilbert

        if self._data is None:
            raise RuntimeError("Data not loaded.")

        # Band-pass filter
        nyq  = self.samplerate / 2.0
        lo   = float(freq_range[0]) / nyq
        hi   = float(freq_range[1]) / nyq
        sos  = butter(order, [lo, hi], btype="band", output="sos")

        data = self._data.astype(np.float64)
        # Apply along time axis (axis 0)
        filtered = sosfiltfilt(sos, data, axis=0)

        # Hilbert phase
        analytic = hilbert(filtered, axis=0)
        phs      = np.angle(analytic)

        out = _copy.deepcopy(self)
        out._data = phs.astype(np.float32)
        return out
