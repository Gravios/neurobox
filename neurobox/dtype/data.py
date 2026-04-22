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
* No MATLAB hash machinery — add when needed.
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

        elif mode == "gauss":
            sigma_sec = kwargs.get("sigma_sec", 0.05)
            sigma_samp = sigma_sec * self.samplerate
            from scipy.ndimage import gaussian_filter1d
            orig_shape = self._data.shape
            self._data = gaussian_filter1d(
                self._data.astype(np.float64).reshape(orig_shape[0], -1),
                sigma=sigma_samp, axis=0
            ).reshape(orig_shape)

        elif mode == "rect":
            window_sec = kwargs.get("window_sec", 0.1)
            window_samp = max(1, int(round(window_sec * self.samplerate)))
            orig_shape = self._data.shape
            self._data = uniform_filter1d(
                self._data.astype(np.float64).reshape(orig_shape[0], -1),
                size=window_samp, axis=0
            ).reshape(orig_shape)

        else:
            raise ValueError(f"Unknown filter mode: {mode!r}")

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
