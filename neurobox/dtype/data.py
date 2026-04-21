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

        If the first index element is an ``(N, 2)`` array of
        ``[start_sec, stop_sec]`` periods, the corresponding segments
        are extracted and concatenated along axis 0 (equivalent to
        MTA's ``SelectPeriods``).

        For everything else, the index is passed directly to the numpy
        array.

        Examples
        --------
        >>> data_in_walk = obj[stc['walk'].data]          # period-select
        >>> channel_3    = obj[:, 2]                      # normal numpy
        >>> obj[np.array([[1.0, 2.0], [5.0, 6.0]])]      # two windows
        """
        if self._data is None:
            raise IndexError("Data has not been loaded yet.")

        if isinstance(idx, tuple):
            first = idx[0]
        else:
            first = idx

        # Period-based selection on first axis
        if (isinstance(first, np.ndarray)
                and first.ndim == 2 and first.shape[1] == 2):
            selected = select_periods(self._data, first, self.samplerate)
            if isinstance(idx, tuple) and len(idx) > 1:
                return selected[(slice(None),) + idx[1:]]
            return selected

        # Normal numpy indexing
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

    def resample(self, new_samplerate: float,
                 method: str = "spline") -> "NBData":
        """Resample the time series to *new_samplerate* Hz.

        Parameters
        ----------
        new_samplerate:
            Target sample rate in Hz.
        method:
            ``'spline'`` (default) uses ``scipy.interpolate.interp1d``
            with cubic spline.  ``'poly'`` uses
            ``scipy.signal.resample_poly`` which is more efficient for
            large factors.
        """
        if self._data is None or new_samplerate == self.samplerate:
            return self

        if self.type_ == "TimeSeries":
            n_in  = self._data.shape[0]
            n_out = int(round(n_in / self.samplerate * new_samplerate))
            orig_shape = self._data.shape
            d = self._data.astype(np.float64).reshape(n_in, -1)

            if method == "poly":
                from math import gcd
                from scipy.signal import resample_poly
                g  = gcd(int(new_samplerate * 100), int(self.samplerate * 100))
                up = int(new_samplerate * 100 // g)
                dn = int(self.samplerate  * 100 // g)
                d  = resample_poly(d, up, dn, axis=0)
                d  = d[:n_out]
            else:
                from scipy.interpolate import interp1d
                t_in  = np.linspace(0, 1, n_in)
                t_out = np.linspace(0, 1, n_out)
                f     = interp1d(t_in, d, kind="cubic", axis=0,
                                 fill_value="extrapolate")
                d     = f(t_out)

            new_shape = (n_out,) + orig_shape[1:]
            self._data = d[:n_out].reshape(new_shape)
            self.samplerate = new_samplerate

        elif self.type_ == "TimePeriods":
            # For period data, rescale the indices
            self._data = self._data * new_samplerate / self.samplerate
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
