"""
epoch.py  —  NBEpoch
====================
Port of MTADepoch.  An NBEpoch is a collection of time periods,
represented either as an ``(N, 2)`` float64 array of
``[start_sec, stop_sec]`` pairs (mode='periods') or as a boolean
``(T,)`` mask array sampled at *samplerate* Hz (mode='mask').

Design differences from MTADepoch
----------------------------------
* **Data stored in seconds** (float64), not sample indices.
  ``samplerate`` is only needed for mask conversion and clamping.
* **Two explicit modes** ('periods' / 'mask') instead of the
  'TimePeriods' / 'TimeSeries' string enum.
* **Arithmetic operators** accept both seconds (floats) and other
  NBEpoch objects.
* **``to_mask`` / ``to_periods``** for explicit conversion.

Sync
----
Every NBEpoch has an optional ``sync`` attribute that is itself a
plain ``(2,)`` float64 array ``[rec_start_sec, rec_stop_sec]``
representing the recording window.  When ``sync`` is set, ``clean()``
will drop / truncate periods that fall outside it.
"""

from __future__ import annotations

import re
from typing import Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _intersect_periods(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return the intersection of two sorted (N,2) period arrays."""
    if a.size == 0 or b.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    out: list[list[float]] = []
    j = 0
    for s, e in a:
        while j < len(b) and b[j, 1] <= s:
            j += 1
        k = j
        while k < len(b) and b[k, 0] < e:
            lo = max(s, b[k, 0])
            hi = min(e, b[k, 1])
            if hi > lo:
                out.append([lo, hi])
            k += 1
    return np.array(out, dtype=np.float64).reshape(-1, 2)


def _union_periods(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return the union of two sorted (N,2) period arrays."""
    if a.size == 0:
        return b
    if b.size == 0:
        return a
    combined = np.concatenate([a, b], axis=0)
    combined = combined[combined[:, 0].argsort()]
    merged = [combined[0].tolist()]
    for s, e in combined[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return np.array(merged, dtype=np.float64)


def _subtract_periods(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return a minus b (set difference on periods)."""
    if a.size == 0:
        return a
    if b.size == 0:
        return a
    result: list[list[float]] = []
    for s, e in a:
        current = [[s, e]]
        for bs, be in b:
            next_segs: list[list[float]] = []
            for cs, ce in current:
                if be <= cs or bs >= ce:
                    next_segs.append([cs, ce])
                else:
                    if cs < bs:
                        next_segs.append([cs, bs])
                    if be < ce:
                        next_segs.append([be, ce])
            current = next_segs
        result.extend(current)
    return np.array(result, dtype=np.float64).reshape(-1, 2)


# ---------------------------------------------------------------------------
# NBEpoch
# ---------------------------------------------------------------------------

class NBEpoch:
    """Collection of time epochs.

    Parameters
    ----------
    data:
        ``(N, 2)`` float64 array of ``[start_sec, stop_sec]`` pairs
        (mode='periods'), or a boolean ``(T,)`` array (mode='mask').
    samplerate:
        Samples per second.  Required for mask operations.
    sync:
        ``(2,)`` array ``[rec_start_sec, rec_stop_sec]`` defining the
        valid recording window.  Used by ``clean()`` to truncate / drop
        out-of-range periods.
    label:
        Human-readable name (e.g. 'walk', 'theta').
    key:
        Single-character shorthand for keyboard / indexing shortcuts.
    mode:
        ``'periods'`` (default) or ``'mask'``.
    """

    def __init__(
        self,
        data: np.ndarray | None = None,
        samplerate: float = 1.0,
        sync: np.ndarray | None = None,
        label: str = "",
        key: str = "",
        mode: str = "periods",
    ) -> None:
        self.samplerate: float = float(samplerate)
        self.sync: np.ndarray | None = (
            np.asarray(sync, dtype=np.float64) if sync is not None else None
        )
        self.label: str = label
        self.key: str = key
        self.mode: str = mode  # 'periods' or 'mask'

        if data is None:
            self.data: np.ndarray = np.empty((0, 2), dtype=np.float64)
        else:
            self.data = np.asarray(data, dtype=np.float64)
            if mode == "periods" and self.data.ndim == 1 and len(self.data) == 2:
                self.data = self.data.reshape(1, 2)

    # ------------------------------------------------------------------ #
    # Representation                                                       #
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        n = len(self.data)
        if self.mode == "periods":
            dur = float(np.diff(self.data, axis=1).sum()) if n > 0 else 0.0
            return (f"NBEpoch(label={self.label!r}, n_periods={n}, "
                    f"total_dur={dur:.2f}s, sr={self.samplerate}Hz)")
        return (f"NBEpoch(label={self.label!r}, mode='mask', "
                f"n_samples={len(self.data)}, sr={self.samplerate}Hz)")

    def __len__(self) -> int:
        return len(self.data)

    def __bool__(self) -> bool:
        return len(self.data) > 0

    def isempty(self) -> bool:
        return len(self.data) == 0

    # ------------------------------------------------------------------ #
    # Conversion                                                           #
    # ------------------------------------------------------------------ #

    def to_mask(self, n_samples: int) -> np.ndarray:
        """Return a boolean mask of length *n_samples*.

        Parameters
        ----------
        n_samples:
            Length of the output mask (= recording length in samples at
            *self.samplerate*).
        """
        mask = np.zeros(n_samples, dtype=bool)
        if self.mode == "mask":
            n = min(n_samples, len(self.data))
            mask[:n] = self.data[:n].astype(bool)
            return mask
        for s, e in self.data:
            i0 = max(0, int(np.round(s * self.samplerate)))
            i1 = min(n_samples, int(np.round(e * self.samplerate)))
            if i1 > i0:
                mask[i0:i1] = True
        return mask

    def to_periods(self) -> "NBEpoch":
        """Convert mask → periods.  No-op if already 'periods'."""
        if self.mode == "periods":
            return self.copy()
        mask = self.data.astype(bool)
        edges = np.diff(np.concatenate([[False], mask, [False]]).astype(np.int8))
        starts = np.where(edges == 1)[0]
        stops  = np.where(edges == -1)[0]
        periods = np.column_stack([starts, stops]).astype(np.float64) / self.samplerate
        return NBEpoch(periods, samplerate=self.samplerate,
                       sync=self.sync, label=self.label, key=self.key,
                       mode="periods")

    def to_mask_epoch(self, n_samples: int) -> "NBEpoch":
        """Return a new NBEpoch in mask mode."""
        mask = self.to_mask(n_samples)
        return NBEpoch(mask, samplerate=self.samplerate,
                       sync=self.sync, label=self.label, key=self.key,
                       mode="mask")

    # ------------------------------------------------------------------ #
    # Duration / coverage                                                  #
    # ------------------------------------------------------------------ #

    @property
    def duration(self) -> float:
        """Total duration in seconds covered by all periods."""
        if self.mode == "periods":
            if self.isempty():
                return 0.0
            return float(np.diff(self.data, axis=1).clip(min=0).sum())
        return float(self.data.astype(bool).sum()) / self.samplerate

    @property
    def n_periods(self) -> int:
        if self.mode == "periods":
            return len(self.data)
        ep = self.to_periods()
        return len(ep.data)

    # ------------------------------------------------------------------ #
    # Arithmetic operators                                                 #
    # ------------------------------------------------------------------ #

    def __and__(self, other: "NBEpoch | np.ndarray") -> "NBEpoch":
        """Intersection.  ``epoch & other_epoch``"""
        a = self._as_periods()
        b = self._coerce(other)._as_periods()
        new_data = _intersect_periods(a, b)
        return NBEpoch(new_data, samplerate=self.samplerate,
                       sync=self.sync,
                       label=f"{self.label}&{getattr(other,'label','')}",
                       key="c", mode="periods")

    def __or__(self, other: "NBEpoch | np.ndarray") -> "NBEpoch":
        """Union.  ``epoch | other_epoch``"""
        a = self._as_periods()
        b = self._coerce(other)._as_periods()
        new_data = _union_periods(a, b)
        return NBEpoch(new_data, samplerate=self.samplerate,
                       sync=self.sync,
                       label=f"{self.label}|{getattr(other,'label','')}",
                       key="c", mode="periods")

    def __add__(self, other: "NBEpoch | float | np.ndarray") -> "NBEpoch":
        """Expand / shift periods.

        ``epoch + [dt_start, dt_end]``  — expand each period by dt_start
        on the left and dt_end on the right (both in seconds).

        ``epoch + other_epoch``  — union of two epoch collections.
        """
        result = self.copy()
        if isinstance(other, NBEpoch):
            return self | other
        b = np.asarray(other, dtype=np.float64)
        if result.mode == "periods":
            if b.shape == () or b.shape == (1,):
                result.data = result.data + float(b)
            else:
                result.data = result.data + b
        result.clean()
        return result

    def __sub__(self, other: "NBEpoch | float | np.ndarray") -> "NBEpoch":
        """Set difference or shrink periods.

        ``epoch - other_epoch``  — remove other_epoch's intervals.
        ``epoch - [dt_start, dt_end]``  — shrink each period.
        """
        result = self.copy()
        if isinstance(other, NBEpoch):
            b = other._as_periods()
            result.data = _subtract_periods(self._as_periods(), b)
        else:
            b = np.asarray(other, dtype=np.float64)
            if result.mode == "periods":
                result.data = result.data - b
        result.clean()
        return result

    # ------------------------------------------------------------------ #
    # Clean / fill                                                         #
    # ------------------------------------------------------------------ #

    def clean(self) -> "NBEpoch":
        """Remove zero/negative-duration periods, sort, merge overlaps,
        and optionally clamp to ``self.sync``."""
        if self.mode != "periods":
            return self
        if self.isempty():
            return self

        # Drop zero/negative duration
        dur = self.data[:, 1] - self.data[:, 0]
        self.data = self.data[dur > 0]

        # Sort by start time
        self.data = self.data[self.data[:, 0].argsort()]

        # Merge overlapping / touching periods
        self.data = _union_periods(self.data, np.empty((0, 2), dtype=np.float64))

        # Clamp to sync window
        if self.sync is not None and len(self.sync) == 2:
            t0, t1 = self.sync
            self.data = self.data[self.data[:, 0] < t1]
            if self.isempty():
                return self
            self.data = self.data[self.data[:, 1] > t0]
            if self.isempty():
                return self
            self.data[:, 0] = np.clip(self.data[:, 0], t0, t1)
            self.data[:, 1] = np.clip(self.data[:, 1], t0, t1)
            dur = self.data[:, 1] - self.data[:, 0]
            self.data = self.data[dur > 0]
        return self

    def fillgaps(self, gap_sec: float = 0.1) -> "NBEpoch":
        """Merge periods separated by gaps smaller than *gap_sec* seconds."""
        if self.mode != "periods" or len(self.data) < 2:
            return self.copy()
        # One-pass merge: build result list, extending last period when gap is small
        merged = [self.data[0].tolist()]
        for s, e in self.data[1:]:
            if s - merged[-1][1] < gap_sec:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        result = self.copy()
        result.data = np.array(merged, dtype=np.float64)
        return result

    # ------------------------------------------------------------------ #
    # Resample (for mask mode or clamping periods to a new sample rate)   #
    # ------------------------------------------------------------------ #

    def resample(self, new_samplerate: float) -> "NBEpoch":
        """Return a copy with samplerate changed.

        For 'periods' mode this is a metadata-only change (the float64
        second values are unchanged).  For 'mask' mode the mask is
        up/downsampled using nearest-neighbour.
        """
        result = self.copy()
        if self.mode == "mask" and new_samplerate != self.samplerate:
            old_n = len(self.data)
            new_n = int(round(old_n / self.samplerate * new_samplerate))
            t_old = np.arange(old_n) / self.samplerate
            t_new = np.arange(new_n) / new_samplerate
            idx   = np.searchsorted(t_old, t_new, side="left").clip(0, old_n - 1)
            result.data = self.data[idx]
        result.samplerate = new_samplerate
        return result

    # ------------------------------------------------------------------ #
    # Copy                                                                 #
    # ------------------------------------------------------------------ #

    def copy(self) -> "NBEpoch":
        return NBEpoch(
            data       = self.data.copy() if self.data is not None else None,
            samplerate = self.samplerate,
            sync       = self.sync.copy() if self.sync is not None else None,
            label      = self.label,
            key        = self.key,
            mode       = self.mode,
        )

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _as_periods(self) -> np.ndarray:
        """Return the period data as an (N,2) float64 array."""
        if self.mode == "periods":
            return self.data
        return self.to_periods().data

    def _coerce(self, other: "NBEpoch | np.ndarray") -> "NBEpoch":
        if isinstance(other, NBEpoch):
            return other
        return NBEpoch(np.asarray(other, dtype=np.float64),
                       samplerate=self.samplerate, mode="periods")

    # ------------------------------------------------------------------ #
    # Static constructors                                                  #
    # ------------------------------------------------------------------ #

    @staticmethod
    def intersect(epochs: list["NBEpoch"]) -> "NBEpoch":
        """Return the intersection of a list of NBEpoch objects."""
        if not epochs:
            return NBEpoch()
        result = epochs[0].copy()
        for ep in epochs[1:]:
            result = result & ep
        return result

    @staticmethod
    def union(epochs: list["NBEpoch"]) -> "NBEpoch":
        """Return the union of a list of NBEpoch objects."""
        if not epochs:
            return NBEpoch()
        result = epochs[0].copy()
        for ep in epochs[1:]:
            result = result | ep
        return result

    @staticmethod
    def from_logical(mask: np.ndarray, samplerate: float,
                     label: str = "", key: str = "") -> "NBEpoch":
        """Create an NBEpoch from a boolean mask array."""
        return NBEpoch(mask.astype(np.float64), samplerate=samplerate,
                       label=label, key=key, mode="mask").to_periods()


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def select_periods(data: np.ndarray, periods: np.ndarray,
                   samplerate: float) -> np.ndarray:
    """Extract and concatenate rows of *data* that fall within *periods*.

    This is the Python equivalent of MTA's ``SelectPeriods``.

    Parameters
    ----------
    data:
        Time-series array, shape ``(T, ...)``.  Time is axis 0.
    periods:
        ``(N, 2)`` float64 array of ``[start_sec, stop_sec]`` or
        integer sample indices when *samplerate* == 1.
    samplerate:
        Sample rate of *data*.

    Returns
    -------
    out : np.ndarray
        Concatenated segments along axis 0.
    """
    segments: list[np.ndarray] = []
    for s, e in periods:
        i0 = int(np.round(s * samplerate))
        i1 = int(np.round(e * samplerate))
        i0 = max(0, i0)
        i1 = min(len(data), i1)
        if i1 > i0:
            segments.append(data[i0:i1])
    return np.concatenate(segments, axis=0) if segments else np.empty_like(data[:0])
