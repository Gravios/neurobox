"""
neurobox.dtype.sync
=====================
StreamSync and TrialWindow — the multi-segment sync system that replaces
the recursive ``MTADepoch.sync`` chain from MATLAB MTA.

Why two types
-------------
The lab's recording setup has a single trusted master clock (e.g.
Neuralynx LFP @ 1250 Hz) and one or more secondary streams (Vicon
motion capture at 119.881 Hz, video at 25 Hz, etc.) that record
shorter chunks delimited by TTL pulses on the master clock.  A
single "session" spans the entire master recording; a "trial" is a
sub-window of master time where some experimental condition held.

Two questions need to be answered, and they're separable:

1. **When was each stream recording, in master-clock seconds?**
   Answered by :class:`StreamSync`.  Lives on each :class:`NBData`.
   Multi-segment to support stop/start workflows used to bound
   clock-drift error.

2. **What sub-window of master time is this trial?**  Answered by
   :class:`TrialWindow`.  Lives on :class:`NBSession` /
   :class:`NBTrial`.  Independent of which streams recorded — a
   trial is a window, not a stream.

The cross-product is what :meth:`NBData.restrict_to_window` does:
slice the stream's data to (window ∩ stream_segments), zero-filling
gaps inside the window where the stream wasn't recording (matching
MATLAB ``MTAData.resync`` semantics).

What this replaces
------------------
The MATLAB ``MTAData.sync`` was an ``MTADepoch`` whose own ``.sync``
was recursively another ``MTADepoch``, with a separate scalar
``origin`` field.  This expressed the same information as
``StreamSync`` + ``TrialWindow`` but obscured it behind a
parent-pointer chain that had no well-defined arithmetic.  The
two-type design here makes every conversion explicit:

* ``StreamSync.local_to_master(local_idx) → master_seconds``
* ``StreamSync.master_to_local(master_t)  → local_idx | None``
* ``StreamSync.slice_for_window(t0, t1)   → list[(local_lo, local_hi)]``
* ``StreamSync.restricted_to_window(t0, t1) → StreamSync``

All conversions go through master-clock seconds; no recursive
walk required.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


__all__ = [
    "StreamSync",
    "TrialWindow",
]


# ─────────────────────────────────────────────────────────────────────── #
# StreamSync                                                                  #
# ─────────────────────────────────────────────────────────────────────── #

@dataclass
class StreamSync:
    """Describes when a recording stream was active on the master clock.

    Attributes
    ----------
    segments : np.ndarray, shape ``(n_segments, 2)``, float64 seconds
        Master-clock periods when this stream was actively recording.
        Multiple rows = disjoint segments (e.g. three Vicon takes
        separated by gaps where Vicon was off but NLX kept running).
        Always sorted by start time.
    samplerate : float
        Native sample rate of THIS stream.  Used for converting
        between master-clock seconds and local sample indices.

    Data layout convention
    ----------------------
    The :class:`NBData.data` array of a stream is one contiguous
    block: sample 0 corresponds to the first sample of segment 0,
    samples roll continuously through all segments with **no gaps in
    the array**.  Gaps in master-clock time become invisible at the
    array level — :meth:`local_to_master` is the only way to recover
    the master-clock alignment.

    Worked example
    --------------
    Vicon @ 120 Hz, three takes during one NLX recording::

        seg 0: master time [10.0, 30.0]  →  local samples 0..2399
        seg 1: master time [50.0, 70.0]  →  local samples 2400..4799
        seg 2: master time [90.0, 100.0] →  local samples 4800..5999
        total local samples = 6000

    Then::

        sync.local_to_master(0)     == 10.0
        sync.local_to_master(2400)  == 50.0   (first sample of seg 1)
        sync.local_to_master(5999)  ≈ 99.992  (last sample of seg 2)

        sync.master_to_local(25.0)  == 1800   (15 s into seg 0 × 120)
        sync.master_to_local(40.0)  is None   (in a gap)
        sync.master_to_local(60.0)  == 3600   (10 s into seg 1)
    """
    segments:   np.ndarray
    samplerate: float

    # ── Construction / normalisation ────────────────────────────── #

    def __post_init__(self) -> None:
        seg = np.asarray(self.segments, dtype=np.float64)
        if seg.ndim == 1 and seg.size == 2:
            seg = seg.reshape(1, 2)
        if seg.size == 0:
            seg = np.zeros((0, 2), dtype=np.float64)
        elif seg.ndim != 2 or seg.shape[1] != 2:
            raise ValueError(
                f"segments must be (n_segments, 2); got shape {seg.shape}"
            )
        if seg.shape[0] > 0 and (seg[:, 1] <= seg[:, 0]).any():
            bad = seg[seg[:, 1] <= seg[:, 0]]
            raise ValueError(
                f"segments must have strictly stop > start; got {bad.tolist()}"
            )
        # Sort by start time
        if seg.shape[0] > 1:
            seg = seg[np.argsort(seg[:, 0])]
            # Reject overlapping segments (gaps must be non-negative)
            for i in range(1, seg.shape[0]):
                if seg[i, 0] < seg[i - 1, 1]:
                    raise ValueError(
                        f"segments overlap: {seg[i-1].tolist()} and "
                        f"{seg[i].tolist()}"
                    )
        self.segments = seg
        self.samplerate = float(self.samplerate)
        if self.samplerate <= 0:
            raise ValueError(f"samplerate must be positive; got {self.samplerate}")

    # ── Properties ──────────────────────────────────────────────── #

    @property
    def n_segments(self) -> int:
        return int(self.segments.shape[0])

    @property
    def is_empty(self) -> bool:
        return self.n_segments == 0

    @property
    def total_samples(self) -> int:
        """Total number of stream samples across all segments."""
        if self.is_empty:
            return 0
        durations = self.segments[:, 1] - self.segments[:, 0]
        return int(np.round((durations * self.samplerate).sum()))

    @property
    def cumulative_samples(self) -> np.ndarray:
        """Sample indices at the start of each segment, shape ``(n+1,)``.

        ``cum[i]`` = total samples in segments ``[0..i)``.  ``cum[-1]``
        equals :attr:`total_samples`.
        """
        if self.is_empty:
            return np.array([0], dtype=np.int64)
        durations     = self.segments[:, 1] - self.segments[:, 0]
        sample_counts = np.round(durations * self.samplerate).astype(np.int64)
        return np.concatenate([[0], np.cumsum(sample_counts)])

    @property
    def master_first(self) -> float:
        """Master-clock second when this stream first started recording."""
        if self.is_empty:
            return float("nan")
        return float(self.segments[0, 0])

    @property
    def master_last(self) -> float:
        """Master-clock second when this stream finished its last segment."""
        if self.is_empty:
            return float("nan")
        return float(self.segments[-1, 1])

    @property
    def master_span(self) -> float:
        """``master_last - master_first`` (includes gaps).  NaN if empty."""
        if self.is_empty:
            return float("nan")
        return self.master_last - self.master_first

    @property
    def gaps(self) -> np.ndarray:
        """Master-clock gaps between segments, shape ``(n_segments-1, 2)``.

        Empty if there are no gaps (single-segment streams).
        """
        if self.n_segments < 2:
            return np.zeros((0, 2), dtype=np.float64)
        return np.column_stack([
            self.segments[:-1, 1],
            self.segments[1:, 0],
        ])

    # ── Conversions ─────────────────────────────────────────────── #

    def local_to_master(self, local_idx: int) -> float:
        """Convert a local sample index to master-clock seconds.

        Raises
        ------
        IndexError
            If ``local_idx`` is outside ``[0, total_samples)``.
        """
        cum = self.cumulative_samples
        total = int(cum[-1])
        if not 0 <= local_idx < total:
            raise IndexError(
                f"local_idx {local_idx} out of range [0, {total})"
            )
        # Find which segment contains local_idx — searchsorted on the
        # cumulative-sample array.
        seg_i = int(np.searchsorted(cum[1:], local_idx, side="right"))
        offset_samples = local_idx - cum[seg_i]
        return float(
            self.segments[seg_i, 0] + offset_samples / self.samplerate
        )

    def master_to_local(self, t: float) -> Optional[int]:
        """Convert a master-clock second to local sample index.

        Returns
        -------
        int or None
            ``None`` if *t* falls outside any of this stream's
            recording segments (including pre-stream, post-stream, and
            inter-segment gaps).
        """
        cum = self.cumulative_samples
        for i in range(self.n_segments):
            seg_t0 = self.segments[i, 0]
            seg_t1 = self.segments[i, 1]
            if seg_t0 <= t < seg_t1:
                offset_samples = (t - seg_t0) * self.samplerate
                return int(cum[i] + np.round(offset_samples))
        return None

    def slice_for_window(
        self, t0: float, t1: float,
    ) -> list[tuple[int, int]]:
        """Local-sample ``(lo, hi)`` ranges that fall in master-time
        ``[t0, t1)``.

        For each segment overlapping the window, returns one tuple
        ``(local_lo, local_hi)`` with the local sample indices that
        contribute.  ``local_hi`` is exclusive (Python half-open
        convention).

        Use this for direct data slicing::

            for lo, hi in sync.slice_for_window(t0, t1):
                buffer.append(stream_data[lo:hi])
        """
        result: list[tuple[int, int]] = []
        cum = self.cumulative_samples
        for i in range(self.n_segments):
            seg_t0 = self.segments[i, 0]
            seg_t1 = self.segments[i, 1]
            overlap_t0 = max(seg_t0, t0)
            overlap_t1 = min(seg_t1, t1)
            if overlap_t0 < overlap_t1:
                lo = int(cum[i] + np.round(
                    (overlap_t0 - seg_t0) * self.samplerate
                ))
                hi = int(cum[i] + np.round(
                    (overlap_t1 - seg_t0) * self.samplerate
                ))
                if hi > lo:
                    result.append((lo, hi))
        return result

    def restricted_to_window(
        self, t0: float, t1: float,
    ) -> "StreamSync":
        """Return a new StreamSync with segments clipped to ``[t0, t1)``."""
        new_segs: list[list[float]] = []
        for seg_t0, seg_t1 in self.segments:
            overlap_t0 = max(float(seg_t0), float(t0))
            overlap_t1 = min(float(seg_t1), float(t1))
            if overlap_t0 < overlap_t1:
                new_segs.append([overlap_t0, overlap_t1])
        return StreamSync(
            segments   = (np.asarray(new_segs, dtype=np.float64)
                            if new_segs else np.zeros((0, 2))),
            samplerate = self.samplerate,
        )

    def valid_mask_in_window(
        self, t0: float, t1: float,
    ) -> np.ndarray:
        """Boolean mask of length ``round((t1 - t0) * samplerate)``.

        ``True`` where this stream was actively recording inside
        ``[t0, t1)``.  Useful for callers that zero-fill gaps and want
        to know which output samples are real vs. gap-fill.
        """
        n_out = int(np.round((t1 - t0) * self.samplerate))
        mask = np.zeros(n_out, dtype=bool)
        for seg_t0, seg_t1 in self.segments:
            overlap_t0 = max(seg_t0, t0)
            overlap_t1 = min(seg_t1, t1)
            if overlap_t0 < overlap_t1:
                lo = int(np.round((overlap_t0 - t0) * self.samplerate))
                hi = int(np.round((overlap_t1 - t0) * self.samplerate))
                lo = max(0, lo); hi = min(n_out, hi)
                if hi > lo:
                    mask[lo:hi] = True
        return mask

    def copy(self) -> "StreamSync":
        return StreamSync(
            segments   = self.segments.copy(),
            samplerate = self.samplerate,
        )

    # ── Constructors ───────────────────────────────────────────── #

    @classmethod
    def continuous(
        cls,
        duration_sec: float,
        samplerate:   float,
        t_start:      float = 0.0,
    ) -> "StreamSync":
        """A single-segment sync covering ``[t_start, t_start + duration)``.

        The default for streams that were recording the whole session
        (typically the master clock itself, e.g. LFP).
        """
        return cls(
            segments   = np.array([[float(t_start),
                                       float(t_start) + float(duration_sec)]]),
            samplerate = samplerate,
        )

    @classmethod
    def from_ttl_pulses(
        cls,
        starts:      np.ndarray,
        stops:       np.ndarray,
        samplerate:  float,
    ) -> "StreamSync":
        """Construct from event-file TTL start/stop pulse times.

        Mirrors the role of ``vstarts`` / ``vstops`` in
        :file:`utilities/sync/sync_nlx_vicon.m` lines 100-115.
        Pulse times are in master-clock seconds; lengths must match.
        """
        starts = np.asarray(starts, dtype=np.float64).ravel()
        stops  = np.asarray(stops,  dtype=np.float64).ravel()
        if starts.size != stops.size:
            raise ValueError(
                f"starts and stops must have same length; "
                f"got {starts.size} and {stops.size}"
            )
        return cls(
            segments   = np.column_stack([starts, stops]),
            samplerate = samplerate,
        )


# ─────────────────────────────────────────────────────────────────────── #
# TrialWindow                                                                 #
# ─────────────────────────────────────────────────────────────────────── #

@dataclass
class TrialWindow:
    """A sub-window of the master clock identifying one trial.

    Attributes
    ----------
    periods : np.ndarray, shape ``(n_periods, 2)``, master-clock seconds
        Time ranges constituting this trial.  Usually a single row
        ``[trial_start, trial_stop]`` for a contiguous trial; multiple
        rows for trials that exclude rest periods or stitched epochs.
    label : str
        Human-readable label (e.g. ``'task1'``).
    name : str
        Trial name as it would appear in a .trl filename
        (e.g. ``'all'``, ``'task1'``).
    """
    periods: np.ndarray
    label:   str = ""
    name:    str = ""

    def __post_init__(self) -> None:
        p = np.asarray(self.periods, dtype=np.float64)
        if p.ndim == 1 and p.size == 2:
            p = p.reshape(1, 2)
        if p.size == 0:
            p = np.zeros((0, 2), dtype=np.float64)
        elif p.ndim != 2 or p.shape[1] != 2:
            raise ValueError(
                f"periods must be (n_periods, 2); got shape {p.shape}"
            )
        if p.shape[0] > 0 and (p[:, 1] <= p[:, 0]).any():
            raise ValueError(
                "all periods must have stop > start"
            )
        self.periods = p

    @property
    def is_empty(self) -> bool:
        return self.periods.shape[0] == 0

    @property
    def t_start(self) -> float:
        """Master time at the start of the first period."""
        return float(self.periods[0, 0]) if not self.is_empty else float("nan")

    @property
    def t_stop(self) -> float:
        """Master time at the end of the last period."""
        return float(self.periods[-1, 1]) if not self.is_empty else float("nan")

    @property
    def total_duration(self) -> float:
        """Sum of all period lengths.  Different from
        ``t_stop - t_start`` when there are multiple periods with gaps."""
        if self.is_empty:
            return 0.0
        return float((self.periods[:, 1] - self.periods[:, 0]).sum())

    def contains(self, t: float) -> bool:
        """True iff *t* is inside any period."""
        if self.is_empty:
            return False
        return bool(((self.periods[:, 0] <= t)
                       & (t < self.periods[:, 1])).any())

    def copy(self) -> "TrialWindow":
        return TrialWindow(
            periods = self.periods.copy(),
            label   = self.label,
            name    = self.name,
        )

    @classmethod
    def whole_session(
        cls,
        sync:  StreamSync,
        label: str = "all",
        name:  str = "all",
    ) -> "TrialWindow":
        """A trial covering the entire span of *sync* (from first segment
        start to last segment stop).  This is the canonical 'all' trial.
        """
        if sync.is_empty:
            return cls(periods=np.zeros((0, 2)), label=label, name=name)
        return cls(
            periods = np.array([[sync.master_first, sync.master_last]]),
            label   = label,
            name    = name,
        )
