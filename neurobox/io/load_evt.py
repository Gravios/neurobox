"""
load_evt.py
===========
Load Neurosuite ``.evt`` event files.

File format
-----------
Plain text, one event per line::

    <timestamp_ms>    <event_label>

Fields are separated by one or more whitespace characters.  The
timestamp is a floating-point value in **milliseconds** (the Neurosuite
convention).  Event labels are arbitrary strings (e.g. ``"Stim on"``,
``"Trig 1"``).

Lines beginning with ``#`` are treated as comments and ignored.

Empty or malformed lines are silently skipped.

Example
-------
::

    123.45    Stim on
    456.78    Stim off
    # This is a comment
    789.00    Reward

"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_evt(
    evt_file: str | Path,
    pattern: str | None = None,
    as_seconds: bool = False,
) -> tuple[np.ndarray, list[str]]:
    """Load a Neurosuite event file.

    Parameters
    ----------
    evt_file:
        Path to the ``.evt`` file.
    pattern:
        When given, only events whose label **contains** this string
        (case-sensitive) are returned.
    as_seconds:
        When *True*, convert timestamps from milliseconds to seconds.

    Returns
    -------
    timestamps : np.ndarray, shape (n_events,), float64
        Event times in milliseconds (or seconds if *as_seconds=True*).
    labels : list[str]
        Event labels parallel to *timestamps*.

    Raises
    ------
    FileNotFoundError
        When *evt_file* does not exist.
    """
    evt_file = Path(evt_file)
    if not evt_file.exists():
        raise FileNotFoundError(f"Event file not found: {evt_file}")

    timestamps: list[float] = []
    labels: list[str] = []

    with open(evt_file, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(None, 1)   # split on first whitespace run
            if len(parts) < 2:
                continue
            try:
                t = float(parts[0])
            except ValueError:
                continue
            label = parts[1].strip()
            if pattern is not None and pattern not in label:
                continue
            timestamps.append(t)
            labels.append(label)

    ts = np.array(timestamps, dtype=np.float64)
    if as_seconds:
        ts = ts / 1000.0

    return ts, labels


def evt_to_periods(
    timestamps: np.ndarray,
    labels: list[str],
    start_pattern: str,
    stop_pattern: str,
) -> np.ndarray:
    """Build a paired ``(start, stop)`` period array from event labels.

    Matches the first *start_pattern* event, then the first subsequent
    *stop_pattern* event, and so on.  Unpaired events at the end are
    discarded.

    Parameters
    ----------
    timestamps:
        Event times (any unit — preserved as-is).
    labels:
        Event labels parallel to *timestamps*.
    start_pattern:
        Substring marking a period start.
    stop_pattern:
        Substring marking a period end.

    Returns
    -------
    periods : np.ndarray, shape (n_periods, 2)
        Each row is ``[start_time, stop_time]``.
    """
    starts = timestamps[np.array([start_pattern in l for l in labels])]
    stops  = timestamps[np.array([stop_pattern  in l for l in labels])]

    pairs: list[tuple[float, float]] = []
    si = 0
    for t_start in starts:
        # Find first stop after this start
        while si < len(stops) and stops[si] <= t_start:
            si += 1
        if si >= len(stops):
            break
        pairs.append((t_start, stops[si]))
        si += 1

    return np.array(pairs, dtype=np.float64).reshape(-1, 2)
