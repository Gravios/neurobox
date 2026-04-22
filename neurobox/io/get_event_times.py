"""
get_event_times.py
==================
Convenience wrappers for extracting TTL event timestamps from a
Neuralynx ``.all.evt`` file, given a session object or a file path.

These are the Python equivalents of MTA ``sync_nlx_events`` and the
inline event-parsing idioms used throughout the MTA sync scripts.

Functions
---------
get_event_times(session_or_path, ttl_label)
    Return timestamps (seconds) for all events whose label contains
    *ttl_label*.

get_ttl_periods(session_or_path, start_ttl, stop_ttl)
    Return (N, 2) float64 array of [start_sec, stop_sec] pairs by
    pairing each start TTL with the next stop TTL.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _resolve_evt_path(session_or_path) -> Path:
    """Return the .all.evt file path from a session object or a string/Path."""
    if isinstance(session_or_path, (str, Path)):
        p = Path(session_or_path)
        if p.suffix == ".evt":
            return p
        # Treat as session base path: add .all.evt suffix
        return p.with_suffix("").parent / (p.stem + ".all.evt")

    # NBSession / NBTrial / SimpleNamespace with spath + name
    session = session_or_path
    name  = getattr(session, "name",  None)
    spath = getattr(session, "spath", None)

    if spath is None or name is None:
        raise ValueError(
            "session must have .spath and .name attributes, "
            "or pass a direct file path string."
        )

    # Try spath first (symlinks), then processed_ephys
    candidates = [Path(spath) / f"{name}.all.evt"]
    paths_obj = getattr(session, "paths", None)
    if paths_obj is not None:
        candidates.append(paths_obj.processed_ephys / f"{name}.all.evt")

    for c in candidates:
        if c.exists():
            return c

    raise FileNotFoundError(
        f"Event file '{name}.all.evt' not found in {spath}"
        + (f" or {paths_obj.processed_ephys}" if paths_obj else "")
    )


def get_event_times(
    session_or_path,
    ttl_label: str,
    as_seconds: bool = True,
) -> np.ndarray:
    """Return timestamps for events whose label contains *ttl_label*.

    Mirrors the MTA pattern::

        evts = LoadEvents(fullfile(Session.spath, [Session.name '.all.evt']));
        eClu = find(~cellfun(@isempty, regexp(evts.Labels, TTLValue)));
        eTime = evts.time(evts.Clu == eClu);

    Parameters
    ----------
    session_or_path:
        An ``NBSession``-like object (with ``.spath`` and ``.name``), or
        a path string to the ``.all.evt`` file / session base.
    ttl_label:
        Substring to match in event labels (e.g. ``'0x0040'``,
        ``'Stim on'``).
    as_seconds:
        Return times in seconds (default True).  When False, return
        the raw millisecond values as stored in the file.

    Returns
    -------
    times : np.ndarray, shape (N,), float64
    """
    from neurobox.io.load_evt import load_evt

    evt_file = _resolve_evt_path(session_or_path)
    ts_ms, labels = load_evt(str(evt_file), as_seconds=False)

    mask = np.array([ttl_label in lbl for lbl in labels], dtype=bool)
    ts   = ts_ms[mask]
    return (ts / 1000.0) if as_seconds else ts


def get_ttl_periods(
    session_or_path,
    start_ttl: str = "0x0040",
    stop_ttl:  str = "0x0000",
) -> np.ndarray:
    """Return (N, 2) float64 array of [start_sec, stop_sec] TTL pairs.

    Parameters
    ----------
    session_or_path:
        NBSession-like or path to ``.all.evt`` file.
    start_ttl:
        TTL label marking recording start (default ``'0x0040'``).
    stop_ttl:
        TTL label marking recording stop  (default ``'0x0000'``).

    Returns
    -------
    periods : np.ndarray, shape (N, 2), float64
        Start/stop pairs in seconds.  Empty array if none found.

    Examples
    --------
    >>> periods = get_ttl_periods(session, '0x0040', '0x0000')
    >>> print(f"{len(periods)} recording blocks, "
    ...       f"total {np.diff(periods).sum():.1f} s")
    """
    from neurobox.io.load_evt import load_evt
    from neurobox.dtype.sync_pipelines import _find_ttl_windows

    evt_file = _resolve_evt_path(session_or_path)
    ts_ms, labels = load_evt(str(evt_file), as_seconds=False)
    ts_sec = ts_ms / 1000.0

    return _find_ttl_windows(ts_sec, labels, start_ttl, stop_ttl)
