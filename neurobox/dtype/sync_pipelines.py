"""
sync_pipelines.py
=================
Synchronisation pipeline functions.  Naming follows the MTA convention:

    sync_<primary>_<secondary>(session, ...)

Primary  = the electrophysiology recording system (master clock).
Secondary = the behavioural / position tracking system.

Primary systems
---------------
``nlx``        Neuralynx — sync via ``.all.evt`` TTL events
``openephys``  Open Ephys — sync via a dedicated ADC pulse channel in ``.lfp``
``ephys``      Generic ephys (same as nlx, fallback)

Secondary systems
-----------------
``vicon``      Vicon C3D / position files (marker CSV or binary .pos)
``optitrack``  Optitrack Motive CSV exports
``spots``      NLX 2-LED tracker, plain binary ``.pos`` (T×4)
``whl``        Neurosuite ``.whl`` binary (already on-clock, no matching needed)

Architecture
------------
Every sync function follows the same four-phase structure:

  Phase 1 — Establish the primary (ephys) recording window
            ``record_sync = [0.0, n_lfp_samples / lfp_sr]``

  Phase 2 — Find secondary-system start/stop times on the primary timeline
            Event-based (nlx): parse ``.all.evt`` for TTL start/stop pairs
            Pulse-based (openephys): threshold-cross a sync ADC channel

  Phase 3 — Match secondary data chunks to ephys windows by duration
            Tolerance: |ephys_dur − chunk_dur| < 0.2 s

  Phase 4 — Populate session fields
            session.sync   = NBEpoch([first_start, last_stop])
            session.xyz    = NBDxyz with origin = first_start, data on [0, span]
            session.lfp    = NBDlfp shell (lazy)
            session.spk    = NBSpk (loaded from .res/.clu)
            session.stc    = empty NBStateCollection

Key relationships
-----------------
* ``record_sync`` is always [0, lfp_duration_sec] — the full LFP file length
* ``session.sync.data`` is a subset of record_sync:
  [first_secondary_start, last_secondary_stop] in ephys seconds
* ``xyz.origin`` = session.sync.data[0,0] = first secondary block start
* The xyz array spans [0, last_stop - first_start] with zero padding between gaps
* ``xyz.sync`` stores the per-block periods in ephys seconds
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from neurobox.dtype.epoch   import NBEpoch
from neurobox.dtype.xyz     import NBDxyz
from neurobox.dtype.lfp     import NBDlfp
from neurobox.dtype.spikes  import NBSpk
from neurobox.dtype.stc     import NBStateCollection
from neurobox.dtype.model   import NBModel


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _load_record_sync(session) -> tuple[float, float, float]:
    """Load par file and determine the full LFP recording window.

    Returns
    -------
    (record_end_sec, wideband_sr, lfp_sr)

    Sets session.par, session.samplerate.
    """
    from neurobox.io import load_par, load_binary

    spath = session.spath
    name  = session.name

    if session.par is None:
        session.par = load_par(str(spath / name))

    par = session.par
    session.samplerate = float(par.acquisitionSystem.samplingRate)
    lfp_sr = float(getattr(par, "lfpSampleRate",
                            getattr(par, "lfp_sample_rate", 1250.0)) or 1250.0)

    # Load one channel of LFP to get exact sample count
    lfp_file = spath / f"{name}.lfp"
    one_ch = load_binary(lfp_file, channels=[0], par=par, channel_first=True)
    record_end_sec = one_ch.shape[1] / lfp_sr

    return record_end_sec, session.samplerate, lfp_sr


def _thresh_cross(signal: np.ndarray, threshold: float,
                  min_dur_samples: int = 0) -> np.ndarray:
    """Return (N, 2) int64 array of [rise_sample, fall_sample] crossings.

    Port of MTA ThreshCross.  Both indices are 0-based (rise is the first
    sample >= threshold, fall is the first sample < threshold after that).
    """
    above = (signal >= threshold).astype(np.int8)
    d     = np.diff(above)
    rises = (np.where(d ==  1)[0] + 1).tolist()
    falls = (np.where(d == -1)[0] + 1).tolist()

    if not rises or not falls:
        return np.empty((0, 2), dtype=np.int64)

    # Align: first event must be a rise
    while falls and falls[0] <= rises[0]:
        falls.pop(0)
    n = min(len(rises), len(falls))
    rises, falls = rises[:n], falls[:n]

    pairs = np.array(list(zip(rises, falls)), dtype=np.int64)
    if min_dur_samples > 0:
        pairs = pairs[(pairs[:, 1] - pairs[:, 0]) >= min_dur_samples]

    return pairs


def _match_chunks_to_windows(
    chunks: list[np.ndarray],
    windows_sec: np.ndarray,
    chunk_sr: float,
    tolerance_sec: float = 0.2,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Match position/tracking data chunks to ephys time windows by duration.

    Parameters
    ----------
    chunks:
        List of position data arrays, one per recording block.
    windows_sec:
        (N, 2) float64 array of [start, stop] in ephys seconds.
    chunk_sr:
        Frame rate of the tracking data.
    tolerance_sec:
        Maximum allowed |window_duration - chunk_duration| for a match.

    Returns
    -------
    matched_chunks : list[np.ndarray]
    matched_windows : np.ndarray, shape (M, 2)  — subset of windows_sec
    """
    matched_chunks:  list[np.ndarray]  = []
    matched_windows: list[list[float]] = []
    used_windows = set()

    for chunk in chunks:
        if chunk is None or chunk.shape[0] == 0:
            continue
        chunk_dur = chunk.shape[0] / chunk_sr
        for wi, (t0, t1) in enumerate(windows_sec):
            if wi in used_windows:
                continue
            win_dur = t1 - t0
            if abs(win_dur - chunk_dur) < tolerance_sec:
                # Shift the window back by one frame (MTA convention):
                # the recorded TTL fires one frame after capture begins
                frame_dt = 1.0 / chunk_sr
                matched_windows.append([t0 - frame_dt, t1 - frame_dt])
                matched_chunks.append(chunk)
                used_windows.add(wi)
                break

    if not matched_chunks:
        return [], np.empty((0, 2), dtype=np.float64)

    return matched_chunks, np.array(matched_windows, dtype=np.float64)


def _build_xyz_array(
    matched_chunks:  list[np.ndarray],
    matched_windows: np.ndarray,
    xyz_sr: float,
) -> np.ndarray:
    """Concatenate matched chunks into a zero-padded contiguous array.

    The output array starts at t = matched_windows[0, 0] (= xyz.origin).
    Gaps between blocks are zero-filled.  Non-zero values within each
    block are preserved; existing zeros are replaced with eps to
    distinguish them from the padding sentinel.
    """
    t0_global   = matched_windows[0, 0]
    t1_global   = matched_windows[-1, 1]
    total_frames = int(np.ceil((t1_global - t0_global) * xyz_sr))

    ref_shape   = matched_chunks[0].shape
    n_markers   = ref_shape[1]
    n_dims      = ref_shape[2]
    xyz_arr     = np.zeros((total_frames, n_markers, n_dims), dtype=np.float64)

    for (t0, _), chunk in zip(matched_windows, matched_chunks):
        i0 = int(round((t0 - t0_global) * xyz_sr))
        i1 = min(i0 + chunk.shape[0], total_frames)
        n  = i1 - i0
        seg = chunk[:n].copy().astype(np.float64)
        # Sentinel: replace in-chunk zeros with eps (MTA convention)
        seg[seg == 0] = np.finfo(np.float32).eps
        xyz_arr[i0:i1] = seg

    return xyz_arr


def _load_mocap_files(session) -> tuple[list[np.ndarray], list[str], float]:
    """Locate and load position files from the session directory.

    Searches for Motive CSV exports in ``spath/`` and ``spath/{maze}/``.

    Returns
    -------
    chunks    : list of (T, N_markers, 3) arrays, one per block
    markers   : list of marker name strings
    xyz_sr    : tracking frame rate (Hz)
    """
    spath      = session.spath
    maze       = getattr(session, "maze", "")
    search_dirs = [spath, spath / maze] if maze else [spath]

    csv_files: list[Path] = []
    for d in search_dirs:
        if d.exists():
            csv_files.extend(sorted(d.glob("*.csv")))

    if not csv_files:
        raise FileNotFoundError(
            f"No Motive CSV files found in {spath} "
            f"(also searched {spath / maze})"
        )

    chunks:  list[np.ndarray] = []
    markers: list[str]        = []
    xyz_sr:  float            = 0.0

    for csv in csv_files:
        try:
            obj = NBDxyz.from_motive_csv(csv)
        except Exception as e:
            print(f"  [warn] {csv.name}: {e}")
            continue
        if obj.data is None or obj.data.shape[0] == 0:
            continue
        if not markers:
            markers = obj.model.markers
            xyz_sr  = obj.samplerate
        chunks.append(obj.data)

    return chunks, markers, xyz_sr


def _populate_session(
    session,
    xyz_arr:         np.ndarray,
    matched_windows: np.ndarray,
    markers:         list[str],
    xyz_sr:          float,
    record_end_sec:  float,
    wideband_sr:     float,
    lfp_sr:          float,
) -> None:
    """Set all session fields after sync matching is resolved.

    Parameters
    ----------
    xyz_arr:
        Contiguous position array (T, N_markers, D) starting at
        matched_windows[0, 0].
    matched_windows:
        (M, 2) float64 array of [start, stop] in ephys seconds for each block.
    """
    spath    = session.spath
    name     = session.name
    filebase = session.filebase

    # ── session.sync : [first_start, last_stop] ─────────────────────────── #
    first_start = matched_windows[0,  0]
    last_stop   = matched_windows[-1, 1]
    session.sync = NBEpoch(
        np.array([[first_start, last_stop]], dtype=np.float64),
        samplerate = 1.0,
        sync       = np.array([0.0, record_end_sec]),
        label      = "sync",
    )

    # ── xyz.sync : all per-block windows ────────────────────────────────── #
    xyz_sync = NBEpoch(
        matched_windows.copy(),
        samplerate = 1.0,
        sync       = np.array([0.0, record_end_sec]),
        label      = "xyz_sync",
    )

    # ── session.xyz ─────────────────────────────────────────────────────── #
    session.xyz = NBDxyz(
        data       = xyz_arr,
        model      = NBModel(markers=markers),
        samplerate = xyz_sr,
        sync       = xyz_sync,
        origin     = first_start,
        path       = spath,
        filename   = f"{filebase}.pos.npz",
        name       = name,
    )

    # ── session.lfp (lazy shell — data loaded on demand) ─────────────────── #
    lfp_sync = NBEpoch(
        np.array([[0.0, record_end_sec]], dtype=np.float64),
        samplerate = 1.0,
        label      = "lfp_sync",
    )
    session.lfp = NBDlfp(
        path       = spath,
        filename   = f"{name}.lfp",
        samplerate = lfp_sr,
        sync       = lfp_sync,
        origin     = 0.0,
        name       = name,
    )

    # ── session.spk ─────────────────────────────────────────────────────── #
    try:
        session.spk = NBSpk.load(
            str(spath / name),
            samplerate = wideband_sr,
        )
    except Exception as e:
        print(f"  [warn] Could not load spikes: {e}")
        session.spk = NBSpk(samplerate=wideband_sr)

    # ── session.stc ─────────────────────────────────────────────────────── #
    session.stc = NBStateCollection(
        path     = spath,
        filename = f"{filebase}.stc.default.pkl",
        mode     = "default",
        sync     = session.sync,
    )


# ---------------------------------------------------------------------------
# NLX event parsing helpers
# ---------------------------------------------------------------------------

def _load_nlx_events(session) -> tuple[np.ndarray, list[str]]:
    """Load and return (timestamps_sec, labels) from the .all.evt file.

    The Neuralynx .all.evt file contains one event per line.
    ``neurobox.io.load_evt`` handles text parsing.
    """
    from neurobox.io import load_evt

    evt_file = session.spath / f"{session.name}.all.evt"
    if not evt_file.exists():
        raise FileNotFoundError(f"Event file not found: {evt_file}")
    ts_ms, labels = load_evt(str(evt_file), as_seconds=False)   # → ms
    return ts_ms / 1000.0, labels   # → seconds


def _find_ttl_windows(
    ts_sec:    np.ndarray,
    labels:    list[str],
    start_ttl: str = "0x0040",
    stop_ttl:  str = "0x0000",
) -> np.ndarray:
    """Build (N, 2) float64 array of [start_sec, stop_sec] TTL windows.

    Pairs each start event with the next stop event.  Handles count
    mismatches by trimming to the shorter list.
    """
    start_mask = np.array([start_ttl in l for l in labels], dtype=bool)
    stop_mask  = np.array([stop_ttl  in l for l in labels], dtype=bool)
    vstarts    = ts_sec[start_mask]
    vstops     = ts_sec[stop_mask]

    # Drop stop events that precede all starts
    while vstops.size and vstops[0] < vstarts[0]:
        vstops = vstops[1:]

    n = min(len(vstarts), len(vstops))
    vstarts, vstops = vstarts[:n], vstops[:n]

    valid = vstops > vstarts
    return np.column_stack([vstarts[valid], vstops[valid]]).astype(np.float64)


# ---------------------------------------------------------------------------
# sync_nlx_vicon  (NLX primary, Vicon / Optitrack secondary)
# ---------------------------------------------------------------------------

def sync_nlx_vicon(
    session,
    ttl_value:    str   = "0x0040",
    stop_ttl:     str   = "0x0000",
    xyz_samplerate: float | None = None,
) -> None:
    """Synchronise Neuralynx ephys with Vicon / Optitrack position data.

    The primary system is NLX.  Synchronisation windows are extracted
    from TTL events in the ``.all.evt`` file.

    Parameters
    ----------
    session:
        NBSession with spath, name, maze, par (or auto-loaded).
    ttl_value:
        TTL label string marking the *start* of each mocap block
        (default ``'0x0040'``).
    stop_ttl:
        TTL label string marking the *stop* (default ``'0x0000'``).
    xyz_samplerate:
        Override tracking frame rate (read from CSV header if None).
    """
    # ── Phase 1: primary recording window ──────────────────────────────── #
    record_end_sec, wideband_sr, lfp_sr = _load_record_sync(session)

    # ── Phase 2: find secondary start/stop on the primary timeline ──────── #
    ts_sec, labels  = _load_nlx_events(session)
    nlx_windows_sec = _find_ttl_windows(ts_sec, labels, ttl_value, stop_ttl)

    if len(nlx_windows_sec) == 0:
        raise RuntimeError(
            f"No TTL pairs found in {session.name}.all.evt "
            f"for start={ttl_value!r} / stop={stop_ttl!r}"
        )

    # ── Phase 3: load and duration-match position data ──────────────────── #
    chunks, markers, xyz_sr = _load_mocap_files(session)
    if xyz_samplerate is not None:
        xyz_sr = xyz_samplerate

    matched_chunks, matched_windows = _match_chunks_to_windows(
        chunks, nlx_windows_sec, xyz_sr
    )
    if not matched_chunks:
        raise RuntimeError(
            "Could not match any mocap chunks to NLX TTL windows.  "
            "Check ttl_value, stop_ttl, or position file frame rate."
        )

    # ── Phase 4: build arrays and populate session ───────────────────────── #
    xyz_arr = _build_xyz_array(matched_chunks, matched_windows, xyz_sr)
    _populate_session(
        session, xyz_arr, matched_windows, markers,
        xyz_sr, record_end_sec, wideband_sr, lfp_sr,
    )


# ---------------------------------------------------------------------------
# sync_nlx_spots  (NLX primary, 2-LED spot tracker .pos secondary)
# ---------------------------------------------------------------------------

def sync_nlx_spots(
    session,
    ttl_value:     str   = "0x0040",
    stop_ttl:      str   = "Stopping Recording",
    spots_samplerate: float = 39.0625,
) -> None:
    """Synchronise Neuralynx ephys with a 2-LED spot tracker.

    The secondary data is a plain binary ``.pos`` file containing
    ``(T, 4)`` int16 values: ``[x_led1, y_led1, x_led2, y_led2]``.
    The file is already stored in the session directory.

    Parameters
    ----------
    session:
        NBSession.
    ttl_value:
        TTL label marking mocap start.
    stop_ttl:
        TTL label marking mocap stop (default NLX stop event).
    spots_samplerate:
        Frame rate of the spot tracker in Hz (default 39.0625 Hz).
    """
    # ── Phase 1 ─────────────────────────────────────────────────────────── #
    record_end_sec, wideband_sr, lfp_sr = _load_record_sync(session)

    # ── Phase 2 ─────────────────────────────────────────────────────────── #
    ts_sec, labels  = _load_nlx_events(session)
    nlx_windows_sec = _find_ttl_windows(ts_sec, labels, ttl_value, stop_ttl)

    # ── Phase 3: load .pos file ─────────────────────────────────────────── #
    pos_file = session.spath / f"{session.name}.pos"
    if not pos_file.exists():
        raise FileNotFoundError(f"Spots position file not found: {pos_file}")

    raw = np.fromfile(str(pos_file), dtype=np.int16).reshape(-1, 4)
    # Reshape to (T, 2_markers, 2_dims): [[x1,y1],[x2,y2]]
    xy = raw.astype(np.float64).reshape(-1, 2, 2)

    # Model: two spot markers
    markers = ["led_front", "led_back"]

    # The .pos file is a single contiguous block — use event window spans
    # to define the sync periods; no duration-matching needed here since
    # the entire file spans [start_ttl, stop_ttl]
    if len(nlx_windows_sec) == 0:
        raise RuntimeError("No TTL pairs found for spots sync.")

    # Use the first start and last stop as the sync window
    first_start = nlx_windows_sec[0, 0] - 1.0 / spots_samplerate
    last_stop   = nlx_windows_sec[-1, 1] - 1.0 / spots_samplerate
    matched_windows = np.array([[first_start, last_stop]], dtype=np.float64)

    # Expand xy to 3D (add a Z=0 column) to match NBDxyz layout
    n_frames = xy.shape[0]
    xyz_arr  = np.concatenate([xy, np.zeros((n_frames, 2, 1))], axis=-1)

    # ── Phase 4 ─────────────────────────────────────────────────────────── #
    _populate_session(
        session, xyz_arr, matched_windows, markers,
        spots_samplerate, record_end_sec, wideband_sr, lfp_sr,
    )


# ---------------------------------------------------------------------------
# sync_nlx_whl  (NLX primary, .whl file secondary)
# ---------------------------------------------------------------------------

def sync_nlx_whl(session) -> None:
    """Synchronise Neuralynx ephys with a ``.whl`` position file.

    The ``.whl`` file (Neurosuite convention) is already on the NLX
    clock — no event matching is needed.  The sync window covers the
    full recording.

    The ``.whl`` file contains ``(T, 4)`` float values:
    ``[x_led1, y_led1, x_led2, y_led2]`` at ``lfp_sr / 32`` Hz.
    """
    # ── Phase 1 ─────────────────────────────────────────────────────────── #
    record_end_sec, wideband_sr, lfp_sr = _load_record_sync(session)

    # ── Load .whl ──────────────────────────────────────────────────────── #
    whl_file = session.spath / f"{session.name}.whl"
    if not whl_file.exists():
        raise FileNotFoundError(f"whl file not found: {whl_file}")

    raw = np.loadtxt(str(whl_file))                  # (T, 4)
    whl_sr = raw.shape[0] / record_end_sec           # infer frame rate
    n_frames = raw.shape[0]

    # Expand to (T, 2_markers, 3) with Z=0
    xy = raw.reshape(n_frames, 2, 2).astype(np.float64)
    xyz_arr = np.concatenate([xy, np.zeros((n_frames, 2, 1))], axis=-1)

    markers = ["led_left", "led_right"]

    # whl is already synchronised — sync covers the full recording
    matched_windows = np.array([[0.0, record_end_sec]], dtype=np.float64)

    # ── Phase 4 ─────────────────────────────────────────────────────────── #
    _populate_session(
        session, xyz_arr, matched_windows, markers,
        whl_sr, record_end_sec, wideband_sr, lfp_sr,
    )


# ---------------------------------------------------------------------------
# sync_openephys_optitrack  (OpenEphys primary, Optitrack/Motive secondary)
# ---------------------------------------------------------------------------

def sync_openephys_optitrack(
    session,
    sync_channel:  int   = 17,
    threshold:     float = 0.5,
    xyz_samplerate: float | None = None,
) -> None:
    """Synchronise Open Ephys ephys with Optitrack/Motive position data.

    The primary system is OpenEphys.  Synchronisation windows are
    extracted from a dedicated TTL pulse channel in the ``.lfp`` file.

    Parameters
    ----------
    session:
        NBSession.
    sync_channel:
        0-based ADC channel number carrying the mocap TTL pulse.
    threshold:
        Normalised detection threshold (0–1, default 0.5).
    xyz_samplerate:
        Override tracking frame rate.
    """
    from neurobox.io import load_binary

    # ── Phase 1 ─────────────────────────────────────────────────────────── #
    record_end_sec, wideband_sr, lfp_sr = _load_record_sync(session)

    # ── Phase 2: pulse-channel threshold crossing ────────────────────────── #
    lfp_file = session.spath / f"{session.name}.lfp"
    pulse = load_binary(
        lfp_file,
        channels      = [sync_channel],
        par           = session.par,
        channel_first = True,
    )[0].astype(np.float64)

    p_mean = pulse.mean()
    p_max  = np.abs(pulse - p_mean).max()
    if p_max < 1e-9:
        raise RuntimeError(
            f"Sync channel {sync_channel} appears flat.  "
            "Check sync_channel index."
        )
    pulse_norm = np.abs(pulse - p_mean) / p_max

    # Crossings → (N, 2) sample-index pairs
    crossings_samp = _thresh_cross(pulse_norm, threshold)
    if len(crossings_samp) == 0:
        raise RuntimeError(
            "No threshold crossings found on sync channel.  "
            "Check sync_channel or threshold."
        )
    # Convert to seconds
    nlx_windows_sec = crossings_samp.astype(np.float64) / lfp_sr

    # ── Phase 3: load and match position chunks ──────────────────────────── #
    chunks, markers, xyz_sr = _load_mocap_files(session)
    if xyz_samplerate is not None:
        xyz_sr = xyz_samplerate

    matched_chunks, matched_windows = _match_chunks_to_windows(
        chunks, nlx_windows_sec, xyz_sr
    )
    if not matched_chunks:
        raise RuntimeError(
            "Could not match any mocap chunks to pulse-channel windows.  "
            "Check sync_channel or threshold."
        )

    # ── Phase 4 ─────────────────────────────────────────────────────────── #
    xyz_arr = _build_xyz_array(matched_chunks, matched_windows, xyz_sr)
    _populate_session(
        session, xyz_arr, matched_windows, markers,
        xyz_sr, record_end_sec, wideband_sr, lfp_sr,
    )


# ---------------------------------------------------------------------------
# sync_openephys_vicon  (OpenEphys primary, Vicon Nexus secondary)
# ---------------------------------------------------------------------------

def sync_openephys_vicon(
    session,
    sync_channel:  int   = 17,
    threshold:     float = 0.5,
    xyz_samplerate: float | None = None,
) -> None:
    """Synchronise Open Ephys ephys with Vicon Nexus position data.

    Identical strategy to sync_openephys_optitrack — the Vicon
    position files are expected as Motive-compatible CSV exports.
    This function is provided as a separate entry point so that
    ``session.create(['openephys', 'vicon'])`` resolves correctly.
    """
    sync_openephys_optitrack(
        session,
        sync_channel   = sync_channel,
        threshold      = threshold,
        xyz_samplerate = xyz_samplerate,
    )


# ---------------------------------------------------------------------------
# Dispatcher (called from NBSession.create)
# ---------------------------------------------------------------------------

# Map (primary, secondary) keyword patterns → pipeline function
_PIPELINE_MAP: list[tuple[tuple[str, ...], tuple[str, ...], callable]] = [
    # OpenEphys + Optitrack/Motive
    (("openephys", "oephys"),          ("optitrack", "motive"),          sync_openephys_optitrack),
    # OpenEphys + Vicon
    (("openephys", "oephys"),          ("vicon",),                       sync_openephys_vicon),
    # NLX + Vicon/Optitrack/Motive (event-based)
    (("nlx", "neuralynx", "ephys"),    ("vicon", "optitrack", "motive"), sync_nlx_vicon),
    # NLX + Spots
    (("nlx", "neuralynx", "ephys"),    ("spots",),                       sync_nlx_spots),
    # NLX + WHL
    (("nlx", "neuralynx", "ephys"),    ("whl",),                         sync_nlx_whl),
]


def dispatch(session, data_loggers: list[str], **kwargs) -> None:
    """Select and run the correct sync pipeline for *data_loggers*.

    Parameters
    ----------
    session:
        NBSession.
    data_loggers:
        List of system names e.g. ``['nlx', 'vicon']``,
        ``['openephys', 'optitrack']``.
    **kwargs:
        Forwarded to the chosen pipeline function.
    """
    dl = [d.lower() for d in data_loggers]

    def _match(patterns: tuple[str, ...]) -> bool:
        return any(p in d for p in patterns for d in dl)

    for primary_patterns, secondary_patterns, func in _PIPELINE_MAP:
        if _match(primary_patterns) and _match(secondary_patterns):
            func(session, **kwargs)
            return

    raise ValueError(
        f"No sync pipeline found for data_loggers={data_loggers!r}.  "
        f"Supported primaries: nlx, openephys.  "
        f"Supported secondaries: vicon, optitrack, motive, spots, whl."
    )
