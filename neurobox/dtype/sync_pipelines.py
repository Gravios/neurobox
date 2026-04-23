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

Secondary systems
-----------------
``vicon``      Vicon / Optitrack Motive CSV exports
``optitrack``  Optitrack Motive CSV exports (alias for vicon)
``spots``      NLX 2-LED tracker, plain binary ``.pos`` (T×4)
``whl``        Neurosuite ``.whl`` binary (already on-clock)

Architecture
------------
Every sync function follows the same four-phase structure:

  Phase 1 — Establish the primary (ephys) recording window
            ``record_sync = [0.0, n_lfp_samples / lfp_sr]``
            Source: one channel of the .lfp file (symlinked or direct).

  Phase 2 — Find secondary-system start/stop times on the primary timeline
            NLX: parse ``.all.evt`` for TTL start/stop pairs
            OpenEphys: threshold-cross a dedicated ADC pulse channel

  Phase 3 — Match secondary data chunks to ephys windows by duration
            Tolerance: |ephys_dur − chunk_dur| < 0.2 s
            One-frame lead correction applied to matched windows (MTA).

  Phase 4 — Populate session fields
            session.sync   = NBEpoch([[first_start, last_stop]])
            session.xyz    = NBDxyz  (data on [0, span], zero-padded gaps)
            session.lfp    = NBDlfp  (lazy shell)
            session.spk    = NBSpk   (loaded from .res/.clu)
            session.stc    = empty NBStateCollection

Key relationships
-----------------
record_sync    = [0, lfp_duration_sec]           — full recording window
session.sync   ⊆ record_sync                     — secondary system active window
xyz.origin     = session.sync.data[0, 0]         — first secondary start
xyz.sync       = per-block windows (M, 2)        — all matched blocks
xyz data       = contiguous array starting at 0  — gaps zero-filled

Path resolution order
---------------------
All functions look for files in this order, stopping at the first hit:
  1. session.spath  — project directory (symlinks to processed files)
  2. session.paths.processed_ephys  — canonical processed ephys path
  3. session.paths.source_mocap     — raw source mocap (CSV files)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from neurobox.dtype.epoch  import NBEpoch
from neurobox.dtype.xyz    import NBDxyz
from neurobox.dtype.lfp    import NBDlfp
from neurobox.dtype.spikes import NBSpk
from neurobox.dtype.stc    import NBStateCollection
from neurobox.dtype.model  import NBModel


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _has_paths(session) -> bool:
    """Return True if the session has an NBSessionPaths object."""
    return hasattr(session, "paths") and session.paths is not None


def _find_file(session, filename: str) -> Path | None:
    """Find *filename* in spath or processed_ephys, return first hit."""
    candidates: list[Path] = [session.spath / filename]
    if _has_paths(session):
        candidates.append(session.paths.processed_ephys / filename)
    for p in candidates:
        if p.exists():
            return p
    return None


def _ephys_base(session) -> str:
    """Return the session base path string for I/O loaders.

    Tries spath (symlinked project dir) first; falls back to
    processed_ephys.
    """
    yaml_in_spath = session.spath / f"{session.name}.yaml"
    if yaml_in_spath.exists():
        return str(session.spath / session.name)
    if _has_paths(session):
        return str(session.paths.processed_ephys / session.name)
    return str(session.spath / session.name)


# ---------------------------------------------------------------------------
# Phase 1: establish recording window
# ---------------------------------------------------------------------------

def _load_record_sync(session) -> tuple[float, float, float]:
    """Load the parameter file and measure the LFP recording length.

    Returns
    -------
    (record_end_sec, wideband_sr, lfp_sr)

    Also sets ``session.par`` and ``session.samplerate``.
    """
    from neurobox.io import load_par, load_binary

    # ── Load par if not already present ───────────────────────────────── #
    if session.par is None:
        session.par = load_par(_ephys_base(session))

    par = session.par
    session.samplerate = float(par.acquisitionSystem.samplingRate)
    from neurobox.io.load_yaml import get_lfp_samplerate
    lfp_sr = get_lfp_samplerate(par)

    # ── Find .lfp file ─────────────────────────────────────────────────── #
    lfp_file = _find_file(session, f"{session.name}.lfp")
    if lfp_file is None:
        raise FileNotFoundError(
            f"LFP file not found for session {session.name!r}.  "
            f"Searched: {session.spath}, "
            f"{getattr(getattr(session,'paths',None),'processed_ephys','N/A')}"
        )

    # Load one channel to get exact sample count
    one_ch = load_binary(
        lfp_file,
        channels      = [0],
        par           = par,
        channel_first = True,
    )
    record_end_sec = one_ch.shape[1] / lfp_sr

    return record_end_sec, session.samplerate, lfp_sr


# ---------------------------------------------------------------------------
# Phase 2 (NLX): TTL event parsing
# ---------------------------------------------------------------------------

def _load_nlx_events(session) -> tuple[np.ndarray, list[str]]:
    """Load timestamps and labels from the ``.all.evt`` file (ms → s).

    Searches spath then processed_ephys.
    """
    from neurobox.io import load_evt

    evt_file = _find_file(session, f"{session.name}.all.evt")
    if evt_file is None:
        raise FileNotFoundError(
            f"Event file '{session.name}.all.evt' not found.  "
            f"Searched: {session.spath}, "
            f"{getattr(getattr(session,'paths',None),'processed_ephys','N/A')}"
        )
    ts_ms, labels = load_evt(str(evt_file), as_seconds=False)
    return ts_ms / 1000.0, labels


def _find_ttl_windows(
    ts_sec:    np.ndarray,
    labels:    list[str],
    start_ttl: str = "0x0040",
    stop_ttl:  str = "0x0000",
) -> np.ndarray:
    """Build ``(N, 2)`` float64 array of ``[start_sec, stop_sec]`` pairs.

    Pairs each start event with the first subsequent stop event.
    Handles mismatched counts by trimming to the shorter list.
    """
    start_mask = np.array([start_ttl in l for l in labels], dtype=bool)
    stop_mask  = np.array([stop_ttl  in l for l in labels], dtype=bool)
    vstarts    = ts_sec[start_mask]
    vstops     = ts_sec[stop_mask]

    # Drop stop events that precede the first start
    while vstops.size and vstops[0] < vstarts[0]:
        vstops = vstops[1:]

    n = min(len(vstarts), len(vstops))
    vstarts, vstops = vstarts[:n], vstops[:n]
    valid = vstops > vstarts
    if not valid.any():
        return np.empty((0, 2), dtype=np.float64)
    return np.column_stack([vstarts[valid], vstops[valid]]).astype(np.float64)


# ---------------------------------------------------------------------------
# Phase 2 (OpenEphys): pulse-channel threshold crossing
# ---------------------------------------------------------------------------

def _thresh_cross(
    signal:          np.ndarray,
    threshold:       float,
    min_dur_samples: int = 0,
) -> np.ndarray:
    """Return ``(N, 2)`` int64 array of ``[rise_sample, fall_sample]``.

    Port of MTA ``ThreshCross``.
    """
    above = (signal >= threshold).astype(np.int8)
    d     = np.diff(above)
    rises = (np.where(d ==  1)[0] + 1).tolist()
    falls = (np.where(d == -1)[0] + 1).tolist()

    if not rises or not falls:
        return np.empty((0, 2), dtype=np.int64)

    while falls and falls[0] <= rises[0]:
        falls.pop(0)
    n = min(len(rises), len(falls))
    pairs = np.array(list(zip(rises[:n], falls[:n])), dtype=np.int64)

    if min_dur_samples > 0:
        pairs = pairs[(pairs[:, 1] - pairs[:, 0]) >= min_dur_samples]
    return pairs


# ---------------------------------------------------------------------------
# Phase 3: load mocap CSV files
# ---------------------------------------------------------------------------

def _mocap_search_dirs(session) -> dict[str, list[Path]]:
    """Return directories to search for position data, keyed by format.

    Search order per format:

    **Processed .mat** (primary — MTA processC3D output):
      1. ``session.paths.processed_mocap``  — canonical processed path
      2. ``session.spath / session.maze``   — symlinked processed

    **Raw CSV** (fallback — Motive CSV exports):
      1. ``session.paths.source_mocap``     — raw source CSVs
      2. Parent of source_mocap             — all mazes under session
      3. ``session.spath / session.maze``   — any CSVs in symlinked dir
      4. ``session.spath``                  — flat fallback

    Returns
    -------
    dict with keys ``'mat'`` and ``'csv'``, each a list of existing Paths.
    """
    mat_dirs: list[Path] = []
    csv_dirs: list[Path] = []
    maze = getattr(session, "maze", "")

    if _has_paths(session):
        # processed_mocap already includes the maze subdir
        mat_dirs.append(session.paths.processed_mocap)
        csv_dirs.append(session.paths.source_mocap)
        csv_dirs.append(session.paths.source_mocap.parent)

    if maze:
        shared = session.spath / maze
        mat_dirs.append(shared)
        csv_dirs.append(shared)
    csv_dirs.append(session.spath)

    return {
        "mat": [d for d in mat_dirs if d.exists()],
        "csv": [d for d in csv_dirs if d.exists()],
    }


def _load_mocap_files(
    session,
    xyz_samplerate: float | None = None,
) -> tuple[list[np.ndarray], list[str], float]:
    """Find and load position data for this session.

    Tries formats in priority order:

    1. Processed ``.mat`` files (MTA ``processC3D`` output) — searched
       in ``session.paths.processed_mocap`` and ``spath/maze/``.
    2. Raw Motive CSV files — searched in ``session.paths.source_mocap``
       and ``spath/``.

    Returns
    -------
    chunks   : list of ``(T, N_markers, 3)`` arrays, one per block
    markers  : marker name strings
    xyz_sr   : tracking frame rate (Hz)
    """
    dirs = _mocap_search_dirs(session)

    # ── Try processed .mat files first ────────────────────────────── #
    for mat_dir in dirs["mat"]:
        try:
            from neurobox.io.load_processed_mat import concatenate_processed_mat
            chunks, markers, xyz_sr = concatenate_processed_mat(mat_dir)
            if xyz_samplerate is not None:
                xyz_sr = xyz_samplerate
            print(f"  mocap source: .mat files in {mat_dir.name}/")
            return chunks, markers, xyz_sr
        except FileNotFoundError:
            continue
        except Exception as exc:
            print(f"  [warn] .mat load failed in {mat_dir}: {exc}")
            continue

    # ── Fall back to raw CSV files ─────────────────────────────────── #
    csv_files: list[Path] = []
    for csv_dir in dirs["csv"]:
        for f in sorted(csv_dir.glob("*.csv")):
            if f not in csv_files:
                csv_files.append(f)

    if not csv_files:
        all_searched = dirs["mat"] + dirs["csv"]
        searched_str = ", ".join(str(d) for d in all_searched)
        raise FileNotFoundError(
            f"No position data found for session {session.name!r}.  "
            f"Searched (mat then csv): {searched_str}"
        )

    chunks:  list[np.ndarray] = []
    markers: list[str]        = []
    xyz_sr:  float            = xyz_samplerate or 0.0

    for csv in csv_files:
        try:
            obj = NBDxyz.from_motive_csv(csv, samplerate=xyz_samplerate)
        except Exception as exc:
            print(f"  [warn] {csv.name}: {exc}")
            continue
        if obj.data is None or obj.data.shape[0] == 0:
            continue
        if not markers:
            markers = obj.model.markers
            if xyz_sr == 0.0:
                xyz_sr = obj.samplerate
        chunks.append(obj.data)

    if not chunks:
        raise RuntimeError(
            f"No usable position data parsed from CSV files in {dirs['csv']}"
        )
    print(f"  mocap source: .csv files ({len(chunks)} chunk(s))")
    return chunks, markers, xyz_sr


# ---------------------------------------------------------------------------
# Phase 3: duration matching
# ---------------------------------------------------------------------------

def _match_chunks_to_windows(
    chunks:        list[np.ndarray],
    windows_sec:   np.ndarray,
    chunk_sr:      float,
    tolerance_sec: float = 0.2,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Match position data chunks to ephys windows by duration.

    Each chunk is matched to the first unmatched window whose duration
    is within *tolerance_sec* seconds of the chunk duration.

    The matched window start/stop are shifted back by one tracking frame
    (MTA convention: the TTL fires one frame after capture begins).

    Returns
    -------
    matched_chunks  : list[np.ndarray]
    matched_windows : np.ndarray, shape ``(M, 2)``  (ephys seconds)
    """
    matched_chunks:  list[np.ndarray]  = []
    matched_windows: list[list[float]] = []
    used = set()

    frame_dt = 1.0 / chunk_sr

    for chunk in chunks:
        if chunk is None or chunk.shape[0] == 0:
            continue
        chunk_dur = chunk.shape[0] / chunk_sr
        for wi, (t0, t1) in enumerate(windows_sec):
            if wi in used:
                continue
            if abs((t1 - t0) - chunk_dur) < tolerance_sec:
                matched_windows.append([t0 - frame_dt, t1 - frame_dt])
                matched_chunks.append(chunk)
                used.add(wi)
                break

    if not matched_chunks:
        return [], np.empty((0, 2), dtype=np.float64)
    return matched_chunks, np.array(matched_windows, dtype=np.float64)


# ---------------------------------------------------------------------------
# Phase 4: build xyz array and populate session
# ---------------------------------------------------------------------------

def _build_xyz_array(
    matched_chunks:  list[np.ndarray],
    matched_windows: np.ndarray,
    xyz_sr:          float,
) -> np.ndarray:
    """Concatenate matched chunks into a zero-padded contiguous array.

    The array starts at ``t = 0`` and spans
    ``[matched_windows[0,0], matched_windows[-1,1]]``.
    Gaps between blocks are zero-padded; in-block zeros are replaced
    with ``eps`` (MTA convention: zeros mark missing data).
    """
    t0_global    = matched_windows[0,  0]
    t1_global    = matched_windows[-1, 1]
    total_frames = int(np.ceil((t1_global - t0_global) * xyz_sr))

    n_markers = matched_chunks[0].shape[1]
    n_dims    = matched_chunks[0].shape[2]
    xyz_arr   = np.zeros((total_frames, n_markers, n_dims), dtype=np.float64)

    for (t0, _), chunk in zip(matched_windows, matched_chunks):
        i0  = int(round((t0 - t0_global) * xyz_sr))
        i1  = min(i0 + chunk.shape[0], total_frames)
        seg = chunk[:i1 - i0].copy().astype(np.float64)
        seg[seg == 0.0] = np.finfo(np.float32).eps
        xyz_arr[i0:i1] = seg

    return xyz_arr


def _populate_session(
    session,
    xyz_arr:         np.ndarray,
    matched_windows: np.ndarray,
    markers:         list[str],
    xyz_sr:          float,
    record_end_sec:  float,
    wideband_sr:     float,
    lfp_sr:          float,
    save_xyz:        bool = True,
) -> None:
    """Assign all session fields after sync matching is resolved.

    Parameters
    ----------
    save_xyz:
        If True (default), save the xyz array to ``session.paths.pos_file``
        via :meth:`~neurobox.dtype.xyz.NBDxyz.save_npy`.
    """
    spath    = session.spath
    name     = session.name
    filebase = session.filebase

    # ── session.sync ────────────────────────────────────────────────────── #
    first_start = float(matched_windows[0,  0])
    last_stop   = float(matched_windows[-1, 1])
    session.sync = NBEpoch(
        np.array([[first_start, last_stop]], dtype=np.float64),
        samplerate = 1.0,
        sync       = np.array([0.0, record_end_sec]),
        label      = "sync",
    )

    # ── xyz.sync (per-block windows) ─────────────────────────────────────── #
    xyz_sync = NBEpoch(
        matched_windows.copy(),
        samplerate = 1.0,
        sync       = np.array([0.0, record_end_sec]),
        label      = "xyz_sync",
    )

    # ── Resolve pos file path ─────────────────────────────────────────────── #
    pos_path = (
        session.paths.pos_file if _has_paths(session)
        else spath / f"{filebase}.pos.npz"
    )

    # ── session.xyz ─────────────────────────────────────────────────────── #
    session.xyz = NBDxyz(
        data       = xyz_arr,
        model      = NBModel(markers=markers),
        samplerate = xyz_sr,
        sync       = xyz_sync,
        origin     = first_start,
        path       = spath,
        filename   = pos_path.name,
        name       = name,
    )
    if save_xyz:
        spath.mkdir(parents=True, exist_ok=True)
        session.xyz.save_npy(pos_path)
        print(f"  xyz saved → {pos_path}")

    # ── session.lfp (lazy shell) ──────────────────────────────────────────── #
    lfp_sync = NBEpoch(
        np.array([[0.0, record_end_sec]], dtype=np.float64),
        samplerate = 1.0,
        label      = "lfp_sync",
    )
    lfp_path = (
        session.paths.lfp_file if _has_paths(session)
        else spath / f"{name}.lfp"
    )
    session.lfp = NBDlfp(
        path       = lfp_path.parent,
        filename   = lfp_path.name,
        samplerate = lfp_sr,
        sync       = lfp_sync,
        origin     = 0.0,
        name       = name,
    )

    # ── session.spk ──────────────────────────────────────────────────────── #
    try:
        session.spk = NBSpk.load(
            _ephys_base(session),
            samplerate = wideband_sr,
        )
        print(f"  spk  loaded: {session.spk.n_units} units  "
              f"{len(session.spk)} spikes")
    except Exception as e:
        print(f"  [warn] Could not load spikes: {e}")
        session.spk = NBSpk(samplerate=wideband_sr)

    # ── session.stc ──────────────────────────────────────────────────────── #
    stc_path = (
        session.paths.stc_file("default") if _has_paths(session)
        else spath / f"{filebase}.stc.default.pkl"
    )
    session.stc = NBStateCollection(
        path     = spath,
        filename = stc_path.name,
        mode     = "default",
        sync     = session.sync,
    )


# ---------------------------------------------------------------------------
# sync_nlx_vicon  (NLX primary, Vicon / Optitrack secondary)
# ---------------------------------------------------------------------------

def sync_nlx_vicon(
    session,
    ttl_value:      str   = "0x0040",
    stop_ttl:       str   = "0x0000",
    xyz_samplerate: float | None = None,
    tolerance_sec:  float = 0.2,
    save_xyz:       bool  = True,
) -> None:
    """Synchronise Neuralynx ephys with Vicon / Optitrack position data.

    Synchronisation windows come from TTL events in ``.all.evt``.

    Parameters
    ----------
    session:
        NBSession.
    ttl_value:
        TTL label marking each mocap block start (default ``'0x0040'``).
    stop_ttl:
        TTL label marking each mocap block stop  (default ``'0x0000'``).
    xyz_samplerate:
        Override tracking frame rate (read from CSV header if None).
    tolerance_sec:
        Duration-matching tolerance in seconds (default 0.2 s).
    save_xyz:
        Save the assembled xyz array to disk (default True).
    """
    print(f"sync_nlx_vicon: {session.name}")

    # Phase 1 ─────────────────────────────────────────────────────────────── #
    record_end_sec, wideband_sr, lfp_sr = _load_record_sync(session)
    print(f"  record: {record_end_sec:.1f} s  wideband {wideband_sr:.0f} Hz  "
          f"LFP {lfp_sr:.0f} Hz")

    # Phase 2 ─────────────────────────────────────────────────────────────── #
    ts_sec, labels  = _load_nlx_events(session)
    nlx_windows_sec = _find_ttl_windows(ts_sec, labels, ttl_value, stop_ttl)
    if len(nlx_windows_sec) == 0:
        raise RuntimeError(
            f"No TTL pairs found in {session.name}.all.evt "
            f"for start={ttl_value!r} / stop={stop_ttl!r}"
        )
    print(f"  TTL windows: {len(nlx_windows_sec)}")

    # Phase 3 ─────────────────────────────────────────────────────────────── #
    chunks, markers, xyz_sr = _load_mocap_files(session, xyz_samplerate)
    print(f"  mocap chunks: {len(chunks)}  markers: {len(markers)}  "
          f"sr: {xyz_sr:.3f} Hz")

    matched_chunks, matched_windows = _match_chunks_to_windows(
        chunks, nlx_windows_sec, xyz_sr, tolerance_sec
    )
    if not matched_chunks:
        raise RuntimeError(
            f"Could not match any of {len(chunks)} mocap chunk(s) to "
            f"{len(nlx_windows_sec)} TTL window(s).  "
            "Check ttl_value, stop_ttl, or xyz_samplerate."
        )
    print(f"  matched: {len(matched_chunks)} chunk(s)")

    # Phase 4 ─────────────────────────────────────────────────────────────── #
    xyz_arr = _build_xyz_array(matched_chunks, matched_windows, xyz_sr)
    _populate_session(
        session, xyz_arr, matched_windows, markers,
        xyz_sr, record_end_sec, wideband_sr, lfp_sr,
        save_xyz=save_xyz,
    )


# ---------------------------------------------------------------------------
# sync_nlx_spots  (NLX primary, 2-LED spot tracker)
# ---------------------------------------------------------------------------

def sync_nlx_spots(
    session,
    ttl_value:        str   = "0x0040",
    stop_ttl:         str   = "Stopping Recording",
    spots_samplerate: float = 39.0625,
    save_xyz:         bool  = True,
) -> None:
    """Synchronise Neuralynx ephys with a 2-LED spot tracker.

    The secondary data is a plain binary ``.pos`` file of ``(T, 4)``
    int16: ``[x_led1, y_led1, x_led2, y_led2]``.  It is treated as a
    single contiguous block spanning the first start to last stop TTL.

    Parameters
    ----------
    session:
        NBSession.
    ttl_value:
        TTL label marking the start of tracking.
    stop_ttl:
        TTL label marking the stop (default ``'Stopping Recording'``).
    spots_samplerate:
        Tracker frame rate in Hz (default 39.0625 Hz).
    save_xyz:
        Save the assembled xyz array to disk (default True).
    """
    print(f"sync_nlx_spots: {session.name}")

    # Phase 1 ─────────────────────────────────────────────────────────────── #
    record_end_sec, wideband_sr, lfp_sr = _load_record_sync(session)

    # Phase 2 ─────────────────────────────────────────────────────────────── #
    ts_sec, labels  = _load_nlx_events(session)
    nlx_windows_sec = _find_ttl_windows(ts_sec, labels, ttl_value, stop_ttl)
    if len(nlx_windows_sec) == 0:
        raise RuntimeError("No TTL pairs found for spots sync.")

    # Phase 3: load .pos ─────────────────────────────────────────────────── #
    pos_file = _find_file(session, f"{session.name}.pos")
    if pos_file is None:
        raise FileNotFoundError(
            f"Spots .pos file not found for {session.name!r}.  "
            f"Searched: {session.spath}"
        )
    raw  = np.fromfile(str(pos_file), dtype=np.int16).reshape(-1, 4)
    xy   = raw.astype(np.float64).reshape(-1, 2, 2)
    n_fr = xy.shape[0]
    xyz_arr = np.concatenate([xy, np.zeros((n_fr, 2, 1))], axis=-1)
    print(f"  spots: {n_fr} frames  sr: {spots_samplerate:.4f} Hz")

    frame_dt = 1.0 / spots_samplerate
    matched_windows = np.array([[
        nlx_windows_sec[0,  0] - frame_dt,
        nlx_windows_sec[-1, 1] - frame_dt,
    ]], dtype=np.float64)

    # Phase 4 ─────────────────────────────────────────────────────────────── #
    _populate_session(
        session, xyz_arr, matched_windows,
        ["led_front", "led_back"],
        spots_samplerate, record_end_sec, wideband_sr, lfp_sr,
        save_xyz=save_xyz,
    )


# ---------------------------------------------------------------------------
# sync_nlx_whl  (NLX primary, .whl file secondary)
# ---------------------------------------------------------------------------

def sync_nlx_whl(
    session,
    save_xyz: bool = True,
) -> None:
    """Synchronise Neuralynx ephys with a ``.whl`` position file.

    The ``.whl`` file is already on the NLX clock — no event matching
    needed.  ``(T, 4)`` floats: ``[x_led1, y_led1, x_led2, y_led2]``.
    """
    print(f"sync_nlx_whl: {session.name}")

    # Phase 1 ─────────────────────────────────────────────────────────────── #
    record_end_sec, wideband_sr, lfp_sr = _load_record_sync(session)

    # Phase 3: load .whl ─────────────────────────────────────────────────── #
    whl_file = _find_file(session, f"{session.name}.whl")
    if whl_file is None:
        raise FileNotFoundError(
            f"whl file not found for {session.name!r}.  "
            f"Searched: {session.spath}"
        )
    raw      = np.loadtxt(str(whl_file))           # (T, 4)
    n_fr     = raw.shape[0]
    whl_sr   = n_fr / record_end_sec
    xy       = raw.reshape(n_fr, 2, 2).astype(np.float64)
    xyz_arr  = np.concatenate([xy, np.zeros((n_fr, 2, 1))], axis=-1)
    print(f"  whl: {n_fr} frames  sr: {whl_sr:.4f} Hz")

    matched_windows = np.array([[0.0, record_end_sec]], dtype=np.float64)

    # Phase 4 ─────────────────────────────────────────────────────────────── #
    _populate_session(
        session, xyz_arr, matched_windows,
        ["led_left", "led_right"],
        whl_sr, record_end_sec, wideband_sr, lfp_sr,
        save_xyz=save_xyz,
    )


# ---------------------------------------------------------------------------
# sync_openephys_optitrack  (OpenEphys primary, Optitrack/Motive secondary)
# ---------------------------------------------------------------------------

def sync_openephys_optitrack(
    session,
    sync_channel:   int   = 17,
    threshold:      float = 0.5,
    xyz_samplerate: float | None = None,
    tolerance_sec:  float = 0.2,
    save_xyz:       bool  = True,
) -> None:
    """Synchronise Open Ephys ephys with Optitrack / Motive position data.

    Synchronisation windows come from threshold-crossings on a dedicated
    TTL pulse channel in the ``.lfp`` file.

    Parameters
    ----------
    session:
        NBSession.
    sync_channel:
        0-based ADC channel carrying the mocap TTL pulse.
    threshold:
        Normalised detection threshold 0–1 (default 0.5).
    xyz_samplerate:
        Override tracking frame rate.
    tolerance_sec:
        Duration-matching tolerance in seconds (default 0.2 s).
    save_xyz:
        Save the assembled xyz array to disk (default True).
    """
    from neurobox.io import load_binary

    print(f"sync_openephys_optitrack: {session.name}  "
          f"sync_channel={sync_channel}")

    # Phase 1 ─────────────────────────────────────────────────────────────── #
    record_end_sec, wideband_sr, lfp_sr = _load_record_sync(session)
    print(f"  record: {record_end_sec:.1f} s  wideband {wideband_sr:.0f} Hz  "
          f"LFP {lfp_sr:.0f} Hz")

    # Phase 2: threshold-cross a pulse channel ─────────────────────────────── #
    lfp_file = _find_file(session, f"{session.name}.lfp")
    if lfp_file is None:
        raise FileNotFoundError(f"LFP file not found for {session.name!r}")

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
            f"Sync channel {sync_channel} appears flat — "
            "check sync_channel index."
        )
    pulse_norm     = np.abs(pulse - p_mean) / p_max
    crossings_samp = _thresh_cross(pulse_norm, threshold)
    if len(crossings_samp) == 0:
        raise RuntimeError(
            "No threshold crossings on sync channel — "
            "check sync_channel or threshold."
        )
    oe_windows_sec = crossings_samp.astype(np.float64) / lfp_sr
    print(f"  pulse windows: {len(oe_windows_sec)}")

    # Phase 3 ─────────────────────────────────────────────────────────────── #
    chunks, markers, xyz_sr = _load_mocap_files(session, xyz_samplerate)
    print(f"  mocap chunks: {len(chunks)}  markers: {len(markers)}  "
          f"sr: {xyz_sr:.3f} Hz")

    matched_chunks, matched_windows = _match_chunks_to_windows(
        chunks, oe_windows_sec, xyz_sr, tolerance_sec
    )
    if not matched_chunks:
        raise RuntimeError(
            f"Could not match any of {len(chunks)} mocap chunk(s) to "
            f"{len(oe_windows_sec)} pulse window(s).  "
            "Check sync_channel, threshold, or xyz_samplerate."
        )
    print(f"  matched: {len(matched_chunks)} chunk(s)")

    # Phase 4 ─────────────────────────────────────────────────────────────── #
    xyz_arr = _build_xyz_array(matched_chunks, matched_windows, xyz_sr)
    _populate_session(
        session, xyz_arr, matched_windows, markers,
        xyz_sr, record_end_sec, wideband_sr, lfp_sr,
        save_xyz=save_xyz,
    )


# ---------------------------------------------------------------------------
# sync_openephys_vicon  (alias — same pulse-channel strategy)
# ---------------------------------------------------------------------------

def sync_openephys_vicon(
    session,
    sync_channel:   int   = 17,
    threshold:      float = 0.5,
    xyz_samplerate: float | None = None,
    tolerance_sec:  float = 0.2,
    save_xyz:       bool  = True,
) -> None:
    """Synchronise Open Ephys ephys with Vicon Nexus position data.

    Same strategy as :func:`sync_openephys_optitrack`.
    """
    sync_openephys_optitrack(
        session,
        sync_channel   = sync_channel,
        threshold      = threshold,
        xyz_samplerate = xyz_samplerate,
        tolerance_sec  = tolerance_sec,
        save_xyz       = save_xyz,
    )


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

# (primary_patterns, secondary_patterns, function)
_PIPELINE_MAP: list[tuple[tuple[str, ...], tuple[str, ...], object]] = [
    (("openephys", "oephys"), ("optitrack", "motive"), sync_openephys_optitrack),
    (("openephys", "oephys"), ("vicon",),              sync_openephys_vicon),
    (("nlx", "neuralynx"),    ("vicon", "optitrack", "motive"), sync_nlx_vicon),
    (("nlx", "neuralynx"),    ("spots",),              sync_nlx_spots),
    (("nlx", "neuralynx"),    ("whl",),                sync_nlx_whl),
]


def dispatch(session, data_loggers: list[str], **kwargs) -> None:
    """Select and run the correct sync pipeline for *data_loggers*.

    Parameters
    ----------
    session:
        NBSession with ``spath``, ``name``, ``maze``, and optionally
        ``paths`` (NBSessionPaths) already set.
    data_loggers:
        List of system names, e.g. ``['nlx', 'vicon']``.
    **kwargs:
        Forwarded to the chosen pipeline function.
    """
    dl = [d.lower() for d in data_loggers]

    def _match(patterns: tuple[str, ...]) -> bool:
        return any(p in d for p in patterns for d in dl)

    for primary_pats, secondary_pats, func in _PIPELINE_MAP:
        if _match(primary_pats) and _match(secondary_pats):
            func(session, **kwargs)
            return

    raise ValueError(
        f"No sync pipeline for data_loggers={data_loggers!r}.\n"
        "Supported primaries  : nlx, neuralynx, openephys, oephys\n"
        "Supported secondaries: vicon, optitrack, motive, spots, whl"
    )
