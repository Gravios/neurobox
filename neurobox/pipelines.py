"""
pipelines.py
============
High-level session setup pipelines.

Port of MTA's ``QuickSessionSetup.m`` and ``QuickTrialSetup.m``, adapted
for the neurobox directory layout and Python idioms.

Functions
---------
quick_session_setup(spec, ...)
    Link data, synchronise, and save an ``NBSession`` from a single
    spec dict (one ``get_session_list`` entry).

batch_session_setup(specs, ...)
    Run ``quick_session_setup`` over a list of spec dicts.

quick_trial_setup(session, ...)
    Create an ``NBTrial`` by trimming / selecting sync blocks from an
    existing ``NBSession``, with optional automatic behaviour labelling.

batch_trial_setup(sessions, ...)
    Run ``quick_trial_setup`` over a list of sessions or specs.

Spec dict keys
--------------
The ``spec`` dict mirrors the MTA ``get_session_list`` struct fields:

``sessionName`` (required)
    Full session name, e.g. ``'sirotaA-jg-05-20120316'``.
``mazeName`` (optional, default ``'cof'``)
    Arena code.
``trialName`` (optional, default ``'all'``)
    Trial label.
``dataLoggers`` (optional, default ``['nlx','vicon']``)
    List of acquisition system names passed to ``NBSession.create()``.
``ttlValue`` (optional, default ``'0x0040'``)
    TTL label for NLX sync events.
``stopTtl`` (optional, default ``'0x0000'``)
    TTL stop label.
``xyzSampleRate`` (optional)
    Override tracking frame rate.
``syncChannel`` (optional)
    ADC channel for OpenEphys sync pulse.
``offsets`` (optional, default ``[0, 0]``)
    ``[start_trim_sec, end_trim_sec]`` — seconds to skip from the start
    or clip from the end of each motion-tracking block.  Positive values
    trim from the start; negative trim from the end.
``dropSyncInd`` (optional)
    0-based indices of sync blocks to exclude from the trial.
``includeSyncInd`` (optional)
    0-based indices of sync blocks to include (all others are dropped).
``mazes`` (optional)
    List of maze codes to link mocap for.  Auto-discovered if absent.
``overwrite`` (optional, default ``False``)
    Re-create even if a ``.ses.pkl`` already exists.
``stcMode`` (optional)
    State-collection mode to load after trial creation.
``autolabel`` (optional, default ``True``)
    Run basic behaviour labelling after trial setup (requires the full
    rat marker set to be present in ``session.xyz.model``).
``projectId`` (optional)
    Override the project ID from ``.env``.
``dataRoot`` (optional)
    Override the data root from ``.env``.

Examples
--------
Single-session, no custom parameters::

    from neurobox.pipelines import quick_session_setup, quick_trial_setup

    session = quick_session_setup({"sessionName": "sirotaA-jg-05-20120316"})
    trial   = quick_trial_setup(session)

Batch from a list::

    specs = [
        {"sessionName": "sirotaA-jg-05-20120316",
         "dataLoggers": ["nlx", "vicon"], "ttlValue": "0x0040"},
        {"sessionName": "sirotaA-jg-06-20120320",
         "offsets": [15, 0]},
    ]
    from neurobox.pipelines import batch_session_setup
    sessions = batch_session_setup(specs)

Custom trial with trimmed sync::

    trial = quick_trial_setup(
        session,
        trial_name    = "run1",
        offsets       = [15, -10],        # skip first 15 s, last 10 s of each block
        drop_sync_ind = [2],              # ignore the third block
    )
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Spec helpers
# ---------------------------------------------------------------------------

def _parse_spec(spec: dict | str) -> dict:
    """Normalise a session spec to a flat dict with canonical keys."""
    if isinstance(spec, str):
        spec = {"sessionName": spec}
    spec = dict(spec)

    # Normalise common aliases
    for alias, canon in (
        ("session_name", "sessionName"),
        ("maze_name",    "mazeName"),
        ("trial_name",   "trialName"),
        ("data_loggers", "dataLoggers"),
        ("ttl_value",    "ttlValue"),
        ("stop_ttl",     "stopTtl"),
        ("xyz_sample_rate", "xyzSampleRate"),
        ("sync_channel",    "syncChannel"),
        ("drop_sync_ind",   "dropSyncInd"),
        ("include_sync_ind","includeSyncInd"),
        ("stc_mode",        "stcMode"),
        ("project_id",      "projectId"),
        ("data_root",       "dataRoot"),
    ):
        if alias in spec and canon not in spec:
            spec[canon] = spec.pop(alias)

    return spec


def _get(spec: dict, key: str, default=None):
    """Case-insensitive get from a spec dict."""
    # Try exact, then lower
    if key in spec:
        return spec[key]
    lower = {k.lower(): v for k, v in spec.items()}
    return lower.get(key.lower(), default)


# ---------------------------------------------------------------------------
# quick_session_setup  (port of QuickSessionSetup)
# ---------------------------------------------------------------------------

def quick_session_setup(
    spec:      dict | str,
    overwrite: bool = False,
) -> "NBSession":
    """Link data and create an ``NBSession`` from a spec dict.

    This is the Python port of MTA's ``QuickSessionSetup``.  It performs:

    1. **Link** — symlink processed data into the project directory via
       :func:`~neurobox.config.link_session`.
    2. **Create** — synchronise ephys with motion capture and populate
       all session fields via :meth:`~neurobox.dtype.session.NBSession.create`.
    3. **Save** — persist the session to ``.ses.pkl``.

    If a ``.ses.pkl`` already exists for this session, ``create()`` is
    skipped unless *overwrite* is True (or ``spec['overwrite']`` is set).

    Parameters
    ----------
    spec:
        Session specification dict or session name string.
        See module docstring for all recognised keys.
    overwrite:
        Re-create even if a ``.ses.pkl`` already exists.

    Returns
    -------
    NBSession
    """
    from neurobox.config.config import link_session, load_config
    from neurobox.dtype.session import NBSession

    spec      = _parse_spec(spec)
    name      = _get(spec, "sessionName")
    if not name:
        raise ValueError("spec must contain 'sessionName'")

    maze      = _get(spec, "mazeName",    "cof")
    mazes     = _get(spec, "mazes",        None)  # None → auto-discover
    overwrite = _get(spec, "overwrite",    overwrite)

    # Resolve project config
    conf: dict = {}
    try:
        conf = load_config()
    except FileNotFoundError:
        pass
    project_id = _get(spec, "projectId") or conf.get("NB_PROJECT_ID", "")
    data_root  = _get(spec, "dataRoot")  or conf.get("NB_DATA_PATH",  "/data")

    if not project_id:
        raise ValueError(
            "projectId is required in the spec dict or in the .env file."
        )

    print(f"\n{'='*60}")
    print(f"quick_session_setup: {name}")
    print(f"{'='*60}")

    # ── Step 1: Link ───────────────────────────────────────────────────── #
    print("[1/3] Linking data...")
    link_session(
        session_name = name,
        project_id   = project_id,
        data_root    = data_root,
        mazes        = mazes,
        overwrite    = False,   # don't stomp existing links on re-runs
    )

    # ── Step 2: Load or create session ────────────────────────────────── #
    session = NBSession(
        session_name = name,
        maze         = maze,
        project_id   = project_id,
        data_root    = data_root,
        overwrite    = overwrite,
    )

    ses_file = session.paths.ses_file
    if ses_file.exists() and not overwrite:
        print(f"[2/3] Session already created — loading {ses_file.name}")
        print("[3/3] Done (skipped sync).")
        return session

    print("[2/3] Synchronising...")
    data_loggers = _get(spec, "dataLoggers",  ["nlx", "vicon"])
    if isinstance(data_loggers, str):
        data_loggers = [data_loggers]

    # Build create() kwargs from spec
    create_kwargs: dict[str, Any] = {}
    if ttl := _get(spec, "ttlValue"):
        create_kwargs["ttl_value"]    = ttl
    if stop := _get(spec, "stopTtl"):
        create_kwargs["stop_ttl"]     = stop
    if sr := _get(spec, "xyzSampleRate"):
        create_kwargs["xyz_samplerate"] = float(sr)
    if ch := _get(spec, "syncChannel"):
        create_kwargs["sync_channel"] = int(ch)

    session.create(data_loggers, **create_kwargs)

    print("[3/3] Done.")
    return session


# ---------------------------------------------------------------------------
# batch_session_setup  (port of looping over get_session_list result)
# ---------------------------------------------------------------------------

def batch_session_setup(
    specs:         list[dict | str],
    overwrite:     bool = False,
    stop_on_error: bool = False,
) -> dict[str, "NBSession | Exception"]:
    """Run :func:`quick_session_setup` for each entry in *specs*.

    Parameters
    ----------
    specs:
        List of spec dicts or session name strings.
    overwrite:
        Passed to every ``quick_session_setup`` call.
    stop_on_error:
        If True, re-raise immediately on failure.  Default False:
        log errors and continue.

    Returns
    -------
    dict mapping session name → ``NBSession`` or caught ``Exception``.
    """
    results: dict[str, Any] = {}
    n = len(specs)

    for i, spec in enumerate(specs, 1):
        spec_d = _parse_spec(spec)
        name   = _get(spec_d, "sessionName", f"entry_{i}")
        print(f"\n[{i}/{n}] {name}")
        try:
            session      = quick_session_setup(spec_d, overwrite=overwrite)
            results[name] = session
        except Exception as exc:
            print(f"  [error] {exc}")
            results[name] = exc
            if stop_on_error:
                raise

    n_ok = sum(1 for v in results.values() if not isinstance(v, Exception))
    print(f"\nbatch_session_setup: {n_ok}/{n} OK")
    return results


# ---------------------------------------------------------------------------
# quick_trial_setup  (port of QuickTrialSetup)
# ---------------------------------------------------------------------------

def quick_trial_setup(
    session_or_spec,
    trial_name:       str              = "all",
    offsets:          list[float]      = (0.0, 0.0),
    drop_sync_ind:    list[int] | None = None,
    include_sync_ind: list[int] | None = None,
    overwrite:        bool             = False,
    autolabel:        bool             = True,
    stc_mode:         str | None       = None,
) -> "NBTrial":
    """Create an ``NBTrial`` from a session with optional sync trimming.

    Port of MTA's ``QuickTrialSetup``.

    The session's ``xyz.sync`` epochs define how many motion-tracking
    blocks were recorded and where they fall on the ephys clock.  This
    function lets you:

    * **Trim** the start/end of each block via *offsets*.
    * **Drop** specific blocks via *drop_sync_ind*.
    * **Select** only certain blocks via *include_sync_ind*.

    The trimmed/filtered sync periods are then used as the trial's sync
    epoch, which constrains all subsequent period-selection operations
    (e.g. ``lfp[stc['walk']]``).

    Parameters
    ----------
    session_or_spec:
        ``NBSession`` object, spec dict, session name string, or
        filebase string (``'name.maze.trial'``).
    trial_name:
        Label for the new trial, e.g. ``'all'`` (default) or ``'run1'``.
    offsets:
        ``[start_trim_sec, end_trim_sec]``.  Positive trims from start,
        negative trims from end.  Applied to *every* sync block.
        Default ``[0, 0]`` (no trimming).
    drop_sync_ind:
        0-based indices of sync blocks to remove.
    include_sync_ind:
        0-based indices of sync blocks to keep.  When both
        *drop_sync_ind* and *include_sync_ind* are given,
        *include_sync_ind* takes precedence.
    overwrite:
        Overwrite an existing ``.trl.pkl`` file.
    autolabel:
        Attempt automatic behaviour labelling after trial creation.
        Requires the full rat head marker set to be present in
        ``session.xyz.model`` (see ``_REQUIRED_AUTOLABEL_MARKERS``).
    stc_mode:
        Load this STC mode after trial creation.

    Returns
    -------
    NBTrial
    """
    from neurobox.dtype.session import NBSession, NBTrial
    from neurobox.dtype.epoch   import NBEpoch

    # ── Resolve session ────────────────────────────────────────────────── #
    if not isinstance(session_or_spec, NBSession):
        spec      = _parse_spec(session_or_spec)
        trial_name = _get(spec, "trialName",  trial_name)
        offsets    = _get(spec, "offsets",     offsets)
        drop_sync_ind  = _get(spec, "dropSyncInd",    drop_sync_ind)
        include_sync_ind = _get(spec, "includeSyncInd", include_sync_ind)
        autolabel  = _get(spec, "autolabel",   autolabel)
        stc_mode   = _get(spec, "stcMode",     stc_mode)
        session    = NBSession.validate(spec)
    else:
        session = session_or_spec

    print(f"\nquick_trial_setup: {session.name}.{session.maze}.{trial_name}")

    # ── Check existing trial ───────────────────────────────────────────── #
    trl_file = session.spath / f"{session.name}.{session.maze}.{trial_name}.trl.pkl"
    if trl_file.exists() and not overwrite:
        print(f"  Loading existing trial from {trl_file.name}")
        trial = NBTrial(session, trial_name=trial_name)
        if stc_mode:
            trial.load("stc", mode=stc_mode)
        return trial

    # ── Build trimmed sync ─────────────────────────────────────────────── #
    if session.xyz is None or session.xyz.sync is None:
        raise RuntimeError(
            f"session.xyz.sync is not set for {session.name!r}.  "
            "Run quick_session_setup() first."
        )

    sync_periods = session.xyz.sync._as_periods().copy()  # (M, 2) float64
    n_blocks     = sync_periods.shape[0]
    print(f"  xyz.sync: {n_blocks} block(s)")

    # ── Apply offsets ──────────────────────────────────────────────────── #
    offsets = list(offsets)
    if len(offsets) != 2:
        raise ValueError(f"offsets must be [start_trim, end_trim], got {offsets}")
    start_trim, end_trim = float(offsets[0]), float(offsets[1])
    if start_trim != 0.0 or end_trim != 0.0:
        sync_periods[:, 0] += start_trim
        sync_periods[:, 1] += end_trim
        # Drop any blocks that became zero-length or negative
        valid = (sync_periods[:, 1] - sync_periods[:, 0]) > 0
        sync_periods = sync_periods[valid]
        print(f"  offsets [{start_trim:+.1f}, {end_trim:+.1f}] s  "
              f"→ {sync_periods.shape[0]} block(s) remaining")

    # ── Apply include/drop logic ───────────────────────────────────────── #
    # include_sync_ind overrides drop_sync_ind (MTA convention)
    keep = np.ones(len(sync_periods), dtype=bool)

    if include_sync_ind is not None:
        inc = np.asarray(include_sync_ind, dtype=int)
        keep[:] = False
        keep[inc[inc < len(sync_periods)]] = True
        print(f"  includeSyncInd={list(inc)}  "
              f"→ {keep.sum()} block(s) kept")
    elif drop_sync_ind is not None:
        drp = np.asarray(drop_sync_ind, dtype=int)
        valid_drp = drp[drp < len(sync_periods)]
        keep[valid_drp] = False
        print(f"  dropSyncInd={list(drp)}  "
              f"→ {keep.sum()} block(s) remaining")

    sync_periods = sync_periods[keep]

    if len(sync_periods) == 0:
        raise RuntimeError(
            "All sync blocks were dropped.  "
            "Check offsets and drop_sync_ind."
        )

    trial_sync = NBEpoch(
        sync_periods,
        samplerate = 1.0,
        label      = trial_name,
    )

    # ── Create and save NBTrial ────────────────────────────────────────── #
    trial = NBTrial(
        session,
        trial_name = trial_name,
        sync       = trial_sync,
        overwrite  = True,
    )
    trial.save(overwrite=overwrite)
    print(f"  Saved → {trl_file.name}")

    # ── Optional stc load ──────────────────────────────────────────────── #
    if stc_mode:
        trial.load("stc", mode=stc_mode)

    # ── Optional autolabel ──────────────────────────────────────────────── #
    if autolabel:
        _try_autolabel(trial)

    return trial


# ---------------------------------------------------------------------------
# Autolabelling  (port of labelBhv prerequisite check in QuickTrialSetup)
# ---------------------------------------------------------------------------

# Markers required for automatic behaviour labelling
_REQUIRED_AUTOLABEL_MARKERS = {
    "spine_lower", "pelvis_root", "spine_middle", "spine_upper",
    "head_back",   "head_left",   "head_front",   "head_right",
}


def _try_autolabel(trial: "NBTrial") -> bool:
    """Attempt basic behaviour labelling if the right markers are present.

    Returns True if labelling was performed, False otherwise.
    """
    xyz = getattr(trial, "xyz", None)
    if xyz is None or xyz.model is None:
        return False

    present = set(xyz.model.markers)
    if not _REQUIRED_AUTOLABEL_MARKERS.issubset(present):
        missing = _REQUIRED_AUTOLABEL_MARKERS - present
        print(f"  autolabel skipped — missing markers: {sorted(missing)}")
        return False

    try:
        _label_basic_states(trial)
        return True
    except Exception as exc:
        print(f"  autolabel warning: {exc}")
        return False


def _label_basic_states(trial: "NBTrial") -> None:
    """Label basic locomotion states from xyz velocity.

    This is a simplified port of MTA's ``labelBhv`` using only position
    data (no LFP theta gating).  It produces states:

    ``walk``   — linear speed > *walk_speed_threshold* cm/s
    ``rear``   — head height above *rear_height_threshold* × median
    ``pause``  — not walking, not rearing

    All other time is labelled ``other``.
    """
    from neurobox.dtype.epoch import NBEpoch

    xyz = trial.xyz
    stc = trial.stc

    if xyz is None or xyz.data is None:
        return

    sr  = xyz.samplerate
    dt  = 1.0 / sr

    # ── Speed (cm/s) from centre-of-mass ─────────────────────────────── #
    com = xyz.data.mean(axis=1)                    # (T, 3)
    com_diff  = np.diff(com, axis=0) / dt          # (T-1, 3)
    speed_2d  = np.sqrt((com_diff[:, :2] ** 2).sum(axis=1))  # XY plane
    speed_2d  = np.concatenate([[0.0], speed_2d])  # (T,)

    # Smooth with 0.2 s boxcar
    window = max(1, int(round(0.2 * sr)))
    from scipy.ndimage import uniform_filter1d
    speed_smooth = uniform_filter1d(speed_2d, size=window)

    # ── Head height ───────────────────────────────────────────────────── #
    head_idx = None
    for m in ("head_back", "head_front"):
        if m in xyz.model.markers:
            head_idx = xyz.model.markers.index(m)
            break
    if head_idx is not None:
        head_z = xyz.data[:, head_idx, 2]
        median_z  = np.median(head_z[head_z > 0]) if (head_z > 0).any() else 1.0
    else:
        head_z = None
        median_z = 1.0

    # ── Thresholds (empirical, from MTA labelBhv defaults) ──────────── #
    walk_thr   = 3.0           # cm/s
    rear_thr   = 1.25 * median_z  # 125% of median head height

    T     = xyz.n_samples
    t_all = np.arange(T) / sr
    sync  = trial.sync._as_periods() if trial.sync is not None else None

    def _make_epoch(mask: np.ndarray, label: str, key: str) -> NBEpoch:
        from neurobox.dtype.epoch import NBEpoch
        ep = NBEpoch.from_logical(mask.astype(bool), samplerate=sr,
                                  label=label, key=key)
        # Merge gaps < 0.1 s, then drop periods < 0.5 s
        ep = ep.fillgaps(gap_sec=0.1)
        if not ep.isempty():
            dur = ep.data[:, 1] - ep.data[:, 0]
            ep.data = ep.data[dur >= 0.5]
        return ep

    walk_mask  = speed_smooth > walk_thr

    if head_z is not None:
        rear_mask = (head_z > rear_thr) & ~walk_mask
    else:
        rear_mask = np.zeros(T, dtype=bool)

    pause_mask = ~walk_mask & ~rear_mask

    for mask, label, key in (
        (walk_mask,  "walk",  "w"),
        (rear_mask,  "rear",  "r"),
        (pause_mask, "pause", "p"),
    ):
        ep = _make_epoch(mask, label, key)
        stc.add_state(ep)

    stc.save(overwrite=True)
    n_states = {lb: np.diff(stc[lb]._as_periods()).sum()
                for lb in ("walk", "rear", "pause")
                if stc.has_state(lb)}
    print("  autolabel:  "
          + "  ".join(f"{lb}={v:.0f}s" for lb, v in n_states.items()))


# ---------------------------------------------------------------------------
# batch_trial_setup
# ---------------------------------------------------------------------------

def batch_trial_setup(
    sessions_or_specs: list,
    trial_name:       str            = "all",
    offsets:          list[float]    = (0.0, 0.0),
    drop_sync_ind:    list[int] | None = None,
    include_sync_ind: list[int] | None = None,
    overwrite:        bool           = False,
    autolabel:        bool           = True,
    stop_on_error:    bool           = False,
) -> dict[str, "NBTrial | Exception"]:
    """Run :func:`quick_trial_setup` over a list of sessions or specs.

    Per-session overrides can be embedded in a spec dict using the
    same keys as described in the module docstring.

    Returns
    -------
    dict mapping session name → ``NBTrial`` or caught ``Exception``.
    """
    from neurobox.dtype.session import NBSession

    results: dict[str, Any] = {}
    n = len(sessions_or_specs)

    for i, entry in enumerate(sessions_or_specs, 1):
        # Resolve name for reporting
        if isinstance(entry, NBSession):
            name = entry.name
        else:
            spec_d = _parse_spec(entry)
            name   = _get(spec_d, "sessionName", f"entry_{i}")

        print(f"\n[{i}/{n}] {name}")
        try:
            trial = quick_trial_setup(
                entry,
                trial_name       = trial_name,
                offsets          = offsets,
                drop_sync_ind    = drop_sync_ind,
                include_sync_ind = include_sync_ind,
                overwrite        = overwrite,
                autolabel        = autolabel,
            )
            results[name] = trial
        except Exception as exc:
            print(f"  [error] {exc}")
            results[name] = exc
            if stop_on_error:
                raise

    n_ok = sum(1 for v in results.values() if not isinstance(v, Exception))
    print(f"\nbatch_trial_setup: {n_ok}/{n} OK")
    return results
