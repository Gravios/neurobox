#!/usr/bin/env python3
"""
demo_sirotaA_jg_05_20120316.py
================================
End-to-end demo: synchronise raw data for session
``sirotaA-jg-05-20120316`` from scratch, save it, then open it in
the MTA Browser.

What this script does
---------------------
1. **Link the session** into the project directory tree.  This
   creates ``<data_root>/project/<project_id>/<session>/`` with
   symlinks to processed ephys + mocap files (no copying).
2. **Synchronise** Neuralynx LFP timestamps with the Vicon mocap
   chunks via :meth:`NBSession.create(['nlx', 'vicon'])`, which
   dispatches to :func:`sync_nlx_vicon`.  The result includes:

   * ``session.window``           — :class:`TrialWindow` covering
     the master-clock span of all matched mocap windows.
   * ``session.xyz.stream_sync``  — :class:`StreamSync` describing
     when Vicon was actively recording.
   * ``session.xyz.recording_windows`` — the per-block ground truth
     used by :meth:`NBData.restrict_to_window` to gap-fill correctly.
   * ``session.lfp.stream_sync``  — single-segment continuous sync
     for the master LFP recording.

   The assembled position array is saved to
   ``<spath>/<filebase>.pos.npz`` and a ``.ses.pkl`` checkpoint is
   written next to it.
3. **Verify** the round-trip: reload the session from disk, build
   a small :class:`NBTrial` with a programmatic
   :class:`TrialWindow`, and check that ``trial.load('xyz')``
   returns the expected sample count.
4. **Open in MTABrowser** so you can scrub through the data and
   label states interactively.

Usage
-----
::

    # Use defaults (reads NB_DATA_PATH / NB_PROJECT_ID from env or
    # from a .env file located via $NB_DOTENV_PATH or cwd/.env;
    # falls back to /data and B01 if neither source has them set)
    python scripts/demo_sirotaA_jg_05_20120316.py

    # Override the data root for one run via env var
    NB_DATA_PATH=/mnt/data python scripts/demo_sirotaA_jg_05_20120316.py

    # Or via CLI flag (highest precedence)
    python scripts/demo_sirotaA_jg_05_20120316.py \
        --project-id B01 --data-root /mnt/data

    # Skip the create() step if the .ses.pkl checkpoint exists
    python scripts/demo_sirotaA_jg_05_20120316.py --no-recreate

    # Just sync, don't open the GUI (useful for batch runs / CI)
    python scripts/demo_sirotaA_jg_05_20120316.py --no-gui

    # Bring up CheckEegStates as well after the browser opens
    python scripts/demo_sirotaA_jg_05_20120316.py --with-eeg-states

The ``data-root`` is resolved by precedence:

  1. ``--data-root`` CLI flag
  2. ``$NB_DATA_PATH`` in the process environment
  3. ``NB_DATA_PATH`` in the ``.env`` file (located via
     ``$NB_DOTENV_PATH`` or ``cwd/.env``)
  4. Hardcoded fallback ``/data``

The same precedence applies to ``--project-id`` /
``$NB_PROJECT_ID`` / ``NB_PROJECT_ID`` in ``.env`` /
fallback ``B01``.

Expected layout under ``--data-root``
-------------------------------------
::

    <data_root>/
        processed/ephys/sirotaA/sirotaA-jg/sirotaA-jg-05/sirotaA-jg-05-20120316/
            sirotaA-jg-05-20120316.yaml
            sirotaA-jg-05-20120316.lfp
            sirotaA-jg-05-20120316.all.evt    (TTL events)
            sirotaA-jg-05-20120316.res.1      (spike times)
            sirotaA-jg-05-20120316.clu.1      (cluster IDs)
            ...
        processed/mocap/sirotaA/sirotaA-jg/sirotaA-jg-05/sirotaA-jg-05-20120316/
            cof/
                sirotaA-jg-05-20120316.Trial001.mat
                sirotaA-jg-05-20120316.Trial002.mat
                ...
        source/mocap/sirotaA/sirotaA-jg/sirotaA-jg-05/sirotaA-jg-05-20120316/
            <Motive .csv files, optional fallback>

If the layout differs, edit ``data_root`` at the top of ``main()``
or pass ``--data-root /your/path``.
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

import numpy as np


SESSION_NAME = "sirotaA-jg-05-20120316"
DEFAULT_PROJECT_ID = "B01"
DEFAULT_MAZE = "cof"


# ─────────────────────────────────────────────────────────────────────── #
# Resolve data_root / project_id from environment                            #
# ─────────────────────────────────────────────────────────────────────── #

def _resolve_env_defaults() -> dict[str, str]:
    """Look up NB_DATA_PATH and NB_PROJECT_ID with the standard
    neurobox precedence:

      1. ``$NB_DATA_PATH`` / ``$NB_PROJECT_ID`` in the process
         environment (12-factor style — handy for one-off runs).
      2. Same keys in a ``.env`` file located via
         :func:`neurobox.config.config.load_config`
         (``$NB_DOTENV_PATH`` or ``cwd/.env``).
      3. Hardcoded fallbacks: ``/data`` and ``B01``.

    Returns a dict ``{data_root, project_id, source}`` where
    ``source`` describes which step provided each value, for
    diagnostic printing.
    """
    out = {"data_root": None, "project_id": None, "source": {}}
    # Step 1: process environment
    env_root = os.environ.get("NB_DATA_PATH")
    env_pid  = os.environ.get("NB_PROJECT_ID")
    if env_root:
        out["data_root"] = env_root
        out["source"]["data_root"] = "$NB_DATA_PATH"
    if env_pid:
        out["project_id"] = env_pid
        out["source"]["project_id"] = "$NB_PROJECT_ID"
    # Step 2: .env file
    if out["data_root"] is None or out["project_id"] is None:
        try:
            from neurobox.config.config import load_config
            conf = load_config()
            if out["data_root"] is None and "NB_DATA_PATH" in conf:
                out["data_root"] = conf["NB_DATA_PATH"]
                out["source"]["data_root"] = ".env (NB_DATA_PATH)"
            if out["project_id"] is None and "NB_PROJECT_ID" in conf:
                out["project_id"] = conf["NB_PROJECT_ID"]
                out["source"]["project_id"] = ".env (NB_PROJECT_ID)"
        except (FileNotFoundError, ImportError):
            pass
    # Step 3: hardcoded fallbacks
    if out["data_root"] is None:
        out["data_root"] = "/data"
        out["source"]["data_root"] = "default"
    if out["project_id"] is None:
        out["project_id"] = DEFAULT_PROJECT_ID
        out["source"]["project_id"] = "default"
    return out


# ─────────────────────────────────────────────────────────────────────── #
# Step 1 — Link the session into the project tree                            #
# ─────────────────────────────────────────────────────────────────────── #

def step_1_link(
    session_name: str,
    project_id:   str,
    data_root:    Path,
    *,
    overwrite:    bool = False,
    verbose:      bool = True,
):
    """Link processed ephys + mocap into ``project/<project_id>/<session>/``.

    Returns the resulting :class:`NBSessionPaths`.  Idempotent — if
    the directory and symlinks already exist, ``link_session`` will
    just verify them unless ``overwrite=True``.
    """
    print(f"\n=== Step 1: link_session ===")
    print(f"  session    : {session_name}")
    print(f"  project_id : {project_id}")
    print(f"  data_root  : {data_root}")

    from neurobox.config.config import link_session
    paths = link_session(
        session_name = session_name,
        project_id   = project_id,
        data_root    = data_root,
        overwrite    = overwrite,
        verbose      = verbose,
    )
    print(f"  → spath: {paths.spath}")
    print(f"    yaml:    {paths.yaml_file}")
    print(f"    lfp :    {paths.lfp_file}")
    return paths


# ─────────────────────────────────────────────────────────────────────── #
# Step 2 — Synchronise raw data (NLX <-> Vicon)                              #
# ─────────────────────────────────────────────────────────────────────── #

def step_2_synchronise(
    session_name: str,
    project_id:   str,
    data_root:    Path,
    maze:         str,
    *,
    recreate:     bool = True,
    ttl_value:    str  = "0x0040",
    stop_ttl:     str  = "0x0000",
):
    """Build the session via :meth:`NBSession.create(['nlx', 'vicon'])`.

    If ``recreate=False`` and the ``.ses.pkl`` checkpoint already
    exists, just load it instead of re-running the sync.
    """
    print(f"\n=== Step 2: synchronise (NLX <-> Vicon) ===")
    from neurobox.dtype import NBSession

    session = NBSession(
        session_name = session_name,
        maze         = maze,
        project_id   = project_id,
        data_root    = data_root,
    )

    ses_pkl = session.spath / f"{session.filebase}.ses.pkl"
    if ses_pkl.exists() and not recreate:
        print(f"  .ses.pkl exists; loading from checkpoint")
        session.load()                     # loads .ses.pkl
        # Restore xyz from the .pos.npz
        try:
            session.load("xyz")
        except FileNotFoundError as e:
            print(f"  [warn] xyz not on disk: {e}")
    else:
        print(f"  running create(['nlx', 'vicon'], "
              f"ttl_value={ttl_value!r})")
        session.create(
            data_loggers = ["nlx", "vicon"],
            ttl_value    = ttl_value,
            stop_ttl     = stop_ttl,
            save_xyz     = True,
        )

    # Round-23 sync state — these are what ``trial.load('xyz')``
    # consults when restricting to a trial window.
    print(f"\n  --- session sync state ---")
    if session.window is not None:
        print(f"  session.window: t={session.window.t_start:.2f}–"
              f"{session.window.t_stop:.2f} s "
              f"({session.window.total_duration:.1f} s total)")
    if session.xyz is not None:
        print(f"  session.xyz: shape={session.xyz.data.shape}, "
              f"sr={session.xyz.samplerate} Hz")
        if session.xyz.recording_windows is not None:
            rw = session.xyz.recording_windows
            print(f"  session.xyz.recording_windows: {len(rw)} blocks")
            for i, (t0, t1) in enumerate(rw[:5]):
                print(f"    block {i}: [{t0:7.2f}, {t1:7.2f}] s "
                      f"({(t1-t0):.2f} s)")
            if len(rw) > 5:
                print(f"    ... and {len(rw) - 5} more")
        if session.xyz.stream_sync is not None:
            ss = session.xyz.stream_sync
            print(f"  session.xyz.stream_sync.master_first/last: "
                  f"{ss.master_first:.2f} / {ss.master_last:.2f}")
    if session.lfp is not None:
        print(f"  session.lfp: sr={session.lfp.samplerate} Hz")
        if session.lfp.stream_sync is not None:
            ls = session.lfp.stream_sync
            print(f"  session.lfp.stream_sync: continuous "
                  f"[{ls.master_first:.2f}, {ls.master_last:.2f}] s")
    if session.spk is not None:
        print(f"  session.spk: {session.spk.n_units} units, "
              f"{len(session.spk)} spikes")

    return session


# ─────────────────────────────────────────────────────────────────────── #
# Step 3 — Verify Trial.load('xyz') round-trip through disk                  #
# ─────────────────────────────────────────────────────────────────────── #

def step_3_trial_load(
    session_name: str,
    project_id:   str,
    data_root:    Path,
    maze:         str,
):
    """Build an NBTrial with a hand-rolled TrialWindow and load xyz.

    This exercises the full disk → restrict_to_window → multi-segment
    zero-fill chain.  The trial here is a 30-second slice in the
    middle of the session — it'll typically straddle one or more
    Vicon stop/restart gaps, so the gap-fill machinery actually does
    something interesting.
    """
    print(f"\n=== Step 3: NBTrial.load('xyz') round-trip ===")
    from neurobox.dtype import NBSession, NBTrial, TrialWindow

    # First peek at the saved session's window so we know what
    # master-time range is meaningful for this dataset.
    session = NBSession(
        session_name = session_name,
        maze         = maze,
        project_id   = project_id,
        data_root    = data_root,
    )
    session.load()
    if session.window is None or session.window.is_empty:
        print(f"  [skip] session.window not populated — nothing to slice")
        return None

    t_start = session.window.t_start
    t_stop  = session.window.t_stop
    span    = t_stop - t_start
    # Take a 30-s window centred around the midpoint, clipped to the
    # session.  Falls back to the full session if it's shorter.
    if span < 30:
        sub_t0, sub_t1 = t_start, t_stop
    else:
        mid = 0.5 * (t_start + t_stop)
        sub_t0 = mid - 15.0
        sub_t1 = mid + 15.0
    print(f"  trial window: [{sub_t0:.2f}, {sub_t1:.2f}] s   "
          f"({sub_t1 - sub_t0:.2f} s)")

    win = TrialWindow(
        periods = np.array([[sub_t0, sub_t1]], dtype=np.float64),
        label   = "demo30s",
        name    = "demo30s",
    )
    trial = NBTrial(
        session_name = session_name,
        maze         = maze,
        trial_name   = "demo30s",
        project_id   = project_id,
        data_root    = data_root,
        window       = win,
    )
    xyz = trial.load("xyz")
    expected_n = int(round((sub_t1 - sub_t0) * xyz.samplerate))
    print(f"  trial.xyz.data.shape:       {xyz.data.shape}")
    print(f"    (expected ~{expected_n} samples = "
          f"{sub_t1 - sub_t0:.1f}s × {xyz.samplerate}Hz)")
    if xyz.stream_sync is not None:
        valid = xyz.stream_sync.valid_mask_in_window(sub_t0, sub_t1)
        print(f"    valid samples (Vicon was on): "
              f"{int(valid.sum())} / {len(valid)} "
              f"({100 * valid.sum() / max(len(valid), 1):.1f}%)")
        print(f"    recorded sub-segments: "
              f"{xyz.stream_sync.n_segments}")
        for i, (t0, t1) in enumerate(xyz.stream_sync.segments[:3]):
            print(f"      [{t0:.2f}, {t1:.2f}]")
    return session


# ─────────────────────────────────────────────────────────────────────── #
# Step 4 — Open in MTABrowser                                                #
# ─────────────────────────────────────────────────────────────────────── #

def step_4_open_browser(
    session,
    *,
    with_eeg_states: bool = False,
    block:           bool = True,
):
    """Hand the session to MTABrowserWindow.

    Uses the existing ``MTABrowserWindow.launch`` helper which
    handles the QApplication lifecycle.
    """
    print(f"\n=== Step 4: open MTA Browser ===")

    # Force the Qt-aware matplotlib backend BEFORE any pyplot import
    import matplotlib
    matplotlib.use("QtAgg")

    from PySide6.QtWidgets import QApplication
    from neurobox.gui.mta_browser import MTABrowserWindow

    app = QApplication.instance() or QApplication(sys.argv)
    win = MTABrowserWindow.launch(session=session, run=False)
    print(f"  browser open at: {win.windowTitle()}")
    print(f"  tabs: " + ", ".join(
        win._tabs.tabText(i) for i in range(win._tabs.count())
    ))

    # Optionally pop CheckEegStates alongside the browser
    if with_eeg_states:
        try:
            from neurobox.gui.check_eeg_states import CheckEegStatesWindow
            eeg = CheckEegStatesWindow.launch(session=session, run=False)
            print(f"  CheckEegStates open at: {eeg.windowTitle()}")
        except Exception as e:
            print(f"  [warn] CheckEegStates failed to open: {e}")
            traceback.print_exc()

    if block:
        print(f"\n  Close the browser window to exit.")
        app.exec()
    return win


# ─────────────────────────────────────────────────────────────────────── #
# Entry point                                                                #
# ─────────────────────────────────────────────────────────────────────── #

def main(argv: list[str] | None = None) -> int:
    env_defaults = _resolve_env_defaults()

    parser = argparse.ArgumentParser(
        prog        = Path(__file__).name,
        description = __doc__.split("Usage")[0].strip(),
        formatter_class = argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--session-name",
        default=SESSION_NAME,
        help=f"Session name (default {SESSION_NAME!r})",
    )
    parser.add_argument(
        "--project-id",
        default=env_defaults["project_id"],
        help=f"Project identifier "
             f"(default {env_defaults['project_id']!r} "
             f"from {env_defaults['source']['project_id']})",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(env_defaults["data_root"]),
        help=f"Root of the data tree "
             f"(default {env_defaults['data_root']!r} "
             f"from {env_defaults['source']['data_root']})",
    )
    parser.add_argument(
        "--maze",
        default=DEFAULT_MAZE,
        help=f"Maze code (default {DEFAULT_MAZE!r})",
    )
    parser.add_argument(
        "--ttl-value",
        default="0x0040",
        help="TTL label marking mocap start (default 0x0040)",
    )
    parser.add_argument(
        "--stop-ttl",
        default="0x0000",
        help="TTL label marking mocap stop (default 0x0000)",
    )
    parser.add_argument(
        "--no-link",
        action="store_true",
        help="Skip step 1 (link_session); assume project tree is set up",
    )
    parser.add_argument(
        "--no-recreate",
        action="store_true",
        help="Skip step 2's create() if .ses.pkl checkpoint exists; "
             "just load from it instead",
    )
    parser.add_argument(
        "--no-trial-check",
        action="store_true",
        help="Skip step 3 (NBTrial round-trip verification)",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Skip step 4 (don't open the browser); useful for "
             "scripted / headless runs",
    )
    parser.add_argument(
        "--with-eeg-states",
        action="store_true",
        help="Also open the CheckEegStates window after the browser",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing symlinks during link_session",
    )
    args = parser.parse_args(argv)

    print(f"=== {Path(__file__).name} ===")
    print(f"Session: {args.session_name}")
    # Annotate with provenance of each setting
    pid_src  = (env_defaults["source"]["project_id"]
                if args.project_id == env_defaults["project_id"]
                else "--project-id flag")
    root_src = (env_defaults["source"]["data_root"]
                if str(args.data_root) == env_defaults["data_root"]
                else "--data-root flag")
    print(f"Project: {args.project_id} (source: {pid_src})")
    print(f"Root   : {args.data_root} (source: {root_src})")
    print(f"Maze   : {args.maze}")

    # Step 1: link
    if not args.no_link:
        try:
            step_1_link(
                session_name = args.session_name,
                project_id   = args.project_id,
                data_root    = args.data_root,
                overwrite    = args.overwrite,
            )
        except Exception as e:
            print(f"\n[error] link_session failed: {e}")
            traceback.print_exc()
            return 2
    else:
        print(f"\n=== Step 1: link_session SKIPPED (--no-link) ===")

    # Step 2: sync (or load checkpoint)
    try:
        session = step_2_synchronise(
            session_name = args.session_name,
            project_id   = args.project_id,
            data_root    = args.data_root,
            maze         = args.maze,
            recreate     = not args.no_recreate,
            ttl_value    = args.ttl_value,
            stop_ttl     = args.stop_ttl,
        )
    except Exception as e:
        print(f"\n[error] synchronisation failed: {e}")
        traceback.print_exc()
        return 3

    # Step 3: trial round-trip verify
    if not args.no_trial_check:
        try:
            step_3_trial_load(
                session_name = args.session_name,
                project_id   = args.project_id,
                data_root    = args.data_root,
                maze         = args.maze,
            )
        except Exception as e:
            print(f"\n[warn] trial round-trip check failed: {e}")
            traceback.print_exc()
            # not fatal — continue to GUI
    else:
        print(f"\n=== Step 3: trial round-trip check "
              f"SKIPPED (--no-trial-check) ===")

    # Step 4: GUI
    if not args.no_gui:
        try:
            step_4_open_browser(
                session,
                with_eeg_states = args.with_eeg_states,
                block           = True,
            )
        except Exception as e:
            print(f"\n[error] GUI failed: {e}")
            traceback.print_exc()
            return 4
    else:
        print(f"\n=== Step 4: GUI SKIPPED (--no-gui) ===")

    print(f"\n=== Done ===")
    return 0


if __name__ == "__main__":            # pragma: no cover
    sys.exit(main())
