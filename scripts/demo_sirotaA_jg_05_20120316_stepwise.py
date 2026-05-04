"""
demo_sirotaA_jg_05_20120316_stepwise.py
========================================
Interactive (cell-based) demo for stepping through the
``sirotaA-jg-05-20120316`` pipeline.

This is the same end-to-end flow as
``demo_sirotaA_jg_05_20120316.py`` but reorganised into
``# %%`` cells you can run one at a time.

How to use
----------
* **VS Code (Pylance / Jupyter extension)** — open the file, each
  ``# %%`` block becomes a runnable cell.  Click "Run Cell" or
  Shift-Enter.
* **IPython REPL** — start ``ipython`` and use
  ``%run -i scripts/demo_sirotaA_jg_05_20120316_stepwise.py`` to
  run the whole thing while keeping all variables in your
  namespace.  To run a single cell manually, copy/paste it.
* **JupyText / Jupyter Lab** — open with the ``percent`` format
  reader; cells render as notebook cells.
* **Plain Python** — ``python scripts/...`` runs every cell
  sequentially.  This won't open the GUI's event loop, so the
  Browser cell at the end will block — pass ``--no-gui`` semantics
  by setting ``OPEN_GUI = False`` near the top.

Required: a project tree under ``DATA_ROOT`` containing
``sirotaA-jg-05-20120316`` (see the docstring of
``demo_sirotaA_jg_05_20120316.py`` for the expected layout).

Tip: re-running cells is safe.  Step 1 (link) is idempotent;
step 2 (sync) skips itself if the .ses.pkl checkpoint already
exists and ``RECREATE = False``.  Tweak any of the constants
in cell 1, then re-run only the cells you need.
"""

# %% [markdown]
# # Setup — knobs and imports
#
# Edit the constants in this cell to point at your dataset.  The
# values below are the canonical ones for the
# ``sirotaA-jg-05-20120316`` session.

# %%
from __future__ import annotations

from pathlib import Path
import numpy as np

# ─────────────────────────────────────────────────────────────────── #
# Session metadata — parsed from get_session_list_v3.m, BehaviorPlaceCode #
# entry for jg05-20120316 (line 1340), with the full inheritance     #
# chain resolved back through the er01-20110719 base case.            #
# ─────────────────────────────────────────────────────────────────── #
#
# Original MATLAB sessionName: 'jg05-20120316'
# Translated to neurobox 4-part naming: 'sirotaA-jg-05-20120316'
#                                        sourceId-userId-subjectId-date

# ── Identity ─────────────────────────────────────────────────────── #
SESSION_NAME = "sirotaA-jg-05-20120316"
SUBJECT      = "jg05"             # MATLAB subject.name
PROJECT_ID   = "general"              # was MATLAB project = 'general'
DATA_ROOT    = Path("/data")
MAZE         = "cof"              # MATLAB mazeName
TRIAL_NAME   = "all"              # MATLAB trialName
DATA_LOGGERS = ["nlx", "vicon"]   # MATLAB dataLoggers / dLoggers

# ── Sync (TTL events in the .all.evt file) ───────────────────────── #
TTL_VALUE = "0x0040"              # MATLAB ttlValue — start-pulse label
STOP_TTL  = "0x0000"              # stop-pulse label (matched pair)
INCLUDE_SYNC_IND: list[int] = []  # MATLAB includeSyncInd — empty means
                                  # use ALL Vicon takes; non-empty would
                                  # be 1-indexed positions to keep

# ── Vicon native rate ────────────────────────────────────────────── #
EXPECTED_XYZ_SR = 119.881035      # MATLAB xyzSampleRate (Hz)
                                  # (auto-detected during sync; this is
                                  # for sanity-checking only)

# ── Trial-window offsets (added to / subtracted from the matched     #
# Vicon windows when defining the 'all' trial) ────────────────────── #
TRIAL_OFFSETS = (10.0, -10.0)     # MATLAB offsets [start_offs, stop_offs]
                                  # (seconds added to the first window
                                  # start and subtracted from the last
                                  # window stop, to skip startup/shutdown
                                  # transients)

# ── Spatial origin / arena offset (used by some xyz transforms) ──── #
X_OFFSET = -20.0                  # MATLAB xOffset
Y_OFFSET = -20.0                  # MATLAB yOffset
Z_OFFSET =   0.0                  # MATLAB zOffset
ROTATION =   0.0                  # MATLAB rotation (radians)

# ── Subject-specific channel groups for LFP analysis ─────────────── #
# These map to MATLAB ``Sessions(end).subject.channelGroup.*`` and
# tell downstream LFP code which channels are which.  jg05 had a
# 70-channel silicon probe; channels are 1-indexed in the MATLAB
# source — pass through to neurobox as-is unless your loader is
# 0-indexed, in which case subtract 1 below.
CH_THETA   = 68                   # primary theta-rhythm channel
CH_THETARC = (57, 64)             # reference channels for theta
                                  #   phase computation
CH_RIPPLE  = list(range(49, 57))  # MATLAB 49:56 → channels for
                                  #   ripple detection (CA1)

# ── Per-subject signal corrections (applied at feature-loading) ──── #
# MATLAB: ``Sessions(end).subject.correction.*``.
#
# These are constant per-subject offsets used to compensate for:
# - thetaPhase: drift of theta-LFP ref channel relative to nominal
#               (added to phz.data in transformations/load_thetarc_phase.m)
# - headYaw:    rotation of the head rigid-body frame around Z
# - headBody:   rotational offset of head relative to body
# - headRoll:   roll offset of head rigid body
# - headCenter: planar offset of the rigid-body centroid from the
#               geometric centre of the marker pattern.
THETA_PHASE_CORRECTION = 0.78539816   # ≈ π/4 rad
HEAD_YAW_CORRECTION    =  0.264       # rad
HEAD_BODY_CORRECTION   = -0.234       # rad
HEAD_ROLL_CORRECTION   = -0.365       # rad
HEAD_CENTER_OFFSET     = (0.0, 0.0)   # (x, y) in mocap units

# ── Anatomical-location flags (used by downstream classifiers) ───── #
# MATLAB: ``Sessions(end).subject.anatLoc.*``.  jg05 was a CA3
# recording — the resolved entry has CA3=False (i.e. NOT in CA3),
# meaning the silicon probe terminated above CA3 in this animal.
# Original interpretation may differ for your downstream code; the
# raw values from get_session_list_v3.m are preserved verbatim.
SUBJECT_IN_CA1 = None             # not set in the resolved entry
SUBJECT_IN_CA3 = False            # MATLAB anatLoc.CA3 (note: also note
                                  # that line 1370–1372 explicitly sets
                                  # CA3=true / DG=false for the next
                                  # session jg05-20120325, so this
                                  # entry inherits whatever was set
                                  # earlier in the chain)
SUBJECT_IN_DG  = None             # not set

# ── Pipeline tags ────────────────────────────────────────────────── #
STC_MODE = "msnn_ppsvd_raux"      # MATLAB stcMode — the trained
                                  # state-classifier model identifier;
                                  # neurobox uses this to find the
                                  # matching .stc.{stcMode}.pkl file

# ── Behaviour flags (script-control, not session metadata) ───────── #
LINK_SESSION = True               # cell 2 — populate project tree
RECREATE     = True               # cell 3 — re-run sync even if
                                  #   .ses.pkl exists
OPEN_GUI     = True               # cell 6 — open MTABrowser at the end

# ── Imports ──────────────────────────────────────────────────────── #
# Force Qt-aware matplotlib backend BEFORE pyplot import — otherwise
# matplotlib picks a tty backend on a headless terminal and the GUI
# breaks.
import matplotlib
matplotlib.use("QtAgg")

from neurobox.dtype import (
    NBSession, NBTrial, TrialWindow,
)


print(f"=== {SESSION_NAME} ===")
print(f"Project: {PROJECT_ID} under {DATA_ROOT}")
print(f"Maze   : {MAZE}")


# %% [markdown]
# # Step 1 — Link the session into the project tree
#
# Creates ``<DATA_ROOT>/project/<PROJECT_ID>/<SESSION_NAME>/`` with
# symlinks pointing at the processed ephys + mocap files.  No
# copying.  Idempotent — running it twice is fine.

# %%
if LINK_SESSION:
    from neurobox.config.config import link_session
    paths = link_session(
        session_name = SESSION_NAME,
        project_id   = PROJECT_ID,
        data_root    = DATA_ROOT,
        overwrite    = False,
        verbose      = True,
    )
    print(f"\nspath:    {paths.spath}")
    print(f"yaml:     {paths.yaml_file}")
    print(f"lfp:      {paths.lfp_file}")
else:
    print("Skipping link_session (LINK_SESSION = False)")


# %% [markdown]
# # Step 2 — Synchronise NLX ↔ Vicon
#
# Build the session by dispatching to ``sync_nlx_vicon``.  This is
# where the multi-segment Vicon takes get matched to TTL events on
# the master clock.
#
# After this cell:
#
# * ``session.window``                  → :class:`TrialWindow`
# * ``session.xyz.recording_windows``   → per-block ground truth
# * ``session.xyz.stream_sync``         → spanning :class:`StreamSync`
# * ``session.lfp.stream_sync``         → continuous master-clock sync
# * ``session.spk``                     → spikes
#
# When ``RECREATE = False`` and the .ses.pkl checkpoint exists we
# skip the sync and just load the cached state.  This makes the
# notebook cheap to re-run.

# %%
session = NBSession(
    session_name = SESSION_NAME,
    maze         = MAZE,
    project_id   = PROJECT_ID,
    data_root    = DATA_ROOT,
)
ses_pkl = session.spath / f"{session.filebase}.ses.pkl"

if ses_pkl.exists() and not RECREATE:
    print(f"Loading from checkpoint: {ses_pkl.name}")
    session.load()
    try:
        session.load("xyz")
    except FileNotFoundError as e:
        print(f"  [warn] xyz not on disk: {e}")
else:
    print(f"Running create({DATA_LOGGERS!r}, "
          f"ttl_value={TTL_VALUE!r})")
    session.create(
        data_loggers = DATA_LOGGERS,
        ttl_value    = TTL_VALUE,
        stop_ttl     = STOP_TTL,
        save_xyz     = True,
    )
print("Done.")


# %% [markdown]
# # Step 3 — Inspect the populated sync state
#
# Look at what the sync pipeline wrote.  The interesting ones for
# multi-segment Vicon are ``recording_windows`` (one row per Vicon
# take) and ``stream_sync`` (a single spanning segment, since the
# data array is session-frame-aligned with internal zero-fills).

# %%
print(f"--- session ---")
print(f"  filebase : {session.filebase}")
print(f"  spath    : {session.spath}")

if session.window is not None:
    w = session.window
    print(f"\n--- session.window ---")
    print(f"  TrialWindow: [{w.t_start:.2f}, {w.t_stop:.2f}] s "
          f"({w.total_duration:.1f} s total)")

if session.xyz is not None:
    xyz = session.xyz
    print(f"\n--- session.xyz ---")
    print(f"  shape : {xyz.data.shape}")
    print(f"  sr    : {xyz.samplerate} Hz")
    print(f"  duration: {xyz.duration:.2f} s")
    print(f"  markers : {xyz.markers}")

    if xyz.recording_windows is not None:
        rw = xyz.recording_windows
        print(f"\n  recording_windows: {len(rw)} blocks")
        for i, (t0, t1) in enumerate(rw):
            print(f"    block {i:2d}: [{t0:8.2f}, {t1:8.2f}] s  "
                  f"({(t1 - t0):.2f} s)")

    if xyz.stream_sync is not None:
        ss = xyz.stream_sync
        print(f"\n  stream_sync: master_first={ss.master_first:.2f} s, "
              f"master_last={ss.master_last:.2f} s, "
              f"span={ss.master_span:.2f} s")

if session.lfp is not None:
    print(f"\n--- session.lfp ---")
    print(f"  sr : {session.lfp.samplerate} Hz")
    if session.lfp.stream_sync is not None:
        ls = session.lfp.stream_sync
        print(f"  stream_sync: [{ls.master_first:.2f}, "
              f"{ls.master_last:.2f}] s "
              f"({ls.n_segments} segment(s))")

if session.spk is not None:
    print(f"\n--- session.spk ---")
    print(f"  units  : {session.spk.n_units}")
    print(f"  spikes : {len(session.spk)}")


# %% [markdown]
# # Step 4 — Build a Trial and load xyz, restricted to the trial window
#
# Build the canonical ``all`` trial for this session, applying the
# ``TRIAL_OFFSETS`` from the MATLAB session list (10 s pad-in at the
# start, 10 s pad-out at the end — to skip mocap-startup transients
# and the experimenter handling the animal at the end).  This window
# typically straddles one or more Vicon stop/restart gaps; the
# :meth:`NBData.restrict_to_window` machinery handles that
# transparently — gap regions are zero-filled in the output array,
# and ``trial.xyz.stream_sync`` reflects only the genuinely-recorded
# sub-segments.

# %%
if session.window is None or session.window.is_empty:
    print("[skip] session.window not populated — nothing to slice")
    trial = None
    trial_xyz = None
else:
    # Apply MATLAB-style offsets to the matched recording span.
    # MATLAB convention: offsets[0] is added to the first window's
    # start (positive = trim later); offsets[1] is added to the
    # last window's stop (negative = trim earlier).
    raw_t0  = session.window.t_start
    raw_t1  = session.window.t_stop
    sub_t0  = raw_t0 + TRIAL_OFFSETS[0]
    sub_t1  = raw_t1 + TRIAL_OFFSETS[1]
    if sub_t1 <= sub_t0:
        print(f"  [warn] offsets {TRIAL_OFFSETS} would empty the trial; "
              f"falling back to raw window")
        sub_t0, sub_t1 = raw_t0, raw_t1

    print(f"Raw session window  : [{raw_t0:.2f}, {raw_t1:.2f}] s")
    print(f"After trial offsets {TRIAL_OFFSETS}: "
          f"[{sub_t0:.2f}, {sub_t1:.2f}] s "
          f"({sub_t1 - sub_t0:.2f} s)")

    win = TrialWindow(
        periods = np.array([[sub_t0, sub_t1]], dtype=np.float64),
        label   = TRIAL_NAME,
        name    = TRIAL_NAME,
    )
    trial = NBTrial(
        session_name = SESSION_NAME,
        maze         = MAZE,
        trial_name   = TRIAL_NAME,
        project_id   = PROJECT_ID,
        data_root    = DATA_ROOT,
        window       = win,
    )
    trial_xyz = trial.load("xyz")

    expected_n = int(round((sub_t1 - sub_t0) * trial_xyz.samplerate))
    print(f"\ntrial.xyz.data.shape    : {trial_xyz.data.shape}")
    print(f"  expected ~{expected_n} samples = "
          f"{sub_t1 - sub_t0:.1f}s × {trial_xyz.samplerate}Hz")

    if trial_xyz.stream_sync is not None:
        valid = trial_xyz.stream_sync.valid_mask_in_window(
            sub_t0, sub_t1,
        )
        pct = 100 * valid.sum() / max(len(valid), 1)
        print(f"\ntrial.xyz.stream_sync:")
        print(f"  recorded sub-segments  : "
              f"{trial_xyz.stream_sync.n_segments}")
        for t0, t1 in trial_xyz.stream_sync.segments:
            print(f"    [{t0:8.2f}, {t1:8.2f}] s")
        print(f"  valid samples (Vicon on): "
              f"{int(valid.sum())} / {len(valid)} ({pct:.1f}%)")


# %% [markdown]
# # Step 5 — Quick visual check of the loaded data
#
# Plot one marker's trajectory across the trial window to confirm
# the data looks sensible.  Gap-filled regions show up as flat
# zeros — that's expected.

# %%
if trial_xyz is not None and trial_xyz.data.size > 0:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    t_axis = np.arange(trial_xyz.n_samples) / trial_xyz.samplerate
    marker_idx = 0
    marker_name = trial_xyz.markers[marker_idx]
    for ax, dim, label in zip(axes, range(3), "XYZ"):
        ax.plot(t_axis, trial_xyz.data[:, marker_idx, dim],
                 lw=0.8)
        ax.set_ylabel(f"{marker_name} {label}")
        # Shade gap regions
        if trial_xyz.stream_sync is not None:
            valid = trial_xyz.stream_sync.valid_mask_in_window(
                0.0, trial_xyz.duration,
            )
            ax.fill_between(
                t_axis, ax.get_ylim()[0], ax.get_ylim()[1],
                where = ~valid,
                alpha = 0.2, color="red",
                label = "Vicon off" if dim == 0 else None,
            )
    axes[0].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("time inside trial (s)")
    fig.suptitle(
        f"{SESSION_NAME} — {marker_name}  "
        f"(red = Vicon-off, gap-filled with zeros)"
    )
    fig.tight_layout()
    plt.show()
else:
    print("(no trial xyz to plot)")


# %% [markdown]
# # Step 6 — Open in MTABrowser
#
# Hand the loaded session to ``MTABrowserWindow``.  Three tabs:
# Data Management, Motion Labelling, and LFP States.  The LFP
# States tab auto-attaches if ``session.lfp`` has data; otherwise
# you can attach it later via ``win.attach_lfp(...)``.
#
# This cell blocks until you close the browser window.  Set
# ``OPEN_GUI = False`` in the first cell to skip it.

# %%
if OPEN_GUI:
    import sys
    from PySide6.QtWidgets import QApplication
    from neurobox.gui.mta_browser import MTABrowserWindow

    app = QApplication.instance() or QApplication(sys.argv)
    win = MTABrowserWindow.launch(session=session, run=False)
    print(f"Browser open at: {win.windowTitle()}")
    print(f"Tabs: " + ", ".join(
        win._tabs.tabText(i) for i in range(win._tabs.count())
    ))
    print("Close the browser window to continue.")
    app.exec()
else:
    print("Skipping GUI (OPEN_GUI = False)")


# %% [markdown]
# # Optional — Sandbox cells for further exploration
#
# Below this point are scratch cells for ad-hoc work.  They use the
# variables defined above (``session``, ``trial``, ``trial_xyz``)
# without redefining anything.

# %% [markdown]
# ## Compute basic kinematic features

# %%
# Uncomment to compute features:
#
# from neurobox.analysis.kinematics import augment_xyz, features
#
# fx = augment_xyz(session.xyz)
# feats = features(fx)
# print(list(feats.keys()))
# print({k: feats[k].shape for k in feats})


# %% [markdown]
# ## Inspect a state collection (if you've already labelled some)

# %%
# Uncomment to load and inspect labelled states:
#
# try:
#     session.load("stc")
#     print(f"{len(session.stc.list_states())} state(s):")
#     for label in session.stc.list_states():
#         ep = session.stc.get_state(label)
#         print(f"  {label:<10s}  {ep.n_periods:4d} periods  "
#               f"total {ep.duration:.1f} s")
# except FileNotFoundError:
#     print("No .stc file yet — label some states in the browser first")


# %% [markdown]
# ## LFP — preselect this session's theta / ripple channels
#
# The MATLAB session-list entry recorded which probe channels are
# the canonical theta-rhythm channel, theta-reference pair, and
# ripple-detection channels.  Use them when extracting LFP for
# downstream analysis.

# %%
# Uncomment to load LFP and slice out the configured channels:
#
# session.load("lfp")
# lfp = session.lfp.data            # shape (n_samples, n_channels)
# theta_trace   = lfp[:, CH_THETA - 1]                      # 1→0 indexing
# thetarc_trace = lfp[:, [c - 1 for c in CH_THETARC]]
# ripple_traces = lfp[:, [c - 1 for c in CH_RIPPLE]]
# print(f"theta channel {CH_THETA}:    "
#       f"shape {theta_trace.shape}")
# print(f"theta-ref channels {CH_THETARC}: "
#       f"shape {thetarc_trace.shape}")
# print(f"ripple channels {CH_RIPPLE}: "
#       f"shape {ripple_traces.shape}")


# %% [markdown]
# ## Apply per-subject head/theta corrections
#
# The MATLAB pipeline adds these constants at feature-loading time
# rather than baking them into the saved data — so any neurobox
# port of ``fet_href_HXY``, ``fet_hba``, ``load_thetarc_phase``,
# etc. should consume them identically.

# %%
# Example — applying THETA_PHASE_CORRECTION to a phase trace:
#
# from neurobox.analysis.lfp.spectral import (
#     SpectralParams, multitaper_spectrogram,
# )
# # ... compute theta-phase phz from session.lfp[:, CH_THETA-1] ...
# phz_corrected = np.mod(phz + THETA_PHASE_CORRECTION + 2*np.pi,
#                          2*np.pi)
#
# Example — applying HEAD_YAW_CORRECTION when computing head
# direction:
#
# from neurobox.analysis.kinematics import head_direction
# raw_yaw = head_direction(session.xyz)
# yaw_corrected = raw_yaw + HEAD_YAW_CORRECTION


# %% [markdown]
# ## Print the resolved metadata as a single dict (for logging)

# %%
metadata = {
    "session_name":          SESSION_NAME,
    "subject":               SUBJECT,
    "project_id":            PROJECT_ID,
    "maze":                  MAZE,
    "trial_name":            TRIAL_NAME,
    "data_loggers":          DATA_LOGGERS,
    "ttl_value":             TTL_VALUE,
    "stop_ttl":              STOP_TTL,
    "expected_xyz_sr":       EXPECTED_XYZ_SR,
    "trial_offsets":         TRIAL_OFFSETS,
    "spatial_offset":        (X_OFFSET, Y_OFFSET, Z_OFFSET),
    "rotation":              ROTATION,
    "channels": {
        "theta":   CH_THETA,
        "thetarc": CH_THETARC,
        "ripple":  CH_RIPPLE,
    },
    "corrections": {
        "theta_phase": THETA_PHASE_CORRECTION,
        "head_yaw":    HEAD_YAW_CORRECTION,
        "head_body":   HEAD_BODY_CORRECTION,
        "head_roll":   HEAD_ROLL_CORRECTION,
        "head_center": HEAD_CENTER_OFFSET,
    },
    "anat_loc": {
        "CA1": SUBJECT_IN_CA1,
        "CA3": SUBJECT_IN_CA3,
        "DG":  SUBJECT_IN_DG,
    },
    "stc_mode":              STC_MODE,
}
import json
print(json.dumps(metadata, indent=2, default=str))
