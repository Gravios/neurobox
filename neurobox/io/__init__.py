"""
neurobox.io
===========
I/O routines for Neurosuite-3 data formats.

Quick reference
---------------
YAML parameter files (neurosuite-3)::

    from neurobox.io import load_par, load_yaml, get_channel_groups
    par = load_par("session")
    ch_groups = get_channel_groups(par)        # [[ch0,ch1,...], ...]

Binary data (wideband .dat / LFP .lfp)::

    from neurobox.io import load_binary
    data = load_binary("session.dat", channels=[0,1,2,3], uv_per_bit=0.195)

Spike times (.res.N / .clu.N — binary, neurosuite-3)::

    from neurobox.io import load_clu_res, spikes_by_unit
    res, clu, shank_map = load_clu_res("session", as_seconds=True)
    spikes = spikes_by_unit(res, clu)          # dict[unit_id → times]

Spike waveforms (.spk.N)::

    from neurobox.io import load_spk, load_spk_from_par
    wf = load_spk_from_par("session", shank=1)  # (n_spikes, n_samp, n_ch)

Event files (.evt)::

    from neurobox.io import load_evt, evt_to_periods
    ts, labels = load_evt("session.evt")
    periods    = evt_to_periods(ts, labels, "Stim on", "Stim off")

TTL event queries (convenience wrappers)::

    from neurobox.io import get_event_times, get_ttl_periods
    times   = get_event_times(session, "0x0040")
    periods = get_ttl_periods(session, "0x0040", "0x0000")

Neuralynx event files (.nev)::

    from neurobox.io import parse_nlx_events
    ttl_times = parse_nlx_events("Events.nev", "TTL Input on AcqSystem")

Processed mocap .mat files (MTA processC3D output)::

    from neurobox.io import load_processed_mat, concatenate_processed_mat
    xyz, markers, sr = load_processed_mat("session_trial001.mat")
    chunks, markers, sr = concatenate_processed_mat("spath/cof/")

Position tracking (Motive CSV)::

    from neurobox.io import load_position_motive_csv
    df = load_position_motive_csv("tracking.csv")

Gap filling (motion-capture dropouts)::

    from neurobox.io import fill_gaps, fill_xyz_gaps, detect_gaps
    xyz_clean = fill_gaps(xyz_arr, samplerate=120., max_gap_sec=0.3)
    fill_xyz_gaps(session.xyz, max_gap_sec=0.3)       # in-place on NBDxyz
"""

from .load_par                 import load_par
from .load_yaml                import load_yaml, get_channel_groups, get_lfp_samplerate
from .load_binary              import load_binary
from .load_clu_res             import load_clu_res, spikes_by_unit
from .load_spk                 import load_spk, load_spk_from_par
from .load_evt                 import load_evt, evt_to_periods
from .parse_events             import parse_nlx_events
from .parse_events             import mocap_events as sync
from .get_event_times          import get_event_times, get_ttl_periods
from .load_processed_mat       import (load_processed_mat,
                                       concatenate_processed_mat)
from .load_position_motive_csv import load_position_motive_csv
from .xyz_fill_gaps            import fill_gaps, fill_xyz_gaps, detect_gaps
from .load_units               import load_units, UnitAnnotation, map_annotations_to_global_ids

__all__ = [
    # Parameter files
    "load_par",
    "load_yaml",
    "get_channel_groups",
    "get_lfp_samplerate",
    # Binary signal data
    "load_binary",
    # Spikes
    "load_clu_res",
    "spikes_by_unit",
    # Waveforms
    "load_spk",
    "load_spk_from_par",
    # Events
    "load_evt",
    "evt_to_periods",
    # TTL convenience
    "get_event_times",
    "get_ttl_periods",
    # Neuralynx
    "parse_nlx_events",
    "sync",
    # Processed mocap (.mat)
    "load_processed_mat",
    "concatenate_processed_mat",
    # Position tracking (CSV)
    "load_position_motive_csv",
    # Gap filling
    "fill_gaps",
    "fill_xyz_gaps",
    "detect_gaps",
    # Unit annotations
    "load_units",
    "UnitAnnotation",
    "map_annotations_to_global_ids",
]
