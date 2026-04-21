"""
neurobox.io
===========
I/O routines for Neurosuite and neurosuite-3 data formats.

Quick reference
---------------
Binary data (wideband .dat / LFP .lfp)::

    from neurobox.io import load_binary
    data = load_binary("session.dat", channels=[0,1,2,3], uv_per_bit=0.195)

Spike times (.res.N / .clu.N — text or binary)::

    from neurobox.io import load_clu_res, spikes_by_unit
    res, clu, shank_map = load_clu_res("session", as_seconds=True)
    spikes = spikes_by_unit(res, clu)          # dict[unit_id → times]

Spike waveforms (.spk.N)::

    from neurobox.io import load_spk, load_spk_from_par
    wf = load_spk_from_par("session", shank=1)  # (n_spikes, n_samp, n_ch)

Parameter files (.yaml / .xml / .par)::

    from neurobox.io import load_par
    par = load_par("session")                  # auto-detects format

YAML parameter files (neurosuite-3)::

    from neurobox.io import load_yaml, get_channel_groups
    par = load_yaml("session.yaml")
    ch_groups = get_channel_groups(par)        # [[ch0,ch1,...], ...]

Event files (.evt)::

    from neurobox.io import load_evt, evt_to_periods
    ts, labels = load_evt("session.evt")
    periods    = evt_to_periods(ts, labels, "Stim on", "Stim off")

Neuralynx event files (.nev)::

    from neurobox.io import parse_nlx_events, sync
    ttl_times = parse_nlx_events("Events.nev", "TTL Input on AcqSystem")

Position tracking (Motive CSV)::

    from neurobox.io import load_position_motive_csv
    df = load_position_motive_csv("tracking.csv")
"""

from .load_par        import load_par
from .load_xml        import load_xml
from .load_yaml       import load_yaml, get_channel_groups
from .load_binary     import load_binary
from .load_clu_res    import load_clu_res, spikes_by_unit
from .load_spk        import load_spk, load_spk_from_par
from .load_evt        import load_evt, evt_to_periods
from .parse_events    import parse_nlx_events
from .parse_events    import mocap_events as sync
from .load_position_motive_csv import load_position_motive_csv

__all__ = [
    # Parameter files
    "load_par",
    "load_xml",
    "load_yaml",
    "get_channel_groups",
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
    # Neuralynx
    "parse_nlx_events",
    "sync",
    # Position tracking
    "load_position_motive_csv",
]
