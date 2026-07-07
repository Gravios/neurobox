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

Spike waveforms (.spk[.method].N)::

    from neurobox.io import load_spk, load_spk_from_par
    wf = load_spk_from_par("session", shank=1)  # (n_spikes, n_samp, n_ch)
    # .spk is Shared under neurosuite-3 naming — falls back to legacy names

Spike features (.fet.<method>.N — binary, neurosuite-3)::

    from neurobox.io import load_fet
    fet = load_fet("session.fet.standard.1")  # FetData(features, timestamps, n_dimensions)

PCA basis (.pca.<method>.N — binary, neurosuite-3)::

    from neurobox.io import load_pca
    basis = load_pca("session.pca.standard.1", n_samples=32)   # PcaBasis(means, eigenvectors, ...)

Hierarchical clustering siblings (.clc.<method>.N / .clp.<method>.N)::

    from neurobox.io import (
        load_clc, load_clp, build_atom_to_fiber, build_fiber_to_atoms,
    )
    atoms    = load_clc("session.clc.standard.1")               # per-spike atom IDs
    clp_map  = load_clp("session.clp.standard.1")               # (parent_of, header)
    a_to_f   = build_atom_to_fiber(clp_map)                     # dict[atom_id → fiber_id]
    f_to_as  = build_fiber_to_atoms(clp_map)                    # dict[fiber_id → [atom_ids]]

Writing neurosuite-3 files (byte-exact, round-trip with loaders)::

    from neurobox.io import (
        save_res, save_clu, save_clc, save_clp,
        save_spk, save_fet, save_pca,
    )
    # Path naming lives in NBSessionPaths — pass the resolved path in:
    save_fet(p.fet_file(1, method="stderiv"), features, timestamps)
    save_pca(p.pca_file(1, method="stderiv"), means, eigenvectors)
    save_spk(p.spk_file(1), waveforms)             # .spk is Shared — standard tag
    save_clu(p.clu_ns3_file(1), cluster_ids)
    save_clc(p.clc_file(1), atom_ids)
    save_clp(p.clp_file(1), parent_of)
    save_res(p.res_ns3_file(1), timestamps)

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
from .load_fet                 import load_fet, FetData
from .load_pca                 import load_pca, PcaBasis
from .load_clc_clp             import (
    load_clc, load_clp, ClpMap,
    build_atom_to_fiber, build_fiber_to_atoms,
)
from .ns3_writers              import (
    save_res, save_clu, save_clc, save_clp,
    save_spk, save_fet, save_pca,
)
from .load_evt                 import load_evt, evt_to_periods
from .parse_events             import parse_nlx_events
from .parse_events             import mocap_events as sync
from .get_event_times          import get_event_times, get_ttl_periods
from .load_processed_mat       import (load_processed_mat,
                                       concatenate_processed_mat)
from .load_position_motive_csv import load_position_motive_csv
from .xyz_fill_gaps            import fill_gaps, fill_xyz_gaps, detect_gaps
from .load_units               import load_units, UnitAnnotation, map_annotations_to_global_ids
from .data_hash                 import data_hash
from .cached_compute            import cached_compute, cache_path_for

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
    # PCA features + basis (neurosuite-3)
    "load_fet",
    "FetData",
    "load_pca",
    "PcaBasis",
    # Hierarchical clustering siblings
    "load_clc",
    "load_clp",
    "ClpMap",
    "build_atom_to_fiber",
    "build_fiber_to_atoms",
    # Neurosuite-3 writers (round-trip with load_*)
    "save_res",
    "save_clu",
    "save_clc",
    "save_clp",
    "save_spk",
    "save_fet",
    "save_pca",
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
    # Hashing for cache keys
    "data_hash",
    "cached_compute",
    "cache_path_for",
]
