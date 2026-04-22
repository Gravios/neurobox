"""
neurobox
========
Analysis toolbox for silicon-probe electrophysiology, integrated with
neurosuite-3.

Submodules
----------
neurobox.io       — I/O for .yaml / .dat / .lfp / .res / .clu / .spk / .evt
neurobox.dtype    — NBSession, NBTrial, NBSpk, NBDxyz, NBDlfp, NBEpoch,
                    NBStateCollection, NBModel, NBSessionPaths, …
neurobox.config   — configure_project, link_session, load_config
neurobox.utils    — sync utilities, file helpers
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("neurobox")
except PackageNotFoundError:
    __version__ = "0.1.0-dev"

# Public re-exports — keep this list explicit so `from neurobox import *`
# gives a clean, predictable namespace.

from neurobox.dtype import (
    Struct,
    NBEpoch,
    select_periods,
    NBData,
    NBModel,
    NBSpk,
    NBDxyz,
    NBDlfp,
    NBStateCollection,
    NBSessionPaths,
    parse_session_name,
    build_session_name,
    NBSession,
    NBTrial,
)

from neurobox.io import (
    load_par,
    load_yaml,
    get_channel_groups,
    load_binary,
    load_clu_res,
    spikes_by_unit,
    load_spk,
    load_spk_from_par,
    load_evt,
    evt_to_periods,
    parse_nlx_events,
    get_event_times,
    get_ttl_periods,
    load_processed_mat,
    concatenate_processed_mat,
    fill_gaps,
    fill_xyz_gaps,
    detect_gaps,
)

from neurobox.config.config import (
    configure_project,
    discover_mazes,
    link_session,
    link_sessions,
    link_session_status,
    load_config,
)
from neurobox.pipelines import (
    quick_session_setup,
    batch_session_setup,
    quick_trial_setup,
    batch_trial_setup,
)
from neurobox.analysis import neuron_quality, NeuronQualityResult

__all__ = [
    "__version__",
    # dtype
    "Struct",
    "NBEpoch",
    "select_periods",
    "NBData",
    "NBModel",
    "NBSpk",
    "NBDxyz",
    "NBDlfp",
    "NBStateCollection",
    "NBSessionPaths",
    "parse_session_name",
    "build_session_name",
    "NBSession",
    "NBTrial",
    # io
    "load_par",
    "load_yaml",
    "get_channel_groups",
    "load_binary",
    "load_clu_res",
    "spikes_by_unit",
    "load_spk",
    "load_spk_from_par",
    "load_evt",
    "evt_to_periods",
    "parse_nlx_events",
    "get_event_times",
    "get_ttl_periods",
    "load_processed_mat",
    "concatenate_processed_mat",
    "fill_gaps",
    "fill_xyz_gaps",
    "detect_gaps",
    # config
    "configure_project",
    "discover_mazes",
    "link_session",
    "link_sessions",
    "link_session_status",
    "load_config",
    # pipelines
    "quick_session_setup",
    "batch_session_setup",
    "quick_trial_setup",
    "batch_trial_setup",
    # analysis
    "neuron_quality",
    "NeuronQualityResult",
]
