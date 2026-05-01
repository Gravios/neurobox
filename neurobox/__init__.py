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
    __version__ = "0.1.2"

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
    NBDang,
    NBDufr,
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
    get_lfp_samplerate,
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
    load_units,
    UnitAnnotation,
    map_annotations_to_global_ids,
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
from neurobox.analysis import (
    neuron_quality, NeuronQualityResult,
    transform_origin, TransformResult,
    SpectralParams, SpectrumResult,
    multitaper_spectrogram, multitaper_coherogram,
    multitaper_cross_spectrogram, multitaper_psd,
    whiten_ar, fet_spec,
    butter_filter, filter0, fir_filter,
    OscillationResult, detect_oscillations, detect_ripples,
    local_minima, thresh_cross, within_ranges,
    CSDResult, current_source_density,
    join_ranges, intersect_ranges, subtract_ranges, complement_ranges,
    RayleighResult,
    circ_mean, circ_r,
    rayleigh_test, ppc,
    von_mises_fit, von_mises_pdf, von_mises_rvs,
    bessel_ratio_inverse,
    FDRResult, fdr_bh,
    BinSmoothResult, bin_smooth,
    ccg, trains_to_ccg, CCGResult,
    occupancy_map, OccupancyResult,
    place_field, PlaceFieldResult,
    place_field_stats, Patch, UnitStats,
)

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
    "NBDang",
    "NBDufr",
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
    "get_lfp_samplerate",
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
    "transform_origin",
    "TransformResult",
    "SpectralParams",
    "SpectrumResult",
    "multitaper_spectrogram",
    "multitaper_coherogram",
    "multitaper_cross_spectrogram",
    "multitaper_psd",
    "whiten_ar",
    "fet_spec",
    "butter_filter",
    "filter0",
    "fir_filter",
    "OscillationResult",
    "detect_oscillations",
    "detect_ripples",
    "local_minima",
    "thresh_cross",
    "within_ranges",
    "CSDResult",
    "current_source_density",
    "join_ranges",
    "intersect_ranges",
    "subtract_ranges",
    "complement_ranges",
    "RayleighResult",
    "circ_mean",
    "circ_r",
    "rayleigh_test",
    "ppc",
    "von_mises_fit",
    "von_mises_pdf",
    "von_mises_rvs",
    "bessel_ratio_inverse",
    "FDRResult",
    "fdr_bh",
    "BinSmoothResult",
    "bin_smooth",
    "ccg",
    "trains_to_ccg",
    "CCGResult",
    # spatial
    "occupancy_map",
    "OccupancyResult",
    "place_field",
    "PlaceFieldResult",
    "place_field_stats",
    "Patch",
    "UnitStats",
    # hmm — requires `pip install 'neurobox[hmm]'`
    "gauss_hmm",
    "HMMResult",
    # unit annotations
    "load_units",
    "UnitAnnotation",
    "map_annotations_to_global_ids",
]


# ─────────────────────────────────────────────────────────────────────────── #
# Lazy HMM import at top level                                                #
# ─────────────────────────────────────────────────────────────────────────── #

def __getattr__(name: str):
    if name in ("gauss_hmm", "HMMResult"):
        from .analysis.stats import hmm as _hmm
        return getattr(_hmm, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
