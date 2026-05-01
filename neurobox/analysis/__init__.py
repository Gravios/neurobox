"""
neurobox.analysis
=================
Post-processing and quality-assessment modules.
"""
from .neuron_quality    import neuron_quality, NeuronQualityResult
from .lfp               import (
    SpectralParams, SpectrumResult,
    multitaper_spectrogram, multitaper_coherogram,
    multitaper_cross_spectrogram, multitaper_psd,
    whiten_ar, fet_spec,
    butter_filter, filter0, fir_filter,
    OscillationResult, detect_oscillations, detect_ripples,
    local_minima, thresh_cross, within_ranges,
    CSDResult, current_source_density,
    join_ranges, intersect_ranges, subtract_ranges, complement_ranges,
)
from .stats             import (
    RayleighResult,
    circ_mean, circ_r,
    rayleigh_test, ppc,
    von_mises_fit, von_mises_pdf, von_mises_rvs,
    bessel_ratio_inverse,
    FDRResult, fdr_bh,
    BinSmoothResult, bin_smooth,
)
from .spikes            import (
    ccg, trains_to_ccg, CCGResult,
)
from .spatial           import (
    occupancy_map, OccupancyResult,
    place_field, PlaceFieldResult,
)
from .transform_origin  import transform_origin, TransformResult


# ─────────────────────────────────────────────────────────────────────────── #
# Lazy HMM import — see neurobox.analysis.stats.__init__ for rationale.       #
# ─────────────────────────────────────────────────────────────────────────── #

def __getattr__(name: str):
    if name in ("gauss_hmm", "HMMResult"):
        from .stats import hmm as _hmm
        return getattr(_hmm, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "neuron_quality", "NeuronQualityResult",
    "transform_origin", "TransformResult",
    "SpectralParams", "SpectrumResult",
    "multitaper_spectrogram", "multitaper_coherogram",
    "multitaper_cross_spectrogram", "multitaper_psd",
    "whiten_ar", "fet_spec",
    "butter_filter", "filter0", "fir_filter",
    "OscillationResult", "detect_oscillations", "detect_ripples",
    "local_minima", "thresh_cross", "within_ranges",
    "CSDResult", "current_source_density",
    "join_ranges", "intersect_ranges", "subtract_ranges", "complement_ranges",
    "RayleighResult",
    "circ_mean", "circ_r",
    "rayleigh_test", "ppc",
    "von_mises_fit", "von_mises_pdf", "von_mises_rvs",
    "bessel_ratio_inverse",
    "FDRResult", "fdr_bh",
    "BinSmoothResult", "bin_smooth",
    "ccg", "trains_to_ccg", "CCGResult",
    "occupancy_map", "OccupancyResult",
    "place_field", "PlaceFieldResult",
    # hmm — requires `pip install 'neurobox[hmm]'`
    "gauss_hmm", "HMMResult",
]
