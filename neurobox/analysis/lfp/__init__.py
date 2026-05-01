"""
neurobox.analysis.lfp
=====================
LFP signal processing and spectral analysis.
"""
from .spectral import (
    SpectralParams,
    SpectrumResult,
    multitaper_spectrogram,
    multitaper_coherogram,
    multitaper_cross_spectrogram,
    multitaper_psd,
    whiten_ar,
    fet_spec,
)
from .filtering import (
    butter_filter,
    filter0,
    fir_filter,
)
from .oscillations import (
    OscillationResult,
    detect_oscillations,
    detect_ripples,
    local_minima,
    thresh_cross,
    within_ranges,
)
from .csd import (
    CSDResult,
    current_source_density,
)
from .ranges import (
    join_ranges,
    intersect_ranges,
    subtract_ranges,
    complement_ranges,
)

__all__ = [
    # spectral
    "SpectralParams",
    "SpectrumResult",
    "multitaper_spectrogram",
    "multitaper_coherogram",
    "multitaper_cross_spectrogram",
    "multitaper_psd",
    "whiten_ar",
    "fet_spec",
    # filtering
    "butter_filter",
    "filter0",
    "fir_filter",
    # oscillations
    "OscillationResult",
    "detect_oscillations",
    "detect_ripples",
    "local_minima",
    "thresh_cross",
    "within_ranges",
    # csd
    "CSDResult",
    "current_source_density",
    # ranges
    "join_ranges",
    "intersect_ranges",
    "subtract_ranges",
    "complement_ranges",
]
