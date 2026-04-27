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

__all__ = [
    "SpectralParams",
    "SpectrumResult",
    "multitaper_spectrogram",
    "multitaper_coherogram",
    "multitaper_cross_spectrogram",
    "multitaper_psd",
    "whiten_ar",
    "fet_spec",
]
