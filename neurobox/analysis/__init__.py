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
)
from .transform_origin  import transform_origin, TransformResult

__all__ = [
    "neuron_quality", "NeuronQualityResult",
    "transform_origin", "TransformResult",
]
