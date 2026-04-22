"""
neurobox.analysis
=================
Post-processing and quality-assessment modules.
"""
from .neuron_quality import neuron_quality, NeuronQualityResult

__all__ = ["neuron_quality", "NeuronQualityResult"]
