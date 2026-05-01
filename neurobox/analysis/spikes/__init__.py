"""
neurobox.analysis.spikes
=========================
Spike-train analysis: cross-correlograms, phase locking, ISI statistics.
"""

from .ccg import ccg, trains_to_ccg, CCGResult, is_compiled

__all__ = [
    "ccg",
    "trains_to_ccg",
    "CCGResult",
    "is_compiled",
]
