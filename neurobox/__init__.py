"""
neurobox
========
Analysis toolbox for silicon-probe electrophysiology, integrated with
neurosuite-3 (github.com/Gravios/neurosuite-3).

Submodules
----------
neurobox.io       — loading .dat / .lfp / .res / .clu / .spk / .evt /
                    .xml / .yaml session parameter files
neurobox.dtype    — core data-structure types (Struct, NBSession, NBData)
neurobox.config   — project path configuration
neurobox.utils    — sync utilities, file helpers
"""

from .io import *
from .dtype import *
