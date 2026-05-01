"""
neurobox.analysis.stats
========================
Statistical primitives — circular statistics, multiple-comparison
correction, smoothing, HMM utilities.
"""

from .circular import (
    RayleighResult,
    circ_mean,
    circ_r,
    rayleigh_test,
    ppc,
    von_mises_fit,
    von_mises_pdf,
    von_mises_rvs,
    bessel_ratio_inverse,
)
from .multcomp import (
    FDRResult,
    fdr_bh,
)
from .smoothing import (
    BinSmoothResult,
    bin_smooth,
)

__all__ = [
    # circular
    "RayleighResult",
    "circ_mean",
    "circ_r",
    "rayleigh_test",
    "ppc",
    "von_mises_fit",
    "von_mises_pdf",
    "von_mises_rvs",
    "bessel_ratio_inverse",
    # multcomp
    "FDRResult",
    "fdr_bh",
    # smoothing
    "BinSmoothResult",
    "bin_smooth",
]
