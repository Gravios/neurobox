"""
neurobox.analysis.stats
========================
Statistical primitives — circular statistics, multiple-comparison
correction, smoothing, HMM utilities.

The :func:`gauss_hmm` / :class:`HMMResult` symbols require the
optional ``hmmlearn`` dependency.  They're imported lazily so the rest
of the module loads without it.  Install with::

    pip install 'neurobox[hmm]'
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
    # CircStat2012a additions
    circ_dist, circ_dist2,
    circ_var, circ_std,
    circ_median,
    circ_kappa,
    circ_moment,
    circ_skewness, circ_kurtosis,
    circ_axial,
    circ_ang2rad, circ_rad2ang,
)
from .multcomp import (
    FDRResult,
    fdr_bh,
)
from .smoothing import (
    BinSmoothResult,
    bin_smooth,
)


# ─────────────────────────────────────────────────────────────────────────── #
# Lazy HMM import — only triggered if the user actually accesses these names. #
# The import inside gauss_hmm itself raises a helpful ImportError when        #
# hmmlearn is missing.                                                         #
# ─────────────────────────────────────────────────────────────────────────── #

def __getattr__(name: str):
    if name in ("gauss_hmm", "HMMResult"):
        from . import hmm as _hmm
        return getattr(_hmm, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
    # CircStat2012a — Berens's MATLAB Circular Statistics Toolbox
    "circ_dist", "circ_dist2",
    "circ_var", "circ_std",
    "circ_median",
    "circ_kappa",
    "circ_moment",
    "circ_skewness", "circ_kurtosis",
    "circ_axial",
    "circ_ang2rad", "circ_rad2ang",
    # multcomp
    "FDRResult",
    "fdr_bh",
    # smoothing
    "BinSmoothResult",
    "bin_smooth",
    # hmm (lazy — requires `pip install 'neurobox[hmm]'`)
    "gauss_hmm",
    "HMMResult",
]
