"""
neurobox.analysis.stats.multcomp
=================================
Multiple-comparison correction.

Port of :file:`labbox/Stats/multcomp/fdr_bh.m` (David Groppe, public
domain, BSD).  Implements the Benjamini-Hochberg (1995) and
Benjamini-Yekutieli (2001) procedures for controlling the false
discovery rate (FDR) of a family of hypothesis tests.

==========================  =============================================
labbox                       neurobox
==========================  =============================================
Stats/multcomp/fdr_bh.m      :func:`fdr_bh`
Stats/multcomp/FDR.m         (subsumed — same algorithm)
==========================  =============================================

Comparison with :mod:`scipy.stats`
----------------------------------
:func:`scipy.stats.false_discovery_control` (added in SciPy 1.11)
implements the same procedure.  This port differs in three ways:

1. **Return signature** — ours matches the labbox 4-tuple
   ``(h, crit_p, adj_ci_cvrg, adj_p)`` so existing MATLAB call sites
   port over with minimal changes.
2. **Multidimensional input** — preserves the input shape on output
   (scipy returns 1-D adjusted p-values regardless).
3. **'pdep' / 'dep'** keywords — match the labbox vocabulary.

If you don't need labbox-compatible return values, prefer the scipy
function for new code.

References
----------
* Benjamini, Y. & Hochberg, Y. (1995).  *J. R. Stat. Soc. B* 57:289-300.
* Benjamini, Y. & Yekutieli, D. (2001).  *Ann. Statist.* 29:1165-1188.
* Benjamini, Y. & Yekutieli, D. (2005).  *J. Am. Stat. Assoc.* 100:71-81.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class FDRResult:
    """Output of :func:`fdr_bh`.

    Attributes
    ----------
    h:
        Boolean array, same shape as ``pvals``.  ``True`` where the null
        hypothesis is rejected at FDR level ``q``.
    crit_p:
        Critical p-value — all ``pvals[i] ≤ crit_p`` are significant.
        ``0.0`` when no p-values are significant.
    adj_ci_cvrg:
        FCR-adjusted confidence-interval coverage for the rejected
        hypotheses (Benjamini & Yekutieli, 2005).  Use this as the
        coverage when constructing CIs for parameters whose tests
        rejected.  ``NaN`` if no rejections.
    adj_p:
        FDR-adjusted p-values, same shape as ``pvals``.  All
        ``adj_p[i] ≤ q`` are significant.  Note: adjusted p-values can
        exceed 1 under the BY (``method='dep'``) procedure.
    """

    h:           np.ndarray
    crit_p:      float
    adj_ci_cvrg: float
    adj_p:       np.ndarray


def fdr_bh(
    pvals:  np.ndarray,
    q:      float = 0.05,
    method: str = "pdep",
) -> FDRResult:
    """Benjamini-Hochberg FDR correction.

    Port of :file:`labbox/Stats/multcomp/fdr_bh.m`.

    Parameters
    ----------
    pvals:
        Array of p-values.  Any shape.  All values must be in ``[0, 1]``.
    q:
        Target false discovery rate.  Default 0.05.
    method:
        ``'pdep'`` (default) — original Benjamini-Hochberg.  Valid
        when tests are **independent** or **positively dependent**
        (PRDS — e.g. positively-correlated Gaussians).

        ``'dep'`` — Benjamini-Yekutieli variant.  Valid for **arbitrary
        dependence**, but less powerful (the threshold is divided by
        ``Σ_{i=1}^m 1/i ≈ ln(m)``).  Use this when you can't justify
        positive dependence — e.g. correlations of unknown sign.

    Returns
    -------
    :class:`FDRResult`

    Examples
    --------
    Basic usage::

        import numpy as np
        from neurobox.analysis.stats import fdr_bh

        pvals = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205])
        result = fdr_bh(pvals, q=0.05)
        print(result.h)        # [ True  True False False False False False False]
        print(result.crit_p)   # 0.008
        print(result.adj_p)

    Two-dimensional p-value matrix from a per-channel × per-event test::

        # pvals.shape = (n_channels, n_events)
        result = fdr_bh(pvals, q=0.01, method='dep')
        # result.h has the same (n_channels, n_events) shape
    """
    pvals = np.asarray(pvals, dtype=np.float64)
    if pvals.size == 0:
        raise ValueError("pvals is empty")
    if np.any(pvals < 0) or np.any(pvals > 1):
        raise ValueError("pvals must be in [0, 1]")
    if method not in ("pdep", "dep"):
        raise ValueError(f"method must be 'pdep' or 'dep', got {method!r}")
    if not (0 < q < 1):
        raise ValueError(f"q must be in (0, 1), got {q}")

    original_shape = pvals.shape
    flat = pvals.ravel()
    m = flat.size

    sort_idx   = np.argsort(flat, kind="stable")
    p_sorted   = flat[sort_idx]
    unsort_idx = np.argsort(sort_idx, kind="stable")
    rank       = np.arange(1, m + 1, dtype=np.float64)

    if method == "pdep":
        thresh = rank * q / m
        wtd_p  = m * p_sorted / rank
    else:  # 'dep' — Benjamini-Yekutieli
        denom = m * np.sum(1.0 / rank)
        thresh = rank * q / denom
        wtd_p  = denom * p_sorted / rank

    # ── Adjusted p-values ──────────────────────────────────────────────── #
    # Cumulative running minimum from the right, applied to wtd_p.
    # MATLAB uses a more obscure "fill-from-sorted-wtd" loop that's
    # equivalent.  np.minimum.accumulate on the reversed array is the
    # standard idiom and is faster.
    adj_p_sorted = np.minimum.accumulate(wtd_p[::-1])[::-1]
    # Cap at 1.0 for the BH ('pdep') case to match scipy / R p.adjust.
    # The labbox code does NOT cap (and notes adj_p > 1 can occur for
    # 'dep'); we cap only when it makes statistical sense (the BH
    # procedure produces values that should be valid p-values).
    if method == "pdep":
        adj_p_sorted = np.minimum(adj_p_sorted, 1.0)
    adj_p_flat = adj_p_sorted[unsort_idx]
    adj_p = adj_p_flat.reshape(original_shape)

    # ── Rejection set & critical p-value ──────────────────────────────── #
    rej = p_sorted <= thresh
    if np.any(rej):
        max_id      = int(np.flatnonzero(rej)[-1])
        crit_p      = float(p_sorted[max_id])
        h           = (pvals <= crit_p)
        adj_ci_cvrg = float(1.0 - thresh[max_id])
    else:
        crit_p      = 0.0
        h           = np.zeros_like(pvals, dtype=bool)
        adj_ci_cvrg = float("nan")

    return FDRResult(
        h           = h,
        crit_p      = crit_p,
        adj_ci_cvrg = adj_ci_cvrg,
        adj_p       = adj_p,
    )
