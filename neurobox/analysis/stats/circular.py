"""
neurobox.analysis.stats.circular
=================================
Circular statistics for phase-locking and directional preference analysis.

Port of the Stats/ helpers in labbox (Ken Harris / Anton Sirota / Evgeny
Resnik) that build on Fisher (1993) *Statistical Analysis of Circular
Data*:

==========================  =====================================================
labbox                       neurobox
==========================  =====================================================
TF/circmean.m                :func:`circ_mean`, :func:`circ_r`
Stats/RayleighTest.m         :func:`rayleigh_test` (single-vector or per-cluster)
Stats/RayleighTestUnbiased.m :func:`rayleigh_test` with ``unbiased=True``
Stats/PPC.m                  :func:`ppc`
Stats/VonMisesFit.m          :func:`von_mises_fit`
Stats/VonMisesPdf.m          :func:`von_mises_pdf`
Stats/VonMisesRnd.m          :func:`von_mises_rvs`
Stats/BesselRatInv.m         :func:`bessel_ratio_inverse`
==========================  =====================================================

Conventions
-----------
* All angles are in **radians**.
* Phases are interpreted on the circle ``[-π, π)`` for outputs; inputs may
  be on any convention since they are reduced via ``exp(iθ)``.
* The null hypothesis for :func:`rayleigh_test` is uniformity on the
  circle.  The asymptotic series with O(1/n²) correction is used —
  matches Fisher (1993) p. 70 and the labbox implementation exactly.
* :func:`ppc` is the unbiased estimator of squared resultant length from
  Vinck et al. (2010) *NeuroImage* 51:112-122.

Note on per-cluster outputs
---------------------------
The labbox ``RayleighTest`` accepts an optional ``clu`` vector to compute
per-cluster statistics in a single vectorised pass.  This module
preserves that pattern through :func:`rayleigh_test` accepting a
``clusters`` keyword.  The result fields then become arrays of length
``n_clusters`` instead of scalars.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.special import i0, i1


# ─────────────────────────────────────────────────────────────────────────── #
# Result containers                                                            #
# ─────────────────────────────────────────────────────────────────────────── #

@dataclass
class RayleighResult:
    """Output of :func:`rayleigh_test`.

    For the single-vector form (no ``clusters`` argument), every field is
    a Python scalar / numpy scalar.  For the per-cluster form, every
    field is a 1-D array of length ``max_clusters`` (entries with no
    spikes are NaN — matches the labbox convention).

    Attributes
    ----------
    p:
        Rayleigh test p-value(s).  Probability of observing the given
        resultant length under uniformity.
    th0:
        Mean angle in radians, in ``[-π, π)``.
    r:
        Mean resultant length, in ``[0, 1]``.  When ``unbiased=True`` and
        ``n < 1000``, this is computed as ``sqrt(max(PPC, 0))`` instead.
    log_z:
        ``log(n * r²)`` — the Rayleigh statistic in log space.
    kappa:
        Maximum-likelihood estimate of the von Mises concentration.  Uses
        the Fisher (1993) small-sample correction for ``n ≤ 15``.
    n:
        Sample size(s).
    """

    p:      np.ndarray | float
    th0:    np.ndarray | float
    r:      np.ndarray | float
    log_z:  np.ndarray | float
    kappa:  np.ndarray | float
    n:      np.ndarray | int


# ─────────────────────────────────────────────────────────────────────────── #
# Mean direction and resultant length                                          #
# ─────────────────────────────────────────────────────────────────────────── #

def circ_mean(theta: np.ndarray, axis: Optional[int] = None) -> np.ndarray | float:
    """Mean direction of a sample of angles.

    Port of :file:`labbox/TF/circmean.m` (theta output).

    Parameters
    ----------
    theta:
        Angles in radians.
    axis:
        Axis along which to take the mean.  ``None`` (default) flattens
        first.

    Returns
    -------
    mu : float or ndarray
        Mean direction in radians, in ``[-π, π)``.

    Notes
    -----
    The labbox ``circmean`` had a long-standing normalisation bug
    (corrected in 2013) where ``r`` was scaled down by sample size.  Our
    port uses the corrected formula from the start.
    """
    z = np.exp(1j * np.asarray(theta))
    return np.angle(np.mean(z, axis=axis))


def circ_r(theta: np.ndarray, axis: Optional[int] = None) -> np.ndarray | float:
    """Mean resultant length of a sample of angles.

    Port of :file:`labbox/TF/circmean.m` (r output).

    Parameters
    ----------
    theta:
        Angles in radians.
    axis:
        Axis along which to compute.  ``None`` flattens first.

    Returns
    -------
    r : float or ndarray
        Mean resultant length in ``[0, 1]``.  ``r = 1`` indicates perfect
        concentration; ``r = 0`` indicates uniformity.
    """
    z = np.exp(1j * np.asarray(theta))
    return np.abs(np.mean(z, axis=axis))


# ─────────────────────────────────────────────────────────────────────────── #
# Bessel-ratio inverse and Von Mises ML                                       #
# ─────────────────────────────────────────────────────────────────────────── #

def bessel_ratio_inverse(r: np.ndarray) -> np.ndarray:
    """Inverse of the Bessel-function ratio ``I₁(κ) / I₀(κ)``.

    Port of :file:`labbox/Stats/BesselRatInv.m`.

    Implements the three-piece approximation from Fisher (1993) p. 51:

    * ``r ∈ [0, 0.53)``  : ``κ = 2r + r³ + 5r⁵/6``
    * ``r ∈ [0.53, 0.85)``: ``κ = -0.4 + 1.39r + 0.43/(1 - r)``
    * ``r ∈ [0.85, 1.0]`` : ``κ = 1 / (r³ - 4r² + 3r)``

    Parameters
    ----------
    r:
        Mean resultant length(s) in ``[0, 1]``.

    Returns
    -------
    kappa : ndarray
        Concentration estimate, same shape as ``r``.  NaN where ``r`` is
        outside ``[0, 1]``.
    """
    r = np.asarray(r, dtype=np.float64)
    out = np.full_like(r, np.nan)

    g0 = (r >= 0)    & (r < 0.53)
    g1 = (r >= 0.53) & (r < 0.85)
    g2 = (r >= 0.85) & (r <= 1)

    out[g0] = 2 * r[g0] + r[g0] ** 3 + (5 * r[g0] ** 5) / 6
    out[g1] = -0.4 + 1.39 * r[g1] + 0.43 / (1.0 - r[g1])
    # Guard against division by zero at r=1 — replace with a large finite kappa.
    denom = r[g2] ** 3 - 4 * r[g2] ** 2 + 3 * r[g2]
    with np.errstate(divide="ignore", invalid="ignore"):
        out[g2] = np.where(denom != 0, 1.0 / denom, np.inf)
    return out


def _kappa_small_sample_correction(kml: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Apply the labbox/Fisher small-n correction to a κ MLE.

    Vectorised port of the inner block of :file:`Stats/VonMisesFit.m` /
    :file:`Stats/RayleighTest.m`:

    * ``n ≤ 15`` and ``kml < 2``: ``k = max(kml − 2/(n·kml), 0)``
    * ``n ≤ 15`` and ``kml ≥ 2``: ``k = (n−1)³ kml / (n³ + n)``
    * ``n  > 15``                  : no correction.
    """
    kml = np.asarray(kml, dtype=np.float64)
    n   = np.asarray(n,   dtype=np.float64)
    out = kml.copy()

    small_low  = (n <= 15) & (kml <  2)
    small_high = (n <= 15) & (kml >= 2)

    with np.errstate(divide="ignore", invalid="ignore"):
        out[small_low]  = np.maximum(kml[small_low] - 2.0 / (n[small_low] * kml[small_low]), 0.0)
        out[small_high] = ((n[small_high] - 1) ** 3 * kml[small_high]
                           / (n[small_high] ** 3 + n[small_high]))
    return out


def von_mises_fit(theta: np.ndarray) -> tuple[float, float]:
    """Maximum-likelihood von Mises fit to a sample of angles.

    Port of :file:`labbox/Stats/VonMisesFit.m`.

    Parameters
    ----------
    theta:
        Angles in radians.  1-D.

    Returns
    -------
    mu : float
        Mean direction in radians, in ``[-π, π)``.
    kappa : float
        ML estimate of the concentration parameter, with the Fisher
        small-sample correction applied for ``n ≤ 15``.

    Examples
    --------
    >>> rng = np.random.default_rng(0)
    >>> theta = rng.vonmises(mu=0.5, kappa=4.0, size=500)
    >>> mu, kappa = von_mises_fit(theta)
    """
    theta = np.asarray(theta, dtype=np.float64).ravel()
    n = theta.size
    if n == 0:
        return float("nan"), float("nan")
    mu = circ_mean(theta)
    r  = circ_r(theta)
    kml = bessel_ratio_inverse(np.atleast_1d(r))[0]
    kappa = _kappa_small_sample_correction(np.atleast_1d(kml), np.atleast_1d(n))[0]
    return float(mu), float(kappa)


def von_mises_pdf(
    theta: np.ndarray,
    mu: float,
    kappa: float,
) -> np.ndarray:
    """Probability density of the von Mises distribution.

    Port of :file:`labbox/Stats/VonMisesPdf.m`.

    .. math::

       p(\\theta \\mid \\mu, \\kappa) = \\frac{e^{\\kappa \\cos(\\theta - \\mu)}}{2 \\pi I_0(\\kappa)}

    Parameters
    ----------
    theta:
        Angle(s) in radians.
    mu:
        Mean direction.
    kappa:
        Concentration parameter (``≥ 0``).

    Returns
    -------
    p : ndarray
        Density evaluated at each ``theta``.
    """
    theta = np.asarray(theta, dtype=np.float64)
    return np.exp(kappa * np.cos(theta - mu)) / (2.0 * np.pi * i0(kappa))


def von_mises_rvs(
    mu: float,
    kappa: float,
    size: int = 1,
    random_state: Optional[int | np.random.Generator] = None,
) -> np.ndarray:
    """Sample from a von Mises distribution.

    Port of :file:`labbox/Stats/VonMisesRnd.m`.

    Parameters
    ----------
    mu:
        Mean direction in radians.
    kappa:
        Concentration parameter (``≥ 0``).
    size:
        Number of samples to draw.
    random_state:
        Seed or :class:`numpy.random.Generator` instance.

    Returns
    -------
    theta : ndarray
        Random angles in radians, in ``[-π, π)``.

    Notes
    -----
    Defers to :meth:`numpy.random.Generator.vonmises`, which uses the
    Best-Fisher (1979) algorithm — the same approach as the labbox
    function.
    """
    rng = (random_state if isinstance(random_state, np.random.Generator)
           else np.random.default_rng(random_state))
    return rng.vonmises(mu=mu, kappa=kappa, size=size)


# ─────────────────────────────────────────────────────────────────────────── #
# PPC (Vinck et al. 2010)                                                      #
# ─────────────────────────────────────────────────────────────────────────── #

def ppc(theta: np.ndarray) -> float:
    """Pairwise Phase Consistency (Vinck et al., 2010).

    Port of :file:`labbox/Stats/PPC.m`.

    Unbiased estimator of the squared resultant length:

    .. math::

       \\mathrm{PPC} = \\frac{2}{n(n-1)} \\sum_{i < j} \\cos(\\theta_i - \\theta_j)

    PPC is unbiased for small samples — unlike the resultant length ``r``
    which has positive bias of order ``1/n``.  Vinck et al. (2010,
    *NeuroImage* 51:112-122) recommend it for spike-LFP phase locking
    when spike counts are low.

    Parameters
    ----------
    theta:
        Angles in radians (1-D).

    Returns
    -------
    ppc : float
        Estimator value.  May be **negative** for very anti-clustered
        samples; the caller should clip to 0 if treating it as r².

    Notes
    -----
    Uses the closed-form ``∑∑ cos(θᵢ - θⱼ) = (∑ cos θᵢ)² + (∑ sin θᵢ)² - n``,
    which is O(n) in time vs the labbox's O(n²) double-loop.  For
    n=10⁴ this is ~10⁵× faster.

    Specifically:
        ``2 ∑_{i<j} cos(θᵢ - θⱼ) = ((∑ cos θᵢ)² + (∑ sin θᵢ)²) - n``

    so

        ``PPC = (S² - n) / (n (n - 1))``  where  ``S² = (∑ cos θᵢ)² + (∑ sin θᵢ)²``.
    """
    theta = np.asarray(theta, dtype=np.float64).ravel()
    n = theta.size
    if n < 2:
        return float("nan")
    sum_cos = np.sum(np.cos(theta))
    sum_sin = np.sum(np.sin(theta))
    s_squared = sum_cos ** 2 + sum_sin ** 2
    return float((s_squared - n) / (n * (n - 1)))


# ─────────────────────────────────────────────────────────────────────────── #
# Rayleigh test                                                                #
# ─────────────────────────────────────────────────────────────────────────── #

def _rayleigh_p_from_z(z: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Asymptotic Rayleigh p-value with O(1/n²) correction.

    Wilkie (1983) / Fisher (1993) p. 70 series — same as labbox::

        p = exp(-z) * (
                1
                + (2z - z²) / (4n)
                - (24z - 132z² + 76z³ - 9z⁴) / (288 n²)
            )
    """
    z = np.asarray(z, dtype=np.float64)
    n = np.asarray(n, dtype=np.float64)
    correction = (
        1.0
        + (2 * z - z ** 2) / (4 * n)
        - (24 * z - 132 * z ** 2 + 76 * z ** 3 - 9 * z ** 4) / (288 * n ** 2)
    )
    return np.exp(-z) * correction


def rayleigh_test(
    theta: np.ndarray,
    clusters: Optional[np.ndarray] = None,
    max_clusters: Optional[int] = None,
    unbiased: bool = False,
    unbiased_n_threshold: int = 1000,
) -> RayleighResult:
    """Rayleigh test for departure from uniformity on the circle.

    Port of :file:`labbox/Stats/RayleighTest.m` (with ``unbiased=False``)
    and :file:`labbox/Stats/RayleighTestUnbiased.m` (with
    ``unbiased=True``).

    Tests the null hypothesis that ``theta`` is uniformly distributed on
    the circle, using the asymptotic Rayleigh statistic with the
    second-order correction from Fisher (1993) p. 70.

    Parameters
    ----------
    theta:
        Angles in radians.  1-D array.
    clusters:
        Optional integer cluster labels (one per element of ``theta``)
        for per-cluster statistics — equivalent to looping the test over
        each cluster, but vectorised.  Cluster labels must be ``≥ 1``.
        Entries with ``clusters[i] ≤ 0`` are dropped.
    max_clusters:
        Output array length when ``clusters`` is given.  Defaults to
        ``max(clusters)``.  Useful when comparing across recordings to
        get a stable per-unit indexing — clusters with no firing in this
        period get NaN entries (matching the labbox convention).
    unbiased:
        If True, use ``r = sqrt(max(PPC, 0))`` instead of the resultant
        length when ``n < unbiased_n_threshold``.  This is the
        Vinck-corrected variant from
        :file:`labbox/Stats/RayleighTestUnbiased.m`.
    unbiased_n_threshold:
        Sample-size cutoff below which the unbiased estimator kicks in
        (default 1000, matching the labbox).

    Returns
    -------
    :class:`RayleighResult`

    Examples
    --------
    Single-vector usage::

        from neurobox.analysis.stats.circular import rayleigh_test
        result = rayleigh_test(spike_phases)
        if result.p < 0.05:
            print(f"phase-locked at θ₀ = {result.th0:.3f} rad, r = {result.r:.3f}")

    Per-cluster usage (equivalent to looping over units)::

        # spike_phases and unit_ids both length N_spikes
        result = rayleigh_test(spike_phases, clusters=unit_ids, max_clusters=200)
        # result.p has shape (200,); units with no spikes are NaN.
    """
    theta = np.asarray(theta, dtype=np.float64).ravel()

    # ── Per-cluster path ────────────────────────────────────────────────── #
    if clusters is not None:
        clusters = np.asarray(clusters).ravel()
        if clusters.size != theta.size:
            raise ValueError(
                f"theta and clusters must have same length: "
                f"{theta.size} vs {clusters.size}"
            )
        # Drop non-positive labels (matches labbox: clu>0)
        valid = clusters > 0
        theta    = theta[valid]
        clusters = clusters[valid].astype(np.int64)

        if max_clusters is None:
            max_clusters = int(clusters.max()) if clusters.size > 0 else 0

        # Sums over each cluster of exp(iθ)
        z = np.exp(1j * theta)
        m_real = np.bincount(clusters, weights=z.real, minlength=max_clusters + 1)[1:max_clusters + 1]
        m_imag = np.bincount(clusters, weights=z.imag, minlength=max_clusters + 1)[1:max_clusters + 1]
        n_per  = np.bincount(clusters, minlength=max_clusters + 1)[1:max_clusters + 1].astype(np.int64)

        present = n_per > 0
        m_complex = (m_real + 1j * m_imag)
        # Avoid division warning for empty clusters
        with np.errstate(invalid="ignore"):
            m_mean = np.where(present, m_complex / np.where(n_per > 0, n_per, 1), 0)
        th0 = np.angle(m_mean)
        r   = np.abs(m_mean)

        if unbiased:
            small = present & (n_per < unbiased_n_threshold)
            if np.any(small):
                # PPC per cluster, restricted to small ones
                ppc_vals = np.full(max_clusters, np.nan)
                for k in np.flatnonzero(small):
                    sel = clusters == (k + 1)
                    if np.sum(sel) >= 2:
                        ppc_vals[k] = ppc(theta[sel])
                # Replace r where unbiased is requested and PPC is finite
                r[small] = np.sqrt(np.maximum(ppc_vals[small], 0.0))

        # Statistic and p-value
        z_stat = n_per * r ** 2
        p     = np.full(max_clusters, np.nan)
        log_z = np.full(max_clusters, np.nan)
        kappa = np.full(max_clusters, np.nan)
        with np.errstate(divide="ignore"):
            log_z[present] = np.log(z_stat[present])
        p[present] = _rayleigh_p_from_z(z_stat[present], n_per[present])

        kml = np.full(max_clusters, np.nan)
        kml[present] = bessel_ratio_inverse(r[present])
        kappa[present] = _kappa_small_sample_correction(kml[present], n_per[present])

        # Empty-cluster cells → NaN for th0 / r as well (mathematically
        # they are 0, but for pipeline use NaN is more informative).
        th0[~present] = np.nan
        r[~present]   = np.nan

        return RayleighResult(p=p, th0=th0, r=r, log_z=log_z, kappa=kappa, n=n_per)

    # ── Single-vector path ──────────────────────────────────────────────── #
    n = theta.size
    if n == 0:
        return RayleighResult(p=float("nan"), th0=float("nan"), r=float("nan"),
                              log_z=float("nan"), kappa=float("nan"), n=0)

    z = np.exp(1j * theta)
    m = np.mean(z)
    th0 = float(np.angle(m))
    r_default = float(np.abs(m))

    if unbiased and n < unbiased_n_threshold:
        r = float(np.sqrt(max(ppc(theta), 0.0)))
    else:
        r = r_default

    z_stat = n * r ** 2
    log_z = float(np.log(z_stat)) if z_stat > 0 else float("-inf")
    p = float(_rayleigh_p_from_z(np.atleast_1d(z_stat), np.atleast_1d(n))[0])

    # Concentration κ
    _, kappa = von_mises_fit(theta)

    return RayleighResult(p=p, th0=th0, r=r, log_z=log_z, kappa=kappa, n=n)
