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


# ─────────────────────────────────────────────────────────────────────────── #
# CircStat2012a ports — Berens's MATLAB Circular Statistics Toolbox          #
#                                                                             #
# These are the essential descriptive-statistics functions used throughout    #
# the MTA pipeline and beyond.  They mirror the Berens API (matlab            #
# Circular Statistics Toolbox version 2012a) so that MTA call sites port      #
# cleanly.  Hypothesis-testing functions (Rayleigh, V-test, Watson-Williams)  #
# are NOT included here — those that the lab uses are already covered by      #
# :func:`rayleigh_test` and :func:`ppc` above; the rest can be added on       #
# demand.                                                                     #
# ─────────────────────────────────────────────────────────────────────────── #


def circ_dist(alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Pairwise circular difference ``alpha - beta`` wrapped to ``(-π, π]``.

    Port of :file:`labbox/Stats/CircStat2012a/circ_dist.m` (Berens, P., 2009).

    Parameters
    ----------
    alpha, beta:
        Angles in radians.  Either same shape, or *beta* may be a single
        angle broadcast against *alpha*.

    Returns
    -------
    np.ndarray
        Element-wise signed angular distance.
    """
    alpha = np.asarray(alpha, dtype=np.float64)
    beta  = np.asarray(beta,  dtype=np.float64)
    return np.angle(np.exp(1j * alpha) / np.exp(1j * beta))


def circ_dist2(alpha: np.ndarray, beta: np.ndarray | None = None) -> np.ndarray:
    """All-pairs circular difference ``alpha_i - beta_j``.

    Port of :file:`labbox/Stats/CircStat2012a/circ_dist2.m`.

    Parameters
    ----------
    alpha:
        ``(n,)`` angles in radians.
    beta:
        ``(m,)`` angles in radians.  ``None`` → use *alpha*.

    Returns
    -------
    np.ndarray, shape ``(n, m)``
        ``out[i, j] = circ_dist(alpha[i], beta[j])``.
    """
    a = np.asarray(alpha, dtype=np.float64).ravel()
    b = a if beta is None else np.asarray(beta, dtype=np.float64).ravel()
    return np.angle(np.exp(1j * a)[:, None] / np.exp(1j * b)[None, :])


def circ_var(theta: np.ndarray, axis: Optional[int] = None) -> np.ndarray | float:
    """Circular variance ``1 − r``.

    Port of :file:`labbox/Stats/CircStat2012a/circ_var.m` (the unweighted,
    unbinned form — pass weighted data through :func:`circ_r` directly if
    you need the binned-correction version).

    Parameters
    ----------
    theta:
        Angles in radians.
    axis:
        Axis along which to compute.  ``None`` flattens first.

    Returns
    -------
    float or ndarray
        ``S = 1 − r`` in ``[0, 1]``.  ``S = 0`` ⇒ perfect concentration;
        ``S = 1`` ⇒ uniform.
    """
    return 1.0 - circ_r(theta, axis=axis)


def circ_std(theta: np.ndarray, axis: Optional[int] = None,
             ) -> tuple[np.ndarray | float, np.ndarray | float]:
    """Circular standard deviation.

    Port of :file:`labbox/Stats/CircStat2012a/circ_std.m`.

    Parameters
    ----------
    theta:
        Angles in radians.
    axis:
        Axis along which to compute.  ``None`` flattens first.

    Returns
    -------
    s : float or ndarray
        Angular deviation ``√(2(1 − r))`` (Zar 26.20).
    s0 : float or ndarray
        Circular standard deviation ``√(−2 log r)`` (Zar 26.21).
    """
    r = circ_r(theta, axis=axis)
    s  = np.sqrt(2.0 * (1.0 - r))
    with np.errstate(divide="ignore", invalid="ignore"):
        s0 = np.sqrt(-2.0 * np.log(r))
    return s, s0


def circ_median(theta: np.ndarray, axis: int = 0) -> np.ndarray:
    """Circular median direction.

    Port of :file:`labbox/Stats/CircStat2012a/circ_median.m`.

    For each column (along *axis*), finds the direction *m* that
    minimises ``|N_+(m) − N_-(m)|`` where ``N_±(m)`` are the counts of
    angles falling on each side of the diameter through *m*.  Slow for
    large samples (O(n²) memory via the all-pairs distance matrix).

    Parameters
    ----------
    theta:
        Angles in radians.  Must be 1-D or 2-D.
    axis:
        Reduction axis.  Default 0.

    Returns
    -------
    np.ndarray
        Median angle(s) in ``[0, 2π)``.
    """
    theta = np.asarray(theta, dtype=np.float64)
    if theta.ndim == 1:
        return _circ_median_1d(theta)
    if theta.ndim != 2:
        raise ValueError(
            f"circ_median supports 1-D or 2-D input; got {theta.ndim}-D"
        )
    if axis == 0:
        return np.array([_circ_median_1d(theta[:, j])
                         for j in range(theta.shape[1])])
    if axis == 1:
        return np.array([_circ_median_1d(theta[i, :])
                         for i in range(theta.shape[0])])
    raise ValueError("circ_median axis must be 0 or 1")


def _circ_median_1d(beta: np.ndarray) -> float:
    beta = np.mod(beta, 2 * np.pi)
    n = beta.size
    if n == 0:
        return float("nan")
    dd = circ_dist2(beta, beta)            # (n, n)
    m1 = np.sum(dd >= 0, axis=0)
    m2 = np.sum(dd <= 0, axis=0)
    dm = np.abs(m1 - m2)
    if n % 2 == 1:
        m   = dm.min()
        idx = [int(np.argmin(dm))]
    else:
        m   = dm.min()
        idx = list(np.where(dm == m)[0][:2])

    md = circ_mean(beta[idx])
    cmean = circ_mean(beta)
    if abs(circ_dist(cmean, md)) > abs(circ_dist(cmean, md + np.pi)):
        md = float(np.mod(md + np.pi, 2 * np.pi))
    return float(md)


def circ_kappa(theta: np.ndarray) -> float:
    """Approximate ML estimate of the von Mises concentration κ.

    Port of :file:`labbox/Stats/CircStat2012a/circ_kappa.m`.

    Uses Fisher's three-piece approximation in *r* (mean resultant
    length) plus the small-sample correction for ``n < 15``.

    Parameters
    ----------
    theta:
        Either a sample of angles in radians or a precomputed mean
        resultant length *r* (the MATLAB convention: when length 1, the
        input is treated as *r*).

    Returns
    -------
    kappa : float
    """
    theta = np.asarray(theta, dtype=np.float64).ravel()
    n = theta.size
    if n > 1:
        R = float(circ_r(theta))
    else:
        R = float(theta[0])

    if R < 0.53:
        kappa = 2.0 * R + R**3 + 5.0 * R**5 / 6.0
    elif R < 0.85:
        kappa = -0.4 + 1.39 * R + 0.43 / (1.0 - R)
    else:
        kappa = 1.0 / (R**3 - 4.0 * R**2 + 3.0 * R)

    if 1 < n < 15:
        if kappa < 2:
            kappa = max(kappa - 2.0 / (n * kappa), 0.0)
        else:
            kappa = (n - 1)**3 * kappa / (n**3 + n)
    return float(kappa)


def circ_moment(
    theta:  np.ndarray,
    p:      int = 1,
    centered: bool = False,
    axis:   Optional[int] = None,
) -> tuple[np.ndarray | complex,
           np.ndarray | float,
           np.ndarray | float]:
    """Complex p-th circular moment.

    Port of :file:`labbox/Stats/CircStat2012a/circ_moment.m` (unweighted form).

    Parameters
    ----------
    theta:
        Angles in radians.
    p:
        Moment order.  Default 1.
    centered:
        If True, compute the *central* moment (about the circular
        mean).  Default False.
    axis:
        Reduction axis.  ``None`` flattens first.

    Returns
    -------
    mp : complex or ndarray
        Complex p-th moment.
    rho_p : float or ndarray
        Magnitude.
    mu_p : float or ndarray
        Angle.
    """
    theta = np.asarray(theta, dtype=np.float64)
    if centered:
        mean_dir = circ_mean(theta, axis=axis)
        theta = circ_dist(theta, mean_dir if axis is None
                                  else np.expand_dims(mean_dir, axis=axis))
    n = theta.shape[axis] if axis is not None else theta.size
    cbar = np.sum(np.cos(p * theta), axis=axis) / n
    sbar = np.sum(np.sin(p * theta), axis=axis) / n
    mp = cbar + 1j * sbar
    return mp, np.abs(mp), np.angle(mp)


def circ_skewness(
    theta: np.ndarray,
    axis:  Optional[int] = None,
) -> tuple[np.ndarray | float, np.ndarray | float]:
    """Pewsey + Fisher angular skewness.

    Port of :file:`labbox/Stats/CircStat2012a/circ_skewness.m`.

    Parameters
    ----------
    theta:
        Angles in radians.
    axis:
        Reduction axis.  ``None`` flattens first.

    Returns
    -------
    b : float or ndarray
        Skewness (Pewsey 2004): ``mean(sin(2(θ − θ̄)))``.
    b0 : float or ndarray
        Alternative skewness (Fisher 1995 §2.3.5):
        ``ρ₂ sin(μ₂ − 2θ̄) / (1 − r)^{3/2}``.
    """
    R     = circ_r(theta, axis=axis)
    theta_bar = circ_mean(theta, axis=axis)
    _, rho2, mu2 = circ_moment(theta, p=2, centered=False, axis=axis)
    theta_arr = np.asarray(theta, dtype=np.float64)
    if axis is not None:
        theta_bar_b = np.expand_dims(theta_bar, axis=axis)
    else:
        theta_bar_b = theta_bar
    n = theta_arr.shape[axis] if axis is not None else theta_arr.size
    b  = np.sum(np.sin(2.0 * circ_dist(theta_arr, theta_bar_b)),
                axis=axis) / n
    b0 = rho2 * np.sin(circ_dist(mu2, 2.0 * theta_bar)) / (1.0 - R) ** 1.5
    return b, b0


def circ_kurtosis(
    theta: np.ndarray,
    axis:  Optional[int] = None,
) -> tuple[np.ndarray | float, np.ndarray | float]:
    """Pewsey + Fisher angular kurtosis.

    Port of :file:`labbox/Stats/CircStat2012a/circ_kurtosis.m`.

    Returns
    -------
    k : float or ndarray
        Kurtosis (Pewsey 2004): ``mean(cos(2(θ − θ̄)))``.
    k0 : float or ndarray
        Alternative kurtosis (Fisher 1995 §2.3.5):
        ``(ρ₂ cos(μ₂ − 2θ̄) − r⁴) / (1 − r)²``.
    """
    R     = circ_r(theta, axis=axis)
    theta_bar = circ_mean(theta, axis=axis)
    _, rho2, mu2 = circ_moment(theta, p=2, centered=False, axis=axis)
    theta_arr = np.asarray(theta, dtype=np.float64)
    if axis is not None:
        theta_bar_b = np.expand_dims(theta_bar, axis=axis)
    else:
        theta_bar_b = theta_bar
    n = theta_arr.shape[axis] if axis is not None else theta_arr.size
    k  = np.sum(np.cos(2.0 * circ_dist(theta_arr, theta_bar_b)),
                axis=axis) / n
    k0 = (rho2 * np.cos(circ_dist(mu2, 2.0 * theta_bar)) - R**4) \
         / (1.0 - R) ** 2
    return k, k0


def circ_axial(alpha: np.ndarray, p: int = 1) -> np.ndarray:
    """Transform p-axial data to a common ``[0, 2π)`` scale.

    Port of :file:`labbox/Stats/CircStat2012a/circ_axial.m`.

    Multiplies *alpha* by *p* and reduces modulo ``2π``.  Useful for
    converting axial data (e.g. orientation angles in ``[0, π)``,
    where 0 and π represent the same axis) onto the full circle so
    that linear circular statistics apply.

    Parameters
    ----------
    alpha:
        Angles in radians.
    p:
        Number of modes (axes).  ``p = 2`` for axial data.  Default 1.

    Returns
    -------
    np.ndarray
        Transformed angles in ``[0, 2π)``.
    """
    return np.mod(np.asarray(alpha, dtype=np.float64) * p, 2.0 * np.pi)


def circ_ang2rad(alpha_deg: np.ndarray) -> np.ndarray:
    """Convert degrees → radians.

    Port of :file:`labbox/Stats/CircStat2012a/circ_ang2rad.m`.  Trivial
    wrapper over :func:`numpy.deg2rad` provided for API parity.
    """
    return np.deg2rad(np.asarray(alpha_deg, dtype=np.float64))


def circ_rad2ang(alpha_rad: np.ndarray) -> np.ndarray:
    """Convert radians → degrees.

    Port of :file:`labbox/Stats/CircStat2012a/circ_rad2ang.m`.  Trivial
    wrapper over :func:`numpy.rad2deg` provided for API parity.
    """
    return np.rad2deg(np.asarray(alpha_rad, dtype=np.float64))
