"""
neurobox.analysis.decoding.bayesian
====================================
Bayesian population decoding from boxcar-smoothed unit firing rates.

Port of :file:`MTA/analysis/decode_ufr_boxcar.m` (Anton Sirota et al.).

What the decoder does
---------------------
For each time window, given:
  * a per-unit firing-rate vector ``cufr`` of shape ``(n_units,)``
  * a per-unit place-field ``ratemap`` of shape ``(n_bins, n_units)``

Compute the unnormalised posterior on spatial bins under a
Poisson firing model with a flat prior on position::

    log P(x | spikes) ∝ Σᵤ [ cufr[u] · log r_u(x)  -  τ · r_u(x) ]
                      = log_ratemap @ cufr  +  prior_per_bin

where ``τ`` is the boxcar window length and ``prior_per_bin =
-Σᵤ r_u(x) · τ`` is the negative expected total-spike count at
each bin.

From the normalised posterior ``E[bin]`` we extract five
position estimates per time step:

  * ``max``  — bin of the posterior peak
  * ``com``  — posterior-weighted centre of mass
  * ``sax``  — *smoothed* posterior weighted: multiply E by a
    Gaussian centred on ``max``, renormalise, then COM
  * ``lom``  — log-posterior weighted COM (uses ``log10(E) + 8``,
    clipped at zero)
  * ``lax``  — smoothed log-posterior weighted COM

These five measures give different trade-offs between localisation
sharpness (``max`` is sharpest) and tail sensitivity (``lom`` /
``lax`` weight low-probability bins more heavily).

Performance
-----------
The MATLAB original loops over time bins.  This Python port also
loops, but uses numpy-vectorised inner expressions; on a 250 Hz
session of 1 hour with 50 units and 25-bin maps, expect ~10 s on a
modern laptop.  For substantially longer sessions or denser maps,
the bottleneck is the Gaussian-weight construction at the peak.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class DecodingResult:
    """Output of :func:`decode_ufr_boxcar`.

    All ``(N, ...)`` arrays are aligned by row; ``ind`` gives the
    index in the input ``ufr`` time axis at which each row was
    decoded.

    Attributes
    ----------
    ind:
        ``(N,)`` time-bin indices (in samples) into the input UFR
        array.  Not uniform — only time-bins with at least one spike
        are kept.
    max:
        ``(N, n_dims)`` position at the posterior peak.
    com:
        ``(N, n_dims)`` posterior-weighted centre of mass.
    sax:
        ``(N, n_dims)`` Gaussian-smoothed posterior-weighted COM.
    lom:
        ``(N, n_dims)`` log-posterior weighted COM.
    lax:
        ``(N, n_dims)`` Gaussian-smoothed log-posterior weighted COM.
    post:
        ``(N,)`` normalised posterior value at the peak bin.
    ucnt:
        ``(N,)`` number of units active in each window.
    uinc:
        ``(N, n_units)`` boolean — which units contributed to each
        decoded sample.
    smoothing_weights:
        Diagonal of the Gaussian smoothing matrix used for sax/lax.
    window:
        Total decoding window in seconds (= 2 × half-spike-window).
    samplerate:
        Samplerate of the input UFR.
    """

    ind:               np.ndarray
    max:               np.ndarray
    com:               np.ndarray
    sax:               np.ndarray
    lom:               np.ndarray
    lax:               np.ndarray
    post:              np.ndarray
    ucnt:              np.ndarray
    uinc:              np.ndarray
    smoothing_weights: np.ndarray
    window:            float
    samplerate:        float

    @property
    def n(self) -> int:
        return int(self.ind.shape[0])


# ─────────────────────────────────────────────────────────────────────────── #
# Posterior-stat helper                                                        #
# ─────────────────────────────────────────────────────────────────────────── #

def _gaussian_smooth_weights(
    bin_coords:        np.ndarray,
    peak_coords:       np.ndarray,
    inv_smoothing:     np.ndarray,
) -> np.ndarray:
    """Per-bin Gaussian weights centred on ``peak_coords``.

    ``weight[b] = exp(-Δᵀ Σ⁻¹ Δ)`` with ``Δ = bin_coords[b] - peak_coords``.

    Parameters
    ----------
    bin_coords:
        ``(n_bins, n_dims)`` bin centre positions.
    peak_coords:
        ``(n_dims,)`` location of the peak.
    inv_smoothing:
        ``(n_dims, n_dims)`` inverse covariance.
    """
    delta = bin_coords - peak_coords      # (n_bins, n_dims)
    quad  = np.einsum("bi,ij,bj->b", delta, inv_smoothing, delta)
    return np.exp(-quad)


def _decode_one_window(
    cufr:           np.ndarray,
    log_ratemap:    np.ndarray,
    prior_per_bin:  np.ndarray,
    bin_coords:     np.ndarray,
    inv_smoothing:  np.ndarray,
    n_dims:         int,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray]:
    """One time-bin decode → (post, max, com, sax, lom, lax).

    Returns
    -------
    post:    posterior value at peak (float)
    max_pos: (n_dims,) coordinates of peak
    com:     (n_dims,) posterior-weighted COM
    sax:     (n_dims,) smoothed-posterior weighted COM
    lom:     (n_dims,) log-posterior weighted COM
    lax:     (n_dims,) smoothed-log-posterior weighted COM
    """
    # Posterior (unnormalised then normalised) — line 142-143 in MATLAB
    log_post = prior_per_bin + log_ratemap @ (cufr + np.finfo(float).eps)
    # Subtract max for numerical stability before exp
    log_post -= log_post.max()
    E = np.exp(log_post)
    E /= np.nansum(E)                                      # (n_bins,)

    # Peak — line 146
    tbin = int(np.argmax(E))
    post_peak = float(E[tbin])
    max_pos = bin_coords[tbin].copy()

    # COM — line 156
    com = (bin_coords * E[:, None]).sum(axis=0)            # (n_dims,)

    # SAX — Gaussian-smoothed COM — lines 150-155
    g = _gaussian_smooth_weights(bin_coords, max_pos, inv_smoothing)
    g_norm = np.nansum(g)
    if g_norm > 0:
        g = g / g_norm
    w = g * E
    w_norm = np.nansum(w)
    if w_norm > 0:
        w = w / w_norm
    sax = (bin_coords * w[:, None]).sum(axis=0)

    # LOM — log-posterior COM — lines 158-162
    log_E = np.log10(E + np.finfo(float).tiny) + 8.0
    log_E[log_E <= 0.0] = np.finfo(float).eps
    log_E_norm = np.nansum(log_E)
    if log_E_norm > 0:
        log_E = log_E / log_E_norm
    lom = (bin_coords * log_E[:, None]).sum(axis=0)

    # LAX — smoothed-log COM — lines 164-169
    g2 = _gaussian_smooth_weights(bin_coords, max_pos, inv_smoothing)
    g2_norm = np.nansum(g2)
    if g2_norm > 0:
        g2 = g2 / g2_norm
    w2 = g2 * log_E
    w2_norm = np.nansum(w2)
    if w2_norm > 0:
        w2 = w2 / w2_norm
    lax = (bin_coords * w2[:, None]).sum(axis=0)

    return post_peak, max_pos, com, sax, lom, lax


# ─────────────────────────────────────────────────────────────────────────── #
# Public API                                                                   #
# ─────────────────────────────────────────────────────────────────────────── #

def decode_ufr_boxcar(
    ufr:                 np.ndarray,
    ratemap:             np.ndarray,
    bin_coords:          np.ndarray,
    *,
    half_spike_window_s: float       = 0.022,
    spike_window_s:      float       = 0.008,
    smoothing_weights:   "np.ndarray | tuple[float, ...]" = (800.0**2, 800.0**2),
    samplerate:          float       = 250.0,
    min_active_units:    float       = 0.5,
    edge_pad:            int         = 10,
) -> DecodingResult:
    """Bayesian position decoding from boxcar-smoothed firing rates.

    Port of :file:`MTA/analysis/decode_ufr_boxcar.m`.

    Parameters
    ----------
    ufr:
        ``(T, n_units)`` boxcar-smoothed firing rates in spikes/s.
        Typically computed via :meth:`NBDufr.compute(mode='boxcar')`.
    ratemap:
        ``(n_bins, n_units)`` per-unit rate maps **flattened over
        spatial dimensions and masked** to the bins of interest.
        Use :func:`prepare_ratemap` to build this from a
        :class:`PlaceFieldResult` and a tensor mask.
    bin_coords:
        ``(n_bins, n_dims)`` spatial coordinates of each surviving
        bin in the same units used downstream (typically mm).
        Use :func:`prepare_bin_coords`.
    half_spike_window_s:
        Half the spike-counting window in seconds.  Window is
        ``[t - half, t + half]``, total ``2 * half`` seconds long.
        Default 0.022 (matches MATLAB).
    spike_window_s:
        UFR-construction window in seconds — used to scale the
        prior term.  Default 0.008 (matches MATLAB ``ufrWindow``).
    smoothing_weights:
        Per-spatial-dim Gaussian σ² (mm²).  Default
        ``(800², 800²)``.  Pass an ``(n_dims, n_dims)`` matrix to
        get a non-diagonal covariance.
    samplerate:
        UFR samplerate in Hz.  Default 250 Hz.
    min_active_units:
        Minimum sum of UFR within the window for a time step to be
        decoded (matches MATLAB's ``cufr > 0.5`` check on the sum).
    edge_pad:
        Number of additional padding samples to skip at the end of
        the trace (matches MATLAB's ``-10`` in the loop bound).

    Returns
    -------
    DecodingResult
    """
    ufr = np.asarray(ufr, dtype=np.float64)
    ratemap = np.asarray(ratemap, dtype=np.float64)
    bin_coords = np.asarray(bin_coords, dtype=np.float64)

    if ufr.ndim != 2:
        raise ValueError(f"ufr must be (T, n_units); got {ufr.shape}")
    if ratemap.ndim != 2:
        raise ValueError(f"ratemap must be (n_bins, n_units); got {ratemap.shape}")
    if bin_coords.ndim != 2:
        raise ValueError(
            f"bin_coords must be (n_bins, n_dims); got {bin_coords.shape}"
        )
    if ratemap.shape[1] != ufr.shape[1]:
        raise ValueError(
            f"unit count mismatch: ratemap has {ratemap.shape[1]} units, "
            f"ufr has {ufr.shape[1]}"
        )
    if ratemap.shape[0] != bin_coords.shape[0]:
        raise ValueError(
            f"bin count mismatch: ratemap has {ratemap.shape[0]} bins, "
            f"bin_coords has {bin_coords.shape[0]}"
        )

    n_dims = bin_coords.shape[1]

    # Build smoothing covariance and its inverse
    sw = np.asarray(smoothing_weights, dtype=np.float64)
    if sw.ndim == 1:
        smoothing_matrix = np.diag(sw)
    else:
        smoothing_matrix = sw
    if smoothing_matrix.shape != (n_dims, n_dims):
        raise ValueError(
            f"smoothing_weights must be ({n_dims},) or ({n_dims},{n_dims}); "
            f"got shape {smoothing_matrix.shape}"
        )
    inv_smoothing = np.linalg.inv(smoothing_matrix)

    # ── Posterior prep — lines 96-99 of MATLAB ──────────────────────── #
    rmap = ratemap.copy()
    rmap[~np.isfinite(rmap)] = 0.0       # nan → 0
    rmap = rmap + 1e-3                    # avoid log(0)
    log_rmap = np.log(rmap)
    prior_per_bin = -rmap.sum(axis=1) * spike_window_s     # (n_bins,)

    # ── Window setup ────────────────────────────────────────────────── #
    half_n = int(round(half_spike_window_s * samplerate))
    if half_n < 1:
        raise ValueError(
            f"half_spike_window_s={half_spike_window_s} too small for "
            f"samplerate={samplerate}"
        )

    T, n_units = ufr.shape

    # Pre-allocate (rough upper bound: every time bin has activity)
    out_ind  = []
    out_post = []
    out_max  = []
    out_com  = []
    out_sax  = []
    out_lom  = []
    out_lax  = []
    out_ucnt = []
    out_uinc = []

    # MATLAB main loop — lines 124-175
    for t in range(half_n, T - half_n - edge_pad):
        # Sum spike rates within ±half_n
        cufr = ufr[t - half_n : t + half_n + 1, :].sum(axis=0)   # (n_units,)
        active = cufr > 0
        n_active = int(active.sum())
        if n_active == 0:
            continue
        # MATLAB's cufr-sum threshold isn't applied per-unit; it's the
        # sum over units that needs to exceed min_active_units, equivalent to
        # ``any(cufr > 0.5)``.
        if cufr.sum() <= min_active_units:
            continue

        post, mx, com, sax, lom, lax = _decode_one_window(
            cufr, log_rmap, prior_per_bin,
            bin_coords, inv_smoothing, n_dims,
        )

        out_ind.append(t)
        out_post.append(post)
        out_max.append(mx)
        out_com.append(com)
        out_sax.append(sax)
        out_lom.append(lom)
        out_lax.append(lax)
        out_ucnt.append(n_active)
        out_uinc.append(active.copy())

    return DecodingResult(
        ind  = np.asarray(out_ind, dtype=np.int64),
        max  = np.asarray(out_max, dtype=np.float64) if out_max
               else np.zeros((0, n_dims)),
        com  = np.asarray(out_com, dtype=np.float64) if out_com
               else np.zeros((0, n_dims)),
        sax  = np.asarray(out_sax, dtype=np.float64) if out_sax
               else np.zeros((0, n_dims)),
        lom  = np.asarray(out_lom, dtype=np.float64) if out_lom
               else np.zeros((0, n_dims)),
        lax  = np.asarray(out_lax, dtype=np.float64) if out_lax
               else np.zeros((0, n_dims)),
        post = np.asarray(out_post, dtype=np.float64),
        ucnt = np.asarray(out_ucnt, dtype=np.int32),
        uinc = (
            np.stack(out_uinc, axis=0) if out_uinc
            else np.zeros((0, n_units), dtype=bool)
        ),
        smoothing_weights = np.diag(smoothing_matrix),
        window            = 2.0 * half_spike_window_s,
        samplerate        = float(samplerate),
    )


# ─────────────────────────────────────────────────────────────────────────── #
# Helpers — turning a PlaceFieldResult into the inputs decode needs            #
# ─────────────────────────────────────────────────────────────────────────── #

def prepare_ratemap(
    pf,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Flatten a :class:`PlaceFieldResult` into the ``(n_bins, n_units)``
    array expected by :func:`decode_ufr_boxcar`, applying a maze mask.

    Parameters
    ----------
    pf:
        :class:`PlaceFieldResult`.  Iter-0 (deterministic) maps are used.
    mask:
        Optional ``bin_shape``-shaped boolean mask.  ``True`` bins are
        kept; ``False`` bins are dropped.  ``None`` keeps all bins.

    Returns
    -------
    ratemap : np.ndarray, shape ``(n_kept_bins, n_units)``
    """
    rm0 = pf.rate_map[..., 0]                  # drop iter axis: (..., n_units)
    if mask is not None:
        flat = rm0.reshape(-1, rm0.shape[-1])
        return flat[mask.ravel(), :]
    return rm0.reshape(-1, rm0.shape[-1])


def prepare_bin_coords(
    pf,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Flatten ``pf.bin_centres`` into a ``(n_bins, n_dims)`` array.

    Bins are listed in row-major order over the spatial axes
    (matching :func:`prepare_ratemap`).  Pass the same ``mask`` you
    passed there to keep the rows aligned.
    """
    grids = np.meshgrid(*pf.bin_centres, indexing="ij")
    coords = np.stack([g.ravel() for g in grids], axis=1)   # (n_bins, n_dims)
    if mask is not None:
        coords = coords[mask.ravel(), :]
    return coords
