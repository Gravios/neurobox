"""
neurobox.analysis.lfp.csd
==========================
Current source density (CSD) analysis.

Port of :file:`labbox/TF/CurSrcDns.m` (Anton Sirota / Evgeny Resnik).
Implements the standard discrete one-dimensional CSD via a centred
second spatial derivative:

.. math::

   \\mathrm{CSD}(z, t) = -\\frac{1}{(h \\Delta z)^2}
       \\big( V(z + h \\Delta z, t) - 2 V(z, t) + V(z - h \\Delta z, t) \\big)

where :math:`V` is the LFP, :math:`\\Delta z` is the inter-contact pitch,
and :math:`h` is the integer step (``step=1`` → adjacent channels,
``step=2`` → skip one).

The negative sign uses the Pitts-Mitzdorf-Nicholson convention where
**current sinks come out positive** (i.e. inflow of positive ions →
positive CSD value).  This matches the labbox ``CurSrcDns`` output and
is the convention used in the bulk of the laminar-CSD literature
(Mitzdorf 1985 *Physiol Rev* 65:37, Nicholson & Freeman 1975
*J Neurophysiol* 38:356).

Public API
----------
``current_source_density``  — port of ``CurSrcDns``; returns CSD + axes
``CSDResult``               — container with raw CSD, optional
                              interpolated CSD, axis vectors

Relationship to ``NBDlfp.csd()``
--------------------------------
``NBDlfp.csd()`` already provides a centred second-difference CSD on an
:class:`NBDlfp` object, but with two differences from the labbox /
laminar-physiology convention:

1. **Sign**: ``NBDlfp.csd()`` returns ``+ d²V/dz²`` (no negation).  This
   module returns ``- d²V/dz²``, so positive values are sinks.
2. **Spatial scaling**: ``NBDlfp.csd()`` uses ``(2 h Δz)²`` in the
   denominator for an ``h``-channel skip.  The textbook centred
   finite-difference uses ``(h Δz)²`` — the spacing between **center
   and a neighbour**, not the full peak-to-peak span.  This module
   uses the textbook form, so its output differs from
   ``NBDlfp.csd()`` by a factor of ``-1/4`` for ``h=1``.

Use ``NBDlfp.csd()`` when you want the result wrapped in an
:class:`NBDlfp` for further filtering/sync.  Use
:func:`current_source_density` when you want raw arrays plus the
sink-positive sign convention and visualisation-ready interpolation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ─────────────────────────────────────────────────────────────────────────── #
# Result container                                                             #
# ─────────────────────────────────────────────────────────────────────────── #

@dataclass
class CSDResult:
    """Output of :func:`current_source_density`.

    Attributes
    ----------
    csd:
        CSD on the original (un-interpolated) channel grid, shape
        ``(T, n_channels - 2*step)``.  Sign convention: positive values
        are sinks.
    csd_interp:
        Bicubically-interpolated CSD for visualisation, shape
        ``(T_interp, n_ch_interp)`` where each axis is upsampled by
        ``2 ** interp_levels``.  ``None`` if ``interp_levels=0``.
    t:
        Time axis in **milliseconds** (matches labbox convention).  Length
        matches ``csd.shape[0]``.  If ``interp_levels > 0``, ``t_interp``
        gives the upsampled axis.
    z:
        Channel axis (1-based channel indices, matching MATLAB) for the
        raw CSD.  Length matches ``csd.shape[1]``.
    t_interp:
        Upsampled time axis for ``csd_interp``, or ``None``.
    z_interp:
        Upsampled channel axis for ``csd_interp``, or ``None``.
    samplerate:
        Sample rate (Hz).  Carried through for the user.
    step:
        Integer channel step used for the second difference.
    pitch_um:
        Inter-contact pitch in micrometres.  ``None`` if no physical
        pitch was provided (then ``csd`` is in units of ``V / step²``).
    """

    csd:        np.ndarray
    t:          np.ndarray
    z:          np.ndarray
    samplerate: float
    step:       int
    csd_interp: Optional[np.ndarray] = None
    t_interp:   Optional[np.ndarray] = None
    z_interp:   Optional[np.ndarray] = None
    pitch_um:   Optional[float]      = None


# ─────────────────────────────────────────────────────────────────────────── #
# CSD (port of CurSrcDns)                                                      #
# ─────────────────────────────────────────────────────────────────────────── #

def current_source_density(
    lfp: np.ndarray,
    samplerate: float = 1250.0,
    step: int = 2,
    pitch_um: Optional[float] = None,
    t_range_ms: Optional[np.ndarray] = None,
    channels: Optional[np.ndarray] = None,
    interp_levels: int = 0,
) -> CSDResult:
    """One-dimensional current source density.

    Port of :file:`labbox/TF/CurSrcDns.m` with two semantic upgrades:

    1. **Correct finite-difference scaling** — denominator is
       ``(step * pitch)²``, not ``step²`` (see module docstring).
    2. **Optional physical units** — pass ``pitch_um`` to get output in
       V/m² when LFP is in volts.  Without it, output is in
       V/sample² (matches CurSrcDns exactly).

    Parameters
    ----------
    lfp:
        LFP array, shape ``(T, n_channels)``.  **Time-first** (this
        module enforces this convention; the MATLAB source warns when
        the input looks transposed).
    samplerate:
        LFP sample rate in Hz.  Default 1250 (typical Neurosuite
        ``.lfp`` rate).
    step:
        Integer channel step for the centred second difference.  Default
        2 (matches labbox ``CurSrcDns`` default).  ``step=1`` uses
        adjacent channels; ``step=2`` skips one (gives slightly smoother
        output in the presence of channel-correlated noise, at the cost
        of one channel of resolution at each end).
    pitch_um:
        Inter-contact spacing in micrometres.  If provided, output is
        in physical units (assuming ``lfp`` is in volts).  If ``None``
        (default), denominator is ``step²`` and output units are
        ``V / sample²`` (matches labbox).
    t_range_ms:
        Optional time axis in milliseconds.  If ``None``, defaults to
        ``np.arange(T) / (samplerate / 1000)``.
    channels:
        Optional 1-based channel indices for the input columns.  If
        ``None``, defaults to ``1, 2, …, n_channels``.  Used only to
        build the output ``z`` axis.
    interp_levels:
        If positive, returns a bicubically-interpolated CSD upsampled
        by ``2 ** interp_levels`` along **both** axes.  Default 0
        (no interpolation).  Set to 3 to mirror the labbox default
        ``interp2(csd, 3)``.

    Returns
    -------
    :class:`CSDResult`

    Examples
    --------
    Raw CSD on adjacent channels::

        from neurobox.analysis.lfp import current_source_density
        result = current_source_density(lfp_array, samplerate=1250, step=1)
        # result.csd has shape (T, n_channels - 2)

    With physical units and interpolation for a heatmap::

        result = current_source_density(
            lfp_array, samplerate=1250, step=2,
            pitch_um=50.0, interp_levels=3,
        )
        # plt.imshow(result.csd_interp.T, aspect='auto',
        #            extent=[result.t_interp[0], result.t_interp[-1],
        #                    result.z_interp[-1], result.z_interp[0]])
    """
    lfp = np.asarray(lfp, dtype=np.float64)
    if lfp.ndim != 2:
        raise ValueError(f"lfp must be 2-D (T, n_channels); got shape {lfp.shape}")
    n_t, n_ch = lfp.shape
    if n_t < n_ch:
        # Same warning as the MATLAB version: shape almost certainly transposed.
        import warnings
        warnings.warn(
            f"lfp shape {lfp.shape}: T < n_channels suggests the array is "
            "transposed (this module expects time-first).",
            UserWarning,
            stacklevel=2,
        )

    if step < 1:
        raise ValueError(f"step must be ≥ 1, got {step}")
    if 2 * step >= n_ch:
        raise ValueError(
            f"step={step} too large for {n_ch} channels: need 2*step < n_channels"
        )

    # Centred second difference with sink-positive sign.
    ch_idx = np.arange(step, n_ch - step)             # centres
    csd_raw = -(lfp[:, ch_idx + step]
                - 2.0 * lfp[:, ch_idx]
                + lfp[:, ch_idx - step])
    if pitch_um is not None:
        # Convert µm → m so output is in [LFP units] / m² (V/m² for LFP in V).
        pitch_m = pitch_um * 1e-6
        csd_raw /= (step * pitch_m) ** 2
    else:
        # No physical pitch given — output is [LFP units] / sample² (matches
        # CurSrcDns).  This is dimensionally inconsistent but useful for
        # visualisation when channel spacing is unknown.
        csd_raw /= step ** 2

    # Axis construction
    if t_range_ms is None:
        t = np.arange(n_t) / (samplerate / 1000.0)
    else:
        t = np.asarray(t_range_ms, dtype=np.float64)
        if t.size != n_t:
            raise ValueError(f"t_range_ms must have length {n_t}, got {t.size}")
    if channels is None:
        channels = np.arange(1, n_ch + 1)             # 1-based, MATLAB convention
    else:
        channels = np.asarray(channels)
        if channels.size != n_ch:
            raise ValueError(f"channels must have length {n_ch}, got {channels.size}")
    z = channels[ch_idx]

    csd_interp = None
    t_interp = None
    z_interp = None
    if interp_levels > 0:
        # MATLAB ``interp2(csd, k, 'linear')`` doubles the resolution k times,
        # bilinearly.  Use scipy.ndimage.zoom which does the same in n-D.
        from scipy.ndimage import zoom
        factor = 2 ** interp_levels
        # zoom over both axes; order=1 → bilinear (matches 'linear').
        csd_interp = zoom(csd_raw, zoom=(factor, factor), order=1, prefilter=False)
        t_interp = np.linspace(t[0], t[-1], csd_interp.shape[0])
        # Channel axis with half-channel padding at each end (matches labbox)
        z_interp = np.linspace(z[0] - 0.5, z[-1] + 0.5, csd_interp.shape[1])

    return CSDResult(
        csd        = csd_raw,
        t          = t,
        z          = z,
        samplerate = float(samplerate),
        step       = int(step),
        csd_interp = csd_interp,
        t_interp   = t_interp,
        z_interp   = z_interp,
        pitch_um   = pitch_um,
    )
