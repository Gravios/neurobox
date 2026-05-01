"""
neurobox.analysis.decoding.theta_phase
=======================================
Theta-phase extraction from LFP — port of
:file:`MTA/transformations/load_theta_phase.m`.

The MATLAB function did three things:

1. Bandpass-filter LFP between 5 and 13 Hz.
2. Hilbert-transform to get instantaneous phase.
3. Optionally add a per-subject calibration offset
   (``Trial.meta.correction.thetaPhase``) and resample.

The neurobox port operates on plain ndarrays so it can be called
without a Trial object.  The bandpass-filter uses the existing
:func:`butter_filter`.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import hilbert

from neurobox.analysis.lfp.filtering import butter_filter


def theta_phase(
    lfp:               np.ndarray,
    samplerate:        float,
    band:              tuple[float, float] = (5.0, 13.0),
    correction:        float = 0.0,
    resample_to:       float | None = None,
    butter_order:      int = 4,
) -> np.ndarray:
    """Extract instantaneous theta phase from an LFP trace.

    Port of :file:`MTA/transformations/load_theta_phase.m`.

    Parameters
    ----------
    lfp:
        ``(T,)`` or ``(T, n_channels)`` LFP signal.  Multi-channel
        input returns a per-channel phase trace; channels are
        processed independently.
    samplerate:
        LFP samplerate in Hz.
    band:
        ``(low, high)`` Hz for the bandpass filter.  Default
        ``(5, 13)``.
    correction:
        Per-subject phase offset added before the ``mod 2π`` wrap
        (matches MATLAB's ``Trial.meta.correction.thetaPhase``).
    resample_to:
        Optional target samplerate.  ``None`` (default) returns the
        phase at the input rate.  When given, uses linear unwrapping
        and stride sampling — for matching the MATLAB `phz.resample`
        behaviour you'd typically pass the xyz samplerate (250 Hz).
    butter_order:
        Bandpass-filter order.  Default 4 (matches MATLAB).

    Returns
    -------
    phase : np.ndarray
        Phase in radians, wrapped to ``[0, 2π)``.  Shape
        ``(T_out,)`` or ``(T_out, n_channels)`` matching the input.
    """
    lfp = np.asarray(lfp, dtype=np.float64)
    one_d = lfp.ndim == 1
    if one_d:
        lfp = lfp[:, None]

    # Bandpass filter
    filtered = butter_filter(
        lfp,
        cutoff     = list(band),
        samplerate = samplerate,
        order      = butter_order,
        btype      = "bandpass",
    )

    # Hilbert → instantaneous phase
    analytic = hilbert(filtered, axis=0)
    phase    = np.angle(analytic)

    # Unwrap (MATLAB does `unwrap(phz.data)` to get continuous values
    # before resampling — necessary so resample doesn't smear phase
    # discontinuities)
    phase_unwrapped = np.unwrap(phase, axis=0)

    if resample_to is not None and resample_to != samplerate:
        # Stride sampling — MATLAB uses interp1 internally on the
        # unwrapped phase, which is roughly equivalent for stable
        # theta phase.  For a pure linear interpolation use np.interp.
        n_in  = phase_unwrapped.shape[0]
        n_out = int(round(n_in * resample_to / samplerate))
        t_in  = np.arange(n_in) / samplerate
        t_out = np.arange(n_out) / resample_to
        cols = []
        for ch in range(phase_unwrapped.shape[1]):
            cols.append(np.interp(t_out, t_in, phase_unwrapped[:, ch]))
        phase_unwrapped = np.stack(cols, axis=1)

    # Apply correction and wrap
    phase_out = np.mod(phase_unwrapped + correction + 2 * np.pi, 2 * np.pi)

    if one_d:
        phase_out = phase_out[:, 0]
    return phase_out
