"""
neurobox.analysis.transformations.thetarc_phase
================================================
Bipolar-reference variant of theta-phase extraction.

Port of :file:`MTA/transformations/load_thetarc_phase.m`.

Identical algorithm to the regular :func:`theta_phase` (round 10) but
takes a 2-channel LFP, computes the bipolar derivative
``lfp[:, 1] - lfp[:, 0]``, and runs the standard 5–13 Hz bandpass +
Hilbert + optional resample + correction-offset chain on the
result.

Useful for radial-component theta phase derived from a current-source
density-style channel pair.
"""

from __future__ import annotations

import numpy as np

from neurobox.analysis.decoding.theta_phase import theta_phase


def thetarc_phase(
    lfp:           np.ndarray,
    samplerate:    float,
    band:          tuple[float, float] = (5.0, 13.0),
    correction:    float = 0.0,
    resample_to:   float | None = None,
    butter_order:  int = 4,
) -> np.ndarray:
    """Theta phase of the bipolar derivative of a 2-channel LFP.

    Port of :file:`MTA/transformations/load_thetarc_phase.m`.

    Parameters
    ----------
    lfp:
        ``(T, 2)`` LFP signal — typically the two channels straddling
        the hippocampal pyramidal layer for a radial-current-source
        derivative.
    samplerate:
        LFP samplerate in Hz.
    band:
        ``(low, high)`` Hz for the bandpass filter.  Default
        ``(5, 13)``.
    correction:
        Per-subject phase offset added before the ``mod 2π`` wrap.
    resample_to:
        Optional target samplerate; ``None`` preserves the input.
    butter_order:
        Bandpass-filter order.  Default 4 (matches MATLAB).

    Returns
    -------
    phase : ``(T_out,)``
        Phase in radians, wrapped to ``[0, 2π)``.
    """
    lfp = np.asarray(lfp, dtype=np.float64)
    if lfp.ndim != 2 or lfp.shape[1] != 2:
        raise ValueError(
            f"thetarc_phase expects (T, 2) LFP; got {lfp.shape}"
        )
    diff = lfp[:, 1] - lfp[:, 0]
    return theta_phase(
        diff,
        samplerate    = samplerate,
        band          = band,
        correction    = correction,
        resample_to   = resample_to,
        butter_order  = butter_order,
    )
