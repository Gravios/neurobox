"""
neurobox.analysis.transformations.misc
=======================================
Small one-off transformations from :file:`MTA/utilities/transforms/`
that don't fit into a thematic submodule.

What's here
-----------
* :func:`make_uniform_distr` — empirical-CDF flattening.  Port of
  :file:`MakeUniformDistr.m`.
* :func:`shilbert` — analytic-signal Hilbert transform.  Thin wrapper
  over :func:`scipy.signal.hilbert`; mirrors the MATLAB convenience
  shim :file:`MTA/utilities/transforms/Shilbert.m`.
* :func:`my_theta_phase` — bandpass + Hilbert + (un)wrap convenience,
  matching the MATLAB :file:`MTA/utilities/transforms/myThetaPhase.m`.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import hilbert

from neurobox.analysis.lfp.filtering import butter_filter


__all__ = ["make_uniform_distr", "shilbert", "my_theta_phase"]


def make_uniform_distr(
    x:        np.ndarray,
    a:        float | None = None,
    b:        float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """ECDF transform — flatten an input distribution onto ``[a, b]``.

    Port of :file:`MTA/utilities/transforms/MakeUniformDistr.m`.

    Replaces each value in *x* by its rank-fraction (``ecdf``) scaled
    to the range ``[a, b]``.  If *x* is uniform on ``[a, b]`` the
    output equals the input.

    Parameters
    ----------
    x:
        ``(N,)`` real input.
    a, b:
        Output range.  Default ``(min(x), max(x))``.

    Returns
    -------
    out : np.ndarray, shape ``(N,)``
        Transformed values.
    sorted_x : np.ndarray, shape ``(N,)``
        Sorted *x* (matches MATLAB's second output argument).
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    a = float(np.min(x)) if a is None else float(a)
    b = float(np.max(x)) if b is None else float(b)
    if x.size == 0:
        return x.copy(), x.copy()
    n = x.size
    order = np.argsort(x, kind="stable")
    rank = np.empty(n, dtype=np.float64)
    rank[order] = (np.arange(1, n + 1, dtype=np.float64)) / n
    out = (b - a) * rank + a
    return out, x[order]


def shilbert(
    xr:        np.ndarray,
    n:         int | None = None,
) -> np.ndarray:
    """Discrete-time analytic signal via Hilbert transform.

    Port of :file:`MTA/utilities/transforms/Shilbert.m`, which itself
    is a copy of MATLAB's Signal Processing Toolbox ``hilbert``.
    This Python equivalent is a thin wrapper around
    :func:`scipy.signal.hilbert` for API parity.

    Parameters
    ----------
    xr:
        Real-valued ``(T,)`` or ``(T, C)`` input.  Imaginary parts are
        discarded with a warning.
    n:
        Optional zero-padding / truncation length along the time axis.

    Returns
    -------
    np.ndarray
        Complex analytic signal, same shape as *xr* (or shape with
        time axis = *n* when *n* is given).
    """
    if np.iscomplexobj(xr):
        import warnings
        warnings.warn("shilbert ignoring imaginary part of input.",
                      stacklevel=2)
        xr = np.real(xr)
    xr = np.asarray(xr, dtype=np.float64)
    return hilbert(xr, N=n, axis=0)


def my_theta_phase(
    eeg:           np.ndarray,
    samplerate:    float = 1250.0,
    freq_range:    tuple[float, float] = (4.0, 10.0),
    butter_order:  int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Theta-band instantaneous phase, amplitude, unwrapped phase, and filtered LFP.

    Port of :file:`MTA/utilities/transforms/myThetaPhase.m`.

    The MATLAB original used a Chebyshev type-II filter; this port
    follows the live (un-commented) code path which uses a 2nd-order
    Butterworth bandpass, matching the Sirota lab pipeline
    in practice.

    Parameters
    ----------
    eeg:
        ``(T,)`` LFP / EEG trace.
    samplerate:
        Hz.  Default 1250.
    freq_range:
        Bandpass band in Hz.  Default ``(4, 10)`` matches MATLAB.
    butter_order:
        Butterworth order.  Default 2 matches MATLAB.

    Returns
    -------
    theta_phase : np.ndarray, shape ``(T,)``
        Instantaneous phase in radians, wrapped to ``(-π, π]``.
    theta_amp : np.ndarray, shape ``(T,)``
        Instantaneous amplitude.
    tot_phase : np.ndarray, shape ``(T,)``
        Unwrapped phase.
    eegf : np.ndarray, shape ``(T,)``
        Bandpass-filtered LFP.
    """
    eeg = np.asarray(eeg, dtype=np.float64).ravel()
    eegf = butter_filter(
        eeg, cutoff=list(freq_range), samplerate=samplerate,
        order=butter_order, btype="bandpass",
    )
    eegf = eegf - np.mean(eegf)
    h = hilbert(eegf)
    theta_phase = np.angle(h)
    theta_amp   = np.abs(h)
    tot_phase   = np.unwrap(theta_phase)
    return theta_phase, theta_amp, tot_phase, eegf
