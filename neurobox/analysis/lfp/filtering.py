"""
neurobox.analysis.lfp.filtering
================================
Time-domain filter primitives used throughout the analysis pipeline.

Port of three labbox/TF helpers (Ken Harris / Anton Sirota):

==========  ==============================================================
labbox      neurobox
==========  ==============================================================
ButFilter   :func:`butter_filter`   — zero-phase Butterworth (filtfilt)
Filter0     :func:`filter0`         — zero-phase FIR (manual delay-shift)
FirFilter   :func:`fir_filter`      — FIR design + zero-phase application
==========  ==============================================================

Why a separate module
---------------------
``NBData.filter()`` already provides Butterworth band-pass / low-pass /
high-pass via ``scipy.signal.sosfiltfilt`` for in-place filtering of any
``NBData`` subclass.  The helpers here serve a different purpose:

* They operate on **plain numpy arrays**, so they can be called from
  array-only code paths (oscillation detection, EMG removal, the
  multi-taper feature extractors) without instantiating an ``NBDlfp``.
* They take cutoff in **Hz** with a ``samplerate`` argument — closer to
  the labbox/MTA call sites which pass ``Fs`` explicitly.
* ``filter0`` and ``fir_filter`` cover designs that ``NBData.filter()``
  does not.

Conventions
-----------
* Input arrays are ``(T,)`` or ``(T, C)``.  The time axis is **always
  axis 0**.  Higher-dimensional inputs are flattened over the channel
  axes for filtering and reshaped on return.
* Cutoffs are in Hz.  Internally they are converted to normalised
  frequencies via ``Wn = cutoff / (samplerate / 2)``.
* All three functions preserve dtype where possible but compute in
  ``float64``.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.signal import butter, sosfiltfilt, filtfilt, firwin, firls, lfilter


# ─────────────────────────────────────────────────────────────────────────── #
# Internal helpers                                                            #
# ─────────────────────────────────────────────────────────────────────────── #

def _as_2d_time_first(x: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
    """Reshape ``x`` to ``(T, -1)`` and return the original shape."""
    arr = np.asarray(x)
    if arr.ndim == 0:
        raise ValueError("Cannot filter a scalar.")
    if arr.ndim == 1:
        return arr.astype(np.float64, copy=True).reshape(-1, 1), arr.shape
    return arr.astype(np.float64, copy=True).reshape(arr.shape[0], -1), arr.shape


def _restore_shape(y: np.ndarray, original_shape: tuple[int, ...]) -> np.ndarray:
    """Inverse of :func:`_as_2d_time_first`."""
    return y.reshape(original_shape)


def _normalise_cutoff(
    cutoff: float | Sequence[float],
    samplerate: float,
) -> float | list[float]:
    """Convert Hz cutoff(s) to ``Wn = f / (Fs / 2)``."""
    nyq = samplerate / 2.0
    if np.isscalar(cutoff):
        return float(cutoff) / nyq
    return [float(c) / nyq for c in cutoff]


def _validate_btype(btype: str) -> str:
    """Map labbox flag aliases to the scipy.signal name."""
    table = {
        "low":      "lowpass",
        "lowpass":  "lowpass",
        "high":     "highpass",
        "highpass": "highpass",
        "band":     "bandpass",
        "bandpass": "bandpass",
        "stop":     "bandstop",
        "bandstop": "bandstop",
    }
    if btype not in table:
        raise ValueError(
            f"Unknown btype {btype!r}.  "
            f"Use one of {sorted(set(table))}."
        )
    return table[btype]


# ─────────────────────────────────────────────────────────────────────────── #
# ButFilter                                                                    #
# ─────────────────────────────────────────────────────────────────────────── #

def butter_filter(
    x: np.ndarray,
    cutoff: float | Sequence[float],
    samplerate: float,
    order: int = 4,
    btype: str = "lowpass",
) -> np.ndarray:
    """Zero-phase Butterworth filter.

    Port of :file:`labbox/TF/ButFilter.m`.

    The labbox call ``ButFilter(x, n, wn, flag)`` takes ``wn`` in
    *normalised* units already (Hz / (Fs / 2)).  This function takes
    ``cutoff`` in **Hz** and an explicit ``samplerate`` so call sites
    don't have to pre-divide.

    Parameters
    ----------
    x:
        Input signal, shape ``(T,)`` or ``(T, C)`` (time-first).
    cutoff:
        Cutoff frequency in Hz.  Scalar for ``'lowpass'`` / ``'highpass'``,
        ``[low, high]`` for ``'bandpass'`` / ``'bandstop'``.
    samplerate:
        Sample rate of ``x`` in Hz.
    order:
        Filter order (default 4).  This is the order *passed to*
        :func:`scipy.signal.butter`; ``filtfilt`` doubles the effective
        order.  labbox typically uses 2 or 4.
    btype:
        ``'lowpass'`` (alias ``'low'``), ``'highpass'`` (``'high'``),
        ``'bandpass'`` (``'band'``), or ``'bandstop'`` (``'stop'``).

    Returns
    -------
    y : ndarray
        Filtered signal, same shape and dtype-promoted to ``float64``.

    Notes
    -----
    Implemented with second-order-sections (``output='sos'``) and
    :func:`scipy.signal.sosfiltfilt`.  This is numerically more stable
    than the ``[b, a]`` form MATLAB returns, especially for higher
    orders.
    """
    btype_n = _validate_btype(btype)
    wn = _normalise_cutoff(cutoff, samplerate)

    arr2d, original_shape = _as_2d_time_first(x)
    sos = butter(order, wn, btype=btype_n, output="sos")
    y2d = sosfiltfilt(sos, arr2d, axis=0)
    return _restore_shape(y2d, original_shape)


# ─────────────────────────────────────────────────────────────────────────── #
# Filter0                                                                      #
# ─────────────────────────────────────────────────────────────────────────── #

def filter0(b: np.ndarray, x: np.ndarray, shift: bool = True) -> np.ndarray:
    """Zero-phase FIR filter via manual delay correction.

    Port of :file:`labbox/TF/Filter0.m`.

    Applies an FIR filter ``b`` with a one-pass causal :func:`scipy.signal.lfilter`
    and corrects the group delay by shifting the output left by
    ``(len(b) - 1) / 2`` samples.  This matches the MATLAB ``Filter0``
    semantics, which differ from ``filtfilt`` (forward-backward and
    therefore filters by ``|H(f)|^2``).

    Use ``filter0`` when you specifically want one-pass filtering with
    the magnitude response equal to the FIR design (typically for
    spectrum estimation).  Use ``filtfilt``-style functions
    (``butter_filter``, ``fir_filter``) for phase-faithful applications.

    Parameters
    ----------
    b:
        FIR filter coefficients, length ``nb`` (must be odd if
        ``shift=True``).
    x:
        Input signal, shape ``(T,)`` or ``(T, C)``.
    shift:
        If True (default), apply the labbox delay-correction step.  If
        False, behave as plain :func:`scipy.signal.lfilter`.

    Returns
    -------
    y : ndarray
        Filtered signal of the same shape as ``x``.

    Raises
    ------
    ValueError
        If ``shift=True`` and ``len(b)`` is even (the original MATLAB
        function rejects this case explicitly: ``'filter order should
        be odd'``).
    """
    b = np.asarray(b, dtype=np.float64).ravel()
    nb = b.size
    if shift and nb % 2 != 1:
        raise ValueError("filter length must be odd when shift=True")

    arr2d, original_shape = _as_2d_time_first(x)
    n_t = arr2d.shape[0]

    # MATLAB Filter0 reflects the last nb samples to suppress the right-edge
    # transient (the "10.3.21 AS" addition).  We do the same.
    extension = arr2d[-1:-nb - 1:-1, :]
    if extension.shape[0] != nb:
        # Signal shorter than nb samples; skip the extension trick.
        extension = np.empty((0, arr2d.shape[1]), dtype=arr2d.dtype)
    extended = np.concatenate([arr2d, extension], axis=0)

    if shift:
        y_full, zf = lfilter(b, 1.0, extended, axis=0, zi=np.zeros((nb - 1, extended.shape[1])))
        s = (nb - 1) // 2
        # MATLAB: y = [y0(shift+1:end,:) ; z(1:shift,:)]
        y_full = np.concatenate([y_full[s:, :], zf[:s, :]], axis=0)
    else:
        y_full = lfilter(b, 1.0, extended, axis=0)

    # Trim back to the original length
    y2d = y_full[:n_t, :]
    return _restore_shape(y2d, original_shape)


# ─────────────────────────────────────────────────────────────────────────── #
# FirFilter                                                                    #
# ─────────────────────────────────────────────────────────────────────────── #

def fir_filter(
    x: np.ndarray,
    cutoff: float | Sequence[float],
    samplerate: float,
    order: int = 0,
    btype: str = "lowpass",
    design: str = "fir1",
) -> tuple[np.ndarray, np.ndarray]:
    """FIR filter design and zero-phase application.

    Port of :file:`labbox/TF/FirFilter.m`.

    Designs an FIR filter using :func:`scipy.signal.firwin` (matches
    MATLAB ``fir1``) or :func:`scipy.signal.firls` and applies it via
    :func:`scipy.signal.filtfilt`.

    Parameters
    ----------
    x:
        Input signal, shape ``(T,)`` or ``(T, C)``.
    cutoff:
        Cutoff frequency in Hz.  Scalar for ``'lowpass'`` / ``'highpass'``,
        ``[low, high]`` for ``'bandpass'`` / ``'bandstop'``.
    samplerate:
        Sample rate of ``x`` in Hz.
    order:
        Filter order.  ``0`` (the labbox default) selects an adaptive
        order: ``3 * round(2 / wn_min)``, with a 15-tap floor.
    btype:
        ``'lowpass'``, ``'highpass'``, or ``'bandpass'``.  ``firls`` does
        not implement bandstop here.
    design:
        ``'fir1'`` (default — windowed, equivalent to MATLAB ``fir1``)
        or ``'firls'`` (least-squares, transition-band design).

    Returns
    -------
    y : ndarray
        Filtered signal, same shape as ``x``.
    coeffs : ndarray
        FIR coefficients used (length ``order + 1``).

    Notes
    -----
    The labbox FirFilter applies the filter with ``filtfilt`` (the in-file
    comment notes this was changed from ``Filter0`` for compatibility
    with newer MATLAB).  We follow that — call :func:`filter0` directly
    if you need the older single-pass behaviour.
    """
    btype_n = _validate_btype(btype)
    wn = _normalise_cutoff(cutoff, samplerate)

    # Adaptive order selection (labbox FirFilter, "n==0 case")
    MIN_N = 15
    MIN_FAC = 3
    if order == 0:
        if btype_n in ("bandpass", "highpass"):
            wn_min = wn[0] if isinstance(wn, list) else wn
            order = MIN_FAC * int(2 / wn_min)
        elif btype_n == "lowpass":
            wn_max = wn[1] if isinstance(wn, list) else wn
            order = MIN_FAC * int(2 / wn_max)
        else:
            raise ValueError(
                f"Adaptive order (order=0) not supported for btype={btype!r}."
            )
        order = max(order, MIN_N)
    # scipy.signal.firwin and firls require **odd** numtaps for type-I
    # band-pass / high-pass / band-stop filters (zero gain at Nyquist
    # would otherwise be impossible).  numtaps = order + 1, so we want
    # order *even*.  This is one place we deviate from the labbox MATLAB
    # — the original forces order odd (numtaps even = type-II), which
    # MATLAB tolerates but scipy rejects.  Even-order, odd-length is the
    # textbook convention.
    if order % 2 != 0:
        order += 1

    if design == "fir1":
        # scipy.signal.firwin is the equivalent of MATLAB fir1.
        # numtaps = order + 1; pass_zero controls btype.
        numtaps = order + 1
        if btype_n == "lowpass":
            coeffs = firwin(numtaps, wn, pass_zero=True)
        elif btype_n == "highpass":
            coeffs = firwin(numtaps, wn, pass_zero=False)
        elif btype_n == "bandpass":
            coeffs = firwin(numtaps, wn, pass_zero=False)
        elif btype_n == "bandstop":
            coeffs = firwin(numtaps, wn, pass_zero=True)
        else:
            raise ValueError(f"Unhandled btype {btype!r}")
    elif design == "firls":
        TRANS = 0.15
        if btype_n == "bandpass":
            f = [0.0, (1 - TRANS) * wn[0], wn[0], wn[1], (1 + TRANS) * wn[1], 1.0]
            m = [0,   0,                  1,     1,     0,                   0]
        elif btype_n == "highpass":
            f = [0.0, (1 - TRANS) * wn,   wn,    1.0]
            m = [0,   0,                  1,     1]
        elif btype_n == "lowpass":
            f = [0.0, wn,                 (1 + TRANS) * wn, 1.0]
            m = [1,   1,                  0,                0]
        else:
            raise ValueError(f"firls not implemented for btype={btype!r}")
        # firls also takes numtaps = order + 1
        coeffs = firls(order + 1, f, m)
    else:
        raise ValueError(f"Unknown design {design!r}.  Use 'fir1' or 'firls'.")

    arr2d, original_shape = _as_2d_time_first(x)
    y2d = filtfilt(coeffs, 1.0, arr2d, axis=0)
    return _restore_shape(y2d, original_shape), coeffs
