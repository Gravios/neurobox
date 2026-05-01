"""
neurobox.analysis.lfp.oscillations
===================================
Oscillation- and event-detection primitives.

Port of four labbox helpers + two composites (Anton Sirota / Ken Harris):

==============================  =============================================
labbox                          neurobox
==============================  =============================================
TF/LocalMinima.m                :func:`local_minima`
TF/ThreshCross.m                :func:`thresh_cross`
Helper/WithinRanges.m           :func:`within_ranges`
TF/DetectOscillations.m         :func:`detect_oscillations`
TF/DetectRipples.m              :func:`detect_ripples`
==============================  =============================================

The composite ``detect_oscillations`` is what the MTA analysis pipeline
actually calls (theta detection, gamma bursts, ripple-band events).  It
chains :func:`butter_filter` from :mod:`neurobox.analysis.lfp.filtering`
with the three primitives in this module.  ``detect_ripples`` is a thin
wrapper around it with hippocampal-ripple-specific defaults.

Conventions
-----------
* Time is in **samples** for the index outputs and **Hz** for ``samplerate``.
* Periods are returned as ``(N, 2)`` int arrays of half-open ``[start,
  stop)`` sample intervals — one row per detected event.  Convert to
  seconds with ``periods / samplerate`` or wrap in an :class:`NBEpoch`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from .filtering import butter_filter

# Try the compiled within_ranges kernel first; fall back to pure numpy.
# The Cython kernel is markedly faster at large ``n_labels`` (>= 100).
try:
    from ._within_ranges_engine import within_ranges_matrix_engine as _wr_matrix_c
    _WR_USING_CYTHON = True
except ImportError:                                              # pragma: no cover
    _wr_matrix_c = None
    _WR_USING_CYTHON = False
from ._within_ranges_python_fallback import within_ranges_matrix_engine_python


def _within_ranges_matrix_dispatch(
    x: np.ndarray,
    starts: np.ndarray,
    stops: np.ndarray,
    range_label_zero_based: np.ndarray,
    n_labels: int,
) -> np.ndarray:
    """Compute the matrix-mode boolean output via the fastest available engine.

    Used internally by :func:`within_ranges`.

    The Cython kernel consumes a pre-sorted *event stream*.  Build it
    here with ``np.lexsort`` so tied times are processed in start →
    point → stop order (this realises the labbox inclusive-on-both-ends
    semantics).
    """
    if _WR_USING_CYTHON:
        # Build merged event stream: starts (kind=0), points (kind=1), stops (kind=2)
        n_pts    = x.size
        n_ranges = starts.size
        n_events = 2 * n_ranges + n_pts
        event_time  = np.empty(n_events, dtype=np.float64)
        event_kind  = np.empty(n_events, dtype=np.int64)
        event_label = np.empty(n_events, dtype=np.int64)
        event_pidx  = np.empty(n_events, dtype=np.int64)

        # Slice [0:n_ranges] = starts
        event_time [:n_ranges] = starts
        event_kind [:n_ranges] = 0
        event_label[:n_ranges] = range_label_zero_based
        event_pidx [:n_ranges] = -1

        # Slice [n_ranges:n_ranges+n_pts] = points
        event_time [n_ranges:n_ranges + n_pts] = x
        event_kind [n_ranges:n_ranges + n_pts] = 1
        event_label[n_ranges:n_ranges + n_pts] = -1
        event_pidx [n_ranges:n_ranges + n_pts] = np.arange(n_pts, dtype=np.int64)

        # Slice [n_ranges+n_pts:] = stops
        event_time [n_ranges + n_pts:] = stops
        event_kind [n_ranges + n_pts:] = 2
        event_label[n_ranges + n_pts:] = range_label_zero_based
        event_pidx [n_ranges + n_pts:] = -1

        # Sort primarily by time, secondarily by kind (start < point < stop).
        order = np.lexsort((event_kind, event_time))
        event_kind  = np.ascontiguousarray(event_kind [order])
        event_label = np.ascontiguousarray(event_label[order])
        event_pidx  = np.ascontiguousarray(event_pidx [order])

        return _wr_matrix_c(event_kind, event_label, event_pidx,
                            n_pts, n_labels)
    else:
        return within_ranges_matrix_engine_python(
            x, starts, stops, range_label_zero_based, n_labels,
        )


# ─────────────────────────────────────────────────────────────────────────── #
# WithinRanges                                                                 #
# ─────────────────────────────────────────────────────────────────────────── #

def within_ranges(
    x: np.ndarray,
    ranges: np.ndarray,
    range_label: Optional[np.ndarray] = None,
    mode: str = "matrix",
) -> np.ndarray:
    """Test whether each element of *x* falls within any of the given ranges.

    Port of :file:`labbox/Helper/WithinRanges.m`.

    The labbox version uses an O(N log N) sort/cumsum trick for speed.
    Since numpy's :func:`numpy.searchsorted` is just as fast and far
    clearer, we use that instead — the output is identical.

    Parameters
    ----------
    x:
        1-D array of points to test.
    ranges:
        ``(N, 2)`` array of ``[start, stop]`` ranges.  Endpoints are
        **inclusive on both sides** (matches the labbox semantics).
    range_label:
        Optional length-N array of integer labels in ``[1, n_labels]``.
        Defaults to all ones.  Only used when ``mode='matrix'``.
    mode:
        ``'matrix'`` (default): boolean ``(len(x), n_labels)`` output —
        column ``j-1`` is True for points falling in any range with
        label ``j``.
        ``'vector'``: integer ``(len(x),)`` output — value ``j`` if the
        point belongs to a range with label ``j``, ``0`` otherwise.
        Raises ``ValueError`` if a point belongs to more than one range.
        ``'flat'``: convenience boolean ``(len(x),)`` — True for any
        match.  No labbox equivalent; equivalent to
        ``within_ranges(...).any(axis=1)``.

    Returns
    -------
    out : ndarray
        Shape and dtype depend on ``mode`` (see above).
    """
    x = np.asarray(x).ravel()
    ranges = np.asarray(ranges)
    if ranges.size == 0:
        if mode == "matrix":
            n_labels = 1 if range_label is None else int(np.max(range_label))
            return np.zeros((x.size, n_labels), dtype=bool)
        if mode == "vector":
            return np.zeros(x.size, dtype=np.int64)
        return np.zeros(x.size, dtype=bool)

    if ranges.ndim == 1 and ranges.size == 2:
        ranges = ranges.reshape(1, 2)
    if ranges.ndim != 2 or ranges.shape[1] != 2:
        raise ValueError("ranges must have shape (N, 2)")
    if np.any(ranges[:, 1] < ranges[:, 0]):
        raise ValueError("End must come after Start in each range")

    starts = ranges[:, 0]
    stops  = ranges[:, 1]
    n_ranges = ranges.shape[0]

    if range_label is None:
        range_label = np.ones(n_ranges, dtype=np.int64)
    else:
        range_label = np.asarray(range_label, dtype=np.int64).ravel()
        if range_label.size != n_ranges:
            raise ValueError("range_label must have one entry per range")

    if mode == "flat":
        # Fast path: just the OR over all ranges.
        # idx = number of starts ≤ x − number of stops < x
        # If idx > 0, x is inside some range.
        idx = (np.searchsorted(np.sort(starts), x, side="right")
               - np.searchsorted(np.sort(stops), x, side="left"))
        return idx > 0

    if mode == "matrix":
        n_labels = int(range_label.max())
        # range_label is 1-based in the public API.  Convert to 0-based
        # for the engine.
        rl_zero = (range_label - 1).astype(np.int64, copy=False)
        out_uint8 = _within_ranges_matrix_dispatch(
            x.astype(np.float64, copy=False),
            starts.astype(np.float64, copy=False),
            stops.astype(np.float64, copy=False),
            rl_zero,
            n_labels,
        )
        return out_uint8.astype(bool)

    if mode == "vector":
        out = np.zeros(x.size, dtype=np.int64)
        # Build per-point membership count to detect overlap.
        for k in range(1, int(range_label.max()) + 1):
            sel = range_label == k
            if not sel.any():
                continue
            s_sorted = np.sort(starts[sel])
            e_sorted = np.sort(stops[sel])
            inside = (np.searchsorted(s_sorted, x, side="right")
                      > np.searchsorted(e_sorted, x, side="left"))
            if np.any(out[inside] != 0):
                raise ValueError("Some points belong to more than one range")
            out[inside] = k
        return out

    raise ValueError(f"Unknown mode {mode!r}.  Use 'matrix', 'vector', or 'flat'.")


# ─────────────────────────────────────────────────────────────────────────── #
# ThreshCross                                                                  #
# ─────────────────────────────────────────────────────────────────────────── #

def thresh_cross(
    x: np.ndarray,
    threshold: float,
    min_interval: int = 0,
) -> np.ndarray:
    """Find contiguous periods where *x* exceeds ``threshold``.

    Port of :file:`labbox/TF/ThreshCross.m`.

    Parameters
    ----------
    x:
        1-D signal.
    threshold:
        Scalar threshold.  Periods are sample ranges where ``x > threshold``.
    min_interval:
        Minimum period length in samples.  Shorter periods are dropped.
        Set to 0 to keep every crossing pair.

    Returns
    -------
    periods : ndarray
        ``(N, 2)`` int array of ``[start, stop]`` sample indices, one
        row per period.  Empty if no crossings are detected.

    Notes
    -----
    Edge handling matches the original:

    * If the signal starts above threshold, the first period starts at
      sample 0.
    * If the signal ends above threshold, the last period stops at
      ``len(x) - 1``.
    * Zero-touching is *not* handled — assumes ``x`` does not equal
      threshold for long stretches (same caveat as the MATLAB version).
    """
    x = np.asarray(x).ravel()
    above = x > threshold
    dx = np.diff(above.astype(np.int8))
    left  = np.flatnonzero(dx == 1)   # rising edges
    right = np.flatnonzero(dx == -1)  # falling edges

    if left.size == 0 and right.size == 0:
        # Either the signal never crosses, or it stays above the threshold
        # for the entire duration.
        if above.all():
            periods = np.array([[0, len(x) - 1]], dtype=np.int64)
            if min_interval > 0 and (periods[0, 1] - periods[0, 0]) <= min_interval:
                return np.empty((0, 2), dtype=np.int64)
            return periods
        return np.empty((0, 2), dtype=np.int64)

    # Mirror MATLAB's edge handling, but extended to handle the cases the
    # MATLAB version misses (signal starts/ends above threshold and only
    # one of left/right is empty).
    if left.size == 0:
        # Signal started above threshold and only fell — period is [0, right[-1]].
        left = np.array([0], dtype=np.int64)
    elif right.size == 0:
        # Signal rose and stayed above — period is [left[0], len(x)-1].
        right = np.array([len(x) - 1], dtype=np.int64)
    else:
        if right[-1] < left[-1]:
            right = np.append(right, len(x) - 1)
        if right[0] < left[0]:
            left = np.insert(left, 0, 0)

    periods = np.column_stack([left, right]).astype(np.int64)

    if min_interval > 0:
        keep = (periods[:, 1] - periods[:, 0]) > min_interval
        periods = periods[keep]
    return periods


# ─────────────────────────────────────────────────────────────────────────── #
# LocalMinima                                                                  #
# ─────────────────────────────────────────────────────────────────────────── #

def local_minima(
    x: np.ndarray,
    not_closer_than: int = 1,
    less_than: Optional[float] = None,
    max_results: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Find local minima in a 1-D array with separation and value constraints.

    Port of :file:`labbox/TF/LocalMinima.m`.

    The original docstring (Ken Harris): *"This program is the curse of
    my life. Why can't things be simple?"*  Reproduced here with the
    same semantics:

    * Endpoints are **never** counted as minima.
    * Plateau handling: when the signal goes down → flat → up, only the
      earliest sample of equality counts as the minimum.
    * Pairs of minima closer than ``not_closer_than`` samples are
      collapsed by keeping the lower of the two.

    Parameters
    ----------
    x:
        1-D signal.
    not_closer_than:
        Minimum spacing (in samples) between successive minima.  Default
        1 — adjacent minima are kept.
    less_than:
        Only minima whose value is below this threshold count.  Default
        None (no threshold).  Applying this is faster than computing all
        minima and filtering afterwards.
    max_results:
        Limit the number of minima returned.  Positive integer →
        ``n`` smallest by value.  Negative integer → ``|n|`` largest by
        value.  None → all minima (default).

    Returns
    -------
    mins : ndarray
        Sample indices of the detected minima.
    values : ndarray
        ``x[mins]`` for convenience.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n_points = x.size

    if less_than is None:
        below = np.arange(n_points)
    else:
        below = np.flatnonzero(x < less_than)

    if below.size == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float64)

    x_below = x[below]

    # MATLAB's tricky bit: handle gaps in `below` by treating the boundary
    # of each contiguous run as a "wall" so we don't claim a minimum that
    # was actually filtered out by less_than.  The sentinel values must be
    # one *outside* the valid index range (the MATLAB `[0; …]` and
    # `[…; nPoints+1]` use 1-indexed sentinels — translated to 0-indexed
    # Python that becomes -1 and n_points).
    gap_left  = np.flatnonzero(np.diff(np.concatenate([[-1],       below])) > 1)
    gap_right = np.flatnonzero(np.diff(np.concatenate([below, [n_points]])) > 1)

    s_diff   = np.sign(np.diff(x_below))
    left_sign  = np.concatenate([[1.0], s_diff])
    left_sign[gap_left] = -1
    right_sign = np.concatenate([s_diff, [-1.0]])
    right_sign[gap_right] = 1

    # Resolve trailing zeros in right_sign with the next non-zero
    # (matches the MATLAB ``for i=fliplr(Zeros(:)')`` loop).
    zeros_idx = np.flatnonzero(right_sign == 0)
    for i in zeros_idx[::-1]:
        if i + 1 < right_sign.size:
            right_sign[i] = right_sign[i + 1]

    mins = below[(left_sign < 0) & (right_sign > 0)]

    # Enforce separation: collapse pairs closer than not_closer_than samples.
    if not_closer_than > 1 and mins.size > 1:
        while True:
            too_close = np.flatnonzero(np.diff(mins) < not_closer_than)
            if too_close.size == 0:
                break
            # For each too-close pair, drop the higher-valued one.
            offset = (x[mins[too_close + 1]] > x[mins[too_close]]).astype(np.int64)
            # offset = 1 → drop right; offset = 0 → drop left
            drop = too_close + offset
            keep = np.ones(mins.size, dtype=bool)
            keep[np.unique(drop)] = False
            mins = mins[keep]

    if max_results is not None and max_results != 0:
        if mins.size == 0:
            mins = np.full(abs(max_results), -1, dtype=np.int64)
            return mins, np.full(abs(max_results), np.nan)
        order = np.argsort(x[mins])
        if max_results < 0:
            order = order[::-1]
        k = abs(max_results)
        if mins.size < k:
            # Pad with NaNs / -1 to match the labbox output shape.
            sel_mins = np.full(k, -1, dtype=np.int64)
            sel_vals = np.full(k, np.nan, dtype=np.float64)
            sel_mins[:mins.size] = mins[order]
            sel_vals[:mins.size] = x[mins[order]]
            return sel_mins, sel_vals
        sel = order[:k]
        return mins[sel], x[mins[sel]]

    return mins, x[mins]


# ─────────────────────────────────────────────────────────────────────────── #
# DetectOscillations                                                           #
# ─────────────────────────────────────────────────────────────────────────── #

@dataclass
class OscillationResult:
    """Output of :func:`detect_oscillations`.

    Attributes
    ----------
    peaks:
        ``(P,)`` int array of sample indices — one per detected
        oscillation peak (local maximum of the smoothed RMS envelope).
    power:
        ``(P,)`` envelope amplitude at each peak (signal units, same as
        ``x``).
    z_power:
        ``(P,)`` z-scored amplitude at each peak (across the full
        recording).
    periods:
        ``(P, 2)`` int array — half-open ``[start, stop)`` sample
        indices of the oscillation period containing each peak.
    duration_ms:
        ``(P,)`` period length in milliseconds.
    troughs:
        Sample indices of all band-pass-filtered signal troughs falling
        within the detected periods.  Each trough is a candidate
        oscillation cycle reference for phase analyses.
    threshold:
        Tuple ``(peak_threshold, duration_threshold)`` actually used.
    samplerate:
        Sample rate the result was computed at.
    """

    peaks:        np.ndarray
    power:        np.ndarray
    z_power:      np.ndarray
    periods:      np.ndarray
    duration_ms:  np.ndarray
    troughs:      np.ndarray
    threshold:    tuple[float, float]
    samplerate:   float


def detect_oscillations(
    x: np.ndarray,
    freq_range: tuple[float, float],
    samplerate: float = 1250.0,
    min_cycles: int = 7,
    threshold: float | Sequence[float] = 90.0,
    max_duration_ms: Optional[float] = None,
    min_interval: Optional[int] = None,
) -> OscillationResult:
    """Detect oscillatory bursts in a frequency band.

    Port of :file:`labbox/TF/DetectOscillations.m`.

    Algorithm
    ---------
    1. Band-pass-filter ``x`` to ``freq_range`` with a 2nd-order
       Butterworth (zero-phase).
    2. Compute the squared envelope and smooth it with a Gaussian window
       of length ``min_cycles`` cycles at the lower edge of
       ``freq_range``.  This gives the RMS power.
    3. Pick the *peak* threshold either as a percentile (``threshold ≥ 20``)
       or as a multiple of standard deviation (``threshold < 20``).
       The duration threshold defaults to half the median of the peak
       amplitudes (or the second element of ``threshold`` if given).
    4. Find peaks of the smoothed envelope above the peak threshold,
       and the surrounding periods where the envelope exceeds the
       duration threshold for at least ``min_cycles`` cycles.
    5. Optionally drop periods longer than ``max_duration_ms``.
    6. Return peaks, periods, durations, and the band-pass-filtered
       trough indices within each period.

    Parameters
    ----------
    x:
        1-D LFP signal.
    freq_range:
        ``(f_low, f_high)`` in Hz.
    samplerate:
        Hz.  Default 1250 (typical Neurosuite ``.lfp`` rate).
    min_cycles:
        Minimum number of cycles of the band centre that an
        oscillation must contain.  Default 7 (the labbox default).
    threshold:
        Peak detection threshold.  Both elements are interpreted using
        the **same** rule, decided by inspecting whether *any* element
        exceeds 20:

        * If ``any(threshold) ≥ 20``: each element is a **percentile**
          of the envelope (e.g. ``90`` → 90th percentile, ``80`` →
          80th).
        * Otherwise: each element is a **σ multiplier**
          (``thr * std(amp) + mean(amp)``).  Use ``3`` or ``5`` for
          "≥ 3 σ above mean".

        Concretely:

        * Scalar (e.g. ``5`` or ``90``) — peak threshold.  Duration
          threshold defaults to ``0.5 * median(|amp[peaks]|)``.
        * Length-2 sequence ``(peak, duration)`` — both transformed
          together.  E.g. ``(5, 2)`` → peak at 5 σ, duration at 2 σ.
    max_duration_ms:
        Drop periods longer than this many milliseconds.  None (default)
        keeps everything.
    min_interval:
        Minimum number of samples between successive peaks.  Defaults to
        ``1.5 * MinDuration`` (samples).

    Returns
    -------
    :class:`OscillationResult`

    Examples
    --------
    Detect theta bursts on an LFP channel sampled at 1250 Hz::

        from neurobox.analysis.lfp.oscillations import detect_oscillations
        result = detect_oscillations(lfp_data, (6.0, 12.0), samplerate=1250)
        theta_periods_sec = result.periods / result.samplerate
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    f_low, f_high = float(freq_range[0]), float(freq_range[1])
    if f_low >= f_high:
        raise ValueError("freq_range must be (low, high) with low < high")

    # 1. Band-pass filter
    fx = butter_filter(x, [f_low, f_high], samplerate, order=2, btype="bandpass")

    # 2. RMS envelope: equivalent to MATLAB ``sqrt(filtfilt(Window, 1, fx²))``.
    #
    # filtfilt applies the FIR window twice (forward + backward), so the
    # effective smoothing kernel is the autocorrelation of `window`.  For a
    # symmetric gaussian, this is just `window ⊗ window` — a wider
    # gaussian (σ' = σ √2).  Computing this with a single-pass
    # `oaconvolve` (overlap-add FFT convolution) is **~6.8× faster** than
    # `filtfilt` for the long kernels typical here (~1000 taps for theta
    # at 1250 Hz), and produces identical results modulo edge effects.
    # Profiling on 1-hour LFP: filtfilt 1242 ms → oaconvolve 182 ms.
    min_duration = samplerate / np.mean(freq_range) * min_cycles  # samples
    if min_interval is None:
        min_interval = int(1.5 * min_duration)

    win_len = int(round(samplerate / f_low * min_cycles))
    win_len = max(win_len, 3)
    # gausswin(WinLen, MinCycles): MATLAB's gausswin(L, alpha) has
    # std-dev = (L - 1) / (2 * alpha).  Use the same here.
    n = np.arange(win_len)
    alpha = float(min_cycles)
    sigma_w = (win_len - 1) / (2.0 * alpha)
    window = np.exp(-0.5 * ((n - (win_len - 1) / 2.0) / sigma_w) ** 2)
    window /= window.sum()

    from scipy.signal import oaconvolve
    # window_ff = window ⊗ window (the effective kernel of filtfilt(window, 1, .))
    window_ff = np.convolve(window, window)
    window_ff /= window_ff.sum()  # ensure unit-area smoothing
    amp = np.sqrt(np.maximum(oaconvolve(fx ** 2, window_ff, mode="same"), 0.0))
    z_amp = (amp - amp.mean()) / amp.std() if amp.std() > 0 else amp - amp.mean()

    # 3. Threshold logic
    # Mirrors DetectOscillations.m: if ANY element > 20, treat the WHOLE
    # vector as percentiles; otherwise treat the WHOLE vector as σ
    # multipliers (Thresh = thr * std + mean).  This was originally
    # applied only to the first element — fixed in round 3 to match the
    # MATLAB behaviour and make the duration threshold scale-correct.
    if np.isscalar(threshold):
        thr_arr = np.atleast_1d(np.asarray(threshold, dtype=np.float64))
    else:
        thr_arr = np.asarray(threshold, dtype=np.float64).ravel()

    if np.any(thr_arr > 20):
        thresh = np.percentile(amp, thr_arr)
    else:
        thresh = thr_arr * amp.std() + amp.mean()
    peak_thr = float(thresh[0])

    # 4. Peak detection on -amp (LocalMinima of -amp ≡ peaks of amp).
    peaks, _ = local_minima(-amp, not_closer_than=int(min_interval),
                            less_than=-peak_thr)
    # Note: the labbox call passes -peak_thr as `LessThan` for `LocalMinima(-amp)`.
    # The result is "peaks of amp where amp > peak_thr".

    # 5. Duration threshold
    if thr_arr.size == 1:
        if peaks.size == 0:
            dur_thr = 0.5 * peak_thr
        else:
            dur_thr = 0.5 * float(np.median(np.abs(amp[peaks])))
    else:
        dur_thr = float(thresh[1])

    osc_periods = thresh_cross(amp, dur_thr, min_interval=int(min_duration))

    # 6. Filter on max duration
    if max_duration_ms is not None and osc_periods.size > 0:
        max_samples = max_duration_ms * samplerate / 1000.0
        keep = (osc_periods[:, 1] - osc_periods[:, 0]) < max_samples
        osc_periods = osc_periods[keep]

    # Keep only peaks that lie within remaining periods.
    if peaks.size > 0 and osc_periods.size > 0:
        in_period = within_ranges(peaks, osc_periods, mode="flat")
        peaks = peaks[in_period]

    if peaks.size > 0 and osc_periods.size > 0:
        # Map each peak to the period it belongs to.
        starts = osc_periods[:, 0]
        stops  = osc_periods[:, 1]
        # For each peak: find the index of the period whose [start, stop]
        # contains it.  Periods are non-overlapping → searchsorted works.
        idx = np.searchsorted(starts, peaks, side="right") - 1
        valid = (idx >= 0) & (idx < osc_periods.shape[0])
        valid &= (peaks <= stops[np.clip(idx, 0, osc_periods.shape[0] - 1)])
        peaks = peaks[valid]
        idx = idx[valid]
        period_per_peak = osc_periods[idx]
        duration_ms = (period_per_peak[:, 1] - period_per_peak[:, 0]) / samplerate * 1000.0
    else:
        period_per_peak = np.empty((0, 2), dtype=np.int64)
        duration_ms = np.array([], dtype=np.float64)

    # 7. Troughs of the band-pass signal within periods.
    fx_troughs, _ = local_minima(fx, not_closer_than=1, less_than=0.0)
    if fx_troughs.size > 0 and osc_periods.size > 0:
        in_per = within_ranges(fx_troughs, osc_periods, mode="flat")
        troughs = fx_troughs[in_per]
    else:
        troughs = np.array([], dtype=np.int64)

    return OscillationResult(
        peaks       = peaks.astype(np.int64),
        power       = amp[peaks] if peaks.size > 0 else np.array([], dtype=np.float64),
        z_power     = z_amp[peaks] if peaks.size > 0 else np.array([], dtype=np.float64),
        periods     = period_per_peak.astype(np.int64),
        duration_ms = duration_ms,
        troughs     = troughs.astype(np.int64),
        threshold   = (peak_thr, dur_thr),
        samplerate  = float(samplerate),
    )


# ─────────────────────────────────────────────────────────────────────────── #
# DetectRipples                                                                #
# ─────────────────────────────────────────────────────────────────────────── #

def detect_ripples(
    lfp: np.ndarray,
    samplerate: float = 1250.0,
    freq_range: tuple[float, float] = (100.0, 250.0),
    threshold: float | Sequence[float] = (5.0, 2.0),
    min_cycles: int = 5,
) -> OscillationResult:
    """Detect hippocampal sharp-wave ripples.

    Port of :file:`labbox/TF/DetectRipples.m` (Anton Sirota, 2013-12-17).

    The labbox ``DetectRipples`` reads the LFP from disk, dispatches to
    ``DetectOscillations`` with ripple-specific defaults (100-250 Hz
    band, 5 cycles minimum, ``[5, 2]`` σ threshold), and writes
    ``.spw`` / ``.evt`` files for Neuroscope.  This port handles only
    the *detection* — file I/O is left to the caller, who can wrap the
    output in an :class:`NBEpoch` or write event files using
    :mod:`neurobox.io`.

    Parameters
    ----------
    lfp:
        1-D LFP signal from a hippocampal pyramidal-layer channel.
    samplerate:
        Hz.  Default 1250 (typical Neurosuite ``.lfp`` rate).
    freq_range:
        Ripple band in Hz.  Default (100, 250) — matches the labbox
        default.  Some labs prefer (140, 230); pass it explicitly to
        change.
    threshold:
        Either a scalar (peak threshold only — duration threshold
        defaults to half the median peak amplitude), or a length-2
        ``(peak_thr, duration_thr)`` pair.  Default ``(5, 2)`` —
        meaning "5 σ above mean" for peak and "2 σ above mean" for
        duration (interpreted as σ multipliers because each value is
        ``< 20``; see :func:`detect_oscillations`).
    min_cycles:
        Minimum number of ripple cycles per detected event.  Default 5
        (labbox default).  Smaller values catch more events but include
        more noise; larger values are stricter.

    Returns
    -------
    :class:`OscillationResult`
        Same fields as :func:`detect_oscillations`.  ``peaks`` is the
        sample index of each ripple peak; ``periods`` is the
        ``[start, stop]`` window where the envelope crosses the duration
        threshold; ``duration_ms`` is the ripple length; ``troughs`` are
        all band-pass-filtered LFP troughs falling inside ripple
        periods (use these as cycle references for phase analyses).

    Examples
    --------
    Standard SWR detection on a CA1 pyramidal channel::

        from neurobox.analysis.lfp import detect_ripples
        result = detect_ripples(ca1_lfp, samplerate=1250)
        ripple_times_sec = result.peaks / result.samplerate
        ripple_periods = result.periods / result.samplerate

    To save as :class:`NBEpoch`::

        from neurobox.dtype import NBEpoch
        ripples_ep = NBEpoch(
            data=result.periods / result.samplerate,
            samplerate=result.samplerate,
            label="ripples",
            key="r",
        )

    Notes
    -----
    The labbox ``DetectRipples`` uses ``Threshold = [5, 2]`` by default.
    The first element is the peak threshold; the second is the duration
    threshold, both in σ units.  This is more permissive than the
    Csicsvari/Ylinen tradition (peak threshold of 7 σ); tighten by
    passing ``threshold=(7, 2)`` if you want a more conservative cut.
    """
    return detect_oscillations(
        x=lfp,
        freq_range=freq_range,
        samplerate=samplerate,
        min_cycles=min_cycles,
        threshold=threshold,
    )
