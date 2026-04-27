"""
neurobox.analysis.lfp.spectral
==============================
Multi-taper spectral analysis of LFP and related continuous signals.

Port of the labbox TF toolkit (Ken Harris / Anton Sirota / Partha Mitra)
and the MTA ``fet_spec`` feature extractor.

Public API
----------
``SpectralParams``          — parameter dataclass (replaces ``mtparam`` / ``def_spec_parm``)
``multitaper_spectrogram``  — power spectrogram per channel  (↔ ``mtcsglong``)
``multitaper_coherogram``   — coherence + phase spectrogram  (↔ ``mtchglong``)
``multitaper_cross_spectrogram`` — complex cross-spectrum    (↔ ``mtcsdglong``)
``multitaper_psd``          — single-segment PSD             (↔ ``mtcsd``)
``whiten_ar``               — AR pre-whitening               (↔ ``WhitenSignal``)
``SpectrumResult``          — result container
``fet_spec``                — MTA-compatible feature wrapper

All spectral functions share the same DPSS taper engine and parameter
conventions so that results are directly comparable.

Parameter conventions
---------------------
All frequencies are in **Hz**.
Output power is in **signal units² / Hz** (single-sided, normalised so
that the integral over frequency equals the signal variance).
Coherence is a real value in ``[0, 1]``; phase is in ``[-π, π]`` radians.

Output array axis order (time-first, C-contiguous)
---------------------------------------------------
``power``    : ``(T, F, C)``       — time × freq × channel
``coherence``: ``(T, F, C, C)``    — time × freq × ch_i × ch_j
``phase``    : ``(T, F, C, C)``    — same
``cross``    : ``(T, F, C, C)``    — complex, same

MATLAB correspondence
---------------------
::

  MATLAB mtcsglong  → multitaper_spectrogram   → result.power  (T, F, C)
  MATLAB mtchglong  → multitaper_coherogram    → result.coherence (T,F,C,C)
                                                  result.phase
  MATLAB mtcsdglong → multitaper_cross_spectrogram → result.cross (T,F,C,C)
  MATLAB mtcsd      → multitaper_psd           → result.power  (F, C, C)
  MATLAB WhitenSignal → whiten_ar
  MATLAB fet_spec   → fet_spec
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# scipy.signal.windows.dpss is the correct path in scipy ≥ 1.1
from scipy.signal.windows import dpss as _dpss
from scipy.linalg         import solve_toeplitz


# ─────────────────────────────────────────────────────────────────────────── #
# Parameter dataclass (replaces mtparam + def_spec_parm)                     #
# ─────────────────────────────────────────────────────────────────────────── #

@dataclass
class SpectralParams:
    """Multi-taper spectrogram parameters.

    Mirrors the ``parspec`` struct passed to ``mtcsglong`` / ``mtchglong`` /
    ``mtcsdglong`` / ``mtcsd``.

    Parameters
    ----------
    samplerate:
        Signal sample rate in Hz (``Fs`` in MATLAB).
    n_fft:
        FFT length in samples (``nFFT``).  Defaults to ``win_len``.
    win_len:
        Window length in samples (``WinLength``).
    n_overlap:
        Overlap between successive windows in samples (``nOverlap``).
        Default is ``win_len // 2``.
    nw:
        Time-bandwidth product for DPSS tapers (``NW``).  Must satisfy
        ``1 < NW < win_len / 2``.  Default 3.
    n_tapers:
        Number of tapers to use.  Default ``2 * nw - 1``.
    detrend:
        Detrending applied to each window before FFT.
        ``'linear'`` (default), ``'constant'``, or ``None``.
    freq_range:
        ``(f_low, f_high)`` Hz.  Frequencies outside this band are
        discarded from the output.  Default ``(0, samplerate / 2)``.
    block_size:
        Number of time windows processed per memory block.  Default 256.
        Reduce if RAM is limited.

    Class methods
    -------------
    ``for_lfp(samplerate)``   — standard LFP parameters (1–140 Hz)
    ``for_xyz(samplerate)``   — kinematics / behavioural feature parameters (0.1–50 Hz)
    ``for_wideband(samplerate)`` — wideband (1–500 Hz or Nyquist)
    """

    samplerate: float    = 1250.0
    n_fft:      int      = 1024
    win_len:    int      = 1024
    n_overlap:  int      = 512
    nw:         float    = 3.0
    n_tapers:   Optional[int] = None          # resolved in __post_init__
    detrend:    Optional[str] = "linear"
    freq_range: tuple[float, float] = (1.0, 140.0)
    block_size: int      = 256

    def __post_init__(self) -> None:
        if self.n_tapers is None:
            self.n_tapers = int(2 * self.nw) - 1
        if self.n_fft < self.win_len:
            self.n_fft = self.win_len

    # ── Preset constructors ──────────────────────────────────────────────── #

    @classmethod
    def for_lfp(cls, samplerate: float = 1250.0) -> "SpectralParams":
        """Standard LFP parameters: 2¹⁰ window, 1–140 Hz.  NW=3."""
        return cls(
            samplerate  = samplerate,
            n_fft       = 2048,
            win_len     = 1024,
            n_overlap   = 512,
            nw          = 3.0,
            freq_range  = (1.0, 140.0),
        )

    @classmethod
    def for_xyz(cls, samplerate: float = 120.0) -> "SpectralParams":
        """Kinematics parameters: 2⁷ window, 0.1–50 Hz.  NW=3."""
        return cls(
            samplerate  = samplerate,
            n_fft       = 256,
            win_len     = 128,
            n_overlap   = int(128 * 0.875),
            nw          = 3.0,
            freq_range  = (0.1, 50.0),
        )

    @classmethod
    def for_wideband(cls, samplerate: float = 20000.0) -> "SpectralParams":
        """Wideband parameters up to Nyquist.  NW=3."""
        return cls(
            samplerate  = samplerate,
            n_fft       = 4096,
            win_len     = 4096,
            n_overlap   = 2048,
            nw          = 3.0,
            freq_range  = (1.0, samplerate / 2 - 1.0),
        )

    # ── Derived quantities ───────────────────────────────────────────────── #

    @property
    def step(self) -> int:
        """Window step in samples."""
        return self.win_len - self.n_overlap

    @property
    def freq_resolution(self) -> float:
        """Frequency resolution in Hz (bandwidth per bin)."""
        return self.samplerate / self.n_fft

    @property
    def time_resolution(self) -> float:
        """Time resolution (step size) in seconds."""
        return self.step / self.samplerate

    def freq_axis(self) -> np.ndarray:
        """Return the frequency axis for the specified ``freq_range``."""
        f_full = np.fft.rfftfreq(self.n_fft, 1.0 / self.samplerate)
        mask = (f_full >= self.freq_range[0]) & (f_full <= self.freq_range[1])
        return f_full[mask]

    def __repr__(self) -> str:
        return (
            f"SpectralParams(sr={self.samplerate}Hz, win={self.win_len}, "
            f"step={self.step}, NW={self.nw}, K={self.n_tapers}, "
            f"freq=[{self.freq_range[0]},{self.freq_range[1]}]Hz)"
        )


# ─────────────────────────────────────────────────────────────────────────── #
# Result container                                                             #
# ─────────────────────────────────────────────────────────────────────────── #

@dataclass
class SpectrumResult:
    """Container for multi-taper spectral output.

    Attributes
    ----------
    freqs : (F,) ndarray
        Frequency axis in Hz.
    times : (T,) ndarray
        Centre-time of each window in seconds (``NaN`` for single-segment
        results from :func:`multitaper_psd`).
    power : ndarray
        Power spectral density, shape ``(T, F, C)`` or ``(F, C)`` for
        single-segment.  Units: signal² / Hz.
    coherence : ndarray or None
        Coherence magnitude ``[0, 1]``, shape ``(T, F, C, C)`` or
        ``(F, C, C)``.  Diagonal = 1 (self-coherence).
    phase : ndarray or None
        Cross-channel phase angle in radians, shape same as ``coherence``.
        ``None`` when not computed.
    cross : ndarray or None
        Complex cross-spectrum, shape same as ``coherence``.
        Diagonal = auto-spectrum.
    params : SpectralParams
        The parameters used to compute this result.
    """
    freqs:     np.ndarray
    times:     np.ndarray
    power:     np.ndarray
    params:    SpectralParams
    coherence: Optional[np.ndarray] = None
    phase:     Optional[np.ndarray] = None
    cross:     Optional[np.ndarray] = None

    @property
    def n_times(self) -> int:
        return int(self.power.shape[0]) if self.power.ndim > 2 else 1

    @property
    def n_freqs(self) -> int:
        return len(self.freqs)

    @property
    def n_channels(self) -> int:
        return int(self.power.shape[-1])

    def __repr__(self) -> str:
        return (
            f"SpectrumResult(T={self.n_times}, F={self.n_freqs}, "
            f"C={self.n_channels}, "
            f"freq=[{self.freqs[0]:.1f},{self.freqs[-1]:.1f}]Hz)"
        )


# ─────────────────────────────────────────────────────────────────────────── #
# Internal DPSS taper engine                                                  #
# ─────────────────────────────────────────────────────────────────────────── #

def _get_tapers(win_len: int, nw: float, n_tapers: int) -> np.ndarray:
    """Return DPSS tapers, shape ``(win_len, n_tapers)``."""
    tapers, _ = _dpss(win_len, nw, n_tapers, return_ratios=True)
    # scipy returns (n_tapers, win_len); MATLAB convention is (win_len, n_tapers)
    return tapers.T  # → (win_len, n_tapers)


def _compute_periodogram(
    segment: np.ndarray,          # (win_len, C)
    tapers:  np.ndarray,          # (win_len, K)
    n_fft:   int,
    f_sel:   np.ndarray,          # boolean or index array into rfft output
    detrend: Optional[str],
) -> np.ndarray:
    """Compute tapered periodogram for one window.

    Returns complex array of shape ``(F, K, C)`` where F = len(f_sel),
    K = n_tapers, C = n_channels.  Normalised as in ``mtchglong`` (sqrt(2)).
    """
    T, C = segment.shape

    if detrend == "linear":
        segment = _detrend_linear(segment)
    elif detrend == "constant":
        segment = segment - segment.mean(axis=0, keepdims=True)

    # Taper: (T,K) x (T,C) → (T, K, C) via broadcasting
    # tapers: (T, K) → (T, K, 1); segment: (T, 1, C)
    tapered = tapers[:, :, None] * segment[:, None, :]   # (T, K, C)

    # FFT along time axis (axis=0), full rfft
    fft_out = np.fft.rfft(tapered, n=n_fft, axis=0)      # (nFFT//2+1, K, C)

    # Select frequency bins and normalise
    per = fft_out[f_sel, :, :] * np.sqrt(2.0)            # (F, K, C)
    return per


def _detrend_linear(x: np.ndarray) -> np.ndarray:
    """Remove best-fit linear trend from each column of x (T, C)."""
    T = x.shape[0]
    t = np.arange(T, dtype=np.float64)
    t -= t.mean()
    slope = (t @ x) / (t @ t)
    return x - t[:, None] * slope[None, :]


def _cross_products_block(
    periodogram: np.ndarray,       # (F, K, C, N_chunk)
    n_tapers:    int,
) -> np.ndarray:
    """Taper-averaged cross-spectral matrix for one block.

    Returns complex array ``(F, C, C, N_chunk)`` where
    ``out[f, i, j, n] = mean_k( per[f,k,i,n] * conj(per[f,k,j,n]) )``.
    """
    F, K, C, N = periodogram.shape
    # Vectorised outer product over channels
    # x: (F, K, C, N) → (F, K, C, 1, N)
    # y: (F, K, C, N) → (F, K, 1, C, N)
    x = periodogram[:, :, :, None, :]   # (F, K, C, 1, N)
    y = periodogram[:, :, None, :, :]   # (F, K, 1, C, N)
    # Cross: (F, K, C, C, N)
    cross = (x * y.conj()).mean(axis=1)  # average over tapers → (F, C, C, N)
    return cross                          # (F, C, C, N)


# ─────────────────────────────────────────────────────────────────────────── #
# Core block-chunked computation engine                                       #
# ─────────────────────────────────────────────────────────────────────────── #

def _multitaper_engine(
    x:        np.ndarray,        # (T_signal, C)
    params:   SpectralParams,
    progress: "callable | None" = None,   # progress(n_done, n_total)
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Shared computation kernel for all spectral functions.

    Returns
    -------
    cross : complex ndarray, shape ``(T_windows, F, C, C)``
        Taper-averaged cross-spectral matrix.
    freqs : (F,) ndarray
    times : (T_windows,) ndarray  — window start times in seconds
    """
    if x.ndim == 1:
        x = x[:, None]
    T_sig, C = x.shape

    p        = params
    step     = p.step
    win_len  = p.win_len
    n_fft    = p.n_fft
    sr       = p.samplerate

    # Number of complete windows
    n_windows = max(1, (T_sig - win_len) // step + 1)

    # Frequency selection
    f_full = np.fft.rfftfreq(n_fft, 1.0 / sr)
    f_mask = (f_full >= p.freq_range[0]) & (f_full <= p.freq_range[1])
    f_sel  = np.where(f_mask)[0]
    freqs  = f_full[f_sel]
    F      = len(f_sel)

    # Window start times (begin of each window, seconds)
    times = np.arange(n_windows) * step / sr

    # DPSS tapers
    tapers = _get_tapers(win_len, p.nw, p.n_tapers)   # (win_len, K)
    K      = p.n_tapers

    # Output buffer
    cross_out = np.zeros((n_windows, F, C, C), dtype=np.complex128)

    # Block-chunked processing
    block_size  = p.block_size
    n_blocks    = int(np.ceil(n_windows / block_size))

    for blk in range(n_blocks):
        i0 = blk * block_size
        i1 = min(i0 + block_size, n_windows)
        n_chunk = i1 - i0

        # Collect tapered periodograms for all windows in this block
        # Shape: (F, K, C, n_chunk)
        per_block = np.zeros((F, K, C, n_chunk), dtype=np.complex128)
        for j in range(n_chunk):
            win_idx   = i0 + j
            seg_start = win_idx * step
            seg_end   = seg_start + win_len
            seg       = x[seg_start:seg_end, :]   # (win_len, C)
            # Handle last segment shorter than win_len
            if seg.shape[0] < win_len:
                pad       = np.zeros((win_len - seg.shape[0], C))
                seg       = np.vstack([seg, pad])
            per_block[:, :, :, j] = _compute_periodogram(
                seg, tapers, n_fft, f_sel, p.detrend
            )

        # Cross-spectral matrix for this block: (F, C, C, n_chunk)
        cross_blk = _cross_products_block(per_block, K)
        # cross_blk has shape (F, C, C, n_chunk); reorder → (n_chunk, F, C, C)
        cross_out[i0:i1] = np.moveaxis(cross_blk, -1, 0)
        if progress is not None:
            progress(i1, n_windows)

    return cross_out, freqs, times


# ─────────────────────────────────────────────────────────────────────────── #
# Public spectral functions                                                   #
# ─────────────────────────────────────────────────────────────────────────── #

def multitaper_spectrogram(
    x:        np.ndarray,
    params:   Optional[SpectralParams] = None,
    samplerate: Optional[float] = None,
    progress: "callable | None" = None,
) -> SpectrumResult:
    """Compute a multi-taper power spectrogram.

    Direct port of ``mtcsglong`` (Ken Harris / Anton Sirota).  Iterates
    ``mtchglong`` over each channel independently; here we compute all
    channels in a single vectorised pass.

    Parameters
    ----------
    x:
        Input signal, shape ``(T,)`` or ``(T, C)``.
    params:
        :class:`SpectralParams` instance.  If *None*, a default LFP
        spectrogram is constructed using *samplerate*.
    samplerate:
        Ignored when *params* is provided.  Used to build default params.

    Returns
    -------
    :class:`SpectrumResult` with ``power`` of shape ``(T_windows, F, C)``.
    Units: signal² / Hz (single-sided PSD).
    """
    if params is None:
        params = SpectralParams.for_lfp(samplerate or 1250.0)

    cross, freqs, times = _multitaper_engine(x, params, progress=progress)
    # Diagonal of cross gives auto-spectrum (real; complex part == 0)
    # Shape: (T, F, C, C) → diagonal over last two axes → (T, F, C)
    power_raw = np.einsum("...ii->...i", cross).real   # (T, F, C)

    # Normalise to PSD: divide by sampling rate to get units²/Hz
    power = power_raw / params.samplerate

    return SpectrumResult(freqs=freqs, times=times, power=power, params=params)


def multitaper_coherogram(
    x:          np.ndarray,
    params:     Optional[SpectralParams] = None,
    samplerate: Optional[float] = None,
    return_phase: bool = True,
    progress:   "callable | None" = None,
) -> SpectrumResult:
    """Compute a multi-taper coherogram and phase-ogram.

    Port of ``mtchglong``.  Returns channel-pair coherence and phase
    as a function of time and frequency.

    Parameters
    ----------
    x:
        Input signal, shape ``(T,)`` or ``(T, C)``.
    params:
        :class:`SpectralParams`.
    return_phase:
        If True (default), also compute and return the instantaneous
        phase of each cross-spectrum.

    Returns
    -------
    :class:`SpectrumResult` with:

    ``power``     : ``(T, F, C)``    — auto-spectrum (PSD) per channel
    ``coherence`` : ``(T, F, C, C)`` — magnitude coherence ∈ [0,1]
    ``phase``     : ``(T, F, C, C)`` — cross-phase angle (radians)
    """
    if params is None:
        params = SpectralParams.for_lfp(samplerate or 1250.0)

    cross, freqs, times = _multitaper_engine(x, params, progress=progress)
    # cross: (T, F, C, C) complex

    # Auto-spectra on the diagonal
    auto = np.einsum("...ii->...i", cross).real   # (T, F, C)
    power = auto / params.samplerate

    # Coherence: |S_xy| / sqrt(S_xx * S_yy)
    auto_i = auto[:, :, :, None]    # (T, F, C, 1)
    auto_j = auto[:, :, None, :]    # (T, F, 1, C)
    denom  = np.sqrt(auto_i * auto_j)
    denom  = np.where(denom > 0, denom, np.inf)
    cohere = np.abs(cross) / denom  # (T, F, C, C)

    phase = np.angle(cross) if return_phase else None

    return SpectrumResult(
        freqs=freqs, times=times, power=power,
        coherence=cohere, phase=phase, params=params,
    )


def multitaper_cross_spectrogram(
    x:        np.ndarray,
    params:   Optional[SpectralParams] = None,
    samplerate: Optional[float] = None,
    progress: "callable | None" = None,
) -> SpectrumResult:
    """Compute the complex cross-spectral matrix (not normalised to coherence).

    Port of ``mtcsdglong``.  Returns the raw taper-averaged cross-spectrum
    ``S_xy(t, f)`` without normalisation.  Coherence can be derived as
    ``|S_xy| / sqrt(S_xx * S_yy)``.

    Parameters
    ----------
    x:
        Input signal, shape ``(T,)`` or ``(T, C)``.

    Returns
    -------
    :class:`SpectrumResult` with:

    ``power`` : ``(T, F, C)``    — auto-spectrum (diagonal of cross)
    ``cross`` : ``(T, F, C, C)`` — complex cross-spectrum
    """
    if params is None:
        params = SpectralParams.for_lfp(samplerate or 1250.0)

    cross, freqs, times = _multitaper_engine(x, params, progress=progress)
    auto  = np.einsum("...ii->...i", cross).real
    power = auto / params.samplerate

    return SpectrumResult(
        freqs=freqs, times=times, power=power,
        cross=cross, params=params,
    )


def multitaper_psd(
    x:        np.ndarray,
    params:   Optional[SpectralParams] = None,
    samplerate: Optional[float] = None,
    average:  bool = True,
) -> SpectrumResult:
    """Compute the multi-taper PSD (time-averaged, single-segment result).

    Port of ``mtcsd`` (single call without sliding window).  Treats the
    entire input as one epoch and averages over tapers.  The result has no
    time dimension when ``average=True``.

    Parameters
    ----------
    x:
        Input signal, shape ``(T,)`` or ``(T, C)``.
    average:
        If True (default), returns time-averaged PSD with no T dimension.

    Returns
    -------
    :class:`SpectrumResult` with:

    ``power``     : ``(F, C)`` (averaged) or ``(T, F, C)``
    ``coherence`` : ``(F, C, C)`` or ``(T, F, C, C)``
    ``cross``     : ``(F, C, C)`` or ``(T, F, C, C)``  — complex
    """
    if params is None:
        params = SpectralParams.for_lfp(samplerate or 1250.0)

    cross, freqs, times = _multitaper_engine(x, params)

    if average:
        cross  = cross.mean(axis=0)        # (F, C, C)
        auto   = np.einsum("...ii->...i", cross).real  # (F, C)
        power  = auto / params.samplerate
        auto_i = auto[:, :, None]
        auto_j = auto[:, None, :]
        denom  = np.sqrt(auto_i * auto_j)
        denom  = np.where(denom > 0, denom, np.inf)
        cohere = np.abs(cross) / denom
        phase  = np.angle(cross)
        times  = np.array([np.nan])
    else:
        auto   = np.einsum("...ii->...i", cross).real
        power  = auto / params.samplerate
        auto_i = auto[:, :, :, None]
        auto_j = auto[:, :, None, :]
        denom  = np.sqrt(auto_i * auto_j)
        denom  = np.where(denom > 0, denom, np.inf)
        cohere = np.abs(cross) / denom
        phase  = np.angle(cross)

    return SpectrumResult(
        freqs=freqs, times=times, power=power,
        coherence=cohere, phase=phase, cross=cross,
        params=params,
    )


# ─────────────────────────────────────────────────────────────────────────── #
# AR whitening  (port of WhitenSignal.m)                                     #
# ─────────────────────────────────────────────────────────────────────────── #

def whiten_ar(
    x:           np.ndarray,
    ar_order:    int   = 2,
    common_ar:   bool  = True,
    ar_model:    Optional[np.ndarray] = None,
    window_sec:  Optional[float] = None,
    samplerate:  float = 1250.0,
) -> tuple[np.ndarray, np.ndarray]:
    """AR pre-whitening of a multi-channel signal.

    Port of ``WhitenSignal.m`` (Anton Sirota).  Fits an autoregressive model
    to the data and applies it as a prediction-error filter, flattening the
    power spectrum.  Used in ``fet_spec`` before computing spectrograms to
    reduce spectral leakage.

    Parameters
    ----------
    x:
        Input signal, shape ``(T,)`` or ``(T, C)``.
    ar_order:
        AR model order.  Default 2.
    common_ar:
        If True (default), fit one model to channel 0 and apply it to all
        channels.  Mirrors ``CommonAR=1`` in MATLAB.
    ar_model:
        Pre-computed AR coefficients ``(ar_order + 1,)``.  If provided,
        skip fitting.
    window_sec:
        If set, refit the AR model every ``window_sec`` seconds.
        ``None`` → single fit over the whole signal.
    samplerate:
        Only used when ``window_sec`` is set.

    Returns
    -------
    y : ndarray, same shape as *x*
        Whitened signal.
    ar_coeff : ndarray, shape ``(ar_order + 1,)``
        AR coefficients ``[1, a₁, a₂, …]`` (prediction-error filter
        numerator; same as MATLAB ``arburg`` output).
    """
    from scipy.signal import lfilter

    if x.ndim == 1:
        x = x[:, None]
        squeeze = True
    else:
        squeeze = False

    T, C = x.shape
    y    = np.zeros_like(x)

    # Compute or validate the AR model
    def _fit_ar(seg: np.ndarray) -> np.ndarray:
        """Burg AR estimation on a 1-D segment."""
        n = len(seg)
        p = ar_order
        # Burg method via scipy correlation
        r = np.correlate(seg, seg, mode='full')[n - 1:]   # autocorrelation
        r = r[:p + 1]
        # Yule-Walker via Toeplitz solve
        a = solve_toeplitz(r[:p], -r[1:p + 1])
        return np.concatenate([[1.0], a])                  # [1, a1, a2, ...]

    def _apply_ar(seg: np.ndarray, a: np.ndarray) -> np.ndarray:
        """Apply AR as a forward difference filter: y = filter(a, 1, x)."""
        return lfilter(a, [1.0], seg)

    if ar_model is not None:
        a = np.asarray(ar_model, dtype=np.float64)
    else:
        a = None

    # Determine segment boundaries
    if window_sec is not None:
        win_samp = int(round(window_sec * samplerate))
        n_wins   = int(np.ceil(T / win_samp))
        segs     = [(i * win_samp, min((i + 1) * win_samp, T))
                    for i in range(n_wins)]
    else:
        segs = [(0, T)]

    for (s, e) in segs:
        if a is None or window_sec is not None:
            if common_ar:
                a_seg = _fit_ar(x[s:e, 0])
                a     = a_seg
            else:
                a = None    # will be re-fitted per channel

        for ch in range(C):
            if not common_ar:
                a_ch = _fit_ar(x[s:e, ch])
            else:
                a_ch = a
            y[s:e, ch] = _apply_ar(x[s:e, ch], a_ch)

    ar_final = a if a is not None else np.array([1.0])

    if squeeze:
        y = y[:, 0]

    return y, ar_final


# ─────────────────────────────────────────────────────────────────────────── #
# MTA-compatible feature wrapper  (port of fet_spec.m)                       #
# ─────────────────────────────────────────────────────────────────────────── #

def fet_spec(
    data:        np.ndarray,
    samplerate:  float,
    params:      Optional[SpectralParams] = None,
    mode:        str   = "power",
    whiten:      bool  = True,
    ar_model:    Optional[np.ndarray] = None,
    cross:       bool  = False,
    pad_to:      Optional[int] = None,
) -> tuple["np.ndarray", "SpectrumResult"]:
    """MTA-compatible spectral feature extractor.

    Port of ``fet_spec.m``.  Computes a time-frequency representation of
    *data*, optionally after AR whitening, and returns the result padded
    back to the original signal length.

    Used by the ``label_lfp_states`` pipeline to produce LFP spectrogram
    features for theta/non-theta state detection.

    Parameters
    ----------
    data:
        Input signal, shape ``(T,)`` or ``(T, C)``.  Should already be
        restricted to the trial sync window.
    samplerate:
        Sample rate of *data* in Hz.
    params:
        Spectral parameters.  Defaults to ``SpectralParams.for_lfp(samplerate)``.
    mode:
        ``'power'``     → power spectrogram (``mtcsglong``)
        ``'coherence'`` → coherogram (``mtchglong``)
        ``'cross'``     → complex cross-spectrum (``mtcsdglong``)
    whiten:
        If True (default), apply AR pre-whitening before spectral analysis.
        Mirrors ``wsig=True`` in MATLAB.
    ar_model:
        Pre-computed AR model.  If provided and *whiten* is True, this is
        used instead of fitting a new model.
    cross:
        If True, compute all channel pairs (cross-spectra / coherence).
        If False (default), compute per-channel only.
    pad_to:
        Pad the output time axis back to this many samples at the original
        *samplerate*.  ``None`` → no padding (return windowed result as-is).

    Returns
    -------
    output : ndarray
        Padded spectral feature array with shape ``(pad_to, F, C)`` (or
        ``(pad_to, F, C, C)`` for cross-mode), where the time axis is at
        the original signal sample rate.  Zero-padded at start and end
        to align windows with the original signal.
    result : :class:`SpectrumResult`
        Full spectral result object (not padded) for further analysis.

    Notes
    -----
    The time axis is padded following the same convention as the original
    MATLAB ``fet_spec.m``: each window is centred at ``(window_start +
    win_len / 2)``, and the output is zero-padded at both ends.
    """
    if params is None:
        params = SpectralParams.for_lfp(samplerate)

    if data.ndim == 1:
        data = data[:, None]
    T, C = data.shape

    # 1. AR whitening
    if whiten:
        data_w, _ = whiten_ar(
            data, ar_model=ar_model, samplerate=samplerate
        )
    else:
        data_w  = data
        ar_out  = None

    # Filter out NaN / Inf rows (port of nniz logic)
    valid = np.all(np.isfinite(data_w) & (data_w != 0), axis=1)
    data_clean = np.where(valid[:, None], data_w, 0.0)

    # 2. Compute the requested spectral representation
    if mode == "power":
        result = multitaper_spectrogram(data_clean, params)
        feat   = result.power
    elif mode == "coherence":
        result = multitaper_coherogram(data_clean, params)
        feat   = result.coherence if cross else result.power
    elif mode == "cross":
        result = multitaper_cross_spectrogram(data_clean, params)
        feat   = result.cross if cross else result.power
    else:
        raise ValueError(f"Unknown mode {mode!r}. Use 'power', 'coherence', or 'cross'.")

    # 3. Pad back to original signal length at spectral time resolution
    if pad_to is not None:
        feat_out = _pad_spectral_output(feat, params, T, pad_to, samplerate)
    else:
        feat_out = feat

    return feat_out, result


def _pad_spectral_output(
    feat:        np.ndarray,   # (T_win, F, ...) windowed output
    params:      SpectralParams,
    n_samples:   int,
    pad_to:      int,
    samplerate:  float,
) -> np.ndarray:
    """Pad windowed spectral output to the original signal length.

    Mirrors the padding logic in ``fet_spec.m``::

        ts = ts + (WinLength/2) / Fs    % shift to window centres
        ssr = 1/diff(ts)                % spectral sample rate
        pad = round([ts(1), T/Fs - ts(end)] * ssr)
        output = cat(0, zeros(pad_pre), output, zeros(pad_post))

    Returns zero-padded array of shape ``(pad_to, F, ...)``.
    """
    T_win = feat.shape[0]
    ssr   = 1.0 / params.time_resolution   # spectral samples / sec

    # Window centre times (seconds from start of signal)
    t_centres = (
        np.arange(T_win) * params.step / params.samplerate
        + params.win_len / 2.0 / params.samplerate
    )

    if T_win < 2:
        pad_pre  = 0
        pad_post = pad_to - T_win
    else:
        pad_pre  = max(0, round(t_centres[0]  * ssr) - 1)
        pad_post = max(0, pad_to - T_win - pad_pre)

    rest_shape = feat.shape[1:]
    pre_block  = np.zeros((pad_pre,  *rest_shape), dtype=feat.dtype)
    post_block = np.zeros((pad_post, *rest_shape), dtype=feat.dtype)
    padded     = np.concatenate([pre_block, feat, post_block], axis=0)

    # Final trim / extend to exactly pad_to
    if padded.shape[0] > pad_to:
        padded = padded[:pad_to]
    elif padded.shape[0] < pad_to:
        extra  = np.zeros((pad_to - padded.shape[0], *rest_shape), dtype=feat.dtype)
        padded = np.concatenate([padded, extra], axis=0)

    return padded
