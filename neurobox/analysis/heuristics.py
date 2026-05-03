"""
neurobox.analysis.heuristics
=============================
Threshold-based behavioural state detectors.

These are **fallback baselines** for behavioural state labelling — they
work without a trained classifier and produce reasonable rear / walk /
head-movement / sniff / shake periods using simple signal-processing
heuristics on motion-capture or LFP data.

Use cases:

* Bootstrapping a classifier (round 12's ``bhv_nn``) on a fresh dataset
  before any hand-labels exist.
* Quick-and-dirty state labels for sessions where running the full
  classifier is overkill.
* Sanity-checking classifier output against simple physical
  expectations.

Ports of the portable subset of :file:`MTA/heuristics/`:

* :file:`MTA/heuristics/rear.m`      → :func:`rear`
* :file:`MTA/heuristics/walk.m`      → :func:`walk`
* :file:`MTA/heuristics/head.m`      → :func:`head_movement`
* :file:`MTA/heuristics/walk_ang.m`  → :func:`walk_angle_subdivide`
* :file:`MTA/heuristics/nrhp.m`      → :func:`non_rearing_head_periods`
* :file:`MTA/heuristics/MTAHvel.m`   → :func:`velocity_threshold_periods`
* :file:`MTA/heuristics/sniff.m`     → :func:`sniff_periods`
* :file:`MTA/heuristics/shake.m`     → :func:`shake_periods`
* :file:`MTA/heuristics/theta.m`     → :func:`read_theta_periods`

Not ported (and why)
--------------------
* :file:`groom.m`, :file:`turn.m`, :file:`vel_ssa.m` — exploratory
  scripts with hardcoded session names and mid-edits, not library
  functions.
* :file:`immobile.m` — syntactically broken in the source repo.
* :file:`hsd.m`, :file:`nrhp5.m` — trivial 1-2 line wrappers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from scipy.signal import medfilt

from neurobox.analysis.lfp.filtering import butter_filter
from neurobox.analysis.lfp.oscillations import thresh_cross
from neurobox.dtype.epoch import NBEpoch
from neurobox.dtype.xyz import NBDxyz


__all__ = [
    "rear",
    "walk",
    "head_movement",
    "walk_angle_subdivide",
    "non_rearing_head_periods",
    "velocity_threshold_periods",
    "sniff_periods",
    "shake_periods",
    "read_theta_periods",
]


# ─────────────────────────────────────────────────────────────────────── #
# Helpers                                                                    #
# ─────────────────────────────────────────────────────────────────────── #

def _xy_speed(xyz: NBDxyz, marker: str) -> np.ndarray:
    """XY-plane speed in samples-per-frame for *marker*.

    Mirrors MATLAB's ``xyzvel`` — finite-difference of xy positions.
    Returns shape ``(T,)``.
    """
    midx = xyz.model.index(marker)
    pos  = xyz.data[:, midx, :2]
    diffs = np.diff(pos, axis=0)
    speed = np.linalg.norm(diffs, axis=1)
    # Pad to T
    return np.concatenate([[speed[0] if speed.size else 0.0], speed])


def _com_xy_speed(xyz: NBDxyz, markers: Sequence[str]) -> np.ndarray:
    """Speed of the centre of mass of *markers* in xy."""
    indices = [xyz.model.index(m) for m in markers
               if m in xyz.model.markers]
    if not indices:
        return np.zeros(xyz.data.shape[0])
    com = xyz.data[:, indices, :2].mean(axis=1)         # (T, 2)
    diffs = np.diff(com, axis=0)
    speed = np.linalg.norm(diffs, axis=1)
    return np.concatenate([[speed[0] if speed.size else 0.0], speed])


def _epoch_from_periods(
    periods:       np.ndarray,
    samplerate:    float,
    label:         str,
    key:           str = "",
) -> NBEpoch:
    """Wrap a ``(N, 2)`` periods array as an :class:`NBEpoch`."""
    return NBEpoch(
        data       = periods.astype(np.float64),
        samplerate = samplerate,
        label      = label,
        key        = key,
        mode       = "periods",
    )


# ─────────────────────────────────────────────────────────────────────── #
# rear                                                                       #
# ─────────────────────────────────────────────────────────────────────── #

def rear(
    xyz:                NBDxyz,
    *,
    method:             str = "com",
    rear_threshold:     float = 50.0,
    minimum_interval:   int = 64,
    head_marker:        str = "head_front",
    n_hmm_states:       int = 2,
    min_inter_rear_duration: int = 10,
    rng:                Optional[np.random.Generator | int] = None,
) -> NBEpoch:
    """Detect rearing periods.

    Port of :file:`MTA/heuristics/rear.m`.

    Two methods are supported:

    * ``'com'`` (default) — threshold the **rear feature**
      ``|head_z − spine_lower_z| × pitch(pelvis_root → spine_middle)``
      after a 1 Hz low-pass filter.  Matches MATLAB ``case 'com'``.
    * ``'hmm'`` — fit a 2-state Gaussian HMM to the same feature; the
      higher-mean state is "rearing".  Requires ``hmmlearn`` (install
      with ``pip install 'neurobox[hmm]'``).

    Parameters
    ----------
    xyz:
        :class:`NBDxyz` with at least ``head_front``, ``spine_lower``,
        ``pelvis_root``, ``spine_middle`` markers.  Resampled
        internally to ≤ 120 Hz to match MATLAB.
    method:
        ``'com'`` or ``'hmm'``.
    rear_threshold:
        For ``method='com'`` — feature threshold for rearing.  Default
        50 matches MATLAB.
    minimum_interval:
        Minimum duration (in samples at the working samplerate) of a
        rear period.  Default 64 matches MATLAB.
    head_marker:
        Head marker for the z-difference.  Default ``'head_front'``
        matches MATLAB.
    n_hmm_states:
        For ``method='hmm'``.
    min_inter_rear_duration:
        For ``method='hmm'`` — adjacent rears separated by less than
        this many samples are merged.

    Returns
    -------
    NBEpoch
        Rearing periods.  Samplerate matches the (possibly resampled)
        working rate ≤ 120 Hz.
    """
    # MATLAB: if xyz.sampleRate>120, xyz.resample(120)
    work = xyz
    if xyz.samplerate > 120:
        work = xyz.resample(120.0)
    fs = float(work.samplerate)

    # Build the rear feature
    head_z = work.data[:, work.model.index(head_marker),       2]
    sl_z   = work.data[:, work.model.index("spine_lower"),     2]
    pr_xy  = work.data[:, work.model.index("pelvis_root"),    :3]
    sm_xy  = work.data[:, work.model.index("spine_middle"),   :3]

    # Pitch angle from pelvis_root → spine_middle (atan2 of the
    # vertical component over the horizontal magnitude — MATLAB
    # ``ang(:,'pelvis_root','spine_middle',2)`` is the elevation angle)
    seg = sm_xy - pr_xy                                # (T, 3)
    horiz_mag = np.linalg.norm(seg[:, :2], axis=1)
    pitch = np.arctan2(seg[:, 2], horiz_mag)           # (T,)

    rear_feat = np.abs(head_z - sl_z) * pitch
    rear_feat = np.where(np.isfinite(rear_feat), rear_feat, 0.0)

    if method == "com":
        # MATLAB applies a 1 Hz Butterworth lowpass to the feature
        smoothed = butter_filter(
            rear_feat, cutoff=1.0, samplerate=fs, btype="lowpass", order=3,
        )
        above = (smoothed > rear_threshold).astype(np.float64)
        periods = thresh_cross(above, threshold=0.5,
                                min_interval=minimum_interval)
    elif method == "hmm":
        from neurobox.analysis.stats.gauss_hmm import gauss_hmm
        # gauss_hmm expects a 2D feature array
        states = gauss_hmm(rear_feat[:, None], n_states=n_hmm_states)
        # Identify the rearing state as the one with highest mean feature
        means = []
        for s in range(n_hmm_states):
            mask = states.state_sequence == s
            means.append(rear_feat[mask].mean() if mask.any() else 0.0)
        rear_state_idx = int(np.argmax(means))
        rearing = (states.state_sequence == rear_state_idx).astype(np.float64)
        # Force endpoints to zero (MATLAB lines 28-29)
        if rearing.size > 0:
            rearing[0]  = 0
            rearing[-1] = 0
        periods = thresh_cross(rearing, threshold=0.5,
                                min_interval=minimum_interval)

        # Merge close-adjacent rears (MATLAB lines 31-39)
        if periods.shape[0] > 1 and min_inter_rear_duration > 0:
            inter = periods[1:, 0] - periods[:-1, 1]
            merge_idx = np.flatnonzero(inter < min_inter_rear_duration)
            if merge_idx.size:
                # Build merged periods
                keep = np.ones(periods.shape[0], dtype=bool)
                for mi in merge_idx:
                    periods[mi, 1] = periods[mi + 1, 1]
                    keep[mi + 1] = False
                periods = periods[keep]
    else:
        raise ValueError(f"method must be 'com' or 'hmm'; got {method!r}")

    return _epoch_from_periods(periods, fs, label="rear", key="r")


# ─────────────────────────────────────────────────────────────────────── #
# walk                                                                       #
# ─────────────────────────────────────────────────────────────────────── #

def walk(
    xyz:               NBDxyz,
    *,
    method:            str = "com",
    walk_threshold:    float = 2.0,
    minimum_interval:  int = 40,
    median_window:     int = 32,
    marker:            str = "spine_lower",
) -> NBEpoch:
    """Detect walking periods via velocity thresholding.

    Port of :file:`MTA/heuristics/walk.m`.

    Methods:

    * ``'vel'`` — speed of *marker* (default ``spine_lower``) through
      a 20-sample median filter, threshold ``walk_threshold * fs / 10``.
    * ``'com'`` (default) — speed of the body-marker centre of mass,
      median-filtered with window ``median_window``, threshold
      ``walk_threshold * fs / 10``.
    * ``'head'`` — same algorithm but on the head COM.
    * ``'fet'`` — return the feature trace itself (NBEpoch with one
      "feature" pseudo-period spanning the recording — caller can use
      the ``data`` attribute).

    Parameters
    ----------
    xyz:
        :class:`NBDxyz`.
    method:
        See above.
    walk_threshold:
        Speed threshold.  Default 2.0 matches MATLAB ``'com'`` mode.
        ``'vel'`` mode in MATLAB defaults to 2.5 — pass explicitly.
    minimum_interval:
        Minimum duration of a walk period in samples.  Default 40
        matches MATLAB ``'com'``.
    median_window:
        Median-filter window in samples.  Default 32.
    marker:
        For ``method='vel'`` — single marker to track.

    Returns
    -------
    NBEpoch
        Walking periods (or single feature pseudo-period for
        ``method='fet'``).
    """
    fs = float(xyz.samplerate)
    spine_markers = ["spine_lower", "pelvis_root", "spine_middle",
                     "spine_upper"]
    head_markers  = ["head_back", "head_left", "head_front",
                     "head_right"]

    if method == "vel":
        speed = _xy_speed(xyz, marker) * fs / 10.0
        feat = medfilt(speed, kernel_size=21)
    elif method == "com":
        speed = _com_xy_speed(xyz, spine_markers) * fs / 10.0
        feat  = medfilt(speed, kernel_size=median_window | 1)  # ensure odd
    elif method == "head":
        speed = _com_xy_speed(xyz, head_markers)
        feat  = medfilt(speed, kernel_size=median_window | 1)
    elif method == "fet":
        speed = _com_xy_speed(xyz, spine_markers) * fs / 10.0
        feat  = medfilt(speed, kernel_size=median_window | 1)
        # Return as a feature trace dressed up as a single-period epoch
        return _epoch_from_periods(
            np.array([[0, len(feat)]]),
            fs, label="walk_fet", key="f",
        )
    else:
        raise ValueError(
            f"method must be 'vel', 'com', 'head', or 'fet'; got {method!r}"
        )

    above = (feat > walk_threshold).astype(np.float64)
    periods = thresh_cross(above, threshold=0.5, min_interval=minimum_interval)
    return _epoch_from_periods(periods, fs, label="walk", key="w")


# ─────────────────────────────────────────────────────────────────────── #
# head_movement                                                              #
# ─────────────────────────────────────────────────────────────────────── #

def head_movement(
    xyz:               NBDxyz,
    *,
    method:            str = "com",
    head_threshold:    float = 2.0,
    minimum_interval:  int = 40,
    median_window:     int = 32,
) -> NBEpoch:
    """Detect head-movement periods.

    Port of :file:`MTA/heuristics/head.m`.

    Methods:

    * ``'vel'`` — speed of ``head_front``, median-filtered.
    * ``'com'`` (default) — speed of head-marker COM, median-filtered.

    Parameters
    ----------
    xyz, method, head_threshold, minimum_interval, median_window:
        Same semantics as :func:`walk`.

    Returns
    -------
    NBEpoch
    """
    fs = float(xyz.samplerate)
    head_markers = ["head_back", "head_left", "head_front", "head_right"]

    if method == "vel":
        speed = _xy_speed(xyz, "head_front") * fs / 10.0
        feat  = medfilt(speed, kernel_size=21)
    elif method == "com":
        speed = _com_xy_speed(xyz, head_markers) * fs / 10.0
        feat  = medfilt(speed, kernel_size=median_window | 1)
    else:
        raise ValueError(f"method must be 'vel' or 'com'; got {method!r}")

    above = (feat > head_threshold).astype(np.float64)
    periods = thresh_cross(above, threshold=0.5, min_interval=minimum_interval)
    return _epoch_from_periods(periods, fs, label="head_move", key="m")


# ─────────────────────────────────────────────────────────────────────── #
# walk_angle_subdivide                                                       #
# ─────────────────────────────────────────────────────────────────────── #

def walk_angle_subdivide(
    walk_epoch:    NBEpoch,
    xyz:           NBDxyz,
    *,
    angle_threshold: float = -0.45,
    head_back_marker: str = "head_back",
    head_front_marker: str = "head_front",
    minimum_interval: int = 20,
) -> tuple[NBEpoch, NBEpoch]:
    """Subdivide walking periods by head pitch angle.

    Port of :file:`MTA/heuristics/walk_ang.m` (``'threshold'`` mode).

    Splits *walk_epoch* into "low-angle" (head pitched down) and
    "high-angle" sub-periods, by intersecting the walk mask with the
    head-pitch threshold mask.

    Parameters
    ----------
    walk_epoch:
        Walking-period :class:`NBEpoch`.
    xyz:
        :class:`NBDxyz` with the head markers.
    angle_threshold:
        Pitch threshold in radians.  Periods where the
        ``head_back→head_front`` pitch is below this value are "lang"
        (low-angle); above is "hang" (high-angle).  Default -0.45
        matches MATLAB.
    head_back_marker, head_front_marker:
        Marker names defining the head-pitch axis.

    Returns
    -------
    (lang, hang) : tuple of NBEpoch
        Two state epochs:

        * ``lang`` — low-angle walk
        * ``hang`` — high-angle walk
    """
    hb = xyz.data[:, xyz.model.index(head_back_marker),  :3]
    hf = xyz.data[:, xyz.model.index(head_front_marker), :3]
    seg = hf - hb
    horiz = np.linalg.norm(seg[:, :2], axis=1)
    pitch = np.arctan2(seg[:, 2], horiz)               # (T,)

    fs = float(xyz.samplerate)
    n_samples = xyz.data.shape[0]
    walk_mask = walk_epoch.to_mask(n_samples) if hasattr(walk_epoch, "to_mask") \
                else _periods_to_mask(walk_epoch.data, n_samples)

    low_mask  = (pitch < angle_threshold) & walk_mask
    high_mask = (pitch > angle_threshold) & walk_mask

    low_periods  = _mask_to_periods(low_mask, min_interval=minimum_interval)
    high_periods = _mask_to_periods(high_mask, min_interval=minimum_interval)

    return (
        _epoch_from_periods(low_periods,  fs, label="walk_lang", key="p"),
        _epoch_from_periods(high_periods, fs, label="walk_hang", key="c"),
    )


def _mask_to_periods(mask: np.ndarray, *, min_interval: int = 0) -> np.ndarray:
    """Convert a boolean mask to ``(N, 2)`` start-end periods (right-open)."""
    mask = np.asarray(mask).astype(bool).ravel()
    if mask.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    edges = np.diff(mask.astype(int))
    starts = np.flatnonzero(edges == 1) + 1
    ends   = np.flatnonzero(edges == -1) + 1
    if mask[0]:
        starts = np.r_[0, starts]
    if mask[-1]:
        ends = np.r_[ends, mask.size]
    periods = np.column_stack([starts, ends]).astype(np.float64)
    if min_interval > 0:
        durations = periods[:, 1] - periods[:, 0]
        periods = periods[durations >= min_interval]
    return periods


def _periods_to_mask(periods: np.ndarray, n_samples: int) -> np.ndarray:
    mask = np.zeros(n_samples, dtype=bool)
    for s, e in periods:
        s = int(max(0, s))
        e = int(min(n_samples, e))
        mask[s:e] = True
    return mask


# ─────────────────────────────────────────────────────────────────────── #
# non_rearing_head_periods                                                   #
# ─────────────────────────────────────────────────────────────────────── #

def non_rearing_head_periods(
    rear_epoch:     NBEpoch,
    head_epoch:     NBEpoch,
    n_samples:      int,
    *,
    samplerate:     Optional[float] = None,
    trim_seconds:   float = 5.0,
) -> NBEpoch:
    """Head-movement periods outside of rears (with margin).

    Port of :file:`MTA/heuristics/nrhp.m`.

    Trims each rearing period by ``trim_seconds`` on each side, then
    intersects the **complement** of the trimmed rears with the
    head-movement mask.

    Parameters
    ----------
    rear_epoch, head_epoch:
        :class:`NBEpoch` objects.  Must share the same samplerate.
    n_samples:
        Total number of samples in the recording (needed to compute
        the complement of the rear mask).
    samplerate:
        Override samplerate (otherwise uses ``rear_epoch.samplerate``).
    trim_seconds:
        How much to extend each rear period before excluding it.
        Default 5 matches MATLAB.

    Returns
    -------
    NBEpoch
        Non-rearing head-movement periods.
    """
    fs = float(samplerate if samplerate is not None
                else rear_epoch.samplerate)
    trim = int(round(fs * trim_seconds))

    # Expand rears by trim on each side
    rear_periods = np.asarray(rear_epoch.data, dtype=np.float64)
    if rear_periods.size:
        expanded = np.column_stack([
            np.maximum(0,            rear_periods[:, 0] - trim),
            np.minimum(n_samples,    rear_periods[:, 1] + trim),
        ])
        # Build the rear-extended mask
        rear_mask = np.zeros(n_samples, dtype=bool)
        for s, e in expanded:
            rear_mask[int(s):int(e)] = True
    else:
        rear_mask = np.zeros(n_samples, dtype=bool)

    head_mask = (head_epoch.to_mask(n_samples)
                  if hasattr(head_epoch, "to_mask")
                  else _periods_to_mask(head_epoch.data, n_samples))

    nrhp_mask = head_mask & ~rear_mask
    periods = _mask_to_periods(nrhp_mask)

    return _epoch_from_periods(periods, fs, label="nrhp", key="n")


# ─────────────────────────────────────────────────────────────────────── #
# velocity_threshold_periods (MTAHvel.m)                                     #
# ─────────────────────────────────────────────────────────────────────── #

def velocity_threshold_periods(
    xyz:               NBDxyz,
    *,
    marker:            str = "spine_lower",
    threshold:         float = 2.0,
    minimum_interval_seconds: float = 0.5,
    filter_window_seconds: float = 0.25,
) -> NBEpoch:
    """Periods where xy-velocity of *marker* exceeds *threshold*.

    Port of :file:`MTA/heuristics/MTAHvel.m`.

    Default behaviour matches MATLAB: low-pass filter at 1/0.25 = 4 Hz,
    then threshold the xy-speed at 2 mm/sample.

    Parameters
    ----------
    xyz:
        :class:`NBDxyz`.
    marker:
        Marker name.  MATLAB uses ``Trial.trackingMarker`` which is
        typically ``spine_lower``.
    threshold:
        Speed threshold (in xyz position units per sample).
    minimum_interval_seconds:
        Minimum period duration in seconds.  Default 0.5 matches MATLAB.
    filter_window_seconds:
        Smoothing window in seconds.  Default 0.25 matches MATLAB.

    Returns
    -------
    NBEpoch
        Active-velocity periods.
    """
    fs = float(xyz.samplerate)
    speed = _xy_speed(xyz, marker)
    # Low-pass at 1/filter_window_seconds Hz to mirror MATLAB's
    # gtwin(0.25, fs) Gaussian smooth followed by velocity diff
    cutoff = 1.0 / filter_window_seconds
    smoothed = butter_filter(speed, cutoff=cutoff, samplerate=fs,
                              btype="lowpass", order=3)
    above = (smoothed > threshold).astype(np.float64)
    min_interval = int(round(fs * minimum_interval_seconds))
    periods = thresh_cross(above, threshold=0.5, min_interval=min_interval)
    return _epoch_from_periods(periods, fs, label="vel", key="v")


# ─────────────────────────────────────────────────────────────────────── #
# sniff_periods — spectral detector (8-14 Hz)                                #
# ─────────────────────────────────────────────────────────────────────── #

def sniff_periods(
    xyz:               NBDxyz,
    *,
    sniff_threshold:   float = 10**-3.5,
    n_fft:             int = 1024,
    window_size:       int = 256,
    freq_band:         tuple[float, float] = (8.0, 14.0),
    head_marker:       str = "head_back",
    spine_marker:      str = "spine_upper",
) -> NBEpoch:
    """Detect sniffing periods by spectral power in 8-14 Hz band.

    Port of :file:`MTA/heuristics/sniff.m`.

    The "sniff feature" is the distance between *head_marker* and
    *spine_marker* over time.  Sniffing produces 8-14 Hz oscillations
    in this distance.  This function:

    1. Computes ``|head_marker - spine_marker|`` per frame.
    2. Whitens the signal with an AR pre-emphasis filter.
    3. Computes a multitaper spectrogram.
    4. Sums power in the 8-14 Hz band.
    5. Thresholds at *sniff_threshold*.

    Parameters
    ----------
    xyz:
        :class:`NBDxyz` with *head_marker* and *spine_marker*.
    sniff_threshold:
        Power threshold.  Default ``10^-3.5`` matches MATLAB.
    n_fft, window_size, freq_band:
        Spectrogram parameters.  Defaults match MATLAB.
    head_marker, spine_marker:
        Markers used for the inter-distance signal.

    Returns
    -------
    NBEpoch
        Sniffing periods, in samples at the spectrogram frame rate.

    Notes
    -----
    Output sample rate is the **spectrogram frame rate**, not the
    original *xyz* rate.  Resample with ``epoch.resample(target_fs)``
    if you want to align with another signal.
    """
    from neurobox.analysis.lfp.spectral import (
        SpectralParams, multitaper_spectrogram, whiten_ar,
    )

    fs = float(xyz.samplerate)
    h_idx = xyz.model.index(head_marker)
    s_idx = xyz.model.index(spine_marker)
    distance = np.linalg.norm(
        xyz.data[:, h_idx, :] - xyz.data[:, s_idx, :], axis=1,
    )                                                    # (T,)
    distance = np.where(np.isfinite(distance), distance, 0.0)
    # AR-whitening fails on perfectly-constant input.
    if distance.std() < 1e-9:
        return _epoch_from_periods(
            np.empty((0, 2)), fs, label="sniff", key="s",
        )

    whitened, _ = whiten_ar(distance)

    overlap = max(0, window_size - max(1, window_size // 4))
    params = SpectralParams(
        samplerate    = fs,
        n_fft         = n_fft,
        win_len       = window_size,
        n_overlap     = overlap,
        freq_range    = (0.1, 20.0),
    )
    spec = multitaper_spectrogram(whitened, params)
    # spec.power shape is (T_windows, F, C) for time-varying power.
    # Squeeze the channel axis (input is 1-D) and select the band.
    power = spec.power
    if power.ndim == 3:
        power = power[..., 0]                            # (T_w, F)
    in_band = (spec.freqs >= freq_band[0]) & (spec.freqs <= freq_band[1])
    band_power = power[:, in_band].mean(axis=1)         # (T_w,)

    # Spectrogram frame rate
    if spec.times.size > 1:
        frame_dt = float(spec.times[1] - spec.times[0])
    else:
        frame_dt = 1.0 / fs
    frame_fs = 1.0 / frame_dt

    above = (band_power > sniff_threshold).astype(np.float64)
    periods = thresh_cross(above, threshold=0.5, min_interval=0)
    return _epoch_from_periods(periods, frame_fs, label="sniff", key="s")


# ─────────────────────────────────────────────────────────────────────── #
# shake_periods — spectral detector (14-16 Hz)                                #
# ─────────────────────────────────────────────────────────────────────── #

def shake_periods(
    xyz:               NBDxyz,
    *,
    shake_threshold:   float = 1e-4,
    n_fft:             int = 256,
    window_size:       int = 64,
    freq_band:         tuple[float, float] = (14.0, 16.0),
) -> NBEpoch:
    """Detect head-shake periods by spectral power in 14-16 Hz band.

    Port of :file:`MTA/heuristics/shake.m`.

    The "shake feature" is the angular deflection between the
    spine_lower→pelvis_root and spine_lower→spine_middle azimuths,
    unwrapped to handle 2π discontinuities.  Shakes produce
    14-16 Hz oscillations.

    Parameters
    ----------
    xyz:
        :class:`NBDxyz` with the spine markers.
    shake_threshold:
        Power threshold.  Default ``1e-4`` matches MATLAB.
    n_fft, window_size, freq_band:
        Spectrogram parameters.

    Returns
    -------
    NBEpoch
        Shake periods at spectrogram frame rate.
    """
    from neurobox.analysis.lfp.spectral import (
        SpectralParams, multitaper_spectrogram, whiten_ar,
    )

    fs = float(xyz.samplerate)
    sl = xyz.data[:, xyz.model.index("spine_lower"),   :2]
    pr = xyz.data[:, xyz.model.index("pelvis_root"),   :2]
    sm = xyz.data[:, xyz.model.index("spine_middle"),  :2]

    az_sl_pr = np.arctan2(pr[:, 1] - sl[:, 1], pr[:, 0] - sl[:, 0])
    az_sl_sm = np.arctan2(sm[:, 1] - sl[:, 1], sm[:, 0] - sl[:, 0])
    # Unwrap to remove the cumulative 2π jumps (MATLAB cslpv/cslsm)
    cslpv = np.unwrap(az_sl_pr)
    cslsm = np.unwrap(az_sl_sm)
    shake_signal = cslsm - cslpv
    shake_signal = np.where(np.isfinite(shake_signal), shake_signal, 0.0)
    # AR-whitening fails on perfectly-constant input.  Add a tiny
    # noise floor to keep the autocorrelation matrix non-singular.
    if shake_signal.std() < 1e-9:
        return _epoch_from_periods(
            np.empty((0, 2)), fs, label="shake", key="k",
        )

    whitened, _ = whiten_ar(shake_signal)

    overlap = max(0, window_size - max(1, window_size // 4))
    params = SpectralParams(
        samplerate   = fs,
        n_fft        = n_fft,
        win_len      = window_size,
        n_overlap    = overlap,
        freq_range   = (5.0, 50.0),
    )
    spec = multitaper_spectrogram(whitened, params)
    power = spec.power
    if power.ndim == 3:
        power = power[..., 0]
    in_band = (spec.freqs >= freq_band[0]) & (spec.freqs <= freq_band[1])
    band_power = power[:, in_band].mean(axis=1)

    if spec.times.size > 1:
        frame_fs = 1.0 / float(spec.times[1] - spec.times[0])
    else:
        frame_fs = fs

    above = (band_power > shake_threshold).astype(np.float64)
    periods = thresh_cross(above, threshold=0.5, min_interval=0)
    return _epoch_from_periods(periods, frame_fs, label="shake", key="k")


# ─────────────────────────────────────────────────────────────────────── #
# read_theta_periods                                                         #
# ─────────────────────────────────────────────────────────────────────── #

def read_theta_periods(
    sts_path:    Path | str,
    samplerate:  float,
    *,
    label:       str = "theta",
    key:         str = "t",
) -> NBEpoch:
    """Read theta periods from a CheckEegStates ``.sts.theta`` file.

    Port of :file:`MTA/heuristics/theta.m` (``'sts2epoch'`` mode).

    The ``.sts.theta`` file format is a plain ASCII file with one
    period per line, two whitespace-separated integers (start, end)
    in samples at the LFP samplerate.

    Parameters
    ----------
    sts_path:
        Path to the ``.sts.theta`` file.
    samplerate:
        Samplerate of the entries (typically the LFP sample rate).
    label, key:
        Output epoch label / key.

    Returns
    -------
    NBEpoch
        Theta periods.
    """
    sts_path = Path(sts_path)
    data = np.loadtxt(sts_path, dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] != 2:
        raise ValueError(
            f"expected 2-column theta-period file, got shape {data.shape}"
        )
    return NBEpoch(
        data       = data,
        samplerate = float(samplerate),
        label      = label,
        key        = key,
        mode       = "periods",
    )
