"""
heuristic_labeling.py
=====================
Stage 1 of the behaviour-labelling pipeline: threshold-cascade
detection of ``walk`` and ``rear`` periods from marker trajectories.

Port of :file:`MTA/classifiers/label_behavior_with_heuristics.m`.

The MATLAB original produces three states — ``gper`` (the whole
recording), ``walk``, and ``rear`` — and writes them straight to a
``.stc`` file on disk.  This port returns an
:class:`~neurobox.dtype.NBStateCollection` instead; saving is the
caller's decision.

Algorithm
---------
1. Low-pass the marker positions and derive four angle features from
   the spine chain (``sfet``, ``bfet``, ``afet``, ``hfet``).
2. Slide a window (``window=64``, ``overlap=8``) over the positions
   and over each angle feature, accumulating the per-window mean and
   variance of the within-window displacement.  In MATLAB this is
   five near-identical 20-line blocks; here it is one vectorised
   helper, :func:`windowed_trajectory_stats`, called five times.
3. Run a cascade of ~15 thresholds over the resulting traces to whittle
   candidate walk periods down.  Every threshold is a field of
   :class:`HeuristicThresholds`.
4. Detect rears from head-height × pelvis pitch, and subtract them
   (with asymmetric padding) from the walk periods.

The ``bfet`` defect
-------------------
MATLAB line 138 builds the ``btraj`` window stack from ``hfet``
(2-column, head only) rather than ``bfet`` (5-column, full spine
chain).  ``bfet`` is therefore computed and discarded, and the derived
``bf`` trace is numerically identical to ``hf`` — yet ``bf`` is
load-bearing: it drives the ``body_turn`` threshold that subtracts
turn intervals out of the walk periods.

This is preserved by default, because the surrounding thresholds were
tuned against this behaviour and "fixing" it silently would
invalidate every historical label.  Set
``HeuristicThresholds.reproduce_bfet_defect = False`` to use ``bfet``
as the MATLAB evidently intended, but treat the result as a different
labeller and re-validate against hand-labelled sessions before
trusting it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from neurobox.dtype import NBDxyz, NBDang, NBEpoch, NBStateCollection
from neurobox.analysis.lfp.filtering import butter_filter
from neurobox.analysis.lfp.oscillations import thresh_cross
from neurobox.analysis.lfp.ranges import subtract_ranges
from neurobox.analysis.stats.circular import circ_dist, circ_mean, circ_var


__all__ = [
    "HeuristicThresholds",
    "windowed_trajectory_stats",
    "label_with_heuristics",
]


# ─────────────────────────────────────────────────────────────────────── #
# Thresholds                                                              #
# ─────────────────────────────────────────────────────────────────────── #

@dataclass(frozen=True)
class HeuristicThresholds:
    """Every magic number from the MATLAB cascade, named and defaulted.

    The MATLAB source hardcodes these inline with no provenance; the
    values here are transcribed verbatim.  They were tuned on the
    Sirota-lab rat preparation at ~120 Hz motion capture and are not
    expected to transfer unchanged to another rig.
    """

    # ── windowing ──────────────────────────────────────────────────── #
    window:             int = 64    # winlen  — coarse pass
    overlap:            int = 8     # nOverlap
    fine_window:        int = 16    # nwinlen — second pass
    fine_overlap:       int = 2     # nnOverlap
    position_lowpass_hz: float = 50.0
    filter_order:       int = 3

    # ── walk cascade, in application order ─────────────────────────── #
    walk_feature:       float = 1.6    # wft   — log10 mean(meanD*varD)
    angle_feature:      float = -1.6   # aft   (computed, not used downstream)
    head_feature:       float = 0.003  # hft   (computed, not used downstream)
    walk_distance_1:    float = 1.85   # wdist_thresh, pass 1
    walk_angle:         float = 0.90   # wang_thresh (applied twice)
    body_turn:          float = 0.02   # btdt  — max|bf| per period
    body_turn_mask:     float = 0.02   # baf   — mask threshold
    min_duration:       int   = 32     # wper_dur_thresh (applied twice)
    walk_distance_2:    float = 1.60   # wdist_thresh, pass 2 — lower on purpose
    merge_gap:          int   = 16     # merge periods closer than this
    traj_dot:           float = 0.0    # twdt  — median trajectory alignment
    traj_dot_norm:      float = 10.0   # ntwdt — normalised by duration
    walk_distance_3:    float = 1.80   # endpoint displacement, final pass
    fine_walk_feature:  float = 1.49   # wf_mov_thresh

    # ── rear detection ─────────────────────────────────────────────── #
    rear_feature:       float = 45.0   # rearThresh
    rear_lowpass_hz:    float = 1.0
    rear_min_interval:  int   = 64     # minimum_interval
    rear_height:        float = 170.0  # rper_height_thresh
    rear_pad_before:    int   = 64     # subtracted from walk as [-64, +24]
    rear_pad_after:     int   = 24

    # ── general-period trim ────────────────────────────────────────── #
    gper_trim:          int   = 5      # aper + [5, -5]

    # ── fidelity switch ────────────────────────────────────────────── #
    reproduce_bfet_defect: bool = True
    """MATLAB builds ``btraj`` from ``hfet`` instead of ``bfet`` — see
    the module docstring.  *True* reproduces it; *False* uses ``bfet``.
    """


# ─────────────────────────────────────────────────────────────────────── #
# Windowed trajectory statistics                                          #
# ─────────────────────────────────────────────────────────────────────── #

def windowed_trajectory_stats(
    x:        np.ndarray,
    window:   int,
    overlap:  int,
    *,
    circular: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-window mean and variance of the within-window displacement.

    Replaces five near-identical loop blocks in the MATLAB original.
    For each of *overlap* phase offsets, the signal is reshaped into
    consecutive *window*-length chunks, each chunk is expressed
    relative to its own first sample, and the mean and variance across
    the window are taken.

    Parameters
    ----------
    x:
        Shape ``(T, C)``.  ``T`` must be a multiple of *window*.
    window, overlap:
        Window length in samples, and the number of evenly-spaced
        phase offsets (``overlap=8`` means a new window every
        ``window/8`` samples).
    circular:
        When *True* use circular mean/variance — required for the
        angle-derived features, matching MATLAB's ``circ_mean`` /
        ``circ_var``.

    Returns
    -------
    (mean, var):
        Both shape ``(T // window * overlap, C)``.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    T, C = x.shape
    n_chunk = T // window
    n_out = n_chunk * overlap
    mean = np.zeros((n_out, C))
    var = np.zeros((n_out, C))
    step = window // overlap

    for i in range(overlap):
        rolled = np.roll(x, -i * step, axis=0)
        # (window, n_chunk, C) — MATLAB reshape is column-major
        chunks = rolled[: n_chunk * window].reshape(n_chunk, window, C)
        chunks = chunks.transpose(1, 0, 2)
        rel = chunks - chunks[0:1, :, :]      # relative to window start
        if circular:
            m = circ_mean(rel, axis=0)
            v = np.stack(
                [[circ_var(rel[:, k, c]) for c in range(C)]
                 for k in range(n_chunk)])
        else:
            m = rel.mean(axis=0)
            v = rel.var(axis=0, ddof=1)
        mean[i::overlap, :] = m
        var[i::overlap, :] = v
    return mean, var


def _magnitude_pairs(a: np.ndarray, n_markers: int) -> np.ndarray:
    """Collapse interleaved ``(x, y)`` columns to per-marker magnitude.

    MATLAB: ``sqrt(sum(reshape(v, [], nMarkers, 2).^2, 3))``.
    """
    return np.sqrt((a.reshape(a.shape[0], n_markers, 2) ** 2).sum(axis=2))


def _lowpass(x: np.ndarray, cutoff: float, fs: float, order: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        return butter_filter(x, cutoff, fs, order=order, btype="lowpass")
    flat = x.reshape(x.shape[0], -1)
    out = np.empty_like(flat)
    for c in range(flat.shape[1]):
        out[:, c] = butter_filter(
            flat[:, c], cutoff, fs, order=order, btype="lowpass")
    return out.reshape(x.shape)


def _gauss_smooth(x: np.ndarray, n: int) -> np.ndarray:
    """MATLAB ``Filter0(gausswin(n)/sum(gausswin(n)), x)``."""
    a = 2.5
    k = np.arange(n) - (n - 1) / 2.0
    w = np.exp(-0.5 * (a * k / ((n - 1) / 2.0)) ** 2)
    w /= w.sum()
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        return np.convolve(x, w, mode="same")
    return np.column_stack(
        [np.convolve(x[:, c], w, mode="same") for c in range(x.shape[1])])


def _merge_close(periods: np.ndarray, gap: int) -> np.ndarray:
    """Merge periods separated by fewer than *gap* samples."""
    if periods.size == 0:
        return periods
    out = [periods[0].tolist()]
    for start, stop in periods[1:]:
        if start - out[-1][1] < gap:
            out[-1][1] = stop
        else:
            out.append([start, stop])
    return np.asarray(out, dtype=periods.dtype)


def _period_reduce(values: np.ndarray, periods: np.ndarray, fn) -> np.ndarray:
    """Apply *fn* to ``values[a:b]`` for each ``[a, b]`` period."""
    if periods.size == 0:
        return np.zeros(0)
    n = values.shape[0]
    out = np.zeros(periods.shape[0])
    for i, (a, b) in enumerate(periods):
        a = int(np.clip(a, 0, n - 1))
        b = int(np.clip(b, a + 1, n))
        out[i] = fn(values[a:b])
    return out


# ─────────────────────────────────────────────────────────────────────── #
# Main entry point                                                        #
# ─────────────────────────────────────────────────────────────────────── #

def label_with_heuristics(
    xyz:        NBDxyz,
    *,
    thresholds: HeuristicThresholds | None = None,
    mode:       str = "auto_wbhr",
) -> NBStateCollection:
    """Detect ``gper`` / ``walk`` / ``rear`` by threshold cascade.

    Port of :file:`MTA/classifiers/label_behavior_with_heuristics.m`.

    Parameters
    ----------
    xyz:
        Marker positions.  Requires the spine chain
        (``spine_lower``, ``pelvis_root``, ``spine_middle``,
        ``spine_upper``) plus ``head_back`` and ``head_front``.
    thresholds:
        :class:`HeuristicThresholds`; defaults transcribe MATLAB.
    mode:
        StateCollection mode tag.  MATLAB uses ``'auto_wbhr'``.

    Returns
    -------
    NBStateCollection
        With states ``gper`` (key ``a``), ``walk`` (``w``) and
        ``rear`` (``r``).  Period data is in **seconds** (the NBEpoch
        contract); the epochs carry ``samplerate=xyz.samplerate`` as
        provenance metadata only.

    Notes
    -----
    Unlike the MATLAB original this function has no side effects — it
    neither writes ``.stc`` files nor mutates a session object.  Call
    :meth:`NBStateCollection.save` yourself if you want it persisted.
    """
    th = thresholds or HeuristicThresholds()

    required = ("spine_lower", "pelvis_root", "spine_middle",
                "spine_upper", "head_back", "head_front")
    missing = [m for m in required if m not in xyz.markers]
    if missing:
        raise KeyError(
            f"label_with_heuristics requires markers {missing}, "
            f"absent from xyz (has {list(xyz.markers)})")

    fs = float(xyz.samplerate)
    ang_full = NBDang.from_xyz(xyz)

    # Low-pass the positions (MATLAB: ButFilter 3rd order, 50 Hz)
    fxyz = xyz.copy()
    fxyz.data = _lowpass(
        xyz.data.astype(np.float64), th.position_lowpass_hz, fs,
        th.filter_order)

    # Trim to a whole number of windows
    T = (fxyz.data.shape[0] // th.window) * th.window
    if T < th.window * 2:
        raise ValueError(
            f"xyz is too short: {fxyz.data.shape[0]} samples cannot fill "
            f"two windows of {th.window}")

    traj_fs = (fs / th.window) * th.overlap

    body = ["spine_lower", "pelvis_root", "spine_middle", "head_back"]
    pos = fxyz.sel(body, [0, 1])[:T]                      # (T, 4, 2)
    pos_flat = pos.reshape(T, -1)                          # (T, 8)

    yaw = lambda a, b: ang_full.between(a, b, "theta")[:T]

    # sfet — successive yaw differences along the spine chain
    sfet = np.column_stack([
        circ_dist(yaw("pelvis_root",  "spine_middle"),
                  yaw("spine_lower",  "pelvis_root")),
        circ_dist(yaw("spine_middle", "spine_upper"),
                  yaw("pelvis_root",  "spine_middle")),
        circ_dist(yaw("spine_upper",  "head_back"),
                  yaw("spine_middle", "spine_upper")),
        circ_dist(yaw("head_back",    "head_front"),
                  yaw("spine_upper",  "head_back")),
    ])
    # bfet — full spine chain yaw (see module docstring: MATLAB discards it)
    bfet = np.column_stack([
        yaw("spine_lower",  "pelvis_root"),
        yaw("pelvis_root",  "spine_middle"),
        yaw("spine_middle", "spine_upper"),
        yaw("spine_upper",  "head_back"),
        yaw("head_back",    "head_front"),
    ])
    afet = bfet[:, :3]
    hfet = bfet[:, [2, 4]]

    # ── Windowed statistics ─────────────────────────────────────────── #
    v_mean, v_var = windowed_trajectory_stats(pos_flat, th.window, th.overlap)
    v_mean_d = _magnitude_pairs(v_mean, len(body))
    v_var_d = _magnitude_pairs(v_var, len(body))

    s_mean, s_var = windowed_trajectory_stats(
        sfet, th.window, th.overlap, circular=True)
    a_mean, a_var = windowed_trajectory_stats(
        afet, th.window, th.overlap, circular=True)
    h_mean, h_var = windowed_trajectory_stats(
        hfet, th.window, th.overlap, circular=True)
    b_src = hfet if th.reproduce_bfet_defect else bfet
    b_mean, b_var = windowed_trajectory_stats(
        b_src, th.window, th.overlap, circular=True)

    # ── Derived traces ──────────────────────────────────────────────── #
    vmv = v_mean_d * v_var_d
    with np.errstate(divide="ignore", invalid="ignore"):
        wf = np.nanmean(np.log10(vmv[:, :2]), axis=1)
    wf = np.nan_to_num(wf, nan=-np.inf)

    a_var_d = np.sqrt((a_var ** 2).sum(axis=1))
    af_raw = circ_mean(a_mean, axis=1) * a_var_d
    with np.errstate(divide="ignore", invalid="ignore"):
        af = np.log10(np.abs(
            _lowpass(af_raw, 3.0, traj_fs, th.filter_order)))

    s_var_d = np.sqrt((s_var ** 2).sum(axis=1))
    sf = _gauss_smooth(circ_mean(s_mean, axis=1) * s_var_d.mean(axis=0), 21) \
        if s_var_d.ndim > 1 else _gauss_smooth(
            circ_mean(s_mean, axis=1) * s_var_d, 21)

    h_var_d = np.sqrt((h_var ** 2).sum(axis=1))
    hf = _gauss_smooth(circ_mean(h_mean, axis=1) * h_var_d, 21)

    b_var_d = np.sqrt((b_var ** 2).sum(axis=1))
    bf = _gauss_smooth(circ_mean(b_mean, axis=1) * b_var_d, 21)

    # ── Trajectory-alignment dot products ──────────────────────────── #
    # MATLAB lines 170-176.  Note two things that are easy to get
    # backwards:
    #   * the "normalisation" divides by the SQUARED norm (v/|v|²),
    #     not by |v| — so these are not unit vectors;
    #   * ``dvtm`` uses the two NORMALISED vectors while ``ndvtm``
    #     uses the two RAW ones, despite the ``n`` prefix suggesting
    #     the opposite.  ``dvtm`` is therefore small and ``ndvtm``
    #     unbounded, which is why their thresholds are 0 and 10.
    idx = np.round(np.linspace(th.window - 1, T - 1, v_mean.shape[0])
                   ).astype(int)
    dtraj = pos[idx] - pos[idx][:, 0:1, :]                 # (n, 4, 2) raw
    vtraj_mean = v_mean.reshape(-1, len(body), 2)          # (n, 4, 2) raw

    def _sqnorm_normalise(v: np.ndarray) -> np.ndarray:
        sq = (v ** 2).sum(axis=2, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.where(sq > 0, v / sq, 0.0)
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    dvtm = (_sqnorm_normalise(vtraj_mean)
            * _sqnorm_normalise(dtraj)).sum(axis=2)        # both normalised
    ndvtm = (dtraj * vtraj_mean).sum(axis=2)               # both raw

    # ── Walk cascade ────────────────────────────────────────────────── #
    per = thresh_cross((wf > th.walk_feature).astype(float), 0.5, 5)
    per = np.round(per * fs / traj_fs).astype(int)

    fpos = _gauss_smooth(pos[:, 0, :], 65)

    def _path_length(p):
        return np.log10(max(
            np.sqrt((np.diff(p, axis=0) ** 2).sum(axis=1)).sum(), 1e-12))

    if per.size:
        score = _period_reduce(fpos, per, _path_length)
        per = per[score > th.walk_distance_1]

    if per.size:
        score = np.abs(_period_reduce(
            sfet, per, lambda s: s.sum() / max(len(s), 1)))
        per = per[score < th.walk_angle]

    # body-turn subtraction (this is where the bfet defect bites)
    if per.size:
        tper = np.round(per / fs * traj_fs).astype(int)
        btpa = _period_reduce(bf, tper, lambda v: np.max(np.abs(v)))
        tper = tper[btpa > th.body_turn]
        baf = np.zeros_like(bf)
        for a, b in tper:
            a, b = int(np.clip(a, 0, len(bf) - 1)), int(np.clip(b, 0, len(bf)))
            baf[a:b] = np.abs(bf[a:b])
        bwn = thresh_cross((baf > th.body_turn_mask).astype(float), 0.5, 1)
        if bwn.size:
            bwn = np.round(bwn / traj_fs * fs).astype(int)
            per = subtract_ranges(per, bwn)

    if per.size:
        per = per[np.diff(per, axis=1).ravel() > th.min_duration]
    if per.size:
        score = _period_reduce(fpos, per, _path_length)
        per = per[score > th.walk_distance_2]
    if per.size:
        per = _merge_close(per, th.merge_gap)
        per = per[np.diff(per, axis=1).ravel() > th.min_duration]

    if per.size:
        tper = np.round(per / fs * traj_fs).astype(int)
        twpa = _period_reduce(
            dvtm[:, 1:3].sum(axis=1), tper, np.median)
        per = per[twpa > th.traj_dot]

    if per.size:
        tper = np.round(per / fs * traj_fs).astype(int)
        # MATLAB divides the median by the period duration in
        # trajectory samples, then compares against 10.
        ntwpa = _period_reduce(
            ndvtm[:, 1:3].sum(axis=1), tper,
            lambda v: np.median(v) / max(len(v), 1))
        per = per[ntwpa > th.traj_dot_norm]

    # Rasterise the surviving periods, knock out any trajectory sample
    # whose mean alignment (over all markers but the first) is
    # negative, then re-extract periods.  MATLAB lines 294-299.
    if per.size:
        vdplts = np.zeros(dtraj.shape[0])
        for a, b in np.round(per / fs * traj_fs).astype(int):
            a = int(np.clip(a, 0, len(vdplts) - 1))
            b = int(np.clip(b, a + 1, len(vdplts)))
            vdplts[a:b] = 1.0
        vdplts[ndvtm[:, 1:].mean(axis=1) < 0] = 0.0
        per = thresh_cross(vdplts, 0.5, 2)
        if per.size:
            per = np.round(per / traj_fs * fs).astype(int)

    if per.size:
        score = _period_reduce(
            fpos, per,
            lambda p: np.log10(max(
                np.linalg.norm(p[-1] - p[0]), 1e-12)))
        per = per[score > th.walk_distance_3]

    if per.size:
        score = np.abs(_period_reduce(
            sfet, per, lambda s: s.sum() / max(len(s), 1)))
        per = per[score < th.walk_angle]

    # ── Fine-scale confirmation pass ────────────────────────────────── #
    if per.size:
        f_mean, f_var = windowed_trajectory_stats(
            pos_flat, th.fine_window, th.fine_overlap)
        fmv = (_magnitude_pairs(f_mean, len(body))
               * _magnitude_pairs(f_var, len(body)))
        with np.errstate(divide="ignore", invalid="ignore"):
            fwf = _gauss_smooth(
                np.nanmean(np.log10(fmv[:, :2]), axis=1), 11)
        fwf = np.nan_to_num(fwf, nan=-np.inf)
        fine_fs = (fs / th.fine_window) * th.fine_overlap
        mov = np.zeros_like(fwf)
        for a, b in np.round(per / fs * fine_fs).astype(int):
            a, b = int(np.clip(a, 0, len(mov) - 1)), int(np.clip(b, 0, len(mov)))
            mov[a:b] = 1.0
        mov[fwf < th.fine_walk_feature] = 0.0
        per = thresh_cross(mov, 0.5, 5)
        if per.size:
            per = np.round(per / fine_fs * fs).astype(int)

    # ── Rear detection ──────────────────────────────────────────────── #
    z_head = xyz.sel(["head_front"], [2]).ravel()[:T]
    z_low = xyz.sel(["spine_lower"], [2]).ravel()[:T]
    pitch_pm = ang_full.between("pelvis_root", "spine_middle", "phi")[:T]
    rear_trace = np.abs(z_head - z_low) * pitch_pm
    rear_trace = np.nan_to_num(rear_trace, nan=0.0)
    rear_trace = _lowpass(rear_trace, th.rear_lowpass_hz, fs, th.filter_order)
    rper = thresh_cross(
        (rear_trace > th.rear_feature).astype(float), 0.5, th.rear_min_interval)
    if rper.size:
        peak = _period_reduce(z_head, rper, np.max)
        rper = rper[peak > th.rear_height]
    if rper.size and per.size:
        pad = rper + np.array([-th.rear_pad_before, th.rear_pad_after])
        per = subtract_ranges(per, np.clip(pad, 0, T))

    # ── Assemble ────────────────────────────────────────────────────── #
    # NBEpoch 'periods' data is in SECONDS by contract (its resample()
    # is metadata-only for periods mode, and to_mask() multiplies by
    # samplerate).  The internal cascade works in sample indices at
    # ``fs``; convert on the way out.  An earlier revision stored the
    # raw indices here, which silently broke every seconds-assuming
    # consumer downstream (stc2mat masks scaled by fs², bootstrap rows
    # drawn from the wrong frames).
    def _sec(idx_periods: np.ndarray) -> np.ndarray:
        if idx_periods.size == 0:
            return np.zeros((0, 2), dtype=np.float64)
        return (np.clip(np.atleast_2d(idx_periods), 0, T)
                .astype(np.float64) / fs)

    stc = NBStateCollection(mode=mode)
    gper = np.array([[th.gper_trim, max(T - th.gper_trim, th.gper_trim + 1)]],
                    dtype=np.int64)
    stc.add_state(NBEpoch(_sec(gper), samplerate=fs,
                          label="gper", key="a"))
    stc.add_state(NBEpoch(_sec(per if per.size else np.zeros((0, 2))),
                          samplerate=fs, label="walk", key="w"))
    stc.add_state(NBEpoch(_sec(rper if rper.size else np.zeros((0, 2))),
                          samplerate=fs, label="rear", key="r"))
    return stc
