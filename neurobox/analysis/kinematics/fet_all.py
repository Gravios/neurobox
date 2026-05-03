"""
neurobox.analysis.kinematics.fet_all
======================================
The 59-column "kitchen-sink" feature set used by the lab's behaviour
classifier pipeline.

Port of:

* :file:`MTA/features/fet_all.m`     → :func:`fet_all_features`
* :file:`MTA/features/gen_fet_lsppc.m` → :func:`lower_spine_yaw_ppc`

What it computes
----------------
59 features per frame, designed to be exhaustive enough that any
behavioural state can be discriminated from any other:

* 12 cols: filtered marker-pair pitches (elevation angles)
* 12 cols: pitch derivatives (raw → Butterworth → diff)
* 14 cols: yaw → angular-velocity-squared
* 1 col : phase-pair coherence (PPC) of marker XY trajectories
* 5 cols: filtered z-positions of spine markers
* 6 cols: log10 xy speed of selected markers
* 6 cols: log10 z speed of selected markers
* 1 col : body-axis projected acceleration ("BFET")
* 1 col : spine straightness (sum-of-segments / endpoint-distance)
* 1 col : angular variance across spine

Total = 59 columns.  The MATLAB original was the input to the
hierarchical-MI feature selector (:func:`select_features_hmi`).

This implementation matches the MATLAB column ordering exactly so
that the per-state ``stateOrd``/``fetInds`` arrays produced by the
selector can be applied to either MATLAB-generated or Python-generated
features.

Differences from MATLAB
-----------------------
* Caller is responsible for pre-processing *xyz* (e.g. via
  :func:`preproc_xyz_spline_spine_head_eqd`) before passing it in.
* The :func:`map_to_reference_session` step (lines 22-30 of
  :file:`fet_all.m`) is now caller's responsibility; this function
  expects already-aligned input.
* The :func:`gen_fet_lsppc` MAT-file caching is replaced by an
  in-memory return.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.signal import butter, filtfilt

from neurobox.analysis.lfp.filtering import butter_filter
from neurobox.analysis.stats.circular import circ_dist, circ_mean, ppc
from neurobox.dtype.ang import NBDang
from neurobox.dtype.xyz import NBDxyz


__all__ = [
    "FetAllResult",
    "fet_all_features",
    "lower_spine_yaw_ppc",
]


# ─────────────────────────────────────────────────────────────────────── #
# Result                                                                     #
# ─────────────────────────────────────────────────────────────────────── #

@dataclass
class FetAllResult:
    """Output of :func:`fet_all_features`.

    Attributes
    ----------
    data : np.ndarray, shape ``(T, 59)``
        Feature matrix at the requested *samplerate*.
    column_names : tuple[str, ...]
        Per-column descriptions (length 59).
    samplerate : float
        Output sample rate.
    """
    data:         np.ndarray
    column_names: tuple[str, ...]
    samplerate:   float


# ─────────────────────────────────────────────────────────────────────── #
# lower_spine_yaw_ppc — port of gen_fet_lsppc.m                              #
# ─────────────────────────────────────────────────────────────────────── #

def lower_spine_yaw_ppc(
    xyz:       NBDxyz,
    *,
    shift:     int = 5,
    markers:   tuple[str, ...] = (
        "spine_lower", "pelvis_root", "spine_middle", "spine_upper",
        "head_back", "head_front",
    ),
) -> np.ndarray:
    """Phase-pair coherence of yaw across spine + head markers.

    Port of :file:`MTA/features/gen_fet_lsppc.m`.

    For each frame *t*:

    1. Compute the XY displacement vector of every marker between
       *t-shift* and *t+shift* (a centred finite-difference).
    2. Convert each marker's vector to its yaw angle (atan2).
    3. Apply :func:`neurobox.analysis.stats.circular.ppc` across the
       markers listed in *markers* — a high PPC value means all
       markers are heading in nearly the same direction.

    Parameters
    ----------
    xyz:
        :class:`NBDxyz`.
    shift:
        Half-width of the centred finite-difference, in samples.
        Default 5 matches MATLAB.
    markers:
        Subset of markers contributing to the PPC.  Markers not
        present are skipped.

    Returns
    -------
    np.ndarray, shape ``(T,)``
        Phase-pair coherence per frame.
    """
    T, N, _ = xyz.data.shape
    fwd = np.roll(xyz.data[:, :, :2], -shift, axis=0)
    bwd = np.roll(xyz.data[:, :, :2],  shift, axis=0)
    diff = fwd - bwd                              # (T, N, 2)
    yaw = np.arctan2(diff[:, :, 1], diff[:, :, 0])  # (T, N)

    # Subset to the requested markers (those present)
    indices = [xyz.model.index(m) for m in markers
               if m in xyz.model.markers]
    if not indices:
        return np.zeros(T, dtype=np.float64)
    yaw_subset = yaw[:, indices]                  # (T, n_markers)

    out = np.zeros(T, dtype=np.float64)
    for t in range(T):
        out[t] = ppc(yaw_subset[t])
    return out


# ─────────────────────────────────────────────────────────────────────── #
# fet_all_features — port of fet_all.m                                       #
# ─────────────────────────────────────────────────────────────────────── #

def _resample_array(arr: np.ndarray, src_sr: float, tgt_sr: float
                     ) -> np.ndarray:
    """Linear-interp resample a (T, ...) array along axis 0."""
    if abs(src_sr - tgt_sr) < 1e-9:
        return arr.copy()
    T = arr.shape[0]
    duration = T / src_sr
    n_tgt = int(round(duration * tgt_sr))
    t_src = np.arange(T) / src_sr
    t_tgt = np.arange(n_tgt) / tgt_sr
    if arr.ndim == 1:
        return np.interp(t_tgt, t_src, arr)
    out_shape = (n_tgt,) + arr.shape[1:]
    out = np.empty(out_shape, dtype=arr.dtype)
    flat = arr.reshape(T, -1)
    for j in range(flat.shape[1]):
        out.reshape(n_tgt, -1)[:, j] = np.interp(t_tgt, t_src, flat[:, j])
    return out


def _butter_lp(arr: np.ndarray, cutoff: float, fs: float, order: int = 3
                ) -> np.ndarray:
    """Per-column Butterworth low-pass.

    For 1-D input, just call :func:`butter_filter`.  For 2-D (T, M)
    input, filter every column independently.
    """
    if arr.ndim == 1:
        return butter_filter(arr, cutoff=cutoff, samplerate=fs,
                              btype="lowpass", order=order)
    out = np.empty_like(arr)
    for c in range(arr.shape[1]):
        out[:, c] = butter_filter(arr[:, c].astype(np.float64),
                                    cutoff=cutoff, samplerate=fs,
                                    btype="lowpass", order=order)
    return out


def fet_all_features(
    xyz:           NBDxyz,
    *,
    samplerate:    float = 20.0,
    pitch_filter_hz: float = 5.0,
    speed_filter_hz: float = 2.5,
    angle_window_samples: Optional[int] = None,
) -> FetAllResult:
    """Compute the canonical 59-column kitchen-sink feature set.

    Port of :file:`MTA/features/fet_all.m`.

    Parameters
    ----------
    xyz:
        Pre-processed :class:`NBDxyz`.  Should have spine markers,
        head markers, and (after :func:`augment_xyz`) bcom / hcom /
        acom derived markers.
    samplerate:
        Output sample rate.  MATLAB default 20 Hz.
    pitch_filter_hz:
        Low-pass cutoff for pitch derivatives.  Default 5 Hz matches
        MATLAB.
    speed_filter_hz:
        Low-pass cutoff for marker speeds.  Default 2.5 Hz matches
        MATLAB.
    angle_window_samples:
        Window length for the yaw → angular-velocity-squared
        accumulation.  Default ``samplerate`` matches MATLAB
        ``newSampleRate`` semantics.

    Returns
    -------
    FetAllResult
        ``data`` shape ``(T_out, 59)``.
    """
    if angle_window_samples is None:
        angle_window_samples = int(round(samplerate))

    src_sr = float(xyz.samplerate)

    # Pre-build a copy of xyz at the source rate but ButFiltered for
    # the "filtered xyz" branch (fxyz in MATLAB).
    fxyz_data = np.empty_like(xyz.data)
    for m in range(xyz.data.shape[1]):
        for d in range(3):
            fxyz_data[:, m, d] = butter_filter(
                xyz.data[:, m, d], cutoff=2.5, samplerate=src_sr,
                btype="lowpass", order=3,
            )
    fxyz = NBDxyz(fxyz_data, model=xyz.model, samplerate=src_sr,
                   name=xyz.name + "_fxyz")

    # Build NBDang objects on both raw and filtered xyz
    ang  = NBDang.from_xyz(xyz)
    fang = NBDang.from_xyz(fxyz)

    # Helper: get an inter-marker angle component
    def A(angobj: NBDang, mi: str, mj: str, comp: str) -> np.ndarray:
        if mi not in xyz.model.markers or mj not in xyz.model.markers:
            return np.zeros(xyz.data.shape[0])
        return angobj.between(mi, mj, component=comp)

    # ── Cols 1-12: filtered pitches between marker pairs ──────────── #
    # Mirrors MATLAB lines 72-83
    pitch_pairs = [
        ("spine_lower", "pelvis_root"),
        ("spine_lower", "spine_middle"),
        ("spine_lower", "bcom"),
        ("spine_lower", "hcom"),
        ("pelvis_root", "spine_middle"),
        ("pelvis_root", "spine_upper"),
        ("spine_middle", "spine_upper"),
        ("spine_middle", "hcom"),
        ("spine_upper", "hcom"),
        ("spine_upper", "hcom"),     # MATLAB line 81 duplicates this
        ("bcom", "hcom"),
        ("bcom", "acom"),
    ]
    cols_1_12 = np.column_stack([A(fang, mi, mj, "phi")
                                   for mi, mj in pitch_pairs])

    # ── Cols 13-24: pitch derivatives ─────────────────────────────────#
    # MATLAB lines 89-104:
    # raw pitches, butter-LP @ 5 Hz, then diff
    raw_pitch_pairs = [
        ("spine_lower", "pelvis_root"),
        ("spine_lower", "spine_middle"),
        ("spine_lower", "bcom"),
        ("spine_lower", "hcom"),
        ("pelvis_root", "spine_middle"),
        ("pelvis_root", "spine_upper"),
        ("spine_middle", "spine_upper"),
        ("spine_middle", "hcom"),
        ("spine_upper", "hcom"),
        ("bcom", "hcom"),
        ("bcom", "acom"),
        ("hcom", "acom"),
    ]
    raw_pitches = np.column_stack([A(ang, mi, mj, "phi")
                                     for mi, mj in raw_pitch_pairs])
    raw_pitches_lp = _butter_lp(raw_pitches, pitch_filter_hz, src_sr)
    cols_13_24_raw = np.diff(raw_pitches_lp, axis=0)
    # Pad to original length
    cols_13_24 = np.vstack([cols_13_24_raw, cols_13_24_raw[-1:]])

    # ── Cols 25-38: yaw → angular velocity squared ────────────────────#
    # MATLAB lines 110-133.  14 yaw pairs (note line 119/120 duplicate
    # spine_upper-hcom).  Filter @ 5 Hz, then circ_dist with 1-shift,
    # window-sum the squares, log10.
    yaw_pairs = [
        ("spine_lower", "pelvis_root"),
        ("spine_lower", "spine_middle"),
        ("spine_lower", "bcom"),
        ("spine_lower", "hcom"),
        ("pelvis_root", "spine_middle"),
        ("pelvis_root", "spine_upper"),
        ("spine_middle", "spine_upper"),
        ("spine_middle", "hcom"),
        ("spine_upper", "hcom"),
        ("spine_upper", "hcom"),     # duplicate per MATLAB line 120
        ("head_back",  "hcom"),
        ("bcom",       "hcom"),
        ("bcom",       "acom"),
        ("hcom",       "acom"),
    ]
    raw_yaws = np.column_stack([A(ang, mi, mj, "theta")
                                  for mi, mj in yaw_pairs])
    yaws_lp = _butter_lp(raw_yaws, pitch_filter_hz, src_sr)
    # circ_dist with shift-1; multiply by samplerate to make velocity
    yaw_vel = circ_dist(yaws_lp, np.roll(yaws_lp, 1, axis=0)) * src_sr
    # Window-sum of squares (centred)
    win = angle_window_samples
    half = win // 2
    yaw_vel_sq = yaw_vel ** 2
    cols_25_38 = np.empty_like(yaw_vel_sq)
    # Use a uniform_filter1d-style accumulator
    from scipy.ndimage import uniform_filter1d
    for c in range(yaw_vel_sq.shape[1]):
        # Window-SUM (not mean) over `win` samples
        cols_25_38[:, c] = (
            uniform_filter1d(yaw_vel_sq[:, c], size=win, mode="nearest")
            * win
        )
    # Centre-shift by half-window (MATLAB circshift(-round(samplerate/2)))
    cols_25_38 = np.roll(cols_25_38, -half, axis=0)
    cols_25_38 = np.where(cols_25_38 < 1e-6, 1e-6, cols_25_38)
    cols_25_38 = np.log10(cols_25_38)

    # ── Col 39: PPC of lower-spine trajectory yaw ─────────────────────#
    ppc_raw = lower_spine_yaw_ppc(xyz)
    # Filter @ 2 Hz
    col_39 = butter_filter(ppc_raw, cutoff=2.0, samplerate=src_sr,
                             btype="lowpass", order=3)

    # ── Cols 40-44: z-positions of 5 spine markers (filtered) ─────── #
    spine5 = [
        "spine_lower", "pelvis_root", "spine_middle", "spine_upper",
        "head_back",
    ]
    cols_40_44 = np.column_stack([
        fxyz.data[:, fxyz.model.index(m), 2]
        if m in fxyz.model.markers else np.zeros(fxyz.data.shape[0])
        for m in spine5
    ])

    # ── Cols 45-50: log10 xy-speed of 6 markers ─────────────────────── #
    xy_speed_markers = [
        "spine_lower", "spine_upper", "head_front",
        "bcom", "hcom", "acom",
    ]
    xy_speeds = []
    for m in xy_speed_markers:
        if m in xyz.model.markers:
            sp = xyz.vel(markers=[m], dims=[0, 1])      # (T, 1)
            sp = sp[:, 0]
        else:
            sp = np.zeros(xyz.data.shape[0])
        sp_lp = butter_filter(sp, cutoff=speed_filter_hz, samplerate=src_sr,
                                btype="lowpass", order=3)
        sp_lp = np.where(sp_lp < 0, 0.1, sp_lp)
        xy_speeds.append(np.log10(np.maximum(sp_lp, 1e-9)))
    cols_45_50 = np.column_stack(xy_speeds)

    # ── Cols 51-56: log10 z-speed (filtered) of same 6 markers ─────── #
    z_speeds = []
    for m in xy_speed_markers:
        if m in fxyz.model.markers:
            sp = fxyz.vel(markers=[m], dims=[2])[:, 0]
        else:
            sp = np.zeros(fxyz.data.shape[0])
        sp_lp = butter_filter(sp, cutoff=speed_filter_hz, samplerate=src_sr,
                                btype="lowpass", order=3)
        sp_lp = np.where(sp_lp < 0, 0.1, sp_lp)
        z_speeds.append(np.log10(np.maximum(sp_lp, 1e-9)))
    cols_51_56 = np.column_stack(z_speeds)

    # ── Col 57: BFET — body-axis-projected acceleration ───────────── #
    # MATLAB lines 147-157: filter xyz @ 20 Hz, take centred-diff @
    # 0.15s shift, project onto spine_upper - spine_lower direction,
    # then log10|·| × sign(·)
    mxyz_data = np.empty_like(xyz.data)
    for m in range(xyz.data.shape[1]):
        for d in range(3):
            mxyz_data[:, m, d] = butter_filter(
                xyz.data[:, m, d], cutoff=20.0, samplerate=src_sr,
                btype="lowpass", order=3,
            )
    tsh = int(round(0.15 * src_sr))
    diff_xyz = np.roll(mxyz_data, -tsh, axis=0) - np.roll(mxyz_data, tsh, axis=0)
    # Body axis: spine_upper - spine_lower (T, 3)
    sl_idx = xyz.model.index("spine_lower")
    su_idx = xyz.model.index("spine_upper")
    body_axis = mxyz_data[:, su_idx, :] - mxyz_data[:, sl_idx, :]
    norm = np.linalg.norm(body_axis, axis=1, keepdims=True)
    body_axis_unit = body_axis / np.where(norm > 0, norm, 1.0)  # (T, 3)
    # Project diff_xyz of marker 0 (spine_lower) onto body_axis
    # MATLAB does this for all markers and reshapes; we take col 1 only
    # since col 57 is the per-frame scalar.  Use spine_lower's diff.
    proj_sl = (diff_xyz[:, sl_idx, :] * body_axis_unit).sum(axis=1)
    col_57 = np.log10(np.maximum(np.abs(proj_sl), 1e-9)) * np.sign(proj_sl)

    # ── Col 58: SS — spine straightness ─────────────────────────── #
    # MATLAB lines 159-164: from spline-spine, sd = inter-segment
    # distances, sn = sum(sd[2:end-1]) / sd[end].  We approximate
    # using the ratio of summed adjacent-marker distances to the
    # endpoint distance.
    spine_seq = ["spine_lower", "pelvis_root", "spine_middle", "spine_upper"]
    spine_indices = [xyz.model.index(m) for m in spine_seq
                      if m in xyz.model.markers]
    spine_pos = xyz.data[:, spine_indices, :]      # (T, 4, 3)
    seg_lens = np.linalg.norm(np.diff(spine_pos, axis=1), axis=2)  # (T, 3)
    end_dist = np.linalg.norm(spine_pos[:, -1] - spine_pos[:, 0], axis=1)
    end_dist = np.maximum(end_dist, 1e-9)
    col_58 = seg_lens.sum(axis=1) / end_dist        # 1.0 = perfectly straight

    # ── Col 59: AV — angular variance ───────────────────────────── #
    # MATLAB lines 167-172: sum of consecutive circ_dist's around the
    # spine, deviation from circular mean.
    yaw_sl_pr = A(fang, "spine_lower",  "pelvis_root",  "theta")
    yaw_pr_sm = A(fang, "pelvis_root",  "spine_middle", "theta")
    yaw_sm_su = A(fang, "spine_middle", "spine_upper",  "theta")
    yaw_su_hc = A(fang, "spine_upper",  "hcom",         "theta")
    sang_col1 = circ_dist(yaw_sl_pr, yaw_pr_sm)
    sang_col2 = circ_dist(yaw_pr_sm, yaw_sm_su)
    sang_col3 = circ_dist(yaw_sm_su, yaw_su_hc)
    sang_sum = sang_col1 + sang_col2 + sang_col3
    av_mean = circ_mean(sang_sum)
    col_59 = np.abs(circ_dist(sang_sum, np.full_like(sang_sum, av_mean)))

    # Stack all columns (in MATLAB order)
    full = np.column_stack([
        cols_1_12,                                 # cols 1-12
        cols_13_24,                                # cols 13-24
        cols_25_38,                                # cols 25-38
        col_39[:, None],                            # col 39
        cols_40_44,                                # cols 40-44
        cols_45_50,                                # cols 45-50
        cols_51_56,                                # cols 51-56
        col_57[:, None],                            # col 57
        col_58[:, None],                            # col 58
        col_59[:, None],                            # col 59
    ])

    # Scrub non-finite values
    full = np.where(np.isfinite(full), full, 0.0)

    # Resample to target rate
    if abs(samplerate - src_sr) > 1e-9:
        full = _resample_array(full, src_sr, samplerate)

    column_names = (
        # Filtered pitches
        *[f"phi_{a}_{b}_filt" for a, b in pitch_pairs],
        # Pitch derivatives
        *[f"d_phi_{a}_{b}" for a, b in raw_pitch_pairs],
        # Yaw → angular velocity squared
        *[f"d2theta_{a}_{b}" for a, b in yaw_pairs],
        # PPC
        "lower_spine_yaw_ppc",
        # Z positions
        *[f"z_{m}" for m in spine5],
        # XY log speeds
        *[f"log10_xy_speed_{m}" for m in xy_speed_markers],
        # Z log speeds
        *[f"log10_z_speed_{m}" for m in xy_speed_markers],
        # BFET, SS, AV
        "bfet_body_proj_accel",
        "ss_spine_straightness",
        "av_angular_variance",
    )
    assert len(column_names) == 59, (
        f"fet_all expected 59 columns, got {len(column_names)}"
    )

    return FetAllResult(
        data         = full,
        column_names = column_names,
        samplerate   = float(samplerate),
    )
