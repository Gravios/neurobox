"""
fet_mis.py
==========
The ``fet_mis`` feature set — the default feature basis for the
behaviour-labelling pipeline (``label_behavior`` → ``label_bhv_msnn``).

Port of :file:`MTA/features/fet_mis.m`.

Eighteen columns, in MATLAB order:

===  ==========================  ===========================================
 #   Title                       Content
===  ==========================  ===========================================
 1   Pitch BLBM                  elevation spine_lower → spine_middle
 2   Pitch BLHC                  elevation spine_lower → hcom
 3   Pitch BPBU                  elevation pelvis_root → spine_upper
 4   Pitch BMBU                  elevation spine_middle → spine_upper
 5   Pitch BMHC                  elevation spine_middle → hcom
 6   PPC traj yaw                pairwise phase consistency of marker yaw
 7   Z BL                        height of spine_lower
 8   Z BP                        height of pelvis_root
 9   Z BM                        height of spine_middle
10   Z BU                        height of spine_upper
11   XY Speed BL                 log10 speed, spine_lower
12   XY Speed BU                 log10 speed, spine_upper
13   XY Speed HF                 log10 speed, hcom
14   XY Speed AC                 log10 speed, acom
15   Z Speed HF                  log10 vertical speed, hcom
16   Z Speed AC                  log10 vertical speed, acom
17   Spine Sinuosity             spline arc length / end-to-end distance
18   mean(d(yaw BLBPBMBUHC)/dt)  summed inter-segment yaw deviation
===  ==========================  ===========================================

Notes on fidelity
-----------------
* The PPC column is computed at the **source** sample rate and only
  then low-passed and resampled, matching MATLAB — where it came from
  a cached ``.lsppc`` file written at the original rate.  Computing it
  after downsampling would silently change the meaning of the ``shift``
  parameter (5 samples is 5/120 s, not 5/12 s).
* MATLAB's ``MTADang`` is 1-indexed on its component axis:
  ``ang(:,i,j,1)`` is azimuth and ``ang(:,i,j,2)`` elevation.  neurobox
  uses named components (``'theta'`` / ``'phi'``) over a 0-indexed axis;
  the mapping is ``1 → 'theta'``, ``2 → 'phi'``.
* Sinuosity divides by the **wrap-around** segment of the spline
  (``sd[:, -1]``), which is the first-to-last point distance — i.e. the
  chord.  The numerator deliberately skips the first segment, following
  the MATLAB ``sum(sd(:,2:end-1),2)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from neurobox.dtype import NBDxyz, NBDang, NBDfet
from neurobox.analysis.lfp.filtering import butter_filter
from neurobox.analysis.stats.circular import circ_dist, circ_mean
from neurobox.analysis.kinematics.spline_spine import (
    preproc_xyz_spline_spine_head_eqd, spline_spine,
)
from neurobox.analysis.kinematics.fet_all import lower_spine_yaw_ppc
from neurobox.analysis.kinematics.helpers import finite_nonzero_mask


__all__ = ["FetMisConfig", "fet_mis", "FET_MIS_TITLES", "FET_MIS_DESCRIPTIONS"]


FET_MIS_TITLES: tuple[str, ...] = (
    "Pitch BLBM", "Pitch BLHC", "Pitch BPBU", "Pitch BMBU", "Pitch BMHC",
    "PPC traj yaw",
    "Z BL", "Z BP", "Z BM", "Z BU",
    "XY Speed BL", "XY Speed BU", "XY Speed HF", "XY Speed AC",
    "Z Speed HF", "Z Speed AC",
    "Spine Sinuosity",
    "mean(d(yaw BLBPBMBUHC)/dt)",
)

FET_MIS_DESCRIPTIONS: tuple[str, ...] = (
    "Pitch of body lower to body middle relative to xy plane",
    "Pitch of body lower to head COM relative to xy plane",
    "Pitch of body pelvis to body upper relative to xy plane",
    "Pitch of body middle to body upper relative to xy plane",
    "Pitch of body middle to head COM relative to xy plane",
    "Low-pass filtered pairwise phase consistency (PPC) of the yaw of "
    "marker trajectories along the rostro-caudal axis",
    "Low-pass filtered height of the lower spine marker",
    "Low-pass filtered height of the pelvic marker",
    "Low-pass filtered height of the middle spine marker",
    "Low-pass filtered height of the upper spine marker",
    "Low-pass filtered speed in the xy plane of the spine lower marker",
    "Low-pass filtered speed in the xy plane of the spine upper marker",
    "Low-pass filtered speed in the xy plane of the head COM marker",
    "Low-pass filtered speed in the xy plane of the body+head COM marker",
    "Low-pass filtered speed in the z axis of the head COM marker",
    "Low-pass filtered speed in the z axis of the body+head COM marker",
    "Length of the spine divided by the distance between its endpoints",
    "Mean yaw speed of the vector from the lower body to the head COM",
)


@dataclass(frozen=True)
class FetMisConfig:
    """Parameters for :func:`fet_mis`.

    Defaults reproduce :file:`MTA/features/fet_mis.m` exactly.  The
    dataclass is frozen and hashable so it can key a
    :func:`neurobox.io.cached_compute` cache — replacing the
    concatenated ``MTAC_BATCH+...`` model-identity string the MATLAB
    pipeline used.
    """
    sample_rate:        float = 12.0    # newSampleRate
    xy_lowpass_hz:      float = 2.4     # fxyz ButFilter cutoff
    z_speed_lowpass_hz: float = 2.5     # fvelz ButFilter cutoff
    ppc_lowpass_hz:     float = 2.0     # lsppc ButFilter cutoff
    filter_order:       int   = 3       # ButFilter order, all three above
    ppc_shift:          int   = 5       # centred finite-difference half-width
    n_spline_interp:    int   = 100     # spline points for sinuosity
    xy_speed_floor:     float = 1e-4    # clamp before log10
    z_speed_floor:      float = 0.1     # value substituted for <= 0
    spline_markers:     tuple[str, ...] = (
        "spine_lower", "pelvis_root", "spine_middle", "spine_upper", "hcom",
    )


def _lowpass(x: np.ndarray, cutoff: float, fs: float, order: int) -> np.ndarray:
    """Zero-phase Butterworth low-pass over the leading (time) axis."""
    if x.ndim == 1:
        return butter_filter(x, cutoff, fs, order=order, btype="lowpass")
    out = np.empty_like(x, dtype=np.float64)
    flat = x.reshape(x.shape[0], -1)
    res = np.empty_like(flat, dtype=np.float64)
    for c in range(flat.shape[1]):
        res[:, c] = butter_filter(
            flat[:, c], cutoff, fs, order=order, btype="lowpass")
    return res.reshape(out.shape)


def _spine_sinuosity(points: np.ndarray) -> np.ndarray:
    """Arc length / chord length of the fitted spline.

    *points* has shape ``(T, n_interp, 3)``.  Reproduces::

        sd = sqrt(sum((ss - circshift(ss,-1,2)).^2, 3));
        sn = sum(sd(:,2:end-1),2) ./ sd(:,end);

    ``circshift(...,-1,2)`` wraps, so the final column of ``sd`` is the
    last-point-to-first-point distance — the chord.
    """
    seg = np.sqrt(((points - np.roll(points, -1, axis=1)) ** 2).sum(axis=2))
    chord = seg[:, -1]
    arc = seg[:, 1:-1].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = arc / chord
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def fet_mis(
    xyz:      NBDxyz,
    *,
    config:   FetMisConfig | None = None,
) -> NBDfet:
    """Compute the ``fet_mis`` 18-column feature set.

    Parameters
    ----------
    xyz:
        Source marker positions at the acquisition sample rate.  Must
        contain the spine chain plus ``hcom`` / ``acom`` — run
        :func:`~neurobox.analysis.kinematics.augment.augment_xyz`
        first if the COM markers aren't present.
    config:
        :class:`FetMisConfig`; defaults reproduce MATLAB.

    Returns
    -------
    NBDfet
        Shape ``(T', 18)`` at ``config.sample_rate``, with
        :data:`FET_MIS_TITLES` as column titles.

    Raises
    ------
    KeyError
        If a required marker is absent from *xyz*.
    """
    cfg = config or FetMisConfig()

    required = ("spine_lower", "pelvis_root", "spine_middle",
                "spine_upper", "hcom", "acom")
    missing = [m for m in required if m not in xyz.markers]
    if missing:
        raise KeyError(
            f"fet_mis requires markers {missing} which are absent from xyz. "
            "Run augment_xyz() to synthesise the COM markers."
        )

    src_fs = float(xyz.samplerate)

    # ── PPC at the SOURCE rate (see module docstring) ───────────────── #
    ppc_src = lower_spine_yaw_ppc(xyz, shift=cfg.ppc_shift)
    ppc_src = _lowpass(ppc_src, cfg.ppc_lowpass_hz, src_fs, cfg.filter_order)

    # ── Preprocess + downsample ─────────────────────────────────────── #
    pxyz = preproc_xyz_spline_spine_head_eqd(
        xyz, target_markers=cfg.spline_markers, n_interp=cfg.n_spline_interp)
    pxyz = pxyz.resample(cfg.sample_rate)
    fs = float(pxyz.samplerate)
    T = pxyz.data.shape[0]

    # Resample the source-rate PPC onto the feature grid.
    ppc = np.interp(
        np.linspace(0.0, 1.0, T),
        np.linspace(0.0, 1.0, ppc_src.shape[0]),
        ppc_src,
    )

    # ── Filtered positions ──────────────────────────────────────────── #
    fxyz = pxyz.copy()
    fxyz.data = _lowpass(
        pxyz.data.astype(np.float64), cfg.xy_lowpass_hz, fs, cfg.filter_order)

    # ── Speeds ──────────────────────────────────────────────────────── #
    velxy = fxyz.vel(["spine_lower", "spine_upper", "hcom", "acom"], [0, 1])
    velxy = np.maximum(velxy, cfg.xy_speed_floor)
    velxy = np.log10(velxy)

    velz = fxyz.vel(["hcom", "acom"], [2])
    velz = _lowpass(velz, cfg.z_speed_lowpass_hz, fs, cfg.filter_order)
    velz = np.where(velz <= 0.0, cfg.z_speed_floor, velz)
    velz = np.log10(velz)

    # ── Inter-marker angles ─────────────────────────────────────────── #
    ang = NBDang.from_xyz(fxyz)
    pitch = lambda a, b: ang.between(a, b, "phi")     # MATLAB component 2
    yaw   = lambda a, b: ang.between(a, b, "theta")   # MATLAB component 1

    # ── Spine sinuosity ─────────────────────────────────────────────── #
    spline = spline_spine(pxyz, markers=cfg.spline_markers,
                          n_interp=cfg.n_spline_interp)
    sinuosity = _spine_sinuosity(spline.points)

    # ── Summed inter-segment yaw deviation ──────────────────────────── #
    sang = np.column_stack([
        circ_dist(yaw("spine_lower", "pelvis_root"),
                  yaw("pelvis_root", "spine_middle")),
        circ_dist(yaw("pelvis_root", "spine_middle"),
                  yaw("spine_middle", "spine_upper")),
        circ_dist(yaw("spine_middle", "spine_upper"),
                  yaw("spine_upper", "hcom")),
    ])
    ssum = sang.sum(axis=1)
    angular_dev = np.abs(ssum - circ_mean(ssum))

    # ── Assemble ────────────────────────────────────────────────────── #
    data = np.column_stack([
        pitch("spine_lower",  "spine_middle"),
        pitch("spine_lower",  "hcom"),
        pitch("pelvis_root",  "spine_upper"),
        pitch("spine_middle", "spine_upper"),
        pitch("spine_middle", "hcom"),
        ppc,
        fxyz.sel(["spine_lower"],  [2]).squeeze(-1),
        fxyz.sel(["pelvis_root"],  [2]).squeeze(-1),
        fxyz.sel(["spine_middle"], [2]).squeeze(-1),
        fxyz.sel(["spine_upper"],  [2]).squeeze(-1),
        velxy,
        velz,
        sinuosity,
        angular_dev,
    ]).astype(np.float64)

    # MATLAB: fet.data(~nniz(xyz),:) = 0
    bad = ~finite_nonzero_mask(pxyz.data.reshape(T, -1).sum(axis=1))
    data[bad, :] = 0.0
    data[~np.isfinite(data)] = 0.0

    return NBDfet(
        data         = data,
        columns      = list(FET_MIS_TITLES),
        samplerate   = fs,
        label        = "fet_mis",
        name         = "req20160310_selected_features",
        key          = "m",
        titles        = list(FET_MIS_TITLES),
        descriptions  = list(FET_MIS_DESCRIPTIONS),
    )
