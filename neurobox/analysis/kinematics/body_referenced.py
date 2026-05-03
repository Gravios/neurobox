"""
neurobox.analysis.kinematics.body_referenced
=============================================
Body-referenced position and motion features.

Ports of:

* :file:`MTA/features/fet_bref.m`     → :func:`body_referenced_features`
* :file:`MTA/features/fet_bref_BXY.m` → :func:`body_referenced_xy_features`

What it computes
----------------
For each frame and each tracked body marker
``(spine_lower, pelvis_root, spine_middle, spine_upper, hcom)``, four
quantities are computed in a coordinate frame **rotated to align with
the rat's body axis** (the line from spine_lower to spine_upper):

1. **Position** parallel to the body axis (forward/back)
2. **Position** perpendicular to the body axis (left/right)
3. **Translational motion** parallel to the body axis (3-sample
   forward-backward shift)
4. **Translational motion** perpendicular to the body axis

Plus the **z-coordinate** and **z-velocity** of each marker.

This gives a 20-dimensional feature vector per frame
(2 in-plane × 5 markers × 2 [position, motion] = 20) plus 5 z-position
+ 5 dz = 30 dimensions total in the canonical ordering, matching
MATLAB's ``fet_bref`` output column count.

Why "body-referenced"?
----------------------
By rotating into the rat's body frame, the features are **invariant to
the rat's heading**: a head-turn that's "rightward in the world" looks
like the same feature value whether the rat is facing north or south.
Without this rotation, the features would be dominated by the rat's
random navigation direction rather than its motor activity.

Differences from MATLAB
-----------------------
* The ``procOpts`` argument selecting between
  ``SPLINE_SPINE_HEAD_EQI`` / ``SPLINE_SPINE_HEAD_EQD`` is replaced by
  taking an already-preprocessed *xyz* directly.  Pre-process with
  :func:`neurobox.analysis.kinematics.spline_spine.preproc_xyz_spline_spine_head_eqd`
  (or `_eqi`) before calling.
* The lazy resample-after-compute pattern is replaced by a
  ``samplerate`` keyword that explicitly resamples the output
  feature.
* The optional spectrogram-based feature transformation
  (commented out in MATLAB lines 115-128) is not ported — use the
  multitaper helpers in :mod:`neurobox.analysis.lfp.spectral` if
  needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from neurobox.analysis.lfp.filtering import butter_filter
from neurobox.dtype.fet import NBDfet
from neurobox.dtype.xyz import NBDxyz


__all__ = [
    "BodyReferencedFeatures",
    "body_referenced_features",
    "body_referenced_xy_features",
]


# Standard markers used by MATLAB fet_bref
DEFAULT_BODY_MARKERS = (
    "spine_lower", "pelvis_root", "spine_middle", "spine_upper", "hcom",
)
DEFAULT_SHIFT_SAMPLES = 3
DEFAULT_FILTER_CUTOFF_HZ = 5.0


@dataclass
class BodyReferencedFeatures:
    """Output of :func:`body_referenced_features`.

    Attributes
    ----------
    fet : np.ndarray, shape ``(T, 30)``
        Feature columns in the canonical MATLAB order:

        - ``[0:10]``  walkFetRot — body-frame XY positions (5 markers × 2 axes)
        - ``[10:15]`` z-position of each marker
        - ``[15:25]`` dwalkFetRot — body-frame XY translational motion
        - ``[25:30]`` dz of each marker
    column_names : tuple[str, ...]
        Per-column descriptions matching MATLAB's ``featureTitles``.
    samplerate : float
    """
    fet:           np.ndarray
    column_names:  tuple[str, ...]
    samplerate:    float

    @property
    def n_features(self) -> int:
        return self.fet.shape[1]


# ─────────────────────────────────────────────────────────────────────── #
# Helpers                                                                    #
# ─────────────────────────────────────────────────────────────────────── #

def _circ_shift(arr: np.ndarray, shift: int, axis: int = 0) -> np.ndarray:
    """MATLAB-style ``circshift`` along *axis*.

    NumPy's :func:`np.roll` is the direct equivalent.
    """
    return np.roll(arr, shift, axis=axis)


def _body_axis_unit_vectors(
    spine_lower:    np.ndarray,    # (T, 2)
    spine_upper:    np.ndarray,    # (T, 2)
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-frame body-axis unit vectors.

    The body axis is ``spine_upper - spine_lower``.  Returns the
    forward unit vector and a 90°-rotated (lateral) unit vector.

    Parameters
    ----------
    spine_lower, spine_upper:
        ``(T, 2)`` xy-positions per frame.

    Returns
    -------
    forward : np.ndarray, shape ``(T, 2)``
    lateral : np.ndarray, shape ``(T, 2)``
    """
    mvec = spine_upper - spine_lower               # (T, 2)
    norm = np.linalg.norm(mvec, axis=1, keepdims=True)
    safe = np.where(norm > 0, norm, 1.0)
    forward = mvec / safe                          # (T, 2) unit vector
    # 90° rotation: (x, y) → (-y, x)
    lateral = np.column_stack([-forward[:, 1], forward[:, 0]])
    return forward, lateral


def _body_referenced_xy(
    xyz:            NBDxyz,
    body_markers:   Sequence[str],
    *,
    shift_samples:  int = DEFAULT_SHIFT_SAMPLES,
    bcom_filtered:  Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute walkFetRot (positional) and dwalkFetRot (translational).

    Returns
    -------
    walk_fet_rot : np.ndarray, shape ``(T, 2, n_markers)``
        Positional feature: (marker_xy − bcom_xy) projected onto the
        body-frame forward / lateral axes.
    dwalk_fet_rot : np.ndarray, shape ``(T, 2, n_markers)``
        Translational feature: marker xy at ``t+shift`` minus xy at
        ``t-shift``, projected onto body-frame axes.
    """
    sl_idx = xyz.model.index("spine_lower")
    su_idx = xyz.model.index("spine_upper")

    # XY of body-axis endpoints (used for the body-frame rotation)
    forward, lateral = _body_axis_unit_vectors(
        xyz.data[:, sl_idx, :2],
        xyz.data[:, su_idx, :2],
    )

    n_markers = len(body_markers)
    T = xyz.data.shape[0]

    # bcom = body centre of mass (lower 4 spine markers).  Filtered
    # version (MATLAB 'fbcom') only needed for some bref variants;
    # the canonical fet_bref uses the *unfiltered* bcom in the position
    # term (line 86 of fet_bref.m: cvec uses xyz(:,'bcom',[1,2])).
    if bcom_filtered is None:
        spine_indices = [
            xyz.model.index(m) for m in
            ("spine_lower", "pelvis_root", "spine_middle", "spine_upper")
            if m in xyz.model.markers
        ]
        bcom = np.mean(xyz.data[:, spine_indices, :2], axis=1)
    else:
        bcom = bcom_filtered

    # Position term (walkFetRot)
    walk_fet_rot = np.zeros((T, 2, n_markers), dtype=np.float64)
    # Translational term (dwalkFetRot)
    dwalk_fet_rot = np.zeros((T, 2, n_markers), dtype=np.float64)

    for k, name in enumerate(body_markers):
        if name not in xyz.model.markers:
            continue
        midx = xyz.model.index(name)
        m_xy = xyz.data[:, midx, :2]                       # (T, 2)

        # Position term: marker − bcom, projected onto body-frame axes
        pos_vec = m_xy - bcom                              # (T, 2)
        walk_fet_rot[:, 0, k] = np.einsum("ij,ij->i", pos_vec, forward)
        walk_fet_rot[:, 1, k] = np.einsum("ij,ij->i", pos_vec, lateral)

        # Translation term: m(t+shift) - m(t-shift)
        tvec = (
            _circ_shift(m_xy, -shift_samples, axis=0)
            - _circ_shift(m_xy,  shift_samples, axis=0)
        )                                                  # (T, 2)
        dwalk_fet_rot[:, 0, k] = np.einsum("ij,ij->i", tvec, forward)
        dwalk_fet_rot[:, 1, k] = np.einsum("ij,ij->i", tvec, lateral)

    return walk_fet_rot, dwalk_fet_rot


# ─────────────────────────────────────────────────────────────────────── #
# Public API                                                                 #
# ─────────────────────────────────────────────────────────────────────── #

def body_referenced_features(
    xyz:           NBDxyz,
    *,
    body_markers:  Sequence[str] = DEFAULT_BODY_MARKERS,
    shift_samples: int = DEFAULT_SHIFT_SAMPLES,
    samplerate:    Optional[float] = None,
    as_nbdfet:     bool = False,
) -> BodyReferencedFeatures | NBDfet:
    """Compute the canonical fet_bref feature set.

    Port of :file:`MTA/features/fet_bref.m`.

    For each of 5 body markers, computes 4 features:

    * 2 positional (xy in body frame, relative to body centre of mass)
    * 2 translational (xy motion in body frame, ``shift_samples``-step diff)

    Plus per-marker z-position and dz, giving 30 columns total.

    Parameters
    ----------
    xyz:
        Augmented :class:`NBDxyz`.  Should have been preprocessed
        through one of the spline-spine modes (see
        :mod:`neurobox.analysis.kinematics.spline_spine`) for
        cross-session comparability.  Must contain ``spine_lower``,
        ``spine_upper``, and the body markers.
    body_markers:
        Markers to feature-ise.  Default
        ``('spine_lower', 'pelvis_root', 'spine_middle',
        'spine_upper', 'hcom')`` matches MATLAB.
    shift_samples:
        Time-shift for the translational difference.  Default 3
        matches MATLAB.
    samplerate:
        Optional resample target.  ``None`` (default) → preserve
        *xyz*'s rate.
    as_nbdfet:
        If True, wrap the result in an :class:`NBDfet`.  Default False
        returns :class:`BodyReferencedFeatures`.

    Returns
    -------
    BodyReferencedFeatures or NBDfet
    """
    n_markers = len(body_markers)
    T = xyz.data.shape[0]

    # Position + translation terms in body frame
    walk_fet_rot, dwalk_fet_rot = _body_referenced_xy(
        xyz, body_markers, shift_samples=shift_samples,
    )

    # z-position
    z_pos = np.zeros((T, n_markers), dtype=np.float64)
    for k, name in enumerate(body_markers):
        if name not in xyz.model.markers:
            continue
        z_pos[:, k] = xyz.data[:, xyz.model.index(name), 2]
    # MATLAB: zvec(~nniz(zvec(:)))=0;
    z_pos[~np.isfinite(z_pos)] = 0.0

    # dz (3-sample shift, like the xy translation)
    dz = np.zeros((T, n_markers), dtype=np.float64)
    for k, name in enumerate(body_markers):
        if name not in xyz.model.markers:
            continue
        z = xyz.data[:, xyz.model.index(name), 2]
        dz[:, k] = (
            _circ_shift(z, -shift_samples) - _circ_shift(z, shift_samples)
        )
    dz[~np.isfinite(dz)] = 0.0

    # Concatenate in MATLAB order:
    # [walkFetRot(forward,lateral × markers), zvec, dwalkFetRot, dzvec]
    walk_flat  = walk_fet_rot.transpose(0, 2, 1).reshape(T, -1)    # (T, n*2)
    dwalk_flat = dwalk_fet_rot.transpose(0, 2, 1).reshape(T, -1)
    fet = np.concatenate([walk_flat, z_pos, dwalk_flat, dz], axis=1)

    # Zero out invalid frames (MATLAB: fet.data(~nniz(xyz),:)=0;)
    finite_xyz = np.isfinite(xyz.data).all(axis=(1, 2))
    fet[~finite_xyz, :] = 0.0

    # Column-name labels
    column_names: list[str] = []
    for m in body_markers:
        column_names.append(f"{m}_body_fwd_pos")
        column_names.append(f"{m}_body_lat_pos")
    for m in body_markers:
        column_names.append(f"{m}_z_pos")
    for m in body_markers:
        column_names.append(f"{m}_body_fwd_dx")
        column_names.append(f"{m}_body_lat_dx")
    for m in body_markers:
        column_names.append(f"{m}_dz")

    out_sr = float(xyz.samplerate)
    if samplerate is not None and samplerate != out_sr:
        # Linear resample of feature columns
        from scipy.signal import resample_poly
        from math import gcd
        old_sr = out_sr
        new_sr = float(samplerate)
        # Use simple ratio approximation
        from fractions import Fraction
        ratio = Fraction(new_sr / old_sr).limit_denominator(10_000)
        up, down = ratio.numerator, ratio.denominator
        fet = resample_poly(fet, up, down, axis=0)
        out_sr = new_sr

    result = BodyReferencedFeatures(
        fet           = fet,
        column_names  = tuple(column_names),
        samplerate    = out_sr,
    )

    if as_nbdfet:
        return NBDfet(
            data       = fet,
            samplerate = out_sr,
            columns    = list(column_names),
            name       = "fet_bref",
            label      = "bref",
            key        = "b",
        )
    return result


def body_referenced_xy_features(
    xyz:           NBDxyz,
    *,
    body_markers:  Sequence[str] = DEFAULT_BODY_MARKERS,
    shift_samples: int = DEFAULT_SHIFT_SAMPLES,
    samplerate:    Optional[float] = None,
    as_nbdfet:     bool = False,
) -> BodyReferencedFeatures | NBDfet:
    """Body-referenced **XY-only** feature set (no z components).

    Port of :file:`MTA/features/fet_bref_BXY.m`.

    Returns 20 features per frame: same as
    :func:`body_referenced_features` but without the ``z_pos`` and
    ``dz`` columns.  Used by ``MjgER2016_figure_BhvClassification.m``
    for the kinematics-only branch of the classifier.

    Parameters
    ----------
    xyz, body_markers, shift_samples, samplerate, as_nbdfet:
        See :func:`body_referenced_features`.

    Returns
    -------
    BodyReferencedFeatures or NBDfet
    """
    full = body_referenced_features(
        xyz,
        body_markers  = body_markers,
        shift_samples = shift_samples,
        samplerate    = samplerate,
        as_nbdfet     = False,
    )
    n_m = len(body_markers)
    # Slice columns: keep walkFetRot (first 2*n_m) and dwalkFetRot
    # (columns 3*n_m : 5*n_m), drop z-pos and dz.
    walk_cols  = slice(0, 2 * n_m)
    dwalk_cols = slice(3 * n_m, 5 * n_m)
    keep_walk  = full.fet[:, walk_cols]
    keep_dwalk = full.fet[:, dwalk_cols]
    fet = np.concatenate([keep_walk, keep_dwalk], axis=1)

    keep_walk_names  = full.column_names[walk_cols]
    keep_dwalk_names = full.column_names[dwalk_cols]
    column_names = tuple(keep_walk_names) + tuple(keep_dwalk_names)

    result = BodyReferencedFeatures(
        fet           = fet,
        column_names  = column_names,
        samplerate    = full.samplerate,
    )

    if as_nbdfet:
        return NBDfet(
            data       = fet,
            samplerate = full.samplerate,
            columns    = list(column_names),
            name       = "fet_bref_BXY",
            label      = "bref_xy",
            key        = "x",
        )
    return result
