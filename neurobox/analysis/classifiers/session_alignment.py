"""
neurobox.analysis.classifiers.session_alignment
=================================================
Per-feature, behavioural-manifold-conditioned offset correction
between sessions.

Port of :file:`MTA/@MTADfet/map_to_reference_session.m` (Anton
Sirota / Justin Graboski, c. 2016) and its helper
:file:`MTA/utilities/mean_embeded_feature_vbvhza.m`.

What it does
------------
When you train a classifier on session A and apply it to session B,
the same feature can have a different absolute scale across sessions
— marker placement varies between animals, motion-capture coordinate
frames drift over recording sessions, and individual differences in
body geometry shift feature distributions.  Normalising by population
mean / std (round 12's :class:`FeatureNormalisation`) corrects for
*global* scale but not for *configuration-dependent* offsets.

This algorithm corrects for the latter:

1. For both target and reference session, estimate a 3-D behavioural
   manifold parameterised by ``(log10(body_velocity),
   log10(head_velocity), head_height)``.  Bin samples into a 20×20×20
   grid (matches MATLAB).
2. Within active, non-rear, non-groom periods, compute the **mean
   feature value** per occupied bin in each session.
3. Compute the **median per-bin difference** between target and
   reference: this is the offset *mshift*.  Use ``circ_dist`` for
   circular features and plain subtraction for linear features.
4. **Subtract** *mshift* from the target feature.

Result: the target feature now has the same expected value as the
reference within every cell of the behavioural manifold, so a
classifier trained on one transfers to the other.

How it differs from the MATLAB port
-----------------------------------
* The MATLAB function had a 30-case switch over feature names
  (``fet_bref``, ``fet_HB_pitch``, ``fet_tsne_rev*``, etc.) that
  hardcoded which feature columns are circular and which are linear.
  This port replaces the switch with explicit *circular_columns* /
  *linear_columns* arguments — cleaner and not tied to the lab's
  specific feature-naming history.

* The MATLAB ``mean_embeded_feature_vbvhza`` had a fallback heuristic
  for sessions without hand-labels (a ``cos(spine_lower→hcom angle)``
  threshold to remove rears).  This port requires the caller to
  pass an explicit *active_mask* — be honest about what's being
  excluded.

* The MATLAB version recomputed the reference feature lazily via
  ``feval(features.label, RefTrial, ...)``.  This port requires
  *reference_feature* and *reference_xyz* as arguments — the caller
  decides how to compute them.

* ``preproc_xyz('trb')`` (spline-spine, deferred from round 10) is
  replaced by :func:`neurobox.analysis.kinematics.augment_xyz` which
  provides ``bcom``, ``hcom``, ``acom``.  The behavioural-manifold
  axes (body/head velocity, head height) are well-defined without
  the spline-spine remix.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from neurobox.analysis.stats.circular import circ_dist
from neurobox.dtype.fet import NBDfet
from neurobox.dtype.xyz import NBDxyz


__all__ = [
    "BehaviouralManifoldStats",
    "behavioural_manifold_stats",
    "map_to_reference_session",
]


# ─────────────────────────────────────────────────────────────────────── #
# Constants — match MATLAB defaults                                          #
# ─────────────────────────────────────────────────────────────────────── #

NBINS = 20
VEL_HISTOGRAM_BOUNDARIES    = np.linspace(-3.0, 2.0, NBINS)   # log10(cm/s)
HEIGHT_HISTOGRAM_BOUNDARIES = np.linspace(0.0, 200.0, NBINS)  # mm


@dataclass
class BehaviouralManifoldStats:
    """Per-bin statistics on the (log_vbody, log_vhead, z_head) manifold.

    Attributes
    ----------
    mean : np.ndarray, shape ``(20, 20, 20, n_features)``
        Mean feature value per bin.  NaN for unoccupied bins.
    std : np.ndarray, shape ``(20, 20, 20, n_features)``
        Within-bin standard deviation.  NaN for unoccupied bins.
    count : np.ndarray, shape ``(20, 20, 20)``
        Number of valid samples per bin (shared across all features).
    """
    mean:  np.ndarray
    std:   np.ndarray
    count: np.ndarray


# ─────────────────────────────────────────────────────────────────────── #
# Helpers                                                                    #
# ─────────────────────────────────────────────────────────────────────── #

def _log_velocity(xyz: NBDxyz, marker: str) -> np.ndarray:
    """Log10 of speed (cm/s) of *marker* with the standard 2.4 Hz lowpass.

    Reproduces the MATLAB::

        vxy = xyz.copy.filter('ButFilter', 3, 2.4, 'low')
        vxy = vel(vxy, [marker, ...])
        vxy(vxy < 1e-3) = 1e-3
        vxy = log10(vxy)
    """
    work = xyz.copy() if hasattr(xyz, "copy") else NBDxyz(
        xyz.data.copy(), model=xyz.model, samplerate=xyz.samplerate,
    )
    work.filter(mode="butter", cutoff=2.4, btype="low", order=3)
    speed = work.vel(markers=[marker])         # (T,) or (T, 1)
    speed = np.asarray(speed).reshape(-1)
    speed[speed < 1e-3] = 1e-3
    return np.log10(speed)


def _bin_indices(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Histogram bin indices in ``[0, len(edges)-1)`` or -1 for out-of-range.

    Mirrors MATLAB's ``histc``: samples outside the edge range become 0
    in MATLAB, which is then dropped via the ``nniz`` mask.  We use -1
    as the out-of-range sentinel.
    """
    idx = np.digitize(values, edges, right=False) - 1
    idx[idx < 0] = -1
    idx[idx >= len(edges) - 1] = -1
    idx[~np.isfinite(values)]  = -1
    return idx


def _per_bin_mean_std(
    values:    np.ndarray,           # (T, n_features)
    bin_idx:   np.ndarray,           # (T, 3) — 0..NBINS-1 or -1
    n_bins:    int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-bin mean, std, count over a 3-D grid.

    Returns ``(mean, std, count)`` where mean/std are
    ``(n_bins, n_bins, n_bins, n_features)`` and count is
    ``(n_bins, n_bins, n_bins)``.  Empty bins are NaN in mean/std.
    """
    valid = np.all(bin_idx >= 0, axis=1)
    bx = bin_idx[valid]
    vv = values[valid]
    n_features = vv.shape[1]

    shape = (n_bins, n_bins, n_bins)
    mean = np.full(shape + (n_features,), np.nan, dtype=np.float64)
    std  = np.full(shape + (n_features,), np.nan, dtype=np.float64)
    count = np.zeros(shape, dtype=np.int64)

    if vv.size == 0:
        return mean, std, count

    # Linear bin index for grouping
    lin = (bx[:, 0] * n_bins + bx[:, 1]) * n_bins + bx[:, 2]
    order = np.argsort(lin, kind="stable")
    lin_s = lin[order]
    vv_s = vv[order]

    # Find runs of equal lin
    changes = np.flatnonzero(np.diff(lin_s)) + 1
    starts  = np.concatenate(([0], changes))
    ends    = np.concatenate((changes, [lin_s.size]))

    for s, e in zip(starts, ends):
        bin_lin = int(lin_s[s])
        group = vv_s[s:e]
        # Decompose linear → (i, j, k)
        ij, k = divmod(bin_lin, n_bins)
        i,  j = divmod(ij,      n_bins)
        count[i, j, k] = e - s
        # nan-aware reductions
        with np.errstate(all="ignore"):
            mean[i, j, k, :] = np.nanmean(group, axis=0)
            std [i, j, k, :] = np.nanstd (group, axis=0)
    return mean, std, count


# ─────────────────────────────────────────────────────────────────────── #
# Public API                                                                 #
# ─────────────────────────────────────────────────────────────────────── #

def behavioural_manifold_stats(
    feature:       np.ndarray | NBDfet,
    xyz:           NBDxyz,
    *,
    active_mask:   np.ndarray,
    body_marker:   str = "bcom",
    head_marker:   str = "hcom",
    height_marker: str = "acom",
    height_dim:    int = 2,
    n_bins:        int = NBINS,
    vel_edges:     np.ndarray = VEL_HISTOGRAM_BOUNDARIES,
    height_edges:  np.ndarray = HEIGHT_HISTOGRAM_BOUNDARIES,
) -> BehaviouralManifoldStats:
    """Per-bin mean / std / count of *feature* on the 3-D behavioural manifold.

    Port of :file:`MTA/utilities/mean_embeded_feature_vbvhza.m`.

    The manifold axes are:

    * ``log10(speed of body_marker)`` — typically ``bcom``.
    * ``log10(speed of head_marker)`` — typically ``hcom``.
    * z-coordinate of *height_marker* — typically ``acom``.

    Parameters
    ----------
    feature:
        ``(T, n_features)`` array (or :class:`NBDfet`) — values to
        average per manifold bin.  Must be at the same samplerate as
        *xyz*.
    xyz:
        Augmented :class:`NBDxyz`.  Must contain *body_marker*,
        *head_marker* and *height_marker* (use
        :func:`neurobox.analysis.kinematics.augment_xyz` first).
    active_mask:
        ``(T,)`` boolean — typically the ``a-m-r`` (active minus
        groom minus rear) periods, resampled to *xyz*'s rate.
    body_marker, head_marker, height_marker:
        Marker names for the three manifold axes.  Defaults match
        MATLAB.
    height_dim:
        Spatial-axis index for *height_marker*.  Default 2 (z).
    n_bins:
        Bins per axis.  Default 20 (matches MATLAB).
    vel_edges, height_edges:
        Bin edges for the velocity (log10 cm/s) and height (mm) axes.

    Returns
    -------
    BehaviouralManifoldStats
    """
    if isinstance(feature, NBDfet):
        feat_arr = np.asarray(feature.data, dtype=np.float64)
    else:
        feat_arr = np.asarray(feature, dtype=np.float64)
    if feat_arr.ndim == 1:
        feat_arr = feat_arr[:, None]
    T, n_features = feat_arr.shape

    if active_mask.shape != (T,):
        raise ValueError(
            f"active_mask shape {active_mask.shape} != ({T},)"
        )

    # Build manifold coordinates
    log_v_body = _log_velocity(xyz, body_marker)
    log_v_head = _log_velocity(xyz, head_marker)

    h_idx = xyz.model.index(height_marker)
    z_anterior = xyz.data[:, h_idx, height_dim]

    # All three must have length T
    if not (log_v_body.size == log_v_head.size == z_anterior.size == T):
        raise ValueError(
            f"length mismatch: log_v_body={log_v_body.size}, "
            f"log_v_head={log_v_head.size}, z_anterior={z_anterior.size}, "
            f"feat_arr={T}"
        )

    bin_v_body = _bin_indices(log_v_body, vel_edges)
    bin_v_head = _bin_indices(log_v_head, vel_edges)
    bin_z      = _bin_indices(z_anterior, height_edges)

    bin_idx = np.column_stack([bin_v_body, bin_v_head, bin_z])

    # Drop samples outside the active mask
    bin_idx[~active_mask, :] = -1

    return BehaviouralManifoldStats(*_per_bin_mean_std(
        feat_arr, bin_idx, n_bins,
    ))


def map_to_reference_session(
    target_feature:    np.ndarray | NBDfet,
    target_xyz:        NBDxyz,
    target_active:     np.ndarray,
    reference_feature: np.ndarray | NBDfet,
    reference_xyz:     NBDxyz,
    reference_active:  np.ndarray,
    *,
    circular_columns:  Sequence[int] = (),
    linear_columns:    Sequence[int] | None = None,
    minimum_occupancy_seconds: float = 1.0,
    body_marker:       str = "bcom",
    head_marker:       str = "hcom",
    height_marker:     str = "acom",
    height_dim:        int = 2,
    n_bins:            int = NBINS,
    vel_edges:         np.ndarray = VEL_HISTOGRAM_BOUNDARIES,
    height_edges:      np.ndarray = HEIGHT_HISTOGRAM_BOUNDARIES,
    return_offsets:    bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Bring *target_feature* onto *reference_feature*'s scale via per-feature offsets.

    Port of :file:`MTA/@MTADfet/map_to_reference_session.m`.

    The algorithm computes one scalar offset per feature column —
    the median per-manifold-bin difference between target and
    reference — and subtracts it (or applies ``circ_dist`` for
    circular features).

    Parameters
    ----------
    target_feature:
        ``(T_target, n_features)`` feature array (or :class:`NBDfet`).
        Modified out-of-place; original is not changed.
    target_xyz:
        Augmented :class:`NBDxyz` for the target session.
    target_active:
        ``(T_target,)`` boolean — active, non-groom, non-rear samples.
    reference_feature, reference_xyz, reference_active:
        Same as the target but for the reference session.
    circular_columns:
        Column indices that are angular (in radians).  These use
        ``circ_dist`` instead of subtraction for the offset.  Default
        ``()`` — no circular columns.
    linear_columns:
        Column indices to apply the *linear* (subtraction) offset to.
        ``None`` (default) → all columns not in *circular_columns*.
        Pass an explicit list to skip some columns entirely.
    minimum_occupancy_seconds:
        Minimum per-bin occupancy (in seconds at the target's
        samplerate) required for that bin to contribute to the
        median-offset estimate.  Matches MATLAB default of 1 s.
    body_marker, head_marker, height_marker, height_dim,
    n_bins, vel_edges, height_edges:
        Forwarded to :func:`behavioural_manifold_stats`.
    return_offsets:
        If True, also return the per-column offset vector.  Default False.

    Returns
    -------
    aligned : np.ndarray, shape ``(T_target, n_features)``
        Target feature with offsets applied.  Same dtype as input.
    offsets : np.ndarray, shape ``(n_features,)``  *(only if return_offsets=True)*
        The offset that was subtracted (or circ_dist'd) per column.
        NaN for columns not in *linear_columns* or *circular_columns*.

    Notes
    -----
    Caller responsibilities:

    * Both feature arrays must be at the same samplerate as their
      respective *xyz*.  Resample upstream if needed.
    * The reference feature must be the **same kind** of feature as
      the target (e.g. both ``fet_HB_pitchB``).  This function does
      not validate that.
    * Zero-valued samples in the target feature are preserved (set to
      zero after the offset is applied), matching MATLAB's
      ``zind = features.data(...) == 0`` patch.
    """
    # Coerce inputs
    if isinstance(target_feature, NBDfet):
        target_arr = np.asarray(target_feature.data, dtype=np.float64).copy()
        target_sr  = float(target_feature.samplerate)
    else:
        target_arr = np.asarray(target_feature, dtype=np.float64).copy()
        target_sr  = float(target_xyz.samplerate)
    if target_arr.ndim == 1:
        target_arr = target_arr[:, None]

    if isinstance(reference_feature, NBDfet):
        ref_arr = np.asarray(reference_feature.data, dtype=np.float64)
    else:
        ref_arr = np.asarray(reference_feature, dtype=np.float64)
    if ref_arr.ndim == 1:
        ref_arr = ref_arr[:, None]

    if target_arr.shape[1] != ref_arr.shape[1]:
        raise ValueError(
            f"target/reference feature column count mismatch: "
            f"{target_arr.shape[1]} vs {ref_arr.shape[1]}"
        )
    n_features = target_arr.shape[1]

    circular_set = set(int(c) for c in circular_columns)
    if linear_columns is None:
        linear_set = set(range(n_features)) - circular_set
    else:
        linear_set = set(int(c) for c in linear_columns)
    if circular_set & linear_set:
        raise ValueError(
            f"columns {circular_set & linear_set} are listed as both "
            f"circular and linear"
        )

    # Compute per-bin manifold means in both sessions
    target_stats = behavioural_manifold_stats(
        target_arr, target_xyz, active_mask=target_active,
        body_marker=body_marker, head_marker=head_marker,
        height_marker=height_marker, height_dim=height_dim,
        n_bins=n_bins, vel_edges=vel_edges, height_edges=height_edges,
    )
    ref_stats = behavioural_manifold_stats(
        ref_arr, reference_xyz, active_mask=reference_active,
        body_marker=body_marker, head_marker=head_marker,
        height_marker=height_marker, height_dim=height_dim,
        n_bins=n_bins, vel_edges=vel_edges, height_edges=height_edges,
    )

    min_count = int(round(minimum_occupancy_seconds * target_sr))

    offsets = np.full(n_features, np.nan, dtype=np.float64)
    columns_to_apply = sorted(linear_set | circular_set)
    for f in columns_to_apply:
        is_circular = f in circular_set
        # Per-bin valid mask
        nnz = (
            np.isfinite(target_stats.mean[..., f]) &
            np.isfinite(ref_stats.mean[..., f]) &
            (target_stats.count > min_count) &
            (ref_stats.count > min_count)
        )
        if not nnz.any():
            continue
        t_vals = target_stats.mean[..., f][nnz]
        r_vals = ref_stats.mean[..., f][nnz]
        if is_circular:
            mzd = np.asarray(circ_dist(t_vals, r_vals))
        else:
            mzd = t_vals - r_vals
        mshift = float(np.nanmedian(mzd))
        offsets[f] = mshift

        # Apply
        zero_mask = target_arr[:, f] == 0.0
        if is_circular:
            target_arr[:, f] = np.asarray(
                circ_dist(target_arr[:, f], np.full_like(target_arr[:, f], mshift))
            )
        else:
            target_arr[:, f] = target_arr[:, f] - mshift
        target_arr[zero_mask, f] = 0.0

    if return_offsets:
        return target_arr, offsets
    return target_arr
