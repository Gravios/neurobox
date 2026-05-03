"""
neurobox.analysis.placefields.egocentric
==========================================
Egocentric ratemaps — firing rate as a function of where the place-field
centre appears in the rat's head-fixed frame.

Ports of:

* :file:`MTA/analysis/compute_ego_ratemap.m`         → :func:`compute_ego_ratemap`
* :file:`MTA/analysis/compute_egocentric_ratemap.m`  → :func:`compute_ego_ratemap` (alias)

Plus the **conditioned** variants from MTA — these are the same
egocentric ratemap but restricted to subsets of frames where various
additional features fall in specified bins:

* :file:`compute_egohba_ratemap.m` (theta-phase × head-body angle)
* :file:`compute_egohbahvl_ratemap.m` (theta-phase × HBA × head velocity-lateral)
* :file:`compute_egohrl_ratemap.m` (theta-phase × head roll)
* :file:`compute_egohvf_ratemap.m` (theta-phase × head velocity-forward)
* :file:`compute_egohvl_ratemap.m` (theta-phase × head velocity-lateral)
* :file:`compute_egoradial_ratemap.m` (theta-phase × radial position)
* :file:`compute_egothp_ratemap.m` (theta-phase only)

Rather than port 8 near-duplicate functions, this module provides:

* :func:`compute_ego_ratemap` — the canonical egocentric ratemap.
* :func:`compute_ego_ratemap_conditioned` — runs the same algorithm
  separately for each cell of an N-D conditioning grid.
  Reproduce any MTA variant by passing the right
  ``conditioning_features`` and ``conditioning_bins``.

What's the egocentric coordinate?
---------------------------------
For each frame ``t``:

  1. Build the rat's head-fixed orthonormal basis (``hvec``) from
     ``nose - hcom``, optionally rotated by a per-subject yaw correction.
  2. Compute the **world-frame vector** from the rat's head-of-mass
     to the unit's place-field centre: ``mxp - hcom_position[t]``.
  3. **Rotate** that world-frame vector into the rat's head frame
     using ``hvec[t]``: that's the egocentric position
     ``ego_xy[t] = (mxp - hcom_pos[t]) @ hvec[t]``.

The result is a per-frame 2-D position in head-coords; the egocentric
ratemap is then just the standard ratemap on this trajectory.

Reproducing MTA variants
------------------------
The MATLAB ``compute_egohba_ratemap.m`` sweeps a 3×3 grid (3 theta-phase
bins × 3 head-body-angle bins), training a separate egocentric ratemap
for each cell.  In Python::

    phz = theta_phase(...)                                    # (T,)
    hba = head_body_angle(...)                                # (T,)

    grid = compute_ego_ratemap_conditioned(
        spk, aug, units=range(50), pft=pft,
        conditioning_features = {"phz": phz, "hba": hba},
        conditioning_bins     = {"phz": np.linspace(0.5, 2*pi-0.5, 4),
                                 "hba": [-1.2, -0.2, 0.2, 1.2]},
        bin_size  = 20,
        boundary  = [(-410, 410), (-410, 410)],
        samplerate= 30.0,
    )
    # grid["phz=0,hba=1"] = PlaceFieldResult for phase-bin 0 × hba-bin 1
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from neurobox.analysis.spatial.place_fields import (
    place_field, PlaceFieldResult,
)
from neurobox.dtype.epoch import NBEpoch
from neurobox.dtype.spikes import NBSpk
from neurobox.dtype.xyz import NBDxyz

from .directional_zones import field_centres_from_result


__all__ = [
    "egocentric_position",
    "compute_ego_ratemap",
    "compute_ego_ratemap_conditioned",
]


# ─────────────────────────────────────────────────────────────────── #
# Helpers                                                             #
# ─────────────────────────────────────────────────────────────────── #

def _build_head_basis(
    xyz:                 NBDxyz,
    head_yaw_correction: float = 0.0,
    nose_marker:         str = "nose",
    head_centre_marker:  str = "hcom",
) -> np.ndarray:
    """Build the per-frame head-fixed orthonormal basis.

    Returns ``hvec`` of shape ``(T, 2, 2)`` where the first column is
    the forward direction (nose − hcom, normalised) and the second
    column is its 90° counter-clockwise rotation, with both rotated
    by *head_yaw_correction* radians.
    """
    nose_idx = xyz.model.index(nose_marker)
    hcom_idx = xyz.model.index(head_centre_marker)

    forward = xyz.data[:, nose_idx, :2] - xyz.data[:, hcom_idx, :2]   # (T, 2)
    norms = np.linalg.norm(forward, axis=1, keepdims=True)
    safe = np.where(norms > 0, norms, 1.0)
    forward = forward / safe                                            # (T, 2)
    perp = np.column_stack([-forward[:, 1], forward[:, 0]])             # 90° CCW
    hvec = np.stack([forward, perp], axis=2)                            # (T, 2, 2)

    # Apply yaw correction
    if head_yaw_correction != 0.0:
        c, s = np.cos(head_yaw_correction), np.sin(head_yaw_correction)
        R = np.array([[c, -s], [s, c]], dtype=np.float64)
        hvec = np.einsum("tij,jk->tik", hvec, R)
    return hvec


def egocentric_position(
    xyz:                  NBDxyz,
    field_centre:         np.ndarray,
    *,
    head_yaw_correction:  float = 0.0,
    head_centre_correction: tuple[float, float] = (0.0, 0.0),
    head_centre_marker:   str = "hcom",
    nose_marker:          str = "nose",
) -> np.ndarray:
    """Per-frame ``(T, 2)`` position of *field_centre* in the rat's head frame.

    The MATLAB pipeline computes this as::

        ego_xy[t] = (mxp - hcom[t]) @ hvec[t] + head_centre_correction

    where ``hvec[t]`` is the head-fixed orthonormal basis at frame
    ``t``, derived from ``nose − hcom`` with yaw correction.

    Parameters
    ----------
    xyz:
        Augmented :class:`NBDxyz` containing both *head_centre_marker*
        (typically ``hcom``) and *nose_marker* (typically ``nose``).
    field_centre:
        ``(2,)`` xy of the place-field centre in world coords.
    head_yaw_correction:
        Per-subject head-yaw rotation applied to the head basis,
        in radians.  Default 0.
    head_centre_correction:
        Optional ``(dx, dy)`` translation in head-frame coords added
        after rotation.  Default ``(0, 0)``.

    Returns
    -------
    ego_xy : np.ndarray, shape ``(T, 2)``
        The place-field centre's xy position in the head-fixed frame
        at every frame of *xyz*.
    """
    field_centre = np.asarray(field_centre, dtype=np.float64).ravel()
    if field_centre.shape != (2,):
        raise ValueError(f"field_centre must be (2,); got {field_centre.shape}")

    hvec = _build_head_basis(xyz, head_yaw_correction,
                              nose_marker, head_centre_marker)
    hcom_idx = xyz.model.index(head_centre_marker)
    hcom_pos = xyz.data[:, hcom_idx, :2]                                 # (T, 2)

    relative = field_centre[None, :] - hcom_pos                          # (T, 2)
    # ego_xy[t, k] = sum_j relative[t, j] * hvec[t, j, k]
    ego = np.einsum("tj,tjk->tk", relative, hvec)
    ego = ego + np.asarray(head_centre_correction, dtype=np.float64)[None, :]
    return ego


# ─────────────────────────────────────────────────────────────────── #
# Canonical egocentric ratemap                                         #
# ─────────────────────────────────────────────────────────────────── #

def compute_ego_ratemap(
    spikes:               NBSpk,
    xyz:                  NBDxyz,
    units:                Sequence[int],
    *,
    pft:                  PlaceFieldResult,
    bin_size:             float | Sequence[float] = 20.0,
    boundary:             Sequence[Sequence[float]] = ((-410, 410), (-410, 410)),
    smoothing_sigma:      float | Sequence[float] | None = 2.0,
    samplerate:           float | None = None,
    state:                NBEpoch | None = None,
    head_yaw_correction:  float = 0.0,
    head_centre_correction: tuple[float, float] = (0.0, 0.0),
    head_centre_marker:   str = "hcom",
    nose_marker:          str = "nose",
    n_iter:               int = 1,
    bootstrap_fraction:   float = 0.0,
    rng:                  np.random.Generator | int | None = None,
    **place_field_kwargs,
) -> dict[int, PlaceFieldResult]:
    """Egocentric ratemap — firing as a function of place-field-centre in head frame.

    Port of :file:`MTA/analysis/compute_ego_ratemap.m` /
    :file:`compute_egocentric_ratemap.m`.

    For each unit in *units*, computes a separate 2-D ratemap where
    the position axes are the (x, y) coordinates of that unit's
    place-field centre **as seen in the rat's head-fixed frame**.
    Returns a dict mapping unit ID to its :class:`PlaceFieldResult`.

    Parameters
    ----------
    spikes:
        Spike times + clusters.
    xyz:
        Augmented :class:`NBDxyz` containing both *head_centre_marker*
        and *nose_marker*.
    units:
        Unit IDs to compute for.
    pft:
        2-D *world-frame* placefield result used to look up each unit's
        place-field centre.
    bin_size, boundary, smoothing_sigma:
        Spatial-binning parameters for the egocentric ratemap.  Default
        ``20`` mm bins on a ±410 mm field — matches MATLAB.
    samplerate:
        Resample target.  ``None`` keeps *xyz* rate.
    state:
        Optional :class:`NBEpoch` restricting which frames count
        (typically a theta-period mask, minus rear/groom/sit).  Default
        None → use all frames.
    head_yaw_correction, head_centre_correction:
        Per-subject corrections — see :func:`egocentric_position`.
    head_centre_marker, nose_marker:
        Marker names.  Defaults match MATLAB.
    n_iter, bootstrap_fraction, rng:
        Forwarded to :func:`place_field` for bootstrap iteration.
    **place_field_kwargs:
        Any other :func:`place_field` keyword (``min_occupancy``,
        ``min_spikes``, ``halfsample``, etc.).

    Returns
    -------
    dict[unit_id → PlaceFieldResult]
        One result per unit.  Unit IDs whose centre couldn't be
        located in *pft* are silently dropped.
    """
    fs = float(samplerate) if samplerate is not None else float(xyz.samplerate)

    # Resolve all unit centres up front (drops any that couldn't be located)
    centres = field_centres_from_result(pft, units)
    valid_units = []
    valid_centres = []
    for k, uid in enumerate(units):
        if np.isfinite(centres[k]).all():
            valid_units.append(int(uid))
            valid_centres.append(centres[k])

    out: dict[int, PlaceFieldResult] = {}
    for uid, centre in zip(valid_units, valid_centres):
        ego_xy = egocentric_position(
            xyz, centre,
            head_yaw_correction = head_yaw_correction,
            head_centre_correction = head_centre_correction,
            head_centre_marker = head_centre_marker,
            nose_marker = nose_marker,
        )
        out[uid] = place_field(
            spikes, ego_xy, units=[uid],
            bin_size = bin_size,
            boundary = boundary,
            samplerate = fs,
            state = state,
            smoothing_sigma = smoothing_sigma,
            n_iter = n_iter,
            bootstrap_fraction = bootstrap_fraction,
            rng = rng,
            **place_field_kwargs,
        )
    return out


# ─────────────────────────────────────────────────────────────────── #
# Conditioned variant — the egohba / egohvl / egohvf / etc. family    #
# ─────────────────────────────────────────────────────────────────── #

def compute_ego_ratemap_conditioned(
    spikes:                 NBSpk,
    xyz:                    NBDxyz,
    units:                  Sequence[int],
    *,
    pft:                    PlaceFieldResult,
    conditioning_features:  Mapping[str, np.ndarray],
    conditioning_bins:      Mapping[str, Sequence[float]],
    bin_size:               float | Sequence[float] = 20.0,
    boundary:               Sequence[Sequence[float]] = ((-410, 410), (-410, 410)),
    smoothing_sigma:        float | Sequence[float] | None = 2.0,
    samplerate:             float | None = None,
    base_state:             NBEpoch | None = None,
    head_yaw_correction:    float = 0.0,
    head_centre_correction: tuple[float, float] = (0.0, 0.0),
    head_centre_marker:     str = "hcom",
    nose_marker:            str = "nose",
    n_iter:                 int = 1,
    bootstrap_fraction:     float = 0.0,
    rng:                    np.random.Generator | int | None = None,
    **place_field_kwargs,
) -> dict[tuple[int, ...], dict[int, PlaceFieldResult]]:
    """Egocentric ratemap conditioned on an N-D feature grid.

    Port of the MTA conditioned-egocentric family (``compute_egohba_ratemap``,
    ``compute_egohvl_ratemap``, ``compute_egohvf_ratemap``,
    ``compute_egohrl_ratemap``, ``compute_egoradial_ratemap``,
    ``compute_egothp_ratemap``).

    For each cell in the conditioning grid, restricts frames to those
    where every feature falls in the corresponding bin and computes
    a fresh egocentric ratemap.

    Parameters
    ----------
    conditioning_features:
        Dict of feature name → ``(T,)`` array of per-frame values.
    conditioning_bins:
        Dict of feature name → bin edges (``len = n_bins + 1``).
        Frames where the feature falls outside any bin are dropped.
    base_state:
        Additional epoch mask applied to **every** cell.  Typically a
        theta-period mask minus rear/groom/sit (matches MATLAB).
    bin_size, boundary, smoothing_sigma, samplerate, head_*,
    n_iter, bootstrap_fraction, rng, **place_field_kwargs:
        See :func:`compute_ego_ratemap`.

    Returns
    -------
    dict[grid_index_tuple → dict[unit_id → PlaceFieldResult]]
        Outer dict is keyed by the per-feature bin indices (in the
        order they were provided in *conditioning_features*).
        Inner dict matches :func:`compute_ego_ratemap` output.

        For example, with ``conditioning_features = {"phz": ..., "hba": ...}``
        and 3 bins each, the keys are ``(0, 0)``, ``(0, 1)``, ..., ``(2, 2)``
        and the *first* index corresponds to ``phz`` (insert order).

    Notes
    -----
    The feature arrays must be at the same samplerate as *xyz*.  In
    MATLAB this was enforced by resampling everything to *xyz* before
    bin-restricting; here it is the caller's responsibility.
    """
    if not conditioning_features:
        raise ValueError(
            "conditioning_features must not be empty; "
            "use compute_ego_ratemap for the unconditioned variant."
        )
    feature_names = list(conditioning_features.keys())
    for name in feature_names:
        if name not in conditioning_bins:
            raise ValueError(
                f"feature {name!r} has no entry in conditioning_bins"
            )

    fs = float(samplerate) if samplerate is not None else float(xyz.samplerate)

    # Resolve unit centres upfront
    centres = field_centres_from_result(pft, units)
    valid_pairs = [(int(uid), centres[k])
                   for k, uid in enumerate(units)
                   if np.isfinite(centres[k]).all()]

    # Pre-compute per-frame bin indices for each feature.  Frames
    # outside any bin get index -1 and are dropped.
    T = xyz.data.shape[0]
    feature_inds: dict[str, np.ndarray] = {}
    feature_n_bins: dict[str, int] = {}
    for name in feature_names:
        edges = np.asarray(conditioning_bins[name], dtype=np.float64)
        vals  = np.asarray(conditioning_features[name], dtype=np.float64).ravel()
        if vals.size != T:
            raise ValueError(
                f"feature {name!r} length {vals.size} ≠ xyz length {T}"
            )
        idx = np.digitize(vals, edges, right=False) - 1
        idx[(idx < 0) | (idx >= len(edges) - 1)] = -1
        idx[~np.isfinite(vals)] = -1
        feature_inds[name] = idx
        feature_n_bins[name] = len(edges) - 1

    # Iterate over the conditioning grid
    grid_shape = tuple(feature_n_bins[name] for name in feature_names)
    out: dict[tuple[int, ...], dict[int, PlaceFieldResult]] = {}

    base_mask = None
    if base_state is not None:
        base_mask = base_state.to_mask(T)

    for grid_idx in np.ndindex(*grid_shape):
        # Build the per-frame mask: all features in their respective bins
        cell_mask = np.ones(T, dtype=bool)
        for name, gi in zip(feature_names, grid_idx):
            cell_mask &= (feature_inds[name] == gi)
        if base_mask is not None:
            cell_mask &= base_mask
        if not cell_mask.any():
            out[grid_idx] = {}
            continue
        # Convert to NBEpoch mask
        cell_state = NBEpoch(
            data       = cell_mask,
            samplerate = fs,
            mode       = "mask",
        )

        per_unit: dict[int, PlaceFieldResult] = {}
        for uid, centre in valid_pairs:
            ego_xy = egocentric_position(
                xyz, centre,
                head_yaw_correction = head_yaw_correction,
                head_centre_correction = head_centre_correction,
                head_centre_marker = head_centre_marker,
                nose_marker = nose_marker,
            )
            per_unit[uid] = place_field(
                spikes, ego_xy, units=[uid],
                bin_size = bin_size,
                boundary = boundary,
                samplerate = fs,
                state = cell_state,
                smoothing_sigma = smoothing_sigma,
                n_iter = n_iter,
                bootstrap_fraction = bootstrap_fraction,
                rng = rng,
                **place_field_kwargs,
            )
        out[grid_idx] = per_unit
    return out
