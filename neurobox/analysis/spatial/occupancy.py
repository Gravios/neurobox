"""
neurobox.analysis.spatial.occupancy
====================================
Spatial occupancy (dwell-time) maps.

Port of :file:`MTA/utilities/generate_occupancy_map.m` (Anton Sirota /
Justin Graboski).  This is the dwell-time half of a place-field
calculation, factored out for cases where you want occupancy alone
(e.g. visualising a trajectory's spatial coverage, normalising
spike-density estimates).

The companion :func:`neurobox.analysis.spatial.place_field` uses the
same binning + smoothing internally so the two stay numerically
consistent.

Convention
----------
* Inputs are :class:`NBDxyz` (position) optionally restricted by an
  :class:`NBEpoch` (state), or a plain ``(T, n_dims)`` numpy array
  with explicit samplerate.  The :class:`NBEpoch` is **resampled
  internally** to match position samplerate — the caller does not need
  to align them by hand.
* Bin sizes are in the **same units as the position data** (typically
  millimetres for motion-capture).
* Smoothing σ is in **bins** (matches the labbox convention).
* The output ``occupancy`` is in **seconds** of dwell time per bin.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.ndimage import gaussian_filter

from neurobox.dtype.epoch import NBEpoch
from neurobox.dtype.xyz   import NBDxyz


# ─────────────────────────────────────────────────────────────────────────── #
# Result container                                                             #
# ─────────────────────────────────────────────────────────────────────────── #

@dataclass
class OccupancyResult:
    """Output of :func:`occupancy_map`.

    Attributes
    ----------
    occupancy:
        ``(n_bins_dim1, n_bins_dim2, ...)`` array of dwell time in
        **seconds** per bin.  Smoothed if ``smoothing_sigma`` was set.
        Bins below the occupancy threshold are NaN.
    occupancy_raw:
        Same shape, **unsmoothed** dwell times in seconds — useful when
        the caller wants to apply their own smoothing kernel or
        threshold logic.
    bin_edges:
        List of ``(n_bins+1,)`` arrays — one per dimension.
    bin_centres:
        List of ``(n_bins,)`` arrays — one per dimension.
    occupancy_mask:
        Boolean array of the same shape as ``occupancy`` — ``True``
        where dwell time exceeds ``min_occupancy``.
    samplerate:
        Position sample rate used for the calculation.
    """

    occupancy:        np.ndarray
    occupancy_raw:    np.ndarray
    bin_edges:        list[np.ndarray]
    bin_centres:      list[np.ndarray]
    occupancy_mask:   np.ndarray
    samplerate:       float


# ─────────────────────────────────────────────────────────────────────────── #
# Bin construction                                                             #
# ─────────────────────────────────────────────────────────────────────────── #

def _build_bins(
    boundary: Sequence[Sequence[float]],
    bin_size: float | Sequence[float],
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    """Build per-axis edges, centres, and bin counts from boundary + size.

    Mirrors the labbox formula:

        Nbin = round(|boundary[i,1] - boundary[i,0]| / bin_size[i])
    """
    n_dims = len(boundary)
    bd = np.asarray(boundary, dtype=np.float64)
    if bd.shape != (n_dims, 2):
        raise ValueError(f"boundary must have shape (n_dims, 2); got {bd.shape}")
    if np.any(bd[:, 1] <= bd[:, 0]):
        raise ValueError("each boundary row must satisfy max > min")

    if np.isscalar(bin_size):
        bs = np.full(n_dims, float(bin_size))
    else:
        bs = np.asarray(bin_size, dtype=np.float64).ravel()
        if bs.size != n_dims:
            raise ValueError(
                f"bin_size must be scalar or length {n_dims}; got {bs.size}"
            )

    extent = bd[:, 1] - bd[:, 0]
    n_bins = np.maximum(np.round(extent / bs).astype(np.int64), 1)

    edges:   list[np.ndarray] = []
    centres: list[np.ndarray] = []
    for i in range(n_dims):
        # Span the actual boundary by linspace (doesn't enforce exact bin_size,
        # but matches the labbox which also uses k = Nbin / extent and may
        # adjust the effective bin width slightly to fit Nbin bins).
        e = np.linspace(bd[i, 0], bd[i, 1], n_bins[i] + 1)
        edges.append(e)
        centres.append(0.5 * (e[:-1] + e[1:]))
    return edges, centres, n_bins


# ─────────────────────────────────────────────────────────────────────────── #
# Public API                                                                   #
# ─────────────────────────────────────────────────────────────────────────── #

def occupancy_map(
    xyz:                NBDxyz | np.ndarray,
    bin_size:           float | Sequence[float],
    boundary:           Sequence[Sequence[float]],
    state:              NBEpoch | None = None,
    samplerate:         float | None = None,
    smoothing_sigma:    float | Sequence[float] | None = None,
    min_occupancy:      float = 0.06,
) -> OccupancyResult:
    """Spatial occupancy (dwell-time) map.

    Port of :file:`MTA/utilities/generate_occupancy_map.m`.

    Parameters
    ----------
    xyz:
        Position data.  Either:

        * :class:`NBDxyz` with shape ``(T, n_markers, n_dims)`` — a
          single tracking marker is selected via the model's first
          marker, and only the first ``len(boundary)`` spatial
          dimensions are used.  Use :meth:`NBDxyz.sel` upstream for
          finer control.
        * Plain ndarray of shape ``(T, n_dims)`` — must match the
          dimensionality of ``boundary``.
    bin_size:
        Spatial bin size.  Scalar (isotropic) or sequence of length
        ``n_dims``.  Same units as the position data (typically mm).
    boundary:
        ``(n_dims, 2)`` sequence of ``[min, max]`` per dimension.
        Position samples falling outside the boundary are dropped.
    state:
        Optional :class:`NBEpoch` restricting which time samples are
        counted (e.g. ``stc['walk']``).  Resampled internally to match
        ``xyz`` samplerate.  ``None`` (default) → use all samples.
    samplerate:
        Required when ``xyz`` is a plain ndarray; ignored when ``xyz``
        is :class:`NBDxyz` (the samplerate is read off the object).
    smoothing_sigma:
        Gaussian smoothing σ in **bins** (matches the labbox
        convention).  Scalar (isotropic) or sequence of length
        ``n_dims``.  ``None`` (default) → no smoothing.
    min_occupancy:
        Bins with smoothed dwell time below this threshold (in
        **seconds**) are masked out (NaN in the output).  Default
        ``0.06`` matches the labbox default.

    Returns
    -------
    :class:`OccupancyResult`

    Examples
    --------
    Single-state map from an :class:`NBDxyz` and a state mask::

        from neurobox.analysis.spatial import occupancy_map
        result = occupancy_map(
            xyz, bin_size=30,
            boundary=[(-500, 500), (-500, 500)],
            state=stc['walk'],
            smoothing_sigma=2.2,
        )

    Plain ndarray input (e.g. for a synthetic trajectory)::

        result = occupancy_map(
            traj, bin_size=10,
            boundary=[(0, 100), (0, 100)],
            samplerate=120.0,
        )
    """
    # ── Resolve xyz to (T, n_dims) array + samplerate ─────────────────── #
    n_dims_target = len(boundary)
    if isinstance(xyz, NBDxyz):
        if samplerate is not None and samplerate != xyz.samplerate:
            raise ValueError(
                f"samplerate={samplerate} contradicts xyz.samplerate={xyz.samplerate}"
            )
        sr = float(xyz.samplerate)
        # Respect state via NBData period-selection (auto-resamples the epoch)
        if state is not None and not state.isempty():
            base = xyz[state]                       # (T, n_markers, n_dims)
        else:
            base = xyz.data
        # Reduce to (T, n_dims_target) — first marker, first n dims (matches
        # labbox: pos = sq(pos(:, trackingMarker, ismember('xyz', type)))).
        if base.ndim == 3:
            xyz_arr = base[:, 0, :n_dims_target]
        elif base.ndim == 2 and base.shape[1] >= n_dims_target:
            xyz_arr = base[:, :n_dims_target]
        else:
            raise ValueError(
                f"NBDxyz data has shape {base.shape}; cannot reduce to "
                f"(T, {n_dims_target})"
            )
    else:
        if samplerate is None:
            raise ValueError("samplerate is required when xyz is a plain ndarray")
        sr = float(samplerate)
        xyz_arr = np.asarray(xyz, dtype=np.float64)
        if xyz_arr.ndim != 2 or xyz_arr.shape[1] != n_dims_target:
            raise ValueError(
                f"xyz must have shape (T, {n_dims_target}); got {xyz_arr.shape}"
            )
        if state is not None and not state.isempty():
            mask = state.resample(sr).to_mask(xyz_arr.shape[0])
            xyz_arr = xyz_arr[mask]

    # ── Drop NaN/Inf rows (matches labbox `nniz`, except we keep zeros — #
    # the MATLAB nniz drops rows that are exactly zero, which is a       #
    # subtle bug there: a rat actually at the maze origin would be       #
    # silently excluded.  We follow the saner convention.)               #
    finite = np.isfinite(xyz_arr).all(axis=1)
    xyz_arr = xyz_arr[finite]

    # ── Build the bin grid ────────────────────────────────────────────── #
    bin_edges, bin_centres, n_bins = _build_bins(boundary, bin_size)

    # ── Histogram → counts → seconds ──────────────────────────────────── #
    counts, _ = np.histogramdd(xyz_arr, bins=bin_edges)
    occ_raw   = counts / sr

    # ── Smoothing ─────────────────────────────────────────────────────── #
    if smoothing_sigma is not None:
        if np.isscalar(smoothing_sigma):
            sigma = float(smoothing_sigma)
        else:
            sigma_arr = np.asarray(smoothing_sigma, dtype=np.float64).ravel()
            if sigma_arr.size != n_dims_target:
                raise ValueError(
                    f"smoothing_sigma must be scalar or length {n_dims_target}; "
                    f"got {sigma_arr.size}"
                )
            sigma = sigma_arr  # type: ignore[assignment]
        occ_smoothed = gaussian_filter(occ_raw, sigma=sigma, mode="constant")
    else:
        occ_smoothed = occ_raw.copy()

    # ── Mask under-occupied bins ──────────────────────────────────────── #
    occupancy_mask = occ_smoothed > min_occupancy
    occupancy = np.where(occupancy_mask, occ_smoothed, np.nan)

    return OccupancyResult(
        occupancy      = occupancy,
        occupancy_raw  = occ_raw,
        bin_edges      = bin_edges,
        bin_centres    = bin_centres,
        occupancy_mask = occupancy_mask,
        samplerate     = sr,
    )
