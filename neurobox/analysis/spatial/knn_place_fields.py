"""
neurobox.analysis.spatial.knn_place_fields
============================================
k-NN-based place-field rate maps with optional block-shuffle
bootstrap.

Port of:

* :file:`MTA/MTAAknnpfs_bs.m` (the inner algorithm) →
  :func:`knn_place_field`
* :file:`MTA/analysis/placefields/PlotKNNPF.m` → the kernel inside
  :func:`knn_place_field`
* :file:`MTA/analysis/placefields/compute_pfstats_bs.m` →
  :func:`compute_pfstats_bs`

What it computes
----------------
For each spatial bin centre, find the *k* nearest position samples
in the recording, and average the unit firing rate over those samples.
The resulting rate map is **smoother** than the standard
spike-count-divided-by-occupancy approach because it uses a
data-adaptive kernel: the effective bandwidth contracts in densely
visited regions and expands in sparsely visited regions.

The bootstrap is a **block-shuffle** of the firing-rate / position
correspondence.  At each iteration:

1. Split the recording into ``samples_per_block`` chunks.
2. Randomly partition the chunks into two halves.
3. Re-shuffle one half independently and re-compute the rate map.

This yields ``n_iter`` rate maps that share the same spatial
trajectory but have firing rates effectively decorrelated from
position — useful as a null distribution for spatial-information
significance testing.

Differences from MATLAB
-----------------------
* Uses :class:`scipy.spatial.cKDTree` for the k-NN search rather than
  the MATLAB-only ``sort(distw)`` brute-force computation.
* Returns a :class:`PlaceFieldResult` (the same dataclass used by
  :func:`neurobox.analysis.spatial.place_field`) so that downstream
  code (including :func:`place_field_stats`) works unchanged.
* The MATLAB `parfor` loop over bootstrap iterations is replaced by
  a serial loop with an optional `n_jobs` parallel backend (joblib).
  Most of the cost is the k-NN search which is shared across iters,
  so parallelism gives only a small speedup.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Sequence

import numpy as np
from scipy.spatial import cKDTree

from neurobox.analysis.spatial.place_fields import PlaceFieldResult
from neurobox.dtype.epoch import NBEpoch
from neurobox.dtype.spikes import NBSpk
from neurobox.dtype.ufr import NBDufr
from neurobox.dtype.xyz import NBDxyz


__all__ = [
    "knn_place_field",
    "compute_pfstats_bs",
    "PfsBsResult",
]


# ─────────────────────────────────────────────────────────────────────── #
# Helpers                                                                    #
# ─────────────────────────────────────────────────────────────────────── #

def _bin_centres(
    boundary:  Sequence[Sequence[float]],
    bin_size:  Sequence[float],
) -> tuple[list[np.ndarray], np.ndarray]:
    """Build bin-centre arrays + a flat ``(n_bins_total, n_dims)`` grid.

    Mirrors the cell-array `Bins` in MATLAB ``PlotKNNPF`` lines 12-17.
    """
    centres = []
    for (lo, hi), bd in zip(boundary, bin_size):
        n_bins = int(np.round((hi - lo) / bd))
        # Bin centres at lo + (i + 0.5) * bd
        c = lo + (np.arange(n_bins) + 0.5) * bd
        centres.append(c)
    grid = np.stack(np.meshgrid(*centres, indexing="ij"), axis=-1)
    flat_grid = grid.reshape(-1, len(centres))
    return centres, flat_grid


def _knn_rate_map(
    pos:               np.ndarray,        # (T, n_dims)
    ufr:               np.ndarray,        # (T,) firing rate
    grid:              np.ndarray,        # (n_bins_total, n_dims)
    *,
    n_neighbors:       int,
    dist_threshold:    float,
    stat:              str,
    tree:              Optional[cKDTree] = None,
) -> np.ndarray:
    """k-NN ratemap kernel — port of :file:`PlotKNNPF.m`.

    For each bin grid point, take the *k* nearest position samples
    and average their firing rate.  Bins whose median k-NN distance
    exceeds *dist_threshold* are masked (NaN).

    Parameters
    ----------
    pos:
        ``(T, n_dims)`` xy (or xyz) positions.
    ufr:
        ``(T,)`` unit firing rate at the same samplerate.
    grid:
        ``(n_bins_total, n_dims)`` flat grid of bin-centre coords.
    n_neighbors:
        *k* in k-NN.  Default 60 in MATLAB callers.
    dist_threshold:
        Bins where the median of the *k* nearest distances exceeds
        this are masked.  Units match *pos*.  Default 125 in MATLAB.
    stat:
        ``'mean'`` (default in :func:`knn_place_field`) or
        ``'median'`` (MATLAB default for ``stat_fun``).
    tree:
        Pre-built :class:`cKDTree` over *pos*.  If None, built here.

    Returns
    -------
    np.ndarray, shape ``(n_bins_total,)``
        Rate-map values at each bin centre, NaN where masked.
    """
    if tree is None:
        finite = np.isfinite(pos).all(axis=1)
        tree = cKDTree(pos[finite])
        ufr_ok = ufr[finite]
    else:
        ufr_ok = ufr

    if tree.n < n_neighbors:
        # Not enough samples for k-NN
        return np.full(grid.shape[0], np.nan)

    dists, idx = tree.query(grid, k=n_neighbors)
    # MATLAB uses `nanmedian` for the threshold check
    median_dist = np.median(dists, axis=1)             # (n_bins_total,)
    if stat == "mean":
        rate = np.mean(ufr_ok[idx], axis=1)
    elif stat == "median":
        rate = np.median(ufr_ok[idx], axis=1)
    else:
        raise ValueError(f"stat must be 'mean' or 'median'; got {stat!r}")

    rate = np.where(median_dist > dist_threshold, np.nan, rate)
    return rate


# ─────────────────────────────────────────────────────────────────────── #
# Public API                                                                 #
# ─────────────────────────────────────────────────────────────────────── #

def knn_place_field(
    spikes:           NBSpk,
    xyz:              NBDxyz | np.ndarray,
    units:            int | Sequence[int] | None = None,
    *,
    bin_size:         float | Sequence[float],
    boundary:         Sequence[Sequence[float]],
    state:            Optional[NBEpoch] = None,
    samplerate:       Optional[float] = 10.0,
    n_neighbors:      int = 60,
    dist_threshold:   float = 125.0,
    n_iter:           int = 1,
    block_size_seconds: float = 1.0,
    stat:             Literal["mean", "median"] = "mean",
    tracking_marker:  str = "spine_lower",
    rng:              Optional[np.random.Generator | int] = None,
) -> PlaceFieldResult:
    """k-NN-based place-field rate map with optional bootstrap.

    Port of :file:`MTA/MTAAknnpfs_bs.m`.

    Parameters
    ----------
    spikes:
        :class:`NBSpk` with cluster IDs and spike times.
    xyz:
        :class:`NBDxyz` (will be subset to *tracking_marker*) or a
        plain ``(T, n_dims)`` position array.
    units:
        Cluster IDs to compute.  ``None`` → all clusters in *spikes*.
    bin_size:
        Spatial bin size, scalar or per-dim.  MATLAB default 20 mm.
    boundary:
        ``[[xlo, xhi], [ylo, yhi], ...]``.  Position samples outside
        are still used (KDTree handles it), but bins outside are not
        produced.
    state:
        Behavioural-state mask :class:`NBEpoch`.  Only samples inside
        contribute to the rate map.  ``None`` → use all samples.
    samplerate:
        Resample target for the position trace (and corresponding
        firing rate).  MATLAB default 10 Hz makes the k-NN tractable
        for long sessions.  Pass ``None`` to keep the native rate.
    n_neighbors:
        *k* in k-NN.  MATLAB default 60.
    dist_threshold:
        Bins whose median k-NN distance exceeds this are masked.
        MATLAB default 125 (mm).
    n_iter:
        Number of bootstrap resampling iterations.  Iter 0 is always
        the deterministic map; iters 1..n_iter-1 are block-shuffled.
        Pass 1 for no bootstrap.  MATLAB default 1.
    block_size_seconds:
        Block size for the shuffle, in seconds.  MATLAB default 1 s
        means each block is roughly the autocorrelation timescale of
        slow position drift, so blocks are quasi-independent.
    stat:
        Statistic across the k neighbours.  ``'mean'`` (default) or
        ``'median'`` (MATLAB default).
    tracking_marker:
        Marker name when *xyz* is an :class:`NBDxyz`.  Default
        ``'spine_lower'``.
    rng:
        For reproducibility.

    Returns
    -------
    PlaceFieldResult
        Same shape as the standard
        :func:`neurobox.analysis.spatial.place_field` output:
        ``rate_map`` is ``(*bin_shape, n_units, n_iter)``.
    """
    rng = (np.random.default_rng(rng) if not isinstance(rng, np.random.Generator)
           else rng)

    # Resolve positions
    if isinstance(xyz, NBDxyz):
        midx = xyz.model.index(tracking_marker)
        n_dims = len(boundary)
        if hasattr(xyz, "samplerate"):
            xyz_sr = float(xyz.samplerate)
        else:
            xyz_sr = float(samplerate or 1.0)
        pos_full = xyz.data[:, midx, :n_dims]      # (T, n_dims)
    else:
        pos_full = np.asarray(xyz, dtype=np.float64)
        if pos_full.ndim == 1:
            pos_full = pos_full[:, None]
        xyz_sr = float(samplerate or 1.0)

    n_dims = pos_full.shape[1]

    # Resolve bin_size as per-dim
    if np.isscalar(bin_size):
        bin_size_arr = np.full(n_dims, float(bin_size))
    else:
        bin_size_arr = np.asarray(bin_size, dtype=np.float64).ravel()
        if bin_size_arr.size != n_dims:
            raise ValueError(
                f"bin_size length {bin_size_arr.size} ≠ n_dims {n_dims}"
            )

    # Build the bin grid
    centres, flat_grid = _bin_centres(boundary, bin_size_arr)
    bin_shape = tuple(c.size for c in centres)
    n_bins_total = int(np.prod(bin_shape))

    # Build / resample firing-rate time series at the working samplerate
    work_sr = float(samplerate or xyz_sr)
    duration_sec = pos_full.shape[0] / xyz_sr

    if units is None:
        unit_ids = sorted(spikes.by_unit().keys())
    elif isinstance(units, (int, np.integer)):
        unit_ids = [int(units)]
    else:
        unit_ids = list(int(u) for u in units)

    ufr_obj = NBDufr.compute(
        spikes, samplerate=work_sr, duration_sec=duration_sec,
        units=unit_ids, window=0.8, mode="boxcar",
    )
    # ufr_obj.data shape (T_work, n_units)
    ufr_full = ufr_obj.data
    if ufr_full.shape[1] != len(unit_ids):
        # Safety check
        unit_ids_present = list(ufr_obj.unit_ids)
    else:
        unit_ids_present = unit_ids

    # Resample position to work_sr
    pos_work = _resample_position(pos_full, xyz_sr, work_sr)
    # Crop both to the shorter
    n_work = min(pos_work.shape[0], ufr_full.shape[0])
    pos_work = pos_work[:n_work]
    ufr_full = ufr_full[:n_work]

    # Apply the state mask
    if state is not None:
        if hasattr(state, "to_mask"):
            mask = state.resample(work_sr).to_mask(n_work) \
                   if float(state.samplerate) != work_sr \
                   else state.to_mask(n_work)
        else:
            mask = np.asarray(state, dtype=bool)[:n_work]
        pos_work = pos_work[mask]
        ufr_full = ufr_full[mask]

    # Drop non-finite frames
    finite = np.isfinite(pos_work).all(axis=1)
    pos_work = pos_work[finite]
    ufr_full = ufr_full[finite]

    # Build the KDTree once over the full position trace — re-used
    # across units (the same nearest-neighbour mapping applies to all
    # units' firing rates).
    tree = cKDTree(pos_work) if pos_work.shape[0] >= n_neighbors \
            else None

    # Allocate output
    rate_map_flat = np.full(
        (n_bins_total, len(unit_ids), n_iter), np.nan, dtype=np.float64,
    )

    # Iter 0 — deterministic
    if tree is not None:
        for ui in range(len(unit_ids)):
            rate_map_flat[:, ui, 0] = _knn_rate_map(
                pos_work, ufr_full[:, ui], flat_grid,
                n_neighbors=n_neighbors, dist_threshold=dist_threshold,
                stat=stat, tree=tree,
            )

    # Iters 1..n_iter-1 — block-shuffle the firing rate / position
    # correspondence.  We build a random permutation of "block IDs"
    # and apply it to ufr to break the spike–position pairing.
    if n_iter > 1 and tree is not None:
        T_work = pos_work.shape[0]
        samples_per_block = max(1, int(round(work_sr * block_size_seconds)))
        n_blocks = T_work // samples_per_block
        trim = T_work - n_blocks * samples_per_block
        if trim > 0:
            # MATLAB drops the trim, so do we
            pos_trim = pos_work[:T_work - trim]
            ufr_trim = ufr_full[:T_work - trim]
            tree_trim = cKDTree(pos_trim)
        else:
            pos_trim, ufr_trim, tree_trim = pos_work, ufr_full, tree

        block_starts = np.arange(n_blocks) * samples_per_block
        block_idx = (
            block_starts[:, None] + np.arange(samples_per_block)[None, :]
        )                                          # (n_blocks, spb)

        for it in range(1, n_iter):
            perm = rng.permutation(n_blocks)
            shuffled_idx = block_idx[perm].ravel()
            ufr_shuf = ufr_trim[shuffled_idx]
            for ui in range(len(unit_ids)):
                rate_map_flat[:, ui, it] = _knn_rate_map(
                    pos_trim, ufr_shuf[:, ui], flat_grid,
                    n_neighbors=n_neighbors, dist_threshold=dist_threshold,
                    stat=stat, tree=tree_trim,
                )

    # Reshape to (*bin_shape, n_units, n_iter)
    rate_map = rate_map_flat.reshape(bin_shape + (len(unit_ids), n_iter))
    occupancy_mask = np.isfinite(rate_map[..., 0, 0])

    # Pack into PlaceFieldResult — most fields are best-effort
    # placeholders since the KNN method doesn't compute them.
    bin_edges = []
    for c, bd in zip(centres, bin_size_arr):
        # Centres → edges
        bin_edges.append(np.r_[c - bd / 2, c[-1] + bd / 2])

    n_units = len(unit_ids)
    return PlaceFieldResult(
        rate_map        = rate_map,
        occupancy       = np.full(bin_shape, np.nan, dtype=np.float64),
        spike_count     = np.full(rate_map.shape, np.nan, dtype=np.float64),
        occupancy_mask  = occupancy_mask,
        bin_edges       = bin_edges,
        bin_centres     = centres,
        spatial_info    = np.full((n_units, n_iter), np.nan,
                                    dtype=np.float64),
        sparsity        = np.full((n_units, n_iter), np.nan,
                                    dtype=np.float64),
        mean_rate       = np.array([np.nanmean(ufr_full[:, ui])
                                      for ui in range(n_units)],
                                     dtype=np.float64).reshape(-1, 1)
                            * np.ones((1, n_iter)),
        unit_ids        = np.asarray(unit_ids, dtype=np.int32),
        n_spikes        = np.array([
            int(((ufr_full[:, ui] * (1.0 / work_sr)).sum()))
            for ui in range(n_units)
        ], dtype=np.int64),
        samplerate      = work_sr,
        skaggs_correct  = False,
    )


def _resample_position(
    pos:    np.ndarray,
    src_sr: float,
    tgt_sr: float,
) -> np.ndarray:
    """Linear-interp resample a position trace.

    Decimation-style; if *tgt_sr* > *src_sr*, just upsamples by
    interpolation.  No anti-aliasing — fine for kinematic time-series
    that are already low-pass-filtered upstream.
    """
    if abs(src_sr - tgt_sr) < 1e-9:
        return pos.copy()
    T = pos.shape[0]
    duration = T / src_sr
    n_tgt = int(round(duration * tgt_sr))
    t_src = np.arange(T) / src_sr
    t_tgt = np.arange(n_tgt) / tgt_sr
    out = np.empty((n_tgt, pos.shape[1]), dtype=np.float64)
    for d in range(pos.shape[1]):
        out[:, d] = np.interp(t_tgt, t_src, pos[:, d])
    return out


# ─────────────────────────────────────────────────────────────────────── #
# compute_pfstats_bs                                                         #
# ─────────────────────────────────────────────────────────────────────── #

@dataclass
class PfsBsResult:
    """Output of :func:`compute_pfstats_bs`.

    Per-state, per-unit, per-bootstrap-iter peak-patch summary stats.

    Attributes
    ----------
    states : tuple[str, ...]
    unit_ids : np.ndarray, shape ``(n_units,)``
    peak_patch_area : np.ndarray, shape ``(n_states, n_iter, n_units)``
        Area (mm²) of the patch with the highest peak firing rate
        per (state, iter, unit).
    peak_patch_com : np.ndarray, shape ``(n_states, n_iter, n_units, n_dims)``
        Centre of mass of that same patch.  NaN if the unit has no
        patch above threshold.
    peak_patch_rate : np.ndarray, shape ``(n_states, n_iter, n_units)``
        Peak firing rate within that patch.  NaN if no patch.
    pf_stats : list[list[UnitStats]]
        Full per-unit stats from
        :func:`neurobox.analysis.spatial.place_field_stats`,
        outer index = state, inner = unit.
    """
    states:           tuple[str, ...]
    unit_ids:         np.ndarray
    peak_patch_area:  np.ndarray
    peak_patch_com:   np.ndarray
    peak_patch_rate:  np.ndarray
    pf_stats:         list


def compute_pfstats_bs(
    pf_per_state:    "Mapping[str, PlaceFieldResult]",
    *,
    units:           Optional[Sequence[int]] = None,
    max_n_patches:   int = 2,
    threshold_method: str = "percentile",
    threshold_pct:   float = 90.0,
) -> PfsBsResult:
    """Aggregate per-patch placefield stats across states and bootstrap iters.

    Port of :file:`MTA/analysis/placefields/compute_pfstats_bs.m`.

    Given a dict of ``{state_name: PlaceFieldResult}`` (typically from
    :func:`knn_place_field` called once per state with the same units
    and bins), this function:

    1. Calls :func:`place_field_stats` on each state's rate-map
       result with ``mode='per_iter'`` to get per-iter patch detection.
    2. For each (state, iter, unit), picks the patch with the
       **highest peak firing rate × area** (matching MATLAB
       ``find(max(x.patchPFR(...).*x.patchArea(...))==...)``).
    3. Collects that patch's area, COM and peak rate into a regular
       3-D array suitable for downstream group statistics.

    Parameters
    ----------
    pf_per_state:
        Dict-like mapping state name → :class:`PlaceFieldResult`.
        All entries must share the same ``unit_ids`` and the same
        ``rate_map.shape[-1]`` (number of bootstrap iterations).
    units:
        Subset to compute.  Default uses ``unit_ids`` of the first
        state's result.
    max_n_patches:
        Forwarded to :func:`place_field_stats`.
    threshold_method, threshold_pct:
        Forwarded to :func:`place_field_stats`.

    Returns
    -------
    PfsBsResult
    """
    # Lazy import to avoid circular
    from neurobox.analysis.spatial.place_field_stats import (
        place_field_stats,
    )

    states = tuple(pf_per_state.keys())
    if not states:
        raise ValueError("pf_per_state must be non-empty")
    first_pf = pf_per_state[states[0]]
    if units is None:
        unit_ids = list(first_pf.unit_ids)
    else:
        unit_ids = list(int(u) for u in units)

    n_states = len(states)
    n_iter = first_pf.rate_map.shape[-1]
    n_dims = first_pf.rate_map.ndim - 2
    n_units = len(unit_ids)

    peak_patch_area = np.full((n_states, n_iter, n_units), np.nan,
                                dtype=np.float64)
    peak_patch_com  = np.full((n_states, n_iter, n_units, n_dims), np.nan,
                                dtype=np.float64)
    peak_patch_rate = np.full((n_states, n_iter, n_units), np.nan,
                                dtype=np.float64)

    pf_stats_per_state = []
    for s_idx, state_name in enumerate(states):
        pf = pf_per_state[state_name]
        stats = place_field_stats(
            pf,
            units             = unit_ids,
            max_n_patches     = max_n_patches,
            threshold_method  = threshold_method,
            threshold_pct     = threshold_pct,
            mode              = "per_iter",
        )
        pf_stats_per_state.append(stats)

        for u_idx, unit_stats in enumerate(stats):
            patches = unit_stats.patches
            if not patches:
                continue
            for it in range(n_iter):
                # For each patch, grab its iter-it stats; pick the
                # patch with max (peak_rate * area) at this iteration.
                per_it_score = []
                for p in patches:
                    if p.peak_rate_iter is None or p.area_iter is None:
                        # n_iter == 1 case
                        peak = p.peak_rate
                        area = p.area
                        com  = p.center_of_mass
                    else:
                        peak = float(p.peak_rate_iter[it])
                        area = float(p.area_iter[it])
                        com  = (p.com_iter[it]
                                 if p.com_iter is not None
                                 else p.center_of_mass)
                    if np.isfinite(peak) and np.isfinite(area):
                        per_it_score.append((peak * area, peak, area, com))
                if not per_it_score:
                    continue
                _, peak, area, com = max(per_it_score, key=lambda x: x[0])
                peak_patch_area[s_idx, it, u_idx] = area
                peak_patch_rate[s_idx, it, u_idx] = peak
                peak_patch_com [s_idx, it, u_idx, :] = com

    return PfsBsResult(
        states           = states,
        unit_ids         = np.asarray(unit_ids, dtype=np.int32),
        peak_patch_area  = peak_patch_area,
        peak_patch_com   = peak_patch_com,
        peak_patch_rate  = peak_patch_rate,
        pf_stats         = pf_stats_per_state,
    )
