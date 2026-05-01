"""
neurobox.analysis.spatial.place_fields
=======================================
Place-field rate maps with optional bootstrap / half-sample / shuffle.

Port of :file:`MTA/@MTAApfs/MTAApfs.m` (the calculator subset) and
:file:`MTA/analysis/placefields/PlotPF.m` (Anton Sirota / Justin
Graboski).

What was kept
-------------
* The core algorithm: bin position, bin spikes, smooth both with an
  n-D gaussian, divide them, mask under-occupied bins.
* Bootstrap / half-sample / position-shuffle iteration for confidence
  intervals and null distributions.
* Skaggs (1993) spatial information and sparsity.

What was dropped
----------------
* The persistence layer (load-from-disk, "merge new units into existing
  data struct").  Use ``np.savez`` / ``pickle`` on the result.
* The three-way constructor dispatch on input type.
* The ``compute_pfs`` plug-point — write your own kernel if needed.
* The hard-coded 16 Hz position resample — use :meth:`NBDxyz.resample`
  upstream if you want to downsample.
* The ``MaxRate`` / ``maxRateInd`` / ``maxRatePos`` fields, which were
  allocated but never written by ``MTAApfs``.

What was corrected
------------------
* The Skaggs spatial-information formula is occupancy-weighted by
  default (matches Skaggs 1993).  The labbox version weights uniformly
  over occupied bins, which is incorrect — pass ``skaggs_correct=False``
  for bit-for-bit labbox compatibility.
* Same correction for sparsity.
* The ``nniz`` "drop zero rows" filter is replaced with
  ``np.isfinite`` only — the labbox version silently dropped any
  position sample that happened to be exactly zero on any axis.
* Edge cases that crashed the labbox (empty epoch, single iteration
  with ``halfsample=True``) are handled cleanly.

Conventions
-----------
* :class:`NBSpk`, :class:`NBDxyz`, and :class:`NBEpoch` are accepted
  natively.  Sample-rate alignment is automatic.
* ``bin_size`` is in **position-data units** (typically mm).
* ``smoothing_sigma`` is in **bins** (matches the labbox convention).
* Output ``rate_map`` is in **Hz** (spikes/second).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.ndimage import gaussian_filter

from neurobox.dtype.epoch  import NBEpoch
from neurobox.dtype.spikes import NBSpk
from neurobox.dtype.xyz    import NBDxyz

from .occupancy import _build_bins


# ─────────────────────────────────────────────────────────────────────────── #
# Result container                                                             #
# ─────────────────────────────────────────────────────────────────────────── #

@dataclass
class PlaceFieldResult:
    """Output of :func:`place_field`.

    Attributes
    ----------
    rate_map:
        ``(*bin_shape, n_units, n_iter)`` array of firing rates in Hz.
        NaN where the occupancy mask is False.  When ``n_units == 1``
        and ``n_iter == 1`` the trailing singleton axes are still
        present; squeeze them at the call site if convenient.
    occupancy:
        ``(*bin_shape,)`` smoothed occupancy in seconds.
    spike_count:
        ``(*bin_shape, n_units, n_iter)`` raw smoothed spike counts.
    occupancy_mask:
        ``(*bin_shape,)`` boolean — ``True`` where rate map is valid.
    bin_edges:
        List of ``(n_bins+1,)`` arrays — one per dimension.
    bin_centres:
        List of ``(n_bins,)`` arrays — one per dimension.
    spatial_info:
        ``(n_units, n_iter)`` Skaggs spatial information in bits/spike.
    sparsity:
        ``(n_units, n_iter)`` Skaggs sparsity in ``[0, 1]``.
    mean_rate:
        ``(n_units, n_iter)`` mean firing rate in Hz, computed within
        the occupancy mask.
    unit_ids:
        ``(n_units,)`` cluster IDs corresponding to the rate-map axes.
    n_spikes:
        ``(n_units,)`` raw spike count per unit (after state masking,
        before any bootstrap resampling).
    samplerate:
        Position sample rate used for the calculation.
    skaggs_correct:
        Whether the standard (occupancy-weighted) or labbox
        (uniform-weighted) Skaggs formula was used.
    """

    rate_map:        np.ndarray
    occupancy:       np.ndarray
    spike_count:     np.ndarray
    occupancy_mask:  np.ndarray
    bin_edges:       list[np.ndarray]
    bin_centres:     list[np.ndarray]
    spatial_info:    np.ndarray
    sparsity:        np.ndarray
    mean_rate:       np.ndarray
    unit_ids:        np.ndarray
    n_spikes:        np.ndarray
    samplerate:      float
    skaggs_correct:  bool


# ─────────────────────────────────────────────────────────────────────────── #
# Internals                                                                    #
# ─────────────────────────────────────────────────────────────────────────── #

def _spike_to_position_index(
    spike_times_sec: np.ndarray,
    pos_original_indices: np.ndarray,
    samplerate: float,
) -> np.ndarray:
    """Map spike times (seconds) to indices into the state-masked position array.

    The ``pos_original_indices`` are the original-time-base sample indices of
    the rows that survived state masking.  We round each spike time to its
    position-sample index, then ``searchsorted`` finds where it falls in
    the surviving-indices array.
    """
    if pos_original_indices.size == 0 or spike_times_sec.size == 0:
        return np.empty(0, dtype=np.int64)
    spike_pos_idx = np.round(spike_times_sec * samplerate).astype(np.int64)
    # Find each spike's position in the surviving-index array
    insertion = np.searchsorted(pos_original_indices, spike_pos_idx, side="left")
    # Keep only spikes whose insertion point lands on an exact match
    valid = (insertion < pos_original_indices.size) & np.isin(
        spike_pos_idx, pos_original_indices, assume_unique=False
    )
    return insertion[valid]


def _state_mask_and_compress(
    xyz: NBDxyz | np.ndarray,
    state: NBEpoch | None,
    samplerate: float | None,
    n_dims: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (compressed_xyz, original_indices, samplerate).

    ``original_indices`` tells us the original-time-base index of each
    surviving row of ``compressed_xyz`` — needed to map spike times into
    the compressed coordinate system.
    """
    if isinstance(xyz, NBDxyz):
        if samplerate is not None and samplerate != xyz.samplerate:
            raise ValueError(
                f"samplerate={samplerate} contradicts xyz.samplerate={xyz.samplerate}"
            )
        sr = float(xyz.samplerate)
        full = xyz.data
        if full.ndim == 3:
            full = full[:, 0, :n_dims]
        elif full.ndim == 2:
            full = full[:, :n_dims]
        else:
            raise ValueError(f"xyz.data has unexpected ndim={full.ndim}")
    else:
        if samplerate is None:
            raise ValueError("samplerate is required when xyz is a plain ndarray")
        sr = float(samplerate)
        full = np.asarray(xyz, dtype=np.float64)
        if full.ndim != 2 or full.shape[1] < n_dims:
            raise ValueError(
                f"xyz must have at least {n_dims} columns; got shape {full.shape}"
            )
        full = full[:, :n_dims]

    n_total = full.shape[0]
    if state is not None and not state.isempty():
        mask = state.resample(sr).to_mask(n_total)
    else:
        mask = np.ones(n_total, dtype=bool)
    # Also drop non-finite rows
    mask = mask & np.isfinite(full).all(axis=1)
    original_indices = np.flatnonzero(mask)
    compressed = full[mask]
    return compressed, original_indices, sr


def _bin_positions(
    pos: np.ndarray,
    bin_edges: list[np.ndarray],
    n_bins: np.ndarray,
) -> np.ndarray:
    """Return ``(K, n_dims)`` integer bin indices, with out-of-range rows dropped."""
    n_dims = len(bin_edges)
    if pos.shape[0] == 0:
        return np.empty((0, n_dims), dtype=np.int64)
    idx = np.empty_like(pos, dtype=np.int64)
    for i in range(n_dims):
        # np.digitize: 1-indexed; bins=edges[i] (ascending)
        # We want 0-indexed [0, n_bins[i]) bins.
        d = np.digitize(pos[:, i], bin_edges[i]) - 1
        d = np.where(d == n_bins[i], n_bins[i] - 1, d)  # last edge = closed
        idx[:, i] = d
    in_range = np.all((idx >= 0) & (idx < n_bins[None, :]), axis=1)
    return idx[in_range]


def _accumarray_nd(
    indices: np.ndarray,
    shape: Sequence[int],
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """numpy equivalent of MATLAB ``accumarray(indices, weights, shape)``."""
    out = np.zeros(shape, dtype=np.float64)
    if indices.shape[0] == 0:
        return out
    flat = np.ravel_multi_index(tuple(indices.T), shape)
    if weights is None:
        np.add.at(out.ravel(), flat, 1.0)
    else:
        np.add.at(out.ravel(), flat, weights)
    return out


def _shuffle_position_blocks(
    pos: np.ndarray,
    block_size_samples: int,
    rng: np.random.Generator,
    dims_to_shuffle: Sequence[int],
) -> np.ndarray:
    """Block-permute ``pos`` along axis 0 for the selected dims only.

    Trailing samples that don't fill a full block are kept in place
    (matches the labbox semantics: ``size(p)/blockSize`` is always integer
    after the trim earlier in MTAApfs).
    """
    n_t = pos.shape[0]
    n_full = (n_t // block_size_samples) * block_size_samples
    if n_full == 0 or len(dims_to_shuffle) == 0:
        return pos.copy()
    out = pos.copy()
    head = out[:n_full].copy()
    n_blocks = n_full // block_size_samples
    perm = rng.permutation(n_blocks)
    head_blocks = head.reshape(n_blocks, block_size_samples, pos.shape[1])
    permuted = head_blocks[perm]
    # Only the selected dims are permuted; the rest keep their original order.
    final_blocks = head_blocks.copy()
    for d in dims_to_shuffle:
        final_blocks[:, :, d] = permuted[:, :, d]
    out[:n_full] = final_blocks.reshape(n_full, pos.shape[1])
    return out


def _skaggs(
    rate: np.ndarray,
    occ: np.ndarray,
    mask: np.ndarray,
    skaggs_correct: bool,
) -> tuple[float, float, float]:
    """Compute mean rate, spatial information, and sparsity.

    With ``skaggs_correct=True`` (default), uses the standard Skaggs
    (1993) formulae with occupancy-probability weighting:

        I = Σ p(x) (λ(x)/λ̄) log₂(λ(x)/λ̄)         bits/spike
        s = (Σ p(x) λ(x))² / Σ p(x) λ(x)²

    With ``skaggs_correct=False``, replicates the labbox bug-compatible
    versions which weight uniformly over occupied bins:

        I = Σ (1/N) (λ(x)/λ̄) log₂(λ(x)/λ̄)         bits/sample
        s = (mean rate over occupied)² / mean(rate²)

    where N = number of occupied bins.

    Returns ``(mean_rate, spatial_information, sparsity)``.  All NaN
    when no bins are occupied.
    """
    occ_in    = occ[mask]
    rate_in   = rate[mask]
    if occ_in.size == 0 or np.sum(occ_in) <= 0:
        return float("nan"), float("nan"), float("nan")

    total_occ = float(np.sum(occ_in))
    p_occ     = occ_in / total_occ                                      # probability per bin

    # Mean rate weighted by occupancy probability — this is the firing-rate
    # expectation under the rat's spatial distribution.  Both Skaggs forms
    # use this for the normaliser.
    mean_rate = float(np.sum(p_occ * rate_in))
    if mean_rate <= 0:
        return mean_rate, float("nan"), float("nan")

    # Mask out bins with zero rate to avoid log(0)·0 (defined as 0 in the
    # Skaggs convention but produces NaN under naive arithmetic).
    nz = (rate_in > 0) & np.isfinite(rate_in)
    rate_nz   = rate_in[nz]
    p_nz      = p_occ[nz]

    if skaggs_correct:
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = rate_nz / mean_rate
            si = float(np.nansum(p_nz * ratio * np.log2(ratio)))
            spar_num = float(np.sum(p_occ * rate_in)) ** 2
            spar_den = float(np.sum(p_occ * rate_in ** 2))
            sparsity = spar_num / spar_den if spar_den > 0 else float("nan")
    else:
        # labbox uniform weighting over occupied bins
        n_occ = float(np.sum(mask))
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = rate_nz / mean_rate
            si = float(np.nansum((1.0 / n_occ) * ratio * np.log2(ratio)))
            mean_uniform   = float(np.mean(rate_in))
            mean_sq_uniform = float(np.mean(rate_in ** 2))
            sparsity = (mean_uniform ** 2 / mean_sq_uniform
                        if mean_sq_uniform > 0 else float("nan"))

    return mean_rate, si, sparsity


# ─────────────────────────────────────────────────────────────────────────── #
# Core kernel — single (unit, iteration) rate map                              #
# ─────────────────────────────────────────────────────────────────────────── #

def _compute_one_rate_map(
    spike_pos: np.ndarray,
    marker_pos: np.ndarray,
    bin_edges: list[np.ndarray],
    n_bins: np.ndarray,
    smoothing_sigma: np.ndarray | None,
    samplerate: float,
    min_occupancy: float,
    skaggs_correct: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    """Compute one rate map.  Returns
    ``(rate_map, occupancy, spike_count, mask, mean_rate, si, sparsity)``."""
    grid_shape = tuple(n_bins.tolist())

    # Bin positions and spikes
    pos_idx     = _bin_positions(marker_pos, bin_edges, n_bins)
    spike_idx   = _bin_positions(spike_pos,  bin_edges, n_bins)

    occupancy   = _accumarray_nd(pos_idx,   grid_shape) / samplerate    # seconds
    spike_count = _accumarray_nd(spike_idx, grid_shape)                  # raw count

    # Smooth (gaussian, mode='constant' → zero-pad, matches MATLAB convn 'same')
    if smoothing_sigma is not None:
        sigma = smoothing_sigma if smoothing_sigma.size > 1 else float(smoothing_sigma[0])
        s_occ   = gaussian_filter(occupancy,   sigma=sigma, mode="constant")
        s_count = gaussian_filter(spike_count, sigma=sigma, mode="constant")
    else:
        s_occ, s_count = occupancy, spike_count

    # Occupancy mask — labbox uses minimally-smoothed occupancy and a
    # weird hardcoded threshold; we use min_occupancy (in seconds) on the
    # smoothed occupancy directly.  Same intent, simpler.
    mask = s_occ > min_occupancy

    with np.errstate(divide="ignore", invalid="ignore"):
        rate_map = np.where(mask, s_count / np.maximum(s_occ, np.finfo(float).eps), np.nan)

    mean_rate, si, sparsity = _skaggs(rate_map, occupancy, mask, skaggs_correct)
    return rate_map, s_occ, s_count, mask, mean_rate, si, sparsity


# ─────────────────────────────────────────────────────────────────────────── #
# Public API                                                                   #
# ─────────────────────────────────────────────────────────────────────────── #

def place_field(
    spikes:              NBSpk,
    xyz:                 NBDxyz | np.ndarray,
    units:               int | Sequence[int] | None = None,
    *,
    bin_size:            float | Sequence[float],
    boundary:            Sequence[Sequence[float]],
    state:               NBEpoch | None = None,
    samplerate:          float | None = None,
    smoothing_sigma:     float | Sequence[float] | None = 2.2,
    min_occupancy:       float = 0.06,
    min_spikes:          int   = 11,
    n_iter:              int   = 1,
    bootstrap_fraction:  float = 0.0,
    halfsample:          bool  = False,
    pos_shuffle:         bool  = False,
    pos_shuffle_dims:    Sequence[int] | None = None,
    shuffle_block_size:  float = 1.0,
    skaggs_correct:      bool  = True,
    rng:                 np.random.Generator | int | None = None,
) -> PlaceFieldResult:
    """Compute place-field rate maps for one or more units.

    Port of :file:`MTA/@MTAApfs/MTAApfs.m` and
    :file:`MTA/analysis/placefields/PlotPF.m`, with the persistence /
    incremental-update layer stripped out (use ``np.savez`` / ``pickle``
    on the result).  See the module docstring for what was kept,
    dropped, and corrected.

    Parameters
    ----------
    spikes:
        :class:`NBSpk` containing spike times and cluster IDs.  The
        ``samplerate`` of this object is the original recording rate
        (e.g. 20 000 Hz) — only used for spike-time → position-index
        conversion (it doesn't have to equal the position rate).
    xyz:
        Position data.  Either:

        * :class:`NBDxyz` of shape ``(T, n_markers, n_dims)`` — the
          first marker and the first ``len(boundary)`` spatial dims
          are used.  Use :meth:`NBDxyz.sel` upstream for finer control.
        * Plain ndarray of shape ``(T, n_dims)`` with explicit
          ``samplerate``.
    units:
        Cluster ID(s) to compute.  ``None`` (default) → all units
        present in ``spikes``.  Scalar accepted for a single unit.
    bin_size:
        Spatial bin size.  Scalar (isotropic) or sequence of length
        ``n_dims``.  Same units as the position data (typically mm).
    boundary:
        ``(n_dims, 2)`` sequence of ``[min, max]`` per dimension.
        Position samples falling outside are dropped.
    state:
        Optional :class:`NBEpoch` restricting which time samples are
        counted.  Resampled internally to match the position
        samplerate.  ``None`` → all samples.
    samplerate:
        Required when ``xyz`` is a plain ndarray; ignored when ``xyz``
        is :class:`NBDxyz`.
    smoothing_sigma:
        Gaussian σ in **bins** (matches the labbox convention).  Scalar
        or sequence of length ``n_dims``.  ``None`` disables smoothing.
        Default 2.2 (the labbox default for 30 mm bins).
    min_occupancy:
        Bins with smoothed dwell time below this threshold (in
        **seconds**) are masked out.  Default 0.06 (labbox default).
    min_spikes:
        Units with fewer than this many spikes (after state masking,
        before any resampling) are returned with all-NaN rate maps.
        Default 11 (labbox default).
    n_iter:
        Number of rate-map iterations.  ``1`` (default) → a single
        deterministic map.  ``> 1`` → multiple iterations using the
        bootstrap / halfsample / pos_shuffle controls.
    bootstrap_fraction:
        Fraction of spikes to resample with replacement on each
        iteration.  ``0.0`` (default) disables bootstrap; e.g. ``1.0``
        resamples ``n_spikes`` spikes per iteration.  Has no effect
        when ``n_iter == 1``.
    halfsample:
        If ``True``, each iteration uses a random half of the position
        blocks (paired complementary halves are produced for adjacent
        iterations: iter 0 + iter 1 = full data; iter 2 + iter 3 = full
        data; etc.).  Requires ``n_iter`` to be even.  Default ``False``.
    pos_shuffle:
        If ``True``, block-permute the position before binning on each
        iteration — produces a null distribution.  Default ``False``.
    pos_shuffle_dims:
        Which spatial dimensions to shuffle.  Useful for "shuffle x but
        not y" tests.  Default ``None`` → all dims.
    shuffle_block_size:
        Block size for half-sample / position shuffle, in **seconds**.
        Default 1.0.
    skaggs_correct:
        Use the standard (occupancy-weighted) Skaggs formulae.  Set
        ``False`` for bit-for-bit labbox compatibility (uniform
        weighting over occupied bins).  Default ``True``.
    rng:
        Seed or :class:`numpy.random.Generator` for reproducibility.

    Returns
    -------
    :class:`PlaceFieldResult`

    Examples
    --------
    Basic 2-D rate map for unit 5 during walking::

        from neurobox.analysis.spatial import place_field
        result = place_field(
            spikes, xyz, units=5,
            bin_size=30,
            boundary=[(-500, 500), (-500, 500)],
            state=stc['walk'],
            smoothing_sigma=2.2,
        )

    Bootstrap with 100 iterations::

        result = place_field(
            spikes, xyz, units=range(1, 50),
            bin_size=30,
            boundary=[(-500, 500), (-500, 500)],
            state=stc['walk'],
            n_iter=100,
            bootstrap_fraction=1.0,
            rng=42,
        )
        # result.rate_map.shape = (n_bins_x, n_bins_y, 49, 100)
        # result.spatial_info.shape = (49, 100)
    """
    # ── Argument validation and normalisation ─────────────────────────── #
    n_dims = len(boundary)
    if n_iter < 1:
        raise ValueError(f"n_iter must be ≥ 1, got {n_iter}")
    if halfsample and n_iter % 2 != 0:
        raise ValueError("halfsample=True requires n_iter to be even")
    if bootstrap_fraction < 0:
        raise ValueError(f"bootstrap_fraction must be ≥ 0, got {bootstrap_fraction}")

    rng_ = (rng if isinstance(rng, np.random.Generator)
            else np.random.default_rng(rng))

    if isinstance(units, (int, np.integer)):
        unit_ids = np.array([int(units)], dtype=np.int64)
    elif units is None:
        unit_ids = np.unique(spikes.clu).astype(np.int64)
    else:
        unit_ids = np.asarray(units, dtype=np.int64).ravel()

    if smoothing_sigma is not None:
        sigma_arr = (np.full(n_dims, float(smoothing_sigma))
                     if np.isscalar(smoothing_sigma)
                     else np.asarray(smoothing_sigma, dtype=np.float64).ravel())
        if sigma_arr.size != n_dims:
            raise ValueError(
                f"smoothing_sigma must be scalar or length {n_dims}; "
                f"got {sigma_arr.size}"
            )
    else:
        sigma_arr = None

    # ── State-mask the position and remember original indices ─────────── #
    pos_compressed, pos_original_indices, pos_sr = _state_mask_and_compress(
        xyz, state, samplerate, n_dims
    )

    # ── Build the bin grid ────────────────────────────────────────────── #
    bin_edges, bin_centres, n_bins = _build_bins(boundary, bin_size)
    grid_shape = tuple(n_bins.tolist())

    # ── Allocate outputs ──────────────────────────────────────────────── #
    n_units = unit_ids.size
    rate_map_full     = np.full(grid_shape + (n_units, n_iter), np.nan)
    spike_count_full  = np.zeros(grid_shape + (n_units, n_iter), dtype=np.float64)
    spatial_info_full = np.full((n_units, n_iter), np.nan)
    sparsity_full     = np.full((n_units, n_iter), np.nan)
    mean_rate_full    = np.full((n_units, n_iter), np.nan)
    n_spikes_full     = np.zeros(n_units, dtype=np.int64)
    occupancy_iter0   = np.zeros(grid_shape, dtype=np.float64)
    mask_iter0        = np.zeros(grid_shape, dtype=bool)

    # ── If position is empty, return all-NaN ──────────────────────────── #
    if pos_compressed.shape[0] == 0:
        return PlaceFieldResult(
            rate_map=rate_map_full, occupancy=occupancy_iter0,
            spike_count=spike_count_full, occupancy_mask=mask_iter0,
            bin_edges=bin_edges, bin_centres=bin_centres,
            spatial_info=spatial_info_full, sparsity=sparsity_full,
            mean_rate=mean_rate_full, unit_ids=unit_ids,
            n_spikes=n_spikes_full, samplerate=pos_sr,
            skaggs_correct=skaggs_correct,
        )

    # ── Per-iteration position arrays ─────────────────────────────────── #
    block_size_samples = max(1, int(round(shuffle_block_size * pos_sr)))
    n_full_blocks      = pos_compressed.shape[0] // block_size_samples
    n_full             = n_full_blocks * block_size_samples

    if pos_shuffle:
        sh_dims = (list(range(n_dims)) if pos_shuffle_dims is None
                   else list(pos_shuffle_dims))

    # Half-sample position-block masks (per iteration)
    if halfsample:
        halfsize = (n_full_blocks - n_full_blocks % 2) // 2
        halfblock_indices = np.zeros((n_iter, halfsize), dtype=np.int64)
        for j in range(0, n_iter, 2):
            perm = rng_.permutation(n_full_blocks)
            halfblock_indices[j]     = perm[:halfsize]
            halfblock_indices[j + 1] = perm[halfsize:2 * halfsize]

    # ── Per-unit, per-iteration loop ──────────────────────────────────── #
    spike_table = spikes.by_unit()  # dict: unit_id → spike-times-array (in seconds)

    for u_idx, unit in enumerate(unit_ids.tolist()):
        all_unit_spikes = spike_table.get(int(unit), np.empty(0))
        # State-restrict the unit's spikes — same epoch as for position
        if state is not None and not state.isempty():
            ep = state.resample(pos_sr)
            from neurobox.dtype.spikes import _restrict_times
            unit_spike_times = _restrict_times(all_unit_spikes, ep)
        else:
            unit_spike_times = all_unit_spikes

        # Spike → compressed-position-index mapping
        spike_pos_idx = _spike_to_position_index(
            unit_spike_times, pos_original_indices, pos_sr
        )
        n_spk = spike_pos_idx.size
        n_spikes_full[u_idx] = n_spk

        if n_spk < min_spikes:
            continue  # leave outputs NaN

        for it in range(n_iter):
            # Determine which spikes contribute on this iteration
            if bootstrap_fraction > 0 and (n_iter > 1 or bootstrap_fraction != 1.0):
                # Sample with replacement
                n_resampled = max(1, int(round(n_spk * bootstrap_fraction)))
                pick = rng_.integers(0, n_spk, size=n_resampled)
                this_spike_idx = spike_pos_idx[pick]
            else:
                this_spike_idx = spike_pos_idx

            # Determine which position samples are visible
            if halfsample:
                hb = halfblock_indices[it]
                # Build sample mask from block indices
                pos_used_mask = np.zeros(n_full, dtype=bool)
                # Vectorised: for each chosen block, set its samples True
                starts = hb * block_size_samples
                for s in starts:
                    pos_used_mask[s:s + block_size_samples] = True
                marker_pos = pos_compressed[:n_full][pos_used_mask]
                # Also drop any spikes whose (compressed) position index
                # falls outside the half-sample mask
                spike_in_used = pos_used_mask[this_spike_idx[
                    this_spike_idx < n_full
                ]]
                kept = this_spike_idx[this_spike_idx < n_full][spike_in_used]
                # Re-index to the half-sample's local coordinates
                # (cumulative-sum of pos_used_mask up to each kept index, minus 1)
                cumsum = np.cumsum(pos_used_mask) - 1
                spike_pos_local = cumsum[kept]
                spike_pos = marker_pos[spike_pos_local] if spike_pos_local.size > 0 \
                    else np.empty((0, n_dims))
            else:
                marker_pos = pos_compressed
                # Drop spike indices outside the (possibly trimmed) marker_pos.
                kept = this_spike_idx[this_spike_idx < marker_pos.shape[0]]
                spike_pos = marker_pos[kept]

            # Optional position shuffle (after halfsample selection)
            if pos_shuffle:
                marker_pos = _shuffle_position_blocks(
                    marker_pos, block_size_samples, rng_, sh_dims
                )
                # spike_pos was indexed before the shuffle; recompute it as
                # the shuffled positions at the same row indices
                if not halfsample:
                    spike_pos = marker_pos[kept] if kept.size > 0 \
                        else np.empty((0, n_dims))

            (rate_map, s_occ, s_count, mask,
             mean_rate, si, spar) = _compute_one_rate_map(
                spike_pos       = spike_pos,
                marker_pos      = marker_pos,
                bin_edges       = bin_edges,
                n_bins          = n_bins,
                smoothing_sigma = sigma_arr,
                samplerate      = pos_sr,
                min_occupancy   = min_occupancy,
                skaggs_correct  = skaggs_correct,
            )

            rate_map_full[..., u_idx, it]    = rate_map
            spike_count_full[..., u_idx, it] = s_count
            spatial_info_full[u_idx, it]     = si
            sparsity_full[u_idx, it]         = spar
            mean_rate_full[u_idx, it]        = mean_rate

            if u_idx == 0 and it == 0:
                occupancy_iter0[:] = s_occ
                mask_iter0[:]      = mask

    return PlaceFieldResult(
        rate_map        = rate_map_full,
        occupancy       = occupancy_iter0,
        spike_count     = spike_count_full,
        occupancy_mask  = mask_iter0,
        bin_edges       = bin_edges,
        bin_centres     = bin_centres,
        spatial_info    = spatial_info_full,
        sparsity        = sparsity_full,
        mean_rate       = mean_rate_full,
        unit_ids        = unit_ids,
        n_spikes        = n_spikes_full,
        samplerate      = pos_sr,
        skaggs_correct  = skaggs_correct,
    )
