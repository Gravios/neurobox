"""
neurobox.analysis.spatial.place_field_stats
============================================
Per-patch statistics for place-field rate maps.

Port of :file:`MTA/analysis/placefields/PlaceFieldStats.m` (Anton
Sirota / Justin Graboski), with two improvements over the MATLAB
original.

What the MATLAB version did
---------------------------
For each unit it:
  1. Computed the deterministic rate map.
  2. Found contiguous patches above a threshold (90th percentile by
     default).
  3. Sorted patches by area, kept the largest *N*.
  4. Reported per-patch area, peak/mean firing rate, centre of mass,
     and the pixel indices.

It also had stubs (commented-out in the MATLAB) for repeating step
2-4 on each bootstrap iteration of an :class:`MTAApfs_bs` rate map,
to get per-patch confidence intervals.  The bootstrap loop was
abandoned in MATLAB because of speed.

What this port does differently
-------------------------------
1. **Fixed-patch mode (recommended, default).**  Patches are defined
   *once* on the deterministic rate map.  For each bootstrap iter we
   restrict to those patch masks and compute peak/mean/area/COM
   *within them*.  This gives stable patch identity across
   iterations (patch 1 is always the same blob), so the confidence
   interval has a coherent spatial interpretation —
   "how does the peak rate inside the main field vary across
   resamples?"  The MATLAB approach re-segmented each iter
   independently, producing patch labels that don't correspond
   across iterations.

2. **Per-iter mode (matches MATLAB semantics).**  Re-segments each
   iter separately.  Useful for population-level "how many distinct
   fields per iter" statistics.  Off by default.

3. **Variable-length patch lists** instead of NaN-padded fixed-size
   arrays.  Each patch carries its own pixel-coordinate array — no
   ``maxBinCount = 0.22 * total_bins`` truncation.

4. **Threshold options** kept from MATLAB and one new:
   - ``"percentile"`` (default, 90th percentile of finite bins) —
     matches MATLAB default.
   - ``"halfmax"`` (peak-rate / 2) — MATLAB stub, never reached.
   - ``"absolute"`` (caller-supplied Hz) — new, useful when the
     caller has a session-wide or shuffled threshold.

Inputs
------
The function takes a :class:`PlaceFieldResult` (output of
:func:`neurobox.analysis.spatial.place_field`) — no need to wrap a
plain ndarray.  Bootstrap iterations and per-unit slicing are
handled by indexing into ``rate_map`` directly.

Examples
--------
Basic, single-unit, deterministic map only::

    from neurobox.analysis.spatial import place_field, place_field_stats
    pf  = place_field(spk, xyz, units=5,
                      bin_size=30, boundary=[(-500, 500), (-500, 500)],
                      smoothing_sigma=2.2)
    stats = place_field_stats(pf, max_n_patches=2)
    main_field = stats[0].patches[0]   # largest patch of unit 0
    print(main_field.peak_rate, main_field.area, main_field.center_of_mass)

Bootstrap with fixed patches::

    pf = place_field(spk, xyz, units=range(50),
                     bin_size=30, boundary=[(-500, 500), (-500, 500)],
                     n_iter=200, bootstrap_fraction=1.0, rng=0)
    stats = place_field_stats(pf, mode="fixed_patches")
    # Per-patch CIs across iterations
    main_field_pfr_ci = np.percentile(
        stats[5].patches[0].peak_rate_iter, [2.5, 97.5]
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence, Union

import numpy as np
from scipy.ndimage import label as _ndi_label

from .place_fields import PlaceFieldResult


# ─────────────────────────────────────────────────────────────────────────── #
# Result containers                                                            #
# ─────────────────────────────────────────────────────────────────────────── #

@dataclass
class Patch:
    """One detected patch in a rate map.

    Attributes
    ----------
    pixel_indices:
        ``(n_pixels, n_dims)`` integer array of bin indices.
    pixel_coords:
        ``(n_pixels, n_dims)`` float array of bin-centre positions
        in the same units as ``boundary`` (typically mm).
    area:
        Patch area in **squared position-units** (e.g. mm²).
        Computed as ``n_pixels × prod(bin_dims)``.
    peak_rate:
        Maximum firing rate within the patch (Hz).
    mean_rate:
        Mean firing rate within the patch (Hz).
    center_of_mass:
        Rate-weighted centre of mass in position-units, length =
        n_dims.
    peak_rate_iter:
        ``(n_iter,)`` peak rate per bootstrap iteration.  ``None``
        when no bootstrap was run (n_iter == 1).
    mean_rate_iter:
        ``(n_iter,)`` mean rate per iteration.  ``None`` if no
        bootstrap.
    area_iter:
        ``(n_iter,)`` area per iteration (only meaningful in
        ``per_iter`` mode where re-segmentation can change the patch
        size; in ``fixed_patches`` mode it equals ``area`` repeated).
    com_iter:
        ``(n_iter, n_dims)`` centre-of-mass per iteration.
    """

    pixel_indices:    np.ndarray
    pixel_coords:     np.ndarray
    area:             float
    peak_rate:        float
    mean_rate:        float
    center_of_mass:   np.ndarray
    peak_rate_iter:   "np.ndarray | None" = None
    mean_rate_iter:   "np.ndarray | None" = None
    area_iter:        "np.ndarray | None" = None
    com_iter:         "np.ndarray | None" = None

    @property
    def n_pixels(self) -> int:
        return int(self.pixel_indices.shape[0])


@dataclass
class UnitStats:
    """Patch statistics for one unit.

    Attributes
    ----------
    unit_id:
        Cluster identifier this entry refers to (matches
        ``PlaceFieldResult.unit_ids``).
    peak_rate:
        Whole-map peak rate (Hz).  NaN when the unit has too few
        spikes for a rate map.
    rate_threshold:
        Threshold (Hz) used for patch detection.
    threshold_method:
        Which method produced ``rate_threshold``.
    patches:
        List of :class:`Patch`, sorted by area descending.  Up to
        ``max_n_patches`` entries; can be empty if no pixels exceed
        the threshold.
    """

    unit_id:           int
    peak_rate:         float
    rate_threshold:    float
    threshold_method:  str
    patches:           list[Patch] = field(default_factory=list)

    @property
    def n_patches(self) -> int:
        return len(self.patches)


# ─────────────────────────────────────────────────────────────────────────── #
# Threshold helpers                                                            #
# ─────────────────────────────────────────────────────────────────────────── #

def _resolve_threshold(
    rate_map_2d:    np.ndarray,
    method:         str,
    percentile:     float,
    absolute:       float | None,
) -> float:
    """Compute the rate threshold according to *method*.

    Parameters
    ----------
    rate_map_2d:
        Rate map for one unit, one iteration.
    method:
        One of ``'percentile'``, ``'halfmax'``, ``'absolute'``.
    percentile:
        Percentile cutoff for the ``'percentile'`` method.
    absolute:
        Absolute threshold (Hz) for the ``'absolute'`` method.
    """
    finite = rate_map_2d[np.isfinite(rate_map_2d)]
    if finite.size == 0:
        return float("nan")

    if method == "percentile":
        return float(np.percentile(finite, percentile))
    if method == "halfmax":
        peak = float(np.nanmax(finite))
        return 0.5 * peak
    if method == "absolute":
        if absolute is None:
            raise ValueError("absolute threshold requires `threshold_value=...`")
        return float(absolute)
    raise ValueError(
        f"Unknown threshold method {method!r}; "
        f"expected 'percentile' / 'halfmax' / 'absolute'"
    )


# ─────────────────────────────────────────────────────────────────────────── #
# Patch detection on a single rate map                                         #
# ─────────────────────────────────────────────────────────────────────────── #

# 8-connected structuring element matches MATLAB bwboundaries default.
_CONN_8 = np.ones((3, 3), dtype=np.int64)


def _detect_patches(
    rate_map:       np.ndarray,
    threshold:      float,
    bin_centres:    list[np.ndarray],
    bin_dims:       np.ndarray,
    max_n_patches:  int,
) -> list[dict]:
    """Detect patches above ``threshold`` and return per-patch stats.

    Returns a list of dicts (one per kept patch) with keys
    ``mask, indices, coords, area, peak_rate, mean_rate, com``.
    Sorted by area descending, truncated to ``max_n_patches``.
    """
    if not np.isfinite(threshold):
        return []

    binary = np.where(np.isfinite(rate_map) & (rate_map > threshold),
                      True, False)
    n_dims = rate_map.ndim

    if n_dims == 2:
        labels, n_blobs = _ndi_label(binary, structure=_CONN_8)
    else:
        # Default cross-shaped connectivity (face-touching) for 3-D+
        labels, n_blobs = _ndi_label(binary)

    if n_blobs == 0:
        return []

    pixel_area = float(np.prod(bin_dims))
    patches: list[dict] = []

    for blob_id in range(1, n_blobs + 1):
        mask = (labels == blob_id)
        idx  = np.argwhere(mask)                                # (n_pix, n_dims)
        npix = idx.shape[0]
        if npix == 0:
            continue
        # Bin-centre coordinates: lookup per axis
        coords = np.column_stack([
            bin_centres[d][idx[:, d]] for d in range(n_dims)
        ])
        rates = rate_map[mask]
        finite_rates = rates[np.isfinite(rates)]
        if finite_rates.size == 0:
            continue

        total_rate = float(np.nansum(rates))
        if total_rate > 0:
            com = (rates[:, None] * coords).sum(axis=0) / total_rate
        else:
            com = coords.mean(axis=0)

        patches.append(dict(
            mask        = mask,
            indices     = idx,
            coords      = coords,
            area        = pixel_area * npix,
            peak_rate   = float(np.nanmax(rates)),
            mean_rate   = float(np.nanmean(rates)),
            com         = com,
        ))

    patches.sort(key=lambda p: p["area"], reverse=True)
    return patches[:max_n_patches]


# ─────────────────────────────────────────────────────────────────────────── #
# Per-iteration stats given fixed masks                                        #
# ─────────────────────────────────────────────────────────────────────────── #

def _stats_within_mask(
    rate_map:    np.ndarray,
    mask:        np.ndarray,
    coords:      np.ndarray,
    bin_dims:    np.ndarray,
) -> tuple[float, float, float, np.ndarray]:
    """Compute (area, peak, mean, COM) for *rate_map* restricted to *mask*."""
    pixel_area = float(np.prod(bin_dims))
    rates_here = rate_map[mask]
    if rates_here.size == 0 or not np.any(np.isfinite(rates_here)):
        return (float("nan"),) * 3 + (np.full(rate_map.ndim, np.nan),)
    finite_rates = rates_here[np.isfinite(rates_here)]
    area = pixel_area * finite_rates.size
    peak = float(np.nanmax(rates_here))
    mean = float(np.nanmean(rates_here))
    total = float(np.nansum(rates_here))
    if total > 0:
        # Use only the finite-rate pixels for the COM weighting
        finite_mask = np.isfinite(rates_here)
        weighted = rates_here[finite_mask][:, None] * coords[finite_mask]
        com = weighted.sum(axis=0) / float(np.nansum(rates_here[finite_mask]))
    else:
        com = coords.mean(axis=0)
    return area, peak, mean, com


# ─────────────────────────────────────────────────────────────────────────── #
# Public API                                                                   #
# ─────────────────────────────────────────────────────────────────────────── #

def place_field_stats(
    pf:               PlaceFieldResult,
    *,
    units:            int | Sequence[int] | None = None,
    max_n_patches:    int = 2,
    threshold_method: Literal["percentile", "halfmax", "absolute"] = "percentile",
    threshold_value:  float | None = None,
    threshold_pct:    float = 90.0,
    mode:             Literal["fixed_patches", "per_iter"] = "fixed_patches",
) -> list[UnitStats]:
    """Compute per-patch statistics for each unit in a rate-map result.

    See the module docstring for an explanation of the two modes.

    Parameters
    ----------
    pf:
        Output of :func:`neurobox.analysis.spatial.place_field`.
    units:
        Subset of unit IDs to compute.  ``None`` (default) → all units
        present in *pf*.
    max_n_patches:
        Maximum number of patches to keep per unit, sorted by area
        descending.  Default 2 (matches MATLAB).
    threshold_method:
        How to derive the firing-rate cutoff for patch detection.

        ``'percentile'`` (default)
            ``threshold_pct``-th percentile of the finite rate-map
            bins.  Matches MATLAB default.
        ``'halfmax'``
            Half of the whole-map peak rate.
        ``'absolute'``
            Use ``threshold_value`` directly.
    threshold_value:
        Cutoff in Hz for ``threshold_method='absolute'``.
    threshold_pct:
        Percentile for ``threshold_method='percentile'`` (default 90).
    mode:
        ``'fixed_patches'`` (default)
            Detect patches once on the deterministic map (iter 0) and
            compute per-iter statistics within those fixed masks.
            Stable patch identity across iterations.
        ``'per_iter'``
            Re-detect patches on every iteration independently.
            Matches MATLAB semantics; patch identities don't
            correspond across iterations.

    Returns
    -------
    list[:class:`UnitStats`]
        Same length and order as ``units`` (or ``pf.unit_ids`` if
        ``units is None``).
    """
    if mode not in ("fixed_patches", "per_iter"):
        raise ValueError(
            f"mode must be 'fixed_patches' or 'per_iter'; got {mode!r}"
        )

    if units is None:
        unit_ids = list(pf.unit_ids)
    elif isinstance(units, (int, np.integer)):
        unit_ids = [int(units)]
    else:
        unit_ids = list(units)

    rate_map_full = pf.rate_map     # (*bin_shape, n_units, n_iter)
    n_dims = rate_map_full.ndim - 2
    n_iter = rate_map_full.shape[-1]

    # Bin geometry
    bin_centres = pf.bin_centres
    bin_dims = np.array([
        float(np.diff(c[:2]).item()) if len(c) > 1 else 1.0
        for c in bin_centres
    ])

    out: list[UnitStats] = []

    for unit_id in unit_ids:
        # Index into the unit axis
        if unit_id not in pf.unit_ids:
            raise ValueError(
                f"unit {unit_id} not in PlaceFieldResult.unit_ids"
            )
        u_idx = int(np.where(pf.unit_ids == unit_id)[0][0])

        # Iter-0 (deterministic) map for patch detection
        rate0 = rate_map_full[..., u_idx, 0]
        if not np.any(np.isfinite(rate0)):
            out.append(UnitStats(
                unit_id          = unit_id,
                peak_rate        = float("nan"),
                rate_threshold   = float("nan"),
                threshold_method = threshold_method,
            ))
            continue

        # Whole-map peak (for reporting)
        peak_overall = float(np.nanmax(rate0))

        # Threshold from iter 0
        thr = _resolve_threshold(
            rate0, threshold_method, threshold_pct, threshold_value
        )

        # Detect patches once on iter 0 — these define the patch identities
        det_patches = _detect_patches(
            rate0, thr, bin_centres, bin_dims, max_n_patches
        )

        if not det_patches:
            out.append(UnitStats(
                unit_id          = unit_id,
                peak_rate        = peak_overall,
                rate_threshold   = thr,
                threshold_method = threshold_method,
            ))
            continue

        # ── Build Patch objects ──────────────────────────────────────── #
        patches: list[Patch] = []
        for p in det_patches:
            patch = Patch(
                pixel_indices  = p["indices"],
                pixel_coords   = p["coords"],
                area           = p["area"],
                peak_rate      = p["peak_rate"],
                mean_rate      = p["mean_rate"],
                center_of_mass = p["com"],
            )

            # ── Bootstrap statistics ─────────────────────────────────── #
            if n_iter > 1:
                if mode == "fixed_patches":
                    # Apply the same mask to every iteration
                    iter_peak = np.full(n_iter, np.nan)
                    iter_mean = np.full(n_iter, np.nan)
                    iter_area = np.full(n_iter, np.nan)
                    iter_com  = np.full((n_iter, n_dims), np.nan)
                    for it in range(n_iter):
                        rmap_it = rate_map_full[..., u_idx, it]
                        a, pk, mn, com = _stats_within_mask(
                            rmap_it, p["mask"], p["coords"], bin_dims
                        )
                        iter_area[it]    = a
                        iter_peak[it]    = pk
                        iter_mean[it]    = mn
                        iter_com[it, :]  = com
                    patch.area_iter      = iter_area
                    patch.peak_rate_iter = iter_peak
                    patch.mean_rate_iter = iter_mean
                    patch.com_iter       = iter_com
                # per_iter mode handled below — patches are reused only
                # for the iter-0 entries; per-iter sub-results are
                # collected into a separate sequence.

            patches.append(patch)

        # ── per_iter mode: re-segment each iter, store as flat list ── #
        if n_iter > 1 and mode == "per_iter":
            # For 'per_iter', patches detected per iteration are kept as
            # separate per-iter lists.  We attach them to each iter-0
            # patch by spatial proximity (closest COM in iter 0).
            # Allocate per-patch buffers
            iter_peak = np.full((len(patches), n_iter), np.nan)
            iter_mean = np.full((len(patches), n_iter), np.nan)
            iter_area = np.full((len(patches), n_iter), np.nan)
            iter_com  = np.full((len(patches), n_iter, n_dims), np.nan)
            iter_peak[:, 0] = [p.peak_rate for p in patches]
            iter_mean[:, 0] = [p.mean_rate for p in patches]
            iter_area[:, 0] = [p.area      for p in patches]
            for k, p in enumerate(patches):
                iter_com[k, 0] = p.center_of_mass

            iter0_coms = np.stack([p.center_of_mass for p in patches])

            for it in range(1, n_iter):
                rmap_it = rate_map_full[..., u_idx, it]
                if not np.any(np.isfinite(rmap_it)):
                    continue
                # Recompute threshold per iter (matches MATLAB intent)
                thr_it = _resolve_threshold(
                    rmap_it, threshold_method, threshold_pct, threshold_value
                )
                it_patches = _detect_patches(
                    rmap_it, thr_it, bin_centres, bin_dims, max_n_patches
                )
                # Match each detected patch to the closest iter-0 COM
                used = np.zeros(len(patches), dtype=bool)
                for ip in it_patches:
                    dists = np.linalg.norm(iter0_coms - ip["com"], axis=1)
                    # Forbid double-assignment
                    dists[used] = np.inf
                    if not np.isfinite(dists).any():
                        break
                    best = int(np.argmin(dists))
                    used[best] = True
                    iter_area[best, it]    = ip["area"]
                    iter_peak[best, it]    = ip["peak_rate"]
                    iter_mean[best, it]    = ip["mean_rate"]
                    iter_com[best, it, :]  = ip["com"]

            for k, p in enumerate(patches):
                p.area_iter      = iter_area[k]
                p.peak_rate_iter = iter_peak[k]
                p.mean_rate_iter = iter_mean[k]
                p.com_iter       = iter_com[k]

        out.append(UnitStats(
            unit_id          = unit_id,
            peak_rate        = peak_overall,
            rate_threshold   = thr,
            threshold_method = threshold_method,
            patches          = patches,
        ))

    return out
