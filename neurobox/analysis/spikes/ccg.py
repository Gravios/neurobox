"""
neurobox.analysis.spikes.ccg
=============================
Cross- and auto-correlograms.

Port of two labbox functions (Ken Harris / Anton Sirota):

================================  =============================================
labbox                            neurobox
================================  =============================================
TF/CCG.m                           :func:`ccg`
TF/Trains2CCG.m                    :func:`trains_to_ccg`
TF/CCGHeart.c (mex extension)      :mod:`._ccg_engine` (Cython)
                                   :mod:`._ccg_python_fallback` (numpy)
================================  =============================================

Inner-loop strategy
-------------------
The hot path is a Cython kernel (:mod:`._ccg_engine`).  When the
compiled extension isn't available, the wrapper transparently falls
back to a vectorised pure-Python implementation
(:mod:`._ccg_python_fallback`) that is algorithmically identical but
roughly 5-15× slower.  Code that uses :func:`ccg` doesn't need to know
which path is active.

Sign convention
---------------
The labbox / Harris CCG uses the convention that:

* ``ccg(:, i, j)`` is the histogram of "times of cell j relative to
  cell i" — i.e., positive lag = j fired *after* i.
* The MATLAB code applies ``flipud`` to the lower triangle to enforce
  this for both halves.

We replicate that convention exactly, so the output is a drop-in
replacement for the labbox ``CCG`` for downstream code (peak finding,
significance testing, etc.) that depends on the lag sign.

Normalizations
--------------
Five options, matching ``CCG.m``:

============== =========================================================
``"count"``    Raw bin counts.
``"hz"``       Conditional intensity of cell j given cell i fired,
               in Hz.  Default — most useful for visual inspection.
``"hz2"``      Joint intensity in Hz².
``"scale"``    Coincidence rate divided by expected rate under
               independence.  Asymptotes to 1.
``"none"``     Same as ``"count"`` (alias).
============== =========================================================

The ``"hz"`` mode normalises by the *centre* group's spike count
(``ccg[:, i, j] / (bin_size * n_i)``).  The ``"scale"`` mode normalises
by both groups' rates.

Edge bias correction
--------------------
When ``epochs`` is given, the time integral over which lag-``τ`` pairs
can occur is shorter than the integral for lag-0 pairs (no pair can
have lag ``τ`` if neither spike falls in an epoch wide enough to
contain both ends).  This induces a triangular bias in the CCG.
:func:`ccg` corrects for this exactly (per the labbox formula) when
``epochs`` is provided.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

import numpy as np

# Try the compiled kernel first; fall back to pure Python if not available.
try:
    from ._ccg_engine import compute_ccg_counts, compute_ccg_counts_with_pairs
    _USING_CYTHON = True
except ImportError:                                              # pragma: no cover
    from ._ccg_python_fallback import (                          # type: ignore[no-redef]
        compute_ccg_counts,
        compute_ccg_counts_with_pairs,
    )
    _USING_CYTHON = False


__all__ = ["ccg", "trains_to_ccg", "CCGResult", "is_compiled"]


# ─────────────────────────────────────────────────────────────────────────── #
# Result container                                                             #
# ─────────────────────────────────────────────────────────────────────────── #

@dataclass
class CCGResult:
    """Output of :func:`ccg` / :func:`trains_to_ccg`.

    Attributes
    ----------
    ccg:
        Cross-correlogram array, shape ``(n_bins, n_groups, n_groups)``.
        ``ccg[:, i, j]`` is the histogram of "cell j relative to cell
        i".  Dtype is ``float64`` for normalised modes,
        ``int64`` for ``"count"``.
    t:
        Lag axis in **milliseconds**, length ``n_bins``.  ``t[half_bins]``
        is 0.
    pairs:
        ``(n_pairs, 2)`` int array of ``[centre_index, other_index]``
        pairs, in the original (unsorted) array indexing.  Empty array
        if ``return_pairs=False``.
    bin_size_samples:
        Bin size in samples (``bin_size_ms * sample_rate / 1000``,
        rounded up).  This is what was actually passed to the kernel.
    sample_rate:
        Sample rate (Hz).  Used to convert ``t`` to ms.
    n_groups:
        Number of groups in the output (``len(group_subset)``).
    group_subset:
        Group labels selected for output, in display order.
    normalization:
        Normalization mode actually used.
    n_spikes_per_group:
        Spike count per group (post-filtering).
    axis_unit:
        Display unit for the y axis (e.g. ``"(Hz)"``, ``"(Spikes)"``).
    """

    ccg: np.ndarray
    t: np.ndarray
    pairs: np.ndarray
    bin_size_samples: int
    sample_rate: float
    n_groups: int
    group_subset: np.ndarray
    normalization: str
    n_spikes_per_group: np.ndarray
    axis_unit: str = ""


# ─────────────────────────────────────────────────────────────────────────── #
# Helpers                                                                      #
# ─────────────────────────────────────────────────────────────────────────── #

def _bias_for_epochs(
    epochs: np.ndarray,
    n_bins: int,
    half_bins: int,
    bin_size: float,
) -> tuple[np.ndarray, float]:
    """Edge-bias correction factor when restricting to ``epochs``.

    Direct port of the bias formula in :file:`labbox/TF/CCG.m` lines 162-175::

        nTerm  = [HalfBins:-1:1, 0.25, 1:HalfBins]
        for each epoch:
            EpochLen   = epoch[1] - epoch[0]
            EpochBias  = max(0, EpochLen - nTerm * BinSize) * BinSize
            Bias      += EpochBias
        Bias /= TotLen * BinSize

    Returns ``(bias, total_length)`` with ``bias.shape == (n_bins,)``.
    """
    n_term = np.concatenate([
        np.arange(half_bins, 0, -1, dtype=np.float64),
        np.array([0.25], dtype=np.float64),
        np.arange(1, half_bins + 1, dtype=np.float64),
    ])
    bias = np.zeros(n_bins, dtype=np.float64)
    total_length = 0.0
    for ep_start, ep_stop in epochs:
        epoch_len = ep_stop - ep_start
        epoch_bias = np.clip(epoch_len - n_term * bin_size, 0.0, None) * bin_size
        bias += epoch_bias
        total_length += epoch_len
    if total_length > 0:
        bias /= total_length * bin_size
    else:
        bias[:] = 1.0  # avoid division by zero; all epochs empty
    # Guard against zero entries (gives 0/0 in the divide); replace with 1 so
    # the corresponding output bin stays 0.
    bias[bias == 0.0] = 1.0
    return bias, total_length


def _within_ranges_flat(x: np.ndarray, ranges: np.ndarray) -> np.ndarray:
    """Boolean mask: which elements of ``x`` lie within any range.

    Inclusive at both ends.  Internal helper to avoid a circular import
    of :func:`neurobox.analysis.lfp.oscillations.within_ranges`.
    """
    if ranges.size == 0:
        return np.zeros(x.size, dtype=bool)
    starts = np.sort(ranges[:, 0])
    stops  = np.sort(ranges[:, 1])
    return (np.searchsorted(starts, x, side="right")
            > np.searchsorted(stops, x, side="left"))


# ─────────────────────────────────────────────────────────────────────────── #
# CCG                                                                          #
# ─────────────────────────────────────────────────────────────────────────── #

def ccg(
    times: np.ndarray,
    groups: np.ndarray,
    bin_size: int | float,
    half_bins: int,
    sample_rate: float = 20000.0,
    group_subset: Optional[Iterable[int]] = None,
    normalization: str = "hz",
    epochs: Optional[np.ndarray] = None,
    return_pairs: bool = False,
) -> CCGResult:
    """Cross- and auto-correlogram of multiple spike trains.

    Port of :file:`labbox/TF/CCG.m`.

    Parameters
    ----------
    times:
        1-D spike times (in samples — *not* seconds, matching labbox).
        **Need not be sorted** — sorting is done internally.
    groups:
        1-D group labels (one per spike).  Any integer labels are
        accepted; ``group_subset`` selects which to include.
        Alternatively pass a single scalar ``1`` to treat all spikes as
        one group.
    bin_size:
        Bin size **in samples** (matching the labbox ``BinSize`` units —
        :func:`trains_to_ccg` takes ms instead).
    half_bins:
        Number of bins on each side of zero.  Total bins = ``1 + 2*half_bins``.
    sample_rate:
        Sample rate in Hz.  Default 20000 (typical Neurosuite ``.dat`` rate).
        Used **only** for the ``t`` (millisecond) axis and for
        normalisation.
    group_subset:
        Iterable of group labels to include in the output, in the order
        they should appear.  Defaults to ``np.unique(groups)``.
    normalization:
        ``"count"``, ``"hz"`` (default), ``"hz2"``, ``"scale"``, or
        ``"none"`` (= ``"count"``).
    epochs:
        Optional ``(N, 2)`` array of ``[start, stop]`` time ranges (in
        the same units as ``times``) restricting which spikes are used
        and triggering edge-bias correction.
    return_pairs:
        If True, populate the ``pairs`` field of the result with the
        index pairs that contributed to the histogram.  Doubles runtime
        and increases memory.  Default False.

    Returns
    -------
    :class:`CCGResult`

    Examples
    --------
    >>> # Auto-correlogram of unit 5, ±50 ms in 1 ms bins (1250 Hz LFP)
    >>> result = ccg(spike_times, unit_ids,
    ...              bin_size=20, half_bins=50,        # 20 samples = 1 ms at 20 kHz
    ...              sample_rate=20000.0,
    ...              group_subset=[5], normalization='hz')
    >>> result.ccg.shape  # (101, 1, 1)
    (101, 1, 1)
    """
    # ── Validate / coerce inputs ────────────────────────────────────────── #
    times  = np.asarray(times,  dtype=np.float64).ravel()
    groups = np.atleast_1d(np.asarray(groups))

    # Single-group shortcut: scalar groups argument
    if groups.size == 1:
        groups = np.ones(times.size, dtype=np.int64)
        if group_subset is None:
            group_subset = np.array([1])
        n_groups = 1
    else:
        groups = groups.ravel().astype(np.int64)
        if groups.size != times.size:
            raise ValueError(
                f"times and groups must have same length: "
                f"{times.size} vs {groups.size}"
            )
        if group_subset is None:
            group_subset = np.unique(groups)
        else:
            group_subset = np.asarray(list(group_subset), dtype=np.int64)
        n_groups = group_subset.size

    group_subset = np.asarray(group_subset, dtype=np.int64)
    n_bins = 1 + 2 * int(half_bins)

    # ── Filter spikes ───────────────────────────────────────────────────── #
    if epochs is not None:
        epochs = np.asarray(epochs, dtype=np.float64)
        if epochs.ndim != 2 or epochs.shape[1] != 2:
            raise ValueError(f"epochs must have shape (N, 2), got {epochs.shape}")
        in_groups  = np.isin(groups, group_subset)
        in_finite  = np.isfinite(times)
        in_range   = _within_ranges_flat(times, epochs)
        included   = in_groups & in_finite & in_range
        # Warn if any epoch gap is shorter than the CCG span.
        if epochs.shape[0] > 1:
            gap_lens = epochs[1:, 0] - epochs[:-1, 1]
            too_short = gap_lens < bin_size * (half_bins + 0.5)
            if too_short.any():
                import warnings
                warnings.warn(
                    f"epochs {np.flatnonzero(too_short).tolist()} are "
                    f"followed by gaps shorter than the CCG span; edge bins "
                    f"may be biased",
                    UserWarning,
                    stacklevel=2,
                )
    else:
        in_groups = np.isin(groups, group_subset)
        in_finite = np.isfinite(times)
        included  = in_groups & in_finite

    times_in  = times[included]
    groups_in = groups[included]

    # Empty-ish guard (matches labbox)
    if times_in.size <= 1:
        empty_ccg = np.zeros((n_bins, n_groups, n_groups),
                             dtype=np.int64 if normalization in ("count", "none") else np.float64)
        t_axis = 1000.0 * np.arange(-half_bins, half_bins + 1) * bin_size / sample_rate
        return CCGResult(
            ccg=empty_ccg,
            t=t_axis,
            pairs=np.empty((0, 2), dtype=np.int64),
            bin_size_samples=int(bin_size),
            sample_rate=float(sample_rate),
            n_groups=n_groups,
            group_subset=group_subset,
            normalization=normalization,
            n_spikes_per_group=np.zeros(n_groups, dtype=np.int64),
            axis_unit="",
        )

    # ── Map group labels → contiguous 0-indexed cluster IDs ─────────────── #
    label_to_clu = {int(g): i for i, g in enumerate(group_subset)}
    clu_in = np.array([label_to_clu[int(g)] for g in groups_in], dtype=np.int64)

    # ── Sort by time (kernel requires it) ───────────────────────────────── #
    sort_ix = np.argsort(times_in, kind="stable")
    times_sorted = np.ascontiguousarray(times_in[sort_ix], dtype=np.float64)
    clu_sorted   = np.ascontiguousarray(clu_in[sort_ix],   dtype=np.int64)

    # ── Run the kernel ──────────────────────────────────────────────────── #
    if return_pairs:
        counts, raw_pairs = compute_ccg_counts_with_pairs(
            times_sorted, clu_sorted, float(bin_size), int(half_bins), int(n_groups),
        )
        # Map sorted indices back to the original `times` array indexing.
        included_idx = np.flatnonzero(included)
        sorted_to_original = included_idx[sort_ix]
        pairs = sorted_to_original[raw_pairs]
    else:
        counts = compute_ccg_counts(
            times_sorted, clu_sorted, float(bin_size), int(half_bins), int(n_groups),
        )
        pairs = np.empty((0, 2), dtype=np.int64)

    # ── Lag axis in ms ──────────────────────────────────────────────────── #
    t_axis = 1000.0 * np.arange(-half_bins, half_bins + 1) * bin_size / sample_rate

    # ── Bias correction (epochs) and normalisation ──────────────────────── #
    if epochs is not None:
        bias, t_range = _bias_for_epochs(epochs, n_bins, int(half_bins), float(bin_size))
    else:
        bias = np.ones(n_bins, dtype=np.float64)
        t_range = float(times_in.max() - times_in.min()) if times_in.size > 1 else 1.0

    n_per = np.array([
        int(np.sum(clu_sorted == g)) for g in range(n_groups)
    ], dtype=np.int64)

    norm_mode = "count" if normalization == "none" else normalization
    if norm_mode not in ("count", "hz", "hz2", "scale"):
        raise ValueError(
            f"Unknown normalization {normalization!r}; must be one of "
            f"'count', 'hz', 'hz2', 'scale', 'none'."
        )

    # Sign-convention note
    # --------------------
    # The labbox ``CCG.m`` does ``ccg(:, g1, g2) = flipud(counts(:, g1, g2))``
    # for the upper triangle, because MATLAB's column-major reshape of the
    # CCGHeart C output ends up with ``Counts(Bin, Mark2, Mark1)`` — i.e., the
    # second axis is the *other* group and the third is the *centre* group.
    # Our Cython kernel (and the numpy fallback) write directly into a
    # row-major array indexed ``counts[bin, mark1=centre, mark2=other]``.  The
    # natural axis order is therefore the OPPOSITE of MATLAB's, and the
    # flipud belongs on the *lower* triangle, not the upper.  The end result
    # is identical: ``ccg[:, x, y]`` at positive lag means ``y fired after
    # x``, matching the labbox docstring ("conditional intensity of cell y
    # given cell x fired").
    if norm_mode == "count":
        ccg_out = np.zeros((n_bins, n_groups, n_groups), dtype=np.int64)
        axis_unit = "(Spikes)"
    else:
        ccg_out = np.zeros((n_bins, n_groups, n_groups), dtype=np.float64)
        if norm_mode == "hz":
            axis_unit = "(Hz)"
        elif norm_mode == "hz2":
            axis_unit = "(Hz^2)"
        else:  # scale
            axis_unit = "(Scaled)"

    for g1 in range(n_groups):
        for g2 in range(g1, n_groups):
            cnt = counts[:, g1, g2].astype(np.float64)
            if norm_mode == "count":
                factor = 1.0
            elif norm_mode == "hz":
                if n_per[g1] == 0:
                    factor = 0.0
                else:
                    factor = sample_rate / (bin_size * n_per[g1])
            elif norm_mode == "hz2":
                factor = (sample_rate * sample_rate) / (t_range * bin_size)
            else:  # scale
                if n_per[g1] == 0 or n_per[g2] == 0:
                    factor = 0.0
                else:
                    factor = t_range / (bin_size * n_per[g1] * n_per[g2])
            normed = cnt * factor / bias
            if norm_mode == "count":
                ccg_out[:, g1, g2] = normed.astype(np.int64)
                ccg_out[:, g2, g1] = normed[::-1].astype(np.int64)
            else:
                ccg_out[:, g1, g2] = normed
                ccg_out[:, g2, g1] = normed[::-1]

    return CCGResult(
        ccg=ccg_out,
        t=t_axis,
        pairs=pairs,
        bin_size_samples=int(bin_size),
        sample_rate=float(sample_rate),
        n_groups=n_groups,
        group_subset=group_subset,
        normalization=norm_mode,
        n_spikes_per_group=n_per,
        axis_unit=axis_unit,
    )


# ─────────────────────────────────────────────────────────────────────────── #
# Trains2CCG                                                                   #
# ─────────────────────────────────────────────────────────────────────────── #

def trains_to_ccg(
    trains: Sequence[np.ndarray],
    groups: Optional[Sequence[np.ndarray]] = None,
    bin_size_ms: float = 1.0,
    half_bins: int = 50,
    sample_rate: float = 20000.0,
    normalization: str = "scale",
    period: Optional[np.ndarray] = None,
    if_in: bool = True,
) -> CCGResult:
    """Concatenate spike trains and compute their pairwise CCG.

    Port of :file:`labbox/TF/Trains2CCG.m`.

    Convenience wrapper around :func:`ccg`: takes a list of spike-time
    vectors (one per "train"), tags each with a unique group label,
    optionally restricts to a period mask, then dispatches to
    :func:`ccg`.

    Parameters
    ----------
    trains:
        List of 1-D spike-time arrays, one per train.  Times are in
        samples at ``sample_rate`` Hz.
    groups:
        Optional list of 1-D arrays, same length as ``trains``.  If a
        group vector has only one element (or is None), the
        corresponding train gets a single new group label.  If it has
        multiple elements (one per spike), each unique value becomes its
        own output group.  Mirrors the labbox behaviour.
    bin_size_ms:
        Bin size in **milliseconds** (note: :func:`ccg` takes samples).
    half_bins:
        Number of bins on each side of zero.
    sample_rate:
        Hz.  Default 20000.
    normalization:
        Same as :func:`ccg`; default ``"scale"`` (matches labbox).
    period:
        Optional ``(N, 2)`` array of time ranges (in samples) to keep.
        Only spikes in these ranges are used.
    if_in:
        Combined with ``period``: if True (default), keep spikes
        **inside** the periods; if False, keep spikes outside.

    Returns
    -------
    :class:`CCGResult`

    Examples
    --------
    >>> # Auto + cross CCG of 3 units, ±25 ms in 1 ms bins
    >>> result = trains_to_ccg(
    ...     trains=[unit1_times, unit2_times, unit3_times],
    ...     bin_size_ms=1.0, half_bins=25,
    ...     sample_rate=20000.0,
    ... )
    """
    if not trains:
        raise ValueError("trains must contain at least one spike train")

    # Drop empty trains and their group entries (matches labbox)
    if groups is None:
        groups_per_train: list[np.ndarray | None] = [None] * len(trains)
    else:
        if len(groups) != len(trains):
            raise ValueError(
                f"len(groups) must equal len(trains): {len(groups)} vs {len(trains)}"
            )
        groups_per_train = list(groups)

    keep_idx = [
        i for i, tr in enumerate(trains)
        if np.asarray(tr).size > 0 and (
            groups_per_train[i] is None or np.asarray(groups_per_train[i]).size > 0
        )
    ]
    trains = [trains[i] for i in keep_idx]
    groups_per_train = [groups_per_train[i] for i in keep_idx]
    n_trains = len(trains)
    if n_trains == 0:
        raise ValueError("all trains are empty")

    # Build a combined times/groups vector with new contiguous labels.
    combined_times: list[np.ndarray] = []
    combined_groups: list[np.ndarray] = []
    next_label = 1
    for tr, gr in zip(trains, groups_per_train):
        tr_arr = np.asarray(tr, dtype=np.float64).ravel()
        if gr is None or np.asarray(gr).size < 2:
            new_groups = np.full(tr_arr.size, next_label, dtype=np.int64)
            next_label += 1
        else:
            gr_arr = np.asarray(gr).ravel()
            if gr_arr.size != tr_arr.size:
                raise ValueError(
                    f"groups vector for a train must match train length: "
                    f"{gr_arr.size} vs {tr_arr.size}"
                )
            uniq = np.unique(gr_arr)
            remap = {int(u): next_label + k for k, u in enumerate(uniq)}
            new_groups = np.array([remap[int(g)] for g in gr_arr], dtype=np.int64)
            next_label += uniq.size
        combined_times.append(tr_arr)
        combined_groups.append(new_groups)

    times_all  = np.concatenate(combined_times)
    groups_all = np.concatenate(combined_groups)

    # Period filtering
    if period is not None:
        period = np.asarray(period, dtype=np.float64)
        if period.ndim != 2 or period.shape[1] != 2:
            raise ValueError(f"period must have shape (N, 2), got {period.shape}")
        in_periods = _within_ranges_flat(times_all, period)
        keep = in_periods if if_in else ~in_periods
        times_all  = times_all[keep]
        groups_all = groups_all[keep]

    # Convert ms → samples
    bin_size_samples = int(np.ceil(bin_size_ms * sample_rate / 1000.0))

    return ccg(
        times=times_all,
        groups=groups_all,
        bin_size=bin_size_samples,
        half_bins=half_bins,
        sample_rate=sample_rate,
        group_subset=np.arange(1, next_label),
        normalization=normalization,
    )


# ─────────────────────────────────────────────────────────────────────────── #
# Diagnostic                                                                   #
# ─────────────────────────────────────────────────────────────────────────── #

def is_compiled() -> bool:
    """Return True if the Cython kernel is in use, False if the Python fallback.

    Useful for performance-sensitive tests and benchmarks.

    Examples
    --------
    >>> from neurobox.analysis.spikes import is_compiled
    >>> if not is_compiled():
    ...     print("Warning: using slow Python fallback for CCG.")
    """
    return _USING_CYTHON
