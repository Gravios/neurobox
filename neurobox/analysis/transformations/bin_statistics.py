"""
neurobox.analysis.transformations.bin_statistics
================================================
Per-bin descriptive statistics for 2-D and 3-D discretised data.

Ports of:
    MTA/transformations/compute_2d_discrete_func.m
    MTA/transformations/compute_2d_discrete_stats.m
    MTA/transformations/compute_2d_discrete_circ_stats.m
    MTA/transformations/compute_3d_discrete_stats.m

The MATLAB originals all share the same shape: take input arrays
``xcomp.data``, ``ycomp.data`` (and ``zcomp.data`` for the 3-D case),
discretise each via ``edges``, then call ``accumarray`` with various
reducer functions (``@sum``, ``@mean``, ``@median``, ``@std``,
``circ_*``).  The four MATLAB files differ only in which reducers
they apply.

Python implementation
---------------------
This port factors out the discretisation step into a small helper
:class:`BinAxis` and exposes three functions:

* :func:`bin_statistic_2d`  — ``count`` + ``mean`` + ``median`` + ``std``
* :func:`bin_statistic_2d_circ` — circular mean / median / std
* :func:`bin_statistic_3d` — ``count`` + ``mean`` + ``std``

For arbitrary reducers (matching MATLAB's ``compute_2d_discrete_func``)
use :func:`bin_statistic`, which takes a callable.

The MATLAB ``zcomp.data`` is returned as the named field of the result
dataclass (``BinStats2D`` etc.).  Bin centres and edge handling
match MATLAB's ``discretize`` semantics: a sample at exactly ``edge[k]``
goes into bin ``k-1`` (left-inclusive, right-exclusive), and samples
outside the edge range are dropped.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy import stats as _scipy_stats


# ─────────────────────────────────────────────────────────────────────── #
# Shared discretiser                                                        #
# ─────────────────────────────────────────────────────────────────────── #

@dataclass
class BinAxis:
    """One axis of bin edges + centres + indices.

    Mirrors MATLAB's ``xcomp`` / ``ycomp`` / ``zcomp`` struct argument.
    Construct from a data vector and edge array; centres are computed
    automatically; indices via :func:`numpy.digitize`.
    """
    data:  np.ndarray
    edges: np.ndarray
    centres: np.ndarray = None              # type: ignore
    inds:    np.ndarray = None              # type: ignore

    def __post_init__(self) -> None:
        self.edges   = np.asarray(self.edges,  dtype=np.float64).ravel()
        self.data    = np.asarray(self.data,   dtype=np.float64).ravel()
        self.centres = 0.5 * (self.edges[:-1] + self.edges[1:])
        # MATLAB discretize: NaN for points outside edges; bins are
        # left-inclusive / right-exclusive except the last which is
        # closed on both sides.  numpy's digitize gives 0..len(edges)
        # with 0 = below, len(edges) = above; we shift so 1 ⇒ "in
        # first bin", convert out-of-range to the sentinel -1 below.
        idx = np.digitize(self.data, self.edges, right=False) - 1
        idx[idx < 0]                        = -1
        idx[self.data == self.edges[-1]]    = len(self.centres) - 1
        idx[idx >= len(self.centres)]       = -1
        self.inds = idx


# ─────────────────────────────────────────────────────────────────────── #
# Result dataclasses                                                        #
# ─────────────────────────────────────────────────────────────────────── #

@dataclass
class BinStats2D:
    """Result of :func:`bin_statistic_2d`."""
    xctr:   np.ndarray
    yctr:   np.ndarray
    count:  np.ndarray
    mean:   np.ndarray
    median: np.ndarray
    std:    np.ndarray


@dataclass
class BinStats2DCirc:
    """Result of :func:`bin_statistic_2d_circ` — circular reductions."""
    xctr:   np.ndarray
    yctr:   np.ndarray
    count:  np.ndarray
    mean:   np.ndarray
    median: np.ndarray
    std:    np.ndarray


@dataclass
class BinStats3D:
    """Result of :func:`bin_statistic_3d`."""
    xctr:   np.ndarray
    yctr:   np.ndarray
    zctr:   np.ndarray
    count:  np.ndarray
    mean:   np.ndarray
    std:    np.ndarray


@dataclass
class BinStatsArbitrary:
    """Result of :func:`bin_statistic` — arbitrary reducer."""
    xctr:  np.ndarray
    yctr:  np.ndarray
    data:  np.ndarray
    count: np.ndarray


# ─────────────────────────────────────────────────────────────────────── #
# Internal helpers                                                          #
# ─────────────────────────────────────────────────────────────────────── #

def _build_axis(arg) -> BinAxis:
    """Accept either a pre-built :class:`BinAxis` or ``(data, edges)`` tuple."""
    if isinstance(arg, BinAxis):
        return arg
    if isinstance(arg, dict):
        return BinAxis(data=arg["data"], edges=arg["edges"])
    data, edges = arg
    return BinAxis(data=data, edges=edges)


def _accum_2d(
    xind:  np.ndarray,
    yind:  np.ndarray,
    vals:  np.ndarray,
    shape: tuple[int, int],
    reducer: Callable[[np.ndarray], float],
    fill:  float = 0.0,
) -> np.ndarray:
    """Per-bin reduce — equivalent of MATLAB ``accumarray(subs, vals, sz, fcn)``.

    Empty bins become *fill*.
    """
    out = np.full(shape, fill, dtype=np.float64)
    if vals.size == 0:
        return out
    # Linear bin index
    lin = xind * shape[1] + yind
    order = np.argsort(lin, kind="stable")
    lin_s, vals_s = lin[order], vals[order]
    # split points where lin changes
    changes = np.flatnonzero(np.diff(lin_s)) + 1
    starts  = np.concatenate(([0], changes))
    ends    = np.concatenate((changes, [lin_s.size]))
    for s, e in zip(starts, ends):
        bin_lin = int(lin_s[s])
        ix, iy  = divmod(bin_lin, shape[1])
        out[ix, iy] = reducer(vals_s[s:e])
    return out


def _accum_3d(
    xind:  np.ndarray,
    yind:  np.ndarray,
    zind:  np.ndarray,
    vals:  np.ndarray,
    shape: tuple[int, int, int],
    reducer: Callable[[np.ndarray], float],
    fill:  float = 0.0,
) -> np.ndarray:
    out = np.full(shape, fill, dtype=np.float64)
    if vals.size == 0:
        return out
    lin = (xind * shape[1] + yind) * shape[2] + zind
    order = np.argsort(lin, kind="stable")
    lin_s, vals_s = lin[order], vals[order]
    changes = np.flatnonzero(np.diff(lin_s)) + 1
    starts  = np.concatenate(([0], changes))
    ends    = np.concatenate((changes, [lin_s.size]))
    for s, e in zip(starts, ends):
        bin_lin = int(lin_s[s])
        ixy, iz = divmod(bin_lin, shape[2])
        ix,  iy = divmod(ixy,     shape[1])
        out[ix, iy, iz] = reducer(vals_s[s:e])
    return out


# ─────────────────────────────────────────────────────────────────────── #
# Public API                                                                #
# ─────────────────────────────────────────────────────────────────────── #

def _valid_2d(xax: BinAxis, yax: BinAxis, vals: np.ndarray) -> np.ndarray:
    return (
        (xax.inds >= 0) & (yax.inds >= 0) &
        np.isfinite(vals) & (vals != 0.0)
    )


def _valid_3d(xax, yax, zax, vals):
    return (
        (xax.inds >= 0) & (yax.inds >= 0) & (zax.inds >= 0) &
        np.isfinite(vals) & (vals != 0.0)
    )


def bin_statistic_2d(
    xcomp: "BinAxis | tuple",
    ycomp: "BinAxis | tuple",
    vcomp: "BinAxis | tuple | np.ndarray",
) -> BinStats2D:
    """2-D per-bin count + mean + median + std.

    Port of :file:`MTA/transformations/compute_2d_discrete_stats.m`.

    Parameters
    ----------
    xcomp, ycomp:
        :class:`BinAxis` or ``(data, edges)`` tuple for each axis.
    vcomp:
        Either a :class:`BinAxis` / ``(data, edges)`` tuple (in which
        case ``vcomp.data`` are the values to reduce), or a plain
        ``(N,)`` array of values.

    Returns
    -------
    :class:`BinStats2D`
        ``count``, ``mean``, ``median``, ``std`` are all
        ``(len(xctr), len(yctr))`` arrays.

    Notes
    -----
    Validity mask follows MATLAB ``nniz``: samples that are NaN, Inf
    or zero are dropped before reduction.
    """
    xax = _build_axis(xcomp)
    yax = _build_axis(ycomp)
    if isinstance(vcomp, np.ndarray):
        vals = np.asarray(vcomp, dtype=np.float64).ravel()
    elif isinstance(vcomp, BinAxis):
        vals = vcomp.data
    else:
        vals = _build_axis(vcomp).data

    if not (xax.data.size == yax.data.size == vals.size):
        raise ValueError(
            f"size mismatch: x={xax.data.size}, y={yax.data.size}, "
            f"v={vals.size}"
        )

    valid = _valid_2d(xax, yax, vals)
    xind, yind = xax.inds[valid], yax.inds[valid]
    vv = vals[valid]
    shape = (len(xax.centres), len(yax.centres))

    count  = _accum_2d(xind, yind, np.ones_like(vv), shape, np.sum)
    mean   = _accum_2d(xind, yind, vv, shape, np.mean)
    median = _accum_2d(xind, yind, vv, shape, np.median)
    std    = _accum_2d(xind, yind, vv, shape, lambda a: np.std(a, ddof=0))

    return BinStats2D(
        xctr=xax.centres, yctr=yax.centres,
        count=count, mean=mean, median=median, std=std,
    )


def bin_statistic_2d_circ(
    xcomp: "BinAxis | tuple",
    ycomp: "BinAxis | tuple",
    vals:  np.ndarray,
) -> BinStats2DCirc:
    """2-D per-bin **circular** count + mean + median + std (radians).

    Port of :file:`MTA/transformations/compute_2d_discrete_circ_stats.m`.

    The reducers are :func:`scipy.stats.circmean`,
    :func:`scipy.stats.circstd`, and a numpy-native circular median.
    Inputs in ``vals`` are assumed to be angles in radians.
    """
    xax = _build_axis(xcomp)
    yax = _build_axis(ycomp)
    vals = np.asarray(vals, dtype=np.float64).ravel()
    if not (xax.data.size == yax.data.size == vals.size):
        raise ValueError(
            f"size mismatch: x={xax.data.size}, y={yax.data.size}, "
            f"v={vals.size}"
        )

    # vals==0 is allowed for angles, so don't drop it; only drop non-finite
    valid = (
        (xax.inds >= 0) & (yax.inds >= 0) & np.isfinite(vals)
    )
    xind, yind = xax.inds[valid], yax.inds[valid]
    vv = vals[valid]
    shape = (len(xax.centres), len(yax.centres))

    # Use canonical CircStat helpers ported in round 14
    from neurobox.analysis.stats.circular import circ_median as _circ_median_canonical

    def _circ_mean(a):
        return float(_scipy_stats.circmean(a, low=-np.pi, high=np.pi))

    def _circ_std(a):
        return float(_scipy_stats.circstd(a, low=-np.pi, high=np.pi))

    def _circ_median(a):
        # Canonical Berens CircStat circ_median; degrades gracefully on empty
        # bins.  Inputs in (-π, π]; output may be in [0, 2π) — wrap back.
        if a.size == 0:
            return 0.0
        m = float(_circ_median_canonical(a))
        # Wrap to (-π, π] for consistency with mean / std outputs
        return float(np.angle(np.exp(1j * m)))

    count  = _accum_2d(xind, yind, np.ones_like(vv), shape, np.sum)
    mean   = _accum_2d(xind, yind, vv, shape, _circ_mean)
    median = _accum_2d(xind, yind, vv, shape, _circ_median)
    std    = _accum_2d(xind, yind, vv, shape, _circ_std)

    return BinStats2DCirc(
        xctr=xax.centres, yctr=yax.centres,
        count=count, mean=mean, median=median, std=std,
    )


def bin_statistic_3d(
    xcomp: "BinAxis | tuple",
    ycomp: "BinAxis | tuple",
    zcomp: "BinAxis | tuple",
    vals:  np.ndarray,
) -> BinStats3D:
    """3-D per-bin count + mean + std.

    Port of :file:`MTA/transformations/compute_3d_discrete_stats.m`.
    """
    xax = _build_axis(xcomp)
    yax = _build_axis(ycomp)
    zax = _build_axis(zcomp)
    vals = np.asarray(vals, dtype=np.float64).ravel()
    if not (xax.data.size == yax.data.size == zax.data.size == vals.size):
        raise ValueError(
            f"size mismatch: x={xax.data.size}, y={yax.data.size}, "
            f"z={zax.data.size}, v={vals.size}"
        )

    valid = _valid_3d(xax, yax, zax, vals)
    xind, yind, zind = xax.inds[valid], yax.inds[valid], zax.inds[valid]
    vv = vals[valid]
    shape = (len(xax.centres), len(yax.centres), len(zax.centres))

    count = _accum_3d(xind, yind, zind, np.ones_like(vv), shape, np.sum)
    mean  = _accum_3d(xind, yind, zind, vv, shape, np.mean)
    std   = _accum_3d(xind, yind, zind, vv, shape, lambda a: np.std(a, ddof=0))

    return BinStats3D(
        xctr=xax.centres, yctr=yax.centres, zctr=zax.centres,
        count=count, mean=mean, std=std,
    )


def bin_statistic(
    xcomp:        "BinAxis | tuple",
    ycomp:        "BinAxis | tuple",
    vals:         np.ndarray,
    func_handle:  Callable[[np.ndarray], float],
    fill_value:   float = 0.0,
) -> BinStatsArbitrary:
    """2-D per-bin reduction with a user-supplied callable.

    Port of :file:`MTA/transformations/compute_2d_discrete_func.m`.

    Parameters
    ----------
    xcomp, ycomp:
        :class:`BinAxis` or ``(data, edges)`` tuple.
    vals:
        ``(N,)`` array of values to reduce.
    func_handle:
        Reducer callable accepting a 1-D array and returning a scalar
        (e.g. ``np.sum``, ``np.mean``, ``len``).
    fill_value:
        Value placed in empty bins.  Default 0.

    Returns
    -------
    :class:`BinStatsArbitrary`
    """
    xax = _build_axis(xcomp)
    yax = _build_axis(ycomp)
    vals = np.asarray(vals, dtype=np.float64).ravel()
    if not (xax.data.size == yax.data.size == vals.size):
        raise ValueError(
            f"size mismatch: x={xax.data.size}, y={yax.data.size}, "
            f"v={vals.size}"
        )

    valid = _valid_2d(xax, yax, vals)
    xind, yind = xax.inds[valid], yax.inds[valid]
    vv = vals[valid]
    shape = (len(xax.centres), len(yax.centres))

    data  = _accum_2d(xind, yind, vv,                shape, func_handle, fill_value)
    count = _accum_2d(xind, yind, np.ones_like(vv),  shape, np.sum)
    return BinStatsArbitrary(
        xctr=xax.centres, yctr=yax.centres, data=data, count=count,
    )
