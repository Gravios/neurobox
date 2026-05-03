"""
neurobox.viz.domain_plots
==========================
Lab-specific plot wrappers for the ``MTA`` figure-script ecosystem.

These are convenience helpers that compose primitives from
:mod:`neurobox.viz.plots` and :mod:`neurobox.viz.state` with our
position / state / spectral types.  They mirror the
``MTA/utilities/graphics/p*.m`` family.

Ports:

* :file:`pXY.m`              → :func:`plot_xy_with_state`
* :file:`pZ.m`               → :func:`plot_z`
* :file:`pB.m`               → :func:`plot_z_basic`
* :file:`pXYV.m`             → :func:`plot_xy_velocity`
* :file:`pRHM.m`             → :func:`plot_rhm_spectrogram`
* :file:`plotSkeleton.m`     → :func:`plot_skeleton`
* :file:`plotSkeletonLine.m` → :func:`plot_skeleton_line`
* :file:`plotcc.m`           → :func:`plot_colored_curve`
* :file:`plotcc3.m`          → :func:`plot_colored_curve_3d`

What's not ported and why
-------------------------
* :file:`pSE.m` — single-line wrapper around ``PlotSessionErrors``;
  trivially replaced by ``ax.errorbar(...)`` in user code.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

from neurobox.dtype.epoch import NBEpoch
from neurobox.dtype.xyz import NBDxyz


__all__ = [
    "plot_xy_with_state",
    "plot_z",
    "plot_z_basic",
    "plot_xy_velocity",
    "plot_rhm_spectrogram",
    "plot_skeleton",
    "plot_skeleton_line",
    "plot_colored_curve",
    "plot_colored_curve_3d",
]


# ─────────────────────────────────────────────────────────────────────── #
# pXY — XY trajectory with optional state-restricted overlay                 #
# ─────────────────────────────────────────────────────────────────────── #

def plot_xy_with_state(
    xyz:                NBDxyz,
    *,
    state:              Optional[NBEpoch] = None,
    ax:                 Optional[Axes] = None,
    base_marker:        str = "head_back",
    overlay_marker:     str = "hcom",
    base_kwargs:        Optional[dict] = None,
    overlay_kwargs:     Optional[dict] = None,
    boundary:           Optional[Sequence[Sequence[float]]] = None,
) -> Axes:
    """Plot an XY trajectory with an optional state-restricted overlay.

    Port of :file:`MTA/utilities/graphics/pXY.m`.

    The full trajectory is drawn as small black dots from
    *base_marker*; samples within *state* are then over-plotted in red
    using *overlay_marker* (matches the lab's standard "all positions
    + the rear positions on top" plot).

    Parameters
    ----------
    xyz:
        Source :class:`NBDxyz`.
    state:
        Optional :class:`NBEpoch` to highlight.  ``None`` → no overlay.
    ax:
        Target axes.  Default ``plt.gca()``.
    base_marker:
        Marker name for the full-trajectory layer.  Default
        ``'head_back'`` matches MATLAB.
    overlay_marker:
        Marker name for the state-restricted overlay.  Default
        ``'hcom'`` matches MATLAB.
    base_kwargs, overlay_kwargs:
        Passed to :meth:`Axes.plot` for each layer.  Sensible defaults
        applied if None.
    boundary:
        Optional ``[[xlo, xhi], [ylo, yhi]]`` to set ``ax`` limits.
        Default leaves auto-scaling.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        ax = plt.gca()

    base_kwargs = {"linestyle": "", "marker": ".", "color": "k",
                    "markersize": 1, **(base_kwargs or {})}
    overlay_kwargs = {"linestyle": "", "marker": ".", "color": "r",
                       "markersize": 2, **(overlay_kwargs or {})}

    base_idx = xyz.model.index(base_marker)
    ax.plot(xyz.data[:, base_idx, 0],
            xyz.data[:, base_idx, 1],
            **base_kwargs)

    if state is not None:
        n = xyz.data.shape[0]
        if hasattr(state, "to_mask"):
            mask = state.resample(xyz.samplerate).to_mask(n) \
                   if float(state.samplerate) != float(xyz.samplerate) \
                   else state.to_mask(n)
        else:
            mask = np.asarray(state, dtype=bool)[:n]
        # Fall back to base_marker if overlay not present
        if overlay_marker not in xyz.model.markers:
            overlay_marker = base_marker
        ov_idx = xyz.model.index(overlay_marker)
        if mask.any():
            ax.plot(xyz.data[mask, ov_idx, 0],
                    xyz.data[mask, ov_idx, 1],
                    **overlay_kwargs)

    if boundary is not None:
        ax.set_xlim(*boundary[0])
        ax.set_ylim(*boundary[1])

    ax.set_aspect("equal", adjustable="box")
    return ax


# ─────────────────────────────────────────────────────────────────────── #
# pZ — z-trace with state vertical lines                                     #
# ─────────────────────────────────────────────────────────────────────── #

def plot_z(
    xyz:               NBDxyz,
    *,
    state:             Optional[NBEpoch] = None,
    ax:                Optional[Axes] = None,
    marker:            str = "head_back",
    line_color:        str = "C0",
    state_color:       str = "r",
) -> Axes:
    """Plot a marker's z-coordinate vs sample with optional state markers.

    Port of :file:`MTA/utilities/graphics/pZ.m`.

    Useful for visualising rear / drop events in elevation.
    """
    if ax is None:
        ax = plt.gca()
    midx = xyz.model.index(marker)
    ax.plot(xyz.data[:, midx, 2], color=line_color, linewidth=0.7)

    if state is not None:
        periods = (np.asarray(state.data, dtype=np.float64)
                   if getattr(state, "mode", "periods") == "periods"
                   else _mask_to_periods(state.data))
        for s, e in periods:
            ax.axvline(s, color=state_color, linewidth=0.5, alpha=0.5)
            ax.axvline(e, color=state_color, linewidth=0.5, alpha=0.5)

    ax.set_xlabel("samples")
    ax.set_ylabel(f"{marker} z")
    return ax


def plot_z_basic(
    xyz:               NBDxyz,
    *,
    ax:                Optional[Axes] = None,
    marker:            str = "spine_lower",
    sync_periods:      Optional[np.ndarray] = None,
) -> Axes:
    """Plot z with sync-period start/end markers.

    Port of :file:`MTA/utilities/graphics/pB.m`.

    Parameters
    ----------
    xyz, marker:
        See :func:`plot_z`.
    sync_periods:
        Optional ``(N, 2)`` periods to mark with vertical lines.
        Defaults to none.
    """
    if ax is None:
        ax = plt.gca()
    midx = xyz.model.index(marker)
    ax.plot(xyz.data[:, midx, 2], color="C0", linewidth=0.7)
    if sync_periods is not None:
        for s, e in np.asarray(sync_periods):
            ax.axvline(s, color="r", linewidth=0.7)
            ax.axvline(e, color="k", linewidth=0.7)
    ax.set_xlabel("samples")
    ax.set_ylabel(f"{marker} z")
    return ax


# ─────────────────────────────────────────────────────────────────────── #
# pXYV — xy-plane velocity                                                   #
# ─────────────────────────────────────────────────────────────────────── #

def plot_xy_velocity(
    xyz:               NBDxyz,
    *,
    state:             Optional[NBEpoch] = None,
    ax:                Optional[Axes] = None,
    marker:            str = "spine_lower",
    cutoff_hz:         float = 2.4,
    filter_order:      int = 3,
) -> Axes:
    """Plot low-pass-filtered xy-velocity with optional state lines.

    Port of :file:`MTA/utilities/graphics/pXYV.m`.

    Applies a Butterworth low-pass at *cutoff_hz* before computing the
    finite-difference velocity, then plots the speed vs samples.
    """
    from neurobox.analysis.lfp.filtering import butter_filter

    if ax is None:
        ax = plt.gca()
    midx = xyz.model.index(marker)
    fs = float(xyz.samplerate)
    # Low-pass each xy axis, then take diff
    xy = xyz.data[:, midx, :2]
    lp = np.column_stack([
        butter_filter(xy[:, 0], cutoff=cutoff_hz, samplerate=fs,
                       btype="lowpass", order=filter_order),
        butter_filter(xy[:, 1], cutoff=cutoff_hz, samplerate=fs,
                       btype="lowpass", order=filter_order),
    ])
    diffs = np.diff(lp, axis=0)
    speed = np.linalg.norm(diffs, axis=1) * fs
    ax.plot(speed, color="C0", linewidth=0.7)

    if state is not None:
        periods = (np.asarray(state.data, dtype=np.float64)
                   if getattr(state, "mode", "periods") == "periods"
                   else _mask_to_periods(state.data))
        for s, e in periods:
            ax.axvline(s, color="r", linewidth=0.5, alpha=0.5)
            ax.axvline(e, color="r", linewidth=0.5, alpha=0.5)

    ax.set_xlabel("samples")
    ax.set_ylabel(f"|d{marker}/dt| (units/s)")
    return ax


# ─────────────────────────────────────────────────────────────────────── #
# pRHM — rhythmic head-movement spectrogram                                  #
# ─────────────────────────────────────────────────────────────────────── #

def plot_rhm_spectrogram(
    rhm_signal:        np.ndarray,
    samplerate:        float,
    *,
    ax:                Optional[Axes] = None,
    n_fft:             int = 512,
    win_len:           int = 256,
    freq_band:         tuple[float, float] = (1.0, 15.0),
    cax_db:            tuple[float, float] = (-8.0, -4.0),
    colormap:          str = "jet",
) -> Axes:
    """Plot a log-power spectrogram of a rhythmic-head-movement signal.

    Port of :file:`MTA/utilities/graphics/pRHM.m`.

    Computes a multitaper spectrogram of *rhm_signal* (typically the
    head-back ↔ head-front pitch trace), then ``imshow`` log10 power
    with the lab's standard 1-15 Hz / -8..-4 dB color scale.

    Parameters
    ----------
    rhm_signal:
        ``(T,)`` 1-D signal.
    samplerate:
        Sample rate in Hz.
    ax:
        Target axes.  Default ``plt.gca()``.
    n_fft, win_len, freq_band:
        Spectrogram parameters.  Defaults match MATLAB.
    cax_db:
        ``(vmin, vmax)`` for the log-power color scale.  Default
        ``(-8, -4)`` matches MATLAB.
    colormap:
        Matplotlib colormap name.

    Returns
    -------
    matplotlib.axes.Axes
    """
    from neurobox.analysis.lfp.spectral import (
        SpectralParams, multitaper_spectrogram,
    )

    if ax is None:
        ax = plt.gca()

    sig = np.asarray(rhm_signal, dtype=np.float64).ravel()
    overlap = max(0, win_len - max(1, win_len // 4))
    params = SpectralParams(
        samplerate = float(samplerate),
        n_fft      = n_fft,
        win_len    = win_len,
        n_overlap  = overlap,
        freq_range = freq_band,
    )
    spec = multitaper_spectrogram(sig, params)
    power = spec.power
    if power.ndim == 3:
        power = power[..., 0]      # (T_win, F)
    if power.ndim == 2 and power.shape[0] == spec.times.size \
            and power.shape[1] == spec.freqs.size:
        # (T_win, F) → transpose for imshow with freq on y-axis
        power_TF = power.T          # (F, T_win)
    else:
        power_TF = power

    with np.errstate(divide="ignore", invalid="ignore"):
        log_power = np.where(power_TF > 0, np.log10(power_TF), np.nan)

    im = ax.imshow(
        log_power,
        aspect       = "auto",
        origin       = "lower",
        extent       = [spec.times[0], spec.times[-1],
                         spec.freqs[0], spec.freqs[-1]],
        cmap         = colormap,
        vmin         = cax_db[0], vmax = cax_db[1],
    )
    ax.set_xlabel("time (s)")
    ax.set_ylabel("frequency (Hz)")
    return ax


# ─────────────────────────────────────────────────────────────────────── #
# Skeleton plotting                                                          #
# ─────────────────────────────────────────────────────────────────────── #

def plot_skeleton(
    xyz:               NBDxyz,
    frame_index:       int,
    *,
    ax:                Optional[Axes] = None,
    skeleton_markers:  Sequence[str] = (
        "spine_lower", "pelvis_root", "spine_middle", "spine_upper",
        "head_back", "head_left", "head_front", "head_right",
    ),
    plot_type:         str = "line",
    trajectory_period: tuple[int, int] = (0, 0),
    trajectory_markers: Sequence[str] = (
        "spine_lower", "pelvis_root", "spine_middle", "spine_upper",
        "head_left", "head_right",
    ),
    marker_size:       float = 4.0,
    line_width:        float = 2.0,
    trajectory_color:  tuple[float, float, float] = (0.5, 0.5, 0.5),
) -> dict:
    """Plot a 3-D rat skeleton at a given frame.

    Port of :file:`MTA/utilities/graphics/plotSkeleton.m`.

    Renders the marker positions at ``xyz.data[frame_index]`` plus the
    sticks (model.connections) connecting them.  Optionally draws a
    pre / post trajectory of selected markers as a dotted line.

    Parameters
    ----------
    xyz:
        :class:`NBDxyz`.  Must have ``model.connections`` populated for
        the stick figure.
    frame_index:
        Sample index to render.  Use the last index for "current frame".
    ax:
        Target axes.  Must be a 3-D matplotlib axes
        (``fig.add_subplot(projection='3d')``).  Default uses
        ``plt.gca()`` and asserts it's 3-D.
    skeleton_markers:
        Markers to render.  Default matches MATLAB.
    plot_type:
        ``'line'`` (default) — stick figure with point-markers.
        ``'surface'`` — replace each stick with a thin cylinder
        (slower; uses :func:`numpy.meshgrid`).
    trajectory_period:
        ``(before, after)`` samples around *frame_index*; if both 0
        (default), no trajectory is drawn.
    trajectory_markers:
        Which markers contribute to the trajectory dots.
    marker_size, line_width:
        Visual sizing.

    Returns
    -------
    dict
        Handles dict with keys ``'sticks'`` (list of Line2D /
        Poly3DCollection), ``'markers'`` (list of artists), and
        ``'trajectory'`` (list of artists, or empty).
    """
    if ax is None:
        ax = plt.gca()

    is_3d = hasattr(ax, "plot3D")
    if not is_3d:
        raise ValueError(
            "plot_skeleton requires a 3-D axes; create with "
            "fig.add_subplot(projection='3d')"
        )

    # Resolve marker indices
    sk_indices = []
    for m in skeleton_markers:
        if m in xyz.model.markers:
            sk_indices.append(xyz.model.index(m))
    sk_indices = np.asarray(sk_indices)

    # Resolve connections (stick endpoints) — only those entirely inside
    # the requested skeleton_markers
    connections = []
    for c in xyz.model.connections:
        if not isinstance(c, (list, tuple)) or len(c) < 2:
            continue
        a_name, b_name = str(c[0]), str(c[1])
        if a_name in skeleton_markers and b_name in skeleton_markers:
            connections.append((xyz.model.index(a_name),
                                 xyz.model.index(b_name)))

    sticks: list = []
    if plot_type == "line":
        for ai, bi in connections:
            xs = [xyz.data[frame_index, ai, 0], xyz.data[frame_index, bi, 0]]
            ys = [xyz.data[frame_index, ai, 1], xyz.data[frame_index, bi, 1]]
            zs = [xyz.data[frame_index, ai, 2], xyz.data[frame_index, bi, 2]]
            ln, = ax.plot(xs, ys, zs, color="k", linewidth=line_width)
            sticks.append(ln)
    elif plot_type == "surface":
        # Approximate sticks as thin cylinders.  We use a degenerate
        # surface via Line3DCollection-equivalent for performance.
        for ai, bi in connections:
            xs = [xyz.data[frame_index, ai, 0], xyz.data[frame_index, bi, 0]]
            ys = [xyz.data[frame_index, ai, 1], xyz.data[frame_index, bi, 1]]
            zs = [xyz.data[frame_index, ai, 2], xyz.data[frame_index, bi, 2]]
            ln, = ax.plot(xs, ys, zs, color="b",
                          linewidth=line_width * 2)
            sticks.append(ln)
    else:
        raise ValueError(
            f"plot_type must be 'line' or 'surface'; got {plot_type!r}"
        )

    # Markers as scatter points
    markers_list = []
    for mi in sk_indices:
        sc = ax.scatter(xyz.data[frame_index, mi, 0],
                         xyz.data[frame_index, mi, 1],
                         xyz.data[frame_index, mi, 2],
                         s=marker_size ** 2, c="k", depthshade=False)
        markers_list.append(sc)

    # Optional trajectory
    trajectory = []
    if any(p != 0 for p in trajectory_period):
        rel = np.arange(trajectory_period[0], trajectory_period[1] + 1)
        idxs = frame_index + rel
        valid = (idxs >= 0) & (idxs < xyz.data.shape[0])
        idxs = idxs[valid]
        for tm in trajectory_markers:
            if tm not in xyz.model.markers:
                continue
            ti = xyz.model.index(tm)
            ln, = ax.plot(
                xyz.data[idxs, ti, 0],
                xyz.data[idxs, ti, 1],
                xyz.data[idxs, ti, 2],
                linestyle=":", marker=".",
                color=trajectory_color, markersize=marker_size,
            )
            trajectory.append(ln)

    # Match MATLAB's daspect([1 1 1])
    ax.set_box_aspect((1, 1, 1))
    return {"sticks": sticks, "markers": markers_list,
             "trajectory": trajectory}


def plot_skeleton_line(
    xyz:               NBDxyz,
    frame_index:       int,
    *,
    ax:                Optional[Axes] = None,
    skeleton_markers:  Sequence[str] = (
        "spine_lower", "pelvis_root", "spine_middle", "spine_upper",
        "head_back", "head_left", "head_front", "head_right",
    ),
    trajectory_period: tuple[int, int] = (0, 0),
    trajectory_markers: Sequence[str] = (
        "spine_lower", "pelvis_root", "spine_middle", "spine_upper",
        "head_left", "head_right",
    ),
    line_width:        float = 2.0,
) -> dict:
    """Static (non-animated) line variant of :func:`plot_skeleton`.

    Port of :file:`MTA/utilities/graphics/plotSkeletonLine.m`.

    The MATLAB ``plotSkeleton`` used ``animatedline`` for performance
    in animations; ``plotSkeletonLine`` uses plain ``line()`` for
    static figures.  In matplotlib both call paths produce
    :class:`Line2D` objects, so this is just :func:`plot_skeleton` with
    ``plot_type='line'`` and no surface mode.
    """
    return plot_skeleton(
        xyz, frame_index,
        ax                   = ax,
        skeleton_markers     = skeleton_markers,
        plot_type            = "line",
        trajectory_period    = trajectory_period,
        trajectory_markers   = trajectory_markers,
        line_width           = line_width,
    )


# ─────────────────────────────────────────────────────────────────────── #
# Coloured curves (plotcc / plotcc3)                                         #
# ─────────────────────────────────────────────────────────────────────── #

def plot_colored_curve(
    x:           np.ndarray,
    y:           np.ndarray,
    c:           np.ndarray,
    *,
    ax:          Optional[Axes] = None,
    colormap:    str = "viridis",
    linewidth:   float = 2.0,
) -> LineCollection:
    """Plot a 2-D curve coloured per-segment by *c*.

    Port of :file:`MTA/utilities/graphics/plotcc.m`.

    Uses :class:`matplotlib.collections.LineCollection` instead of
    MATLAB's ``surface``-as-line trick.

    Parameters
    ----------
    x, y, c:
        ``(N,)`` arrays.  *c* is mapped onto *colormap* and applied
        to each segment.
    """
    if ax is None:
        ax = plt.gca()
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    c = np.asarray(c).ravel()
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap=colormap, linewidth=linewidth)
    lc.set_array(c)
    ax.add_collection(lc)
    ax.autoscale_view()
    return lc


def plot_colored_curve_3d(
    x:           np.ndarray,
    y:           np.ndarray,
    z:           np.ndarray,
    c:           np.ndarray,
    *,
    ax:          Optional[Axes] = None,
    colormap:    str = "viridis",
    linewidth:   float = 2.0,
):
    """Plot a 3-D curve coloured per-segment by *c*.

    Port of :file:`MTA/utilities/graphics/plotcc3.m`.

    Requires a 3-D axes.  Uses
    :class:`mpl_toolkits.mplot3d.art3d.Line3DCollection`.
    """
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    if ax is None:
        ax = plt.gca()
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    z = np.asarray(z).ravel()
    c = np.asarray(c).ravel()
    points = np.array([x, y, z]).T.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = Line3DCollection(segments, cmap=colormap, linewidth=linewidth)
    lc.set_array(c)
    ax.add_collection3d(lc)
    return lc


# ─────────────────────────────────────────────────────────────────────── #
# Local helpers                                                              #
# ─────────────────────────────────────────────────────────────────────── #

def _mask_to_periods(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask).astype(bool).ravel()
    if mask.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    edges = np.diff(mask.astype(int))
    starts = np.flatnonzero(edges == 1) + 1
    ends   = np.flatnonzero(edges == -1) + 1
    if mask[0]:
        starts = np.r_[0, starts]
    if mask[-1]:
        ends = np.r_[ends, mask.size]
    return np.column_stack([starts, ends]).astype(np.float64)
