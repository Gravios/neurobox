"""
neurobox.viz.plots
===================
Domain plot primitives — port of the matplotlib-translatable subset of
:file:`MTA/utilities/graphics/`.

Ports of:

* :file:`imagescnan.m`     → :func:`imagescnan` — NaN-aware imshow
* :file:`error_ellipse.m`  → :func:`error_ellipse` — covariance-ellipse plot
* :file:`apply_colorbar.m` → :func:`apply_colorbar` — cm-positioned colorbar
* :file:`circular_arrow.m` → :func:`circular_arrow` — arc-shaped arrow
* :file:`draw_arrow.m`     → :func:`draw_arrow` — straight or curved arrow
* :file:`draw_lines.m`     → :func:`draw_lines` — interactive line drawing
* :file:`sbar.m`           → :func:`sbar` — overlaid state-restricted histogram

What's not ported and why
-------------------------
* ``frameSet_series``, ``frameSet2im``, ``frameSet_viewer`` —
  movie-rendering helpers; replaced by ``matplotlib.animation`` in
  user code.
* ``ClusterH2``, ``ClusterPP`` — interactive cluster-cutting GUIs;
  belong in a future ``neurobox.gui`` round, not in viz.
* ``hist2c``, ``histO`` — thin wrappers over MATLAB's ``histc``;
  use ``matplotlib.pyplot.hist2d`` / ``ax.hist`` directly.
* ``label_frames``, ``set_parameters_scatter`` — one-liner helpers.
* The lab-specific domain plotters ``pXY``, ``pZ``, ``pB``, ``pSE``,
  ``pXYV``, ``pRHM`` — these are 5-15 line wrappers over standard
  plotting that depend heavily on lab-specific feature names; port on
  demand alongside the figure scripts that consume them.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.colorbar import Colorbar
from matplotlib.colors import Colormap, ListedColormap, Normalize
from matplotlib.image import AxesImage
from matplotlib.patches import Ellipse, FancyArrow, FancyArrowPatch


__all__ = [
    "imagescnan",
    "error_ellipse",
    "apply_colorbar",
    "circular_arrow",
    "draw_arrow",
    "draw_lines",
    "sbar",
]


# ─────────────────────────────────────────────────────────────────── #
# imagescnan                                                            #
# ─────────────────────────────────────────────────────────────────── #

def imagescnan(
    image_data:           np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    ax:                   Optional[Axes] = None,
    color_limits:         Optional[tuple[float, float] | str] = None,
    data_type:            str = "linear",
    nan_rgb:              tuple[float, float, float] = (0.5, 0.5, 0.5),
    colormap:             str | Colormap = "viridis",
    colormap_flip:        bool = False,
    add_colorbar:         bool = False,
    interpolation:        str = "nearest",
) -> tuple[AxesImage, Optional[Colorbar]]:
    """NaN-aware imshow with explicit NaN colour and clip handling.

    Port of :file:`MTA/utilities/graphics/imagescnan.m`.

    Parameters
    ----------
    image_data:
        Either a 2-D ``(M, N)`` array, or a tuple
        ``(x_axis, y_axis, matrix)`` to set explicit data-coords on
        the imshow domain.
    ax:
        Target matplotlib axes.  Default ``None`` → use ``plt.gca()``.
    color_limits:
        ``(vmin, vmax)`` to set the colour scale, or the string
        ``'sym'`` for symmetric ±max·0.7 (matches MATLAB).  ``None``
        → auto-fit to the non-NaN data range.
    data_type:
        ``'linear'`` (default) or ``'circular'``.  For ``'circular'``,
        no clipping is applied at the limits (values wrap).
    nan_rgb:
        RGB triple in ``[0, 1]`` used to colour NaN pixels.  Default
        mid-grey matches MATLAB's ``[0.5, 0.5, 0.5]``.
    colormap:
        matplotlib colormap name or :class:`Colormap`.  Default
        ``'viridis'`` (matplotlib's analogue of MATLAB's ``parula``).
    colormap_flip:
        Reverse the colormap before applying.
    add_colorbar:
        If True, attach a colorbar to *ax* and return it as the second
        return value.
    interpolation:
        Forwarded to ``ax.imshow``.  Default ``'nearest'`` matches
        MATLAB's ``image()``.

    Returns
    -------
    image : matplotlib.image.AxesImage
        The imshow handle.
    colorbar : matplotlib.colorbar.Colorbar or None
        The colorbar, or None if ``add_colorbar`` was False.
    """
    # Unpack input
    if isinstance(image_data, tuple):
        x_axis, y_axis, matrix = image_data
        x_axis = np.asarray(x_axis, dtype=np.float64)
        y_axis = np.asarray(y_axis, dtype=np.float64)
        matrix = np.asarray(matrix, dtype=np.float64)
        # imshow's `extent` is [left, right, bottom, top]
        dx = (x_axis[1] - x_axis[0]) / 2 if len(x_axis) > 1 else 0.5
        dy = (y_axis[1] - y_axis[0]) / 2 if len(y_axis) > 1 else 0.5
        extent = (x_axis[0] - dx, x_axis[-1] + dx,
                  y_axis[-1] + dy, y_axis[0] - dy)
    else:
        matrix = np.asarray(image_data, dtype=np.float64)
        extent = None

    if ax is None:
        ax = plt.gca()

    # Resolve color limits
    if color_limits is None:
        finite = matrix[np.isfinite(matrix)]
        if finite.size == 0:
            vmin, vmax = 0.0, 1.0
        else:
            vmin, vmax = float(finite.min()), float(finite.max())
            if vmin == vmax:
                vmax = vmin + 1.0
    elif isinstance(color_limits, str) and color_limits == "sym":
        finite = matrix[np.isfinite(matrix)]
        amax = float(np.max(np.abs(finite))) if finite.size else 1.0
        vmin, vmax = -amax * 0.7, amax * 0.7
    else:
        vmin, vmax = float(color_limits[0]), float(color_limits[1])

    # Resolve colormap
    if isinstance(colormap, str):
        cmap = plt.get_cmap(colormap)
    else:
        cmap = colormap
    if colormap_flip:
        cmap = cmap.reversed()

    # Mark NaNs in the colormap
    cmap = cmap.copy() if hasattr(cmap, "copy") else cmap
    cmap.set_bad(color=nan_rgb)

    # Clip to limits for circular data — MATLAB behaviour folds outside
    # values to the limit; for linear data we clip identically.  Both
    # imshow calls also pass `vmin/vmax` so out-of-range values render
    # at the colormap endpoints.
    masked = np.ma.masked_invalid(matrix)

    norm = Normalize(vmin=vmin, vmax=vmax, clip=(data_type == "linear"))

    im = ax.imshow(
        masked,
        cmap          = cmap,
        norm          = norm,
        extent        = extent,
        origin        = "upper",
        interpolation = interpolation,
        aspect        = "auto",
    )

    cb = None
    if add_colorbar:
        cb = ax.figure.colorbar(im, ax=ax)

    return im, cb


# ─────────────────────────────────────────────────────────────────── #
# error_ellipse                                                         #
# ─────────────────────────────────────────────────────────────────── #

def error_ellipse(
    cov:        np.ndarray,
    mu:         Optional[np.ndarray] = None,
    *,
    ax:         Optional[Axes] = None,
    conf:       float = 0.5,
    scale:      float = 1.0,
    n_points:   int = 100,
    clip_radius: float = np.inf,
    **plot_kwargs: Any,
) -> Any:
    """Plot a confidence-ellipse from a 2-D covariance matrix.

    Port of :file:`MTA/utilities/graphics/error_ellipse.m`.

    Parameters
    ----------
    cov:
        ``(2, 2)`` covariance matrix.  Must be positive-definite.
    mu:
        ``(2,)`` ellipse centre.  Default ``(0, 0)``.
    ax:
        Target axes.  Default ``plt.gca()``.
    conf:
        Confidence level in ``(0, 1)``.  Default 0.5 (50 % ellipse).
    scale:
        Scale factor applied to the radius — useful for unit
        conversion (e.g. mm → m).
    n_points:
        Number of points around the ellipse.  Default 100.
    clip_radius:
        Points outside this Euclidean radius from *mu* are clipped
        (rendered as NaN).  Default ``inf`` → no clipping.
    **plot_kwargs:
        Forwarded to ``ax.plot``.  Use ``color=...``, ``linewidth=...``,
        etc.  Note: this port draws the ellipse as a polyline rather
        than as an :class:`matplotlib.patches.Ellipse`, matching
        MATLAB's behaviour and giving the user direct access to the
        line-style kwargs.

    Returns
    -------
    matplotlib.lines.Line2D
        Handle to the plotted ellipse outline.

    Notes
    -----
    The 3-D ellipsoid form of the MATLAB original is **not** ported
    in this round — defer to a future round if 3-D covariance plotting
    becomes necessary.
    """
    cov = np.asarray(cov, dtype=np.float64)
    if cov.shape != (2, 2):
        raise ValueError(
            f"error_ellipse expects a 2x2 covariance matrix; got {cov.shape}"
        )
    eigvals = np.linalg.eigvalsh(cov)
    if (eigvals <= 0).any():
        raise ValueError(
            "covariance matrix must be positive definite "
            f"(eigenvalues = {eigvals.tolist()})"
        )

    if mu is None:
        mu = np.zeros(2, dtype=np.float64)
    else:
        mu = np.asarray(mu, dtype=np.float64).ravel()
        if mu.shape != (2,):
            raise ValueError(f"mu must be (2,); got {mu.shape}")

    if not (0.0 < conf < 1.0):
        raise ValueError(f"conf must be in (0, 1); got {conf}")

    # Quantile factor: sqrt of chi2 quantile with 2 d.o.f.
    from scipy.stats import chi2
    k = float(np.sqrt(chi2.ppf(conf, df=2)))

    # Eigendecomposition for the principal axes
    eigval, eigvec = np.linalg.eigh(cov)
    # MATLAB's `[cos(p), sin(p)] * sqrt(eigval) * eigvec'`
    p = np.linspace(0.0, 2.0 * np.pi, n_points + 1)
    unit = np.column_stack([np.cos(p), np.sin(p)])
    xy = unit @ np.diag(np.sqrt(eigval)) @ eigvec.T
    x = xy[:, 0]
    y = xy[:, 1]

    # Clip far points
    if np.isfinite(clip_radius):
        r = np.hypot(x, y)
        x = np.where(r > clip_radius, np.nan, x)
        y = np.where(r > clip_radius, np.nan, y)

    if ax is None:
        ax = plt.gca()
    line, = ax.plot(scale * (mu[0] + k * x),
                     scale * (mu[1] + k * y),
                     **plot_kwargs)
    return line


# ─────────────────────────────────────────────────────────────────── #
# apply_colorbar                                                        #
# ─────────────────────────────────────────────────────────────────── #

def apply_colorbar(
    ax:        Axes,
    *,
    location:  str = "eastoutside",
    colormap:  Optional[str | Colormap] = None,
    pad_cm:    float = 0.05,
    width_cm:  float = 0.15,
    label:     Optional[str] = None,
) -> Colorbar:
    """Attach a cm-positioned colorbar to *ax*.

    Port of :file:`MTA/utilities/graphics/apply_colorbar.m`.

    Unlike matplotlib's default ``Figure.colorbar`` which steals
    space from the parent axes, this places the colorbar at fixed
    cm offsets so the parent axes keeps its layout-managed position.

    Parameters
    ----------
    ax:
        Parent axes.  Must already contain an image / collection
        with a colormap.
    location:
        ``'eastoutside'`` → right of *ax* (default), or
        ``'southoutside'`` → below.  Other matplotlib locations are
        not supported by this port.
    colormap:
        Optional colormap to apply to *ax*'s image before creating
        the colorbar.  If ``None``, uses the existing colormap.
    pad_cm, width_cm:
        Cm-precise spacing and bar width.  Defaults match MATLAB.
    label:
        Optional colorbar label.

    Returns
    -------
    matplotlib.colorbar.Colorbar
    """
    fig = ax.figure
    if colormap is not None:
        # Apply colormap to the most recent image / collection on ax
        for art in reversed(ax.get_images() + ax.collections):
            if hasattr(art, "set_cmap"):
                art.set_cmap(colormap)
                break

    # Figure-fraction position of the parent axes
    pos = ax.get_position()
    fig_w_in, fig_h_in = fig.get_size_inches()
    fig_w_cm = fig_w_in * 2.54
    fig_h_cm = fig_h_in * 2.54

    pad_frac_x = pad_cm / fig_w_cm
    pad_frac_y = pad_cm / fig_h_cm
    width_frac_x = width_cm / fig_w_cm
    width_frac_y = width_cm / fig_h_cm

    if location == "eastoutside":
        cax_rect = (pos.x1 + pad_frac_x, pos.y0,
                    width_frac_x,         pos.height)
    elif location == "southoutside":
        cax_rect = (pos.x0, pos.y0 - pad_frac_y - width_frac_y,
                    pos.width, width_frac_y)
    else:
        raise ValueError(
            f"location must be 'eastoutside' or 'southoutside'; "
            f"got {location!r}"
        )

    cax = fig.add_axes(cax_rect)
    # Find the most recent mappable on `ax`
    mappables = ax.get_images() + [c for c in ax.collections
                                    if hasattr(c, "get_cmap")]
    if not mappables:
        raise RuntimeError(
            "apply_colorbar: no image or mappable collection found on ax"
        )
    cb = fig.colorbar(mappables[-1], cax=cax,
                       orientation="horizontal" if location == "southoutside"
                                   else "vertical")
    if label is not None:
        cb.set_label(label)
    return cb


# ─────────────────────────────────────────────────────────────────── #
# circular_arrow                                                        #
# ─────────────────────────────────────────────────────────────────── #

def circular_arrow(
    ax:           Axes,
    radius:       float,
    center:       tuple[float, float],
    arrow_angle:  float,
    angle:        float,
    *,
    direction:    int = 1,
    color:        str = "k",
    head_size:    tuple[float, float] = (10.0, 8.0),
    n_points:     int = 200,
    linewidth:    float = 1.0,
) -> tuple[list, list]:
    """Plot a circular arc with one or two arrowheads.

    Port of :file:`MTA/utilities/graphics/circular_arrow.m`.

    Parameters
    ----------
    ax:
        Target axes.
    radius:
        Arc radius in data coords.
    center:
        ``(xc, yc)`` arc centre in data coords.
    arrow_angle:
        Centre angle of the arc, in radians (0 = +x).
    angle:
        Total arc length in radians.
    direction:
        ``+1`` → arrowhead at end of arc; ``-1`` → at start;
        ``2`` → both ends.  Default ``+1``.
    color:
        Line / arrow colour.
    head_size:
        ``(width, length)`` of the arrowhead in data-units.  Defaults
        ``(10, 8)`` are MATLAB-flavoured rough values; tune to your
        figure scale.
    n_points:
        Polyline resolution.  Default 200.

    Returns
    -------
    arcs : list[Line2D]
        Polyline handles.
    arrows : list[FancyArrowPatch]
        Arrowhead handles.
    """
    xc, yc = float(center[0]), float(center[1])
    half = angle / 2.0
    angles = np.linspace(arrow_angle - half, arrow_angle + half, n_points)
    arc_x = xc + radius * np.cos(angles)
    arc_y = yc + radius * np.sin(angles)
    arcs = ax.plot(arc_x, arc_y, color=color, linewidth=linewidth)

    arrows = []
    head_w, head_l = head_size

    def _add_head(start_idx: int, end_idx: int) -> None:
        """Add a FancyArrowPatch from arc[start_idx] → arc[end_idx]."""
        arr = FancyArrowPatch(
            (arc_x[start_idx], arc_y[start_idx]),
            (arc_x[end_idx],   arc_y[end_idx]),
            arrowstyle="-|>",
            mutation_scale=max(head_w, head_l),
            color=color,
            linewidth=0,
        )
        ax.add_patch(arr)
        arrows.append(arr)

    if direction == 1:
        _add_head(-2, -1)
    elif direction == -1:
        _add_head(1, 0)
    elif direction == 2:
        _add_head(-2, -1)
        _add_head(1, 0)
    else:
        raise ValueError(
            f"direction must be -1, 1, or 2; got {direction}"
        )
    return arcs, arrows


# ─────────────────────────────────────────────────────────────────── #
# draw_arrow                                                            #
# ─────────────────────────────────────────────────────────────────── #

def draw_arrow(
    ax:        Axes,
    start:     tuple[float, float],
    end:       tuple[float, float],
    *,
    style:     str = "straight",
    color:     str = "k",
    linewidth: float = 1.5,
    head_size: float = 10.0,
) -> FancyArrowPatch:
    """Draw a straight or curved arrow between two points.

    Replaces the (broken in MATLAB) :file:`draw_arrow.m`.  The MATLAB
    file in the lab repo has syntax errors and was clearly mid-edit;
    this Python port provides the working baseline.

    Parameters
    ----------
    ax:
        Target axes.
    start, end:
        Tail and head positions in data coords.
    style:
        ``'straight'`` (default) → straight arrow;
        ``'curve'`` → arc-shaped using matplotlib's
        ``connectionstyle='arc3'``.
    color, linewidth, head_size:
        Self-explanatory.

    Returns
    -------
    matplotlib.patches.FancyArrowPatch
    """
    if style == "straight":
        connection = "arc3,rad=0"
    elif style == "curve":
        connection = "arc3,rad=0.3"
    else:
        raise ValueError(
            f"style must be 'straight' or 'curve'; got {style!r}"
        )

    arr = FancyArrowPatch(
        start, end,
        arrowstyle    = "-|>",
        mutation_scale = head_size,
        color         = color,
        linewidth     = linewidth,
        connectionstyle = connection,
    )
    ax.add_patch(arr)
    return arr


# ─────────────────────────────────────────────────────────────────── #
# draw_lines (interactive)                                              #
# ─────────────────────────────────────────────────────────────────── #

def draw_lines(
    ax:    Axes,
    *,
    mode:  str = "free",
) -> Callable[[], list[tuple[np.ndarray, np.ndarray]]]:
    """Interactively draw lines on *ax*.

    Port of :file:`MTA/utilities/graphics/draw_lines.m`.

    Click left-mouse to start / extend a line, right-mouse to finish
    the current line.  Press ``q`` to stop drawing entirely.

    Parameters
    ----------
    ax:
        Target axes.
    mode:
        ``'free'`` (default) → just record the lines.
        ``'line_fit'`` → fit a 1st-order polynomial to each line and
        also record the slopes.  Matches MATLAB's
        ``getappdata(src,'mode')`` switch.

    Returns
    -------
    Callable returning ``list[(xs, ys)]`` once the user is done
    drawing.  Call the returned function to retrieve the final
    set of lines.

    Notes
    -----
    Requires an interactive matplotlib backend (TkAgg, QtAgg, etc.).
    Returns an empty list under non-interactive backends like Agg.
    """
    # Mutable container so the closure can append from event callbacks
    lines: list[tuple[list[float], list[float]]] = []
    current: dict[str, Any] = {"line": None, "xs": [], "ys": []}
    fits: list[np.ndarray] = []

    fig = ax.figure
    finished = {"value": False}

    def _on_click(event):
        if event.inaxes is not ax:
            return
        if event.button == 1:                   # left click
            current["xs"].append(event.xdata)
            current["ys"].append(event.ydata)
            if current["line"] is None:
                ln, = ax.plot(current["xs"], current["ys"], "-o")
                current["line"] = ln
            else:
                current["line"].set_data(current["xs"], current["ys"])
            fig.canvas.draw_idle()
        elif event.button == 3:                 # right click → finish
            if current["xs"]:
                xs = np.asarray(current["xs"], dtype=np.float64)
                ys = np.asarray(current["ys"], dtype=np.float64)
                lines.append((xs, ys))
                if mode == "line_fit" and len(xs) >= 2:
                    fits.append(np.polyfit(xs, ys, 1))
            current["line"] = None
            current["xs"] = []
            current["ys"] = []

    def _on_key(event):
        if event.key == "q":
            finished["value"] = True

    cid_click = fig.canvas.mpl_connect("button_press_event", _on_click)
    cid_key   = fig.canvas.mpl_connect("key_press_event",    _on_key)

    def get_lines() -> list[tuple[np.ndarray, np.ndarray]]:
        # Disconnect callbacks now that the user wants the result
        fig.canvas.mpl_disconnect(cid_click)
        fig.canvas.mpl_disconnect(cid_key)
        return [(np.asarray(xs), np.asarray(ys)) for xs, ys in lines]

    return get_lines


# ─────────────────────────────────────────────────────────────────── #
# sbar                                                                  #
# ─────────────────────────────────────────────────────────────────── #

def sbar(
    feature_values:  np.ndarray,
    bin_edges:       np.ndarray,
    state_mask:      np.ndarray,
    *,
    ax:              Optional[Axes] = None,
    background_color: str = "lightgrey",
    state_color:     str = "red",
    state_alpha:     float = 0.5,
) -> tuple:
    """Overlay a state-restricted histogram on a baseline (active) histogram.

    Port of :file:`MTA/utilities/graphics/sbar.m`.

    The MATLAB original expected ``Trial`` and ``fet`` objects and
    looked up the ``a-<state>`` (active-minus-state) baseline from a
    state collection.  This port takes pre-resolved arrays:

    * ``feature_values`` ``(T,)`` — a single feature column.
    * ``bin_edges`` ``(n_bins+1,)`` — histogram bin edges.
    * ``state_mask`` ``(T,)`` — boolean mask for the focal state.

    The baseline histogram is computed over **all** samples (caller
    masks first if a different baseline is desired); the state
    histogram is computed over the masked subset.

    Parameters
    ----------
    feature_values, bin_edges, state_mask:
        See above.
    ax:
        Target axes.  Default ``plt.gca()``.
    background_color, state_color, state_alpha:
        Self-explanatory.

    Returns
    -------
    (BarContainer, BarContainer)
        Background and state-restricted histogram handles.
    """
    feature_values = np.asarray(feature_values, dtype=np.float64).ravel()
    state_mask     = np.asarray(state_mask, dtype=bool).ravel()
    if feature_values.shape != state_mask.shape:
        raise ValueError(
            f"feature_values shape {feature_values.shape} ≠ "
            f"state_mask shape {state_mask.shape}"
        )

    if ax is None:
        ax = plt.gca()

    finite = np.isfinite(feature_values)
    bg_counts, _   = np.histogram(feature_values[finite], bins=bin_edges)
    fg_counts, _   = np.histogram(
        feature_values[finite & state_mask], bins=bin_edges,
    )

    centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    widths  = np.diff(bin_edges)
    bg_h = ax.bar(centres, bg_counts, width=widths,
                   color=background_color, edgecolor="none")
    fg_h = ax.bar(centres, fg_counts, width=widths,
                   color=state_color, alpha=state_alpha, edgecolor="none")
    return bg_h, fg_h
