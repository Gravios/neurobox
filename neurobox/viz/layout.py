"""
neurobox.viz.layout
====================
Paper-figure layout primitives on matplotlib.

Direct ports of:

* :file:`MTA/utilities/graphics/setup_figure.m`        → :func:`setup_figure`
* :file:`MTA/utilities/graphics/setup_axes.m`          → :func:`setup_axes`
* :file:`MTA/utilities/graphics/place_subplot.m`       → :func:`place_subplot`
* :file:`MTA/utilities/graphics/clear_fax.m`           → :func:`clear_fax`
* :file:`MTA/utilities/graphics/clear_axes_labels.m`   → :func:`clear_axes_labels`
* :file:`MTA/utilities/graphics/clear_axes_ticks.m`    → :func:`clear_axes_ticks`
* :file:`MTA/utilities/graphics/emptyAxis.m`           → :func:`empty_axis`
* :file:`MTA/utilities/graphics/linkax.m`              → :func:`linkax`
* :file:`MTA/utilities/graphics/parse_inkscape_layout.m` → :func:`parse_inkscape_layout`

The MATLAB original
-------------------
MTA's ``setup_figure``/``setup_axes`` pair places matplotlib axes at
**exact centimetre coordinates** on a paper-sized page (A4/A3/A2 or
1080p), with a fixed cm-grid of subplot anchor points.  Each axes is
**hash-tagged** so that re-calling ``setup_axes`` with identical
arguments reuses the same axes (clearing it with ``cla``) — this is
what lets the EgoProCode2D / BehaviorPlaceCode figure scripts iterate
on individual subplots without rebuilding the whole figure layout.

Hashing is a SHA-1 of ``(yind, yoffset, xind, xoffset, gYoffset,
gXoffset, hscale, wscale)``.  The hash becomes the matplotlib
``Axes.label`` (matplotlib's analogue of MATLAB's ``Tag``).

Coordinate system
-----------------
matplotlib axes are positioned in **figure-fraction** coords, but the
whole pipeline converts to/from **centimetres** internally so user
code looks identical to the MATLAB version.

For a 21 × 29.7 cm A4 figure, the cm-grid is::

    page.xpos = [marginLeft, marginLeft + (subplot.width + h_padding), …]
    page.ypos = flipped equivalent on the y-axis (top of page = high y in cm)

``setup_axes(fig, yind, yoffset, xind, xoffset, …)`` places an axes at::

    [page.xpos[xind] + xoffset + gXoffset,
     page.ypos[yind] + yoffset + gYoffset,
     subplot.width  * wscale,
     subplot.height * hscale]

— in centimetres, then converted to figure-fraction for matplotlib.

Inkscape import
---------------
:func:`parse_inkscape_layout` reads a JSON file like
``EgoProCode2D-f1-layout.json`` (a flat dict of named regions in
pixel coords) and returns a :class:`Layout` object.  Use
:func:`setup_axes_named` to place axes by region name instead of
grid index.

Differences from MATLAB
-----------------------
* ``DELETE_CURRENT_AXES`` global is replaced by an explicit
  ``delete_existing=True`` argument on :func:`setup_axes`.
* MATLAB's ``hfig.UserData.sax`` cell-array of all axes is replaced
  by an :class:`NBFigure` wrapper carrying the ``axes`` dict; the
  original ``Figure`` is still accessible via ``nbfig.figure``.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


__all__ = [
    "PageOpts",
    "SubplotOpts",
    "FigOpts",
    "Layout",
    "NBFigure",
    "setup_figure",
    "setup_axes",
    "setup_axes_named",
    "place_subplot",
    "clear_fax",
    "clear_axes_labels",
    "clear_axes_ticks",
    "empty_axis",
    "linkax",
    "parse_inkscape_layout",
]


# ─────────────────────────────────────────────────────────────────────── #
# Page format presets — match MATLAB exactly                                #
# ─────────────────────────────────────────────────────────────────────── #

PAGE_FORMATS_CM = {
    # (width, height) for portrait orientation, in cm
    "A4": (21.0, 29.7),
    "A3": (29.7, 42.0),
    "A2": (42.0, 59.4),
}

# 1080p screen format — uses pixel units rather than cm
PAGE_1080P_PX = (1920, 1080)

CM_PER_INCH = 2.54


# ─────────────────────────────────────────────────────────────────────── #
# Result dataclasses                                                         #
# ─────────────────────────────────────────────────────────────────────── #

@dataclass
class PageOpts:
    """Page-level layout parameters (cm or px)."""
    width:         float
    height:        float
    units:         str         # 'centimeters' or 'pixels'
    margin_left:   float
    margin_top:    float
    xpos:          np.ndarray  # (n_x_anchors,) — cm or px
    ypos:          np.ndarray  # (n_y_anchors,) — cm or px (top-down)


@dataclass
class SubplotOpts:
    width:                  float
    height:                 float
    horizontal_padding:     float
    vertical_padding:       float
    units:                  str


@dataclass
class FigOpts:
    """Composite layout state stored on each :class:`NBFigure`."""
    page:    PageOpts
    subplot: SubplotOpts
    format:  str
    layout:  str   # 'portrait' or 'landscape'


@dataclass
class Layout:
    """Imported figure layout (e.g. from an Inkscape SVG export).

    Attributes
    ----------
    body : dict[str, float]
        Page extents.  Keys: ``x``, ``y``, ``width``, ``height``.
    subplots : dict[str, dict[str, float]]
        Named-region rectangles.  Each value has keys ``x``, ``y``,
        ``width``, ``height``.
    units : str
        Source units, typically ``"pixels"`` for Inkscape exports.
    """
    body:     dict[str, float]
    subplots: dict[str, dict[str, float]]
    units:    str = "pixels"


@dataclass
class NBFigure:
    """Thin wrapper around :class:`matplotlib.figure.Figure` carrying
    the layout state needed by :func:`setup_axes`.

    Attributes
    ----------
    figure : matplotlib.figure.Figure
        The underlying matplotlib figure.
    opts : FigOpts
        Page / subplot layout parameters.
    fax : matplotlib.axes.Axes
        Background axes covering the whole page (used for inter-axes
        annotations like rotated labels).  Mirrors MATLAB's
        ``hfig.UserData.fax``.
    axes : dict[str, matplotlib.axes.Axes]
        Hash-tag → Axes registry.  Re-calling :func:`setup_axes` with
        identical args returns the existing axes.
    layout : Optional[Layout]
        Imported layout for :func:`setup_axes_named`.
    """
    figure: Figure
    opts:   FigOpts
    fax:    Axes
    axes:   dict[str, Axes] = field(default_factory=dict)
    layout: Optional[Layout] = None


# ─────────────────────────────────────────────────────────────────────── #
# Helpers                                                                    #
# ─────────────────────────────────────────────────────────────────────── #

def _hash_args(*args: Any) -> str:
    """SHA-1 over a tuple of layout arguments → 16-char hex tag.

    Mirrors MATLAB ``DataHash``.  Truncated to 16 chars because that's
    plenty for axes-label uniqueness within a single figure (and stays
    well below matplotlib's label-length sanity limits).
    """
    payload = repr(tuple(args)).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:16]


def _cm_to_fig_fraction(rect_cm: tuple[float, float, float, float],
                         page_w_cm: float, page_h_cm: float
                         ) -> tuple[float, float, float, float]:
    """Convert a ``[x, y, w, h]`` rectangle in cm to figure-fraction.

    matplotlib's ``Axes.set_position`` and ``Figure.add_axes`` both
    take rectangles in figure-fraction (0..1) coords.  We always
    represent layouts internally in cm and convert here.

    Note: the input is "MATLAB style" (origin = bottom-left, y grows
    upward) which matches matplotlib's convention.
    """
    x_cm, y_cm, w_cm, h_cm = rect_cm
    return (x_cm / page_w_cm,
            y_cm / page_h_cm,
            w_cm / page_w_cm,
            h_cm / page_h_cm)


def _build_xpos_ypos(page_w: float, page_h: float,
                      margin_left: float, margin_top: float,
                      subplot_w: float, subplot_h: float,
                      pad_h: float, pad_v: float
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Compute the ``page.xpos`` / ``page.ypos`` cm-grids.

    Mirrors MATLAB ``setup_figure.m`` lines 132-140.

    ``xpos`` runs left-to-right starting at *margin_left*.
    ``ypos`` runs **top-to-bottom**: ``ypos[0]`` is the highest anchor
    just below the top margin, and successive entries decrease.  This
    matches MATLAB's ``fliplr(...)`` so that ``yind=1`` in MATLAB is
    the topmost row.
    """
    xstep = subplot_w + pad_h
    if xstep <= 0:
        xpos = np.array([margin_left])
    else:
        xpos = np.arange(margin_left, page_w, xstep)

    ystep = subplot_h + pad_v
    if ystep <= 0:
        ypos = np.array([page_h - margin_top - subplot_h])
    else:
        # MATLAB:
        #   page.ypos = fliplr( (page.height - page.marginTop)
        #                     - (subplot.height + verticalPadding)
        #                     * floor((page.height - page.marginTop)
        #                            /(subplot.height + verticalPadding))
        #                     : ystep
        #                     : page.height - page.marginTop - subplot.height )
        # Translate: build ascending y-positions then flip.
        upper = page_h - margin_top - subplot_h
        n_steps = int(np.floor((page_h - margin_top) / ystep))
        lower = (page_h - margin_top) - ystep * n_steps
        ypos_asc = np.arange(lower, upper + 1e-9, ystep)
        ypos = ypos_asc[::-1]
    return xpos, ypos


# ─────────────────────────────────────────────────────────────────────── #
# Public API: setup_figure                                                   #
# ─────────────────────────────────────────────────────────────────────── #

def setup_figure(
    *,
    fig:                       Optional[Figure] = None,
    format:                    str = "A4",
    layout:                    str = "portrait",
    units:                     str = "centimeters",
    subplot_width:             float = 1.15,
    subplot_height:            float = 1.15,
    subplot_padding_horizontal: float = 0.0,
    subplot_padding_vertical:  float = 0.15,
    margin_left:               float = 2.54,
    margin_top:                float = 2.54,
    dpi:                       int = 300,
) -> NBFigure:
    """Create a paper-sized figure with cm-precise positioning.

    Port of :file:`MTA/utilities/graphics/setup_figure.m`.

    Parameters
    ----------
    fig:
        Existing :class:`matplotlib.figure.Figure` to configure.
        ``None`` (default) → create a fresh figure.
    format:
        Page format.  One of ``'A4'``, ``'A3'``, ``'A2'``, ``'1080p'``.
    layout:
        ``'portrait'`` or ``'landscape'``.  Ignored for ``'1080p'``.
    units:
        Internal layout unit.  ``'centimeters'`` (default) or
        ``'pixels'`` (used for ``'1080p'``).
    subplot_width, subplot_height:
        Default subplot size in *units*.  Default 1.15 cm matches
        MATLAB.
    subplot_padding_horizontal, subplot_padding_vertical:
        Spacing between adjacent subplots, in *units*.  Defaults
        ``0.0`` and ``0.15`` cm match MATLAB.
    margin_left, margin_top:
        Page margins, in *units*.  Default 2.54 cm = 1 in matches
        MATLAB.
    dpi:
        Figure resolution.  Default 300 (publication-quality raster).

    Returns
    -------
    NBFigure
        Wrapper carrying ``figure`` (the matplotlib Figure), ``opts``
        (layout parameters), ``fax`` (background axes), and ``axes``
        (hash-tag registry, populated by subsequent
        :func:`setup_axes` calls).

    Examples
    --------
    >>> nbfig = setup_figure(format="A4", layout="portrait")
    >>> sax = setup_axes(nbfig, yind=4, yoffset=-1.0, xind=2, xoffset=0.0)
    """
    # Resolve format → page dimensions
    if format == "1080p":
        units = "pixels"
        page_w, page_h = PAGE_1080P_PX
        # 1080p defaults from MATLAB lines 60-66
        margin_left = 100.0
        margin_top  = 100.0
    elif format in PAGE_FORMATS_CM:
        portrait_w, portrait_h = PAGE_FORMATS_CM[format]
        if layout == "portrait":
            page_w, page_h = portrait_w, portrait_h
        elif layout == "landscape":
            page_w, page_h = portrait_h, portrait_w
        else:
            raise ValueError(
                f"layout must be 'portrait' or 'landscape'; got {layout!r}"
            )
    else:
        raise ValueError(
            f"format must be one of {sorted(PAGE_FORMATS_CM)} or '1080p'; "
            f"got {format!r}"
        )

    # Build the cm-grid of subplot anchors
    xpos, ypos = _build_xpos_ypos(
        page_w, page_h,
        margin_left, margin_top,
        subplot_width, subplot_height,
        subplot_padding_horizontal, subplot_padding_vertical,
    )

    page_opts = PageOpts(
        width = page_w, height = page_h, units = units,
        margin_left = margin_left, margin_top = margin_top,
        xpos = xpos, ypos = ypos,
    )
    subplot_opts = SubplotOpts(
        width  = subplot_width,
        height = subplot_height,
        horizontal_padding = subplot_padding_horizontal,
        vertical_padding   = subplot_padding_vertical,
        units  = units,
    )
    opts = FigOpts(page = page_opts, subplot = subplot_opts,
                    format = format, layout = layout)

    # Create / reset the matplotlib Figure
    if fig is None:
        fig = plt.figure(dpi = dpi)
    fig.clear()
    if units == "centimeters":
        fig.set_size_inches(page_w / CM_PER_INCH, page_h / CM_PER_INCH)
    else:  # pixels
        fig.set_size_inches(page_w / dpi, page_h / dpi)

    # Background axes covering the whole page (for inter-axes annotations).
    # Mirrors MATLAB lines 163-167.
    fax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    fax.set_xlim(0, page_w)
    fax.set_ylim(0, page_h)
    fax.set_axis_off()
    fax.set_facecolor("none")
    # Push background axes to the bottom so subplot axes draw on top
    fax.set_zorder(-1)

    nbfig = NBFigure(figure=fig, opts=opts, fax=fax, axes={}, layout=None)
    # Stash on the underlying figure so other helpers can find it
    fig._nbfig = nbfig
    return nbfig


# ─────────────────────────────────────────────────────────────────────── #
# Public API: setup_axes                                                     #
# ─────────────────────────────────────────────────────────────────────── #

def setup_axes(
    fig_or_nbfig:        "Figure | NBFigure",
    yind:                int,
    yoffset:             float,
    xind:                int,
    xoffset:             float,
    *,
    g_yoffset:           float = 0.0,
    g_xoffset:           float = 0.0,
    hscale:              float = 1.0,
    wscale:              float = 1.0,
    delete_existing:     bool  = False,
    fontsize:            float = 8,
    linewidth:           float = 1.0,
) -> Axes:
    """Place an axes at a cm-precise position relative to the page grid.

    Port of :file:`MTA/utilities/graphics/setup_axes.m`.

    The axes is positioned at::

        x = page.xpos[xind] + xoffset + g_xoffset
        y = page.ypos[yind] + yoffset + g_yoffset
        w = subplot.width  * wscale
        h = subplot.height * hscale

    A SHA-1 tag of ``(yind, yoffset, xind, xoffset, g_yoffset,
    g_xoffset, hscale, wscale)`` is computed.  If an axes with the
    same tag already exists on the figure, it is **reused** (cleared
    with ``cla()``) — this matches MATLAB ``findobj(hfig, 'Tag', hash)``.

    Parameters
    ----------
    fig_or_nbfig:
        Either an :class:`NBFigure` (from :func:`setup_figure`) or a
        bare :class:`matplotlib.figure.Figure` that has been
        configured by :func:`setup_figure` (in which case the
        :class:`NBFigure` is recovered from ``fig._nbfig``).
    yind, xind:
        Anchor indices into ``page.ypos[]`` / ``page.xpos[]``.  These
        are **0-indexed** in the Python port (MATLAB was 1-indexed) —
        subtract 1 from any MATLAB index when porting.
    yoffset, xoffset:
        Per-axes offset from the anchor, in cm.
    g_yoffset, g_xoffset:
        Global offset applied to every axes in this call group.
        Useful for shifting whole panels.
    hscale, wscale:
        Multipliers on the default subplot height / width.
    delete_existing:
        If True and an axes with the same hash exists, delete it
        rather than reuse.  Mirrors MATLAB ``DELETE_CURRENT_AXES``
        global.
    fontsize:
        Default tick-label font size.  Default 8 matches MATLAB.
    linewidth:
        Default axes spine line width.  Default 1.0 matches MATLAB.

    Returns
    -------
    matplotlib.axes.Axes
        Newly created or reused axes.
    """
    # Resolve to an NBFigure
    if isinstance(fig_or_nbfig, NBFigure):
        nbfig = fig_or_nbfig
    elif isinstance(fig_or_nbfig, Figure):
        nbfig = getattr(fig_or_nbfig, "_nbfig", None)
        if nbfig is None:
            raise RuntimeError(
                "Figure was not initialised by setup_figure(); "
                "call setup_figure() first or pass an NBFigure directly."
            )
    else:
        raise TypeError(
            f"first argument must be an NBFigure or matplotlib Figure; "
            f"got {type(fig_or_nbfig).__name__}"
        )

    # Validate indices
    n_x = len(nbfig.opts.page.xpos)
    n_y = len(nbfig.opts.page.ypos)
    if not (0 <= xind < n_x):
        raise IndexError(f"xind={xind} out of range [0, {n_x})")
    if not (0 <= yind < n_y):
        raise IndexError(f"yind={yind} out of range [0, {n_y})")

    tag = _hash_args(yind, yoffset, xind, xoffset,
                      g_yoffset, g_xoffset, hscale, wscale)

    if tag in nbfig.axes:
        existing = nbfig.axes[tag]
        if delete_existing:
            existing.remove()
            del nbfig.axes[tag]
        else:
            existing.cla()
            return existing

    # Compute position in cm
    x_cm = nbfig.opts.page.xpos[xind] + xoffset + g_xoffset
    y_cm = nbfig.opts.page.ypos[yind] + yoffset + g_yoffset
    w_cm = nbfig.opts.subplot.width  * wscale
    h_cm = nbfig.opts.subplot.height * hscale

    rect_frac = _cm_to_fig_fraction(
        (x_cm, y_cm, w_cm, h_cm),
        nbfig.opts.page.width, nbfig.opts.page.height,
    )

    ax = nbfig.figure.add_axes(rect_frac, label=tag)
    ax.tick_params(labelsize=fontsize)
    for spine in ax.spines.values():
        spine.set_linewidth(linewidth)
    nbfig.axes[tag] = ax
    return ax


# ─────────────────────────────────────────────────────────────────────── #
# place_subplot — MATLAB convenience wrapper                                 #
# ─────────────────────────────────────────────────────────────────────── #

def place_subplot(
    nbfig:        NBFigure,
    yind:         int,
    yoffset:      float,
    xind:         int,
    xoffset:      float,
    *,
    scale_width:  float = 1.0,
    scale_height: float = 1.0,
) -> Axes:
    """MATLAB-style :func:`setup_axes` shorthand.

    Port of :file:`MTA/utilities/graphics/place_subplot.m`.  Differs
    from :func:`setup_axes` only in the ``scale_width`` / ``scale_height``
    parameter names (MATLAB's ``scaleWidth``/``scaleHeight``).
    """
    return setup_axes(nbfig, yind, yoffset, xind, xoffset,
                       wscale=scale_width, hscale=scale_height)


# ─────────────────────────────────────────────────────────────────────── #
# Inkscape layout import                                                     #
# ─────────────────────────────────────────────────────────────────────── #

def parse_inkscape_layout(
    filepath:        str | Path,
    element_pattern: str = "",
) -> Layout:
    """Read a JSON figure layout (typically exported from Inkscape).

    Port of :file:`MTA/utilities/graphics/parse_inkscape_layout.m`,
    extended to read either:

    * MATLAB's CSV-ish format (``name,x,y,width,height\\n``).
    * A JSON dict like the EgoProCode2D layout files
      (``{"body":..., "subplots":{"name":{"x":..,"y":..}}}``).

    Either format works — this function detects which from the file
    contents.

    Parameters
    ----------
    filepath:
        Path to the layout file.
    element_pattern:
        Regex; only subplots whose name matches are kept.  Default
        ``""`` keeps everything.

    Returns
    -------
    Layout
        Parsed layout with named subplot rectangles in pixel coords.

    Examples
    --------
    >>> layout = parse_inkscape_layout(
    ...     "EgoProCode2D-f1-layout.json",
    ...     element_pattern="EgoProCode2D-f1-subplot-",
    ... )
    >>> layout.subplots["EgoProCode2D-f1-subplot-placefieldExample"]
    {'x': 192.984375, 'y': 113.59375, 'width': 113.375, 'height': 113.375}
    """
    filepath = Path(filepath)
    text = filepath.read_text()
    if text.lstrip().startswith("{"):
        # JSON format
        raw = json.loads(text)
        body = raw.get("body", {"x": 0, "y": 0, "width": 0, "height": 0})
        subplots = dict(raw.get("subplots", {}))
    else:
        # CSV-ish format: name,x,y,width,height per line
        body = {"x": 0, "y": 0, "width": 0, "height": 0}
        subplots: dict[str, dict[str, float]] = {}
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 5:
                continue
            name = parts[0]
            try:
                x, y, w, h = (float(p) for p in parts[1:5])
            except ValueError:
                continue
            if name == "body":
                body = {"x": x, "y": y, "width": w, "height": h}
            else:
                subplots[name] = {"x": x, "y": y, "width": w, "height": h}

    # Filter by regex if requested
    if element_pattern:
        rx = re.compile(element_pattern)
        subplots = {k: v for k, v in subplots.items() if rx.search(k)}

    return Layout(body=body, subplots=subplots, units="pixels")


def setup_axes_named(
    nbfig:           NBFigure,
    name:            str,
    *,
    layout:          Optional[Layout] = None,
    delete_existing: bool = False,
    fontsize:        float = 8,
    linewidth:       float = 1.0,
) -> Axes:
    """Place an axes at a named region from an imported layout.

    Companion to :func:`parse_inkscape_layout`.  Looks up *name* in
    *layout* (or ``nbfig.layout`` if not given), converts the
    pixel-coord rectangle to the figure's coordinate system, and
    creates the axes.

    Parameters
    ----------
    nbfig:
        :class:`NBFigure` from :func:`setup_figure`.
    name:
        Subplot name as it appears in ``layout.subplots``.
    layout:
        Imported layout.  Default uses ``nbfig.layout`` (set with
        ``nbfig.layout = parse_inkscape_layout(...)`` after creation).
    delete_existing:
        See :func:`setup_axes`.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if layout is None:
        layout = nbfig.layout
    if layout is None:
        raise ValueError(
            "no layout provided; pass layout= or set nbfig.layout"
        )
    if name not in layout.subplots:
        raise KeyError(f"layout has no subplot named {name!r}")
    sub = layout.subplots[name]
    # Convert pixel-coord rectangle from layout space → figure-fraction.
    # Layout uses image-style coords (y=0 at top); flip y.
    body = layout.body
    body_w = body.get("width",  0) or 1.0
    body_h = body.get("height", 0) or 1.0
    x_frac = sub["x"]      / body_w
    w_frac = sub["width"]  / body_w
    h_frac = sub["height"] / body_h
    y_frac = 1.0 - (sub["y"] + sub["height"]) / body_h

    tag = _hash_args("named", name)
    if tag in nbfig.axes:
        existing = nbfig.axes[tag]
        if delete_existing:
            existing.remove()
            del nbfig.axes[tag]
        else:
            existing.cla()
            return existing

    ax = nbfig.figure.add_axes([x_frac, y_frac, w_frac, h_frac], label=tag)
    ax.tick_params(labelsize=fontsize)
    for spine in ax.spines.values():
        spine.set_linewidth(linewidth)
    nbfig.axes[tag] = ax
    return ax


# ─────────────────────────────────────────────────────────────────────── #
# Tick / label clearing helpers                                              #
# ─────────────────────────────────────────────────────────────────────── #

def clear_fax(nbfig: NBFigure) -> None:
    """Clear the background axes ``nbfig.fax`` (cf. MATLAB ``clear_fax.m``).

    Useful when re-rendering a figure: the inter-axes annotations
    (rotated labels, arrows etc.) drawn on the background axes
    accumulate across runs unless explicitly cleared.
    """
    nbfig.fax.cla()
    nbfig.fax.set_xlim(0, nbfig.opts.page.width)
    nbfig.fax.set_ylim(0, nbfig.opts.page.height)
    nbfig.fax.set_axis_off()


def clear_axes_labels(ax: Axes) -> None:
    """Clear tick labels but keep ticks.

    Port of :file:`clear_axes_labels.m`.  Sets both X and Y tick
    labels to empty strings; tick marks remain visible.
    """
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.xaxis.set_ticks_position("top")


def clear_axes_ticks(ax: Axes) -> None:
    """Remove ticks and tick labels entirely.

    Port of :file:`clear_axes_ticks.m`.
    """
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.xaxis.set_ticks_position("top")


def empty_axis(ax: Axes,
                background: tuple[float, float, float] = (0.9, 0.9, 0.9),
                ) -> None:
    """As :func:`clear_axes_ticks` but also paint the axes background.

    Port of :file:`emptyAxis.m`.  Default light-grey background
    matches MATLAB.
    """
    clear_axes_ticks(ax)
    ax.set_facecolor(background)


def linkax(fig: "Figure | NBFigure", mode: str = "xy") -> None:
    """Link axis limits of all axes on *fig*.

    Port of :file:`linkax.m`, which simply called MATLAB's
    ``linkaxes``.

    Parameters
    ----------
    mode:
        ``'x'`` → share x-axis limits, ``'y'`` → share y-axis limits,
        ``'xy'`` → share both (default).
    """
    if isinstance(fig, NBFigure):
        axes = list(fig.axes.values())
    elif isinstance(fig, Figure):
        axes = [a for a in fig.axes if not getattr(a, "_is_fax", False)]
    else:
        raise TypeError(
            f"linkax expects Figure or NBFigure; got {type(fig).__name__}"
        )
    if len(axes) < 2:
        return

    primary = axes[0]
    for ax in axes[1:]:
        if mode in ("x", "xy"):
            ax.sharex(primary)
        if mode in ("y", "xy"):
            ax.sharey(primary)
