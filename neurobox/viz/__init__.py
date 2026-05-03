"""
neurobox.viz
=============
Paper-figure layout and plot primitives — port of MTA's
``utilities/graphics/`` module on top of matplotlib.

Three sub-modules:

* :mod:`neurobox.viz.layout` — paper-figure layout primitives
  (:func:`setup_figure`, :func:`setup_axes`, ``clear_*`` helpers,
  :func:`parse_inkscape_layout`).  Mirrors MATLAB ``setup_figure.m`` /
  ``setup_axes.m`` semantics in cm-precise positioning.
* :mod:`neurobox.viz.plots` — domain plot primitives
  (:func:`imagescnan`, :func:`error_ellipse`, :func:`apply_colorbar`,
  :func:`circular_arrow`, :func:`draw_arrow`, :func:`draw_lines`,
  :func:`sbar`).
* :mod:`neurobox.viz.state` — state-collection plotting helpers
  (:func:`plot_stc`, :func:`plot_stcs`, :func:`plot_state_durations`,
  :func:`plot_features_with_stc`).

The whole module requires matplotlib; install with::

    pip install 'neurobox[viz]'

If matplotlib is not installed, importing ``neurobox.viz`` raises a
clear ImportError with the install hint.
"""

from __future__ import annotations

# Fail-fast at import with a helpful message
try:
    import matplotlib            # noqa: F401  (importability check)
except ImportError as e:         # pragma: no cover
    raise ImportError(
        "neurobox.viz requires matplotlib.  Install with "
        "`pip install 'neurobox[viz]'` (or just `pip install matplotlib`)."
    ) from e

from .layout import (
    setup_figure,
    setup_axes,
    setup_axes_named,
    clear_fax,
    clear_axes_labels,
    clear_axes_ticks,
    empty_axis,
    linkax,
    parse_inkscape_layout,
    place_subplot,
    NBFigure,
    Layout,
)
from .plots import (
    imagescnan,
    error_ellipse,
    apply_colorbar,
    circular_arrow,
    draw_arrow,
    draw_lines,
    sbar,
)
from .state import (
    plot_stc,
    plot_stcs,
    plot_state_durations,
    plot_features_with_stc,
)


__all__ = [
    # layout
    "setup_figure",
    "setup_axes",
    "setup_axes_named",
    "clear_fax",
    "clear_axes_labels",
    "clear_axes_ticks",
    "empty_axis",
    "linkax",
    "parse_inkscape_layout",
    "place_subplot",
    "NBFigure",
    "Layout",
    # plots
    "imagescnan",
    "error_ellipse",
    "apply_colorbar",
    "circular_arrow",
    "draw_arrow",
    "draw_lines",
    "sbar",
    # state
    "plot_stc",
    "plot_stcs",
    "plot_state_durations",
    "plot_features_with_stc",
]
