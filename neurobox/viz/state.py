"""
neurobox.viz.state
===================
State-collection plotting helpers — rasters and duration bars.

Ports of:

* :file:`MTA/utilities/graphics/plot_stc.m`             → :func:`plot_stc`
* :file:`MTA/utilities/graphics/plot_stcs.m`            → :func:`plot_stcs`
* :file:`MTA/utilities/graphics/plot_state_durations.m` → :func:`plot_state_durations`
* :file:`MTA/utilities/graphics/plot_features_with_stc.m` → :func:`plot_features_with_stc`
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.collections import PatchCollection
from matplotlib.colors import to_rgb
from matplotlib.patches import Rectangle


__all__ = [
    "plot_stc",
    "plot_stcs",
    "plot_state_durations",
    "plot_features_with_stc",
]


# Default state-name → matplotlib-colour mapping, matching MATLAB's
# 'brgcmy' single-character codes.
_DEFAULT_STATE_COLORS = {
    "walk":  "blue",
    "rear":  "red",
    "turn":  "green",
    "pause": "cyan",
    "groom": "magenta",
    "sit":   "yellow",
}


def _resolve_colors(states: Sequence[str],
                     state_colors: str | Sequence[str] | None,
                     ) -> list[tuple[float, float, float]]:
    """Return one RGB triple per state.

    *state_colors* may be:

    * ``None`` → derive from ``_DEFAULT_STATE_COLORS`` falling back to
      ``cm.tab10``.
    * a string of single-letter codes (e.g. ``"brgcmy"``) — one char
      per state.
    * a sequence of matplotlib colour specs (one per state).
    """
    n = len(states)
    if state_colors is None:
        out = []
        cmap = plt.get_cmap("tab10")
        for i, name in enumerate(states):
            if name in _DEFAULT_STATE_COLORS:
                out.append(to_rgb(_DEFAULT_STATE_COLORS[name]))
            else:
                out.append(cmap(i % 10)[:3])
        return out
    if isinstance(state_colors, str) and len(state_colors) == n \
            and all(c in "brgcmykw" for c in state_colors):
        return [to_rgb(c) for c in state_colors]
    if isinstance(state_colors, (list, tuple)) and len(state_colors) == n:
        return [to_rgb(c) for c in state_colors]
    raise ValueError(
        f"state_colors length {len(state_colors)} does not match "
        f"number of states {n}"
    )


# ─────────────────────────────────────────────────────────────────── #
# plot_stc                                                              #
# ─────────────────────────────────────────────────────────────────── #

def plot_stc(
    stc:                "NBStateCollection",
    *,
    ax:                 Optional[Axes] = None,
    samplerate:         Optional[float] = None,
    states:             Optional[Sequence[str]] = None,
    state_colors:       str | Sequence[str] | None = None,
    staggered:          bool = True,
    label_method:       str = "text",
    time_unit:          str = "samples",
) -> Axes:
    """Plot a state-collection raster.

    Port of :file:`MTA/utilities/graphics/plot_stc.m`.

    Each state in *states* is rendered as a row of patches: each patch
    spans the period start/end of one bout of that state, and its
    fill colour is the state's assigned colour.  When ``staggered``
    is True (the default), each state gets its own y-row; when False,
    all states overlay at y=0.

    Parameters
    ----------
    stc:
        :class:`NBStateCollection` from
        :mod:`neurobox.dtype.state_collection`.
    ax:
        Target axes.  Default ``plt.gca()``.
    samplerate:
        Resample-target rate for periods-to-x-coords conversion.  If
        ``None`` (default), uses the first state's native rate.  Note
        that *time_unit* still controls whether the x-axis is samples
        or seconds.
    states:
        Sequence of state labels to render in row order (top to
        bottom).  ``None`` → use all states in the collection in their
        native order.  Default MATLAB list:
        ``['walk', 'rear', 'turn', 'pause', 'groom', 'sit']``.
    state_colors:
        See :func:`_resolve_colors`.  Default uses lab-standard
        colours (walk=blue, rear=red, etc.) matching MATLAB
        ``'brgcmy'``.
    staggered:
        If True (default), each state gets its own y-row.  If False,
        all states overlay at y=0..1.
    label_method:
        ``'text'`` (default) → label each y-row with the state name;
        ``''`` → leave y-axis unlabelled.
    time_unit:
        ``'samples'`` (default, MATLAB-equivalent) → x-axis in
        samples; ``'seconds'`` → divide by samplerate.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        ax = plt.gca()

    if states is None:
        # Use all states in collection order
        try:
            states = list(stc.list_states())
        except AttributeError:
            try:
                states = list(stc.labels)
            except AttributeError:
                states = [s.label for s in stc.states]

    n = len(states)
    if n == 0:
        return ax

    colors = _resolve_colors(states, state_colors)

    # Resolve samplerate
    if samplerate is None:
        ep0 = stc[states[0]] if hasattr(stc, "__getitem__") else \
              stc.states[0]
        samplerate = float(ep0.samplerate)

    for s_idx, name in enumerate(states):
        try:
            ep = stc[name]
        except (KeyError, IndexError):
            continue
        # Make sure we're working in samples at the chosen samplerate
        if hasattr(ep, "resample") and float(ep.samplerate) != samplerate:
            ep = ep.resample(samplerate)

        # Get periods in samples (NBEpoch.data in 'periods' mode)
        if getattr(ep, "mode", "periods") != "periods":
            # Convert mask → periods
            from neurobox.dtype.epoch import NBEpoch
            mask = ep.data.astype(bool)
            edges = np.diff(mask.astype(int))
            starts = np.flatnonzero(edges == 1) + 1
            ends   = np.flatnonzero(edges == -1) + 1
            if mask.size and mask[0]:
                starts = np.r_[0, starts]
            if mask.size and mask[-1]:
                ends = np.r_[ends, mask.size]
            periods = np.column_stack([starts, ends]).astype(np.float64)
        else:
            periods = np.asarray(ep.data, dtype=np.float64)
        if periods.size == 0:
            continue

        x_factor = (1.0 / samplerate) if time_unit == "seconds" else 1.0
        y0 = float(s_idx) if staggered else 0.0
        for start, end in periods:
            ax.add_patch(Rectangle(
                (start * x_factor, y0),
                (end - start) * x_factor,
                1.0,
                edgecolor=colors[s_idx],
                facecolor=colors[s_idx],
                linewidth=0.0,
            ))

    ax.set_ylim(0, n if staggered else 1.0)
    # Stretch x-axis to data
    ax.relim()
    ax.autoscale_view(scaley=False)

    if staggered and label_method == "text":
        ax.set_yticks(np.arange(n) + 0.5)
        ax.set_yticklabels(states)
    else:
        ax.set_yticks([])

    return ax


# ─────────────────────────────────────────────────────────────────── #
# plot_stcs — overlay multiple state collections                        #
# ─────────────────────────────────────────────────────────────────── #

def plot_stcs(
    stcs:               Sequence["NBStateCollection"],
    *,
    ax:                 Optional[Axes] = None,
    samplerate:         Optional[float] = None,
    states:             Optional[Sequence[str]] = None,
    state_colors:       str | Sequence[str] | None = None,
    label_method:       str = "text",
    time_unit:          str = "samples",
    row_height:         float = 1.0,
    row_padding:        float = 0.2,
) -> Axes:
    """Plot multiple state collections as stacked rasters.

    Port of :file:`MTA/utilities/graphics/plot_stcs.m`.

    Each collection in *stcs* gets its own block of rows below the
    previous collection's block.  Useful for comparing hand-labels
    against classifier output, or two classifier variants.
    """
    if ax is None:
        ax = plt.gca()

    if not stcs:
        return ax

    # Use the first stc to figure out the per-stc state list
    if states is None:
        first = stcs[0]
        try:
            states = list(first.list_states())
        except AttributeError:
            try:
                states = list(first.labels)
            except AttributeError:
                states = [s.label for s in first.states]

    n_states = len(states)
    block_h = n_states * row_height + row_padding

    for i, stc in enumerate(stcs):
        # Place each stc with a per-block y-offset by faking a sub-axes
        # via patch coordinates.  We just call plot_stc with staggered
        # but shift the resulting y values manually.
        save_ylim = ax.get_ylim()
        # Plot each state row at row_height * (n_states - 1 - row_idx) + i*block_h
        colors = _resolve_colors(states, state_colors)
        if samplerate is None:
            samplerate = float(stc[states[0]].samplerate
                                if hasattr(stc, "__getitem__")
                                else stc.states[0].samplerate)

        for s_idx, name in enumerate(states):
            try:
                ep = stc[name]
            except (KeyError, IndexError):
                continue
            if hasattr(ep, "resample") and float(ep.samplerate) != samplerate:
                ep = ep.resample(samplerate)
            periods = (np.asarray(ep.data, dtype=np.float64)
                       if getattr(ep, "mode", "periods") == "periods"
                       else _mask_to_periods(ep.data))
            if periods.size == 0:
                continue
            x_factor = (1.0 / samplerate) if time_unit == "seconds" else 1.0
            y0 = i * block_h + (n_states - 1 - s_idx) * row_height
            for start, end in periods:
                ax.add_patch(Rectangle(
                    (start * x_factor, y0),
                    (end - start) * x_factor, row_height,
                    edgecolor=colors[s_idx],
                    facecolor=colors[s_idx],
                    linewidth=0.0,
                ))

    total_h = len(stcs) * block_h
    ax.set_ylim(0, total_h)
    ax.relim()
    ax.autoscale_view(scaley=False)

    if label_method == "text":
        # One label per row in the first block; subsequent blocks get
        # numeric prefixes
        labels = []
        ticks  = []
        for i in range(len(stcs)):
            for s_idx, name in enumerate(states):
                ticks.append(i * block_h
                              + (n_states - 1 - s_idx) * row_height
                              + row_height / 2.0)
                labels.append(f"{name}" if i == 0 else f"{name} ({i+1})")
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels)
    return ax


def _mask_to_periods(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask).astype(bool).ravel()
    edges = np.diff(mask.astype(int))
    starts = np.flatnonzero(edges == 1) + 1
    ends   = np.flatnonzero(edges == -1) + 1
    if mask.size and mask[0]:
        starts = np.r_[0, starts]
    if mask.size and mask[-1]:
        ends = np.r_[ends, mask.size]
    return np.column_stack([starts, ends]).astype(np.float64)


# ─────────────────────────────────────────────────────────────────── #
# plot_state_durations                                                  #
# ─────────────────────────────────────────────────────────────────── #

def plot_state_durations(
    stc:               "NBStateCollection",
    *,
    ax:                Optional[Axes] = None,
    states:            Optional[Sequence[str]] = None,
    state_colors:      str | Sequence[str] | None = None,
    theta_state:       Optional[str] = "theta",
    show_theta_split:  bool = True,
) -> Axes:
    """Stacked bar showing total per-state duration, with theta split.

    Port of :file:`MTA/utilities/graphics/plot_state_durations.m`.

    The MATLAB original drew a 2-row stacked bar: top row is per-state
    total durations side-by-side; bottom row breaks each state into
    its theta vs non-theta sub-durations.

    Parameters
    ----------
    stc:
        :class:`NBStateCollection`.
    ax:
        Target axes.  Default ``plt.gca()``.
    states:
        States to include.  Default
        ``['walk', 'rear', 'turn', 'pause', 'groom', 'sit']`` matching
        MATLAB.
    state_colors:
        See :func:`_resolve_colors`.
    theta_state:
        Label of the theta epoch in *stc*.  ``None`` → skip the
        theta-split row.  Default ``'theta'``.
    show_theta_split:
        If True, also render the per-state theta vs non-theta split
        as a second row.  If False, only the top row is drawn.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        ax = plt.gca()
    if states is None:
        states = ["walk", "rear", "turn", "pause", "groom", "sit"]

    colors = _resolve_colors(states, state_colors)

    # Per-state total duration (in samples)
    durations = []
    for name in states:
        try:
            ep = stc[name]
        except (KeyError, IndexError):
            durations.append(0.0)
            continue
        periods = (np.asarray(ep.data, dtype=np.float64)
                   if getattr(ep, "mode", "periods") == "periods"
                   else _mask_to_periods(ep.data))
        if periods.size == 0:
            durations.append(0.0)
        else:
            durations.append(float(np.sum(periods[:, 1] - periods[:, 0])))
    durations = np.asarray(durations)

    # Top row: each state's total
    positions = np.r_[0.0, np.cumsum(durations[:-1])]
    for x0, dur, c in zip(positions, durations, colors):
        ax.add_patch(Rectangle((x0, 1.0), dur, 1.0,
                                  facecolor=c, edgecolor="none"))

    # Bottom row: theta vs non-theta breakdown
    if show_theta_split and theta_state is not None:
        try:
            theta_ep = stc[theta_state]
        except (KeyError, IndexError):
            theta_ep = None

        if theta_ep is not None:
            theta_durations = []
            for name in states:
                try:
                    ep = stc[name] & theta_ep
                except (KeyError, IndexError, TypeError):
                    theta_durations.append(0.0)
                    continue
                periods = (np.asarray(ep.data, dtype=np.float64)
                           if getattr(ep, "mode", "periods") == "periods"
                           else _mask_to_periods(ep.data))
                if periods.size == 0:
                    theta_durations.append(0.0)
                else:
                    theta_durations.append(
                        float(np.sum(periods[:, 1] - periods[:, 0])))
            theta_durations = np.asarray(theta_durations)
            non_theta = durations - theta_durations

            # Interleave: theta block (light), non-theta block (dark)
            # for each state, in the same order as the top row.
            light_colors = colors                           # full saturation
            dark_colors  = [(c[0] * 0.4 + 0.25,
                             c[1] * 0.4 + 0.25,
                             c[2] * 0.4 + 0.25) for c in colors]
            x = 0.0
            for theta_d, non_d, lc, dc in zip(
                    theta_durations, non_theta, light_colors, dark_colors):
                if theta_d > 0:
                    ax.add_patch(Rectangle((x, 0.0), theta_d, 1.0,
                                              facecolor=lc, edgecolor="none"))
                    x += theta_d
                if non_d > 0:
                    ax.add_patch(Rectangle((x, 0.0), non_d, 1.0,
                                              facecolor=dc, edgecolor="none"))
                    x += non_d

    total = float(durations.sum()) if durations.size else 1.0
    ax.set_xlim(0, total)
    ax.set_ylim(0, 2)
    ax.set_xticks([])
    ax.set_yticks([])
    return ax


# ─────────────────────────────────────────────────────────────────── #
# plot_features_with_stc                                                #
# ─────────────────────────────────────────────────────────────────── #

def plot_features_with_stc(
    feature_values:   np.ndarray,
    stc:              "NBStateCollection",
    *,
    samplerate:       float,
    ax:               Optional[Axes] = None,
    states:           Optional[Sequence[str]] = None,
    state_colors:     str | Sequence[str] | None = None,
    state_alpha:      float = 0.2,
    feature_color:    str = "k",
    feature_lw:       float = 0.8,
    time_unit:        str = "seconds",
) -> Axes:
    """Plot a feature trace overlaid with state-coloured background spans.

    Port of :file:`MTA/utilities/graphics/plot_features_with_stc.m`.

    Useful for sanity-checking classifier output: shows the raw feature
    trace with each state as a translucent vertical band.

    Parameters
    ----------
    feature_values:
        ``(T,)`` or ``(T, n_features)`` feature array.  Each column
        becomes its own line.
    stc:
        :class:`NBStateCollection`.
    samplerate:
        Sample rate of *feature_values*.
    ax:
        Target axes.  Default ``plt.gca()``.
    states:
        Which states to overlay.  Default
        ``['walk', 'rear', 'turn', 'pause', 'groom', 'sit']``.
    state_colors:
        See :func:`_resolve_colors`.
    state_alpha:
        Background-span transparency.  Default 0.2.
    feature_color, feature_lw:
        Line style for the feature trace.
    time_unit:
        ``'seconds'`` (default) or ``'samples'``.

    Returns
    -------
    matplotlib.axes.Axes
    """
    feature_values = np.asarray(feature_values, dtype=np.float64)
    if feature_values.ndim == 1:
        feature_values = feature_values[:, None]
    T, n_feat = feature_values.shape
    if states is None:
        states = ["walk", "rear", "turn", "pause", "groom", "sit"]
    colors = _resolve_colors(states, state_colors)

    if ax is None:
        ax = plt.gca()

    # Time axis
    x_factor = (1.0 / samplerate) if time_unit == "seconds" else 1.0
    t = np.arange(T) * x_factor

    # Plot feature lines first
    for j in range(n_feat):
        ax.plot(t, feature_values[:, j], color=feature_color,
                linewidth=feature_lw)

    # Then overlay state spans
    for s_idx, name in enumerate(states):
        try:
            ep = stc[name]
        except (KeyError, IndexError):
            continue
        if hasattr(ep, "resample") and float(ep.samplerate) != samplerate:
            ep = ep.resample(samplerate)
        periods = (np.asarray(ep.data, dtype=np.float64)
                   if getattr(ep, "mode", "periods") == "periods"
                   else _mask_to_periods(ep.data))
        for start, end in periods:
            ax.axvspan(start * x_factor, end * x_factor,
                        facecolor=colors[s_idx], alpha=state_alpha,
                        edgecolor="none", zorder=-1)

    ax.set_xlim(t[0], t[-1] if T > 0 else 1.0)
    return ax
