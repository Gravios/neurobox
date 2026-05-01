"""
neurobox.config.standards
==========================
Canonical reference data: maze geometries, standard marker names,
and standard skeleton connections.

This is the static "standards" data that every project shares.
Originally derived from the data blocks in
:file:`MTA/MTAConfiguration.m` (``MTAMazes``, ``MTAMarkers``,
``MTAMarkerConnections``).  Kept as YAML in
:file:`neurobox/data/standards/` so it can be amended without code
changes.

The project-directory and ``.env``-writing parts of MATLAB's
MTAConfiguration are covered separately by
:func:`neurobox.config.configure_project` / :func:`load_config`;
this module only provides the reference data.

What's here
-----------
:func:`load_mazes`
    Maze geometry table — code, shape, visual + computational extents.
:func:`load_markers`
    Standard marker name list (24 names, in the canonical order).
:func:`load_marker_connections`
    Skeleton-graph edges (23 pairs).
:func:`get_maze`
    Look up a single maze by code.
:func:`make_standard_model`
    Convenience factory that builds an :class:`NBModel` populated with
    a subset of the standard markers and the matching skeleton edges.

Examples
--------
>>> from neurobox.config import load_mazes, get_maze, make_standard_model
>>> mazes = load_mazes()
>>> mazes['cof'].shape
'circle'
>>> mazes['cof'].computational_extent_xyz_mm
((-500.0, 500.0), (-500.0, 500.0), (0.0, 360.0))

>>> get_maze('rov').visual_extent_xyz_mm
((-800.0, 800.0), (-500.0, 500.0), (0.0, 300.0))

Build an NBModel for the standard 11-marker rigid-body subset::

>>> model = make_standard_model([
...     'head_back', 'head_left', 'head_front', 'head_right',
...     'spine_upper', 'spine_middle', 'pelvis_root',
...     'hip_left', 'hip_right', 'knee_left', 'knee_right',
... ])
>>> model.markers[0]
'head_back'
>>> ('head_back', 'head_left') in [tuple(c) for c in model.connections]
True
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

from neurobox.dtype.model import NBModel


# ─────────────────────────────────────────────────────────────────────────── #
# Resource locations                                                           #
# ─────────────────────────────────────────────────────────────────────────── #

_DATA_DIR = (Path(__file__).resolve().parent.parent / "data" / "standards")


# ─────────────────────────────────────────────────────────────────────────── #
# Maze geometry                                                                #
# ─────────────────────────────────────────────────────────────────────────── #

@dataclass(frozen=True)
class MazeInfo:
    """Geometry record for a maze.

    Attributes
    ----------
    code:
        Three-letter maze code, e.g. ``'cof'`` (circle open field).
    shape:
        One of ``'circle'``, ``'rectangle'``, ``'square'``, ``'line'``,
        ``'W'``.
    visual_extent_xyz_mm:
        ``((x_min, x_max), (y_min, y_max), (z_min, z_max))`` — bounding
        region for plotting.
    computational_extent_xyz_mm:
        ``((x_min, x_max), (y_min, y_max), (z_min, z_max))`` — spatial
        domain for analyses (typically slightly larger than visual).
    """

    code:                          str
    shape:                         str
    visual_extent_xyz_mm:          tuple[tuple[float, float], ...]
    computational_extent_xyz_mm:   tuple[tuple[float, float], ...]

    def visual_extent(self, axis: str) -> tuple[float, float]:
        """Return ``(min, max)`` for the named axis (``'x'``, ``'y'``, ``'z'``)."""
        return self.visual_extent_xyz_mm[_axis_index(axis)]

    def computational_extent(self, axis: str) -> tuple[float, float]:
        """Return computational ``(min, max)`` for the named axis."""
        return self.computational_extent_xyz_mm[_axis_index(axis)]


def _axis_index(axis: str) -> int:
    a = axis.lower()
    if a in ("x", "0"): return 0
    if a in ("y", "1"): return 1
    if a in ("z", "2"): return 2
    raise ValueError(f"Unknown axis {axis!r}; expected 'x', 'y', or 'z'")


@lru_cache(maxsize=1)
def _load_mazes_raw() -> list[dict]:
    with open(_DATA_DIR / "mazes.yaml") as fp:
        data = yaml.safe_load(fp)
    return data["mazes"]


def load_mazes() -> dict[str, MazeInfo]:
    """Load all maze records as a code → :class:`MazeInfo` mapping.

    Returns a fresh dict on each call (the underlying data is cached).
    """
    out: dict[str, MazeInfo] = {}
    for raw in _load_mazes_raw():
        out[raw["code"]] = MazeInfo(
            code  = raw["code"],
            shape = raw["shape"],
            visual_extent_xyz_mm = tuple(
                tuple(float(v) for v in row)
                for row in raw["visual_extent_xyz_mm"]
            ),
            computational_extent_xyz_mm = tuple(
                tuple(float(v) for v in row)
                for row in raw["computational_extent_xyz_mm"]
            ),
        )
    return out


def get_maze(code: str) -> MazeInfo:
    """Look up a single maze by its code.

    Raises
    ------
    KeyError
        If ``code`` is not in the standard maze table.
    """
    mazes = load_mazes()
    if code not in mazes:
        raise KeyError(
            f"Unknown maze code {code!r}.  "
            f"Known: {sorted(mazes.keys())}"
        )
    return mazes[code]


# ─────────────────────────────────────────────────────────────────────────── #
# Markers                                                                      #
# ─────────────────────────────────────────────────────────────────────────── #

@lru_cache(maxsize=1)
def _markers_cached() -> tuple[str, ...]:
    with open(_DATA_DIR / "markers.yaml") as fp:
        data = yaml.safe_load(fp)
    return tuple(data["markers"])


def load_markers() -> list[str]:
    """Return the canonical list of marker names recognised by neurobox.

    Order is stable across releases; new markers should be appended.
    Originally seeded from MTA's ``MTAMarkers``.

    Returns a fresh list on each call (callers may safely mutate
    without affecting future calls).
    """
    return list(_markers_cached())


@lru_cache(maxsize=1)
def _connections_cached() -> tuple[tuple[str, str], ...]:
    with open(_DATA_DIR / "marker_connections.yaml") as fp:
        data = yaml.safe_load(fp)
    return tuple((a, b) for a, b in data["connections"])


def load_marker_connections() -> list[tuple[str, str]]:
    """Return the canonical skeleton edges as a list of ``(a, b)`` pairs.

    Used by :func:`make_standard_model` to populate ``NBModel.connections``.
    Returns a fresh list on each call.
    """
    return list(_connections_cached())


def make_standard_model(
    markers: list[str] | None = None,
    *,
    only_present_connections: bool = True,
) -> NBModel:
    """Build an :class:`NBModel` from the canonical marker list.

    Parameters
    ----------
    markers:
        Subset of canonical marker names to include.  ``None``
        (default) → all 24 standard markers.  Names not in the
        canonical list are passed through unchanged (so callers can
        mix in custom marker names).
    only_present_connections:
        When ``True`` (default), only include skeleton edges where
        both endpoints are in the chosen marker list.  When ``False``,
        include all 23 standard edges regardless.

    Returns
    -------
    :class:`NBModel`
        Populated with ``markers`` (in the given order) and the
        relevant subset of skeleton connections.

    Examples
    --------
    Standard 4-marker head rigid body::

        from neurobox.config import make_standard_model
        head = make_standard_model([
            'head_back', 'head_left', 'head_front', 'head_right',
        ])
        # Triangle of head edges is preserved
        assert ['head_front', 'head_left'] in head.connections

    Mix in a custom marker::

        body = make_standard_model([
            'head_back', 'spine_upper', 'pelvis_root', 'custom_widget',
        ])
        # 'custom_widget' kept verbatim; not used in any standard edge.
    """
    if markers is None:
        markers = load_markers()

    if only_present_connections:
        canonical = set(load_markers())
        marker_set = set(markers)
        # Keep edges where both endpoints are in the requested marker set
        edges = [
            list(p) for p in load_marker_connections()
            if p[0] in marker_set and p[1] in marker_set
        ]
    else:
        edges = [list(p) for p in load_marker_connections()]

    return NBModel(markers=list(markers), connections=edges)
