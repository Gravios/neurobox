"""
model.py  —  NBModel
=====================
Lightweight marker-name registry.  Port of the indexing parts of
MTAModel (the rigid-body, VSK-loading, and visualisation machinery
is left to a future iteration).

Responsibilities
----------------
* Map marker names ↔ integer column indices for NBDxyz indexing.
* Provide rigid-body subset construction.
* Store standard marker connections for skeleton plotting.

The class is intentionally thin — no file I/O beyond the
class constructor's ``from_markers`` factory.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class NBModel:
    """Ordered collection of marker names with index helpers.

    Parameters
    ----------
    markers:
        Ordered list of marker name strings.  The order must match
        the second axis of the associated NBDxyz data array.
    connections:
        Optional list of ``[name_a, name_b]`` pairs defining the
        skeleton graph used for plotting.

    Examples
    --------
    >>> model = NBModel(['head_front', 'head_back', 'spine_upper',
    ...                  'spine_middle', 'tail_base'])
    >>> model.index('spine_upper')
    2
    >>> model.indices(['head_front', 'head_back'])
    [0, 1]
    >>> sub = model.subset(['head_front', 'head_back'])
    >>> sub.markers
    ['head_front', 'head_back']
    """

    markers:     list[str]       = field(default_factory=list)
    connections: list[list[str]] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    # Index resolution                                                     #
    # ------------------------------------------------------------------ #

    def index(self, name: str) -> int:
        """Return the 0-based column index for *name*.

        Raises
        ------
        KeyError if the marker is not found.
        """
        try:
            return self.markers.index(name)
        except ValueError:
            raise KeyError(
                f"Marker {name!r} not in model.  "
                f"Available: {self.markers}"
            ) from None

    def indices(self, names: list[str]) -> list[int]:
        """Return column indices for a list of marker names."""
        return [self.index(n) for n in names]

    def resolve(self, ref) -> list[int]:
        """Resolve a flexible marker reference to a list of indices.

        Accepts:
        * ``None``          → all indices
        * ``str``           → single marker name
        * ``list[str]``     → list of marker names
        * ``list[int]``     → pass-through
        * ``int``           → wrap in list
        """
        if ref is None:
            return list(range(len(self.markers)))
        if isinstance(ref, str):
            return [self.index(ref)]
        if isinstance(ref, int):
            return [ref]
        # list: may contain strings or ints
        return [self.index(r) if isinstance(r, str) else int(r) for r in ref]

    # ------------------------------------------------------------------ #
    # Subset construction                                                  #
    # ------------------------------------------------------------------ #

    def subset(self, names: list[str]) -> "NBModel":
        """Return a new NBModel restricted to *names*, preserving order."""
        # Validate
        for n in names:
            self.index(n)
        # Build subset connections (only pairs where both endpoints are in subset)
        name_set = set(names)
        sub_conns = [c for c in self.connections
                     if c[0] in name_set and c[1] in name_set]
        return NBModel(markers=names, connections=sub_conns)

    # ------------------------------------------------------------------ #
    # Properties                                                           #
    # ------------------------------------------------------------------ #

    @property
    def n(self) -> int:
        """Number of markers."""
        return len(self.markers)

    def __len__(self) -> int:
        return len(self.markers)

    def __contains__(self, name: str) -> bool:
        return name in self.markers

    def __repr__(self) -> str:
        return f"NBModel({self.markers!r})"

    # ------------------------------------------------------------------ #
    # Standard marker sets                                                 #
    # ------------------------------------------------------------------ #

    @classmethod
    def default_rat(cls) -> "NBModel":
        """Standard 15-marker Sprague-Dawley rat model matching MTA defaults."""
        markers = [
            "hip_right", "hip_left", "knee_left", "knee_right",
            "tail_base", "tail_end",
            "spine_lower", "pelvis_root", "spine_middle", "spine_upper",
            "head_back", "head_left", "head_front", "head_right", "head_top",
        ]
        connections = [
            ["head_front", "head_left"],
            ["head_front", "head_right"],
            ["head_right", "head_left"],
            ["head_left",  "head_back"],
            ["head_right", "head_back"],
            ["head_back",  "spine_upper"],
            ["spine_upper", "spine_middle"],
            ["spine_middle", "pelvis_root"],
            ["pelvis_root",  "hip_right"],
            ["pelvis_root",  "hip_left"],
            ["pelvis_root",  "spine_lower"],
            ["spine_lower",  "hip_right"],
            ["spine_lower",  "hip_left"],
            ["hip_left",     "hip_right"],
            ["hip_left",     "knee_left"],
            ["hip_right",    "knee_right"],
            ["head_top",     "head_front"],
            ["head_top",     "head_back"],
            ["tail_base",    "spine_lower"],
        ]
        return cls(markers=markers, connections=connections)

    @classmethod
    def from_csv_headers(cls, header_names: list[str]) -> "NBModel":
        """Build a model from Motive CSV column-label strings.

        Motive exports marker position columns labelled like
        ``'RatW:head_front'`` — the part after the colon is the
        marker name.
        """
        cleaned = []
        for h in header_names:
            if ":" in h:
                cleaned.append(h.split(":")[-1].strip())
            else:
                cleaned.append(h.strip())
        return cls(markers=cleaned)
