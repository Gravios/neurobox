"""
load_units.py
=============
Parse the ``units:`` block from an ndManager-yaml parameter file.

The units block records per-cluster manual curation from klusters / phy:

.. code-block:: yaml

    units:
      - group:   7
        cluster: 4
        structure:       CA1
        type:            pyr
        isolationDistance: 18.3
        quality:         good
        notes:           clear waveform

Each entry identifies a cluster by ``(group, cluster)`` — matching the
``(shank, local_cluster_id)`` from the ``.clu.N`` / ``.res.N`` files —
and carries human-curated metadata.

Values of ``'~'``, ``'--'``, ``null``, or empty string are treated as
absent (stored as ``None``).

Usage
-----
::

    from neurobox.io import load_par, load_units

    par   = load_par('jg05-20120316.yaml')
    units = load_units(par)                    # list[UnitAnnotation]

    good  = [u for u in units if u.is_good()]
    print(f"{len(good)}/{len(units)} units flagged as good")

Integration with NBSpk
-----------------------
``NBSpk.load()`` automatically calls ``load_units`` when a par object is
available, storing the result in ``spk.annotations``.  You can then
filter units::

    spk = session.load('spk')
    good_ids = spk.annotated_unit_ids(quality='good')
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Sentinel values used in ndManager-yaml for "no data"
_ABSENT = frozenset({"~", "--", "", "null", None})


def _clean(val: Any) -> Any:
    """Return None for absent sentinels, else the value unchanged."""
    if val in _ABSENT:
        return None
    if isinstance(val, str) and val.strip().lower() in {"~", "--", "null", ""}:
        return None
    return val


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class UnitAnnotation:
    """Curation record for one sorted cluster.

    Attributes
    ----------
    group : int
        Electrode group index (shank number, 1-based in neurosuite).
    cluster : int
        Local cluster ID within the group (as in ``.clu.N``).
    global_id : int | None
        Global cluster ID as used in ``NBSpk.clu`` (set after mapping).
    structure : str | None
        Anatomical structure, e.g. ``'CA1'``, ``'DG'``.
    cell_type : str | None
        Cell type label, e.g. ``'pyr'``, ``'int'``, ``'mua'``.
    isolation_distance : float | None
        L-ratio isolation distance from PCA feature space.
    quality : str | None
        Manual quality tag, e.g. ``'good'``, ``'great'``, ``'mua'``,
        ``'noise'``.  The ``ndm_stripdat`` pipeline uses this field to
        build the autostrip list.
    notes : str | None
        Free-text operator notes.
    """
    group:              int
    cluster:            int
    global_id:          int | None  = None
    structure:          str | None  = None
    cell_type:          str | None  = None
    isolation_distance: float | None = None
    quality:            str | None  = None
    notes:              str | None  = None

    # Quality tags that indicate a well-isolated single unit
    _SINGLE_UNIT_TAGS: frozenset[str] = field(
        default_factory=lambda: frozenset(
            {"good", "great", "excellent", "su", "single_unit", "single-unit",
             "single unit", "accepted"}
        ),
        repr=False,
    )

    def is_good(self, tags: frozenset[str] | None = None) -> bool:
        """Return True if the quality tag is in *tags* (case-insensitive).

        Parameters
        ----------
        tags:
            Set of quality strings to accept.  Defaults to
            ``{'good', 'great', 'excellent', 'su', ...}``.
        """
        if self.quality is None:
            return False
        accept = tags if tags is not None else self._SINGLE_UNIT_TAGS
        return self.quality.strip().lower() in accept

    def is_mua(self) -> bool:
        """Return True if flagged as multi-unit activity."""
        if self.quality is None:
            return False
        return self.quality.strip().lower() in {"mua", "multi", "multi-unit", "multiunit"}

    def __repr__(self) -> str:
        return (f"UnitAnnotation(group={self.group}, cluster={self.cluster}, "
                f"quality={self.quality!r}, type={self.cell_type!r})")


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def load_units(par) -> list[UnitAnnotation]:
    """Parse the ``units:`` block from a par Struct.

    Parameters
    ----------
    par:
        Parameter Struct from :func:`~neurobox.io.load_par`.

    Returns
    -------
    list[UnitAnnotation]
        One entry per annotated cluster.  Empty list when the YAML has
        no ``units:`` block or all entries are absent.

    Notes
    -----
    The ``(group, cluster)`` pair maps to the ``.clu.N`` numbering:
    ``group`` is the shank index (1-based), ``cluster`` is the local
    cluster id within that shank file.  These are the same numbers
    visible in klusters.
    """
    units_raw = getattr(par, "units", None)
    if units_raw is None:
        return []

    # units_raw is a list of Struct objects (one per annotated unit)
    result: list[UnitAnnotation] = []
    for entry in (units_raw if isinstance(units_raw, list) else [units_raw]):
        # Accept both Struct objects and plain dicts
        if hasattr(entry, "__dict__"):
            d = entry.__dict__
        elif isinstance(entry, dict):
            d = entry
        else:
            continue

        group   = d.get("group")
        cluster = d.get("cluster")
        if group is None or cluster is None:
            continue

        try:
            group   = int(group)
            cluster = int(cluster)
        except (TypeError, ValueError):
            continue

        iso_raw = _clean(d.get("isolationDistance"))
        try:
            iso = float(iso_raw) if iso_raw is not None else None
        except (TypeError, ValueError):
            iso = None

        result.append(UnitAnnotation(
            group              = group,
            cluster            = cluster,
            structure          = _clean(d.get("structure")),
            cell_type          = _clean(d.get("type")),
            isolation_distance = iso,
            quality            = _clean(d.get("quality")),
            notes              = _clean(d.get("notes")),
        ))

    return result


def map_annotations_to_global_ids(
    annotations: list[UnitAnnotation],
    spk_map: "np.ndarray",
) -> list[UnitAnnotation]:
    """Set ``global_id`` on each annotation by matching against ``spk.map``.

    ``spk.map`` is an ``(N, 3)`` array:
    ``[global_id, shank (group), local_cluster_id]``.

    Parameters
    ----------
    annotations:
        List from :func:`load_units`.
    spk_map:
        ``NBSpk.map`` array.

    Returns
    -------
    The same list, with ``global_id`` filled in where a match is found.
    """
    import numpy as np
    if spk_map is None or len(spk_map) == 0:
        return annotations
    for ann in annotations:
        matches = np.where(
            (spk_map[:, 1] == ann.group) &
            (spk_map[:, 2] == ann.cluster)
        )[0]
        if len(matches) == 1:
            ann.global_id = int(spk_map[matches[0], 0])
    return annotations
