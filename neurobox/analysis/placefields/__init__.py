"""
neurobox.analysis.placefields
==============================
Place-field-derived per-sample directional / distance scores.

Contents
--------
* :mod:`directional_zones` — Directional Rate Zone (DRZ),
  Directional Distance Zone (DDZ), Gaussian variants (GDZ/GHZ),
  and head-anchored variants (HDZ/HRZ/HPV/TPV).  Ports of
  :file:`MTA/analysis/placefields/compute_*.m`.

The actual placefield computation lives in
:mod:`neurobox.analysis.spatial.place_fields` (round 5);
patch-level statistics in
:mod:`neurobox.analysis.spatial.place_field_stats` (round 9).
"""

from .directional_zones import (
    compute_drz,
    compute_ddz,
    compute_ghz,
    compute_gdz,
    compute_hdz,
    compute_hrz,
    compute_hpv,
    compute_tpv,
    field_centres_from_result,
)
from .egocentric import (
    egocentric_position,
    compute_ego_ratemap,
    compute_ego_ratemap_conditioned,
)


__all__ = [
    "compute_drz",
    "compute_ddz",
    "compute_ghz",
    "compute_gdz",
    "compute_hdz",
    "compute_hrz",
    "compute_hpv",
    "compute_tpv",
    "field_centres_from_result",
    "egocentric_position",
    "compute_ego_ratemap",
    "compute_ego_ratemap_conditioned",
]
