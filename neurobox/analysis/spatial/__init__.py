"""
neurobox.analysis.spatial
==========================
Place fields, occupancy maps, and other spatial-coding analyses.

Port of MTA's :file:`@MTAApfs` and :file:`utilities/generate_occupancy_map.m`,
with the persistence and incremental-update layer stripped out (use
``np.savez`` / ``pickle`` on the result objects).
"""

from .occupancy   import occupancy_map, OccupancyResult
from .place_fields import place_field, PlaceFieldResult
from .place_field_stats import (
    place_field_stats,
    Patch,
    UnitStats,
)

__all__ = [
    "occupancy_map",
    "OccupancyResult",
    "place_field",
    "PlaceFieldResult",
    "place_field_stats",
    "Patch",
    "UnitStats",
]
