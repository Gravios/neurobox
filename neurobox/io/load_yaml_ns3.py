"""
load_yaml_ns3.py
================
Readers for the neurosuite-3 YAML per-shank and session-level
formats: ``.col`` (collision decomposition results) and ``.drift``
(probe drift trajectories).

See the spec files at:
* ``doc/ndmanager-plugins/formats/col.md``
* ``doc/ndmanager-plugins/formats/drift.md``
in the neurosuite-3 repository.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


__all__ = ["load_col", "load_drift"]


def load_col(col_file: str | Path) -> dict[str, Any]:
    """Load a ``.col.<method>.N`` collision-decomposition file.

    Returns the parsed YAML content unchanged — a dict with the
    top-level ``collisions`` key mapping to a dict with ``format``,
    ``spikeGroup``, and ``spikes`` fields.  See the neurosuite-3
    ``col.md`` for the schema.

    Parameters
    ----------
    col_file:
        Path to the ``.col.<method>.N`` YAML file.

    Returns
    -------
    dict
        The parsed YAML document.  Typically::

            {"collisions": {
                "format":     "1.0",
                "spikeGroup": 1,
                "spikes":     [ {spikeIndex: 4721, isCollision: true,
                                 components: [{unit, shift, amplitude}, ...]},
                                ... ]
            }}

    Raises
    ------
    FileNotFoundError
        When *col_file* does not exist.
    """
    col_file = Path(col_file)
    if not col_file.exists():
        raise FileNotFoundError(f"Collision file not found: {col_file}")
    with open(col_file, "r") as fh:
        return yaml.safe_load(fh) or {}


def load_drift(drift_file: str | Path) -> dict[str, Any]:
    """Load a ``.drift`` probe-drift-trajectory file (session-level).

    Returns the parsed YAML content unchanged.  See the neurosuite-3
    ``drift.md`` for the schema.

    Parameters
    ----------
    drift_file:
        Path to the session-level ``.drift`` YAML file.

    Returns
    -------
    dict
        Parsed YAML.  Typically::

            {"drift": {
                "format":    "1.0",
                "method":    "unit_com",
                "windowSec": 60.0,
                "probes":    [ {probeId: 0, shanks: [...]}, ... ],
            }}

    Raises
    ------
    FileNotFoundError
        When *drift_file* does not exist.
    """
    drift_file = Path(drift_file)
    if not drift_file.exists():
        raise FileNotFoundError(f"Drift file not found: {drift_file}")
    with open(drift_file, "r") as fh:
        return yaml.safe_load(fh) or {}
