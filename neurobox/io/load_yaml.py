"""
load_yaml.py
============
Load neurosuite-3 / ndManager-yaml session parameter files into a
:class:`Struct` tree.

YAML parameter file layout (ndManager-yaml format)
---------------------------------------------------
::

    acquisitionSystem:
      nChannels: 96
      nBits: 16
      samplingRate: 32552
      voltageRange: 20
      amplification: 1000
      offset: 0

    fieldPotentials:
      lfpSamplingRate: 1250

    anatomicalDescription:
      channelGroups:
        - channels:
            - {id: 0, skip: 0}
            - {id: 1, skip: 0}
            ...

    spikeDetection:
      channelGroups:
        - channels: [0, 1, 2, 3, 4, 5, 6, 7]
          nSamples: 41
          peakSampleIndex: 20
          nFeatures: 3

    probes:
      - id: 0
        probeFile: /path/to/Buzsaki64L.probe
        label: Buzsaki64L
        channelOffset: 0
        anatomicalGroups: [1, 2, 3, 4, 5, 6, 7, 8]

The returned :class:`Struct` exposes every top-level key as an attribute.
Nested dicts become nested :class:`Struct` objects; lists of dicts become
lists of :class:`Struct` objects; primitive values are left as-is.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from neurobox.dtype import Struct


# ---------------------------------------------------------------------------
# Recursive dict → Struct conversion
# ---------------------------------------------------------------------------

def _to_struct(value: Any) -> Any:
    """Recursively convert dicts → Struct, list-of-dicts → list-of-Struct."""
    if isinstance(value, dict):
        s = Struct()
        for k, v in value.items():
            setattr(s, k, _to_struct(v))
        return s
    if isinstance(value, list):
        return [_to_struct(item) for item in value]
    return value


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_yaml(file_name: str | Path) -> Struct:
    """Parse a neurosuite-3 YAML parameter file.

    Parameters
    ----------
    file_name:
        Path to the ``.yaml`` file.

    Returns
    -------
    par : Struct
        Top-level parameter struct.  Key attributes:

        ``par.acquisitionSystem.nChannels``        — total ADC channel count
        ``par.acquisitionSystem.samplingRate``      — wideband sample rate (Hz)
        ``par.acquisitionSystem.nBits``             — ADC resolution (usually 16)
        ``par.fieldPotentials.lfpSamplingRate``     — LFP sample rate (Hz)
        ``par.spikeDetection.channelGroups``        — list of Struct, one per shank
        ``par.anatomicalDescription.channelGroups`` — list of Struct, one per shank
        ``par.probes``                              — list of Struct (probe metadata)

    Raises
    ------
    FileNotFoundError
        When *file_name* does not exist.
    """
    file_name = Path(file_name)
    if not file_name.exists():
        raise FileNotFoundError(f"YAML parameter file not found: {file_name}")

    with open(file_name, "r") as fh:
        raw = yaml.safe_load(fh) or {}

    return _to_struct(raw)


def get_lfp_samplerate(par, default: float = 1250.0) -> float:
    """Return the LFP sample rate from a parsed parameter Struct.

    Checks locations in priority order:

    1. ``par.fieldPotentials.lfpSamplingRate`` — ndManager-yaml (new format)
    2. ``par.lfpSampleRate``                   — legacy ndManager XML / old YAML

    Parameters
    ----------
    par:
        Parsed parameter Struct from :func:`load_par`.
    default:
        Fallback value when the key is absent (default 1250.0 Hz).

    Returns
    -------
    float
    """
    # 1. New YAML format
    fp = getattr(par, "fieldPotentials", None)
    if fp is not None:
        v = getattr(fp, "lfpSamplingRate", None)
        if v is not None:
            return float(v)
    # 2. Legacy XML / old YAML flat key
    v = getattr(par, "lfpSampleRate", None)
    if v is not None:
        return float(v)
    return default


def get_channel_groups(par: Struct, source: str = "spike") -> list[list[int]]:
    """Extract channel lists for each spike group from a parameter Struct.

    Parameters
    ----------
    par:
        Parsed parameter object from :func:`load_yaml` or
        :func:`~neurobox.io.load_xml.load_xml`.
    source:
        ``"spike"`` (default) → ``spikeDetection.channelGroups``
        ``"anat"``            → ``anatomicalDescription.channelGroups``

    Returns
    -------
    channel_groups : list[list[int]]
        One sublist of 0-based channel indices per shank, in group order.
    """
    if source == "anat":
        top = getattr(par, "anatomicalDescription", None)
    else:
        top = getattr(par, "spikeDetection", None)

    if top is None:
        return []

    groups_attr = getattr(top, "channelGroups", None)
    if groups_attr is None:
        return []

    # YAML format: channelGroups is a list of Struct objects
    if isinstance(groups_attr, list):
        result = []
        for grp in groups_attr:
            if isinstance(grp, Struct):
                chs = getattr(grp, "channels", [])
            elif isinstance(grp, dict):
                chs = grp.get("channels", [])
            else:
                chs = []
            # anatomical groups store dicts {id, skip}; spike groups store ints
            ch_list = []
            for ch in (chs if isinstance(chs, list) else [chs]):
                if isinstance(ch, dict):
                    ch_list.append(int(ch["id"]))
                elif isinstance(ch, Struct):
                    ch_list.append(int(getattr(ch, "id", 0)))
                else:
                    ch_list.append(int(ch))
            result.append(ch_list)
        return result

    # XML format: channelGroups.group is a Struct or list of Structs
    group_list = getattr(groups_attr, "group", None)
    if group_list is None:
        return []
    if not isinstance(group_list, list):
        group_list = [group_list]

    result = []
    for grp in group_list:
        chs = getattr(grp, "channels", None)
        if chs is None:
            result.append([])
            continue
        ch_attr = getattr(chs, "channel", chs)
        if not isinstance(ch_attr, list):
            ch_attr = [ch_attr]
        result.append([int(c) for c in ch_attr])
    return result
