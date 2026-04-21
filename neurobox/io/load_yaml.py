"""
load_yaml.py
============
Load neurosuite-3 YAML session parameter files into a :class:`Struct`
interface compatible with the existing :func:`load_xml` workflow.

YAML parameter file layout (neurosuite-3)
-----------------------------------------
::

    acquisitionSystem:
      nChannels: 64
      nBits: 16
      samplingRate: 20000
      voltageRange: 20
      amplification: 1000
      offset: 0

    spikeDetection:
      channelGroups:
        - channels: [0, 1, 2, 3, 4, 5, 6, 7]
          nSamples: 52
          peakSampleIndex: 26
          nFeatures: 3
          probeId: 0
          shankIndex: 0

    anatomicalDescription:
      channelGroups:
        - channels:
            - {id: 0, skip: 0}
            ...
          probeId: 0
          shankIndex: 0

    probes:
      - probeId: 0
        probeFile: buzsaki64.probe
        channelOffset: 0
        anatomicalGroups: [1, 2, 3, 4, 5, 6, 7, 8]

The returned :class:`Struct` exposes the top-level keys as attributes.
Nested dicts become nested :class:`Struct` objects; lists of dicts
become lists of :class:`Struct` objects; primitive values are left as-is.
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

        ``par.acquisitionSystem.nChannels``   — total ADC channel count
        ``par.acquisitionSystem.samplingRate`` — wideband sample rate (Hz)
        ``par.acquisitionSystem.nBits``        — ADC resolution (usually 16)
        ``par.spikeDetection.channelGroups``   — list of Struct, one per shank
        ``par.anatomicalDescription.channelGroups`` — list of Struct, one per shank
        ``par.probes``                          — list of Struct (may be absent)

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
