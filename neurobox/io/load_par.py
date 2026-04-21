"""
load_par.py
===========
Load the neurosuite-3 YAML session parameter file.

The function accepts either a full path (with or without ``.yaml``
extension) or a bare session base path and appends ``.yaml``
automatically.
"""

from __future__ import annotations

from pathlib import Path

from neurobox.dtype import Struct
from neurobox.io.load_yaml import load_yaml


def load_par(file_name: str | Path) -> Struct:
    """Load a neurosuite-3 YAML parameter file.

    Parameters
    ----------
    file_name:
        Full path to the ``.yaml`` file, or a bare session base path
        (no extension) — ``.yaml`` will be appended automatically.

    Returns
    -------
    par : Struct
        Parsed parameter object.  Key attributes:

        ``par.acquisitionSystem.nChannels``    — total channel count
        ``par.acquisitionSystem.nBits``        — ADC bit depth (usually 16)
        ``par.acquisitionSystem.samplingRate`` — wideband sample rate (Hz)
        ``par.spikeDetection.channelGroups``   — list of Struct, one per shank
        ``par.anatomicalDescription.channelGroups`` — list of Struct, one per shank
        ``par.probes``                          — list of Struct (probe metadata)

    Raises
    ------
    FileNotFoundError
        When the ``.yaml`` file cannot be found.
    """
    path = Path(file_name)
    if path.suffix.lower() != ".yaml":
        path = path.with_suffix(".yaml")

    if not path.exists():
        raise FileNotFoundError(f"YAML parameter file not found: {path}")

    return load_yaml(path)
