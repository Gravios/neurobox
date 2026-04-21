"""
load_spk.py
===========
Load spike waveform files produced by ndm_extractspikes.

File format (``.spk.N``)
--------------------------
Flat binary, little-endian int16, no header.  The three-dimensional
structure is implicit — dimensions must be supplied by the caller:

    (n_spikes × n_samples_per_spike × n_channels_in_group)

Waveforms are stored in C order (spike-major → sample-major →
channel-minor), so the on-disk layout is:

    spk0_samp0_ch0  spk0_samp0_ch1 ... spk0_sampS_ch0 ... spkN_sampS_chC

This matches the layout expected by KlustaKwik and ndm_estimatedrift.

``nSamples`` (number of waveform samples) and ``peakSampleIndex`` come
from the ``spikeDetection.channelGroups[N]`` block in the session
``.xml`` / ``.yaml`` file.

Variant: ``.spkD.N``
---------------------
Produced by ``process_extractspikes_stderiv``.  Identical layout to
``.spk.N`` but stores ``nChannels − 1`` sites (the spatial derivative
between adjacent channels).  Pass ``n_channels_in_group - 1`` as
*n_channels* when loading ``.spkD`` files.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np


def load_spk(
    spk_file: str | Path,
    n_samples: int,
    n_channels: int,
    n_spikes: int | None = None,
    as_float: bool = False,
    uv_per_bit: float | None = None,
) -> np.ndarray:
    """Load a Neurosuite waveform binary file.

    Parameters
    ----------
    spk_file:
        Path to the ``.spk.N`` or ``.spkD.N`` file.
    n_samples:
        Number of samples per waveform snippet (*nSamples* in the YAML /
        XML parameter block for this group, typically 32–64).
    n_channels:
        Number of channels recorded in this spike group (*len(channels)*
        in spikeDetection.channelGroups[N]).  For ``.spkD`` files pass
        ``n_channels_in_group - 1``.
    n_spikes:
        Expected number of spikes.  When *None* the count is inferred
        from the file size.  Providing it explicitly is a sanity check:
        a :class:`ValueError` is raised on mismatch.
    as_float:
        When *True*, cast to float64 before returning.
    uv_per_bit:
        If given, multiply by this factor and return float32 (µV).
        Overrides *as_float*.

    Returns
    -------
    waveforms : np.ndarray, shape ``(n_spikes, n_samples, n_channels)``
        Waveform snippets in int16 (or float32/float64 if converted).

    Raises
    ------
    FileNotFoundError
        When *spk_file* does not exist.
    ValueError
        When file size is inconsistent with the given dimensions, or when
        the inferred spike count does not match *n_spikes*.
    """
    spk_file = Path(spk_file)
    if not spk_file.exists():
        raise FileNotFoundError(f"Spike waveform file not found: {spk_file}")

    frame_size: int = n_samples * n_channels   # int16 values per spike
    file_size_bytes: int = spk_file.stat().st_size
    n_values: int = file_size_bytes // 2       # 2 bytes per int16

    if n_values % frame_size != 0:
        raise ValueError(
            f"{spk_file.name}: file size {file_size_bytes} bytes is not a "
            f"multiple of frame_size {frame_size} (n_samples={n_samples} × "
            f"n_channels={n_channels})."
        )

    inferred_n_spikes: int = n_values // frame_size

    if n_spikes is not None and inferred_n_spikes != n_spikes:
        raise ValueError(
            f"{spk_file.name}: expected {n_spikes} spikes but file "
            f"contains {inferred_n_spikes}."
        )

    raw = np.fromfile(str(spk_file), dtype="<i2", count=inferred_n_spikes * frame_size)
    waveforms = raw.reshape(inferred_n_spikes, n_samples, n_channels)

    if uv_per_bit is not None:
        return waveforms.astype(np.float32) * np.float32(uv_per_bit)
    if as_float:
        return waveforms.astype(np.float64)
    return waveforms


def load_spk_from_par(
    file_base: str | Path,
    shank: int,
    par=None,
    n_spikes: int | None = None,
    uv_per_bit: float | None = None,
) -> np.ndarray:
    """Convenience wrapper that reads *n_samples* and *n_channels*
    from the session parameter object.

    Parameters
    ----------
    file_base:
        Session base path without extension.
    shank:
        1-based shank / spike-group index.
    par:
        Parsed parameter object from :func:`~neurobox.io.load_par`.
        If *None*, the sidecar is loaded automatically.
    n_spikes:
        Optional spike count for validation.
    uv_per_bit:
        Optional µV conversion factor.

    Returns
    -------
    waveforms : np.ndarray, shape ``(n_spikes, n_samples, n_channels)``
    """
    file_base = Path(str(file_base))

    if par is None:
        from neurobox.io.load_par import load_par
        par = load_par(str(file_base))

    # Prefer .spkD if present (standard-deviation-projection variant)
    spkD_path = Path(f"{file_base}.spkD.{shank}")
    spk_path  = Path(f"{file_base}.spk.{shank}")
    use_path  = spkD_path if spkD_path.exists() else spk_path
    is_stderiv = use_path == spkD_path

    # Extract n_samples and n_channels for this group from the parameter file
    # Support both XML (Struct) and YAML (dict-backed Struct) formats
    n_samples_spk  = 32   # sensible default
    n_channels_spk = 8    # sensible default

    try:
        # YAML path: spikeDetection.channelGroups is a list of dicts
        spike_groups = par.spikeDetection.channelGroups
        if isinstance(spike_groups, list):
            grp = spike_groups[shank - 1]
            if isinstance(grp, dict):
                n_samples_spk  = int(grp.get("nSamples", n_samples_spk))
                chs = grp.get("channels", [])
                n_channels_spk = len(chs) if chs else n_channels_spk
            else:
                n_samples_spk  = int(getattr(grp, "nSamples", n_samples_spk))
        # XML path: spikeDetection.channelGroups.group is a list of Structs
        elif hasattr(spike_groups, "group"):
            grp_list = spike_groups.group
            if not isinstance(grp_list, list):
                grp_list = [grp_list]
            grp = grp_list[shank - 1]
            n_samples_spk  = int(getattr(grp, "nSamples", n_samples_spk))
            chs = getattr(grp, "channels", None)
            if chs is not None:
                ch_list = getattr(chs, "channel", chs)
                if not isinstance(ch_list, list):
                    ch_list = [ch_list]
                n_channels_spk = len(ch_list)
    except (AttributeError, IndexError, TypeError):
        pass  # keep defaults

    if is_stderiv:
        n_channels_spk = max(1, n_channels_spk - 1)

    return load_spk(use_path, n_samples_spk, n_channels_spk,
                    n_spikes=n_spikes, uv_per_bit=uv_per_bit)
