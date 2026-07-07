"""
load_spk.py
===========
Load spike waveform files produced by ``ndm_extractspikes`` and
``ndm_reextractspikes`` (neurosuite-3).

File format (``.spk.<method>.N`` or legacy untagged ``.spk.N``)
--------------------------------------------------------------
Flat binary, little-endian int16, no header.  Shape:

    (n_spikes, n_samples_per_spike, n_channels_in_group)

Layout on disk is **sample-major within each spike, channel-minor
within each sample** — equivalent to numpy C order of the shape
above.  Reading with
``np.fromfile(...).reshape(n_spikes, n_samples, n_channels)`` is
correct.

``nSamples`` (number of waveform samples) and ``peakSampleIndex`` come
from the ``spikeDetection.channelGroups[N]`` block in the session
``.xml`` / ``.yaml`` file.

Variant naming (neurosuite-3)
-----------------------------
Under the neurosuite-3 *variant (chain-of-custody) naming convention*,
``.spk`` is a **Shared** artifact — the raw waveform snippets are
method-independent, so one physical copy is shared across variants.
Resolution order for a request with method *m*:

    <base>.spk.<m>.<shank>            preferred
    <base>.spk.standard.<shank>       fallback (if m != 'standard')
    <base>.spk.<shank>                untagged legacy fallback

The retired ``.spkD.N`` name is **gone** — the stderiv transform is
applied downstream at PCA time (``ndm_pca --method stderiv``), which
writes ``.fet.stderiv.N`` directly.  There is no separate stderiv
``.spk`` file to load.  A ``.spk`` that resolves to a raw/untagged
copy is not treated as stderiv even in a stderiv session.
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
    file_base:  str | Path,
    shank:      int,
    par=None,
    n_spikes:   int | None = None,
    uv_per_bit: float | None = None,
    method:     str = "standard",
) -> np.ndarray:
    """Convenience wrapper that reads *n_samples* and *n_channels*
    from the session parameter object.

    Under the neurosuite-3 variant naming convention, ``.spk`` is a
    **Shared** artifact — one physical file is shared across method
    variants because the stderiv transform is applied downstream at
    PCA time.  This wrapper searches, in order:

    1. ``<file_base>.spk.<method>.<shank>``
    2. ``<file_base>.spk.standard.<shank>`` (only if ``method != 'standard'``)
    3. ``<file_base>.spk.<shank>``               (untagged legacy)

    and reads the first one that exists.  If none exist, it raises
    :class:`FileNotFoundError` naming the primary path.

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
    method:
        Variant tag used for the Shared-artifact search.  Defaults
        to ``'standard'``.

    Returns
    -------
    waveforms : np.ndarray, shape ``(n_spikes, n_samples, n_channels)``
    """
    file_base = Path(str(file_base))

    if par is None:
        from neurobox.io.load_par import load_par
        par = load_par(str(file_base))

    # Neurosuite-3 Shared-artifact fall-back for .spk
    candidates: list[Path] = [
        Path(f"{file_base}.spk.{method}.{shank}"),
    ]
    if method != "standard":
        candidates.append(Path(f"{file_base}.spk.standard.{shank}"))
    candidates.append(Path(f"{file_base}.spk.{shank}"))    # untagged legacy

    use_path: Path | None = None
    for cand in candidates:
        if cand.exists():
            use_path = cand
            break
    if use_path is None:
        raise FileNotFoundError(
            f"No .spk file found for shank {shank}: tried "
            + ", ".join(str(c) for c in candidates)
        )

    # Extract n_samples and n_channels for this group from the parameter file
    # Support both XML (Struct) and YAML (dict-backed Struct) formats
    n_samples_spk  = 32   # sensible default
    n_channels_spk = 8    # sensible default

    try:
        # YAML path: spikeDetection.channelGroups is a list of dicts
        # (or Struct-typed entries when loaded via load_par)
        spike_groups = par.spikeDetection.channelGroups
        if isinstance(spike_groups, list):
            grp = spike_groups[shank - 1]
            if isinstance(grp, dict):
                n_samples_spk  = int(grp.get("nSamples", n_samples_spk))
                chs = grp.get("channels", [])
                n_channels_spk = len(chs) if chs else n_channels_spk
            else:
                # Struct-typed (load_par wraps YAML dicts in a Struct)
                n_samples_spk  = int(getattr(grp, "nSamples", n_samples_spk))
                chs = getattr(grp, "channels", None)
                if chs is not None:
                    ch_list = chs if isinstance(chs, list) else [chs]
                    n_channels_spk = len(ch_list) or n_channels_spk
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

    return load_spk(use_path, n_samples_spk, n_channels_spk,
                    n_spikes=n_spikes, uv_per_bit=uv_per_bit)
