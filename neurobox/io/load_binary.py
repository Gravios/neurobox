"""
load_binary.py
==============
Load contiguous binary data files produced by Amplipex / Intan /
NeuraLynx-converted pipelines in the Neurosuite format.

Covers both wide-band (.dat, 20 kHz) and LFP (.lfp, 1250 Hz) files.
The on-disk layout is int16, channels interleaved, little-endian:

    sample_0_ch_0  sample_0_ch_1 ... sample_0_ch_N
    sample_1_ch_0  ...
    ...

Parameters are read from the associated .xml or .yaml session file via
:func:`~neurobox.io.load_par.load_par`.
"""

from __future__ import annotations

import mmap
import os
from pathlib import Path

import numpy as np

from neurobox.dtype import Struct
from neurobox.io.load_par import load_par as _load_par


def load_binary(
    file_name: str | Path,
    channels: list[int],
    par: Struct | None = None,
    periods: np.ndarray | None = None,
    uv_per_bit: float | None = None,
    channel_first: bool = False,
) -> np.ndarray:
    """Load selected channels from a Neurosuite binary file.

    Parameters
    ----------
    file_name:
        Path to the ``.dat`` or ``.lfp`` file.
    channels:
        0-based list of channel indices to extract.  Must be a subset of
        ``[0, par.acquisitionSystem.nChannels)``.
    par:
        Parsed parameter object (from :func:`load_par`).  If *None* the
        function will attempt to locate and load the ``.xml`` / ``.yaml``
        sidecar automatically.
    periods:
        Optional ``(N, 2)`` int array of ``[start_sample, end_sample)``
        intervals to load.  When *None* the entire file is loaded.
    uv_per_bit:
        ADC conversion factor (µV per raw integer count).  Typical values:
        0.195 (Intan RHD/RHS), 0.1 (Amplipex), None → return raw int16.
        When provided the output dtype is float32.
    channel_first:
        If *True* return shape ``(n_channels, n_samples)``; if *False*
        (default) return shape ``(n_samples, n_channels)``.

    Returns
    -------
    data : np.ndarray
        Raw int16 counts, or float32 µV if *uv_per_bit* is given.
        Shape depends on *channel_first*.

    Raises
    ------
    FileNotFoundError
        If *file_name* does not exist.
    ValueError
        If *channels* contains out-of-range indices.
    """
    file_name = Path(file_name)
    if not file_name.exists():
        raise FileNotFoundError(f"Binary file not found: {file_name}")

    # ── Parameter loading ──────────────────────────────────────────────── #
    if par is None:
        stem = str(file_name.with_suffix(""))
        par = _load_par(stem)

    n_channels_total: int = int(par.acquisitionSystem.nChannels)
    n_bits: int = int(par.acquisitionSystem.nBits)
    dtype_raw = np.dtype(f"<i{n_bits // 8}")   # always little-endian

    # ── Validate channel list ──────────────────────────────────────────── #
    channels = list(channels)
    if max(channels) >= n_channels_total or min(channels) < 0:
        raise ValueError(
            f"channels must be in [0, {n_channels_total}); "
            f"got range [{min(channels)}, {max(channels)}]"
        )

    dtype_size: int = dtype_raw.itemsize
    file_size: int = file_name.stat().st_size
    n_samples_total: int = file_size // (dtype_size * n_channels_total)

    # ── Period setup ──────────────────────────────────────────────────── #
    if periods is None:
        periods = np.array([[0, n_samples_total]], dtype=np.int64)
    else:
        periods = np.asarray(periods, dtype=np.int64)
        periods = np.clip(periods, 0, n_samples_total)

    total_out_samples: int = int(np.diff(periods, axis=1).sum())
    n_ch_out: int = len(channels)

    out_dtype = np.float32 if uv_per_bit is not None else dtype_raw
    # Internal layout: (n_channels, n_samples) — transpose at end if needed
    data = np.empty((n_ch_out, total_out_samples), dtype=out_dtype)

    # ── Load each period via mmap ──────────────────────────────────────── #
    col_offset: int = 0

    for period in periods:
        start_samp = int(period[0])
        stop_samp  = int(period[1])
        n_read     = stop_samp - start_samp
        if n_read <= 0:
            continue

        byte_offset: int = start_samp * n_channels_total * dtype_size
        byte_length: int = n_read     * n_channels_total * dtype_size

        fd = os.open(str(file_name), os.O_RDONLY)
        try:
            mm = mmap.mmap(fd, length=byte_length, offset=byte_offset,
                           access=mmap.ACCESS_READ)
            try:
                # shape: (n_read, n_channels_total)
                chunk = np.frombuffer(mm, dtype=dtype_raw).reshape(n_read, n_channels_total).copy()
                chunk_sel = chunk[:, channels].T          # (n_ch_out, n_read)
                if uv_per_bit is not None:
                    data[:, col_offset:col_offset + n_read] = (
                        chunk_sel.astype(np.float32) * uv_per_bit
                    )
                else:
                    data[:, col_offset:col_offset + n_read] = chunk_sel
            finally:
                mm.close()
        finally:
            os.close(fd)

        col_offset += n_read

    return data if channel_first else data.T
