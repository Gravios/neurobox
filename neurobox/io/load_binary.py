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

Parameters are read from the associated .yaml session file via
:func:`~neurobox.io.load_par.load_par`.
"""

from __future__ import annotations

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
        0-based list of channel indices to extract.
    par:
        Parsed parameter object (from :func:`load_par`).  Auto-loaded
        from the ``.yaml`` sidecar if *None*.
    periods:
        Optional ``(N, 2)`` int array of ``[start_sample, end_sample)``
        intervals to load.  *None* → load the entire file.
    uv_per_bit:
        ADC conversion factor (µV per raw integer count).
        ``None`` → return raw int16.  When provided the output is float32.
    channel_first:
        ``True`` → shape ``(n_channels, n_samples)``.
        ``False`` (default) → shape ``(n_samples, n_channels)``.

    Returns
    -------
    np.ndarray
        Raw int16, or float32 µV if *uv_per_bit* is given.

    Raises
    ------
    FileNotFoundError
        When *file_name* does not exist.
    ValueError
        When *channels* contains out-of-range indices.
    """
    file_name = Path(file_name)
    if not file_name.exists():
        raise FileNotFoundError(f"Binary file not found: {file_name}")

    # ── Parameter loading ──────────────────────────────────────────────── #
    if par is None:
        stem = str(file_name.with_suffix(""))
        par = _load_par(stem)

    n_channels_total: int = int(par.acquisitionSystem.nChannels)
    n_bits: int           = int(par.acquisitionSystem.nBits)
    dtype_raw             = np.dtype(f"<i{n_bits // 8}")

    # ── Validate channels ──────────────────────────────────────────────── #
    channels = list(channels)
    if not channels:
        raise ValueError("channels must be a non-empty list.")
    if max(channels) >= n_channels_total or min(channels) < 0:
        raise ValueError(
            f"channels must be in [0, {n_channels_total}); "
            f"got range [{min(channels)}, {max(channels)}]."
        )

    dtype_size: int       = dtype_raw.itemsize
    file_size: int        = file_name.stat().st_size
    n_samples_total: int  = file_size // (dtype_size * n_channels_total)

    # ── Period setup ───────────────────────────────────────────────────── #
    if periods is None:
        periods = np.array([[0, n_samples_total]], dtype=np.int64)
    else:
        periods = np.asarray(periods, dtype=np.int64)
        periods = np.clip(periods, 0, n_samples_total)

    total_out_samples: int = int(np.diff(periods, axis=1).sum())
    n_ch_out: int          = len(channels)

    out_dtype  = np.float32 if uv_per_bit is not None else dtype_raw
    data       = np.empty((n_ch_out, total_out_samples), dtype=out_dtype)
    col_offset = 0

    # ── Load each period via seek + read ───────────────────────────────── #
    with open(str(file_name), "rb") as fh:
        for period in periods:
            start_samp = int(period[0])
            stop_samp  = int(period[1])
            n_read     = stop_samp - start_samp
            if n_read <= 0:
                continue

            fh.seek(start_samp * n_channels_total * dtype_size)
            buf   = fh.read(n_read * n_channels_total * dtype_size)
            chunk = np.frombuffer(buf, dtype=dtype_raw).reshape(n_read, n_channels_total)
            chunk_sel = chunk[:, channels].T      # (n_ch_out, n_read)

            if uv_per_bit is not None:
                data[:, col_offset:col_offset + n_read] = (
                    chunk_sel.astype(np.float32) * uv_per_bit
                )
            else:
                data[:, col_offset:col_offset + n_read] = chunk_sel

            col_offset += n_read

    return data if channel_first else data.T
