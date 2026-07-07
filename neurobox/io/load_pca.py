"""
load_pca.py
===========
Load neurosuite-3 binary ``.pca.<method>.N`` files (or the untagged
legacy ``.pca.N``) — the PCA eigenvector basis used to project
spike waveforms into feature space.

File format
-----------
Binary, little-endian, no separator between sections:

======================  ======================================
Bytes                   Content
======================  ======================================
0 – 4                   ``nCh``      (int32)  — channel count
4 – 8                   ``data2use`` (int32)  — data-selection tag
                                    written by the PCA plugin
                                    (implementation-specific)
8 – 12                  ``nComp``    (int32)  — principal comps
12 – 16                 ``isCentered`` (int32) — 0 / 1 flag
16 – 20                 ``recShift`` (int32)  — sample shift used
                                    when the basis was fit
20 – …                  ``means``       (nCh × nSamples doubles)
…                        ``eigenvecs``   (nCh × nComp × nSamples
                                          doubles)
======================  ======================================

The *nSamples* value is NOT stored in the header — the caller must
supply it (from the ``spikeDetection.channelGroups[N].nSamples``
entry in the session YAML) so we can slice the trailing double
block correctly.

Variant naming (neurosuite-3)
-----------------------------
``.pca`` is a **MethodSpecific** artifact under the neurosuite-3
*variant naming convention* (see
``doc/ndmanager-plugins/formats/naming.md`` in the neurosuite-3
repository).  The canonical layout is::

    <base>.pca.<method>.<shank>

The retired ``.pcaD.N`` name is subsumed by ``method='stderiv'``.
Each variant pairs with the matching ``.fet.<method>.<shank>``.

Written by ``ndm_pca`` (standard) and ``ndm_pca --method stderiv``.
Read by Klusters (realignment / nudge) and by shadow-cluster
reprojection when new spikes arrive after the initial sort.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np


_HEADER_DTYPE = np.dtype("<i4")
_HEADER_NBYTES = 5 * _HEADER_DTYPE.itemsize


class PcaBasis(NamedTuple):
    """Loaded PCA basis for one shank.

    Attributes
    ----------
    n_channels, n_samples, n_components:
        Basis dimensions.  ``n_channels`` and ``n_components`` come
        from the file header; ``n_samples`` is passed by the caller.
    data2use:
        Data-selection tag from the header (a plugin implementation
        detail — preserved verbatim for round-tripping).
    is_centered:
        Whether the eigenvectors were fit on centered data (True/False).
    rec_shift:
        Sample shift used when the basis was fit.
    means:
        Shape ``(n_channels, n_samples)``, float64.  Per-channel
        mean waveform subtracted before projection.
    eigenvectors:
        Shape ``(n_channels, n_components, n_samples)``, float64.
        Principal components per channel.
    """
    n_channels:   int
    n_samples:    int
    n_components: int
    data2use:     int
    is_centered:  bool
    rec_shift:    int
    means:        np.ndarray     # (n_channels, n_samples)                  float64
    eigenvectors: np.ndarray     # (n_channels, n_components, n_samples)    float64


def load_pca(
    pca_file:  str | Path,
    n_samples: int,
) -> PcaBasis:
    """Load a ``.pca.<method>.N`` / legacy ``.pca.N`` file.

    Parameters
    ----------
    pca_file:
        Path to the file.
    n_samples:
        Number of samples per waveform (must match the value used
        when the basis was fit — usually the ``nSamples`` field in
        the session YAML's ``spikeDetection.channelGroups[N]``).

    Returns
    -------
    PcaBasis

    Raises
    ------
    FileNotFoundError
        When *pca_file* does not exist.
    ValueError
        When file size doesn't match the header + expected block sizes.
    """
    pca_file = Path(pca_file)
    if not pca_file.exists():
        raise FileNotFoundError(f"PCA basis file not found: {pca_file}")

    with open(pca_file, "rb") as fh:
        raw_header = fh.read(_HEADER_NBYTES)
        if len(raw_header) < _HEADER_NBYTES:
            raise ValueError(
                f"{pca_file.name}: truncated header "
                f"({len(raw_header)}/{_HEADER_NBYTES} bytes)"
            )
        n_ch, data2use, n_comp, is_centered, rec_shift = (
            int(x) for x in np.frombuffer(raw_header, dtype=_HEADER_DTYPE)
        )

        expected_means_count = n_ch * n_samples
        expected_evec_count  = n_ch * n_comp * n_samples
        expected_bytes = _HEADER_NBYTES + (
            (expected_means_count + expected_evec_count) * 8
        )
        actual = pca_file.stat().st_size
        if actual != expected_bytes:
            raise ValueError(
                f"{pca_file.name}: size mismatch — expected "
                f"{expected_bytes} bytes for header + means "
                f"(n_ch={n_ch} × n_samples={n_samples}) + "
                f"eigenvecs (n_ch × n_comp={n_comp} × n_samples), "
                f"got {actual}"
            )

        means = np.fromfile(
            fh, dtype="<f8", count=expected_means_count,
        ).reshape(n_ch, n_samples)
        eigenvectors = np.fromfile(
            fh, dtype="<f8", count=expected_evec_count,
        ).reshape(n_ch, n_comp, n_samples)

    return PcaBasis(
        n_channels   = n_ch,
        n_samples    = n_samples,
        n_components = n_comp,
        data2use     = data2use,
        is_centered  = bool(is_centered),
        rec_shift    = rec_shift,
        means        = means,
        eigenvectors = eigenvectors,
    )
