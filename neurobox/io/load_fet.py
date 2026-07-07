"""
load_fet.py
===========
Load PCA feature vectors from neurosuite-3 binary ``.fet.<method>.N``
files (or the untagged legacy form ``.fet.N``).

File format
-----------
Binary, little-endian.  Header is a single ``int32_t`` giving
``nDimensions`` (= ``nChannels × nPCs + nExtraFeatures + 1`` — the
last column is the spike timestamp / sample index).  The header is
followed by ``nSpikes × nDimensions × int64_t`` values in row-major
order.

On disk (with ``D = nDimensions``)::

    header int32
    spike0_dim0  spike0_dim1  …  spike0_dim(D-1)
    spike1_dim0  spike1_dim1  …  spike1_dim(D-1)
    …

The last column of each row is the spike timestamp (sample index) —
the same value as the corresponding entry in ``.res``.

Variant naming (neurosuite-3)
-----------------------------
``.fet`` is a **MethodSpecific** artifact under the neurosuite-3
*variant (chain-of-custody) naming convention* (see
``doc/ndmanager-plugins/formats/naming.md`` in the neurosuite-3
repository).  The canonical layout is::

    <base>.fet.<method>.<shank>

The retired ``.fetD.N`` name is subsumed by ``method='stderiv'``.
Every ``.fet.<method>.<shank>`` pairs with a matching
``.pca.<method>.<shank>``.

Written by ``ndm_pca`` (standard) and ``ndm_pca --method stderiv``.
Consumed by KiloKlustaKwik and Klusters.  Because the resolution
rule for MethodSpecific artifacts is **strict** — no cross-variant
fallback — a stderiv session that finds no ``.fet.stderiv.N`` is
an error, not a silent fall-back to the standard variant.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np


class FetData(NamedTuple):
    """Loaded feature-file contents.

    Attributes
    ----------
    features:
        Shape ``(n_spikes, n_dimensions - 1)``, int64.  All feature
        columns (PCA + extras) *excluding* the timestamp column.
    timestamps:
        Shape ``(n_spikes,)``, int64.  Spike sample indices
        (the last column of every on-disk row).
    n_dimensions:
        The raw header value — total dimensions per row *including*
        the timestamp column.  Useful for sanity-checking against
        the paired ``.pca`` file's ``nCh × nComp`` product plus
        the expected extras.
    """
    features:     np.ndarray     # (n_spikes, n_dimensions - 1) int64
    timestamps:   np.ndarray     # (n_spikes,)                    int64
    n_dimensions: int


def load_fet(
    fet_file:  str | Path,
    n_spikes:  int | None = None,
) -> FetData:
    """Load a ``.fet.<method>.N`` / legacy ``.fet.N`` file.

    Parameters
    ----------
    fet_file:
        Path to the feature file.
    n_spikes:
        Optional expected spike count.  If provided, raises
        :class:`ValueError` on mismatch.  When *None* the count is
        inferred from the file size.

    Returns
    -------
    FetData

    Raises
    ------
    FileNotFoundError
        When *fet_file* does not exist.
    ValueError
        On file-size / dimension inconsistency, or an ``n_spikes``
        mismatch.
    """
    fet_file = Path(fet_file)
    if not fet_file.exists():
        raise FileNotFoundError(f"Feature file not found: {fet_file}")

    file_size = fet_file.stat().st_size
    if file_size < 4:
        raise ValueError(
            f"{fet_file.name}: file is too small to contain a header "
            f"(size={file_size} bytes)"
        )

    with open(fet_file, "rb") as fh:
        header_bytes = fh.read(4)
        n_dim = int(np.frombuffer(header_bytes, dtype="<i4")[0])
        if n_dim < 1:
            raise ValueError(
                f"{fet_file.name}: header nDimensions={n_dim} is invalid"
            )
        row_size = n_dim * 8    # int64_t
        payload_bytes = file_size - 4
        if payload_bytes % row_size != 0:
            raise ValueError(
                f"{fet_file.name}: payload {payload_bytes} bytes is not "
                f"a multiple of row size {row_size} "
                f"(n_dim={n_dim} × 8 bytes)"
            )
        inferred_n = payload_bytes // row_size
        if n_spikes is not None and inferred_n != n_spikes:
            raise ValueError(
                f"{fet_file.name}: expected {n_spikes} spikes but file "
                f"contains {inferred_n}"
            )

        payload = np.fromfile(fh, dtype="<i8", count=inferred_n * n_dim)

    payload = payload.reshape(inferred_n, n_dim)
    # Last column is the spike timestamp (sample index)
    timestamps = np.ascontiguousarray(payload[:, -1])
    features   = np.ascontiguousarray(payload[:, :-1])
    return FetData(
        features     = features,
        timestamps   = timestamps,
        n_dimensions = n_dim,
    )
