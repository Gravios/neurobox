"""
ns3_writers.py
==============
Binary writers for the neurosuite-3 per-shank artifact formats.

This module is the single canonical writer for every format
described in ``doc/ndmanager-plugins/formats/*.md`` of the
neurosuite-3 repository.  Reading the same formats lives in the
matching ``load_*.py`` modules.  Path naming is not the writer's
concern — construct paths with
:meth:`neurobox.dtype.paths.NBSessionPaths.ns3_file` and pass them
in.

Supported writers
-----------------

+---------------+-------------------------------+
| Function      | File                          |
+===============+===============================+
| ``save_res``  | ``.res[.method].N``           |
| ``save_clu``  | ``.clu[.method].N``           |
| ``save_clc``  | ``.clc[.method].N``           |
| ``save_clp``  | ``.clp[.method].N``           |
| ``save_spk``  | ``.spk[.method].N``           |
| ``save_fet``  | ``.fet[.method].N``           |
| ``save_pca``  | ``.pca[.method].N``           |
+---------------+-------------------------------+

All writers:

* accept an explicit path (variant-tagged or legacy — the writer
  doesn't inspect it);
* write **little-endian** binary regardless of host endianness;
* atomically rename ``<path>.tmp`` → ``<path>`` on completion so
  a crashed writer leaves no half-file behind;
* validate input shapes and dtypes and raise :class:`ValueError`
  with a helpful message when a mismatch is detected;
* accept ``overwrite=False`` (the default) to protect existing
  files.  Pass ``overwrite=True`` to replace.

Round-tripping
--------------
Every writer's output is bit-identical when reloaded by the
matching :mod:`neurobox.io.load_*` reader.  See
:mod:`tests.test_neurosuite3_formats` for the round-trip suite.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np


__all__ = [
    "save_res",
    "save_clu",
    "save_clc",
    "save_clp",
    "save_spk",
    "save_fet",
    "save_pca",
]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _prepare(path: Any, overwrite: bool) -> Path:
    """Validate the target path and its parent directory."""
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"{path} already exists.  Pass overwrite=True to replace."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _atomic_write(path: Path, write_fn) -> None:
    """Write to ``<path>.tmp`` then rename onto *path*.

    Robust to interruption — a crash mid-write leaves only the
    ``.tmp`` sidecar behind and never a truncated destination file.
    ``write_fn`` receives an open file handle in binary mode.
    """
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        write_fn(fh)
        fh.flush()
        try:
            os.fsync(fh.fileno())
        except OSError:
            # Best-effort — some filesystems (e.g. sshfs) don't support fsync
            pass
    os.replace(tmp, path)


def _as_le(arr: np.ndarray, dtype: str) -> np.ndarray:
    """Convert *arr* to a contiguous little-endian buffer of *dtype*.

    Handles native-endian input on both little- and big-endian
    hosts.  ``dtype`` is a numpy dtype string like ``'<i4'``.
    """
    return np.ascontiguousarray(arr, dtype=dtype)


# ---------------------------------------------------------------------------
# .res.[method].N  —  spike timestamps (int64 LE, no header)
# ---------------------------------------------------------------------------

def save_res(
    path:      str | Path,
    timestamps: np.ndarray,
    *,
    overwrite: bool = False,
) -> Path:
    """Write spike timestamps to a ``.res`` file.

    Parameters
    ----------
    path:
        Destination.  Use
        :meth:`NBSessionPaths.res_ns3_file` (or the legacy
        :meth:`~NBSessionPaths.res_file`) to construct it.
    timestamps:
        1-D array of spike sample indices.  Written as
        little-endian ``int64`` with no header.
    overwrite:
        If *False* (default) refuse to replace an existing file.

    Returns
    -------
    Path
        The destination path.
    """
    ts = np.asarray(timestamps)
    if ts.ndim != 1:
        raise ValueError(
            f"timestamps must be 1-D, got shape {ts.shape}"
        )
    payload = _as_le(ts, "<i8")
    target = _prepare(path, overwrite)
    _atomic_write(target, lambda fh: fh.write(payload.tobytes()))
    return target


# ---------------------------------------------------------------------------
# .clu.[method].N  and  .clc.[method].N  —  cluster IDs (int32 LE)
# ---------------------------------------------------------------------------

def _save_clu_like(
    path:      str | Path,
    cluster_ids: np.ndarray,
    *,
    overwrite: bool,
) -> Path:
    """Underlying writer shared by :func:`save_clu` and :func:`save_clc`.

    Both files use the same on-disk framing:

        int32 header (number of distinct clusters)
        int32[nSpikes]  cluster IDs
    """
    ids = np.asarray(cluster_ids)
    if ids.ndim != 1:
        raise ValueError(
            f"cluster_ids must be 1-D, got shape {ids.shape}"
        )
    payload = _as_le(ids, "<i4")
    # Header = number of distinct cluster IDs present in the array
    # (per clu.md: "int32_t header (number of clusters)").
    n_clusters = int(np.unique(payload).size) if payload.size else 0
    header = np.array([n_clusters], dtype="<i4")
    target = _prepare(path, overwrite)

    def _write(fh):
        fh.write(header.tobytes())
        fh.write(payload.tobytes())

    _atomic_write(target, _write)
    return target


def save_clu(
    path:      str | Path,
    cluster_ids: np.ndarray,
    *,
    overwrite: bool = False,
) -> Path:
    """Write the fiber-layer (or flat) cluster assignments to ``.clu``.

    In a hierarchical session this is the **parent** layer paired
    with a matching :func:`save_clc` / :func:`save_clp` triple.

    Parameters
    ----------
    path:
        Destination.  Use
        :meth:`NBSessionPaths.clu_ns3_file` for variant-tagged form
        or :meth:`NBSessionPaths.clu_file` for legacy untagged.
    cluster_ids:
        1-D array of per-spike cluster IDs.  Convention: 0 =
        noise/artefact, 1 = MUA, ≥ 2 = single units.
    """
    return _save_clu_like(path, cluster_ids, overwrite=overwrite)


def save_clc(
    path:      str | Path,
    atom_ids:  np.ndarray,
    *,
    overwrite: bool = False,
) -> Path:
    """Write the atom-layer (child) cluster assignments to ``.clc``.

    Only meaningful in a hierarchical session.  Aligned per-spike
    with the paired ``.res`` and ``.clu`` files.  The atom→fiber
    map lives in the paired :func:`save_clp` output.

    Parameters
    ----------
    path:
        Destination.  Construct with
        :meth:`NBSessionPaths.clc_file`.
    atom_ids:
        1-D array of per-spike atom IDs.  Atom IDs are *global*
        across the group (``1..nAtoms``), not per-fiber.
    """
    return _save_clu_like(path, atom_ids, overwrite=overwrite)


# ---------------------------------------------------------------------------
# .clp.[method].N  —  atom→fiber map (int32 LE)
# ---------------------------------------------------------------------------

def save_clp(
    path:      str | Path,
    parent_of: np.ndarray,
    *,
    overwrite: bool = False,
) -> Path:
    """Write the atom→fiber (child→parent) linkage to ``.clp``.

    Layout::

        int32 header (highest atom ID written)
        int32[nAtoms]  parent-fiber IDs, indexed by 1-based atom ID

    A parent-fiber ID of ``0`` marks the atom as noise / unmapped.

    Parameters
    ----------
    path:
        Destination.  Construct with
        :meth:`NBSessionPaths.clp_file`.
    parent_of:
        1-D array such that ``parent_of[i] = fiber_id_of_atom_(i+1)``
        (1-based atom IDs per the ``clp.md`` spec).
    """
    arr = np.asarray(parent_of)
    if arr.ndim != 1:
        raise ValueError(
            f"parent_of must be 1-D, got shape {arr.shape}"
        )
    payload = _as_le(arr, "<i4")
    # Header = highest atom ID = n_atoms (atom IDs are 1-based,
    # dense from 1 to nAtoms).  Klusters uses this to sanity-check
    # the payload size on read.
    header = np.array([int(payload.size)], dtype="<i4")
    target = _prepare(path, overwrite)

    def _write(fh):
        fh.write(header.tobytes())
        fh.write(payload.tobytes())

    _atomic_write(target, _write)
    return target


# ---------------------------------------------------------------------------
# .spk.[method].N  —  waveforms (int16 LE, no header)
# ---------------------------------------------------------------------------

def save_spk(
    path:      str | Path,
    waveforms: np.ndarray,
    *,
    overwrite: bool = False,
) -> Path:
    """Write spike waveforms to ``.spk``.

    Layout: flat ``int16`` little-endian, no header.  Shape is
    ``(n_spikes, n_samples, n_channels)`` with channels as the
    fastest-varying axis (sample-major within each spike).  This
    matches numpy C-order for the given shape, so writing the
    contiguous byte buffer is correct.

    Parameters
    ----------
    path:
        Destination.  Use :meth:`NBSessionPaths.spk_file`.  Because
        ``.spk`` is a Shared artifact under the neurosuite-3
        naming convention, one physical file is reused across
        variants — write only one copy, at the ``standard`` method
        tag, unless you have a variant-specific waveform set.
    waveforms:
        3-D array of shape ``(n_spikes, n_samples, n_channels)``,
        dtype coercible to ``int16``.
    """
    wf = np.asarray(waveforms)
    if wf.ndim != 3:
        raise ValueError(
            f"waveforms must be 3-D (n_spikes, n_samples, n_channels), "
            f"got shape {wf.shape}"
        )
    payload = _as_le(wf, "<i2")
    target = _prepare(path, overwrite)
    _atomic_write(target, lambda fh: fh.write(payload.tobytes()))
    return target


# ---------------------------------------------------------------------------
# .fet.[method].N  —  PCA feature vectors (int32 header + int64 payload)
# ---------------------------------------------------------------------------

def save_fet(
    path:       str | Path,
    features:   np.ndarray,
    timestamps: np.ndarray,
    *,
    overwrite:  bool = False,
) -> Path:
    """Write PCA feature vectors to ``.fet``.

    Combines the caller's per-spike ``features`` matrix with the
    aligned ``timestamps`` column into the on-disk row format:

        int32 header = n_dimensions (= features.shape[1] + 1)
        int64[nSpikes, n_dimensions], row-major, last column = timestamp

    Parameters
    ----------
    path:
        Destination.  Use :meth:`NBSessionPaths.fet_file`
        (with ``method`` in ``{'standard', 'stderiv', …}``).
    features:
        Shape ``(n_spikes, n_features)``, dtype coercible to
        ``int64``.  The *features* argument should exclude the
        timestamp column — this function appends it.
    timestamps:
        Shape ``(n_spikes,)``, dtype coercible to ``int64``.

    Round-trip: :func:`~neurobox.io.load_fet.load_fet` splits the
    on-disk row back into ``(features, timestamps)`` — so the
    ``FetData.features`` you get on reload matches the ``features``
    argument here.
    """
    feats = np.asarray(features)
    ts    = np.asarray(timestamps)
    if feats.ndim != 2:
        raise ValueError(
            f"features must be 2-D (n_spikes, n_features), "
            f"got shape {feats.shape}"
        )
    if ts.ndim != 1:
        raise ValueError(
            f"timestamps must be 1-D, got shape {ts.shape}"
        )
    if ts.size != feats.shape[0]:
        raise ValueError(
            f"timestamps has {ts.size} entries but features has "
            f"{feats.shape[0]} rows"
        )

    n_dim = feats.shape[1] + 1     # +1 for the appended timestamp column
    payload = np.empty((feats.shape[0], n_dim), dtype="<i8")
    payload[:, :-1] = feats
    payload[:, -1]  = ts
    header = np.array([n_dim], dtype="<i4")
    target = _prepare(path, overwrite)

    def _write(fh):
        fh.write(header.tobytes())
        fh.write(payload.tobytes())

    _atomic_write(target, _write)
    return target


# ---------------------------------------------------------------------------
# .pca.[method].N  —  PCA eigenvector basis (5-int32 header + doubles)
# ---------------------------------------------------------------------------

def save_pca(
    path:         str | Path,
    means:        np.ndarray,
    eigenvectors: np.ndarray,
    *,
    data2use:     int  = 0,
    is_centered:  bool = True,
    rec_shift:    int  = 0,
    overwrite:    bool = False,
) -> Path:
    """Write a PCA eigenvector basis to ``.pca``.

    Layout::

        int32 nCh
        int32 data2use
        int32 nComp
        int32 isCentered      (0 or 1)
        int32 recShift
        float64[nCh, nSamples]        means
        float64[nCh, nComp, nSamples] eigenvectors

    ``nSamples`` is not written to the header — it is fixed by the
    shapes of ``means`` and ``eigenvectors``, which the reader
    reconstructs by requiring the caller to pass ``n_samples``
    (from the session YAML's ``spikeDetection.channelGroups[N]``).

    Parameters
    ----------
    path:
        Destination.  Use :meth:`NBSessionPaths.pca_file`.
    means:
        Shape ``(n_channels, n_samples)``, dtype coercible to
        ``float64``.  Per-channel mean waveform subtracted before
        projection.
    eigenvectors:
        Shape ``(n_channels, n_components, n_samples)``, dtype
        coercible to ``float64``.  Principal components per
        channel.
    data2use, is_centered, rec_shift:
        Header metadata.  Round-tripped verbatim through
        :class:`~neurobox.io.load_pca.PcaBasis`; see the
        neurosuite-3 ``pca.md`` for their semantics.
    """
    m = np.asarray(means)
    e = np.asarray(eigenvectors)
    if m.ndim != 2:
        raise ValueError(
            f"means must be 2-D (n_channels, n_samples), got shape {m.shape}"
        )
    if e.ndim != 3:
        raise ValueError(
            f"eigenvectors must be 3-D (n_channels, n_components, n_samples), "
            f"got shape {e.shape}"
        )
    n_ch, n_samples = m.shape
    e_ch, n_comp, e_samples = e.shape
    if e_ch != n_ch:
        raise ValueError(
            f"eigenvectors n_channels ({e_ch}) doesn't match means "
            f"({n_ch})"
        )
    if e_samples != n_samples:
        raise ValueError(
            f"eigenvectors n_samples ({e_samples}) doesn't match means "
            f"({n_samples})"
        )

    header = np.array(
        [n_ch, int(data2use), n_comp, int(bool(is_centered)), int(rec_shift)],
        dtype="<i4",
    )
    means_le = _as_le(m, "<f8")
    evecs_le = _as_le(e, "<f8")
    target = _prepare(path, overwrite)

    def _write(fh):
        fh.write(header.tobytes())
        fh.write(means_le.tobytes())
        fh.write(evecs_le.tobytes())

    _atomic_write(target, _write)
    return target
