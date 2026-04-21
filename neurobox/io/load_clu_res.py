"""
load_clu_res.py
===============
Load spike timestamps and cluster assignments from neurosuite-3
binary ``.res.N`` / ``.clu.N`` file pairs.

Binary formats (neurosuite-3)
------------------------------
``.res.N`` — flat little-endian int64, no header.  One timestamp
(sample index) per spike, in time order.

``.clu.N`` — little-endian int32.  First int32 is nClusters (header,
discarded).  Remaining int32 values are cluster IDs, one per spike, in
the same order as ``.res.N``.

Cluster ID conventions
-----------------------
0   — noise / artefact
1   — multi-unit activity (MUA)
≥ 2 — isolated single units (returned by default)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from neurobox.io.load_par import load_par as _load_par


# ---------------------------------------------------------------------------
# Low-level binary readers
# ---------------------------------------------------------------------------

def _read_res(path: Path) -> np.ndarray:
    """Binary .res.N — flat int64 LE, no header."""
    return np.fromfile(str(path), dtype="<i8")


def _read_clu(path: Path) -> np.ndarray:
    """Binary .clu.N — int32 LE; first int32 is nClusters header (dropped)."""
    raw = np.fromfile(str(path), dtype="<i4")
    return raw[1:] if len(raw) > 1 else np.array([], dtype=np.int32)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_clu_res(
    file_base: str | Path,
    shank_groups: list[int] | None = None,
    clusters: list[int] | None = None,
    include_noise: bool = False,
    sampling_rate: float | None = None,
    as_seconds: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load spike times and cluster IDs from all shanks.

    Parameters
    ----------
    file_base:
        Session base path without extension
        (e.g. ``/data/jg05-20120312``).
    shank_groups:
        1-based shank indices to load.  *None* → read from the ``.yaml``
        parameter file, or glob for ``.res.*`` files as fallback.
    clusters:
        Restrict output to these specific (globally-remapped) cluster IDs.
    include_noise:
        When *False* (default) discard cluster IDs 0 (noise) and 1 (MUA).
    sampling_rate:
        Recording sample rate in Hz.  Required when *as_seconds=True*.
        Read from the parameter file automatically when *None*.
    as_seconds:
        Convert spike sample indices to seconds.

    Returns
    -------
    res : np.ndarray, shape (n_spikes,)
        Spike times in samples (or seconds), sorted ascending.
    clu : np.ndarray, shape (n_spikes,)
        Globally-remapped cluster IDs (unique across shanks).
    shank_map : np.ndarray, shape (n_unique_clusters, 2)
        Columns: ``[global_cluster_id, shank_index]``.
    """
    file_base = Path(str(file_base))

    # ── Resolve shank list ─────────────────────────────────────────────── #
    if shank_groups is None:
        yaml_path = file_base.with_suffix(".yaml")
        if yaml_path.exists():
            par = _load_par(str(yaml_path))
            grps = par.spikeDetection.channelGroups if par.spikeDetection else None
            shank_groups = list(range(1, len(grps) + 1)) if grps else None
        if shank_groups is None:
            found = sorted(
                int(p.suffix.lstrip("."))
                for p in file_base.parent.glob(f"{file_base.name}.res.*")
                if p.suffix.lstrip(".").isdigit()
            )
            shank_groups = found if found else [1]

    # ── Resolve sampling rate ──────────────────────────────────────────── #
    if as_seconds and sampling_rate is None:
        try:
            par_obj = _load_par(str(file_base))
            sampling_rate = float(par_obj.acquisitionSystem.samplingRate)
        except Exception:
            raise ValueError(
                "as_seconds=True requires sampling_rate or a readable .yaml file."
            )

    # ── Load shank by shank ────────────────────────────────────────────── #
    all_res:  list[np.ndarray] = []
    all_clu:  list[np.ndarray] = []
    all_map:  list[np.ndarray] = []
    max_clu:  int = 0

    for shank in shank_groups:
        clu_path = Path(f"{file_base}.clu.{shank}")
        res_path = Path(f"{file_base}.res.{shank}")

        if not clu_path.exists() or not res_path.exists():
            continue

        fclu = _read_clu(clu_path)
        fres = _read_res(res_path)

        n = min(len(fclu), len(fres))
        fclu, fres = fclu[:n], fres[:n]
        if n == 0:
            continue

        if not include_noise:
            keep = fclu > 1
            if not keep.any():
                continue
            fclu = fclu[keep]
            fres = fres[keep]

        fclu = fclu + max_clu
        unique_clu = np.unique(fclu)
        max_clu = int(unique_clu.max()) + 1

        all_res.append(fres)
        all_clu.append(fclu)
        all_map.append(
            np.column_stack([unique_clu, np.full(len(unique_clu), shank)])
        )

    if not all_res:
        empty = np.array([], dtype=np.int64)
        return empty, empty.astype(np.int32), np.empty((0, 2), dtype=np.int64)

    res = np.concatenate(all_res)
    clu = np.concatenate(all_clu)
    shank_map = np.concatenate(all_map).astype(np.int64)

    order = np.argsort(res, kind="stable")
    res   = res[order]
    clu   = clu[order]

    if clusters is not None:
        keep = np.isin(clu, clusters)
        res  = res[keep]
        clu  = clu[keep]

    if as_seconds:
        res = res.astype(np.float64) / sampling_rate

    return res, clu, shank_map


def spikes_by_unit(
    res: np.ndarray,
    clu: np.ndarray,
    sampling_rate: float | None = None,
    as_seconds: bool = False,
) -> dict[int, np.ndarray]:
    """Split concatenated spike arrays into a per-unit dict.

    Returns
    -------
    spikes : dict[int, np.ndarray]
        ``cluster_id`` → sorted 1-D array of spike times.
    """
    if as_seconds and sampling_rate is not None:
        res = res.astype(np.float64) / sampling_rate
    return {int(uid): res[clu == uid] for uid in np.unique(clu)}
