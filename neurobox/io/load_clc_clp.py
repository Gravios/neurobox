"""
load_clc_clp.py
================
Load the hierarchical-clustering sibling files ``.clc`` and ``.clp``
produced by neurosuite-3 Klusters.

A hierarchical clustering session for shank *N* is the triple:

* ``.clu.<method>.N`` — **fiber (parent) layer**: per-spike
  assignment to assembled units.  Same shape as any flat ``.clu``.
* ``.clc.<method>.N`` — **atom (child) layer**: per-spike
  assignment to the finest over-split micro-cluster.  Same shape
  as ``.clu``.
* ``.clp.<method>.N`` — **atom→fiber map**: cluster-ID-indexed
  (not spike-indexed) — one ``int32`` parent-fiber ID per atom.

All three files share the same on-disk framing (little-endian
int32 header followed by int32 payload); ``.clc`` reuses the
same reader as ``.clu`` (:func:`neurobox.io.load_clu_res._read_clu`).

Under the neurosuite-3 *variant naming convention*, all three are
**MethodSpecific** and carry the same ``<method>`` tag — so a
stderiv session has the triple
``.clu.stderiv.N`` + ``.clc.stderiv.N`` + ``.clp.stderiv.N``.
Legacy untagged names (``.clu.N``, ``.clc.N``, ``.clp.N``) are
still recognised on read.

Nesting invariant: every atom belongs to *exactly one* fiber — so
for the aligned per-spike ``.clu`` and ``.clc`` arrays, all spikes
carrying a given atom ID must carry the same fiber ID.  On save,
Klusters rewrites ``.clc`` and ``.clp`` together to keep them
consistent with ``.clu``.

Cluster ID conventions (fiber and atom layers alike):

* ``0`` → noise / artefact
* ``1`` → unsorted MUA
* ``≥ 2`` → real clusters (single units for fibers, over-split
  micro-clusters for atoms).  Atom IDs are *global* across the
  group (``1..nAtoms``), not per-fiber.

References:
* ``doc/ndmanager-plugins/formats/naming.md``
* ``doc/ndmanager-plugins/formats/clu.md``
* ``doc/ndmanager-plugins/formats/clc.md``
* ``doc/ndmanager-plugins/formats/clp.md``
* ``doc/klusters/hierarchical-clustering.md``
  (all in the ``neurosuite-3`` repo).
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np


# Re-export the low-level reader from load_clu_res so callers can
# do e.g. `from neurobox.io import _read_clc`.  The file framing is
# identical to `.clu`.
from neurobox.io.load_clu_res import _read_clu as _read_clc


def load_clc(clc_file: str | Path) -> np.ndarray:
    """Load the atom (child-layer) cluster assignments from ``.clc``.

    Returns an ``(n_spikes,)`` int32 array of atom IDs aligned with
    the paired ``.res`` and ``.clu`` files.

    Parameters
    ----------
    clc_file:
        Path to the ``.clc.<method>.N`` (or legacy ``.clc.N``) file.

    Returns
    -------
    np.ndarray
        Shape ``(n_spikes,)``, dtype int32.  Atom IDs (0 = noise,
        1 = unsorted, ≥ 2 = atoms).

    Raises
    ------
    FileNotFoundError
        When *clc_file* does not exist.
    """
    clc_file = Path(clc_file)
    if not clc_file.exists():
        raise FileNotFoundError(f"Child-layer file not found: {clc_file}")
    return _read_clc(clc_file)


class ClpMap(NamedTuple):
    """Atom→fiber map loaded from ``.clp``.

    Attributes
    ----------
    parent_of:
        Shape ``(n_atoms,)``, dtype int32.  ``parent_of[i - 1]`` is
        the parent fiber ID of atom ID ``i`` (1-based atom IDs, per
        the file spec).  A value of ``0`` means the atom is
        noise / unmapped.
    header:
        Raw int32 header value — usually the highest atom ID
        written.  Preserved for round-tripping.
    """
    parent_of: np.ndarray
    header:    int


def load_clp(clp_file: str | Path) -> ClpMap:
    """Load the atom→fiber (child→parent) map from ``.clp``.

    Parameters
    ----------
    clp_file:
        Path to the ``.clp.<method>.N`` (or legacy ``.clp.N``) file.

    Returns
    -------
    ClpMap

    Raises
    ------
    FileNotFoundError
        When *clp_file* does not exist.
    ValueError
        When the file is smaller than the 4-byte header.
    """
    clp_file = Path(clp_file)
    if not clp_file.exists():
        raise FileNotFoundError(f"Linkage file not found: {clp_file}")

    file_size = clp_file.stat().st_size
    if file_size < 4:
        raise ValueError(
            f"{clp_file.name}: too small ({file_size} bytes) — "
            "expected at least a 4-byte int32 header."
        )
    if (file_size - 4) % 4 != 0:
        raise ValueError(
            f"{clp_file.name}: payload ({file_size - 4} bytes) is not "
            "a whole number of int32 entries."
        )

    raw = np.fromfile(str(clp_file), dtype="<i4")
    header    = int(raw[0])
    parent_of = raw[1:]
    return ClpMap(parent_of=parent_of, header=header)


def build_atom_to_fiber(
    clp_map: ClpMap,
    include_zero: bool = False,
) -> dict[int, int]:
    """Convert a :class:`ClpMap` into an atom-ID → fiber-ID dict.

    Parameters
    ----------
    clp_map:
        As returned by :func:`load_clp`.
    include_zero:
        When *False* (default) atoms mapped to fiber ID ``0``
        (noise / unmapped) are omitted from the returned dict.

    Returns
    -------
    dict[int, int]
        ``{atom_id: fiber_id}`` for every atom in the file.
        Atom IDs are 1-based (as in the file layout).
    """
    out: dict[int, int] = {}
    for idx, parent in enumerate(clp_map.parent_of):
        parent = int(parent)
        if not include_zero and parent == 0:
            continue
        # Atom IDs are 1-based per the spec
        out[idx + 1] = parent
    return out


def build_fiber_to_atoms(
    clp_map:      ClpMap,
    include_zero: bool = False,
) -> dict[int, list[int]]:
    """Invert a :class:`ClpMap` into a fiber-ID → list-of-atoms dict.

    Parameters
    ----------
    clp_map:
        As returned by :func:`load_clp`.
    include_zero:
        When *False* (default) atoms mapped to fiber ID ``0`` are
        excluded.

    Returns
    -------
    dict[int, list[int]]
        ``{fiber_id: [atom_id, ...]}``, atom IDs in ascending
        order per fiber.
    """
    out: dict[int, list[int]] = {}
    for idx, parent in enumerate(clp_map.parent_of):
        parent = int(parent)
        if not include_zero and parent == 0:
            continue
        out.setdefault(parent, []).append(idx + 1)
    return out
