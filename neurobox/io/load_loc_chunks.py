"""
load_loc_chunks.py
==================
Readers for the neurosuite-3 small per-shank formats:

* ``.loc.N``    — binary per-spike source locations (5 float32 per spike)
* ``.chunks.N`` — text KiloKlustaKwik chunk boundaries (one per line)

See the spec files at:
* ``doc/ndmanager-plugins/formats/loc.md``
* ``doc/ndmanager-plugins/formats/chunks.md``
in the neurosuite-3 repository.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


__all__ = ["load_loc", "load_chunks"]


# Column meanings for a .loc row.  Kept as a module constant so
# callers and downstream code have a single source of truth.
LOC_COLUMNS: tuple[str, ...] = ("x_s", "y_s", "z_s", "A", "residual")


def load_loc(loc_file: str | Path) -> np.ndarray:
    """Load a ``.loc.N`` per-spike source-location file.

    Binary: no header, one row per spike, 5 ``float32`` values per
    row: ``x_s``, ``y_s``, ``z_s``, ``A``, ``residual``.
    File size = ``n_spikes × 20`` bytes.

    Parameters
    ----------
    loc_file:
        Path to the ``.loc.N`` file.

    Returns
    -------
    np.ndarray
        Shape ``(n_spikes, 5)``, dtype ``float32``.  Columns are
        described by :data:`LOC_COLUMNS`.

    Raises
    ------
    FileNotFoundError
        When *loc_file* does not exist.
    ValueError
        When the file size isn't a multiple of the row size (20 bytes).
    """
    loc_file = Path(loc_file)
    if not loc_file.exists():
        raise FileNotFoundError(f"Location file not found: {loc_file}")
    file_size = loc_file.stat().st_size
    row_size  = 5 * 4     # 5 × float32
    if file_size % row_size != 0:
        raise ValueError(
            f"{loc_file.name}: file size {file_size} bytes is not a "
            f"multiple of row size {row_size} (5 × float32)."
        )
    n_spikes = file_size // row_size
    return np.fromfile(
        str(loc_file), dtype="<f4",
    ).reshape(n_spikes, 5)


def load_chunks(chunks_file: str | Path) -> np.ndarray:
    """Load a ``.chunks.N`` chunk-boundary file.

    Text: one chunk per line, ``start_sample end_sample`` (
    whitespace-separated).  Blank lines and lines beginning with
    ``#`` are ignored.

    Parameters
    ----------
    chunks_file:
        Path to the ``.chunks.N`` text file.

    Returns
    -------
    np.ndarray
        Shape ``(n_chunks, 2)``, dtype ``int64``.  Column 0 is
        ``start_sample``, column 1 is ``end_sample``.  Empty file
        yields an empty ``(0, 2)`` array.

    Raises
    ------
    FileNotFoundError
        When *chunks_file* does not exist.
    ValueError
        When any non-comment, non-blank line does not have exactly
        two integer fields.
    """
    chunks_file = Path(chunks_file)
    if not chunks_file.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_file}")

    rows: list[tuple[int, int]] = []
    with open(chunks_file, "r") as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(
                    f"{chunks_file.name}:{lineno}: expected exactly two "
                    f"integer fields, got {len(parts)} ({line!r})."
                )
            try:
                start, stop = int(parts[0]), int(parts[1])
            except ValueError as exc:
                raise ValueError(
                    f"{chunks_file.name}:{lineno}: non-integer field in "
                    f"line {line!r}"
                ) from exc
            rows.append((start, stop))

    if not rows:
        return np.empty((0, 2), dtype=np.int64)
    return np.asarray(rows, dtype=np.int64)
