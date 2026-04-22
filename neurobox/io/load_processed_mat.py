"""
load_processed_mat.py
=====================
Load MTA-format processed motion-capture ``.mat`` files.

These files are produced by MTA's ``processC3D`` / ``parse_rbo_from_csv``
pipeline and live in::

    /data/processed/mocap/<srcId>/.../<session>/<maze>/

Each ``.mat`` file represents one recording trial block and contains:

  ``xyzpos``     float64 (T, N_markers, 3) — position in mm
  ``markers``    cell array of marker name strings
  ``sampleRate`` double — tracking frame rate in Hz

The public entry point :func:`load_processed_mat` loads a single file.
:func:`concatenate_processed_mat` mirrors MTA ``concatenate_vicon_files``:
it scans a directory for trial blocks, loads them in order, and returns a
list of per-trial chunk arrays.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Single-file loader
# ---------------------------------------------------------------------------

def load_processed_mat(
    mat_path: str | Path,
) -> tuple[np.ndarray, list[str], float]:
    """Load one MTA-processed mocap ``.mat`` file.

    Parameters
    ----------
    mat_path:
        Path to the ``.mat`` file.

    Returns
    -------
    xyzpos : np.ndarray, shape ``(T, N_markers, 3)``, float64
        Position data in mm.
    markers : list[str]
        Ordered marker names.
    samplerate : float
        Tracking frame rate in Hz.
    """
    mat_path = Path(mat_path)
    if not mat_path.exists():
        raise FileNotFoundError(f"MAT file not found: {mat_path}")

    try:
        import scipy.io as sio
    except ImportError as exc:
        raise ImportError(
            "scipy is required to load .mat files: pip install scipy"
        ) from exc

    ds = sio.loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)

    # ── Position data ─────────────────────────────────────────────── #
    if "xyzpos" not in ds:
        raise KeyError(
            f"'xyzpos' not found in {mat_path.name}.  "
            f"Keys: {[k for k in ds if not k.startswith('_')]}"
        )
    xyzpos = np.asarray(ds["xyzpos"], dtype=np.float64)

    # Flat (T, 3*N) → (T, N, 3)
    if xyzpos.ndim == 2 and xyzpos.shape[1] % 3 == 0:
        xyzpos = xyzpos.reshape(xyzpos.shape[0], xyzpos.shape[1] // 3, 3)

    # ── Marker names ──────────────────────────────────────────────── #
    raw = ds.get("markers", None)
    if raw is not None:
        markers = [str(m).strip() for m in np.asarray(raw).flat]
    else:
        markers = [f"marker_{i}" for i in range(xyzpos.shape[1])]

    # ── Sample rate ───────────────────────────────────────────────── #
    sr_raw = ds.get("sampleRate", ds.get("sample_rate", None))
    samplerate = float(np.squeeze(sr_raw)) if sr_raw is not None else 120.0

    return xyzpos, markers, samplerate


# ---------------------------------------------------------------------------
# Directory-level loader  (port of MTA concatenate_vicon_files)
# ---------------------------------------------------------------------------

_TRIAL_PATS: list[re.Pattern] = [
    re.compile(r'[Tt]rial\s*(\d+)', re.IGNORECASE),
    re.compile(r'[Tt]ake\s*(\d+)',  re.IGNORECASE),
    re.compile(r'_(\d+)\.mat$',     re.IGNORECASE),
]


def _sort_key(p: Path) -> tuple[int, str]:
    for pat in _TRIAL_PATS:
        m = pat.search(p.name)
        if m:
            return (int(m.group(1)), p.name)
    return (0, p.name)


def concatenate_processed_mat(
    directory: str | Path,
    glob_pattern: str = "*.mat",
    expected_markers: list[str] | None = None,
) -> tuple[list[np.ndarray], list[str], float]:
    """Load all trial ``.mat`` files from *directory* in sorted order.

    Port of MTA ``concatenate_vicon_files``.

    Parameters
    ----------
    directory:
        Directory to search (``spath/maze/`` or
        ``processed/mocap/.../session/maze/``).
    glob_pattern:
        Glob pattern (default ``'*.mat'``).
    expected_markers:
        If provided, files with a different marker set are skipped
        with a warning rather than raising an error.

    Returns
    -------
    chunks : list[np.ndarray]
        Per-trial ``(T_i, N_markers, 3)`` float64 arrays in order.
    markers : list[str]
        Marker names from the first successfully loaded file.
    samplerate : float
        Tracking frame rate in Hz.
    """
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    mat_files = sorted(directory.glob(glob_pattern), key=_sort_key)
    # Exclude files that look like aggregate outputs, not trial blocks
    mat_files = [
        f for f in mat_files
        if any(pat.search(f.name) for pat in _TRIAL_PATS)
        or glob_pattern != "*.mat"  # if caller gave a specific pattern, trust it
    ]
    if not mat_files:
        # Fallback: accept any .mat file if no trial-pattern match
        mat_files = sorted(directory.glob(glob_pattern), key=_sort_key)

    if not mat_files:
        raise FileNotFoundError(
            f"No .mat files matching {glob_pattern!r} found in {directory}"
        )

    chunks:    list[np.ndarray] = []
    markers:   list[str]        = []
    samplerate: float           = 0.0

    for mat in mat_files:
        try:
            xyz, mkrs, sr = load_processed_mat(mat)
        except Exception as exc:
            print(f"  [warn] {mat.name}: {exc}")
            continue

        if expected_markers and sorted(mkrs) != sorted(expected_markers):
            print(f"  [warn] {mat.name}: markers {mkrs} != expected "
                  f"{expected_markers} — skipping")
            continue

        if not markers:
            markers    = mkrs
            samplerate = sr
        elif mkrs != markers:
            print(f"  [warn] {mat.name}: marker list changed "
                  f"({mkrs} vs {markers}) — skipping")
            continue

        chunks.append(xyz)

    if not chunks:
        raise RuntimeError(
            f"No usable .mat files loaded from {directory}"
        )
    return chunks, markers, samplerate
