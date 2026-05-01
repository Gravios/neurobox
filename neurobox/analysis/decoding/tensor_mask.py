"""
neurobox.analysis.decoding.tensor_mask
=======================================
Maze-shape masks for restricting Bayesian decoding to plausible spatial
bins.  Port of :file:`MTA/analysis/create_tensor_mask.m`.

The MATLAB original has cases for ``circular``, ``square``, and
``line`` mazes (plus stubs for ``circular_bhv`` and ``HB`` that were
never implemented).  This port covers the three live cases.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class CircularBoundary:
    """Circular maze boundary.

    Attributes
    ----------
    radius:
        Maze radius in mm.  Bins at distance ≥ radius from centre are
        masked out.
    """
    radius: float
    shape:  str = "circular"


@dataclass
class SquareBoundary:
    """Square maze boundary.

    Attributes
    ----------
    edge_length:
        Length of each side in mm.  Bins outside ``±edge_length/2`` on
        either axis are masked out.
    """
    edge_length: float
    shape:       str = "square"


@dataclass
class LineBoundary:
    """Linear-track maze boundary.

    The track extends along the first axis to ``±edge_length/2`` and
    is constrained to ``±200 mm`` along the second axis (matches the
    MATLAB hard-coded value for the ``lin`` maze).
    """
    edge_length: float
    shape:       str = "line"


Boundary = CircularBoundary | SquareBoundary | LineBoundary


def create_tensor_mask(
    bin_centres: Sequence[np.ndarray],
    boundary:    Boundary | None = None,
) -> np.ndarray:
    """Build a boolean mask over rate-map bins.

    Port of :file:`MTA/analysis/create_tensor_mask.m`.

    Parameters
    ----------
    bin_centres:
        Per-axis bin-centre arrays (typically ``pf.bin_centres`` from
        a :class:`PlaceFieldResult`).  At least 2-D.
    boundary:
        One of :class:`CircularBoundary`, :class:`SquareBoundary`,
        :class:`LineBoundary`.  Default ``CircularBoundary(radius=440)``,
        matching the MATLAB default.

    Returns
    -------
    mask : np.ndarray of bool, shape ``(len(bin_centres[0]), len(bin_centres[1]))``
        True where the bin is inside the maze.
    """
    if boundary is None:
        boundary = CircularBoundary(radius=440.0)

    if len(bin_centres) < 2:
        raise ValueError(
            "create_tensor_mask: need at least 2 axes of bin centres."
        )

    bx = np.asarray(bin_centres[0])
    by = np.asarray(bin_centres[1])
    nx, ny = len(bx), len(by)

    if isinstance(boundary, CircularBoundary):
        # MATLAB uses index-space (pixel) distance with a half-pixel
        # offset.  The radius in index units is computed by counting
        # how many bins from the centre lie within ±radius of zero.
        # We replicate the index-space construction so the mask shape
        # exactly matches the MATLAB output.
        center_w = nx / 2.0
        center_h = ny / 2.0
        # Number of bins from centre to boundary
        # MATLAB: round(numel(bins{1})/2) - find(bins{1} < -radius, 1, 'last')
        outside_left = np.where(bx < -boundary.radius)[0]
        if outside_left.size > 0:
            radius_idx = round(nx / 2) - (outside_left[-1] + 1)  # +1: MATLAB→Python
        else:
            radius_idx = nx / 2

        # MATLAB meshgrid(1:width, 1:height) → shape (height, width)
        # but neurobox bin convention is (nx, ny), so we transpose at the end.
        W, H = np.meshgrid(np.arange(1, nx + 1),
                           np.arange(1, ny + 1), indexing="xy")
        dist = np.sqrt((W - center_w - 0.5)**2 + (H - center_h - 0.5)**2)
        mask = dist < radius_idx
        # Convert (height, width) → (nx, ny)
        return mask.T

    if isinstance(boundary, SquareBoundary):
        x_ok = np.abs(bx) <= boundary.edge_length / 2.0
        y_ok = np.abs(by) <= boundary.edge_length / 2.0
        return np.outer(x_ok, y_ok)

    if isinstance(boundary, LineBoundary):
        x_ok = np.abs(bx) <= boundary.edge_length / 2.0
        y_ok = np.abs(by) <= 200.0       # MATLAB hard-coded
        return np.outer(x_ok, y_ok)

    raise TypeError(f"Unknown boundary type: {type(boundary).__name__}")
