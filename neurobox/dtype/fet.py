"""
fet.py  —  NBDfet
==================
Lightweight container for derived features computed from NBDxyz / NBSpk /
NBDlfp.  Port of MTA's ``MTADfet`` minus the file-IO / persistence
machinery.

Responsibilities
----------------
* Hold a ``(T, n_features)`` array with named columns.
* Carry a samplerate (so resampling integrates with the rest of NBData).
* Optional human-readable column titles + descriptions for plot labels
  and reports.

The class is deliberately thin — it does not own any computation.
Feature functions (``fet_xy``, ``fet_dxy``, …) construct an NBDfet and
return it; the alignment and analysis machinery operates on NBDfet
objects directly.

Examples
--------
>>> fet = NBDfet(
...     data       = np.column_stack([x, y, head_yaw]),
...     columns    = ['x', 'y', 'head_yaw'],
...     samplerate = 120.0,
...     label      = 'fet_dxy',
... )
>>> fet.shape
(N, 3)
>>> fet['head_yaw']               # 1-D view
>>> fet.sel(['x', 'y'])           # (N, 2) sub-feature
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from neurobox.dtype.data import NBData
from neurobox.dtype.epoch import NBEpoch


class NBDfet(NBData):
    """Named-column feature container.

    Parameters
    ----------
    data:
        ``(T, n_features)`` array.  Trailing dims OK but rare.
    columns:
        Optional list of column names, one per feature.  Default is
        positional indices as strings.
    samplerate:
        Samples per second.
    label:
        Short feature-set identifier (e.g. ``'fet_dxy'``); used by
        :func:`map_to_reference_session` as the schema lookup key.
    name:
        Human-readable name (e.g. ``'Head direction + xy'``).
    titles, descriptions:
        Optional per-column text for plotting; lists the same length
        as *columns*.

    Attributes
    ----------
    columns : list[str]
    titles, descriptions : list[str]
    """

    def __init__(
        self,
        data:        np.ndarray | None = None,
        columns:     Sequence[str] | None = None,
        samplerate:  float = 1.0,
        sync:        NBEpoch | None = None,
        origin:      float = 0.0,
        label:       str = "",
        name:        str = "",
        titles:      Sequence[str] | None = None,
        descriptions: Sequence[str] | None = None,
        key:         str = "",
        path:        Path | str | None = None,
        filename:    str | None = None,
    ) -> None:
        super().__init__(
            path       = path,
            filename   = filename,
            data       = data,
            samplerate = samplerate,
            sync       = sync,
            origin     = origin,
            type_      = "TimeSeries",
            ext        = "fet",
            name       = name,
            label      = label,
            key        = key,
        )
        if data is not None and columns is None:
            n_cols = (
                self._data.shape[1] if self._data is not None and self._data.ndim >= 2
                else 1
            )
            columns = [str(i) for i in range(n_cols)]
        self.columns: list[str] = list(columns) if columns is not None else []
        self.titles:       list[str] = list(titles)       if titles else []
        self.descriptions: list[str] = list(descriptions) if descriptions else []

        # Validate
        if data is not None and self._data is not None and self._data.ndim >= 2:
            n = self._data.shape[1]
            if len(self.columns) != n:
                raise ValueError(
                    f"columns has {len(self.columns)} entries but data has "
                    f"{n} columns"
                )
            if self.titles and len(self.titles) != n:
                raise ValueError(
                    f"titles has {len(self.titles)} entries but data has "
                    f"{n} columns"
                )
            if self.descriptions and len(self.descriptions) != n:
                raise ValueError(
                    f"descriptions has {len(self.descriptions)} entries but "
                    f"data has {n} columns"
                )

    # ── NBData abstract methods ─────────────────────────────────────── #

    def load(self, *args, **kwargs) -> "NBDfet":
        if self._data is None:
            raise NotImplementedError(
                "NBDfet has no on-disk format yet.  Construct from a "
                "computed array."
            )
        return self

    def create(self, *args, **kwargs) -> "NBDfet":
        raise NotImplementedError(
            "NBDfet.create() is not used; construct directly from data."
        )

    # ── Column lookup ───────────────────────────────────────────────── #

    def index(self, name: str) -> int:
        """Return the column index for *name*."""
        try:
            return self.columns.index(name)
        except ValueError:
            raise KeyError(
                f"Feature {name!r} not in NBDfet.  Available: {self.columns}"
            ) from None

    def indices(self, names: Sequence[str]) -> list[int]:
        return [self.index(n) for n in names]

    def sel(self, columns: str | Sequence[str] | None = None) -> np.ndarray:
        """Select one or more columns by name; return the underlying array."""
        if self._data is None:
            raise RuntimeError("NBDfet data not loaded")
        if columns is None:
            return self._data
        if isinstance(columns, str):
            return self._data[:, self.index(columns)]
        idx = self.indices(list(columns))
        return self._data[:, idx]

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.sel(key)
        if isinstance(key, list) and key and isinstance(key[0], str):
            return self.sel(key)
        return super().__getitem__(key)

    @property
    def n_features(self) -> int:
        if self._data is None or self._data.ndim < 2:
            return 0
        return int(self._data.shape[1])

    def __repr__(self) -> str:
        shape = self.shape if self._data is not None else "not loaded"
        return (
            f"NBDfet(label={self.label!r}, columns={self.columns}, "
            f"shape={shape}, samplerate={self.samplerate})"
        )
