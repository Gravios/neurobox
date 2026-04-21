"""
stc.py  —  NBStateCollection
=============================
Port of MTAStateCollection.

Stores a named collection of NBEpoch objects (behavioural states) and
provides a DSL for composing them.

Query language (same as MTA)
-----------------------------
``stc['walk']``          → the 'walk' epoch
``stc['walk&rear']``     → intersection of walk and rear
``stc['walk|rear']``     → union of walk and rear
``stc['walk^rear']``     → same as union (MTA convention)

Multiple operators are evaluated left-to-right with equal precedence.

Design differences from MTAStateCollection
-------------------------------------------
* States stored in a plain ``dict[str → NBEpoch]`` (both label and key
  map to the same NBEpoch object).
* ``__getitem__`` replaces the complex MATLAB subsref.
* ``filter()`` simplified to keyword arguments.
* Persistence via ``pickle`` instead of MATLAB .mat files.
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path

import numpy as np

from neurobox.dtype.epoch import NBEpoch, _intersect_periods, _union_periods


class NBStateCollection:
    """Named collection of behavioural state epochs.

    Parameters
    ----------
    path:
        Directory where the collection is saved/loaded.
    filename:
        File name for persistence (default: auto from filebase + mode).
    mode:
        Labelling mode string (e.g. 'manual', 'auto', 'hmm').
    sync:
        Recording-window NBEpoch shared by all states.
    """

    def __init__(
        self,
        path: Path | str | None = None,
        filename: str | None = None,
        mode: str = "manual",
        sync: NBEpoch | None = None,
    ) -> None:
        self.path:     Path | None     = Path(path) if path is not None else None
        self.filename: str | None      = filename
        self.mode:     str             = mode
        self.sync:     NBEpoch | None  = sync
        self._states:  dict[str, NBEpoch] = {}   # label → epoch
        self._keys:    dict[str, str]     = {}   # key   → label

    # ------------------------------------------------------------------ #
    # State management                                                     #
    # ------------------------------------------------------------------ #

    def add_state(
        self,
        epoch: NBEpoch,
        label: str | None = None,
        key: str | None = None,
    ) -> None:
        """Add or replace a state.

        Parameters
        ----------
        epoch:
            The NBEpoch object representing the state.
        label:
            Overrides ``epoch.label`` if provided.
        key:
            Overrides ``epoch.key`` if provided.
        """
        if label is not None:
            epoch.label = label
        if key is not None:
            epoch.key = key
        lbl = epoch.label
        k   = epoch.key
        if not lbl:
            raise ValueError("Epoch must have a non-empty label.")
        self._states[lbl] = epoch
        if k:
            self._keys[k] = lbl

    def remove_state(self, name: str) -> None:
        """Remove a state by label or key."""
        lbl = self._resolve_label(name)
        if lbl is None:
            return
        ep = self._states.pop(lbl, None)
        if ep and ep.key in self._keys:
            del self._keys[ep.key]

    def get_state(self, name: str) -> NBEpoch:
        """Retrieve a state by label or key."""
        lbl = self._resolve_label(name)
        if lbl is None:
            raise KeyError(f"State {name!r} not found. "
                           f"Available: {self.list_states()}")
        return self._states[lbl]

    def has_state(self, name: str) -> bool:
        return self._resolve_label(name) is not None

    def list_states(self) -> list[str]:
        """Return all state labels."""
        return list(self._states.keys())

    def _resolve_label(self, name: str) -> str | None:
        if name in self._states:
            return name
        if name in self._keys:
            return self._keys[name]
        return None

    # ------------------------------------------------------------------ #
    # Query interface — stc['walk&rear']                                  #
    # ------------------------------------------------------------------ #

    def __getitem__(self, query: str) -> NBEpoch:
        """Retrieve and compose states using the MTA query language.

        Operators (left-to-right, equal precedence):
          ``&``  intersection
          ``|``  union
          ``^``  union (MTA alternate)
          ``+``  union (MTA alternate)
          ``-``  difference (A minus B)

        Examples
        --------
        >>> stc['walk']
        >>> stc['walk&rear']
        >>> stc['walk|rear|grooming']
        >>> stc['walk-rear']
        """
        return self._parse_query(str(query))

    def _parse_query(self, query: str) -> NBEpoch:
        # Split on operators, keeping delimiters
        tokens = re.split(r'([&|^+\-])', query)
        names  = tokens[0::2]
        ops    = tokens[1::2]

        result = self.get_state(names[0].strip())
        for op, name in zip(ops, names[1:]):
            other = self.get_state(name.strip())
            if op in ('&',):
                result = result & other
            elif op in ('|', '^', '+'):
                result = result | other
            elif op == '-':
                result = result - other
            else:
                raise ValueError(f"Unknown operator: {op!r}")
        return result

    # ------------------------------------------------------------------ #
    # Explicit composition methods (complement to string DSL)             #
    # ------------------------------------------------------------------ #

    def intersect(self, *state_names: str) -> NBEpoch:
        """Return the intersection of the named states."""
        epochs = [self.get_state(n) for n in state_names]
        return NBEpoch.intersect(epochs)

    def union(self, *state_names: str) -> NBEpoch:
        """Return the union of the named states."""
        epochs = [self.get_state(n) for n in state_names]
        return NBEpoch.union(epochs)

    def difference(self, base: str, subtract: str) -> NBEpoch:
        """Return base_state minus subtract_state."""
        return self.get_state(base) - self.get_state(subtract)

    # ------------------------------------------------------------------ #
    # State filtering                                                      #
    # ------------------------------------------------------------------ #

    def filter(
        self,
        state: str,
        *,
        exclude: list[str] | None = None,
        min_duration_sec: float | None = None,
        trim_sec: float = 0.0,
        fill_gaps_sec: float | None = None,
    ) -> NBEpoch:
        """Return filtered epochs for a state.

        Parameters
        ----------
        state:
            Target state label or key.
        exclude:
            State names to exclude (proximity-based exclusion not yet
            implemented — currently does set difference).
        min_duration_sec:
            Drop periods shorter than this.
        trim_sec:
            Shrink each period by this amount on both ends.
        fill_gaps_sec:
            Merge gaps shorter than this (applied before filtering).
        """
        result = self.get_state(state).copy()

        if fill_gaps_sec is not None:
            result = result.fillgaps(fill_gaps_sec)

        if exclude:
            for ex_name in exclude:
                result = result - self.get_state(ex_name)

        if trim_sec > 0 and result.mode == "periods" and not result.isempty():
            result.data[:, 0] += trim_sec
            result.data[:, 1] -= trim_sec
            result.clean()

        if min_duration_sec is not None and result.mode == "periods":
            dur = result.data[:, 1] - result.data[:, 0]
            result.data = result.data[dur >= min_duration_sec]

        return result

    # ------------------------------------------------------------------ #
    # State transitions                                                    #
    # ------------------------------------------------------------------ #

    def get_transitions(
        self,
        from_state: str,
        to_state: str,
        window_sec: float = 0.2,
    ) -> np.ndarray:
        """Return (N, 2) periods centred on transitions from→to.

        Parameters
        ----------
        from_state, to_state:
            State labels or keys.
        window_sec:
            Half-window in seconds around each transition point.

        Returns
        -------
        periods : np.ndarray, shape (N, 2)
            Each row is [transition_time - window, transition_time + window].
        """
        ante = self.get_state(from_state)._as_periods()   # antecedent stops
        subq = self.get_state(to_state)._as_periods()     # subsequent starts

        transitions: list[float] = []
        j = 0
        for _, stop_a in ante:
            while j < len(subq) and subq[j, 0] <= stop_a:
                j += 1
            if j < len(subq):
                gap = subq[j, 0] - stop_a
                if 0 <= gap <= window_sec:
                    transitions.append(subq[j, 0])
                    j += 1

        if not transitions:
            return np.empty((0, 2), dtype=np.float64)
        t = np.array(transitions, dtype=np.float64)
        return np.column_stack([t - window_sec, t + window_sec])

    # ------------------------------------------------------------------ #
    # Resync / update                                                      #
    # ------------------------------------------------------------------ #

    def update_sync(self, sync: NBEpoch) -> None:
        """Update recording-window sync for all states."""
        self.sync = sync
        for ep in self._states.values():
            ep.sync = sync.data[:2] if sync is not None else None
            ep.clean()

    # ------------------------------------------------------------------ #
    # Persistence                                                          #
    # ------------------------------------------------------------------ #

    @property
    def fpath(self) -> Path | None:
        if self.path is None or self.filename is None:
            return None
        return self.path / self.filename

    def save(self, overwrite: bool = False) -> None:
        """Persist to pickle."""
        fp = self.fpath
        if fp is None:
            raise ValueError("path and filename must be set before saving.")
        if fp.exists() and not overwrite:
            raise FileExistsError(f"{fp} exists.  Pass overwrite=True to replace.")
        with open(fp, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load_file(cls, path: Path | str) -> "NBStateCollection":
        """Load a saved NBStateCollection from a pickle file."""
        with open(path, "rb") as f:
            return pickle.load(f)

    def isempty(self) -> bool:
        return len(self._states) == 0

    # ------------------------------------------------------------------ #
    # Repr                                                                 #
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        labels = self.list_states()
        return (f"NBStateCollection(mode={self.mode!r}, "
                f"states={labels})")

    def __len__(self) -> int:
        return len(self._states)
