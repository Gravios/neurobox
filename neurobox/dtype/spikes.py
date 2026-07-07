"""
spikes.py  —  NBSpk
====================
Port of MTASpk.

Design differences from MTASpk
--------------------------------
* Spike times (``res``) stored in **seconds** (float64), not sample
  indices.  ``samplerate`` is preserved for back-compat / writing.
* ``__getitem__`` supports both cluster-ID selection and
  epoch-restricted selection:  ``spk[clu_ids]``,
  ``spk[clu_ids, epoch]``.
* ``restrict(epoch)`` returns a new NBSpk limited to spikes within
  the given epoch's periods.
* ``by_unit()`` → ``dict[int → np.ndarray]`` (same as
  ``spikes_by_unit`` in neurobox.io).
* ``load()`` delegates directly to ``neurobox.io.load_clu_res``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from neurobox.dtype.epoch import NBEpoch


class NBSpk:
    """Container for spike times and cluster assignments.

    Attributes
    ----------
    res : np.ndarray, shape (n_spikes,), float64
        Spike times in **seconds**, sorted ascending.
    clu : np.ndarray, shape (n_spikes,), int32
        Globally-remapped cluster ID for each spike.
    map : np.ndarray, shape (n_units, 2), int64
        Columns: ``[global_cluster_id, shank_index]``.
    samplerate : float
        Original recording sample rate (used when writing back to
        binary or converting to sample indices).
    fet : np.ndarray or None
        Spike waveform features, shape ``(n_spikes, n_features)``.
    spk : np.ndarray or None
        Spike waveform snippets, shape
        ``(n_spikes, n_samples, n_channels)``.
    type : str
        Always ``'TimePoints'``.
    """

    def __init__(
        self,
        res: np.ndarray | None = None,
        clu: np.ndarray | None = None,
        map_: np.ndarray | None = None,
        samplerate: float = 20000.0,
        fet: np.ndarray | None = None,
        spk: np.ndarray | None = None,
    ) -> None:
        self.res: np.ndarray        = (np.asarray(res, dtype=np.float64)
                                       if res is not None
                                       else np.array([], dtype=np.float64))
        self.clu: np.ndarray        = (np.asarray(clu, dtype=np.int32)
                                       if clu is not None
                                       else np.array([], dtype=np.int32))
        self.map: np.ndarray        = (np.asarray(map_, dtype=np.int64)
                                       if map_ is not None
                                       else np.empty((0, 3), dtype=np.int64))
        self.samplerate: float      = float(samplerate)
        self.fet: np.ndarray | None = fet
        self.spk: np.ndarray | None = spk
        self.type: str              = "TimePoints"
        self.annotations: list      = []  # list[UnitAnnotation] from YAML units block

        # Identity hash — mirrors NBData.update_hash.
        self.hash: str = ""
        self.update_hash()

    # ------------------------------------------------------------------ #
    # Hash mechanism (MATLAB MTASpk/update_hash.m)                         #
    # ------------------------------------------------------------------ #

    def update_hash(self, modification_hash: str | None = None) -> None:
        """Recompute ``self.hash`` from spike content + transform tag.

        Port of :file:`MTA/@MTASpk/update_hash.m`.

        Spike-train identity is captured by the bytes of *res*, *clu*
        and the cluster ID map *map*.  The samplerate is included so
        resampled / reindexed copies hash differently.
        """
        from neurobox.io.data_hash import data_hash
        self.hash = data_hash([
            self.res.tobytes() if self.res.size > 0 else None,
            self.clu.tobytes() if self.clu.size > 0 else None,
            self.map.tobytes() if self.map.size > 0 else None,
            self.samplerate,
            self.type,
            modification_hash,
        ])

    # ------------------------------------------------------------------ #
    # Representation                                                       #
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        n_spk  = len(self.res)
        n_unit = len(np.unique(self.clu)) if len(self.clu) else 0
        return f"NBSpk(n_spikes={n_spk}, n_units={n_unit}, sr={self.samplerate}Hz)"

    def __len__(self) -> int:
        return len(self.res)

    def isempty(self) -> bool:
        return len(self.res) == 0

    # ------------------------------------------------------------------ #
    # Indexing — spk[unit_ids] or spk[unit_ids, epoch]                   #
    # ------------------------------------------------------------------ #

    def __getitem__(self, idx):
        """Return spike times for the given cluster ID(s).

        Parameters
        ----------
        idx : int | array-like | tuple[ids, NBEpoch]
            ``spk[3]``         → times for cluster 3.
            ``spk[[3, 7]]``    → times for clusters 3 and 7.
            ``spk[3, epoch]``  → times for cluster 3 within epoch.
            ``spk[[3,7], epoch]`` → times for clusters 3,7 within epoch.
        """
        if isinstance(idx, tuple):
            unit_ids, epoch = idx[0], idx[1]
        else:
            unit_ids, epoch = idx, None

        # Scalar → 1-element array
        if np.isscalar(unit_ids):
            unit_ids = [unit_ids]

        mask = np.isin(self.clu, unit_ids)
        times = self.res[mask]

        if epoch is not None:
            times = _restrict_times(times, epoch)

        return times

    # ------------------------------------------------------------------ #
    # Restrict to epoch                                                    #
    # ------------------------------------------------------------------ #

    def restrict(self, epoch: "NBEpoch") -> "NBSpk":
        """Return a new NBSpk with spikes limited to *epoch* periods."""
        if epoch.isempty():
            return NBSpk(samplerate=self.samplerate)
        periods = epoch._as_periods()
        keep    = _within_periods(self.res, periods)
        new_map = self.map.copy()
        # Retain only units still present after restriction
        remaining_units = np.unique(self.clu[keep])
        new_map = new_map[np.isin(new_map[:, 0], remaining_units)]
        return NBSpk(
            res        = self.res[keep],
            clu        = self.clu[keep],
            map_       = new_map,
            samplerate = self.samplerate,
            fet        = self.fet[keep]  if self.fet is not None else None,
            spk        = self.spk[keep]  if self.spk is not None else None,
        )

    # ------------------------------------------------------------------ #
    # Per-unit dict                                                        #
    # ------------------------------------------------------------------ #

    def by_unit(self) -> dict[int, np.ndarray]:
        """Return ``{cluster_id: spike_times}`` dict."""
        return {int(uid): self.res[self.clu == uid]
                for uid in np.unique(self.clu)}

    @property
    def unit_ids(self) -> np.ndarray:
        return np.unique(self.clu)

    @property
    def n_units(self) -> int:
        return len(self.unit_ids)

    # ------------------------------------------------------------------ #
    # Shank mapping helpers                                                #
    # ------------------------------------------------------------------ #

    def shank_for_unit(self, unit_id: int) -> int | None:
        """Return the shank index of a given unit, or None if not found."""
        row = self.map[self.map[:, 0] == unit_id]
        return int(row[0, 1]) if len(row) else None

    def units_on_shank(self, shank: int) -> np.ndarray:
        """Return all unit IDs on *shank*."""
        return self.map[self.map[:, 1] == shank, 0].astype(np.int32)

    # ------------------------------------------------------------------ #
    # Annotation helpers (YAML units: block)                               #
    # ------------------------------------------------------------------ #

    def annotation_for(self, unit_id: int) -> "UnitAnnotation | None":
        """Return the :class:`~neurobox.io.UnitAnnotation` for *unit_id*, or None.

        Uses the ``global_id`` set by :func:`~neurobox.io.map_annotations_to_global_ids`.
        If ``global_id`` was not resolved (e.g. par not available), falls back to
        matching by ``(shank, local_cluster)`` via ``spk.map``.
        """
        if not self.annotations:
            return None
        # Try global_id match first
        for ann in self.annotations:
            if ann.global_id is not None and ann.global_id == unit_id:
                return ann
        # Fallback: match via spk.map (group = shank, cluster = local id)
        row = self.map[self.map[:, 0] == unit_id]
        if len(row) == 0:
            return None
        shank, local_clu = int(row[0, 1]), int(row[0, 2])
        for ann in self.annotations:
            if ann.group == shank and ann.cluster == local_clu:
                return ann
        return None

    def annotated_unit_ids(
        self,
        quality: str | list[str] | None = None,
        cell_type: str | None = None,
        structure: str | None = None,
        require_global_id: bool = True,
    ) -> np.ndarray:
        """Return unit IDs whose annotation matches the given filters.

        Parameters
        ----------
        quality:
            One or more quality strings to accept (case-insensitive).
            ``None`` means accept any annotated unit regardless of quality.
            Common values: ``'good'``, ``'great'``, ``'excellent'``.
        cell_type:
            Filter to units whose ``cell_type`` contains this substring
            (case-insensitive).  E.g. ``'pyr'``, ``'int'``.
        structure:
            Filter to units whose ``structure`` contains this substring
            (case-insensitive).  E.g. ``'CA1'``.
        require_global_id:
            If True (default), only return units whose ``global_id`` was
            successfully mapped from the YAML to the spike data.

        Returns
        -------
        np.ndarray of int
            Sorted array of unit IDs that pass all filters.

        Examples
        --------
        >>> good_ids = spk.annotated_unit_ids(quality='good')
        >>> pyr_ids  = spk.annotated_unit_ids(cell_type='pyr')
        >>> ca1_good = spk.annotated_unit_ids(quality='good', structure='CA1')
        """
        if not self.annotations:
            return np.array([], dtype=np.int32)

        # Normalise quality to a frozenset
        if quality is not None:
            if isinstance(quality, str):
                quality_set = frozenset([quality.strip().lower()])
            else:
                quality_set = frozenset(q.strip().lower() for q in quality)
        else:
            quality_set = None

        ids = []
        for ann in self.annotations:
            if require_global_id and ann.global_id is None:
                continue

            if quality_set is not None:
                q = (ann.quality or "").strip().lower()
                if q not in quality_set:
                    continue

            if cell_type is not None:
                ct = (ann.cell_type or "").lower()
                if cell_type.lower() not in ct:
                    continue

            if structure is not None:
                st = (ann.structure or "").lower()
                if structure.lower() not in st:
                    continue

            if ann.global_id is not None:
                ids.append(ann.global_id)

        return np.unique(np.array(ids, dtype=np.int32))

    @classmethod
    def load(
        cls,
        file_base: str | Path,
        shank_groups: list[int] | None = None,
        include_noise: bool = False,
        include_waveforms: bool = False,
        samplerate: float | None = None,
    ) -> "NBSpk":
        """Load from neurosuite-3 binary .res/.clu files.

        Parameters
        ----------
        file_base:
            Session base path without extension.
        shank_groups:
            Shanks to load (None → all from .yaml or glob).
        include_noise:
            Include cluster IDs 0 (noise) and 1 (MUA).
        include_waveforms:
            Also load waveform snippets from .spk.N files.
        samplerate:
            Recording sample rate.  Read from .yaml if None.
        """
        from neurobox.io import load_clu_res, load_par

        # Load par once upfront so we can share it with waveform loading
        par = None
        if samplerate is None or include_waveforms:
            try:
                par = load_par(str(file_base))
            except Exception:
                pass
        if samplerate is None:
            samplerate = (float(par.acquisitionSystem.samplingRate)
                          if par is not None else 20000.0)

        res_arr, clu_arr, shank_map = load_clu_res(
            file_base,
            shank_groups   = shank_groups,
            include_noise  = include_noise,
            as_seconds     = True,
            sampling_rate  = samplerate,
        )

        spk_wf = None
        if include_waveforms:
            from neurobox.io import load_spk_from_par
            try:
                if par is None:
                    par = load_par(str(file_base))
                chunks = []
                for shk in (shank_groups or list(range(1, 9))):
                    try:
                        chunks.append(load_spk_from_par(file_base, shk, par))
                    except FileNotFoundError:
                        pass
                if chunks:
                    # Chunks are per-shank; concat naively (shape may differ
                    # across shanks if n_channels differs — pad with zeros)
                    max_samp = max(c.shape[1] for c in chunks)
                    max_chan = max(c.shape[2] for c in chunks)
                    padded   = []
                    for c in chunks:
                        p = np.zeros((c.shape[0], max_samp, max_chan), dtype=c.dtype)
                        p[:, :c.shape[1], :c.shape[2]] = c
                        padded.append(p)
                    spk_wf = np.concatenate(padded, axis=0)
            except Exception:
                pass

        obj = cls(
            res        = res_arr,
            clu        = clu_arr,
            map_       = shank_map,
            samplerate = samplerate,
            spk        = spk_wf,
        )

        # Load curation annotations from the units: block (if present)
        if par is not None:
            try:
                from neurobox.io.load_units import (
                    load_units, map_annotations_to_global_ids,
                )
                obj.annotations = load_units(par)
                if obj.annotations and len(shank_map):
                    map_annotations_to_global_ids(obj.annotations, shank_map)
            except Exception:
                pass

        return obj

    def clear(self) -> "NBSpk":
        """Free the in-memory spike arrays. Mirrors MTASpk.clear."""
        self.res = np.array([], dtype=np.float64)
        self.clu = np.array([], dtype=np.int32)
        self.fet = None
        self.spk = None
        return self

    def copy(self) -> "NBSpk":
        """Return a deep copy. Mirrors MTASpk.copy."""
        import copy as _copy
        return _copy.deepcopy(self)

    def save(
        self,
        file_base:  str | Path,
        method:     str  = "standard",
        overwrite:  bool = False,
    ) -> list[Path]:
        """Save this NBSpk back to neurosuite-3 ``.res`` / ``.clu`` files.

        Complement of :meth:`load`.  Splits the globally-remapped
        cluster IDs in :attr:`clu` back into per-shank local IDs
        using :attr:`map`, converts :attr:`res` from seconds back
        to sample indices, and writes one variant-tagged file pair
        per shank present in :attr:`map`.

        Parameters
        ----------
        file_base:
            Session base path without extension — e.g.
            ``/data/sirotaA-jg-05-20120316``.  Combined with the
            ``method`` tag and per-shank suffix to form the output
            path.
        method:
            Variant tag written into the filename
            (``<base>.res.<method>.<shank>`` and
            ``<base>.clu.<method>.<shank>``).  Defaults to
            ``'standard'`` — the neurosuite-3 canonical tag when
            no other variant is in play.
        overwrite:
            Passed through to the underlying writers.  When *False*
            (default), a :class:`FileExistsError` is raised if any
            output file already exists.

        Returns
        -------
        list[pathlib.Path]
            All files written, in shank-then-type order:
            ``[res.<method>.1, clu.<method>.1, res.<method>.2, …]``.

        Raises
        ------
        FileExistsError
            If any target file exists and ``overwrite=False``.
        ValueError
            If the internal state is inconsistent (e.g. ``res`` and
            ``clu`` different lengths, or ``map`` doesn't cover the
            cluster IDs present in ``clu``).
        """
        from neurobox.io.ns3_writers import save_res, save_clu

        file_base = Path(str(file_base))

        # Validate internal state.
        if self.res.size != self.clu.size:
            raise ValueError(
                f"NBSpk.save: res has {self.res.size} entries but "
                f"clu has {self.clu.size}"
            )
        if self.res.size == 0:
            return []
        if self.map.size == 0:
            raise ValueError(
                "NBSpk.save: cluster→shank map is empty; nothing to save.  "
                "This usually means the NBSpk was constructed directly "
                "rather than loaded from disk."
            )

        # Convert res (seconds) → sample indices.
        # Round-half-to-even so we don't accumulate a systematic bias
        # from repeated round-trips through the seconds representation.
        res_samples_all = np.round(
            self.res.astype(np.float64) * self.samplerate
        ).astype(np.int64)

        written: list[Path] = []

        # Sort by shank index (col 1 of map) so we produce a
        # deterministic output order regardless of insertion order.
        shanks = np.unique(self.map[:, 1])
        for shank in shanks:
            shank = int(shank)

            # Rows of the map for this shank — carries the
            # global→local mapping directly (col 2 = original local
            # ID; col 0 = global remap assigned by load_clu_res).
            shank_row_mask = self.map[:, 1] == shank
            shank_rows     = self.map[shank_row_mask]
            shank_globals  = shank_rows[:, 0]

            # Which spike-array entries belong to this shank's clusters?
            spike_mask = np.isin(self.clu, shank_globals)
            if not spike_mask.any():
                continue

            local_res = res_samples_all[spike_mask]
            local_clu_global = self.clu[spike_mask]

            # Inverse-map globals → local per-shank IDs using the
            # global→local mapping stored in col 2 of self.map.
            global_to_local = dict(
                zip(shank_globals.tolist(),
                    shank_rows[:, 2].tolist())
            )
            local_clu = np.fromiter(
                (global_to_local[int(g)] for g in local_clu_global),
                dtype=np.int32,
                count=int(spike_mask.sum()),
            )

            # NBSpk.load(as_seconds=True) sorts across all shanks by
            # time, so re-sort each shank slice by sample so on-disk
            # order is time-ascending (matches what load_clu_res
            # would write for a fresh sort).
            sort_ix = np.argsort(local_res, kind="stable")
            local_res = local_res[sort_ix]
            local_clu = local_clu[sort_ix]

            res_path = file_base.with_name(
                f"{file_base.name}.res.{method}.{shank}"
            )
            clu_path = file_base.with_name(
                f"{file_base.name}.clu.{method}.{shank}"
            )
            written.append(save_res(res_path, local_res, overwrite=overwrite))
            written.append(save_clu(clu_path, local_clu, overwrite=overwrite))

        return written

    def save_unit_set(self, session, name: str, unit_ids) -> None:
        """Save named unit subset to disk. Mirrors MTASpk.set_unit_set.

        Saves <spath>/<filebase>.unit_set.<name>.npy.
        """
        out = session.spath / f"{session.filebase}.unit_set.{name}.npy"
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(out), np.asarray(unit_ids, dtype=np.int32))

    def load_unit_set(self, session, name: str) -> "np.ndarray":
        """Load a named unit set. Mirrors MTASpk.get_unit_set."""
        src = session.spath / f"{session.filebase}.unit_set.{name}.npy"
        if not src.exists():
            raise FileNotFoundError(
                f"Unit set {name!r} not found at {src}. "
                "Create it with spk.save_unit_set(session, name, ids)."
            )
        return np.load(str(src))

    def create(self, *args, **kwargs) -> "NBSpk":
        """Alias for load()."""
        return NBSpk.load(*args, **kwargs)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _within_periods(times: np.ndarray, periods: np.ndarray) -> np.ndarray:
    """Boolean mask: True where times fall within any period.

    Requires *times* to be sorted ascending (guaranteed after
    ``load_clu_res``).  Uses ``np.searchsorted`` for
    O((N + M) log N) complexity instead of O(N × M).
    """
    mask = np.zeros(len(times), dtype=bool)
    for s, e in periods:
        i0 = int(np.searchsorted(times, s, side="left"))
        i1 = int(np.searchsorted(times, e, side="left"))
        if i1 > i0:
            mask[i0:i1] = True
    return mask


def _restrict_times(times: np.ndarray, epoch: "NBEpoch") -> np.ndarray:
    periods = epoch._as_periods()
    return times[_within_periods(times, periods)]
