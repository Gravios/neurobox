"""
paths.py  —  NBSessionPaths
============================
Path resolution for the neurobox data directory structure.

Directory grammar
-----------------
The hierarchy is organised by data lifecycle (source → processed →
project) and then by data type, lab, user, subject, and recording date:

  /data/
    source/<typeId>/<sourceId>/<sourceId>-<userId>/
           <sourceId>-<userId>-<subjectId>/
           <sourceId>-<userId>-<subjectId>-<date>/
               <raw acquisition files>

    processed/<typeId>/<sourceId>/<sourceId>-<userId>/
              <sourceId>-<userId>-<subjectId>/
              <sourceId>-<userId>-<subjectId>-<date>/
                  <standardised output files>
                  <maze>/        ← mocap only

    project/<projectId>/<sourceId>-<userId>-<subjectId>-<date>/
                <symlinks to processed files>
                <analysis outputs: .ses.pkl, .stc.*.pkl, .pos.npz, …>

Type IDs
--------
The ``typeId`` in both source and processed trees is the *standardised*
data type, not the acquisition system name:

  Raw acquisition       →  typeId (processed)
  ─────────────────────────────────────────────
  Neuralynx (.ncs)      →  nlx   (source)
  Open Ephys            →  ephys (source)
  Neuralynx / OE        →  ephys (processed)
  Vicon (.c3d)          →  mocap (source)
  Optitrack CSV         →  mocap (source)
  Vicon / Optitrack     →  mocap (processed)
  Video                 →  video (source only)

Session name decomposition
--------------------------
``session_name`` = ``<sourceId>-<userId>-<subjectId>-<date>``
e.g.             = ``sirotaA-jg-05-20120316``

Parsed right-to-left:
  date      last component, 8 digits  (``20120316``)
  subjectId penultimate, numeric      (``05``)
  userId    antepenultimate, alpha    (``jg``)
  sourceId  everything before that   (``sirotaA``)
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Session name parsing
# ---------------------------------------------------------------------------

_SESSION_RE = re.compile(
    r'^(?P<sourceId>.+?)-(?P<userId>[A-Za-z]+)-(?P<subjectId>\d+)'
    r'-(?P<date>\d{8}[a-zA-Z]*)$'
)


def parse_session_name(name: str) -> dict[str, str]:
    """Decompose a session name into its components.

    Parameters
    ----------
    name:
        Full session name, e.g. ``'sirotaA-jg-05-20120316'``.

    Returns
    -------
    dict with keys ``sourceId``, ``userId``, ``subjectId``, ``date``.

    Raises
    ------
    ValueError if the name does not match the expected pattern.
    """
    m = _SESSION_RE.match(name)
    if m is None:
        raise ValueError(
            f"Session name {name!r} does not match the expected pattern "
            f"'<sourceId>-<userId>-<subjectId>-<date>' "
            f"(e.g. 'sirotaA-jg-05-20120316')."
        )
    return m.groupdict()


def build_session_name(source_id: str, user_id: str,
                        subject_id: str | int, date: str) -> str:
    """Build a canonical session name from its components."""
    return f"{source_id}-{user_id}-{int(subject_id):02d}-{date}"


# ---------------------------------------------------------------------------
# Path builder
# ---------------------------------------------------------------------------

@dataclass
class NBSessionPaths:
    """All standard paths for one recording session.

    Parameters
    ----------
    session_name:
        Full session name, e.g. ``'sirotaA-jg-05-20120316'``.
    data_root:
        Root of the data tree (``NB_DATA_PATH``), e.g. ``Path('/data')``.
    project_id:
        Project identifier, e.g. ``'B01'``.
    maze:
        Maze / arena code for motion-capture lookups, e.g. ``'cof'``.
        Used to resolve processed mocap paths.

    Attributes (read-only after construction)
    -----------------------------------------
    source_id, user_id, subject_id, date:
        Components parsed from ``session_name``.
    spath:
        Project session directory  — where analysis outputs live and
        symlinks to processed data are created.
    """

    session_name: str
    data_root:    Path
    project_id:   str
    maze:         str = "cof"

    # populated in __post_init__
    source_id:    str = field(init=False, repr=False)
    user_id:      str = field(init=False, repr=False)
    subject_id:   str = field(init=False, repr=False)
    date:         str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.data_root = Path(self.data_root)
        parts = parse_session_name(self.session_name)
        self.source_id  = parts["sourceId"]
        self.user_id    = parts["userId"]
        self.subject_id = parts["subjectId"]
        self.date       = parts["date"]

    # ------------------------------------------------------------------ #
    # Internal path builder                                               #
    # ------------------------------------------------------------------ #

    def _hierarchy(self, base: Path, type_id: str) -> Path:
        """Return the 5-level hierarchy path for a given type under *base*."""
        u  = f"{self.source_id}-{self.user_id}"
        us = f"{u}-{self.subject_id}"
        return (base / type_id / self.source_id / u / us / self.session_name)

    # ------------------------------------------------------------------ #
    # Project (analysis) directory                                        #
    # ------------------------------------------------------------------ #

    @property
    def spath(self) -> Path:
        """Project session directory (real directory, not a symlink)."""
        return self.data_root / "project" / self.project_id / self.session_name

    # ------------------------------------------------------------------ #
    # Processed data directories                                          #
    # ------------------------------------------------------------------ #

    @property
    def processed_ephys(self) -> Path:
        """Processed ephys directory for this session."""
        return self._hierarchy(self.data_root / "processed", "ephys")

    @property
    def processed_mocap(self) -> Path:
        """Processed mocap directory for this session (maze-level)."""
        return self._hierarchy(self.data_root / "processed", "mocap") / self.maze

    # ------------------------------------------------------------------ #
    # Subject-ID padding variants                                         #
    # ------------------------------------------------------------------ #
    #
    # The MATLAB session-list convention uses compact subject IDs
    # (``jg05`` → subject "05"), but the Sirota lab's processed-data
    # tree uses zero-padded 6-digit forms (``sirotaA-jg-000005`` →
    # subject "000005").  These helpers let us look up data on disk
    # without baking either convention into the canonical session name.

    def subject_id_padded(self, width: int = 6) -> str:
        """``self.subject_id`` zero-padded to *width* digits."""
        return self.subject_id.zfill(width)

    def _hierarchy_for_subject(
        self,
        base:       Path,
        type_id:    str,
        subject_id: str,
    ) -> Path:
        """Like :meth:`_hierarchy` but with an explicit ``subject_id``.

        When ``subject_id`` differs from ``self.subject_id``, the leaf
        session-name component is rewritten to use it so the on-disk
        directory ``sirotaA-jg-000005-20120316/`` can be reached from
        a session whose canonical name is ``sirotaA-jg-05-20120316``.
        """
        u  = f"{self.source_id}-{self.user_id}"
        us = f"{u}-{subject_id}"
        if subject_id == self.subject_id:
            leaf = self.session_name
        else:
            leaf = f"{self.source_id}-{self.user_id}-{subject_id}-{self.date}"
        return base / type_id / self.source_id / u / us / leaf

    def _resolve_existing(
        self,
        type_id:      str,
        *,
        with_maze:    bool = False,
    ) -> Path:
        """Return the canonical path for *type_id* under ``processed/``
        if it exists; otherwise the 6-digit-padded variant if THAT
        exists; otherwise the canonical path (for sensible error
        messages).
        """
        base = self.data_root / "processed"
        canonical = self._hierarchy(base, type_id)
        if with_maze:
            canonical = canonical / self.maze
        if canonical.exists():
            return canonical
        padded_subj = self.subject_id_padded()
        if padded_subj != self.subject_id:
            padded = self._hierarchy_for_subject(base, type_id, padded_subj)
            if with_maze:
                padded = padded / self.maze
            if padded.exists():
                return padded
        return canonical

    def resolve_processed_ephys(self) -> Path:
        """Return the on-disk processed-ephys directory.

        Tries the canonical (compact subject-ID) path first, then the
        6-digit zero-padded variant.  Falls back to the canonical
        path if neither exists, so error messages still print the
        path the caller would expect.
        """
        return self._resolve_existing("ephys")

    def resolve_processed_mocap(self) -> Path:
        """Return the on-disk processed-mocap maze directory.

        Same canonical → padded resolution as
        :meth:`resolve_processed_ephys`.
        """
        return self._resolve_existing("mocap", with_maze=True)

    def resolve_processed_mocap_session(self) -> Path:
        """Return the on-disk processed-mocap *session* directory
        (parent of all maze subdirectories).

        Useful for maze-discovery — scan the returned directory's
        children to enumerate available mazes.  Same canonical →
        padded resolution as :meth:`resolve_processed_ephys`.
        """
        return self._resolve_existing("mocap")

    # ------------------------------------------------------------------ #
    # Source data directories                                             #
    # ------------------------------------------------------------------ #

    @property
    def source_nlx(self) -> Path:
        """Source Neuralynx directory."""
        return self._hierarchy(self.data_root / "source", "nlx")

    @property
    def source_mocap(self) -> Path:
        """Source motion-capture directory."""
        return self._hierarchy(self.data_root / "source", "mocap")

    @property
    def source_video(self) -> Path:
        """Source video directory."""
        return self._hierarchy(self.data_root / "source", "video")

    # ------------------------------------------------------------------ #
    # Common processed file paths                                         #
    # ------------------------------------------------------------------ #

    @property
    def yaml_file(self) -> Path:
        """Session parameter file (.yaml)."""
        return self.processed_ephys / f"{self.session_name}.yaml"

    @property
    def dat_file(self) -> Path:
        """Wideband binary (.dat)."""
        return self.processed_ephys / f"{self.session_name}.dat"

    @property
    def lfp_file(self) -> Path:
        """Downsampled LFP binary (.lfp)."""
        return self.processed_ephys / f"{self.session_name}.lfp"

    def res_file(self, shank: int) -> Path:
        """Spike timestamps for shank N — canonical path
        ``<base>.res.<shank>``.

        For backward compatibility with legacy (untagged) layouts.
        In neurosuite-3, ``.res`` is a **Shared** artifact — use
        :meth:`resolve_ns3_shared` to search variant-tagged copies
        (``<base>.res.<method>.<shank>``) with fall-back to this
        untagged path.
        """
        return self.processed_ephys / f"{self.session_name}.res.{shank}"

    def clu_file(self, shank: int) -> Path:
        """Cluster assignments (fiber layer, or flat) for shank N —
        canonical untagged path ``<base>.clu.<shank>``.

        For backward compatibility with legacy layouts.  In
        neurosuite-3, ``.clu`` is a **MethodSpecific** artifact and
        the canonical layout is ``<base>.clu.<method>.<shank>``;
        use :meth:`ns3_file` (``type='clu'``) or
        :meth:`resolve_ns3_method_specific` for the variant-tagged
        form.

        In a hierarchical session this file is the *fiber (parent)*
        layer; the sibling :meth:`clc_file` is the *atom (child)*
        layer and :meth:`clp_file` is the atom→fiber map.
        """
        return self.processed_ephys / f"{self.session_name}.clu.{shank}"

    def evt_file(self, suffix: str = "all") -> Path:
        """Event file (.{suffix}.evt)."""
        return self.processed_ephys / f"{self.session_name}.{suffix}.evt"

    # ------------------------------------------------------------------ #
    # Neurosuite-3 variant naming convention                              #
    # ------------------------------------------------------------------ #
    #
    # Neurosuite-3 groups per-shank artifacts into three classes with
    # different naming and resolution rules (see the spec at
    # `doc/ndmanager-plugins/formats/naming.md` in the neurosuite-3
    # repository):
    #
    #   ┌──────────────┬─────────────────────────────────────────────┐
    #   │ Class        │ Canonical layout                             │
    #   ├──────────────┼─────────────────────────────────────────────┤
    #   │ SessionWide  │ <base>.<type>                               │
    #   │              │  (fil, dat, xml, yaml, nrs, par, eeg, lfp)  │
    #   │ MethodSpec.  │ <base>.<type>.<method>.<shank>              │
    #   │              │  (clu, clc, clp, fet, pca, col, model, klg) │
    #   │              │  strict — no cross-variant fall-back        │
    #   │ Shared       │ prefer <base>.<type>.<method>.<shank>       │
    #   │              │  then     <base>.<type>.standard.<shank>    │
    #   │              │  then     <base>.<type>.<shank>  (legacy)   │
    #   │              │  (res, spk)                                 │
    #   └──────────────┴─────────────────────────────────────────────┘
    #
    # ``method`` is a variant tag (e.g. ``'standard'`` for the raw
    # domain, ``'stderiv'`` for the spatial-derivative /
    # temporal-difference domain).  Any string is allowed — new
    # variants are added just by writing files named with them.
    #
    # The retired ``.spkD`` / ``.pcaD`` / ``.fetD`` "D-suffix"
    # names are subsumed by ``.spk.stderiv`` / ``.pca.stderiv`` /
    # ``.fet.stderiv``.  ``.spk`` is Shared, so a stderiv session
    # can still fall through to a raw ``.spk`` because the stderiv
    # transform is applied downstream at PCA time.

    _NS3_SESSION_WIDE: tuple[str, ...] = (
        "fil", "dat", "xml", "yaml", "nrs", "par", "eeg", "lfp",
    )
    _NS3_METHOD_SPECIFIC: tuple[str, ...] = (
        "clu", "clc", "clp", "fet", "pca", "col", "model", "klg",
    )
    _NS3_SHARED: tuple[str, ...] = ("res", "spk")

    def ns3_class(self, type_id: str) -> str:
        """Classify *type_id* per the neurosuite-3 variant naming spec.

        Returns one of ``'session_wide'``, ``'method_specific'``,
        ``'shared'``.  Raises :class:`ValueError` for unknown types.
        """
        if type_id in self._NS3_SESSION_WIDE:
            return "session_wide"
        if type_id in self._NS3_METHOD_SPECIFIC:
            return "method_specific"
        if type_id in self._NS3_SHARED:
            return "shared"
        raise ValueError(
            f"Unknown neurosuite-3 artifact type {type_id!r}.  "
            f"Known types: session-wide={self._NS3_SESSION_WIDE}, "
            f"method-specific={self._NS3_METHOD_SPECIFIC}, "
            f"shared={self._NS3_SHARED}"
        )

    def ns3_file(
        self,
        type_id: str,
        shank:   int | None = None,
        method:  str = "standard",
    ) -> Path:
        """Return the *canonical* variant-tagged path per neurosuite-3.

        This is the pure name-computing helper — no filesystem lookup.

        Parameters
        ----------
        type_id:
            Artifact type (``'clu'``, ``'spk'``, ``'fet'``, …).
        shank:
            Electrode group index (1-based).  Required for
            method-specific and shared types; must be ``None`` for
            session-wide types.
        method:
            Variant tag.  Defaults to ``'standard'``.

        Returns
        -------
        pathlib.Path

        Raises
        ------
        ValueError
            If *shank* is required but missing (or vice versa), or
            if *type_id* is unknown.
        """
        cls = self.ns3_class(type_id)
        if cls == "session_wide":
            if shank is not None:
                raise ValueError(
                    f"neurosuite-3 type {type_id!r} is session-wide; "
                    "shank must be None"
                )
            return self.processed_ephys / f"{self.session_name}.{type_id}"

        if shank is None:
            raise ValueError(
                f"neurosuite-3 type {type_id!r} requires a shank"
            )

        # method-specific + shared use the same canonical filename
        return (
            self.processed_ephys
            / f"{self.session_name}.{type_id}.{method}.{shank}"
        )

    def resolve_ns3(
        self,
        type_id: str,
        shank:   int | None = None,
        method:  str = "standard",
    ) -> tuple[Path, str]:
        """Resolve a neurosuite-3 artifact on disk with class-specific
        fall-back rules.

        See :meth:`ns3_class` for the resolution policy per class:

        * SessionWide     — exactly one candidate; returned as-is.
        * MethodSpecific  — strict; the variant-tagged path is
                            returned regardless of existence.
        * Shared          — try ``<method>``, then ``standard``,
                            then the untagged legacy path; return
                            the first that exists.  Falls through
                            to the variant-tagged path (for use in
                            error messages) if none exist.

        Parameters
        ----------
        type_id, shank, method:
            As for :meth:`ns3_file`.

        Returns
        -------
        (path, resolved_method):
            *path* is the resolved :class:`~pathlib.Path`.
            *resolved_method* is the variant tag of the file that
            was actually found — ``'standard'`` when the untagged
            legacy path was used, or the caller's *method* for a
            MethodSpecific / SessionWide type (they never fall
            back).  For Shared types callers can compare
            ``resolved_method == 'stderiv'`` to reproduce the spec's
            ``resolvedIsStderiv()`` predicate.
        """
        cls = self.ns3_class(type_id)
        if cls == "session_wide":
            return self.ns3_file(type_id, method=method), method

        if cls == "method_specific":
            # Strict — no fall-back to another variant.
            return self.ns3_file(type_id, shank, method=method), method

        # Shared:  <method> → standard → untagged legacy
        base = self.processed_ephys
        primary = base / f"{self.session_name}.{type_id}.{method}.{shank}"
        if primary.exists():
            return primary, method
        if method != "standard":
            std = base / f"{self.session_name}.{type_id}.standard.{shank}"
            if std.exists():
                return std, "standard"
        untagged = base / f"{self.session_name}.{type_id}.{shank}"
        if untagged.exists():
            return untagged, "standard"
        # None found — return the primary path for a clear error msg
        return primary, method

    # ------------------------------------------------------------------ #
    # Type-specific helpers                                               #
    # ------------------------------------------------------------------ #

    def spk_file(
        self,
        shank:  int,
        method: str = "standard",
    ) -> Path:
        """Spike waveforms for shank N (variant-tagged path).

        In neurosuite-3, ``.spk`` is a *Shared* artifact — the raw
        waveform snippets are method-independent, so a stderiv
        session can read the same file as a standard session.
        Use :meth:`resolve_ns3` (``type='spk'``) to search with the
        Shared fall-back rules.
        """
        return self.ns3_file("spk", shank, method=method)

    def fet_file(
        self,
        shank:  int,
        method: str = "standard",
    ) -> Path:
        """PCA feature vectors for shank N — canonical path
        ``<base>.fet.<method>.<shank>``.

        MethodSpecific: strict, no cross-variant fall-back.  The
        retired ``.fetD.N`` name is subsumed by
        ``method='stderiv'``.
        """
        return self.ns3_file("fet", shank, method=method)

    def pca_file(
        self,
        shank:  int,
        method: str = "standard",
    ) -> Path:
        """PCA eigenvector basis for shank N — canonical path
        ``<base>.pca.<method>.<shank>``.

        MethodSpecific.  The retired ``.pcaD.N`` name is subsumed
        by ``method='stderiv'``.
        """
        return self.ns3_file("pca", shank, method=method)

    def clc_file(
        self,
        shank:  int,
        method: str = "standard",
    ) -> Path:
        """Atom (child-layer) cluster assignments for shank N.

        MethodSpecific: strict.  Present only in hierarchical
        sessions; carries the same ``<method>`` tag as the
        matching :meth:`clu_file` / :meth:`clp_file`.
        """
        return self.ns3_file("clc", shank, method=method)

    def clp_file(
        self,
        shank:  int,
        method: str = "standard",
    ) -> Path:
        """Atom→fiber (child→parent) linkage for shank N.

        MethodSpecific.  Carries the same ``<method>`` tag as the
        matching :meth:`clu_file` / :meth:`clc_file` (see
        ``clp.md`` in the neurosuite-3 spec).
        """
        return self.ns3_file("clp", shank, method=method)

    def col_file(
        self,
        shank:  int,
        method: str = "standard",
    ) -> Path:
        """Collision-decomposition results for shank N (YAML,
        MethodSpecific)."""
        return self.ns3_file("col", shank, method=method)

    def clu_ns3_file(
        self,
        shank:  int,
        method: str = "standard",
    ) -> Path:
        """Variant-tagged ``.clu.<method>.<shank>``.

        Separate from :meth:`clu_file` (which returns the untagged
        legacy path for backward compat).  Use this one for new
        neurosuite-3 layouts.
        """
        return self.ns3_file("clu", shank, method=method)

    def res_ns3_file(
        self,
        shank:  int,
        method: str = "standard",
    ) -> Path:
        """Variant-tagged ``.res.<method>.<shank>`` (Shared).

        See :meth:`resolve_ns3` to search with fall-back to the
        untagged legacy ``<base>.res.<shank>``.
        """
        return self.ns3_file("res", shank, method=method)

    # ------------------------------------------------------------------ #
    # Analysis output paths (written to spath)                           #
    # ------------------------------------------------------------------ #

    @property
    def ses_file(self) -> Path:
        return self.spath / f"{self.session_name}.cof.all.ses.pkl"

    def stc_file(self, mode: str = "default") -> Path:
        return self.spath / f"{self.session_name}.cof.all.stc.{mode}.pkl"

    @property
    def pos_file(self) -> Path:
        return self.spath / f"{self.session_name}.cof.all.pos.npz"

    # ------------------------------------------------------------------ #
    # Existence checks                                                    #
    # ------------------------------------------------------------------ #

    def check_processed(self) -> dict[str, bool]:
        """Return a dict of which standard processed files exist."""
        return {
            "yaml":   self.yaml_file.exists(),
            "dat":    self.dat_file.exists(),
            "lfp":    self.lfp_file.exists(),
            "res.1":  self.res_file(1).exists(),
            "clu.1":  self.clu_file(1).exists(),
            "evt":    self.evt_file().exists(),
        }

    def check_source(self) -> dict[str, bool]:
        return {
            "nlx":   self.source_nlx.exists(),
            "mocap": self.source_mocap.exists(),
            "video": self.source_video.exists(),
        }

    # ------------------------------------------------------------------ #
    # Repr                                                                #
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        return (
            f"NBSessionPaths(\n"
            f"  session   = {self.session_name!r}\n"
            f"  spath     = {self.spath}\n"
            f"  ephys     = {self.processed_ephys}\n"
            f"  mocap     = {self.processed_mocap}\n"
            f")"
        )
