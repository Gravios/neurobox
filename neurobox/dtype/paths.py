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
        """Spike timestamps for shank N (.res.N)."""
        return self.processed_ephys / f"{self.session_name}.res.{shank}"

    def clu_file(self, shank: int) -> Path:
        """Cluster assignments for shank N (.clu.N)."""
        return self.processed_ephys / f"{self.session_name}.clu.{shank}"

    def evt_file(self, suffix: str = "all") -> Path:
        """Event file (.{suffix}.evt)."""
        return self.processed_ephys / f"{self.session_name}.{suffix}.evt"

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
