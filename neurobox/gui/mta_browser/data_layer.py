"""
neurobox.gui.mta_browser.data_layer
=====================================
Filesystem scanning and session-discovery helpers, parameterised by a
:class:`NamingConfig` so users can point the browser at projects laid
out under different conventions:

* **neurobox** (default) — canonical 4-part session names
  ``<sourceId>-<userId>-<subjectId>-<date>`` plus per-trial files
  ``<filebase>.ses.pkl`` / ``<filebase>.trl.pkl``.  This matches what
  :class:`~neurobox.dtype.NBSession` and :class:`~neurobox.dtype.NBTrial`
  write to disk.
* **labbox-mta** (legacy) — the pre-2026 MATLAB layout: 2-letter
  subject + numeric ID (``jg05-20120316``) plus ``.ses.mat`` /
  ``.trl.mat`` per-trial files.

The MATLAB ``BSdataManagement_Callback`` (lines 387-526 of
``MTABrowser.m``) walks the project directory using a regex and
groups sessions by ``subject -> date -> maze -> trial``.  This module
extracts those rules into pure functions that the GUI can call
without holding any Qt state.

Both conventions can coexist in the same project tree — the scanner
will try multiple :class:`NamingConfig`s in order and merge the
results.  The default :func:`scan_project` call uses
:func:`default_naming_configs` which prefers neurobox and falls back
to labbox-mta.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional, Pattern

__all__ = [
    "NamingConfig",
    "SessionEntry",
    "ProjectIndex",
    "scan_project",
    "neurobox_naming",
    "labbox_mta_naming",
    "default_naming_configs",
]


# ─────────────────────────────────────────────────────────────────────── #
# NamingConfig                                                                #
# ─────────────────────────────────────────────────────────────────────── #

@dataclass(frozen=True)
class NamingConfig:
    """Describes how session/maze/trial names are encoded on disk.

    Attributes
    ----------
    name :
        Short identifier (used in the GUI preferences UI).
    session_pattern :
        Regex matching session-directory names.  Required named
        groups: ``subject`` and ``date``.  Additional named groups
        are allowed and ignored.
    session_marker_glob :
        Glob pattern (relative to the session dir) that matches
        files identifying mazes — one file per maze.  E.g.
        ``"*.ses.pkl"`` for neurobox or ``"*.ses.mat"`` for legacy
        MATLAB.  The maze is extracted via :attr:`maze_from_filename`.
    trial_marker_glob :
        Same idea for per-trial files.  E.g. ``"*.trl.pkl"``.
    maze_from_filename :
        Function ``filename -> maze | None``.  Skips the file when
        it returns None.  Default: pull the second-to-last
        dot-separated token from the stem.
    trial_from_filename :
        Function ``filename -> (maze, trial) | None``.  Default:
        parse ``<sessionname>.<maze>.<trial>.trl.<ext>``.
    """
    name:                str
    session_pattern:     Pattern[str]
    session_marker_glob: str
    trial_marker_glob:   str
    maze_from_filename:  Callable[[str], Optional[str]]
    trial_from_filename: Callable[[str], Optional[tuple[str, str]]]

    def parse_session_dir(self, name: str) -> Optional[tuple[str, str]]:
        """Return ``(subject, date)`` for a session directory name,
        or None if the name doesn't match this convention."""
        m = self.session_pattern.match(name)
        if m is None:
            return None
        try:
            subject = m.group("subject")
            date    = m.group("date")
        except IndexError:
            return None        # pattern lacks required named groups
        return subject, date


# ─────────────────────────────────────────────────────────────────────── #
# Built-in NamingConfig presets                                              #
# ─────────────────────────────────────────────────────────────────────── #

def _maze_from_dot_stem(filename: str) -> Optional[str]:
    """``some-session.cof.ses.pkl`` → ``"cof"``.

    Returns None if the filename doesn't have at least two
    dot-separated tokens before the marker suffix.
    """
    parts = filename.rsplit(".", 3)
    # parts: [<base>, <maze>, <marker>, <ext>]
    if len(parts) < 4:
        return None
    return parts[1] or None


def _trial_from_dot_stem(filename: str) -> Optional[tuple[str, str]]:
    """``some-session.cof.task1.trl.pkl`` → ``("cof", "task1")``."""
    parts = filename.rsplit(".", 4)
    # parts: [<base>, <maze>, <trial>, <marker>, <ext>]
    if len(parts) < 5:
        return None
    maze, trial = parts[1], parts[2]
    if not maze or not trial:
        return None
    return maze, trial


# Reuse the canonical 4-part regex from neurobox.dtype.paths so both
# the GUI and the dtype layer agree on what a valid neurobox session
# name looks like.
_NEUROBOX_SESSION_RE = re.compile(
    r"^(?P<sourceId>.+?)-(?P<userId>[A-Za-z]+)-"
    r"(?P<subject>\d+)-(?P<date>\d{8}[a-zA-Z]*)$"
)


neurobox_naming = NamingConfig(
    name                = "neurobox",
    session_pattern     = _NEUROBOX_SESSION_RE,
    session_marker_glob = "*.ses.pkl",
    trial_marker_glob   = "*.trl.pkl",
    maze_from_filename  = _maze_from_dot_stem,
    trial_from_filename = _trial_from_dot_stem,
)
"""Canonical neurobox naming.  Matches what :class:`NBSession` and
:class:`NBTrial` write to disk: 4-part session directory name plus
``<filebase>.ses.pkl`` / ``<filebase>.trl.pkl`` per-trial files."""


# Legacy labbox-MTA: 2-letter subject + numeric ID, optional letter
# date suffix, ``.mat`` files.
_LABBOX_MTA_SESSION_RE = re.compile(
    r"^(?P<subject>[a-zA-Z]{1,2}[0-9]{2,4})[-]"
    r"(?P<date>[0-9]{8}[a-zA-Z]*)$"
)

labbox_mta_naming = NamingConfig(
    name                = "labbox-mta",
    session_pattern     = _LABBOX_MTA_SESSION_RE,
    session_marker_glob = "*.ses.mat",
    trial_marker_glob   = "*.trl.mat",
    maze_from_filename  = _maze_from_dot_stem,
    trial_from_filename = _trial_from_dot_stem,
)
"""Legacy labbox/MTA naming.  Used for projects predating the 2026
neurobox port.  Recognises 2-letter subject codes (e.g. ``jg05``)
and ``.ses.mat`` / ``.trl.mat`` per-trial files."""


def default_naming_configs() -> list[NamingConfig]:
    """The default scan precedence: neurobox first, then labbox-mta.

    A session directory is matched by the first config whose
    ``session_pattern`` accepts its name.  Mazes and trials within
    that directory are then resolved using only that config.
    """
    return [neurobox_naming, labbox_mta_naming]


# ─────────────────────────────────────────────────────────────────────── #
# SessionEntry / ProjectIndex                                                 #
# ─────────────────────────────────────────────────────────────────────── #

@dataclass
class SessionEntry:
    """One discovered session with its mazes and trials.

    Attributes
    ----------
    name : str
        Full session name, e.g. ``'sirotaA-jg-05-20120316'`` or
        ``'jg05-20120316'`` (depending on the matched
        :class:`NamingConfig`).
    subject : str
        Subject identifier extracted from ``name`` via the matched
        config's ``session_pattern``.
    date : str
        Date string + optional letter suffix.
    path : Path
        Absolute path to the session directory.
    mazes : list[str]
        Maze identifiers (sorted lexically).
    trials : dict[str, list[str]]
        Mapping from maze → list of trial labels (with ``'all'``
        always present).
    naming : str
        Name of the :class:`NamingConfig` that matched this
        session.
    """
    name:    str
    subject: str
    date:    str
    path:    Path
    mazes:   list[str]            = field(default_factory=list)
    trials:  dict[str, list[str]] = field(default_factory=dict)
    naming:  str                  = ""


@dataclass
class ProjectIndex:
    """Hierarchical index of all sessions in a project tree."""
    project_root: Path
    by_subject:   dict[str, list[SessionEntry]]

    def subjects(self) -> list[str]:
        """Return all subject codes, sorted."""
        return sorted(self.by_subject.keys())

    def dates_for(self, subject: str) -> list[str]:
        """Return dates available for *subject*."""
        return [s.date for s in self.by_subject.get(subject, [])]

    def session_for(self, subject: str, date: str) -> Optional[SessionEntry]:
        """Return the SessionEntry for ``subject``/``date``, or None."""
        for s in self.by_subject.get(subject, []):
            if s.date == date:
                return s
        return None

    def mazes_for(self, subject: str, date: str) -> list[str]:
        s = self.session_for(subject, date)
        return list(s.mazes) if s else []

    def trials_for(
        self, subject: str, date: str, maze: str,
    ) -> list[str]:
        s = self.session_for(subject, date)
        return list(s.trials.get(maze, [])) if s else []


# ─────────────────────────────────────────────────────────────────────── #
# scan_project                                                                #
# ─────────────────────────────────────────────────────────────────────── #

def scan_project(
    project_root: Path | str,
    namings:      Optional[Iterable[NamingConfig]] = None,
) -> ProjectIndex:
    """Walk *project_root* and return a :class:`ProjectIndex`.

    Mirrors the directory-scan behaviour of MATLAB's
    ``BSdataManagement_Callback``, generalised over
    :class:`NamingConfig` so both the new neurobox layout and the
    legacy labbox-mta layout can be discovered.

    Parameters
    ----------
    project_root:
        Root directory containing per-session sub-directories.
    namings:
        Iterable of :class:`NamingConfig` to try, in order.  The
        first config whose ``session_pattern`` matches a directory
        name is used for that session — and *only* that config is
        used to resolve its mazes and trials, so configs don't
        cross-contaminate.  If None, defaults to
        :func:`default_naming_configs`.

    Returns
    -------
    ProjectIndex
        Empty if *project_root* doesn't exist.

    Notes
    -----
    All filesystem iteration is sorted, so results are deterministic
    across platforms.
    """
    project_root = Path(project_root)
    by_subject: dict[str, list[SessionEntry]] = {}
    if not project_root.is_dir():
        return ProjectIndex(project_root=project_root,
                              by_subject=by_subject)

    if namings is None:
        namings = default_naming_configs()
    namings = list(namings)
    if not namings:
        return ProjectIndex(project_root=project_root,
                              by_subject=by_subject)

    for entry in sorted(project_root.iterdir()):
        if not entry.is_dir():
            continue

        # Find the first naming config that recognises this directory
        match = None
        for cfg in namings:
            parsed = cfg.parse_session_dir(entry.name)
            if parsed is not None:
                match = (cfg, parsed)
                break
        if match is None:
            continue
        cfg, (subject, date) = match

        mazes  : list[str]            = []
        trials : dict[str, list[str]] = {}

        # Pass 1 — discover mazes via this config's session-marker glob
        for f in sorted(entry.glob(cfg.session_marker_glob)):
            maze = cfg.maze_from_filename(f.name)
            if maze and maze not in mazes:
                mazes.append(maze)
                trials[maze] = []

        # Pass 2 — discover trials via this config's trial-marker glob
        for f in sorted(entry.glob(cfg.trial_marker_glob)):
            tm = cfg.trial_from_filename(f.name)
            if tm is None:
                continue
            maze, trial = tm
            if maze in trials and trial not in trials[maze]:
                trials[maze].append(trial)

        # Always include 'all' as the default trial — every maze has
        # an implicit 'all'.  Sort the rest for stable display.
        for maze, t_list in trials.items():
            t_list.sort()
            if "all" not in t_list:
                t_list.insert(0, "all")

        sess = SessionEntry(
            name    = entry.name,
            subject = subject,
            date    = date,
            path    = entry,
            mazes   = sorted(mazes),
            trials  = trials,
            naming  = cfg.name,
        )
        by_subject.setdefault(subject, []).append(sess)

    # Sort sessions by date within each subject
    for v in by_subject.values():
        v.sort(key=lambda s: s.date)

    return ProjectIndex(project_root=project_root, by_subject=by_subject)
