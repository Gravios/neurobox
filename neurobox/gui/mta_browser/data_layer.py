"""
neurobox.gui.mta_browser.data_layer
=====================================
Filesystem scanning and session-discovery helpers.

The MATLAB ``BSdataManagement_Callback`` (lines 387-526 of
``MTABrowser.m``) walks the project directory using a regex and
groups sessions by ``subject -> date -> maze -> trial``.  This module
extracts those rules into pure functions that the GUI can call
without holding any Qt state.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

__all__ = [
    "SessionEntry",
    "ProjectIndex",
    "scan_project",
]


# Session naming convention: 2-letter subject + 2-4-digit ID + dash +
# 8-digit date.  E.g. "jg05-20120316", "Ed03-20140624a".
_SESSION_RE = re.compile(r"^([a-zA-Z]{1,2}[0-9]{2,4})[-]([0-9]{8})([a-zA-Z]*)$")
_MAZE_RE    = re.compile(r"\.([^.]+)\.ses(?:\.|$)")
_TRIAL_RE   = re.compile(r"\.([^.]+)\.([^.]+)\.trl(?:\.|$)")


@dataclass
class SessionEntry:
    """One discovered session with its mazes and trials.

    Attributes
    ----------
    name : str
        Full session name, e.g. ``'jg05-20120316'``.
    subject : str
        Subject prefix, e.g. ``'jg05'``.
    date : str
        8-digit date string + suffix, e.g. ``'20120316'`` or
        ``'20140624a'``.
    path : Path
        Absolute path to the session directory.
    mazes : list[str]
        Maze identifiers found in this session (e.g.
        ``['cof', 'sof']``).
    trials : dict[str, list[str]]
        Mapping from maze → list of trial labels (e.g.
        ``{'cof': ['all', 'task1']}``).
    """
    name:    str
    subject: str
    date:    str
    path:    Path
    mazes:   list[str]            = field(default_factory=list)
    trials:  dict[str, list[str]] = field(default_factory=dict)


@dataclass
class ProjectIndex:
    """Hierarchical index of all sessions in a project tree.

    Attributes
    ----------
    project_root : Path
    by_subject : dict[str, list[SessionEntry]]
        Subject → list of session entries (sorted by date).
    """
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


def scan_project(project_root: Path | str) -> ProjectIndex:
    """Walk *project_root* and return a :class:`ProjectIndex`.

    Mirrors the directory-scan behaviour of
    ``BSdataManagement_Callback`` lines 397-503.

    Parameters
    ----------
    project_root:
        Root directory containing per-session sub-directories.

    Returns
    -------
    ProjectIndex
        Empty if *project_root* doesn't exist.
    """
    project_root = Path(project_root)
    by_subject: dict[str, list[SessionEntry]] = {}
    if not project_root.is_dir():
        return ProjectIndex(project_root=project_root, by_subject=by_subject)

    for entry in sorted(project_root.iterdir()):
        if not entry.is_dir():
            continue
        m = _SESSION_RE.match(entry.name)
        if not m:
            continue
        subject = m.group(1)
        date    = m.group(2) + (m.group(3) or "")

        mazes  : list[str]            = []
        trials : dict[str, list[str]] = {}
        # Pass 1 — discover mazes via .ses.mat files.  Sort the
        # filesystem listing so the result is deterministic across
        # platforms (ext4 / btrfs / NFS hand back unsorted entries).
        for f in sorted(entry.iterdir()):
            if f.suffix != ".mat":
                continue
            mm = _MAZE_RE.search(f.name)
            if mm and f.name.endswith(".ses.mat"):
                maze = mm.group(1)
                if maze not in mazes:
                    mazes.append(maze)
                    trials[maze] = []
        # Pass 2 — discover trials via .trl.mat files (also sorted)
        for f in sorted(entry.iterdir()):
            if f.suffix != ".mat":
                continue
            tm = _TRIAL_RE.search(f.name)
            if tm and f.name.endswith(".trl.mat"):
                maze, trial = tm.group(1), tm.group(2)
                if maze in trials and trial not in trials[maze]:
                    trials[maze].append(trial)

        # Always include 'all' as the default trial if the maze has any
        # entry.  MATLAB convention: every maze has an implicit 'all'.
        for maze, t_list in trials.items():
            t_list.sort()
            if "all" not in t_list:
                t_list.insert(0, "all")

        sess = SessionEntry(
            name    = entry.name,
            subject = subject,
            date    = date,
            path    = entry,
            mazes   = mazes,
            trials  = trials,
        )
        by_subject.setdefault(subject, []).append(sess)

    # Sort sessions by date within each subject
    for v in by_subject.values():
        v.sort(key=lambda s: s.date)

    return ProjectIndex(project_root=project_root, by_subject=by_subject)
