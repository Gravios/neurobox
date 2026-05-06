"""
neurobox.config.sessions_json
==============================
Loader for the ``data/sessions.json`` file produced by
:mod:`scripts.convert_session_list`.

That file has two top-level sections:

* ``subjects`` — per-subject constants (``correction``,
  ``channelGroup``, ``anatLoc`` keys that are stable across all
  recordings of an animal).  Keyed by ``subject.name``.
* ``lists``    — per-experiment lists of session entries.  Each
  session has a ``subject: "<name>"`` link and any keys whose values
  vary between recordings (e.g. probe-dependent ``channelGroup``,
  recording-mount-dependent ``correction.headYaw``).

This module loads that file and provides the **effective** view of a
session — the merge of its subject defaults with its per-session
fields.  Session values win on conflict, so an animal's stable
``thetaPhase`` shows up automatically while a session-specific
``headYaw`` correction overrides cleanly.

Public API
----------
:func:`load_sessions_json`
    Read the file from disk and return a :class:`SessionsCatalog`.
:class:`SessionsCatalog`
    Search / iterate / look up sessions and resolve effective values.
:func:`effective_session`
    Lower-level helper — merge raw subject + session dicts.

This module is independent from :mod:`neurobox.config.session_lists`,
which loads the per-list YAML format.  Both end up describing the
same MTA session metadata; the JSON format is more convenient when
you need to query across all 44 lists at once (e.g. "every session
that uses subject ``jg05``").
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional


__all__ = [
    "DEFAULT_SESSIONS_JSON",
    "EffectiveSession",
    "SessionsCatalog",
    "effective_session",
    "load_sessions_json",
]


# Default location of the bundled JSON file (sibling of this module's
# package, under ``<repo>/data/sessions.json``).
DEFAULT_SESSIONS_JSON = (
    Path(__file__).resolve().parents[2] / "data" / "sessions.json"
)

# Sections that get merged from subject → session via ``__or__``-style
# precedence (session keys win).  Must match the converter's
# ``_LIFTABLE_SECTIONS`` constant.
_MERGEABLE_SECTIONS = ("correction", "channelGroup", "anatLoc")


# ─────────────────────────────────────────────────────────────────────── #
# Effective view                                                              #
# ─────────────────────────────────────────────────────────────────────── #

def effective_session(
    session: dict,
    subjects: dict[str, dict],
) -> dict:
    """Return *session* with subject defaults merged in.

    The merge rule for the three :data:`_MERGEABLE_SECTIONS`
    (``correction``, ``channelGroup``, ``anatLoc``) is
    ``subject_section | session_section`` — session keys win on
    conflict.

    Parameters
    ----------
    session:
        One entry from ``lists[<name>]``.  Typically has a bare
        string ``subject`` field linking to *subjects*.
    subjects:
        Top-level ``subjects`` dict from the JSON.  May be empty.

    Returns
    -------
    dict
        A NEW dict (input untouched) with each mergeable section
        holding the effective merged values.  ``subject`` becomes a
        full dict ``{name, correction, channelGroup, anatLoc}`` so
        downstream code can call ``s["subject"]["name"]`` without
        having to know whether it's a string or dict.
    """
    out = {k: v for k, v in session.items()}
    subj_link = out.get("subject")
    if isinstance(subj_link, str):
        subj_record = subjects.get(subj_link, {"name": subj_link})
    elif isinstance(subj_link, dict):
        subj_record = subj_link
    else:
        return out

    # Build the merged subject struct
    merged_subject: dict = {"name": subj_record.get("name", subj_link)}
    for section in _MERGEABLE_SECTIONS:
        sub_sec   = subj_record.get(section, {}) or {}
        sess_sec  = out.get(section, {}) or {}
        if isinstance(sub_sec, dict) and isinstance(sess_sec, dict):
            merged = {**sub_sec, **sess_sec}    # session wins
        elif isinstance(sess_sec, dict) and sess_sec:
            merged = dict(sess_sec)
        elif isinstance(sub_sec, dict):
            merged = dict(sub_sec)
        else:
            merged = {}
        if merged:
            merged_subject[section] = merged
        # Remove the per-section override key from the session view —
        # the merged value lives under merged_subject now.  Callers
        # that want raw session-specific overrides can still read the
        # original dict.
        out.pop(section, None)
    out["subject"] = merged_subject
    return out


# ─────────────────────────────────────────────────────────────────────── #
# EffectiveSession dataclass                                                  #
# ─────────────────────────────────────────────────────────────────────── #

@dataclass(frozen=True)
class EffectiveSession:
    """A single session entry with subject defaults already merged in.

    Attributes
    ----------
    list_name:
        Name of the list this session was found in (e.g.
        ``'BehaviorPlaceCode'``).  Same session may appear in
        multiple lists — see :meth:`SessionsCatalog.find`.
    raw:
        The unmerged session dict from the JSON.  Keep around for
        callers that want to know which keys were session-specific
        overrides versus inherited subject defaults.
    merged:
        The merged session dict — ``raw`` plus subject defaults
        applied to ``correction`` / ``channelGroup`` / ``anatLoc``.
    """
    list_name: str
    raw:       dict
    merged:    dict

    # ── Convenience accessors — read from .merged ─────────────────── #

    @property
    def session_name(self) -> str:
        return self.merged.get("sessionName", "")

    @property
    def maze_name(self) -> str:
        return self.merged.get("mazeName", "")

    @property
    def trial_name(self) -> str:
        return self.merged.get("trialName", "all")

    @property
    def subject(self) -> dict:
        """Merged subject struct — always a dict with at least a
        ``name`` key, plus any merged ``correction`` / ``channelGroup``
        / ``anatLoc``."""
        return self.merged.get("subject", {})

    @property
    def subject_name(self) -> str:
        return self.subject.get("name", "")

    @property
    def correction(self) -> dict:
        return self.subject.get("correction", {})

    @property
    def channel_group(self) -> dict:
        return self.subject.get("channelGroup", {})

    @property
    def anat_loc(self) -> dict:
        return self.subject.get("anatLoc", {})


# ─────────────────────────────────────────────────────────────────────── #
# Catalog                                                                     #
# ─────────────────────────────────────────────────────────────────────── #

@dataclass
class SessionsCatalog:
    """In-memory view of ``data/sessions.json``.

    Use :func:`load_sessions_json` to construct.  Methods:

    * :meth:`list_names`        — names of all session-list groups
    * :meth:`subjects`          — names of all known subjects
    * :meth:`get`               — look up a single session by
                                  ``(list_name, session_name)``
    * :meth:`find`              — find every occurrence of a session
                                  name across all lists
    * :meth:`for_list`          — iterate :class:`EffectiveSession`
                                  for one list
    * :meth:`for_subject`       — iterate every session of one subject
    * :meth:`get_subject`       — raw subject dict (unmerged defaults)
    """
    subjects_raw: dict[str, dict] = field(default_factory=dict)
    lists:        dict[str, list[dict]] = field(default_factory=dict)

    # ── Inventory ─────────────────────────────────────────────────── #

    def list_names(self) -> list[str]:
        return sorted(self.lists.keys())

    def subjects(self) -> list[str]:
        return sorted(self.subjects_raw.keys())

    # ── Subject access ────────────────────────────────────────────── #

    def get_subject(self, name: str) -> Optional[dict]:
        """Raw subject dict (unmerged defaults), or None if unknown."""
        return self.subjects_raw.get(name)

    # ── Session access ────────────────────────────────────────────── #

    def get(
        self,
        list_name:    str,
        session_name: str,
    ) -> Optional[EffectiveSession]:
        """Look up a single session.  Returns None if not found."""
        for s in self.lists.get(list_name, []):
            if s.get("sessionName") == session_name:
                return EffectiveSession(
                    list_name = list_name,
                    raw       = s,
                    merged    = effective_session(s, self.subjects_raw),
                )
        return None

    def find(self, session_name: str) -> list[EffectiveSession]:
        """Find every list containing *session_name* (it can be in
        more than one — the same session often appears in
        ``BehaviorPlaceCode``, ``hand_labeled_jg``, etc.)."""
        out = []
        for list_name, entries in self.lists.items():
            for s in entries:
                if s.get("sessionName") == session_name:
                    out.append(EffectiveSession(
                        list_name = list_name,
                        raw       = s,
                        merged    = effective_session(
                            s, self.subjects_raw,
                        ),
                    ))
        return out

    # ── Iteration ─────────────────────────────────────────────────── #

    def for_list(self, list_name: str) -> Iterator[EffectiveSession]:
        """Yield :class:`EffectiveSession`s for one list."""
        for s in self.lists.get(list_name, []):
            yield EffectiveSession(
                list_name = list_name,
                raw       = s,
                merged    = effective_session(s, self.subjects_raw),
            )

    def for_subject(self, subject: str) -> Iterator[EffectiveSession]:
        """Yield every session whose subject is *subject*, across
        all lists."""
        for list_name, entries in self.lists.items():
            for s in entries:
                subj = s.get("subject")
                if subj == subject or (
                    isinstance(subj, dict) and subj.get("name") == subject
                ):
                    yield EffectiveSession(
                        list_name = list_name,
                        raw       = s,
                        merged    = effective_session(
                            s, self.subjects_raw,
                        ),
                    )

    def __iter__(self) -> Iterator[EffectiveSession]:
        """Yield every session in every list (with duplicates if a
        session appears in multiple lists)."""
        for name in self.lists:
            yield from self.for_list(name)

    def __len__(self) -> int:
        return sum(len(v) for v in self.lists.values())


# ─────────────────────────────────────────────────────────────────────── #
# Loader                                                                      #
# ─────────────────────────────────────────────────────────────────────── #

def load_sessions_json(
    path: Path | str | None = None,
) -> SessionsCatalog:
    """Load and parse a ``sessions.json`` file.

    Parameters
    ----------
    path:
        Path to the JSON file.  Defaults to
        :data:`DEFAULT_SESSIONS_JSON` (the bundled file at
        ``<repo>/data/sessions.json``).

    Returns
    -------
    SessionsCatalog

    Raises
    ------
    FileNotFoundError
        If *path* (or the default) doesn't exist.
    ValueError
        If the file is malformed JSON.

    Backwards compatibility
    -----------------------
    Files written before the ``--lift-subjects`` flag existed have
    no top-level ``subjects`` key.  Such files are loaded with an
    empty subjects dict, so :func:`effective_session` becomes a
    no-op and the catalog still works — every session entry just
    has the full subject struct embedded in-place under the
    ``subject`` key.
    """
    p = Path(path) if path is not None else DEFAULT_SESSIONS_JSON
    if not p.exists():
        raise FileNotFoundError(f"Sessions file not found: {p}")
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"Malformed JSON in {p}: {e}") from e
    if not isinstance(data, dict):
        raise ValueError(
            f"{p}: expected an object at the top level, got {type(data).__name__}"
        )
    return SessionsCatalog(
        subjects_raw = data.get("subjects", {}) or {},
        lists        = data.get("lists",    {}) or {},
    )
