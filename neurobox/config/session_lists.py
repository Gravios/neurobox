"""
neurobox.config.session_lists
==============================
Typed access to MTA-style session metadata stored as YAML.

Replaces the giant ``get_session_list_v3.m`` switch statement with a
collection of YAML files, one per project (``BehaviorPlaceCode``,
``EgoProCode2D``, ``EgoProCode2D_CA3``, ``EgoProCode2d_CA1``).

The YAMLs use ``{xyz_root}`` / ``{nlx_root}`` placeholders for the
machine-specific data roots, which are resolved at load time from
either explicit arguments, environment variables (``NB_XYZ_ROOT`` /
``NB_NLX_ROOT``), or auto-discovered from a ``.env`` file via
:mod:`neurobox.config`.

Public API
----------
:class:`SessionList`
    Container for a YAML file's worth of sessions, with name-based
    lookup, slicing, and bulk conversion to pipeline-spec dicts.
:class:`TrialSpec`
    Dataclass binding a single session's fields together — the
    pipeline spec dict, subject calibration, anatomical labelling.
:class:`SubjectInfo`
    Calibration metadata for one rat: head-rotation corrections,
    LFP channel groups, anatomical region flags.
:func:`load_session_list`
    Load a built-in or external YAML by name or path.
:func:`available_session_lists`
    Names of YAMLs bundled with the package.

Examples
--------
Load all BehaviorPlaceCode trials and run them through pipelines::

    from neurobox.config.session_lists import load_session_list
    from neurobox.pipelines import batch_session_setup

    sl = load_session_list("BehaviorPlaceCode")
    sessions = batch_session_setup([t.spec for t in sl])

Inspect calibration for a specific session::

    trial = sl["jg05-20120329"]
    print(trial.subject.correction.theta_phase)   # → 0.7853981633974483
    print(trial.subject.channel_group.theta)       # → 71
    print(trial.subject.anat_loc.is_ca3)           # → True

Filter by anatomical region::

    ca1_only = [t for t in sl if t.subject.anat_loc.is_ca1]
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterator, Optional, Sequence, Union

import yaml

# Where the bundled YAMLs live
_BUILTIN_DIR = Path(__file__).resolve().parent.parent / "data" / "session_lists"

# Map between user-facing names (camelCase, MATLAB-style) and YAML filenames.
# The MATLAB case names are case-sensitive in the original (e.g. "EgoProCode2d_CA1"
# vs "EgoProCode2D"); we accept both spellings.
_BUILTIN_NAMES: dict[str, str] = {
    "BehaviorPlaceCode":   "behavior_place_code.yaml",
    "EgoProCode2D":        "ego_pro_code_2d.yaml",
    "EgoProCode2D_CA3":    "ego_pro_code_2d_ca3.yaml",
    "EgoProCode2d_CA1":    "ego_pro_code_2d_ca1.yaml",
    "EgoProCode2D_CA1":    "ego_pro_code_2d_ca1.yaml",   # alt spelling
}


# ─────────────────────────────────────────────────────────────────────────── #
# Sub-dataclasses for the subject calibration block                            #
# ─────────────────────────────────────────────────────────────────────────── #

@dataclass(frozen=True)
class HeadCorrection:
    """Head-rotation calibration angles in radians.

    Mirrors the MATLAB ``subject.correction`` struct.

    Attributes
    ----------
    theta_phase:
        Phase offset (rad) used for spike-phase analysis.
    head_yaw, head_body, head_roll:
        Rotation corrections (rad) applied during marker registration.
    head_center:
        ``[x, y]`` translational offset.
    """

    theta_phase: float = 0.0
    head_yaw:    float = 0.0
    head_body:   float = 0.0
    head_roll:   float = 0.0
    head_center: tuple[float, float] = (0.0, 0.0)


@dataclass(frozen=True)
class ChannelGroup:
    """LFP channel groupings for a recording.

    Mirrors ``subject.channelGroup``.  All channel indices are
    **1-based** to match the MATLAB convention; convert to 0-based
    with :meth:`as_zero_based` when calling Python LFP utilities.

    Attributes
    ----------
    theta:
        Theta-reference channel (single).
    thetarc:
        ``[lo, hi]`` channel range used for theta phase reconstruction.
    ripple:
        List of channels in the ripple-detection group.
    """

    theta:   Optional[int]            = None
    thetarc: Optional[tuple[int, int]] = None
    ripple:  Optional[list[int]]      = None

    def as_zero_based(self) -> "ChannelGroup":
        """Return a copy with all 1-based channel indices shifted to 0-based."""
        return ChannelGroup(
            theta   = (self.theta - 1) if self.theta is not None else None,
            thetarc = ((self.thetarc[0] - 1, self.thetarc[1] - 1)
                       if self.thetarc is not None else None),
            ripple  = ([c - 1 for c in self.ripple]
                       if self.ripple is not None else None),
        )


@dataclass(frozen=True)
class AnatLocation:
    """Anatomical-region flags for a recording.

    Mirrors ``subject.anat_loc``.  Multiple flags can be True at once
    (some recordings span CA1 and CA3 with different shanks).
    """

    ca1: bool = False
    ca3: bool = False
    dg:  bool = False

    @property
    def is_ca1(self) -> bool:
        return self.ca1

    @property
    def is_ca3(self) -> bool:
        return self.ca3

    @property
    def is_dg(self) -> bool:
        return self.dg

    def regions(self) -> list[str]:
        """Return the names of all True regions."""
        names = []
        if self.ca1: names.append("CA1")
        if self.ca3: names.append("CA3")
        if self.dg:  names.append("DG")
        return names


@dataclass(frozen=True)
class SubjectInfo:
    """Per-trial subject calibration data.

    Mirrors the MATLAB ``Sessions(i).subject`` struct.

    Attributes
    ----------
    name:
        Subject identifier (e.g. ``'jg05'``, ``'er01'``).
    correction:
        Head-rotation calibration.
    channel_group:
        LFP channel groupings.
    anat_loc:
        Anatomical-region flags.
    """

    name:           str
    correction:     HeadCorrection
    channel_group:  ChannelGroup
    anat_loc:       AnatLocation


# ─────────────────────────────────────────────────────────────────────────── #
# Per-session container                                                        #
# ─────────────────────────────────────────────────────────────────────────── #

@dataclass(frozen=True)
class TrialSpec:
    """A single session/trial metadata entry from a session list.

    Bundles together:

    * The ``pipelines``-compatible ``spec`` dict — pass this to
      :func:`neurobox.pipelines.quick_session_setup`.
    * The :class:`SubjectInfo` — calibration data not used by
      ``pipelines`` but needed for downstream analysis (spike-phase
      computation, channel selection).
    * The flat ``offsets``, ``x_offset``, ``y_offset``, ``z_offset``
      maze-alignment constants — these are MTA-specific and don't go
      into ``pipelines``.
    """

    session_name:     str
    maze_name:        str
    trial_name:       str
    subject:          SubjectInfo
    spec:             dict[str, Any] = field(default_factory=dict)

    # Maze-frame translational offsets (mm).  Not used by pipelines,
    # but used downstream for cross-session maze coregistration.
    x_offset:         float = 0.0
    y_offset:         float = 0.0
    z_offset:         float = 0.0

    # Time-domain offsets [start_trim, end_trim] in samples
    offsets:          tuple[int, int] = (0, 0)

    @property
    def full_name(self) -> str:
        """Canonical ``<session>.<maze>.<trial>`` string."""
        return f"{self.session_name}.{self.maze_name}.{self.trial_name}"

    @property
    def subjects(self) -> list[str]:
        """Multi-subject list (rare; usually ``[self.subject.name]``)."""
        return [self.subject.name]


# ─────────────────────────────────────────────────────────────────────────── #
# Top-level container                                                          #
# ─────────────────────────────────────────────────────────────────────────── #

class SessionList(Sequence[TrialSpec]):
    """Sequence of :class:`TrialSpec` loaded from a YAML file.

    Behaves like a regular sequence — supports ``len()``, integer
    indexing, slicing, and iteration — plus name-based lookup via
    string indexing (returns the **first** trial whose
    ``session_name``, ``full_name``, or partial-match string equals
    the key).

    Built via :func:`load_session_list`; not normally instantiated
    directly.
    """

    def __init__(self, name: str, trials: list[TrialSpec], source: Path | None = None):
        self.name:     str       = name
        self._trials:  list[TrialSpec] = list(trials)
        self.source:   Optional[Path]  = source

    # ── Sequence protocol ────────────────────────────────────────────── #

    def __len__(self) -> int:
        return len(self._trials)

    def __iter__(self) -> Iterator[TrialSpec]:
        return iter(self._trials)

    def __getitem__(self, key) -> Union[TrialSpec, "SessionList"]:
        if isinstance(key, int):
            return self._trials[key]
        if isinstance(key, slice):
            return SessionList(self.name, self._trials[key], self.source)
        if isinstance(key, str):
            # Try exact match on session_name, then full_name, then prefix
            for t in self._trials:
                if t.session_name == key or t.full_name == key:
                    return t
            for t in self._trials:
                if t.session_name.startswith(key):
                    return t
            raise KeyError(f"No session matching {key!r} in {self.name!r}")
        raise TypeError(f"Invalid index type: {type(key).__name__}")

    def __repr__(self) -> str:
        return f"SessionList({self.name!r}, n={len(self)})"

    # ── Convenience ──────────────────────────────────────────────────── #

    @property
    def session_names(self) -> list[str]:
        return [t.session_name for t in self._trials]

    @property
    def subjects(self) -> list[str]:
        """Sorted list of unique subject names across all trials."""
        return sorted({t.subject.name for t in self._trials})

    def filter(self, predicate) -> "SessionList":
        """Return a new SessionList containing trials matching ``predicate``."""
        return SessionList(
            self.name,
            [t for t in self._trials if predicate(t)],
            self.source,
        )

    def by_subject(self, subject: str) -> "SessionList":
        return self.filter(lambda t: t.subject.name == subject)

    def by_region(self, region: str) -> "SessionList":
        """Return trials with the given anatomical region flag set."""
        r = region.upper()
        return self.filter(lambda t: r in t.subject.anat_loc.regions())

    def specs(self) -> list[dict]:
        """Return ``pipelines``-compatible spec dicts for every trial."""
        return [t.spec for t in self._trials]


# ─────────────────────────────────────────────────────────────────────────── #
# YAML → dataclasses                                                           #
# ─────────────────────────────────────────────────────────────────────────── #

_PLACEHOLDER_RE = re.compile(r"\{(xyz_root|nlx_root|ephys_root)\}")


def _resolve_placeholders(value: Any, subs: dict[str, str]) -> Any:
    """Recursively replace ``{xyz_root}`` / ``{nlx_root}`` placeholders."""
    if isinstance(value, str):
        def replace(m):
            key = m.group(1)
            # ephys_root is an alias for nlx_root
            if key == "ephys_root":
                key = "nlx_root"
            return subs.get(key, m.group(0))
        return _PLACEHOLDER_RE.sub(replace, value)
    if isinstance(value, dict):
        return {k: _resolve_placeholders(v, subs) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_placeholders(x, subs) for x in value]
    return value


def _resolve_data_roots(
    xyz_root: Optional[str | Path],
    nlx_root: Optional[str | Path],
) -> dict[str, str]:
    """Resolve xyz_root / nlx_root from args, then env, then None."""
    out = {}
    out["xyz_root"] = (str(xyz_root) if xyz_root is not None
                       else os.environ.get("NB_XYZ_ROOT", ""))
    out["nlx_root"] = (str(nlx_root) if nlx_root is not None
                       else os.environ.get("NB_NLX_ROOT", ""))
    return out


def _build_subject_info(raw_subject: dict | None) -> SubjectInfo:
    """Map the raw YAML subject dict to a SubjectInfo."""
    raw = raw_subject or {}
    corr_raw = raw.get("correction", {}) or {}
    chan_raw = raw.get("channelGroup", {}) or {}
    anat_raw = raw.get("anat_loc", {}) or {}

    # head_center may be a 2-list or a flat sequence
    hc = corr_raw.get("headCenter", (0.0, 0.0))
    if isinstance(hc, (list, tuple)) and len(hc) >= 2:
        head_center = (float(hc[0]), float(hc[1]))
    else:
        head_center = (0.0, 0.0)

    correction = HeadCorrection(
        theta_phase = float(corr_raw.get("thetaPhase", 0.0) or 0.0),
        head_yaw    = float(corr_raw.get("headYaw",    0.0) or 0.0),
        head_body   = float(corr_raw.get("headBody",   0.0) or 0.0),
        head_roll   = float(corr_raw.get("headRoll",   0.0) or 0.0),
        head_center = head_center,
    )

    # thetarc may be a 2-list — coerce to tuple
    thetarc = chan_raw.get("thetarc")
    if thetarc is not None and len(thetarc) >= 2:
        thetarc = (int(thetarc[0]), int(thetarc[1]))
    else:
        thetarc = None
    ripple = chan_raw.get("ripple")
    if ripple is not None:
        ripple = [int(c) for c in ripple]
    theta = chan_raw.get("theta")
    if theta is not None:
        theta = int(theta)

    channel_group = ChannelGroup(theta=theta, thetarc=thetarc, ripple=ripple)
    anat_loc = AnatLocation(
        ca1 = bool(anat_raw.get("CA1", False)),
        ca3 = bool(anat_raw.get("CA3", False)),
        dg  = bool(anat_raw.get("DG",  False)),
    )

    return SubjectInfo(
        name           = str(raw.get("name", "")),
        correction     = correction,
        channel_group  = channel_group,
        anat_loc       = anat_loc,
    )


def _build_pipeline_spec(raw: dict) -> dict[str, Any]:
    """Extract a pipelines-compatible spec dict from a raw session dict.

    The pipelines layer accepts both camelCase and snake_case keys (see
    :func:`neurobox.pipelines._normalise_spec`), so we pass the
    MATLAB-style camelCase keys through unchanged for the fields that
    pipelines recognises.  Subject/anat data is *not* part of the
    spec dict — those live in :class:`SubjectInfo`.
    """
    pipeline_keys = (
        "sessionName", "mazeName", "trialName", "dataLoggers",
        "ttlValue", "stopTtl", "xyzSampleRate", "syncChannel",
        "dropSyncInd", "includeSyncInd", "stcMode", "projectId",
        "dataRoot", "offsets",
    )
    out: dict[str, Any] = {}
    for k in pipeline_keys:
        if k in raw:
            out[k] = raw[k]
    # Map dLoggers (MATLAB) → dataLoggers (pipelines)
    if "dLoggers" in raw and "dataLoggers" not in out:
        out["dataLoggers"] = raw["dLoggers"]
    # Map TTLValue → ttlValue
    if "TTLValue" in raw and "ttlValue" not in out:
        out["ttlValue"] = raw["TTLValue"]
    # dPaths is MTA-specific — not used by pipelines, but useful enough
    # to expose at the top of the spec.
    if "dPaths" in raw:
        out["dPaths"] = raw["dPaths"]
    # mazes / project / hostServer / dataServer carry through as-is for
    # users who consult them directly.
    for k in ("project", "hostServer", "dataServer"):
        if k in raw:
            out[k] = raw[k]
    return out


def _build_trial_spec(raw: dict) -> TrialSpec:
    subject = _build_subject_info(raw.get("subject"))
    spec    = _build_pipeline_spec(raw)
    offsets = raw.get("offsets", (0, 0))
    if not isinstance(offsets, (list, tuple)) or len(offsets) < 2:
        offsets = (0, 0)
    return TrialSpec(
        session_name = str(raw["sessionName"]),
        maze_name    = str(raw.get("mazeName",  "cof")),
        trial_name   = str(raw.get("trialName", "all")),
        subject      = subject,
        spec         = spec,
        x_offset     = float(raw.get("xOffset", 0.0) or 0.0),
        y_offset     = float(raw.get("yOffset", 0.0) or 0.0),
        z_offset     = float(raw.get("zOffset", 0.0) or 0.0),
        offsets      = (int(offsets[0]), int(offsets[1])),
    )


# ─────────────────────────────────────────────────────────────────────────── #
# Public loader functions                                                      #
# ─────────────────────────────────────────────────────────────────────────── #

def available_session_lists() -> list[str]:
    """Names of YAML session lists bundled with the package.

    Returns
    -------
    list[str]
        Canonical names, e.g. ``['BehaviorPlaceCode', 'EgoProCode2D',
        'EgoProCode2D_CA3', 'EgoProCode2d_CA1']``.  Aliases (e.g.
        ``EgoProCode2D_CA1``) are not returned.
    """
    seen: set[str] = set()
    out:  list[str] = []
    for name, fname in _BUILTIN_NAMES.items():
        if fname in seen:
            continue
        seen.add(fname)
        out.append(name)
    return out


def load_session_list(
    name_or_path: str | Path,
    *,
    xyz_root: Optional[str | Path] = None,
    nlx_root: Optional[str | Path] = None,
) -> SessionList:
    """Load a session-list YAML by built-in name or filesystem path.

    Parameters
    ----------
    name_or_path:
        Either a built-in name (``'BehaviorPlaceCode'``,
        ``'EgoProCode2D'``, ``'EgoProCode2D_CA3'``, ``'EgoProCode2d_CA1'``)
        or a path to a YAML file with the same schema.
    xyz_root:
        Replacement for the ``{xyz_root}`` placeholder in ``dPaths.xyz``.
        Falls back to the ``NB_XYZ_ROOT`` environment variable, then to
        an empty string (so unresolved placeholders surface as obvious
        path errors at use time).
    nlx_root:
        Replacement for ``{nlx_root}``.  Falls back to ``NB_NLX_ROOT``.

    Returns
    -------
    :class:`SessionList`
    """
    # Resolve to a path
    name_or_path = str(name_or_path)
    if name_or_path in _BUILTIN_NAMES:
        path = _BUILTIN_DIR / _BUILTIN_NAMES[name_or_path]
        canonical_name = name_or_path
    else:
        path = Path(name_or_path)
        if not path.exists():
            avail = ", ".join(available_session_lists())
            raise FileNotFoundError(
                f"No session list named or found at {name_or_path!r}.  "
                f"Available built-ins: {avail}"
            )
        canonical_name = path.stem

    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict) or "sessions" not in raw:
        raise ValueError(
            f"YAML at {path} must be a mapping with a 'sessions' key"
        )

    subs = _resolve_data_roots(xyz_root, nlx_root)
    raw_sessions = _resolve_placeholders(raw["sessions"], subs)
    if not isinstance(raw_sessions, list):
        raise ValueError(f"'sessions' must be a list; got {type(raw_sessions).__name__}")

    trials = [_build_trial_spec(s) for s in raw_sessions]

    return SessionList(
        name   = raw.get("name", canonical_name),
        trials = trials,
        source = path,
    )
