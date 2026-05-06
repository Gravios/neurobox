#!/usr/bin/env python3
"""Convert MTA get_session_list_v3.m to sessions.json.  See docstring below."""

from __future__ import annotations
__doc__ = """
convert_session_list.py
=======================
Convert MTA get_session_list_v3.m to sessions.json.

Usage
-----
    python scripts/convert_session_list.py \\
        /path/to/get_session_list_v3.m sessions.json

What this produces
------------------
    {
      "lists": {
        "BehaviorPlaceCode": [ {session}, {session}, ... ],
        ...
      }
    }

Each entry is a fully-expanded dict with the ``Sessions(end+1) = Sessions(end)``
inheritance chain resolved and MATLAB math/range expressions evaluated.
The server-specific path fields (xyz_host, nlx_host, dPaths, hostServer,
dataServer) are dropped — these are replaced by the .env file and
NBSessionPaths in neurobox.
"""


import json
import math
import re
import sys
from pathlib import Path
from copy import deepcopy
from typing import Any


# ---------------------------------------------------------------------------
# MATLAB expression evaluator
# ---------------------------------------------------------------------------

def _eval_matlab(expr: str) -> Any:
    expr = expr.strip()
    if not expr:
        return None
    if expr in ("true", "True"):
        return True
    if expr in ("false", "False"):
        return False
    m = re.match(r"^'([^']*)'$", expr)
    if m:
        return m.group(1)
    try:
        return int(expr)
    except ValueError:
        pass
    try:
        return float(expr)
    except ValueError:
        pass
    # Math with pi
    try:
        val = eval(expr, {"__builtins__": {}},
                   {"pi": math.pi, "exp": math.exp, "sqrt": math.sqrt,
                    "sin": math.sin, "cos": math.cos})
        return round(float(val), 8)
    except Exception:
        pass
    # MATLAB range in brackets or bare
    if re.match(r"^\[.*\]$", expr):
        inner = expr[1:-1].strip()
        if ":" in inner:
            return _eval_range(inner)
    if ":" in expr and not expr.startswith("'"):
        return _eval_range(expr)
    # Cell array  {{'a','b'}}  or  {'a','b'}
    m = re.match(r"^\{+(.+?)\}+$", expr, re.DOTALL)
    if m:
        items = _split_top(m.group(1))
        out = []
        for it in items:
            it = it.strip()
            sm = re.match(r"^'([^']*)'$", it)
            if sm:
                out.append(sm.group(1))
        return out if out else None
    # Array literal [a,b,c] or [a b c]
    m = re.match(r"^\[(.+)\]$", expr, re.DOTALL)
    if m:
        inner = m.group(1).replace(";", ",")
        items = _split_top(inner)
        vals = [_eval_matlab(i.strip()) for i in items if i.strip()]
        return vals
    if expr in ("[]", "{}"):
        return []
    return expr  # return as-is (string)


def _eval_range(expr: str) -> list:
    parts = [p.strip() for p in expr.split(":")]
    try:
        if len(parts) == 2:
            return list(range(int(parts[0]), int(parts[1]) + 1))
        if len(parts) == 3:
            return list(range(int(parts[0]), int(parts[2]) + 1, int(parts[1])))
    except (ValueError, TypeError):
        pass
    return []


def _split_top(s: str, sep: str = ",") -> list[str]:
    depth = 0
    buf: list[str] = []
    items: list[str] = []
    for ch in s:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == sep and depth == 0:
            items.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    if buf:
        items.append("".join(buf))
    return items


def _parse_struct_args(args: str) -> dict:
    items = _split_top(args)
    result: dict = {}
    i = 0
    while i + 1 < len(items):
        key_m = re.match(r"^'([^']+)'$", items[i].strip())
        if key_m:
            val_str = items[i + 1].strip()
            val = _eval_matlab(val_str)
            # Recurse for nested struct(...)
            nm = re.match(r"^struct\s*\((.+)\)$", val_str, re.DOTALL)
            if nm:
                val = _parse_struct_args(nm.group(1))
            result[key_m.group(1)] = val
        i += 2
    return result


# ---------------------------------------------------------------------------
# Field normaliser
# ---------------------------------------------------------------------------

_DROP = {
    "xyz_host", "nlx_host", "dPaths", "hostServer", "dataServer",
    "project", "host", "local", "overwrite", "csv", "path", "meta",
    "sessionBase", "primarySubject", "subjects", "rippleDetectionChannels",
}

_RENAME = {
    "TTLValue":        "ttlValue",
    "dLoggers":        "dataLoggers",
    "xOffSet":         "xOffset",
    "yOffSet":         "yOffset",
    "zOffSet":         "zOffset",
}


def _normalise(raw: dict) -> dict:
    out: dict = {}
    for k, v in raw.items():
        if k in _DROP:
            continue
        out[_RENAME.get(k, k)] = v

    out.setdefault("mazeName",        "cof")
    out.setdefault("trialName",       "all")
    out.setdefault("dataLoggers",     ["nlx", "vicon"])
    out.setdefault("ttlValue",        "0x0040")
    out.setdefault("includeSyncInd",  [])
    out.setdefault("offsets",         [0.0, 0.0])
    out.setdefault("xOffset",         0.0)
    out.setdefault("yOffset",         0.0)
    out.setdefault("zOffset",         0.0)
    out.setdefault("rotation",        0.0)
    out.setdefault("stcMode",         "default")
    out.setdefault("thetaRef",        [])
    out.setdefault("thetaRefGeneral", None)

    if isinstance(out.get("dataLoggers"), str):
        out["dataLoggers"] = [out["dataLoggers"]]
    if out.get("offsets") in ([], None):
        out["offsets"] = [0.0, 0.0]
    elif isinstance(out["offsets"], (int, float)):
        out["offsets"] = [float(out["offsets"]), 0.0]
    elif isinstance(out["offsets"], list) and len(out["offsets"]) == 2:
        out["offsets"] = [float(x) for x in out["offsets"]]

    xsr = out.get("xyzSampleRate", 0)
    out["xyzSampleRate"] = float(xsr) if xsr else None

    if "subject" in out and isinstance(out["subject"], dict):
        _normalise_subject(out["subject"])

    return out


def _normalise_subject(s: dict) -> None:
    for k in list(s):
        if isinstance(s[k], dict):
            _normalise_subject(s[k])
    if "anat_loc" in s:
        s["anatLoc"] = s.pop("anat_loc")


# ---------------------------------------------------------------------------
# Case-block parser
# ---------------------------------------------------------------------------

def _strip_comment(line: str) -> str:
    in_str = False
    for i, ch in enumerate(line):
        if ch == "'":
            in_str = not in_str
        elif ch == "%" and not in_str:
            return line[:i]
    return line


def _split_cases(source: str) -> dict[str, str]:
    cases: dict[str, str] = {}
    cur_name: str | None = None
    cur_lines: list[str] = []
    for line in source.splitlines():
        s = _strip_comment(line).rstrip()
        m = re.match(r"^\s*case\s+'([^']+)'", s)
        if m:
            if cur_name:
                cases[cur_name] = "\n".join(cur_lines)
            cur_name  = m.group(1)
            cur_lines = []
            continue
        if re.match(r"^\s*(otherwise|end)\b", s):
            if cur_name:
                cases[cur_name] = "\n".join(cur_lines)
            cur_name  = None
            cur_lines = []
            continue
        if cur_name:
            cur_lines.append(line)
    return cases


def _eval_case(body: str) -> list[dict]:
    sessions: list[dict] = []

    # Remove $$$-commented blocks
    body = re.sub(r"%\s*\$\$\$.*?(?=\n)", "", body)

    # Strip comments and blank lines, join continuation lines
    clean = []
    for raw in body.splitlines():
        line = _strip_comment(raw).rstrip()
        if line.strip():
            clean.append(line)

    joined: list[str] = []
    buf = ""
    for line in clean:
        if line.rstrip().endswith("..."):
            buf += " " + line.rstrip()[:-3]
        else:
            buf += " " + line
            joined.append(buf.strip())
            buf = ""
    if buf.strip():
        joined.append(buf.strip())

    for stmt in joined:
        stmt = stmt.strip().rstrip(";")

        # Sessions(1) = struct(...)  or  Sessions(end+1) = struct(...)
        m = re.match(r"Sessions\s*\(\s*(?:1|end\s*\+\s*1?)\s*\)\s*=\s*(.+)$",
                     stmt, re.DOTALL)
        if m:
            rhs = m.group(1).strip()
            parsed = _parse_rhs(rhs, sessions)
            if parsed is not None:
                sessions.append(_normalise(parsed))
            continue

        # Sessions(end+1) = Sessions(end)  or  = Sessions(1)
        if re.match(r"Sessions\s*\(end\+1?\)\s*=\s*Sessions\s*\(\w+\)", stmt):
            if sessions:
                sessions.append(deepcopy(sessions[-1]))
            continue

        # Sessions(end) = Sessions(end)  (uncommon but present)
        if re.match(r"Sessions\s*\(end\)\s*=\s*Sessions\s*\(\w+\)", stmt):
            if sessions:
                sessions.append(deepcopy(sessions[-1]))
            continue

        # Sessions(end).field.subfield = value
        m3 = re.match(r"Sessions\s*\(end\)\s*\.(\w+(?:\.\w+)*)\s*=\s*(.+)$",
                       stmt, re.DOTALL)
        if m3 and sessions:
            path = m3.group(1).split(".")
            val  = _eval_matlab(m3.group(2).strip())
            _set_nested(sessions[-1], path, val)
            sessions[-1] = _normalise(sessions[-1])

    return sessions


def _parse_rhs(rhs: str, sessions: list[dict]) -> dict | None:
    rhs = rhs.strip().rstrip(";")
    m = re.match(r"^struct\s*\((.+)\)$", rhs, re.DOTALL)
    if m:
        return _parse_struct_args(m.group(1))
    if re.match(r"^Sessions\s*\(1\)$", rhs) and sessions:
        return deepcopy(sessions[0])
    if re.match(r"^Sessions\s*\(end\)$", rhs) and sessions:
        return deepcopy(sessions[-1])
    return None


def _set_nested(d: dict, path: list[str], val: Any) -> None:
    for key in path[:-1]:
        if key not in d or not isinstance(d[key], dict):
            d[key] = {}
        d = d[key]
    d[path[-1]] = val


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _translate_session_name(
    legacy:    str,
    source_id: str = "sirotaA",
) -> str:
    """Translate a MATLAB legacy session name to neurobox 4-part naming.

    The MATLAB convention is ``<userId><subjectId>-<date>``, e.g.
    ``jg05-20120316`` (user ``jg``, subject ``05``, date ``20120316``).
    The neurobox convention (matching :class:`NBSessionPaths`) is
    ``<sourceId>-<userId>-<subjectId>-<date>``, with the four parts
    separated by single hyphens.

    Examples
    --------
    >>> _translate_session_name("jg05-20120316")
    'sirotaA-jg-05-20120316'
    >>> _translate_session_name("Ed03-20140624a")
    'sirotaA-Ed-03-20140624a'
    >>> _translate_session_name("ER06-20130612", source_id="evgenyB")
    'evgenyB-ER-06-20130612'

    Parameters
    ----------
    legacy:
        Original MATLAB session name.
    source_id:
        Source/lab identifier to use as the new first component.
        Defaults to ``'sirotaA'`` since most labbox/MTA data
        originated in the Sirota lab.

    Returns
    -------
    str
        4-part neurobox name.  If *legacy* doesn't match the
        ``<letters><digits>-<date>`` pattern (e.g. it's already in
        4-part form, or some other layout), it's returned unchanged.
    """
    m = re.match(r"^([A-Za-z]+)(\d+)-(\d{8}[a-zA-Z]*)$", legacy)
    if not m:
        return legacy           # leave unrecognised names alone
    user, subj, date = m.group(1), m.group(2), m.group(3)
    return f"{source_id}-{user}-{subj}-{date}"


def _apply_neurobox_naming(
    lists:     dict[str, list[dict]],
    source_id: str = "sirotaA",
) -> tuple[dict[str, list[dict]], int]:
    """Rewrite every ``sessionName`` in *lists* using
    :func:`_translate_session_name`.

    Returns the rewritten dict and the count of names that actually
    changed.  Operates on a deep-copied structure; the input is left
    untouched.
    """
    import copy
    out = copy.deepcopy(lists)
    changed = 0
    for entries in out.values():
        for s in entries:
            old = s.get("sessionName")
            if not isinstance(old, str):
                continue
            new = _translate_session_name(old, source_id=source_id)
            if new != old:
                s["sessionName"] = new
                changed += 1
    return out, changed


# ─────────────────────────────────────────────────────────────────── #
# Subject de-duplication                                                #
# ─────────────────────────────────────────────────────────────────── #
#
# The MATLAB get_session_list_v3.m embeds an entire ``subject`` struct
# into every session entry.  Most fields under that struct are
# properties of the animal and don't change between recordings:
#
#   subject.name                           (always stable)
#   subject.correction.thetaPhase          (stable for most subjects)
#   subject.correction.headBody            (stable)
#   subject.correction.headRoll            (stable)
#
# Other fields legitimately do vary per-session:
#
#   subject.correction.headYaw, headCenter (depends on rigid-body mount)
#   subject.channelGroup.{theta,thetarc,ripple}  (depends on probe location)
#   subject.anatLoc.{CA1,CA3,DG}           (depends on probe location)
#
# When ``--lift-subjects`` is set, the converter:
#   1. Builds a top-level ``subjects`` dict, keyed by subject.name,
#      holding only the keys that are stable across every session of
#      that subject.
#   2. Replaces ``subject: {...struct...}`` in each session entry
#      with the bare string ``subject: "<name>"``.
#   3. Promotes the per-session-varying keys to top-level fields on
#      the session (``correction``, ``channelGroup``, ``anatLoc``).
#
# Effective values at lookup-time are the merge:
#
#   effective.correction = subjects[s.subject].correction | s.correction
#   (session wins on conflict)


_LIFTABLE_SECTIONS = ("correction", "channelGroup", "anatLoc")


def _lift_subjects(
    lists: dict[str, list[dict]],
) -> tuple[dict[str, dict], dict[str, list[dict]], dict[str, int]]:
    """Extract per-subject constants from session entries.

    Returns
    -------
    subjects:
        Mapping ``{subject_name: {name, correction?, channelGroup?,
        anatLoc?}}``.  Sub-dicts only contain keys that were stable
        across all of that subject's sessions.
    new_lists:
        Same shape as input ``lists``, but each session has:
          * ``subject``  → bare string (subject name)
          * ``correction`` / ``channelGroup`` / ``anatLoc`` → top-level
            on the session, holding only the per-session keys that
            *don't* have a stable subject default.
    stats:
        Diagnostic counts: ``{n_subjects, n_sessions, n_with_extra,
        n_total_lifted_keys}``.
    """
    import copy
    from collections import defaultdict

    by_subject: dict[str, list[dict]] = defaultdict(list)
    for entries in lists.values():
        for s in entries:
            subj = s.get("subject", {})
            if isinstance(subj, dict) and subj.get("name"):
                by_subject[subj["name"]].append(s)

    # Build subjects dict + record which keys are stable per subject
    subjects: dict[str, dict] = {}
    stable_keys: dict[str, dict[str, set[str]]] = {}
    for name, entries in by_subject.items():
        subj_default: dict = {"name": name}
        stable_keys[name] = {sec: set() for sec in _LIFTABLE_SECTIONS}
        for section in _LIFTABLE_SECTIONS:
            section_keys: set[str] = set()
            for e in entries:
                sec = e.get("subject", {}).get(section)
                if isinstance(sec, dict):
                    section_keys.update(sec.keys())
            stable: dict = {}
            for k in section_keys:
                vals: list = []
                for e in entries:
                    v = e.get("subject", {}).get(section, {}).get(k)
                    if v is not None:
                        try:
                            vals.append(json.dumps(v))
                        except TypeError:
                            vals.append(str(v))
                # Stable means: every session had this value present
                # AND all values were identical
                if len(vals) == len(entries) and len(set(vals)) == 1:
                    stable[k] = entries[0]["subject"].get(
                        section, {},
                    )[k]
                    stable_keys[name][section].add(k)
            if stable:
                subj_default[section] = stable
        subjects[name] = subj_default

    # Build the rewritten sessions list
    new_lists: dict[str, list[dict]] = {}
    n_sessions = 0
    n_with_extra = 0
    n_total_lifted_keys = 0
    for list_name, entries in lists.items():
        new_entries = []
        for s in entries:
            n_sessions += 1
            out = {k: v for k, v in s.items() if k != "subject"}
            subj_dict = s.get("subject", {})
            subj_name = (subj_dict.get("name")
                         if isinstance(subj_dict, dict) else None)
            if subj_name is None:
                # Session without a subject — leave as-is.
                new_entries.append(out)
                continue
            out["subject"] = subj_name
            # Walk each section, keep only the keys that aren't a
            # subject-default
            had_extras = False
            for section in _LIFTABLE_SECTIONS:
                sec_session = subj_dict.get(section, {}) or {}
                if not isinstance(sec_session, dict):
                    continue
                kept = {
                    k: copy.deepcopy(v) for k, v in sec_session.items()
                    if k not in stable_keys[subj_name][section]
                }
                if kept:
                    out[section] = kept
                    had_extras = True
                    n_total_lifted_keys += len(kept)
            if had_extras:
                n_with_extra += 1
            new_entries.append(out)
        new_lists[list_name] = new_entries

    return subjects, new_lists, {
        "n_subjects":          len(subjects),
        "n_sessions":          n_sessions,
        "n_with_extra":        n_with_extra,
        "n_total_lifted_keys": n_total_lifted_keys,
    }


def convert(
    matlab_path:     str,
    json_path:       str,
    *,
    neurobox_names:  bool = False,
    source_id:       str  = "sirotaA",
    lift_subjects:   bool = False,
) -> None:
    src = Path(matlab_path).read_text(encoding="utf-8", errors="replace")

    # Jump to the switch statement
    m = re.search(r"\bswitch\s+sessionList\b", src)
    if m:
        src = src[m.start():]

    cases = _split_cases(src)
    print(f"Found {len(cases)} case(s)")

    lists: dict[str, list[dict]] = {}
    for name, body in cases.items():
        try:
            entries = _eval_case(body)
        except Exception as exc:
            print(f"  [warn] {name}: {exc}")
            entries = []
        if entries:
            lists[name] = entries
            print(f"  {name}: {len(entries)} session(s)")

    if neurobox_names:
        lists, n_changed = _apply_neurobox_naming(lists, source_id)
        print(f"Translated {n_changed} sessionName values to neurobox "
              f"4-part naming (source_id={source_id!r}).")

    output: dict = {"lists": lists}
    if lift_subjects:
        subjects, lists2, stats = _lift_subjects(lists)
        print(
            f"Lifted {stats['n_subjects']} subjects; "
            f"{stats['n_with_extra']}/{stats['n_sessions']} sessions "
            f"have session-specific overrides "
            f"({stats['n_total_lifted_keys']} keys total)."
        )
        output = {"subjects": subjects, "lists": lists2}

    out = Path(json_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    total = sum(len(v) for v in output.get("lists", {}).values())
    print(f"\nWrote {len(output.get('lists', {}))} lists, "
          f"{total} sessions → {out}")


def _cli_main() -> None:
    """Entry point: nb-convert-sessions"""
    import argparse
    p = argparse.ArgumentParser(
        description="Convert MTA get_session_list_v3.m to sessions.json.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("matlab_file", help="Path to get_session_list_v3.m")
    p.add_argument("output_json", help="Output path for sessions.json")
    p.add_argument(
        "--neurobox-names",
        action="store_true",
        help="Translate sessionName from legacy '<user><id>-<date>' "
             "form to neurobox 4-part form '<source>-<user>-<id>-<date>'",
    )
    p.add_argument(
        "--source-id",
        default="sirotaA",
        help="Source identifier to use as the first component when "
             "translating to neurobox naming",
    )
    p.add_argument(
        "--lift-subjects",
        action="store_true",
        help="Extract per-subject constants (correction, channelGroup, "
             "anatLoc) into a top-level 'subjects' dict.  Per-session "
             "overrides become top-level fields on the session entry.",
    )
    args = p.parse_args()
    convert(
        args.matlab_file, args.output_json,
        neurobox_names = args.neurobox_names,
        source_id      = args.source_id,
        lift_subjects  = args.lift_subjects,
    )


if __name__ == "__main__":
    # Lightweight legacy invocation:  convert <in> <out>
    # Full CLI:                       see _cli_main()
    if any(arg.startswith("--") for arg in sys.argv[1:]):
        _cli_main()
    elif len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <get_session_list_v3.m> <output.json> "
              f"[--neurobox-names] [--source-id NAME] [--lift-subjects]")
        sys.exit(1)
    else:
        convert(sys.argv[1], sys.argv[2])
