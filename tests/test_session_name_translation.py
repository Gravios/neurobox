"""Tests for the legacy → neurobox session-name translation and the
relaxed date regex that supports letter-suffixed dates.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Add scripts/ to path so we can import the converter
SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))


# ─────────────────────────────────────────────────────────────────────── #
# Translation helper                                                          #
# ─────────────────────────────────────────────────────────────────────── #

class TestTranslateSessionName:
    def test_canonical_jg(self):
        from convert_session_list import _translate_session_name
        assert _translate_session_name("jg05-20120316") == \
            "sirotaA-jg-05-20120316"

    def test_canonical_uppercase_user(self):
        from convert_session_list import _translate_session_name
        assert _translate_session_name("ER06-20130612") == \
            "sirotaA-ER-06-20130612"

    def test_mixed_case_user(self):
        from convert_session_list import _translate_session_name
        assert _translate_session_name("Ed10-20140813") == \
            "sirotaA-Ed-10-20140813"

    def test_date_with_letter_suffix(self):
        from convert_session_list import _translate_session_name
        assert _translate_session_name("Ed03-20140624a") == \
            "sirotaA-Ed-03-20140624a"
        assert _translate_session_name("IF14-20190712b") == \
            "sirotaA-IF-14-20190712b"

    def test_custom_source_id(self):
        from convert_session_list import _translate_session_name
        assert _translate_session_name(
            "ER06-20130612", source_id="evgenyB",
        ) == "evgenyB-ER-06-20130612"

    def test_already_translated_left_alone(self):
        """Idempotent — running it twice doesn't double-prefix."""
        from convert_session_list import _translate_session_name
        n = "sirotaA-jg-05-20120316"
        assert _translate_session_name(n) == n

    def test_unrecognised_name_left_alone(self):
        """Names that don't fit <letters><digits>-<date> are returned
        unchanged."""
        from convert_session_list import _translate_session_name
        for n in ["", "weird", "no-dash-here", "2-leading-digits-only"]:
            assert _translate_session_name(n) == n

    def test_preserves_subject_name_field(self):
        """``subject.name`` should NOT be translated even when
        ``sessionName`` is — it's a separate identifier referenced by
        other parts of the codebase."""
        from convert_session_list import _apply_neurobox_naming
        before = {"x": [
            {"sessionName": "jg05-20120316",
             "subject": {"name": "jg05"}},
        ]}
        after, n = _apply_neurobox_naming(before)
        assert n == 1
        assert after["x"][0]["sessionName"] == "sirotaA-jg-05-20120316"
        assert after["x"][0]["subject"]["name"] == "jg05"
        # And the original isn't mutated
        assert before["x"][0]["sessionName"] == "jg05-20120316"


# ─────────────────────────────────────────────────────────────────────── #
# parse_session_name now accepts letter-suffixed dates                       #
# ─────────────────────────────────────────────────────────────────────── #

class TestParseSessionNameRelaxed:
    def test_accepts_plain_date(self):
        from neurobox.dtype.paths import parse_session_name
        assert parse_session_name("sirotaA-jg-05-20120316") == {
            "sourceId":  "sirotaA",
            "userId":    "jg",
            "subjectId": "05",
            "date":      "20120316",
        }

    def test_accepts_date_with_letter_suffix(self):
        from neurobox.dtype.paths import parse_session_name
        assert parse_session_name(
            "sirotaA-Ed-03-20140624a",
        )["date"] == "20140624a"
        assert parse_session_name(
            "sirotaA-IF-14-20190712b",
        )["date"] == "20190712b"

    def test_rejects_bad_format(self):
        from neurobox.dtype.paths import parse_session_name
        with pytest.raises(ValueError):
            parse_session_name("jg05-20120316")          # no sourceId
        with pytest.raises(ValueError):
            parse_session_name("sirotaA-jg-05")          # no date


# ─────────────────────────────────────────────────────────────────────── #
# Round-trip: data/sessions.json all parse cleanly                           #
# ─────────────────────────────────────────────────────────────────────── #

class TestSessionsJsonRoundTrip:
    """The shipped data/sessions.json should be in the new neurobox
    naming and every entry should parse via NBSessionPaths."""

    def _load(self):
        path = Path(__file__).resolve().parent.parent / "data" / "sessions.json"
        if not path.exists():
            pytest.skip(f"{path} not present in this checkout")
        return json.loads(path.read_text())

    def test_all_session_names_parse(self):
        from neurobox.dtype.paths import parse_session_name
        d = self._load()
        unique = set()
        for entries in d.get("lists", {}).values():
            for s in entries:
                if "sessionName" in s:
                    unique.add(s["sessionName"])
        assert len(unique) > 0
        for n in unique:
            parse_session_name(n)            # should not raise

    def test_no_legacy_names_remain(self):
        """No bare ``<letters><digits>-<date>`` names should be left
        in the file after translation."""
        import re
        LEGACY = re.compile(r"^[A-Za-z]+\d+-\d{8}[a-zA-Z]*$")
        d = self._load()
        leftover = []
        for entries in d.get("lists", {}).values():
            for s in entries:
                n = s.get("sessionName")
                if isinstance(n, str) and LEGACY.match(n):
                    leftover.append(n)
        assert not leftover, f"Legacy names found: {leftover[:5]}"

    def test_jg05_session_translated(self):
        d = self._load()
        names = {s.get("sessionName")
                 for entries in d["lists"].values()
                 for s in entries}
        assert "sirotaA-jg-05-20120316" in names


# ─────────────────────────────────────────────────────────────────────── #
# Subject deduplication                                                       #
# ─────────────────────────────────────────────────────────────────────── #

class TestLiftSubjects:
    def test_basic_lift_pulls_stable_keys(self):
        from convert_session_list import _lift_subjects
        # Two sessions, same subject, same headBody, different headYaw.
        # headBody → subject default; headYaw → stays on session.
        lists = {
            "ListA": [
                {"sessionName": "X-jg-05-20120316",
                 "subject": {
                     "name": "jg05",
                     "correction": {"headBody": -0.234, "headYaw": 0.1},
                 }},
                {"sessionName": "X-jg-05-20120317",
                 "subject": {
                     "name": "jg05",
                     "correction": {"headBody": -0.234, "headYaw": 0.2},
                 }},
            ],
        }
        subjects, new_lists, stats = _lift_subjects(lists)
        assert subjects["jg05"]["correction"] == {"headBody": -0.234}
        # Sessions: subject becomes a string, headBody disappears,
        # headYaw stays
        for s in new_lists["ListA"]:
            assert s["subject"] == "jg05"
            assert "headBody" not in s.get("correction", {})
            assert "headYaw" in s["correction"]
        assert stats["n_subjects"] == 1
        assert stats["n_sessions"] == 2
        assert stats["n_with_extra"] == 2

    def test_unstable_field_does_not_promote(self):
        """If a key's value differs across that subject's sessions,
        it stays on the session, NOT in the subject default."""
        from convert_session_list import _lift_subjects
        lists = {
            "ListA": [
                {"sessionName": "a", "subject": {
                    "name": "x",
                    "correction": {"headYaw": 0.1},
                }},
                {"sessionName": "b", "subject": {
                    "name": "x",
                    "correction": {"headYaw": 0.2},
                }},
            ],
        }
        subjects, new_lists, _ = _lift_subjects(lists)
        # headYaw was unstable → no subject default for it
        assert "correction" not in subjects["x"]
        # And it stays on every session
        for s in new_lists["ListA"]:
            assert "headYaw" in s["correction"]

    def test_missing_subject_field_blocks_promotion(self):
        """If the key is present in some sessions but missing in
        others, it CAN'T be a subject default — promote conservatively."""
        from convert_session_list import _lift_subjects
        lists = {
            "ListA": [
                {"sessionName": "a", "subject": {
                    "name": "x", "correction": {"headBody": 1.0}}},
                {"sessionName": "b", "subject": {"name": "x"}},
            ],
        }
        subjects, _, _ = _lift_subjects(lists)
        assert "correction" not in subjects["x"]

    def test_session_without_subject_left_alone(self):
        from convert_session_list import _lift_subjects
        lists = {
            "ListA": [
                {"sessionName": "a"},        # no subject at all
            ],
        }
        subjects, new_lists, stats = _lift_subjects(lists)
        assert subjects == {}
        assert new_lists["ListA"][0] == {"sessionName": "a"}
        assert stats["n_subjects"] == 0

    def test_round_trip_through_effective_session(self):
        """After lift, merging subject + session via
        effective_session reproduces the original subject struct."""
        from convert_session_list import _lift_subjects
        from neurobox.config.sessions_json import effective_session

        original_subject = {
            "name": "jg05",
            "correction": {
                "thetaPhase": 0.78539816,
                "headBody":   -0.234,
                "headRoll":   -0.365,
                "headYaw":     0.264,
                "headCenter":  [0, 0],
            },
            "channelGroup": {"theta": 68},
            "anatLoc":      {"CA3": False},
        }
        lists = {
            "L": [
                {"sessionName": "a", "subject": original_subject},
                {"sessionName": "b", "subject": {**original_subject,
                    "correction": {**original_subject["correction"],
                                    "headYaw": -0.264}}},
            ],
        }
        subjects, new_lists, _ = _lift_subjects(lists)
        # Session 'a' merged should reproduce original
        merged = effective_session(new_lists["L"][0], subjects)
        assert merged["subject"]["name"] == "jg05"
        for k, v in original_subject["correction"].items():
            assert merged["subject"]["correction"][k] == v
        assert merged["subject"]["channelGroup"] == {"theta": 68}
        assert merged["subject"]["anatLoc"] == {"CA3": False}


# ─────────────────────────────────────────────────────────────────────── #
# SessionsCatalog loader                                                      #
# ─────────────────────────────────────────────────────────────────────── #

class TestSessionsCatalog:
    def test_load_default(self):
        from neurobox.config.sessions_json import load_sessions_json
        cat = load_sessions_json()
        assert len(cat) > 0
        assert "BehaviorPlaceCode" in cat.list_names()
        assert "jg05" in cat.subjects()

    def test_get_returns_merged_session(self):
        from neurobox.config.sessions_json import load_sessions_json
        cat = load_sessions_json()
        s = cat.get("BehaviorPlaceCode", "sirotaA-jg-05-20120316")
        assert s is not None
        assert s.subject_name == "jg05"
        # Subject defaults appear in the merged correction
        assert s.correction["thetaPhase"] == 0.78539816
        assert s.correction["headBody"]   == -0.234
        # Session-specific overrides also appear
        assert s.correction["headYaw"] == 0.264

    def test_get_unknown_returns_none(self):
        from neurobox.config.sessions_json import load_sessions_json
        cat = load_sessions_json()
        assert cat.get("NoSuchList", "anything") is None
        assert cat.get("BehaviorPlaceCode", "no-such-session") is None

    def test_find_across_lists(self):
        """Some sessions appear in multiple lists — find() returns
        all hits."""
        from neurobox.config.sessions_json import load_sessions_json
        cat = load_sessions_json()
        hits = cat.find("sirotaA-jg-05-20120316")
        # At least one match (BehaviorPlaceCode should always have it)
        assert len(hits) >= 1
        list_names = {h.list_name for h in hits}
        assert "BehaviorPlaceCode" in list_names

    def test_for_subject_iterates_all_sessions(self):
        from neurobox.config.sessions_json import load_sessions_json
        cat = load_sessions_json()
        sessions = list(cat.for_subject("jg05"))
        assert len(sessions) > 10        # jg05 has many recordings
        for s in sessions:
            assert s.subject_name == "jg05"

    def test_get_subject_returns_raw(self):
        from neurobox.config.sessions_json import load_sessions_json
        cat = load_sessions_json()
        subj = cat.get_subject("jg05")
        assert subj is not None
        assert subj["name"] == "jg05"
        # jg05's stable keys
        assert "correction" in subj
        assert subj["correction"]["thetaPhase"] == 0.78539816

    def test_legacy_format_without_subjects_key(self, tmp_path):
        """Files written before --lift-subjects should still load —
        sessions just have full subject structs in-place."""
        import json
        from neurobox.config.sessions_json import load_sessions_json
        old_format = {
            "lists": {
                "X": [
                    {"sessionName": "a",
                     "subject": {
                         "name": "old",
                         "correction": {"headBody": 1.0},
                     }},
                ],
            }
        }
        p = tmp_path / "old.json"
        p.write_text(json.dumps(old_format))
        cat = load_sessions_json(p)
        s = cat.get("X", "a")
        assert s is not None
        assert s.subject_name == "old"
        assert s.correction == {"headBody": 1.0}

    def test_malformed_json_raises_value_error(self, tmp_path):
        from neurobox.config.sessions_json import load_sessions_json
        p = tmp_path / "bad.json"
        p.write_text("{this is not json")
        import pytest
        with pytest.raises(ValueError, match="Malformed JSON"):
            load_sessions_json(p)

    def test_missing_file_raises(self, tmp_path):
        from neurobox.config.sessions_json import load_sessions_json
        import pytest
        with pytest.raises(FileNotFoundError):
            load_sessions_json(tmp_path / "nonexistent.json")
