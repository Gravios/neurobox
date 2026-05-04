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
