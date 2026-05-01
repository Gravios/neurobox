"""Tests for neurobox.config.session_lists."""

from __future__ import annotations

import math
import os
from pathlib import Path
from textwrap import dedent

import numpy as np
import pytest


# ─────────────────────────────────────────────────────────────────────────── #
# Loader basics                                                                #
# ─────────────────────────────────────────────────────────────────────────── #

class TestLoaderBasics:

    def test_available_session_lists(self):
        from neurobox.config import available_session_lists
        names = available_session_lists()
        assert "BehaviorPlaceCode" in names
        assert "EgoProCode2D" in names
        assert "EgoProCode2D_CA3" in names
        assert "EgoProCode2d_CA1" in names

    def test_load_built_in(self):
        from neurobox.config import load_session_list
        sl = load_session_list("BehaviorPlaceCode")
        assert sl.name == "BehaviorPlaceCode"
        assert len(sl) == 34
        assert sl.source is not None
        assert sl.source.name == "behavior_place_code.yaml"

    def test_load_alt_spelling(self):
        """``EgoProCode2D_CA1`` (uppercase D) is an alias for ``EgoProCode2d_CA1``."""
        from neurobox.config import load_session_list
        a = load_session_list("EgoProCode2d_CA1")
        b = load_session_list("EgoProCode2D_CA1")
        assert len(a) == len(b)
        assert a.session_names == b.session_names

    def test_each_built_in_has_trials(self):
        """Sanity check: each YAML has the expected count of trials."""
        from neurobox.config import load_session_list
        expected = {
            "BehaviorPlaceCode":   34,
            "EgoProCode2D":        33,
            "EgoProCode2D_CA3":     8,
            "EgoProCode2d_CA1":    13,
        }
        for name, count in expected.items():
            sl = load_session_list(name)
            assert len(sl) == count, f"{name}: expected {count}, got {len(sl)}"

    def test_load_external_yaml(self, tmp_path):
        """A YAML at an explicit path can be loaded too."""
        from neurobox.config import load_session_list
        yaml_path = tmp_path / "custom.yaml"
        yaml_path.write_text(dedent("""\
            name: CustomList
            sessions:
              - sessionName: test01-20240101
                subjects: [test]
                subject:
                  name: test
                  correction: {thetaPhase: 0.0}
                  channelGroup: {theta: 1}
                  anat_loc: {CA1: true}
                mazeName: cof
                trialName: all
                xyzSampleRate: 120
        """))
        sl = load_session_list(yaml_path)
        assert len(sl) == 1
        assert sl[0].session_name == "test01-20240101"

    def test_missing_file_raises(self):
        from neurobox.config import load_session_list
        with pytest.raises(FileNotFoundError, match="No session list"):
            load_session_list("ThisDoesNotExist")


# ─────────────────────────────────────────────────────────────────────────── #
# Path placeholder substitution                                                #
# ─────────────────────────────────────────────────────────────────────────── #

class TestPathSubstitution:

    def test_explicit_xyz_root(self):
        from neurobox.config import load_session_list
        sl = load_session_list("BehaviorPlaceCode", xyz_root="/data/xyz",
                               nlx_root="/data/ephys")
        # All dPaths should have substituted
        for t in sl:
            xyz = t.spec.get("dPaths", {}).get("xyz", "")
            assert "{xyz_root}" not in xyz
            if xyz:
                assert xyz.startswith("/data/xyz") or xyz.startswith("/storage/")

    def test_environment_variables(self, monkeypatch):
        from neurobox.config import load_session_list
        monkeypatch.setenv("NB_XYZ_ROOT", "/env/xyz")
        monkeypatch.setenv("NB_NLX_ROOT", "/env/nlx")
        sl = load_session_list("BehaviorPlaceCode")
        # Verify env vars came through on a session that uses both roots
        t = sl["er01-20110719"]
        assert t.spec["dPaths"]["xyz"].startswith("/env/xyz")
        assert t.spec["dPaths"]["nlx"].startswith("/env/nlx")

    def test_explicit_overrides_env(self, monkeypatch):
        from neurobox.config import load_session_list
        monkeypatch.setenv("NB_XYZ_ROOT", "/env/xyz")
        sl = load_session_list("BehaviorPlaceCode", xyz_root="/explicit/xyz")
        t = sl["er01-20110719"]
        assert t.spec["dPaths"]["xyz"].startswith("/explicit/xyz")

    def test_unresolved_placeholders_remain(self):
        """If neither arg nor env is set, placeholder stays empty (path becomes
        relative)."""
        from neurobox.config import load_session_list
        # Make sure the env var isn't set
        for var in ("NB_XYZ_ROOT", "NB_NLX_ROOT"):
            os.environ.pop(var, None)
        sl = load_session_list("BehaviorPlaceCode")
        t = sl["er01-20110719"]
        # The {xyz_root} placeholder is replaced with empty string,
        # leaving e.g. '/er01' (a clearly-not-real path the user can debug)
        assert "{xyz_root}" not in t.spec["dPaths"]["xyz"]


# ─────────────────────────────────────────────────────────────────────────── #
# Sequence protocol                                                            #
# ─────────────────────────────────────────────────────────────────────────── #

class TestSequenceProtocol:

    @pytest.fixture
    def sl(self):
        from neurobox.config import load_session_list
        return load_session_list("BehaviorPlaceCode")

    def test_len(self, sl):
        assert len(sl) == 34

    def test_int_index(self, sl):
        from neurobox.config import TrialSpec
        t = sl[0]
        assert isinstance(t, TrialSpec)
        assert t.session_name == "er01-20110719"

    def test_negative_index(self, sl):
        t = sl[-1]
        assert t.session_name == "FS03-20201222"

    def test_string_index_session_name(self, sl):
        t = sl["jg05-20120316"]
        assert t.session_name == "jg05-20120316"

    def test_string_index_full_name(self, sl):
        t = sl["jg05-20120316.cof.all"]
        assert t.session_name == "jg05-20120316"

    def test_string_index_prefix(self, sl):
        # Prefix match returns the first matching session
        t = sl["jg05-2012031"]
        assert t.session_name.startswith("jg05-2012031")

    def test_string_index_missing(self, sl):
        with pytest.raises(KeyError, match="No session matching"):
            sl["doesnotexist"]

    def test_slice(self, sl):
        from neurobox.config import SessionList
        sub = sl[:3]
        assert isinstance(sub, SessionList)
        assert len(sub) == 3
        assert sub.session_names == sl.session_names[:3]

    def test_iter(self, sl):
        names = [t.session_name for t in sl]
        assert len(names) == len(sl)

    def test_subjects_property(self, sl):
        # All unique subject names, sorted
        assert sl.subjects == sorted({"er01", "ER06", "Ed10", "jg04", "jg05"})


# ─────────────────────────────────────────────────────────────────────────── #
# Filtering                                                                    #
# ─────────────────────────────────────────────────────────────────────────── #

class TestFiltering:

    @pytest.fixture
    def sl(self):
        from neurobox.config import load_session_list
        return load_session_list("BehaviorPlaceCode")

    def test_filter_predicate(self, sl):
        small = sl.filter(lambda t: t.subject.name == "jg05")
        assert all(t.subject.name == "jg05" for t in small)
        # jg05 has many sessions in BehaviorPlaceCode
        assert len(small) > 5

    def test_by_subject(self, sl):
        jg05 = sl.by_subject("jg05")
        assert len(jg05) > 0
        assert all(t.subject.name == "jg05" for t in jg05)

    def test_by_region_ca1(self, sl):
        ca1 = sl.by_region("CA1")
        assert len(ca1) > 0
        assert all(t.subject.anat_loc.is_ca1 for t in ca1)

    def test_by_region_ca3(self, sl):
        ca3 = sl.by_region("CA3")
        assert len(ca3) > 0
        assert all(t.subject.anat_loc.is_ca3 for t in ca3)


# ─────────────────────────────────────────────────────────────────────────── #
# TrialSpec / SubjectInfo content                                              #
# ─────────────────────────────────────────────────────────────────────────── #

class TestTrialSpecContents:

    @pytest.fixture
    def sl(self):
        from neurobox.config import load_session_list
        return load_session_list("BehaviorPlaceCode", xyz_root="/x", nlx_root="/n")

    def test_full_name_format(self, sl):
        t = sl["jg05-20120316"]
        assert t.full_name == "jg05-20120316.cof.all"

    def test_subjects_property(self, sl):
        t = sl["jg05-20120316"]
        assert t.subjects == ["jg05"]

    def test_subject_name(self, sl):
        t = sl["er01-20110719"]
        assert t.subject.name == "er01"

    def test_correction_values_match_matlab(self, sl):
        """Pi expressions in the MATLAB are evaluated to floats correctly."""
        t = sl["er01-20110719"]
        # MATLAB: thetaPhase = pi
        assert t.subject.correction.theta_phase == pytest.approx(math.pi)

        t = sl["ER06-20130624"]
        # MATLAB: thetaPhase = 3*pi/4
        assert t.subject.correction.theta_phase == pytest.approx(3 * math.pi / 4)

        t = sl["jg05-20120329"]
        # MATLAB: thetaPhase = pi/4
        assert t.subject.correction.theta_phase == pytest.approx(math.pi / 4)

    def test_channel_group_ranges_expanded(self, sl):
        """[1:8] etc. are expanded to explicit lists in the loader."""
        t = sl["er01-20110719"]
        assert t.subject.channel_group.theta == 8
        assert t.subject.channel_group.thetarc == (9, 16)
        assert t.subject.channel_group.ripple == list(range(17, 25))

    def test_channel_group_zero_based_conversion(self, sl):
        """The 1-based MATLAB indices can be shifted to 0-based."""
        t = sl["er01-20110719"]
        cg = t.subject.channel_group
        zg = cg.as_zero_based()
        assert zg.theta == cg.theta - 1
        assert zg.thetarc == (cg.thetarc[0] - 1, cg.thetarc[1] - 1)
        assert zg.ripple == [c - 1 for c in cg.ripple]

    def test_anat_loc_flags(self, sl):
        t = sl["er01-20110719"]
        assert t.subject.anat_loc.is_ca3
        assert not t.subject.anat_loc.is_ca1
        assert not t.subject.anat_loc.is_dg
        assert t.subject.anat_loc.regions() == ["CA3"]

    def test_offsets(self, sl):
        # ER06-20130612 has offsets=[0,0]; jg05-20120310 has [8,0]
        t = sl["jg05-20120310"]
        assert t.offsets == (8, 0)

    def test_xyz_offset(self, sl):
        # ER06-20130624 has xOffset=-20, yOffset=-20
        t = sl["ER06-20130624"]
        assert t.x_offset == -20.0
        assert t.y_offset == -20.0

    def test_immutability(self, sl):
        """Dataclasses are frozen — attempting to mutate raises."""
        from dataclasses import FrozenInstanceError
        t = sl[0]
        with pytest.raises(FrozenInstanceError):
            t.session_name = "modified"
        with pytest.raises(FrozenInstanceError):
            t.subject.correction.theta_phase = 0.0


# ─────────────────────────────────────────────────────────────────────────── #
# Pipeline-spec integration                                                    #
# ─────────────────────────────────────────────────────────────────────────── #

class TestPipelineSpec:

    @pytest.fixture
    def sl(self):
        from neurobox.config import load_session_list
        return load_session_list("BehaviorPlaceCode",
                                  xyz_root="/data/xyz", nlx_root="/data/nlx")

    def test_spec_has_required_keys(self, sl):
        t = sl[0]
        assert "sessionName" in t.spec
        assert "mazeName" in t.spec
        assert "trialName" in t.spec
        assert "dataLoggers" in t.spec     # mapped from dLoggers
        assert "ttlValue" in t.spec        # mapped from TTLValue

    def test_spec_excludes_subject_data(self, sl):
        """Subject calibration is on .subject, not .spec — pipelines doesn't need it."""
        t = sl[0]
        assert "subject" not in t.spec

    def test_specs_method(self, sl):
        specs = sl.specs()
        assert len(specs) == len(sl)
        assert all(isinstance(s, dict) and "sessionName" in s for s in specs)

    def test_pipelines_parse_spec(self, sl):
        """The spec dict round-trips through pipelines._parse_spec without error."""
        from neurobox.pipelines import _parse_spec
        for t in sl:
            norm = _parse_spec(t.spec)
            assert norm["sessionName"] == t.session_name


# ─────────────────────────────────────────────────────────────────────────── #
# All four bundled YAMLs load                                                  #
# ─────────────────────────────────────────────────────────────────────────── #

class TestAllBuiltIns:

    @pytest.mark.parametrize("name", [
        "BehaviorPlaceCode", "EgoProCode2D",
        "EgoProCode2D_CA3", "EgoProCode2d_CA1",
    ])
    def test_loads_without_error(self, name):
        from neurobox.config import load_session_list
        sl = load_session_list(name, xyz_root="/x", nlx_root="/n")
        assert len(sl) > 0

    @pytest.mark.parametrize("name", [
        "BehaviorPlaceCode", "EgoProCode2D",
        "EgoProCode2D_CA3", "EgoProCode2d_CA1",
    ])
    def test_all_subjects_have_corrections(self, name):
        """Every trial has finite correction values (no NaN, no None)."""
        from neurobox.config import load_session_list
        sl = load_session_list(name, xyz_root="/x", nlx_root="/n")
        for t in sl:
            c = t.subject.correction
            for v in (c.theta_phase, c.head_yaw, c.head_body, c.head_roll):
                assert np.isfinite(v), \
                    f"{name}/{t.session_name}: non-finite correction value"
