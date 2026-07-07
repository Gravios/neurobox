"""Tests for neurosuite-3 variant-naming path helpers and binary
file-format readers.

The neurosuite-3 *variant (chain-of-custody) naming convention*
splits per-shank artifacts into three classes with distinct
resolution rules; see
``doc/ndmanager-plugins/formats/naming.md`` in the neurosuite-3
repository for the authoritative spec.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


# ─────────────────────────────────────────────────────────────────────── #
# NBSessionPaths.ns3_class                                                   #
# ─────────────────────────────────────────────────────────────────────── #

class TestNs3Class:
    def _paths(self, tmp_path):
        from neurobox.dtype.paths import NBSessionPaths
        return NBSessionPaths(
            "sirotaA-jg-05-20120316", tmp_path, "B01", "cof",
        )

    def test_session_wide(self, tmp_path):
        p = self._paths(tmp_path)
        for t in ("fil", "dat", "xml", "yaml", "nrs", "par", "eeg", "lfp"):
            assert p.ns3_class(t) == "session_wide"

    def test_method_specific(self, tmp_path):
        p = self._paths(tmp_path)
        for t in ("clu", "clc", "clp", "fet", "pca", "col", "model", "klg"):
            assert p.ns3_class(t) == "method_specific"

    def test_shared(self, tmp_path):
        p = self._paths(tmp_path)
        assert p.ns3_class("res") == "shared"
        assert p.ns3_class("spk") == "shared"

    def test_unknown_raises(self, tmp_path):
        p = self._paths(tmp_path)
        with pytest.raises(ValueError, match="Unknown"):
            p.ns3_class("nonexistent")


# ─────────────────────────────────────────────────────────────────────── #
# NBSessionPaths.ns3_file — canonical path computation                       #
# ─────────────────────────────────────────────────────────────────────── #

class TestNs3FileNaming:
    def _paths(self, tmp_path):
        from neurobox.dtype.paths import NBSessionPaths
        return NBSessionPaths(
            "sirotaA-jg-05-20120316", tmp_path, "B01", "cof",
        )

    def test_method_specific_default_standard(self, tmp_path):
        p = self._paths(tmp_path)
        assert p.ns3_file("clu", 1).name == \
            "sirotaA-jg-05-20120316.clu.standard.1"
        assert p.ns3_file("fet", 3).name == \
            "sirotaA-jg-05-20120316.fet.standard.3"

    def test_method_specific_stderiv(self, tmp_path):
        p = self._paths(tmp_path)
        assert p.ns3_file("fet", 1, method="stderiv").name == \
            "sirotaA-jg-05-20120316.fet.stderiv.1"
        assert p.ns3_file("pca", 2, method="stderiv").name == \
            "sirotaA-jg-05-20120316.pca.stderiv.2"

    def test_shared_canonical_uses_method_tag(self, tmp_path):
        p = self._paths(tmp_path)
        # ``ns3_file`` returns the CANONICAL variant-tagged path even for
        # Shared artifacts.  Resolution / fallback is a separate concern.
        assert p.ns3_file("spk", 1).name == \
            "sirotaA-jg-05-20120316.spk.standard.1"
        assert p.ns3_file("res", 1, method="stderiv").name == \
            "sirotaA-jg-05-20120316.res.stderiv.1"

    def test_session_wide_no_shank(self, tmp_path):
        p = self._paths(tmp_path)
        assert p.ns3_file("dat").name  == "sirotaA-jg-05-20120316.dat"
        assert p.ns3_file("yaml").name == "sirotaA-jg-05-20120316.yaml"
        assert p.ns3_file("lfp").name  == "sirotaA-jg-05-20120316.lfp"

    def test_session_wide_rejects_shank(self, tmp_path):
        p = self._paths(tmp_path)
        with pytest.raises(ValueError, match="session-wide"):
            p.ns3_file("dat", shank=1)

    def test_method_specific_requires_shank(self, tmp_path):
        p = self._paths(tmp_path)
        with pytest.raises(ValueError, match="requires a shank"):
            p.ns3_file("clu")

    def test_all_files_under_processed_ephys(self, tmp_path):
        p = self._paths(tmp_path)
        for path in [
            p.ns3_file("dat"),  p.ns3_file("yaml"),
            p.ns3_file("clu", 1), p.ns3_file("spk", 1),
            p.ns3_file("fet", 1, method="stderiv"),
        ]:
            assert path.parent == p.processed_ephys


# ─────────────────────────────────────────────────────────────────────── #
# Type-specific helpers                                                       #
# ─────────────────────────────────────────────────────────────────────── #

class TestTypeSpecificHelpers:
    def _paths(self, tmp_path):
        from neurobox.dtype.paths import NBSessionPaths
        return NBSessionPaths(
            "sirotaA-jg-05-20120316", tmp_path, "B01", "cof",
        )

    def test_spk_file_default_and_method(self, tmp_path):
        p = self._paths(tmp_path)
        assert p.spk_file(1).name == \
            "sirotaA-jg-05-20120316.spk.standard.1"
        assert p.spk_file(1, method="stderiv").name == \
            "sirotaA-jg-05-20120316.spk.stderiv.1"

    def test_fet_pca_variant_pairs(self, tmp_path):
        p = self._paths(tmp_path)
        assert p.fet_file(1).name == \
            "sirotaA-jg-05-20120316.fet.standard.1"
        assert p.fet_file(1, method="stderiv").name == \
            "sirotaA-jg-05-20120316.fet.stderiv.1"
        assert p.pca_file(1).name == \
            "sirotaA-jg-05-20120316.pca.standard.1"
        assert p.pca_file(1, method="stderiv").name == \
            "sirotaA-jg-05-20120316.pca.stderiv.1"

    def test_clc_clp(self, tmp_path):
        p = self._paths(tmp_path)
        assert p.clc_file(1).name == \
            "sirotaA-jg-05-20120316.clc.standard.1"
        assert p.clp_file(1, method="stderiv").name == \
            "sirotaA-jg-05-20120316.clp.stderiv.1"

    def test_col(self, tmp_path):
        p = self._paths(tmp_path)
        assert p.col_file(1).name == \
            "sirotaA-jg-05-20120316.col.standard.1"

    def test_legacy_helpers_still_return_untagged(self, tmp_path):
        """For backward compatibility, the singular-arg legacy
        helpers return the untagged legacy path (``.res.1``, not
        ``.res.standard.1``).  The new API is accessed via
        :meth:`res_ns3_file` and :meth:`clu_ns3_file`."""
        p = self._paths(tmp_path)
        assert p.res_file(1).name == "sirotaA-jg-05-20120316.res.1"
        assert p.clu_file(1).name == "sirotaA-jg-05-20120316.clu.1"

    def test_variant_tagged_res_clu(self, tmp_path):
        p = self._paths(tmp_path)
        assert p.res_ns3_file(1).name == \
            "sirotaA-jg-05-20120316.res.standard.1"
        assert p.clu_ns3_file(1, method="stderiv").name == \
            "sirotaA-jg-05-20120316.clu.stderiv.1"


# ─────────────────────────────────────────────────────────────────────── #
# resolve_ns3 — filesystem lookup with class-specific fallback              #
# ─────────────────────────────────────────────────────────────────────── #

class TestResolveNs3:
    def _paths(self, tmp_path):
        from neurobox.dtype.paths import NBSessionPaths
        p = NBSessionPaths(
            "sirotaA-jg-05-20120316", tmp_path, "B01", "cof",
        )
        p.processed_ephys.mkdir(parents=True, exist_ok=True)
        return p

    # ── SessionWide: returns canonical, method ignored ──────────────── #

    def test_session_wide_no_fallback(self, tmp_path):
        p = self._paths(tmp_path)
        got, m = p.resolve_ns3("yaml")
        assert got == p.processed_ephys / "sirotaA-jg-05-20120316.yaml"
        assert m == "standard"

    # ── MethodSpecific: strict, no fallback ─────────────────────────── #

    def test_method_specific_strict_returns_variant_path(self, tmp_path):
        p = self._paths(tmp_path)
        # Only .fet.standard.1 exists — asking for stderiv returns the
        # stderiv path (not the standard one), even though it does not exist.
        (p.processed_ephys / "sirotaA-jg-05-20120316.fet.standard.1"
         ).write_bytes(b"\x00")
        got, m = p.resolve_ns3("fet", 1, method="stderiv")
        assert got.name == "sirotaA-jg-05-20120316.fet.stderiv.1"
        assert m == "stderiv"
        assert not got.exists()

    def test_method_specific_finds_when_exists(self, tmp_path):
        p = self._paths(tmp_path)
        target = (p.processed_ephys
                  / "sirotaA-jg-05-20120316.fet.stderiv.2")
        target.write_bytes(b"\x00")
        got, m = p.resolve_ns3("fet", 2, method="stderiv")
        assert got == target
        assert m == "stderiv"

    # ── Shared: prefer <method> → standard → untagged ───────────────── #

    def test_shared_prefers_method_tag(self, tmp_path):
        p = self._paths(tmp_path)
        # Create all three; the method-tagged path wins.
        (p.processed_ephys / "sirotaA-jg-05-20120316.spk.stderiv.1"
         ).write_bytes(b"A")
        (p.processed_ephys / "sirotaA-jg-05-20120316.spk.standard.1"
         ).write_bytes(b"B")
        (p.processed_ephys / "sirotaA-jg-05-20120316.spk.1"
         ).write_bytes(b"C")
        got, m = p.resolve_ns3("spk", 1, method="stderiv")
        assert got.read_bytes() == b"A"
        assert m == "stderiv"

    def test_shared_falls_back_to_standard(self, tmp_path):
        p = self._paths(tmp_path)
        # No stderiv, but standard + legacy exist.
        (p.processed_ephys / "sirotaA-jg-05-20120316.spk.standard.1"
         ).write_bytes(b"B")
        (p.processed_ephys / "sirotaA-jg-05-20120316.spk.1"
         ).write_bytes(b"C")
        got, m = p.resolve_ns3("spk", 1, method="stderiv")
        assert got.read_bytes() == b"B"
        assert m == "standard"

    def test_shared_falls_back_to_untagged(self, tmp_path):
        p = self._paths(tmp_path)
        # Only the untagged legacy file exists.
        (p.processed_ephys / "sirotaA-jg-05-20120316.spk.1"
         ).write_bytes(b"C")
        got, m = p.resolve_ns3("spk", 1, method="stderiv")
        assert got.read_bytes() == b"C"
        # Per the spec: a resolved untagged file reports 'standard'.
        assert m == "standard"

    def test_shared_no_files_returns_primary_path(self, tmp_path):
        p = self._paths(tmp_path)
        got, m = p.resolve_ns3("res", 1, method="stderiv")
        assert got.name == "sirotaA-jg-05-20120316.res.stderiv.1"
        assert m == "stderiv"
        assert not got.exists()

    def test_shared_standard_method_only_tries_two_candidates(self, tmp_path):
        """When method='standard', the fallback list is ``standard``
        then untagged.  It shouldn't try 'standard' twice."""
        p = self._paths(tmp_path)
        (p.processed_ephys / "sirotaA-jg-05-20120316.res.1"
         ).write_bytes(b"X")
        got, m = p.resolve_ns3("res", 1)
        assert got.read_bytes() == b"X"
        assert m == "standard"


# ─────────────────────────────────────────────────────────────────────── #
# load_fet — binary .fet reader (variant-tagged and legacy)                  #
# ─────────────────────────────────────────────────────────────────────── #

class TestLoadSpkFromParFallback:
    """Neurosuite-3 makes ``.spk`` a Shared artifact.  The reader
    should try ``.spk.<method>.N``, then ``.spk.standard.N``, then
    the untagged legacy ``.spk.N``."""

    def _minimal_yaml(self, tmp_path, name, n_ch, n_samples):
        yaml_path = tmp_path / f"{name}.yaml"
        yaml_path.write_text(
            "acquisitionSystem:\n"
            "  nChannels: 64\n"
            "  samplingRate: 20000\n"
            "spikeDetection:\n"
            "  channelGroups:\n"
            f"    - channels: {list(range(n_ch))}\n"
            f"      nSamples: {n_samples}\n"
        )

    def _write_spk(self, path, n_spikes, n_samples, n_ch, fill_value=1):
        n_values = n_spikes * n_samples * n_ch
        payload = np.full(n_values, fill_value, dtype="<i2")
        path.write_bytes(payload.tobytes())

    def test_finds_method_tagged(self, tmp_path):
        from neurobox.io.load_spk import load_spk_from_par
        name = "sirotaA-jg-05-20120316"
        n_ch, n_samples, n_spikes = 8, 32, 4
        self._minimal_yaml(tmp_path, name, n_ch, n_samples)
        self._write_spk(
            tmp_path / f"{name}.spk.standard.1",
            n_spikes, n_samples, n_ch, fill_value=7,
        )
        got = load_spk_from_par(tmp_path / name, shank=1)
        assert got.shape == (n_spikes, n_samples, n_ch)
        assert got.flat[0] == 7

    def test_falls_back_to_standard_when_method_missing(self, tmp_path):
        from neurobox.io.load_spk import load_spk_from_par
        name = "sirotaA-jg-05-20120316"
        n_ch, n_samples, n_spikes = 4, 16, 3
        self._minimal_yaml(tmp_path, name, n_ch, n_samples)
        # Only standard exists — requesting stderiv falls back
        self._write_spk(
            tmp_path / f"{name}.spk.standard.1",
            n_spikes, n_samples, n_ch, fill_value=42,
        )
        got = load_spk_from_par(tmp_path / name, shank=1, method="stderiv")
        assert got.shape == (n_spikes, n_samples, n_ch)
        assert got.flat[0] == 42

    def test_falls_back_to_untagged_legacy(self, tmp_path):
        from neurobox.io.load_spk import load_spk_from_par
        name = "sirotaA-jg-05-20120316"
        n_ch, n_samples, n_spikes = 4, 16, 3
        self._minimal_yaml(tmp_path, name, n_ch, n_samples)
        # Only legacy untagged exists
        self._write_spk(
            tmp_path / f"{name}.spk.1",
            n_spikes, n_samples, n_ch, fill_value=99,
        )
        got = load_spk_from_par(tmp_path / name, shank=1)
        assert got.shape == (n_spikes, n_samples, n_ch)
        assert got.flat[0] == 99

    def test_no_spk_file_raises_file_not_found(self, tmp_path):
        from neurobox.io.load_spk import load_spk_from_par
        name = "sirotaA-jg-05-20120316"
        self._minimal_yaml(tmp_path, name, 4, 16)
        with pytest.raises(FileNotFoundError, match="tried"):
            load_spk_from_par(tmp_path / name, shank=1)


# ─────────────────────────────────────────────────────────────────────── #
# Existing formats — regression: unchanged behaviour                          #
# ─────────────────────────────────────────────────────────────────────── #

class TestExistingBinaryFormatsStillWork:
    def test_read_res_int64_le(self, tmp_path):
        from neurobox.io.load_clu_res import _read_res
        p = tmp_path / "s.res.1"
        p.write_bytes(np.array([10, 20, 30], dtype="<i8").tobytes())
        got = _read_res(p)
        np.testing.assert_array_equal(got, [10, 20, 30])

    def test_read_clu_drops_header(self, tmp_path):
        from neurobox.io.load_clu_res import _read_clu
        p = tmp_path / "s.clu.1"
        p.write_bytes(np.array([5, 2, 3, 4], dtype="<i4").tobytes())
        got = _read_clu(p)
        np.testing.assert_array_equal(got, [2, 3, 4])
