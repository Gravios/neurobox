"""Tests for the neurosuite-3 binary writers in
:mod:`neurobox.io.ns3_writers`.

Each writer must:

* produce byte-exact output that the matching reader reproduces
  losslessly, and
* enforce ``overwrite=False`` (the default), atomic replacement,
  little-endian encoding, and input-shape / dtype validation.
"""
from __future__ import annotations

import os
import struct
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest


# ─────────────────────────────────────────────────────────────────────── #
# Round-trip: writer → reader                                                #
# ─────────────────────────────────────────────────────────────────────── #

class TestRoundTripRes:
    def test_basic(self, tmp_path):
        from neurobox.io import save_res
        from neurobox.io.load_clu_res import _read_res
        p = tmp_path / "s.res.standard.1"
        ts = np.array([10, 200, 3_000, 40_000], dtype=np.int64)
        save_res(p, ts)
        np.testing.assert_array_equal(_read_res(p), ts)

    def test_empty(self, tmp_path):
        from neurobox.io import save_res
        from neurobox.io.load_clu_res import _read_res
        p = tmp_path / "s.res.standard.1"
        save_res(p, np.array([], dtype=np.int64))
        assert p.stat().st_size == 0
        got = _read_res(p)
        assert got.size == 0

    def test_dtype_coercion_int32_to_int64(self, tmp_path):
        from neurobox.io import save_res
        from neurobox.io.load_clu_res import _read_res
        p = tmp_path / "s.res.standard.1"
        save_res(p, np.array([1, 2, 3], dtype=np.int32))    # non-int64 input
        got = _read_res(p)
        assert got.dtype == np.int64
        np.testing.assert_array_equal(got, [1, 2, 3])

    def test_2d_input_rejected(self, tmp_path):
        from neurobox.io import save_res
        with pytest.raises(ValueError, match="1-D"):
            save_res(tmp_path / "s.res.1", np.zeros((3, 2), dtype=np.int64))


class TestRoundTripClu:
    def test_basic(self, tmp_path):
        from neurobox.io import save_clu
        from neurobox.io.load_clu_res import _read_clu
        p = tmp_path / "s.clu.standard.1"
        ids = np.array([2, 3, 2, 4, 3, 3], dtype=np.int32)
        save_clu(p, ids)
        np.testing.assert_array_equal(_read_clu(p), ids)

    def test_header_stores_distinct_cluster_count(self, tmp_path):
        """.clu header is the number of distinct cluster IDs (per
        the neurosuite-3 clu.md spec)."""
        from neurobox.io import save_clu
        p = tmp_path / "s.clu.standard.1"
        # 4 distinct IDs: {0, 1, 2, 5}
        save_clu(p, np.array([0, 1, 2, 2, 5, 1, 0], dtype=np.int32))
        with open(p, "rb") as fh:
            header = struct.unpack("<i", fh.read(4))[0]
        assert header == 4

    def test_header_zero_for_empty_input(self, tmp_path):
        from neurobox.io import save_clu
        p = tmp_path / "s.clu.standard.1"
        save_clu(p, np.array([], dtype=np.int32))
        with open(p, "rb") as fh:
            header = struct.unpack("<i", fh.read(4))[0]
        assert header == 0
        assert p.stat().st_size == 4     # only the header

    def test_dtype_coercion_int64_to_int32(self, tmp_path):
        from neurobox.io import save_clu
        from neurobox.io.load_clu_res import _read_clu
        p = tmp_path / "s.clu.standard.1"
        save_clu(p, np.array([2, 3, 4], dtype=np.int64))
        got = _read_clu(p)
        assert got.dtype == np.int32
        np.testing.assert_array_equal(got, [2, 3, 4])


class TestRoundTripClc:
    def test_same_framing_as_clu(self, tmp_path):
        """.clc has the same on-disk layout as .clu — the same
        reader unpacks it correctly."""
        from neurobox.io import save_clc, load_clc
        p = tmp_path / "s.clc.standard.1"
        atoms = np.array([2, 2, 3, 5, 3], dtype=np.int32)
        save_clc(p, atoms)
        got = load_clc(p)
        np.testing.assert_array_equal(got, atoms)


class TestRoundTripClp:
    def test_basic(self, tmp_path):
        from neurobox.io import save_clp, load_clp
        p = tmp_path / "s.clp.standard.1"
        # Atoms 1..5 owned by fibers 2, 2, 3, 0, 3
        parent = np.array([2, 2, 3, 0, 3], dtype=np.int32)
        save_clp(p, parent)
        got = load_clp(p)
        np.testing.assert_array_equal(got.parent_of, parent)
        # Header = highest atom ID written (= n_atoms in dense layout)
        assert got.header == 5

    def test_round_trip_through_atom_to_fiber_map(self, tmp_path):
        from neurobox.io import (
            save_clp, load_clp,
            build_atom_to_fiber, build_fiber_to_atoms,
        )
        p = tmp_path / "s.clp.standard.1"
        parent = np.array([2, 2, 3, 0, 3], dtype=np.int32)
        save_clp(p, parent)
        clp = load_clp(p)
        a_to_f = build_atom_to_fiber(clp)
        assert a_to_f == {1: 2, 2: 2, 3: 3, 5: 3}     # atom 4 (parent=0) dropped
        f_to_as = build_fiber_to_atoms(clp)
        assert f_to_as == {2: [1, 2], 3: [3, 5]}

    def test_rejects_2d(self, tmp_path):
        from neurobox.io import save_clp
        with pytest.raises(ValueError, match="1-D"):
            save_clp(tmp_path / "s.clp.1", np.zeros((3, 2), dtype=np.int32))


class TestRoundTripSpk:
    def test_basic_round_trip(self, tmp_path):
        from neurobox.io import save_spk, load_spk
        p = tmp_path / "s.spk.standard.1"
        n_spikes, n_samples, n_channels = 5, 32, 4
        rng = np.random.default_rng(0)
        wf = rng.integers(-1000, 1000, size=(n_spikes, n_samples, n_channels),
                            dtype=np.int16)
        save_spk(p, wf)
        got = load_spk(p, n_samples=n_samples, n_channels=n_channels)
        np.testing.assert_array_equal(got, wf)
        # File size matches spec: nSpikes × nSamples × nChannels × 2
        assert p.stat().st_size == n_spikes * n_samples * n_channels * 2

    def test_channel_layout_matches_spec(self, tmp_path):
        """Neurosuite-3 spk.md: "sample-major layout (all channels
        for sample 0 first, then all for sample 1, etc.)."  With a
        distinctive per-position value we can verify the byte order
        directly."""
        from neurobox.io import save_spk
        p = tmp_path / "s.spk.standard.1"
        # 1 spike, 3 samples, 2 channels.  Values chosen so on-disk
        # order (sample-major, channels innermost) is: 10, 11, 20, 21, 30, 31.
        wf = np.array(
            [[[10, 11], [20, 21], [30, 31]]],
            dtype=np.int16,
        )
        save_spk(p, wf)
        expected_bytes = np.array([10, 11, 20, 21, 30, 31],
                                     dtype="<i2").tobytes()
        assert p.read_bytes() == expected_bytes

    def test_dtype_coercion_from_float(self, tmp_path):
        from neurobox.io import save_spk, load_spk
        p = tmp_path / "s.spk.standard.1"
        wf = np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float64)
        save_spk(p, wf)
        got = load_spk(p, n_samples=2, n_channels=2)
        assert got.dtype == np.int16
        np.testing.assert_array_equal(got, wf.astype(np.int16))

    def test_rejects_2d(self, tmp_path):
        from neurobox.io import save_spk
        with pytest.raises(ValueError, match="3-D"):
            save_spk(tmp_path / "s.spk.1", np.zeros((3, 4), dtype=np.int16))


class TestRoundTripFet:
    def test_basic(self, tmp_path):
        from neurobox.io import save_fet, load_fet
        p = tmp_path / "s.fet.standard.1"
        feats = np.array(
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            dtype=np.int64,
        )
        ts = np.array([100, 200, 300], dtype=np.int64)
        save_fet(p, feats, ts)
        got = load_fet(p)
        np.testing.assert_array_equal(got.features,   feats)
        np.testing.assert_array_equal(got.timestamps, ts)
        assert got.n_dimensions == feats.shape[1] + 1     # +1 for timestamp

    def test_header_matches_ndim(self, tmp_path):
        from neurobox.io import save_fet
        p = tmp_path / "s.fet.standard.1"
        feats = np.zeros((2, 5), dtype=np.int64)
        ts    = np.arange(2, dtype=np.int64)
        save_fet(p, feats, ts)
        with open(p, "rb") as fh:
            n_dim = struct.unpack("<i", fh.read(4))[0]
        assert n_dim == 6                                # 5 feats + 1 timestamp

    def test_timestamps_length_mismatch(self, tmp_path):
        from neurobox.io import save_fet
        with pytest.raises(ValueError, match="timestamps has"):
            save_fet(
                tmp_path / "s.fet.1",
                features   = np.zeros((3, 2), dtype=np.int64),
                timestamps = np.arange(5, dtype=np.int64),
            )

    def test_1d_features_rejected(self, tmp_path):
        from neurobox.io import save_fet
        with pytest.raises(ValueError, match="features must be 2-D"):
            save_fet(
                tmp_path / "s.fet.1",
                features   = np.zeros(5, dtype=np.int64),
                timestamps = np.arange(5, dtype=np.int64),
            )


class TestRoundTripPca:
    def test_basic(self, tmp_path):
        from neurobox.io import save_pca, load_pca
        p = tmp_path / "s.pca.standard.1"
        n_ch, n_samples, n_comp = 4, 32, 3
        rng = np.random.default_rng(0)
        means = rng.normal(size=(n_ch, n_samples))
        evecs = rng.normal(size=(n_ch, n_comp, n_samples))
        save_pca(p, means, evecs, data2use=42, is_centered=True, rec_shift=-3)

        got = load_pca(p, n_samples=n_samples)
        assert got.n_channels   == n_ch
        assert got.n_components == n_comp
        assert got.data2use     == 42
        assert got.is_centered  is True
        assert got.rec_shift    == -3
        np.testing.assert_allclose(got.means,        means)
        np.testing.assert_allclose(got.eigenvectors, evecs)

    def test_is_centered_false_encoded_as_zero(self, tmp_path):
        from neurobox.io import save_pca, load_pca
        p = tmp_path / "s.pca.standard.1"
        save_pca(
            p,
            means        = np.zeros((2, 4)),
            eigenvectors = np.zeros((2, 1, 4)),
            is_centered  = False,
        )
        assert load_pca(p, n_samples=4).is_centered is False

    def test_shape_mismatch_channels(self, tmp_path):
        from neurobox.io import save_pca
        with pytest.raises(ValueError, match="n_channels"):
            save_pca(
                tmp_path / "s.pca.1",
                means        = np.zeros((2, 4)),
                eigenvectors = np.zeros((3, 1, 4)),     # wrong n_ch
            )

    def test_shape_mismatch_samples(self, tmp_path):
        from neurobox.io import save_pca
        with pytest.raises(ValueError, match="n_samples"):
            save_pca(
                tmp_path / "s.pca.1",
                means        = np.zeros((2, 4)),
                eigenvectors = np.zeros((2, 1, 8)),     # wrong n_samples
            )

    def test_dtype_coercion_float32_to_float64(self, tmp_path):
        from neurobox.io import save_pca, load_pca
        p = tmp_path / "s.pca.standard.1"
        means = np.ones((2, 4), dtype=np.float32) * 2.5
        evecs = np.ones((2, 1, 4), dtype=np.float32) * 0.75
        save_pca(p, means, evecs)
        got = load_pca(p, n_samples=4)
        assert got.means.dtype == np.float64
        np.testing.assert_allclose(got.means, means, rtol=1e-6)
        np.testing.assert_allclose(got.eigenvectors, evecs, rtol=1e-6)


# ─────────────────────────────────────────────────────────────────────── #
# overwrite semantics                                                        #
# ─────────────────────────────────────────────────────────────────────── #

class TestOverwriteSemantics:
    def test_refuses_existing_by_default(self, tmp_path):
        from neurobox.io import save_res
        p = tmp_path / "s.res.1"
        p.write_bytes(b"existing")
        with pytest.raises(FileExistsError, match="already exists"):
            save_res(p, np.array([1, 2], dtype=np.int64))
        # Original preserved
        assert p.read_bytes() == b"existing"

    def test_overwrite_true_replaces(self, tmp_path):
        from neurobox.io import save_res
        from neurobox.io.load_clu_res import _read_res
        p = tmp_path / "s.res.1"
        p.write_bytes(b"existing")
        save_res(p, np.array([1, 2, 3], dtype=np.int64), overwrite=True)
        np.testing.assert_array_equal(_read_res(p), [1, 2, 3])

    def test_all_binary_writers_respect_overwrite_false(self, tmp_path):
        """Sanity: every binary writer honours the overwrite=False default."""
        from neurobox.io import (
            save_res, save_clu, save_clc, save_clp,
            save_spk, save_fet, save_pca,
        )
        # Placeholders exist for every target
        writers_and_args = [
            (save_res, "s.res.1",  {"timestamps":  np.array([1], dtype=np.int64)}),
            (save_clu, "s.clu.1",  {"cluster_ids": np.array([2], dtype=np.int32)}),
            (save_clc, "s.clc.1",  {"atom_ids":    np.array([2], dtype=np.int32)}),
            (save_clp, "s.clp.1",  {"parent_of":   np.array([2], dtype=np.int32)}),
            (save_spk, "s.spk.1",
                {"waveforms": np.zeros((1, 1, 1), dtype=np.int16)}),
            (save_fet, "s.fet.1",
                {"features":   np.zeros((1, 1), dtype=np.int64),
                 "timestamps": np.array([0], dtype=np.int64)}),
            (save_pca, "s.pca.1",
                {"means":        np.zeros((1, 1)),
                 "eigenvectors": np.zeros((1, 1, 1))}),
        ]
        for fn, name, kwargs in writers_and_args:
            path = tmp_path / name
            path.write_bytes(b"x")
            with pytest.raises(FileExistsError):
                fn(path, **kwargs)


# ─────────────────────────────────────────────────────────────────────── #
# Atomic write behaviour                                                     #
# ─────────────────────────────────────────────────────────────────────── #

class TestAtomicWrite:
    def test_uses_tmp_sidecar_then_rename(self, tmp_path, monkeypatch):
        """Verify writes go via a ``.tmp`` sidecar that's renamed
        onto the final path."""
        from neurobox.io import save_res
        from neurobox.io import ns3_writers as w

        seen_paths: list[str] = []
        real_replace = os.replace

        def spy_replace(src, dst):
            seen_paths.append((str(src), str(dst)))
            return real_replace(src, dst)

        monkeypatch.setattr(w.os, "replace", spy_replace)
        p = tmp_path / "s.res.1"
        save_res(p, np.array([1, 2, 3], dtype=np.int64))
        # Exactly one rename, and it went from <name>.tmp → <name>
        assert len(seen_paths) == 1
        src, dst = seen_paths[0]
        assert src == str(p) + ".tmp"
        assert dst == str(p)
        # Sidecar cleaned up
        assert not (tmp_path / "s.res.1.tmp").exists()

    def test_partial_write_leaves_no_destination(self, tmp_path):
        """If the write callback raises, the destination file must
        not appear (though a .tmp file may linger — that's OK)."""
        from neurobox.io import ns3_writers as w
        target = tmp_path / "s.res.1"

        def _boom(fh):
            fh.write(b"partial")
            raise RuntimeError("simulated failure")

        with pytest.raises(RuntimeError, match="simulated"):
            w._atomic_write(target, _boom)
        assert not target.exists()
        # (a .tmp with 'partial' contents may exist; that's expected)


# ─────────────────────────────────────────────────────────────────────── #
# Endianness                                                                 #
# ─────────────────────────────────────────────────────────────────────── #

class TestEndianness:
    """Writers must produce little-endian bytes regardless of host
    endianness, since neurosuite-3 files are always LE."""

    def test_res_bytes_are_little_endian(self, tmp_path):
        from neurobox.io import save_res
        p = tmp_path / "s.res.1"
        save_res(p, np.array([1, 256], dtype=np.int64))
        raw = p.read_bytes()
        # 1 as int64 LE = 01 00 00 00 00 00 00 00
        # 256 as int64 LE = 00 01 00 00 00 00 00 00
        assert raw[:8] == b"\x01\x00\x00\x00\x00\x00\x00\x00"
        assert raw[8:] == b"\x00\x01\x00\x00\x00\x00\x00\x00"

    def test_clu_header_little_endian(self, tmp_path):
        from neurobox.io import save_clu
        p = tmp_path / "s.clu.1"
        # Force header = 256 by providing 256 distinct IDs
        ids = np.arange(256, dtype=np.int32)
        save_clu(p, ids)
        # First 4 bytes = 256 as int32 LE = 00 01 00 00
        assert p.read_bytes()[:4] == b"\x00\x01\x00\x00"

    def test_pca_header_little_endian(self, tmp_path):
        from neurobox.io import save_pca
        p = tmp_path / "s.pca.1"
        # nCh=1, data2use=0, nComp=2, isCentered=1, recShift=0
        save_pca(
            p,
            means        = np.zeros((1, 4)),
            eigenvectors = np.zeros((1, 2, 4)),
            data2use     = 0,
            is_centered  = True,
            rec_shift    = 0,
        )
        # Header = 5 int32s = 20 bytes
        header = p.read_bytes()[:20]
        n_ch, data2use, n_comp, is_c, rec = struct.unpack("<iiiii", header)
        assert (n_ch, data2use, n_comp, is_c, rec) == (1, 0, 2, 1, 0)


# ─────────────────────────────────────────────────────────────────────── #
# Integration with NBSessionPaths                                            #
# ─────────────────────────────────────────────────────────────────────── #

class TestIntegrationWithNs3Paths:
    def test_write_and_resolve_shared_variant(self, tmp_path):
        """save_spk writes a Shared artifact; resolve_ns3 finds it
        under the variant tag it was written with."""
        from neurobox.io import save_spk
        from neurobox.dtype.paths import NBSessionPaths
        p = NBSessionPaths(
            "sirotaA-jg-05-20120316", tmp_path, "B01", "cof",
        )
        wf = np.zeros((2, 4, 3), dtype=np.int16)
        save_spk(p.spk_file(1, method="stderiv"), wf)
        # resolve_ns3 finds it
        got, method = p.resolve_ns3("spk", 1, method="stderiv")
        assert got == p.spk_file(1, method="stderiv")
        assert method == "stderiv"

    def test_write_method_specific_strict_resolves(self, tmp_path):
        """save_fet writes a MethodSpecific artifact; resolve_ns3
        with the same method finds it, with a different method
        returns the (non-existent) target path."""
        from neurobox.io import save_fet
        from neurobox.dtype.paths import NBSessionPaths
        p = NBSessionPaths(
            "sirotaA-jg-05-20120316", tmp_path, "B01", "cof",
        )
        feats = np.zeros((3, 2), dtype=np.int64)
        ts    = np.arange(3,   dtype=np.int64)
        save_fet(p.fet_file(1, method="standard"), feats, ts)

        # Standard resolves cleanly
        target, method = p.resolve_ns3("fet", 1, method="standard")
        assert target.exists()
        assert method == "standard"

        # Stderiv is strict: no fallback to standard
        target, method = p.resolve_ns3("fet", 1, method="stderiv")
        assert not target.exists()
        assert method == "stderiv"

    def test_round_trip_via_paths_helper(self, tmp_path):
        """Full pipeline: build path via NBSessionPaths → save →
        resolve → load."""
        from neurobox.io import save_pca, load_pca
        from neurobox.dtype.paths import NBSessionPaths
        p = NBSessionPaths(
            "sirotaA-jg-05-20120316", tmp_path, "B01", "cof",
        )
        rng = np.random.default_rng(0)
        means = rng.normal(size=(4, 32))
        evecs = rng.normal(size=(4, 2, 32))

        save_pca(
            p.pca_file(1, method="stderiv"),
            means, evecs, data2use=7, is_centered=True, rec_shift=1,
        )
        resolved, method = p.resolve_ns3("pca", 1, method="stderiv")
        assert method == "stderiv"
        got = load_pca(resolved, n_samples=32)
        np.testing.assert_allclose(got.means,        means)
        np.testing.assert_allclose(got.eigenvectors, evecs)
        assert got.data2use   == 7
        assert got.rec_shift  == 1

    def test_creates_parent_directories(self, tmp_path):
        """Writers create parent dirs so callers don't have to
        mkdir the processed_ephys hierarchy manually."""
        from neurobox.io import save_res
        target = tmp_path / "deep" / "nested" / "dir" / "s.res.1"
        save_res(target, np.array([1], dtype=np.int64))
        assert target.exists()
