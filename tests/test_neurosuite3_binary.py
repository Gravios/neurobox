"""Tests for the neurosuite-3 binary readers:
:func:`~neurobox.io.load_fet`, :func:`~neurobox.io.load_pca`, and
the hierarchical-clustering :func:`~neurobox.io.load_clc` /
:func:`~neurobox.io.load_clp` (plus their helpers)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
class TestLoadFet:
    @staticmethod
    def _write_fet(path: Path, feats: np.ndarray, ts: np.ndarray) -> None:
        n_dim = feats.shape[1] + 1
        with open(path, "wb") as fh:
            fh.write(np.int32(n_dim).tobytes())
            payload = np.column_stack([feats, ts]).astype("<i8")
            fh.write(payload.tobytes())

    def test_roundtrip_variant_tagged(self, tmp_path):
        from neurobox.io import load_fet
        feats = np.array(
            [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
            dtype=np.int64,
        )
        ts = np.array([100, 200, 300, 400], dtype=np.int64)
        p = tmp_path / "session.fet.standard.1"
        self._write_fet(p, feats, ts)

        got = load_fet(p)
        np.testing.assert_array_equal(got.features,   feats)
        np.testing.assert_array_equal(got.timestamps, ts)
        assert got.n_dimensions == 4

    def test_roundtrip_stderiv_variant(self, tmp_path):
        """load_fet doesn't care which method — just the file
        contents.  Sanity-check with a stderiv-named file."""
        from neurobox.io import load_fet
        feats = np.zeros((3, 2), dtype=np.int64)
        ts    = np.arange(3, dtype=np.int64)
        p = tmp_path / "session.fet.stderiv.1"
        self._write_fet(p, feats, ts)
        got = load_fet(p)
        assert got.n_dimensions == 3
        np.testing.assert_array_equal(got.timestamps, ts)

    def test_roundtrip_legacy_untagged(self, tmp_path):
        from neurobox.io import load_fet
        feats = np.zeros((2, 2), dtype=np.int64)
        ts    = np.array([10, 20], dtype=np.int64)
        p = tmp_path / "session.fet.1"        # untagged legacy
        self._write_fet(p, feats, ts)
        got = load_fet(p)
        np.testing.assert_array_equal(got.timestamps, ts)

    def test_dtype_int64(self, tmp_path):
        from neurobox.io import load_fet
        feats = np.zeros((10, 2), dtype=np.int64)
        ts    = np.arange(10, dtype=np.int64)
        p = tmp_path / "s.fet.standard.1"
        self._write_fet(p, feats, ts)
        got = load_fet(p)
        assert got.features.dtype   == np.int64
        assert got.timestamps.dtype == np.int64

    def test_n_spikes_validation(self, tmp_path):
        from neurobox.io import load_fet
        feats = np.zeros((5, 2), dtype=np.int64)
        ts    = np.arange(5, dtype=np.int64)
        p = tmp_path / "s.fet.standard.1"
        self._write_fet(p, feats, ts)
        load_fet(p, n_spikes=5)
        with pytest.raises(ValueError, match="expected 99"):
            load_fet(p, n_spikes=99)

    def test_file_not_found(self, tmp_path):
        from neurobox.io import load_fet
        with pytest.raises(FileNotFoundError):
            load_fet(tmp_path / "does-not-exist.fet.standard.1")

    def test_truncated_file(self, tmp_path):
        from neurobox.io import load_fet
        p = tmp_path / "s.fet.standard.1"
        p.write_bytes(b"\x00\x00")
        with pytest.raises(ValueError, match="too small"):
            load_fet(p)

    def test_size_not_multiple_of_row(self, tmp_path):
        from neurobox.io import load_fet
        p = tmp_path / "s.fet.standard.1"
        with open(p, "wb") as fh:
            fh.write(np.int32(3).tobytes())
            fh.write(b"\x00" * 20)
        with pytest.raises(ValueError, match="not a multiple of row size"):
            load_fet(p)


# ─────────────────────────────────────────────────────────────────────── #
# load_pca — binary .pca reader                                              #
# ─────────────────────────────────────────────────────────────────────── #

class TestLoadPca:
    @staticmethod
    def _write_pca(
        path:        Path,
        n_ch:        int,
        n_samples:   int,
        n_comp:      int,
        means:       np.ndarray,
        eigenvecs:   np.ndarray,
        data2use:    int = 0,
        is_centered: int = 1,
        rec_shift:   int = 0,
    ) -> None:
        assert means.shape == (n_ch, n_samples)
        assert eigenvecs.shape == (n_ch, n_comp, n_samples)
        with open(path, "wb") as fh:
            hdr = np.array(
                [n_ch, data2use, n_comp, is_centered, rec_shift],
                dtype="<i4",
            )
            fh.write(hdr.tobytes())
            fh.write(means.astype("<f8").tobytes())
            fh.write(eigenvecs.astype("<f8").tobytes())

    def test_roundtrip_variant_tagged(self, tmp_path):
        from neurobox.io import load_pca
        n_ch, n_samples, n_comp = 4, 32, 3
        rng = np.random.default_rng(0)
        means = rng.normal(size=(n_ch, n_samples))
        evecs = rng.normal(size=(n_ch, n_comp, n_samples))
        p = tmp_path / "session.pca.standard.1"
        self._write_pca(p, n_ch, n_samples, n_comp, means, evecs)

        got = load_pca(p, n_samples=n_samples)
        assert got.n_channels   == n_ch
        assert got.n_samples    == n_samples
        assert got.n_components == n_comp
        assert got.is_centered  is True
        np.testing.assert_allclose(got.means,        means)
        np.testing.assert_allclose(got.eigenvectors, evecs)

    def test_stderiv_variant_file(self, tmp_path):
        """load_pca is naming-agnostic; sanity-check with a stderiv name."""
        from neurobox.io import load_pca
        p = tmp_path / "s.pca.stderiv.1"
        self._write_pca(
            p, n_ch=2, n_samples=8, n_comp=2,
            means     = np.ones((2, 8)),
            eigenvecs = np.ones((2, 2, 8)),
        )
        got = load_pca(p, n_samples=8)
        assert got.n_channels == 2 and got.n_components == 2

    def test_legacy_untagged_file(self, tmp_path):
        from neurobox.io import load_pca
        p = tmp_path / "s.pca.1"      # untagged legacy
        self._write_pca(
            p, n_ch=1, n_samples=4, n_comp=1,
            means     = np.zeros((1, 4)),
            eigenvecs = np.zeros((1, 1, 4)),
        )
        got = load_pca(p, n_samples=4)
        assert got.n_channels == 1

    def test_header_field_passthrough(self, tmp_path):
        from neurobox.io import load_pca
        p = tmp_path / "s.pca.standard.1"
        self._write_pca(
            p, n_ch=2, n_samples=4, n_comp=1,
            means      = np.zeros((2, 4)),
            eigenvecs  = np.zeros((2, 1, 4)),
            data2use   = 42,
            is_centered= 0,
            rec_shift  = -3,
        )
        got = load_pca(p, n_samples=4)
        assert got.data2use    == 42
        assert got.is_centered is False
        assert got.rec_shift   == -3

    def test_wrong_n_samples_raises(self, tmp_path):
        from neurobox.io import load_pca
        p = tmp_path / "s.pca.standard.1"
        self._write_pca(
            p, n_ch=2, n_samples=8, n_comp=1,
            means     = np.zeros((2, 8)),
            eigenvecs = np.zeros((2, 1, 8)),
        )
        with pytest.raises(ValueError, match="size mismatch"):
            load_pca(p, n_samples=16)

    def test_truncated_header(self, tmp_path):
        from neurobox.io import load_pca
        p = tmp_path / "s.pca.standard.1"
        p.write_bytes(b"\x00" * 8)
        with pytest.raises(ValueError, match="truncated header"):
            load_pca(p, n_samples=4)

    def test_file_not_found(self, tmp_path):
        from neurobox.io import load_pca
        with pytest.raises(FileNotFoundError):
            load_pca(tmp_path / "missing.pca.standard.1", n_samples=32)


# ─────────────────────────────────────────────────────────────────────── #
# load_clc / load_clp — hierarchical layers                                  #
# ─────────────────────────────────────────────────────────────────────── #

class TestLoadClcClp:
    @staticmethod
    def _write_int32_framed(path: Path, header: int, payload) -> None:
        with open(path, "wb") as fh:
            fh.write(np.int32(header).tobytes())
            fh.write(np.asarray(payload, dtype="<i4").tobytes())

    def test_load_clc_variant_tagged(self, tmp_path):
        from neurobox.io import load_clc
        p = tmp_path / "session.clc.standard.1"
        self._write_int32_framed(p, header=5, payload=[0, 1, 2, 3, 4])
        got = load_clc(p)
        np.testing.assert_array_equal(got, [0, 1, 2, 3, 4])

    def test_load_clc_legacy_untagged(self, tmp_path):
        from neurobox.io import load_clc
        p = tmp_path / "session.clc.1"    # untagged legacy still works
        self._write_int32_framed(p, header=3, payload=[2, 2, 3])
        got = load_clc(p)
        np.testing.assert_array_equal(got, [2, 2, 3])

    def test_load_clc_missing(self, tmp_path):
        from neurobox.io import load_clc
        with pytest.raises(FileNotFoundError):
            load_clc(tmp_path / "missing.clc.standard.1")

    def test_load_clp_header_and_payload(self, tmp_path):
        from neurobox.io import load_clp
        p = tmp_path / "session.clp.standard.1"
        self._write_int32_framed(p, header=4, payload=[2, 2, 3, 0])
        got = load_clp(p)
        assert got.header == 4
        np.testing.assert_array_equal(got.parent_of, [2, 2, 3, 0])

    def test_load_clp_stderiv_variant(self, tmp_path):
        from neurobox.io import load_clp
        p = tmp_path / "session.clp.stderiv.1"
        self._write_int32_framed(p, header=2, payload=[1, 1])
        got = load_clp(p)
        assert got.header == 2

    def test_load_clp_missing(self, tmp_path):
        from neurobox.io import load_clp
        with pytest.raises(FileNotFoundError):
            load_clp(tmp_path / "missing.clp.standard.1")

    def test_load_clp_truncated_below_header(self, tmp_path):
        from neurobox.io import load_clp
        p = tmp_path / "session.clp.standard.1"
        p.write_bytes(b"\x00\x00")
        with pytest.raises(ValueError, match="too small"):
            load_clp(p)

    def test_load_clp_bad_alignment(self, tmp_path):
        from neurobox.io import load_clp
        p = tmp_path / "session.clp.standard.1"
        p.write_bytes(b"\x00" * 4 + b"\x00" * 3)
        with pytest.raises(ValueError, match="whole number of int32"):
            load_clp(p)

    def test_build_atom_to_fiber_drops_zero_default(self, tmp_path):
        from neurobox.io import load_clp, build_atom_to_fiber
        p = tmp_path / "s.clp.standard.1"
        self._write_int32_framed(p, header=4, payload=[2, 2, 3, 0])
        m = build_atom_to_fiber(load_clp(p))
        assert m == {1: 2, 2: 2, 3: 3}

    def test_build_atom_to_fiber_keeps_zero_when_requested(self, tmp_path):
        from neurobox.io import load_clp, build_atom_to_fiber
        p = tmp_path / "s.clp.standard.1"
        self._write_int32_framed(p, header=4, payload=[2, 2, 3, 0])
        m = build_atom_to_fiber(load_clp(p), include_zero=True)
        assert m == {1: 2, 2: 2, 3: 3, 4: 0}

    def test_build_fiber_to_atoms_grouping(self, tmp_path):
        from neurobox.io import load_clp, build_fiber_to_atoms
        p = tmp_path / "s.clp.standard.1"
        self._write_int32_framed(p, header=5, payload=[2, 3, 2, 3, 3])
        m = build_fiber_to_atoms(load_clp(p))
        assert m == {2: [1, 3], 3: [2, 4, 5]}
