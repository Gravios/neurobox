"""Tests for the neurosuite-3 YAML + small-format writers:
:func:`~neurobox.io.save_col`, :func:`~neurobox.io.save_drift`,
:func:`~neurobox.io.save_loc`, :func:`~neurobox.io.save_chunks`
— and their matching readers."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
class TestRoundTripCol:
    def _sample_spikes(self):
        return [
            {"spikeIndex":  4721,
             "isCollision": True,
             "components":  [
                 {"unit": 3, "shift": -2, "amplitude": 0.97},
                 {"unit": 7, "shift":  8, "amplitude": 0.84},
             ]},
            {"spikeIndex":  4900,
             "isCollision": False,
             "components":  []},
        ]

    def test_basic_round_trip(self, tmp_path):
        from neurobox.io import save_col, load_col
        p = tmp_path / "s.col.standard.1"
        save_col(p, spikes=self._sample_spikes(), spike_group=1)
        doc = load_col(p)
        assert doc["collisions"]["format"]     == "1.0"
        assert doc["collisions"]["spikeGroup"] == 1
        assert doc["collisions"]["spikes"]     == self._sample_spikes()

    def test_top_level_key_ordering(self, tmp_path):
        """The neurosuite-3 spec shows format / spikeGroup / spikes
        in that order; sort_keys=False preserves it."""
        from neurobox.io import save_col
        p = tmp_path / "s.col.standard.1"
        save_col(p, spikes=self._sample_spikes(), spike_group=1)
        text = p.read_text()
        i_fmt   = text.index("format:")
        i_group = text.index("spikeGroup:")
        i_spk   = text.index("spikes:")
        assert i_fmt < i_group < i_spk

    def test_empty_spikes_list_valid(self, tmp_path):
        from neurobox.io import save_col, load_col
        p = tmp_path / "s.col.standard.1"
        save_col(p, spikes=[], spike_group=3)
        assert load_col(p)["collisions"]["spikes"] == []

    def test_non_list_spikes_rejected(self, tmp_path):
        from neurobox.io import save_col
        with pytest.raises(ValueError, match="must be a list"):
            save_col(tmp_path / "s.col.1", spikes={"bad": True}, spike_group=1)


# ─────────────────────────────────────────────────────────────────────── #
# .drift — YAML round-trip                                                   #
# ─────────────────────────────────────────────────────────────────────── #

class TestRoundTripDrift:
    def _sample_probes(self):
        return [
            {"probeId": 0,
             "shanks": [
                 {"shankIndex": 0,
                  "spikeGroup": 1,
                  "nUnitsTotal": 8,
                  "windows": [
                      {"t_start": 0.0,  "t_end":  60.0, "drift_um":  0.0},
                      {"t_start": 60.0, "t_end": 120.0, "drift_um": -1.8},
                  ]},
             ]},
        ]

    def test_basic_round_trip(self, tmp_path):
        from neurobox.io import save_drift, load_drift
        p = tmp_path / "s.drift"
        save_drift(
            p,
            probes     = self._sample_probes(),
            method     = "unit_com",
            window_sec = 60.0,
        )
        doc = load_drift(p)
        assert doc["drift"]["format"]    == "1.0"
        assert doc["drift"]["method"]    == "unit_com"
        assert doc["drift"]["windowSec"] == pytest.approx(60.0)
        assert doc["drift"]["probes"]    == self._sample_probes()

    def test_field_ordering(self, tmp_path):
        """format → method → windowSec → probes per the spec."""
        from neurobox.io import save_drift
        p = tmp_path / "s.drift"
        save_drift(p, probes=self._sample_probes(),
                    method="unit_com", window_sec=60.0)
        text = p.read_text()
        i_fmt = text.index("format:")
        i_mth = text.index("method:")
        i_win = text.index("windowSec:")
        i_prb = text.index("probes:")
        assert i_fmt < i_mth < i_win < i_prb

    def test_non_list_probes_rejected(self, tmp_path):
        from neurobox.io import save_drift
        with pytest.raises(ValueError, match="must be a list"):
            save_drift(
                tmp_path / "s.drift", probes={}, method="x", window_sec=1.0,
            )


# ─────────────────────────────────────────────────────────────────────── #
# .loc — binary round-trip                                                   #
# ─────────────────────────────────────────────────────────────────────── #

class TestRoundTripLoc:
    def test_basic(self, tmp_path):
        from neurobox.io import save_loc, load_loc, LOC_COLUMNS
        assert LOC_COLUMNS == ("x_s", "y_s", "z_s", "A", "residual")

        p = tmp_path / "s.loc.1"
        rng = np.random.default_rng(0)
        locs = rng.normal(size=(20, 5)).astype(np.float32)
        save_loc(p, locs)
        got = load_loc(p)
        assert got.dtype == np.float32
        np.testing.assert_allclose(got, locs, rtol=0, atol=0)

    def test_file_size_is_n_spikes_times_20(self, tmp_path):
        from neurobox.io import save_loc
        p = tmp_path / "s.loc.1"
        save_loc(p, np.zeros((13, 5), dtype=np.float32))
        assert p.stat().st_size == 13 * 20

    def test_rejects_wrong_ncols(self, tmp_path):
        from neurobox.io import save_loc
        with pytest.raises(ValueError, match="5 columns"):
            save_loc(tmp_path / "s.loc.1",
                      np.zeros((10, 4), dtype=np.float32))

    def test_rejects_1d(self, tmp_path):
        from neurobox.io import save_loc
        with pytest.raises(ValueError, match="2-D"):
            save_loc(tmp_path / "s.loc.1", np.zeros(5, dtype=np.float32))

    def test_reader_rejects_wrong_size(self, tmp_path):
        from neurobox.io import load_loc
        p = tmp_path / "s.loc.1"
        p.write_bytes(b"\x00" * 21)     # not a multiple of 20
        with pytest.raises(ValueError, match="multiple of row size"):
            load_loc(p)


# ─────────────────────────────────────────────────────────────────────── #
# .chunks — text round-trip                                                  #
# ─────────────────────────────────────────────────────────────────────── #

class TestRoundTripChunks:
    def test_basic(self, tmp_path):
        from neurobox.io import save_chunks, load_chunks
        p = tmp_path / "s.chunks.1"
        chunks = np.array([[0, 20000], [20000, 40000], [40000, 55000]],
                            dtype=np.int64)
        save_chunks(p, chunks)
        got = load_chunks(p)
        np.testing.assert_array_equal(got, chunks)

    def test_text_layout_one_per_line(self, tmp_path):
        from neurobox.io import save_chunks
        p = tmp_path / "s.chunks.1"
        save_chunks(p, [(0, 100), (100, 250)])
        assert p.read_text() == "0 100\n100 250\n"

    def test_single_chunk_1d_input(self, tmp_path):
        from neurobox.io import save_chunks, load_chunks
        p = tmp_path / "s.chunks.1"
        save_chunks(p, np.array([0, 100], dtype=np.int64))
        got = load_chunks(p)
        np.testing.assert_array_equal(got, [[0, 100]])

    def test_reader_ignores_comments_and_blank_lines(self, tmp_path):
        from neurobox.io import load_chunks
        p = tmp_path / "s.chunks.1"
        p.write_text("# header comment\n0 100\n\n100 200\n# trailing\n")
        got = load_chunks(p)
        np.testing.assert_array_equal(got, [[0, 100], [100, 200]])

    def test_reader_rejects_malformed(self, tmp_path):
        from neurobox.io import load_chunks
        p = tmp_path / "s.chunks.1"
        p.write_text("0 100 200\n")     # three fields
        with pytest.raises(ValueError, match="two integer fields"):
            load_chunks(p)

    def test_empty_chunks(self, tmp_path):
        from neurobox.io import save_chunks, load_chunks
        p = tmp_path / "s.chunks.1"
        save_chunks(p, np.empty((0, 2), dtype=np.int64))
        assert p.read_text() == ""
        got = load_chunks(p)
        assert got.shape == (0, 2)

    def test_rejects_wrong_shape(self, tmp_path):
        from neurobox.io import save_chunks
        with pytest.raises(ValueError):
            save_chunks(tmp_path / "s.chunks.1",
                          np.zeros((3, 3), dtype=np.int64))
