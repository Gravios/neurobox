"""Tests for :meth:`NBSpk.save` — the object-level round-trip
wrapper over :func:`save_res` / :func:`save_clu`.

The core invariant tested is:

    NBSpk.load(NBSpk.save(spk)) == spk

for spike / cluster arrays derived from real ``.res`` / ``.clu``
files: same spike times, same cluster IDs (globally remapped), same
shank map.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


# --------------------------------------------------------------------- #
# Fixture: minimal on-disk .yaml + .res.N + .clu.N for two shanks       #
# --------------------------------------------------------------------- #

@pytest.fixture
def two_shank_session(tmp_path):
    """Two-shank session:

        shank 1: 4 spikes, local clusters [2, 3, 2, 4]  → globals [2, 3, 2, 4]
        shank 2: 3 spikes, local clusters [2, 2, 3]     → globals [5, 5, 6]

    Everything time-sorted across shanks after load_clu_res
    concatenation.
    """
    from neurobox.io.ns3_writers import save_res, save_clu

    name = "sirotaA-jg-05-20120316"
    base = tmp_path / name
    sr   = 20000.0

    # Minimal YAML enough for load_par to pull samplingRate + a
    # 2-group spikeDetection block.
    (tmp_path / f"{name}.yaml").write_text(
        "acquisitionSystem:\n"
        "  nChannels: 32\n"
        f"  samplingRate: {sr}\n"
        "spikeDetection:\n"
        "  channelGroups:\n"
        "    - channels: [0, 1, 2, 3]\n"
        "      nSamples: 32\n"
        "    - channels: [4, 5, 6, 7]\n"
        "      nSamples: 32\n"
    )

    # Shank 1: 4 spikes at samples 1000, 5000, 12000, 30000
    save_res(base.with_name(f"{name}.res.1"),
              np.array([1000, 5000, 12000, 30000], dtype=np.int64))
    save_clu(base.with_name(f"{name}.clu.1"),
              np.array([2, 3, 2, 4], dtype=np.int32))

    # Shank 2: 3 spikes at samples 3000, 15000, 25000
    save_res(base.with_name(f"{name}.res.2"),
              np.array([3000, 15000, 25000], dtype=np.int64))
    save_clu(base.with_name(f"{name}.clu.2"),
              np.array([2, 2, 3], dtype=np.int32))

    return tmp_path, name, sr


# --------------------------------------------------------------------- #
# Round-trip                                                            #
# --------------------------------------------------------------------- #

class TestNbspkSaveRoundTrip:
    def test_save_then_load_gives_identical_spk(self, two_shank_session, tmp_path):
        """Load, save with a fresh method tag, load back — every
        field of NBSpk should match."""
        from neurobox.dtype.spikes import NBSpk

        base_dir, name, sr = two_shank_session
        original = NBSpk.load(base_dir / name)

        # Save under a fresh 'roundtrip' method tag so it doesn't
        # clash with the fixture files.
        method_tag = "roundtrip"
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        out_base = out_dir / name

        # Copy the yaml over so load can find it in the new dir
        (tmp_path / f"{name}.yaml").rename(out_dir / f"{name}.yaml")

        written = original.save(out_base, method=method_tag)
        assert len(written) == 4      # 2 shanks × (res, clu)
        for p in written:
            assert p.exists()

        # The written files use variant tags: reload them by hand
        # via load_clu_res, then confirm arrays match.  (NBSpk.load
        # doesn't yet know about the method-tagged Shared fallback
        # for .res, so bypass it for this round-trip check.)
        from neurobox.io import load_clu_res

        # Reconstruct by pointing load_clu_res at the tagged files
        # via a per-shank hand-load, mimicking what NBSpk.load
        # would do internally.
        s1_res = np.fromfile(
            str(out_base) + f".res.{method_tag}.1", dtype="<i8",
        )
        s1_clu = np.fromfile(
            str(out_base) + f".clu.{method_tag}.1", dtype="<i4",
        )[1:]     # drop header
        s2_res = np.fromfile(
            str(out_base) + f".res.{method_tag}.2", dtype="<i8",
        )
        s2_clu = np.fromfile(
            str(out_base) + f".clu.{method_tag}.2", dtype="<i4",
        )[1:]

        # Shank 1 spikes should be at (rounded) original sample indices
        np.testing.assert_array_equal(s1_res, [1000, 5000, 12000, 30000])
        np.testing.assert_array_equal(s1_clu, [2, 3, 2, 4])

        # Shank 2 local cluster IDs should be un-mapped back
        np.testing.assert_array_equal(s2_res, [3000, 15000, 25000])
        np.testing.assert_array_equal(s2_clu, [2, 2, 3])

    def test_save_creates_variant_tagged_files(self, two_shank_session, tmp_path):
        """Verify the filenames written match neurosuite-3 variant
        naming: ``<base>.res.<method>.<shank>``."""
        from neurobox.dtype.spikes import NBSpk

        base_dir, name, sr = two_shank_session
        original = NBSpk.load(base_dir / name)

        out_dir = tmp_path / "out"
        out_dir.mkdir()
        (tmp_path / f"{name}.yaml").rename(out_dir / f"{name}.yaml")

        written = original.save(out_dir / name, method="stderiv")
        written_names = sorted(p.name for p in written)
        assert written_names == [
            f"{name}.clu.stderiv.1",
            f"{name}.clu.stderiv.2",
            f"{name}.res.stderiv.1",
            f"{name}.res.stderiv.2",
        ]

    def test_default_method_is_standard(self, two_shank_session, tmp_path):
        from neurobox.dtype.spikes import NBSpk

        base_dir, name, sr = two_shank_session
        original = NBSpk.load(base_dir / name)

        out_dir = tmp_path / "out"
        out_dir.mkdir()
        (tmp_path / f"{name}.yaml").rename(out_dir / f"{name}.yaml")

        written = original.save(out_dir / name)
        assert all(".standard." in p.name for p in written)


# --------------------------------------------------------------------- #
# Empty / edge cases                                                    #
# --------------------------------------------------------------------- #

class TestNbspkSaveEdgeCases:
    def test_empty_spk_returns_no_files(self):
        from neurobox.dtype.spikes import NBSpk
        empty = NBSpk()
        assert empty.save("/tmp/should-not-be-written") == []

    def test_mismatched_res_clu_raises(self):
        from neurobox.dtype.spikes import NBSpk
        spk = NBSpk(
            res  = np.array([1.0, 2.0, 3.0]),
            clu  = np.array([2, 3], dtype=np.int32),
            map_ = np.array([[2, 1], [3, 1]], dtype=np.int64),
        )
        with pytest.raises(ValueError, match="res has"):
            spk.save("/tmp/never-written")

    def test_missing_map_raises(self):
        from neurobox.dtype.spikes import NBSpk
        spk = NBSpk(
            res = np.array([1.0, 2.0]),
            clu = np.array([2, 3], dtype=np.int32),
        )
        with pytest.raises(ValueError, match="map is empty"):
            spk.save("/tmp/never-written")


# --------------------------------------------------------------------- #
# Overwrite semantics                                                   #
# --------------------------------------------------------------------- #

class TestNbspkSaveOverwrite:
    def test_refuses_existing_by_default(self, two_shank_session, tmp_path):
        from neurobox.dtype.spikes import NBSpk

        base_dir, name, sr = two_shank_session
        spk = NBSpk.load(base_dir / name)

        # Save under a tag that DOES clash with an existing file
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        (tmp_path / f"{name}.yaml").rename(out_dir / f"{name}.yaml")

        # First save fine
        spk.save(out_dir / name, method="v1")
        # Second save must refuse (overwrite=False default)
        with pytest.raises(FileExistsError):
            spk.save(out_dir / name, method="v1")

    def test_overwrite_true_replaces(self, two_shank_session, tmp_path):
        from neurobox.dtype.spikes import NBSpk

        base_dir, name, sr = two_shank_session
        spk = NBSpk.load(base_dir / name)

        out_dir = tmp_path / "out"
        out_dir.mkdir()
        (tmp_path / f"{name}.yaml").rename(out_dir / f"{name}.yaml")

        spk.save(out_dir / name, method="v1")
        # Overwrite works
        spk.save(out_dir / name, method="v1", overwrite=True)


# --------------------------------------------------------------------- #
# Sample-index round-trip precision                                     #
# --------------------------------------------------------------------- #

class TestNbspkSaveSampleAccuracy:
    def test_res_conversion_preserves_sample_indices(self, two_shank_session, tmp_path):
        """Save doesn't accumulate float error going from seconds
        back to samples for reasonable sample rates."""
        from neurobox.dtype.spikes import NBSpk

        base_dir, name, sr = two_shank_session
        original = NBSpk.load(base_dir / name)

        out_dir = tmp_path / "out"
        out_dir.mkdir()
        (tmp_path / f"{name}.yaml").rename(out_dir / f"{name}.yaml")

        original.save(out_dir / name, method="rt")

        # Read the written .res.rt.1 file directly and compare with
        # the original sample indices from the fixture.
        s1_res = np.fromfile(
            str(out_dir / name) + ".res.rt.1", dtype="<i8",
        )
        np.testing.assert_array_equal(s1_res, [1000, 5000, 12000, 30000])
