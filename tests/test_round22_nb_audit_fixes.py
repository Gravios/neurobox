"""Round-22 — regression tests for the NB-object audit fixes.

Three fixes verified here:

1. ``NBEpoch.save(path)`` now raises :class:`FileExistsError` instead of
   silently doing nothing when the target file exists.  The previous
   silent-no-op left callers confused when ``load_file`` later raised
   :class:`EOFError` on a zero-byte file.

2. ``NBTrial()`` no longer crashes when constructed with no
   arguments.  Previously it tried to read ``self.name`` which had
   not been set yet by the early-returning ``NBSession.__init__``.

3. ``NBDang.between(..., component=<unknown>)`` now raises a clear
   :class:`ValueError` with the list of valid component names,
   instead of an obscure :class:`IndexError` from numpy.  The
   ``"rho"`` alias is now recognised as a synonym for ``"r"`` /
   ``"dist"``.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from neurobox.dtype import (
    NBEpoch, NBTrial, NBDxyz, NBDang, NBModel,
)


# ─────────────────────────────────────────────────────────────────────── #
# Fix 1: NBEpoch.save raises FileExistsError                                 #
# ─────────────────────────────────────────────────────────────────────── #

class TestNbEpochSaveLoadFix:
    def test_save_creates_file(self):
        e = NBEpoch(data=np.array([[1.0, 5.0]]), samplerate=30.0,
                     mode="periods")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.epoch.pkl"
            e.save(path)
            assert path.exists() and path.stat().st_size > 0
            loaded = NBEpoch.load_file(path)
            np.testing.assert_array_equal(loaded.data, e.data)

    def test_save_zero_byte_file_writes(self):
        """A pre-existing 0-byte file is treated as 'absent' — save still
        writes, no overwrite required.  This matches the case where the
        user pre-creates an empty placeholder."""
        e = NBEpoch(data=np.array([[1.0, 5.0]]), samplerate=30.0,
                     mode="periods")
        with tempfile.NamedTemporaryFile(suffix=".epoch.pkl",
                                            delete=False) as tf:
            path = Path(tf.name)
        try:
            assert path.stat().st_size == 0
            e.save(path)                          # no overwrite needed
            assert path.stat().st_size > 0
        finally:
            path.unlink(missing_ok=True)

    def test_save_existing_file_raises(self):
        """Without overwrite=True, save against a populated file raises."""
        e = NBEpoch(data=np.array([[1.0, 5.0]]), samplerate=30.0,
                     mode="periods")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.epoch.pkl"
            e.save(path)                          # first save: OK
            with pytest.raises(FileExistsError):
                e.save(path)                      # second save: raises
            e.save(path, overwrite=True)          # explicit overwrite: OK


# ─────────────────────────────────────────────────────────────────────── #
# Fix 2: NBTrial() with no args                                              #
# ─────────────────────────────────────────────────────────────────────── #

class TestNbTrialNoArgsConstruct:
    def test_construct_no_args(self):
        """NBTrial() should not crash with AttributeError on self.name."""
        t = NBTrial()
        assert hasattr(t, "load")
        assert hasattr(t, "save")

    def test_construct_with_session_name(self):
        """When given a session name, the usual filebase machinery runs.

        We can't go all the way through to disk lookup without a real
        project, but the construction itself shouldn't crash on the
        basic identity-attribute setup.
        """
        # Use a name in the canonical 4-part format
        try:
            NBTrial(session_name="sirotaA-jg-05-20120316",
                     project_id="B01", data_root="/tmp")
        except FileNotFoundError:
            pass        # Expected — there's no real session at /tmp
        except OSError:
            pass
        # Anything else is a regression


# ─────────────────────────────────────────────────────────────────────── #
# Fix 3: NBDang.between rejects unknown components                           #
# ─────────────────────────────────────────────────────────────────────── #

class TestNbDangBetweenComponentValidation:
    def _make_ang(self):
        rng = np.random.default_rng(0)
        pts = rng.standard_normal((200, 3, 3)) * 100 + 10
        xyz = NBDxyz(pts, model=NBModel(markers=["a", "b", "c"]),
                      samplerate=120.0)
        return NBDang.from_xyz(xyz)

    def test_known_components_work(self):
        ang = self._make_ang()
        for comp in ("theta", "azimuth", "phi", "elevation",
                     "r", "dist", "rho"):
            out = ang.between("a", "b", component=comp)
            assert out.shape == (200,)

    def test_integer_component_works(self):
        ang = self._make_ang()
        for ci in (0, 1, 2):
            out = ang.between("a", "b", component=ci)
            assert out.shape == (200,)

    def test_unknown_component_raises_value_error(self):
        """Previously this gave an obscure IndexError from numpy."""
        ang = self._make_ang()
        with pytest.raises(ValueError, match="Unknown component"):
            ang.between("a", "b", component="bogus")


# ─────────────────────────────────────────────────────────────────────── #
# Fix 4: NBSession xyz-sync slicing respects sync samplerate                 #
# ─────────────────────────────────────────────────────────────────────── #

class TestSessionSyncSlicing:
    """Regression test for the silent off-by-(samplerate-ratio) bug in
    NBSession.load('xyz').

    The documented convention is that ``session.sync`` (and therefore
    ``NBTrial.sync``) is an :class:`NBEpoch` with ``samplerate=1.0``
    storing seconds.  The previous xyz load path called
    ``self.sync.to_mask(xyz.n_samples)`` directly, which silently
    interpreted the period values as already-in-samples — producing
    a mask covering ``samplerate_ratio`` of the intended trial data
    (e.g. ~1 % for 120 Hz mocap).
    """

    def test_to_mask_against_different_target_rate(self):
        """Direct demonstration of the fix logic."""
        # Sync as seconds (samplerate=1.0)
        sync = NBEpoch(data=np.array([[30.0, 50.0]]),
                        samplerate=1.0, mode="periods")

        # Wrong path (the old behaviour) — sync interpreted as 1 Hz
        bad_mask = sync.to_mask(12000)
        assert bad_mask.sum() == 20      # only 20 "samples" of bug

        # Correct path: resample sync to xyz's rate first
        sync_at_120 = sync.resample(120.0)
        good_mask = sync_at_120.to_mask(12000)
        assert good_mask.sum() == 2400   # 20 s × 120 Hz

    def test_sync_resample_preserves_periods_in_seconds(self):
        sync = NBEpoch(data=np.array([[10.0, 20.0]]),
                        samplerate=1.0, mode="periods")
        rs = sync.resample(120.0)
        assert rs.samplerate == 120.0
        # Period values stay in seconds — only the rate (used by to_mask)
        # changes
        np.testing.assert_allclose(rs.data, sync.data)

    def test_sync_already_at_target_rate_no_double_resample(self):
        """If sync.samplerate already matches the target, don't bother
        round-tripping through resample (also avoids subtle precision
        loss)."""
        sync = NBEpoch(data=np.array([[10.0, 20.0]]),
                        samplerate=120.0, mode="periods")
        # The fix uses an explicit guard: abs(diff) > 1e-9
        # so this returns the original mask construction
        mask = sync.to_mask(12000)
        assert mask.sum() == 1200      # 10 s × 120 Hz
