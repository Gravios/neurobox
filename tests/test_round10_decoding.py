"""Tests for the Bayesian decoding stack."""

from __future__ import annotations

import numpy as np
import pytest

from neurobox.dtype import NBDxyz, NBModel, NBStateCollection, NBEpoch
from neurobox.analysis.kinematics import augment_xyz
from neurobox.analysis.decoding import (
    decode_ufr_boxcar, DecodingResult,
    prepare_ratemap, prepare_bin_coords,
    accumulate_decoding_vars, AccumulatedDecoding,
    create_tensor_mask,
    CircularBoundary, SquareBoundary, LineBoundary,
    theta_phase, stc2mat,
)


# ─────────────────────────────────────────────────────────────────────────── #
# Synthetic place-field result builder                                         #
# ─────────────────────────────────────────────────────────────────────────── #

class _FakePF:
    """Minimal place-field result for decoder testing."""
    def __init__(self, n_x=25, n_y=25, peaks=None):
        edges = np.linspace(-500, 500, n_x + 1)
        self.bin_centres = [
            0.5 * (edges[:-1] + edges[1:]),
            0.5 * (edges[:-1] + edges[1:]),
        ]
        if peaks is None:
            peaks = [(-300, -300), (-100, 100), (100, -100), (300, 300), (0, 0)]
        n_units = len(peaks)
        self.rate_map = np.zeros((n_x, n_y, n_units, 1))
        gx, gy = np.meshgrid(self.bin_centres[0], self.bin_centres[1], indexing="ij")
        for u, (px, py) in enumerate(peaks):
            self.rate_map[..., u, 0] = 30 * np.exp(
                -((gx - px) ** 2 + (gy - py) ** 2) / (2 * 80 ** 2)
            )


# ─────────────────────────────────────────────────────────────────────────── #
# create_tensor_mask                                                            #
# ─────────────────────────────────────────────────────────────────────────── #

class TestTensorMask:
    def test_circular_default_radius(self):
        bins = [np.linspace(-500, 500, 26), np.linspace(-500, 500, 26)]
        mask = create_tensor_mask(bins, CircularBoundary(radius=440))
        # Centre bin should be in-bounds
        assert mask[12, 12]
        # Corner bin should be out
        assert not mask[0, 0]
        # About 60-70% area filled (circle in square)
        frac = mask.mean()
        assert 0.5 < frac < 0.85

    def test_square(self):
        bins = [np.linspace(-500, 500, 26), np.linspace(-500, 500, 26)]
        mask = create_tensor_mask(bins, SquareBoundary(edge_length=600))
        # Bins beyond ±300 should be excluded
        bx = bins[0]
        for i, x in enumerate(bx):
            for j, y in enumerate(bins[1]):
                expected = (abs(x) <= 300) and (abs(y) <= 300)
                assert mask[i, j] == expected, f"mismatch at ({i},{j})"

    def test_line(self):
        bins = [np.linspace(-1000, 1000, 51), np.linspace(-300, 300, 16)]
        mask = create_tensor_mask(bins, LineBoundary(edge_length=1500))
        # x in [-750, 750], y in [-200, 200]
        assert mask.sum() > 0
        # Bin at x=-1000 should be excluded
        assert not mask[0, mask.shape[1] // 2]

    def test_default_circular(self):
        bins = [np.linspace(-500, 500, 26), np.linspace(-500, 500, 26)]
        m1 = create_tensor_mask(bins)
        m2 = create_tensor_mask(bins, CircularBoundary(radius=440))
        np.testing.assert_array_equal(m1, m2)

    def test_too_few_dims_raises(self):
        with pytest.raises(ValueError, match="at least 2 axes"):
            create_tensor_mask([np.arange(10)])


# ─────────────────────────────────────────────────────────────────────────── #
# prepare_ratemap / prepare_bin_coords                                         #
# ─────────────────────────────────────────────────────────────────────────── #

class TestPrepareRatemap:
    def test_basic_shapes(self):
        pf = _FakePF()
        rm = prepare_ratemap(pf)
        # n_bins = 25*25, n_units = 5
        assert rm.shape == (625, 5)

    def test_with_mask(self):
        pf = _FakePF()
        mask = create_tensor_mask(pf.bin_centres, CircularBoundary(radius=440))
        rm = prepare_ratemap(pf, mask)
        assert rm.shape == (mask.sum(), 5)

    def test_bin_coords_shape(self):
        pf = _FakePF()
        coords = prepare_bin_coords(pf)
        assert coords.shape == (625, 2)

    def test_bin_coords_with_mask(self):
        pf = _FakePF()
        mask = create_tensor_mask(pf.bin_centres, CircularBoundary(radius=440))
        coords = prepare_bin_coords(pf, mask)
        rm = prepare_ratemap(pf, mask)
        # Same row count
        assert coords.shape[0] == rm.shape[0]


# ─────────────────────────────────────────────────────────────────────────── #
# decode_ufr_boxcar                                                            #
# ─────────────────────────────────────────────────────────────────────────── #

class TestDecodeUfrBoxcar:
    def test_recovers_known_position(self):
        """When unit i is firing, posterior peak should be at unit i's place field centre."""
        pf = _FakePF()
        mask = create_tensor_mask(pf.bin_centres, CircularBoundary(radius=440))
        rm = prepare_ratemap(pf, mask)
        coords = prepare_bin_coords(pf, mask)

        T = 200
        ufr = np.zeros((T, 5))
        # Unit 1 fires throughout — peak at (-100, 100)
        ufr[:, 1] = 25.0

        result = decode_ufr_boxcar(
            ufr, rm, coords, samplerate=250.0,
        )
        assert result.n > 0
        # Average decoded peak should be near unit 1's place field
        mean_max = result.max.mean(axis=0)
        np.testing.assert_allclose(mean_max, [-100, 100], atol=40)

    def test_skips_silent_periods(self):
        pf = _FakePF()
        rm = prepare_ratemap(pf)
        coords = prepare_bin_coords(pf)
        T = 200
        ufr = np.zeros((T, 5))
        # Only middle 100 samples have activity
        ufr[100:200, 0] = 10.0
        result = decode_ufr_boxcar(
            ufr, rm, coords, samplerate=250.0,
        )
        # Decoded indices should all be in the active range
        assert result.ind.min() >= 100 - 6   # account for window
        assert result.ind.max() <= 200

    def test_result_has_all_attributes(self):
        pf = _FakePF()
        rm = prepare_ratemap(pf)
        coords = prepare_bin_coords(pf)
        T = 100
        ufr = np.zeros((T, 5))
        ufr[:, 1] = 20.0
        result = decode_ufr_boxcar(ufr, rm, coords, samplerate=250.0)

        for attr in ("ind", "max", "com", "sax", "lom", "lax",
                     "post", "ucnt", "uinc"):
            assert getattr(result, attr) is not None
        # sax, com, lom, lax should all be near each other for sharp posteriors
        assert np.allclose(result.com.mean(axis=0), result.sax.mean(axis=0), atol=50)

    def test_input_validation(self):
        pf = _FakePF()
        rm = prepare_ratemap(pf)
        coords = prepare_bin_coords(pf)

        # Wrong unit count
        with pytest.raises(ValueError, match="unit count mismatch"):
            decode_ufr_boxcar(
                np.zeros((10, 99)), rm, coords, samplerate=250.0,
            )
        # Wrong bin count
        with pytest.raises(ValueError, match="bin count mismatch"):
            decode_ufr_boxcar(
                np.zeros((10, 5)), rm[:50], coords, samplerate=250.0,
            )

    def test_small_window_raises(self):
        pf = _FakePF()
        rm = prepare_ratemap(pf)
        coords = prepare_bin_coords(pf)
        with pytest.raises(ValueError, match="too small"):
            decode_ufr_boxcar(
                np.zeros((10, 5)), rm, coords,
                samplerate=10.0, half_spike_window_s=0.01,
            )

    def test_smoothing_matrix_passed_directly(self):
        pf = _FakePF()
        rm = prepare_ratemap(pf)
        coords = prepare_bin_coords(pf)
        T = 100
        ufr = np.zeros((T, 5))
        ufr[:, 1] = 20.0
        # 2-D smoothing matrix
        result = decode_ufr_boxcar(
            ufr, rm, coords, samplerate=250.0,
            smoothing_weights=np.array([[800**2, 0], [0, 800**2]]),
        )
        assert result.n > 0


# ─────────────────────────────────────────────────────────────────────────── #
# accumulate_decoding_vars                                                     #
# ─────────────────────────────────────────────────────────────────────────── #

class TestAccumulate:
    def _setup(self):
        pf = _FakePF()
        mask = create_tensor_mask(pf.bin_centres, CircularBoundary(radius=440))
        rm = prepare_ratemap(pf, mask)
        coords = prepare_bin_coords(pf, mask)

        T = 1000
        ufr = np.zeros((T, 5))
        ufr[:, 1] = 25.0  # constant unit-1 activity → peak at (-100, 100)
        result = decode_ufr_boxcar(ufr, rm, coords, samplerate=250.0)

        markers = [
            "spine_lower", "pelvis_root", "spine_middle", "spine_upper",
            "head_back", "head_left", "head_front", "head_right",
        ]
        # Rat moving in a circle, head pointing along the trajectory
        # so that the head frame is well-defined (nose ahead of hcom in xy).
        t = np.arange(T) / 250.0
        body_x = 200 * np.cos(t * 0.5)
        body_y = 200 * np.sin(t * 0.5)
        # Heading (tangent to circle)
        heading = t * 0.5 + np.pi / 2
        cos_h, sin_h = np.cos(heading), np.sin(heading)

        # Per-marker offsets in body-local frame (x is forward, y lateral, z height)
        body_offsets = {
            "spine_lower":  (-30,  0, 50),
            "pelvis_root":  (-15,  0, 50),
            "spine_middle": (  0,  0, 50),
            "spine_upper":  ( 15,  0, 50),
            "head_back":    ( 25,  0, 60),
            "head_left":    ( 30, +5, 60),
            "head_front":   ( 35,  0, 60),
            "head_right":   ( 30, -5, 60),
        }
        xyz_data = np.zeros((T, len(markers), 3))
        for i, m in enumerate(markers):
            ox, oy, oz = body_offsets[m]
            xyz_data[:, i, 0] = body_x + ox * cos_h - oy * sin_h
            xyz_data[:, i, 1] = body_y + ox * sin_h + oy * cos_h
            xyz_data[:, i, 2] = oz

        xyz = NBDxyz(xyz_data, model=NBModel(markers=markers), samplerate=250.0)
        return result, augment_xyz(xyz)

    def test_returns_full_struct(self):
        result, xyz_aug = self._setup()
        acc = accumulate_decoding_vars(result, xyz_aug)
        for attr in ("ecom", "esax", "emax", "elom", "elax",
                     "tcom", "tsax", "tmax", "tlom", "tlax",
                     "hvec", "tvec", "xyz"):
            assert hasattr(acc, attr)
            assert getattr(acc, attr).shape[0] == acc.decoding.n

    def test_edge_pad_drops_samples(self):
        result, xyz_aug = self._setup()
        acc = accumulate_decoding_vars(result, xyz_aug, edge_pad=26)
        assert acc.decoding.n == result.n - 52

    def test_samplerate_mismatch_raises(self):
        result, xyz_aug = self._setup()
        # Build an xyz at 120 Hz instead of 250
        xyz_120 = NBDxyz(xyz_aug.data, model=xyz_aug.model, samplerate=120.0)
        with pytest.raises(ValueError, match="samplerate mismatch"):
            accumulate_decoding_vars(result, xyz_120)

    def test_missing_markers_raises(self):
        result, xyz_aug = self._setup()
        # Strip hcom/nose
        xyz_bare = xyz_aug.subset(["spine_lower", "head_back"])
        # Need to manually fix samplerate (subset preserves it)
        with pytest.raises(ValueError, match="must contain"):
            accumulate_decoding_vars(result, xyz_bare)

    def test_head_yaw_rotation(self):
        result, xyz_aug = self._setup()
        a0 = accumulate_decoding_vars(result, xyz_aug, head_yaw=0.0)
        a1 = accumulate_decoding_vars(result, xyz_aug, head_yaw=np.pi / 2)
        # Rotation by 90° should change ecom (different projections)
        assert not np.allclose(a0.ecom, a1.ecom, atol=1.0)


# ─────────────────────────────────────────────────────────────────────────── #
# theta_phase                                                                  #
# ─────────────────────────────────────────────────────────────────────────── #

class TestThetaPhase:
    def test_pure_8hz_recovers_phase(self):
        """Synthesise a clean 8 Hz sinusoid; phase should advance linearly."""
        fs = 1250.0
        t = np.arange(int(5 * fs)) / fs
        lfp = np.cos(2 * np.pi * 8 * t)
        phase = theta_phase(lfp, samplerate=fs)
        # Phase should be wrapped to [0, 2π)
        assert phase.min() >= 0.0
        assert phase.max() < 2 * np.pi
        # Differences (after unwrapping) should be ~constant
        unwrapped = np.unwrap(phase)
        # Avoid edge effects
        a, b, step = int(fs), int(4 * fs), 10
        diffs = np.diff(unwrapped[a:b:step])
        # Expected per-sample increment: 2π * 8 / fs * step
        expected = 2 * np.pi * 8 / fs * step
        np.testing.assert_allclose(diffs.mean(), expected, rtol=0.05)

    def test_correction_offset_applied(self):
        fs = 1250.0
        t = np.arange(int(2 * fs)) / fs
        lfp = np.cos(2 * np.pi * 8 * t)
        p0 = theta_phase(lfp, samplerate=fs, correction=0.0)
        p1 = theta_phase(lfp, samplerate=fs, correction=np.pi / 4)
        # Difference (modulo 2π) should be π/4
        circ_diff = np.angle(np.exp(1j * (p1 - p0)))
        np.testing.assert_allclose(circ_diff, np.pi / 4, atol=1e-6)

    def test_resample(self):
        fs = 1250.0
        t = np.arange(int(2 * fs)) / fs
        lfp = np.cos(2 * np.pi * 8 * t)
        phase = theta_phase(lfp, samplerate=fs, resample_to=250.0)
        # Should have ~ 2 * 250 samples
        assert abs(len(phase) - 500) <= 2

    def test_multichannel(self):
        fs = 1250.0
        t = np.arange(int(2 * fs)) / fs
        lfp = np.column_stack([
            np.cos(2 * np.pi * 8 * t),
            np.cos(2 * np.pi * 8 * t + 0.5),
        ])
        phase = theta_phase(lfp, samplerate=fs)
        assert phase.shape == lfp.shape


# ─────────────────────────────────────────────────────────────────────────── #
# stc2mat                                                                      #
# ─────────────────────────────────────────────────────────────────────────── #

class TestStc2Mat:
    def _make_stc(self):
        stc = NBStateCollection()
        # State A active in [0, 1) s; state B active in [1, 2) s
        stc.add_state(
            NBEpoch(data=np.array([[0.0, 1.0]]), samplerate=250.0,
                    label="alpha", key="a"),
        )
        stc.add_state(
            NBEpoch(data=np.array([[1.0, 2.0]]), samplerate=250.0,
                    label="beta", key="b"),
        )
        return stc

    def test_basic(self):
        stc = self._make_stc()
        smat, names = stc2mat(stc, n_samples=500, samplerate=250.0)
        assert smat.shape == (500, 2)
        assert names == ["alpha", "beta"]
        # Column 0 should be 1 in the first half; column 1 should be 2 in the second half
        assert smat[100, 0] == 1
        assert smat[100, 1] == 0
        assert smat[400, 0] == 0
        assert smat[400, 1] == 2

    def test_subset_states(self):
        stc = self._make_stc()
        smat, names = stc2mat(
            stc, n_samples=500, samplerate=250.0, states=["beta"],
        )
        assert names == ["beta"]
        assert smat.shape == (500, 1)

    def test_unknown_state_silently_skipped(self):
        stc = self._make_stc()
        smat, names = stc2mat(
            stc, n_samples=500, samplerate=250.0,
            states=["alpha", "no_such_state"],
        )
        assert names == ["alpha", "no_such_state"]
        # Second column should be all zeros
        assert (smat[:, 1] == 0).all()
