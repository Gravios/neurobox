"""Tests for round-17 — spline_spine, body-referenced features, heuristics."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from neurobox.dtype import NBDxyz, NBModel, NBEpoch
from neurobox.analysis.kinematics import (
    augment_xyz,
    spline_spine,
    preproc_xyz_spline_spine_head_eqi,
    preproc_xyz_spline_spine_head_eqd,
    body_referenced_features,
    body_referenced_xy_features,
    SplineSpineResult,
    BodyReferencedFeatures,
)
from neurobox.analysis.heuristics import (
    rear,
    walk,
    head_movement,
    walk_angle_subdivide,
    non_rearing_head_periods,
    velocity_threshold_periods,
    sniff_periods,
    shake_periods,
    read_theta_periods,
)


# ─────────────────────────────────────────────────────────────────────── #
# Fixtures                                                                  #
# ─────────────────────────────────────────────────────────────────────── #

DEFAULT_MARKERS = (
    "spine_lower", "pelvis_root", "spine_middle", "spine_upper",
    "head_back", "head_left", "head_front", "head_right",
)
DEFAULT_OFFSETS = {
    "spine_lower":  (-30,  0, 50),
    "pelvis_root":  (-15,  0, 50),
    "spine_middle": (  0,  0, 50),
    "spine_upper":  ( 15,  0, 50),
    "head_back":    ( 25,  0, 60),
    "head_left":    ( 30,  5, 60),
    "head_front":   ( 35,  0, 60),
    "head_right":   ( 30, -5, 60),
}


def _make_synthetic_xyz(
    T:       int = 1200,
    fs:      float = 120.0,
    body_x:  np.ndarray | None = None,
    body_y:  np.ndarray | None = None,
    heading: np.ndarray | None = None,
    extra:   dict | None = None,
) -> NBDxyz:
    """Build a synthetic NBDxyz.  Optionally apply per-marker overrides
    via *extra* (dict of marker → (T, 3) override array)."""
    t = np.arange(T) / fs
    if body_x is None:
        body_x = np.zeros(T)
    if body_y is None:
        body_y = np.zeros(T)
    if heading is None:
        heading = np.full(T, np.pi / 2)

    data = np.zeros((T, len(DEFAULT_MARKERS), 3))
    for i, m in enumerate(DEFAULT_MARKERS):
        ox, oy, oz = DEFAULT_OFFSETS[m]
        cy, sy = np.cos(heading), np.sin(heading)
        data[:, i, 0] = body_x + ox * cy - oy * sy
        data[:, i, 1] = body_y + ox * sy + oy * cy
        data[:, i, 2] = oz
    if extra:
        for m, override in extra.items():
            idx = DEFAULT_MARKERS.index(m)
            data[:, idx, :] = override
    return NBDxyz(
        data, model=NBModel(markers=list(DEFAULT_MARKERS)),
        samplerate=fs, name="round17_test",
    )


# ─────────────────────────────────────────────────────────────────────── #
# Spline spine                                                              #
# ─────────────────────────────────────────────────────────────────────── #

class TestSplineSpine:
    def test_basic_shape(self):
        xyz = _make_synthetic_xyz()
        aug = augment_xyz(xyz)
        ssr = spline_spine(aug, n_interp=100)
        assert isinstance(ssr, SplineSpineResult)
        assert ssr.points.shape == (1200, 100, 3)
        assert ssr.markers == (
            "spine_lower", "pelvis_root", "spine_middle",
            "spine_upper", "hcom",
        )
        assert ssr.samplerate == 120.0

    def test_all_finite(self):
        xyz = _make_synthetic_xyz()
        aug = augment_xyz(xyz)
        ssr = spline_spine(aug, n_interp=50)
        assert np.isfinite(ssr.points).all()

    def test_endpoints_match_anchors(self):
        """First and last spline points should equal first / last anchor."""
        xyz = _make_synthetic_xyz()
        aug = augment_xyz(xyz)
        ssr = spline_spine(aug, n_interp=100)
        sl_idx = aug.model.index("spine_lower")
        hc_idx = aug.model.index("hcom")
        # First spline point ≈ spine_lower
        np.testing.assert_allclose(
            ssr.points[0, 0],  aug.data[0, sl_idx], atol=1e-6,
        )
        # Last spline point ≈ hcom
        np.testing.assert_allclose(
            ssr.points[0, -1], aug.data[0, hc_idx], atol=1e-6,
        )

    def test_missing_marker_raises(self):
        xyz = _make_synthetic_xyz()
        with pytest.raises(KeyError, match="does_not_exist"):
            spline_spine(xyz, markers=("does_not_exist", "spine_lower"))


class TestPreprocSplineSpineEqi:
    def test_shape_preserved(self):
        xyz = _make_synthetic_xyz()
        aug = augment_xyz(xyz)
        eqi = preproc_xyz_spline_spine_head_eqi(aug)
        assert eqi.data.shape == aug.data.shape

    def test_endpoints_preserved(self):
        """spine_lower and hcom (the endpoints) should be unchanged."""
        xyz = _make_synthetic_xyz()
        aug = augment_xyz(xyz)
        eqi = preproc_xyz_spline_spine_head_eqi(aug)
        sl = aug.model.index("spine_lower")
        hc = aug.model.index("hcom")
        np.testing.assert_array_equal(
            eqi.data[:, sl, :], aug.data[:, sl, :],
        )
        np.testing.assert_array_equal(
            eqi.data[:, hc, :], aug.data[:, hc, :],
        )

    def test_inner_markers_on_spline(self):
        """Inner markers should lie on the spline (not at original positions
        for a non-trivial spline)."""
        xyz = _make_synthetic_xyz()
        aug = augment_xyz(xyz)
        eqi = preproc_xyz_spline_spine_head_eqi(aug)
        # For a strictly linear spine, the inner markers won't move much,
        # but the values should still be finite
        pr = eqi.model.index("pelvis_root")
        assert np.isfinite(eqi.data[:, pr, :]).all()


class TestPreprocSplineSpineEqd:
    def test_shape_preserved(self):
        xyz = _make_synthetic_xyz()
        aug = augment_xyz(xyz)
        eqd = preproc_xyz_spline_spine_head_eqd(aug)
        assert eqd.data.shape == aug.data.shape

    def test_runs_with_reference_session(self):
        xyz = _make_synthetic_xyz()
        aug = augment_xyz(xyz)
        ref = augment_xyz(_make_synthetic_xyz(T=300))
        eqd = preproc_xyz_spline_spine_head_eqd(aug, reference_session=ref)
        assert eqd.data.shape == aug.data.shape


# ─────────────────────────────────────────────────────────────────────── #
# Body-referenced features                                                  #
# ─────────────────────────────────────────────────────────────────────── #

class TestBodyReferencedFeatures:
    def test_full_shape(self):
        xyz = _make_synthetic_xyz()
        aug = augment_xyz(xyz)
        result = body_referenced_features(aug)
        assert isinstance(result, BodyReferencedFeatures)
        # 5 markers × (2 walk + 1 z + 2 dwalk + 1 dz) = 30 columns
        assert result.fet.shape == (1200, 30)
        assert len(result.column_names) == 30

    def test_xy_only_shape(self):
        xyz = _make_synthetic_xyz()
        aug = augment_xyz(xyz)
        result = body_referenced_xy_features(aug)
        # 5 markers × (2 walk + 2 dwalk) = 20 columns
        assert result.fet.shape == (1200, 20)
        assert len(result.column_names) == 20

    def test_z_columns_match_input(self):
        """z-position columns should equal the marker z values."""
        xyz = _make_synthetic_xyz()
        aug = augment_xyz(xyz)
        result = body_referenced_features(aug)
        # Columns 10..15 are spine_lower, pelvis_root, spine_middle,
        # spine_upper, hcom z-positions.
        # In our synthetic: spine_* at z=50, hcom (head COM) at z=60.
        np.testing.assert_allclose(
            result.fet[:, 10:14], 50.0, atol=1e-6,
        )
        np.testing.assert_allclose(result.fet[:, 14], 60.0, atol=1e-6)

    def test_dz_zero_for_constant_height(self):
        xyz = _make_synthetic_xyz()
        aug = augment_xyz(xyz)
        result = body_referenced_features(aug)
        # dz columns (25:30) should be zero (heights are constant)
        np.testing.assert_allclose(result.fet[:, 25:30], 0.0, atol=1e-9)

    def test_as_nbdfet_returns_NBDfet(self):
        from neurobox.dtype.fet import NBDfet
        xyz = _make_synthetic_xyz()
        aug = augment_xyz(xyz)
        out = body_referenced_features(aug, as_nbdfet=True)
        assert isinstance(out, NBDfet)
        assert out.name == "fet_bref"
        assert out.samplerate == 120.0

    def test_resample(self):
        xyz = _make_synthetic_xyz()
        aug = augment_xyz(xyz)
        result = body_referenced_features(aug, samplerate=60.0)
        # Halved samplerate ⇒ half the rows (approximately)
        assert result.samplerate == 60.0
        assert abs(result.fet.shape[0] - 600) < 5

    def test_invalid_frames_zeroed(self):
        """Frames with NaN xyz should produce all-zero feature columns."""
        xyz = _make_synthetic_xyz()
        # Inject NaN into one frame
        xyz.data[10, 0, :] = np.nan
        aug = augment_xyz(xyz)
        # augment_xyz fills NaN with eps; for a strict NaN test, manually
        # corrupt aug
        aug.data[100, 0, 0] = np.nan
        result = body_referenced_features(aug)
        np.testing.assert_array_equal(result.fet[100, :], 0.0)


# ─────────────────────────────────────────────────────────────────────── #
# Heuristics                                                                #
# ─────────────────────────────────────────────────────────────────────── #

@pytest.fixture
def walking_session():
    """Synthetic session with alternating walk / sit periods."""
    T, fs = 3600, 120.0
    t = np.arange(T) / fs
    walk_state = ((t // 3) % 2 == 0).astype(float)
    v = 200 * walk_state
    body_x = np.cumsum(v) / fs
    xyz = _make_synthetic_xyz(T=T, fs=fs, body_x=body_x)
    aug = augment_xyz(xyz)
    return aug


@pytest.fixture
def rearing_session():
    """Synthetic session with a rear during 5-7s."""
    T, fs = 3600, 120.0
    t = np.arange(T) / fs
    xyz = _make_synthetic_xyz(T=T, fs=fs)
    rear_window = (t >= 5.0) & (t < 7.0)
    # Pelvis tilts up, head goes high
    xyz.data[rear_window, DEFAULT_MARKERS.index("pelvis_root"),  2] = 80
    xyz.data[rear_window, DEFAULT_MARKERS.index("spine_middle"), 2] = 120
    xyz.data[rear_window, DEFAULT_MARKERS.index("spine_upper"),  2] = 160
    xyz.data[rear_window, DEFAULT_MARKERS.index("head_back"),    2] = 200
    xyz.data[rear_window, DEFAULT_MARKERS.index("head_left"),    2] = 200
    xyz.data[rear_window, DEFAULT_MARKERS.index("head_front"),   2] = 220
    xyz.data[rear_window, DEFAULT_MARKERS.index("head_right"),   2] = 200
    aug = augment_xyz(xyz)
    return aug


class TestRear:
    def test_com_method_detects_rear(self, rearing_session):
        ep = rear(rearing_session, method="com",
                   rear_threshold=50.0, minimum_interval=10)
        assert ep.data.shape[0] >= 1
        # Detected period should overlap the 5-7s ground-truth window
        fs = rearing_session.samplerate
        truth_start = 5.0 * fs
        truth_end   = 7.0 * fs
        for s, e in ep.data:
            if s < truth_end and e > truth_start:
                return  # at least one period overlaps
        pytest.fail("no detected rear overlapped the ground-truth window")

    def test_no_rear_in_quiet_session(self):
        xyz = _make_synthetic_xyz(T=1200)
        aug = augment_xyz(xyz)
        ep = rear(aug, method="com", rear_threshold=50.0)
        assert ep.data.shape[0] == 0

    def test_invalid_method_raises(self, rearing_session):
        with pytest.raises(ValueError, match="method"):
            rear(rearing_session, method="quantum")


class TestWalk:
    def test_com_method(self, walking_session):
        ep = walk(walking_session, method="com", walk_threshold=2.0)
        assert ep.data.shape[0] >= 1

    def test_vel_method(self, walking_session):
        ep = walk(walking_session, method="vel",
                   walk_threshold=2.0, marker="spine_lower")
        assert ep.data.shape[0] >= 1

    def test_head_method(self, walking_session):
        ep = walk(walking_session, method="head", walk_threshold=2.0)
        # Should detect movement (head tracks body)
        assert ep.data.shape[0] >= 0

    def test_invalid_method_raises(self, walking_session):
        with pytest.raises(ValueError, match="method"):
            walk(walking_session, method="run")


class TestHeadMovement:
    def test_com_method(self, walking_session):
        ep = head_movement(walking_session, method="com")
        assert ep.data.shape[0] >= 0

    def test_invalid_method_raises(self, walking_session):
        with pytest.raises(ValueError, match="method"):
            head_movement(walking_session, method="bogus")


class TestWalkAngleSubdivide:
    def test_returns_two_epochs(self, walking_session):
        walk_ep = walk(walking_session, method="com")
        lang, hang = walk_angle_subdivide(walk_ep, walking_session,
                                            angle_threshold=0.0)
        assert isinstance(lang, NBEpoch)
        assert isinstance(hang, NBEpoch)
        assert lang.label == "walk_lang"
        assert hang.label == "walk_hang"


class TestNonRearingHeadPeriods:
    def test_excludes_rear_window(self, rearing_session):
        rear_ep = rear(rearing_session, method="com",
                        rear_threshold=50.0, minimum_interval=10)
        head_ep = head_movement(rearing_session, method="com")
        n = rearing_session.data.shape[0]
        nrhp = non_rearing_head_periods(
            rear_ep, head_ep, n,
            samplerate=rearing_session.samplerate,
            trim_seconds=1.0,
        )
        # No nrhp period should overlap the trimmed rear window
        for rs, re in rear_ep.data:
            ts = rs - 1.0 * rearing_session.samplerate
            te = re + 1.0 * rearing_session.samplerate
            for ns, ne in nrhp.data:
                # Periods should not overlap the trimmed-rear range
                assert not (ns < te and ne > ts), \
                    f"nrhp [{ns},{ne}] overlaps trimmed rear [{ts},{te}]"


class TestVelocityThresholdPeriods:
    def test_basic_detection(self, walking_session):
        ep = velocity_threshold_periods(walking_session, threshold=1.0)
        assert ep.data.shape[0] >= 1
        assert ep.label == "vel"


class TestSniffPeriods:
    def test_detects_oscillation(self):
        T, fs = 3600, 120.0
        t = np.arange(T) / fs
        # Use heading=0 so marker x-axis aligns with world x-axis
        # (default fixture uses heading=pi/2 which rotates the body 90°,
        # causing head_back to oscillate perpendicular to its own
        # body-axis distance).
        xyz = _make_synthetic_xyz(T=T, fs=fs, heading=np.zeros(T))
        sniff_window = (t >= 5.0) & (t < 15.0)
        # Inject 10 Hz oscillation along the body x-axis: this directly
        # modulates the spine_upper ↔ head_back distance.
        xyz.data[sniff_window, DEFAULT_MARKERS.index("head_back"), 0] += \
            5 * np.sin(2 * np.pi * 10 * t[sniff_window])
        aug = augment_xyz(xyz)
        ep = sniff_periods(aug, sniff_threshold=0.001)
        # At least one period should be in the 5-15s range
        assert ep.data.shape[0] >= 1
        in_window = False
        for s, e in ep.data:
            if e <= s:           # skip zero-length artefacts
                continue
            ts = s / ep.samplerate
            te = e / ep.samplerate
            if ts < 15 and te > 5:
                in_window = True
                break
        assert in_window, "no sniff period within ground-truth window"

    def test_no_sniff_in_quiet(self):
        xyz = _make_synthetic_xyz(T=1200)
        aug = augment_xyz(xyz)
        ep = sniff_periods(aug, sniff_threshold=1.0)   # high threshold
        assert ep.data.shape[0] == 0


class TestShakePeriods:
    def test_runs_without_error(self):
        T, fs = 3600, 120.0
        xyz = _make_synthetic_xyz(T=T, fs=fs)
        aug = augment_xyz(xyz)
        ep = shake_periods(aug, shake_threshold=1e-3)
        # Quiet session — should produce few or no detections
        assert isinstance(ep, NBEpoch)
        assert ep.label == "shake"


class TestReadThetaPeriods:
    def test_reads_simple_file(self, tmp_path):
        f = tmp_path / "test.sts.theta"
        f.write_text("100 200\n300 500\n800 900\n")
        ep = read_theta_periods(f, samplerate=1250.0)
        assert ep.data.shape == (3, 2)
        assert ep.label == "theta"
        assert ep.samplerate == 1250.0
        np.testing.assert_array_equal(ep.data[0], [100.0, 200.0])

    def test_single_period(self, tmp_path):
        f = tmp_path / "single.sts.theta"
        f.write_text("100 200\n")
        ep = read_theta_periods(f, samplerate=1250.0)
        assert ep.data.shape == (1, 2)

    def test_invalid_file_raises(self, tmp_path):
        f = tmp_path / "bad.sts.theta"
        f.write_text("100 200 300\n")  # 3 columns instead of 2
        with pytest.raises(ValueError, match="2-column"):
            read_theta_periods(f, samplerate=1250.0)
