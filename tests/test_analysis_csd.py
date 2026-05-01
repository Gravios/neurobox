"""Tests for neurobox.analysis.lfp.csd."""

from __future__ import annotations

import numpy as np
import pytest


class TestCurrentSourceDensity:

    def test_shape_step_one(self):
        from neurobox.analysis.lfp import current_source_density
        rng = np.random.default_rng(0)
        lfp = rng.standard_normal((1000, 8))
        result = current_source_density(lfp, samplerate=1250.0, step=1)
        # n_channels - 2*step = 8 - 2 = 6
        assert result.csd.shape == (1000, 6)
        assert result.t.size == 1000
        assert result.z.size == 6

    def test_shape_step_two(self):
        from neurobox.analysis.lfp import current_source_density
        rng = np.random.default_rng(0)
        lfp = rng.standard_normal((500, 8))
        result = current_source_density(lfp, samplerate=1250.0, step=2)
        # n_channels - 2*step = 8 - 4 = 4
        assert result.csd.shape == (500, 4)

    def test_sink_positive_convention(self):
        """A current sink (positive ion inflow) → positive CSD value.

        Construct a synthetic potential field with a parabolic minimum
        (a sink) at channel 4.  The CSD of a parabolic V dip is constant
        and positive everywhere it is well-defined.
        """
        from neurobox.analysis.lfp import current_source_density
        n_t, n_ch = 100, 9
        z = np.arange(n_ch, dtype=np.float64)
        # V(z) = +(z - 4)² → CSD = -d²V/dz² = -2 (negative, source)
        # V(z) = -(z - 4)² → CSD = -d²V/dz² = +2 (positive, sink)
        v = -(z - 4.0) ** 2
        lfp = np.broadcast_to(v, (n_t, n_ch)).copy()
        result = current_source_density(lfp, samplerate=1000.0, step=1)
        # All output values should be positive (sink)
        assert np.all(result.csd > 0), \
            f"sink convention violated: csd={result.csd[0]}"
        # Numerical second difference of -(z-4)² is +2 everywhere
        np.testing.assert_allclose(result.csd, 2.0, rtol=1e-12)

    def test_finite_difference_with_pitch(self):
        """With pitch_um, output should equal -∂²V/∂z² in V/m²."""
        from neurobox.analysis.lfp import current_source_density
        # V = -z² (parabola); ∂²V/∂z² = -2; -∂²V/∂z² = +2.
        # With z in metres, CSD = 2 V/m² regardless of step.
        n_ch = 10
        pitch_um = 50.0
        pitch_m = pitch_um * 1e-6  # for the analytic check
        z_m = np.arange(n_ch) * pitch_m
        v = -(z_m ** 2)  # V in volts (assume so)
        lfp = np.broadcast_to(v, (10, n_ch)).copy()
        result = current_source_density(lfp, samplerate=1000.0, step=1,
                                        pitch_um=pitch_um)
        # Expected: CSD = -∂²V/∂z² = 2 V/m²
        np.testing.assert_allclose(result.csd, 2.0, rtol=1e-10)

    def test_step_too_large_raises(self):
        from neurobox.analysis.lfp import current_source_density
        lfp = np.zeros((100, 4))
        with pytest.raises(ValueError, match="step"):
            current_source_density(lfp, step=2)  # 2*step >= n_channels=4

    def test_step_zero_raises(self):
        from neurobox.analysis.lfp import current_source_density
        with pytest.raises(ValueError, match="step"):
            current_source_density(np.zeros((100, 8)), step=0)

    def test_transpose_warning(self):
        """When T < n_channels, a UserWarning is issued (matches labbox)."""
        from neurobox.analysis.lfp import current_source_density
        # 5 timepoints, 16 channels — almost certainly transposed.
        lfp = np.zeros((5, 16))
        with pytest.warns(UserWarning, match="transposed"):
            current_source_density(lfp, step=1)

    def test_default_t_axis_in_ms(self):
        from neurobox.analysis.lfp import current_source_density
        n_t = 1000
        lfp = np.zeros((n_t, 8))
        result = current_source_density(lfp, samplerate=1000.0, step=1)
        # 1000 samples at 1000 Hz → 1000 ms
        assert abs(result.t[-1] - 999.0) < 1e-9
        assert result.t[0] == 0.0

    def test_interp_levels(self):
        from neurobox.analysis.lfp import current_source_density
        rng = np.random.default_rng(0)
        lfp = rng.standard_normal((100, 8))
        result = current_source_density(lfp, samplerate=1000.0,
                                        step=1, interp_levels=2)
        # interp_levels=2 → 2² = 4× upsampling
        assert result.csd_interp.shape[0] == 4 * 100
        assert result.csd_interp.shape[1] == 4 * 6  # 6 = 8 - 2*1
        assert result.t_interp.size == 4 * 100
        assert result.z_interp.size == 4 * 6

    def test_no_interp_returns_none(self):
        from neurobox.analysis.lfp import current_source_density
        lfp = np.zeros((50, 6))
        result = current_source_density(lfp, step=1, interp_levels=0)
        assert result.csd_interp is None
        assert result.t_interp is None
        assert result.z_interp is None

    def test_custom_channels(self):
        from neurobox.analysis.lfp import current_source_density
        lfp = np.zeros((10, 5))
        # Pretend channels are physical IDs [10, 12, 14, 16, 18]
        result = current_source_density(lfp, step=1,
                                        channels=np.array([10, 12, 14, 16, 18]))
        # z axis after step=1 trimming: middle 3 channels [12, 14, 16]
        np.testing.assert_array_equal(result.z, [12, 14, 16])

    def test_result_carries_metadata(self):
        from neurobox.analysis.lfp import current_source_density, CSDResult
        lfp = np.zeros((10, 6))
        result = current_source_density(lfp, samplerate=2500.0, step=1,
                                        pitch_um=25.0)
        assert isinstance(result, CSDResult)
        assert result.samplerate == 2500.0
        assert result.step == 1
        assert result.pitch_um == 25.0
