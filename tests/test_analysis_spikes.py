"""Tests for neurobox.analysis.spikes.ccg.

The same test suite runs against both kernels (Cython compiled and pure-Python
fallback) via parametrisation, so we can verify they produce identical output.
"""

from __future__ import annotations

import numpy as np
import pytest


# ─────────────────────────────────────────────────────────────────────────── #
# Reference implementation — exhaustive O(n²) for small inputs                  #
# ─────────────────────────────────────────────────────────────────────────── #

def _ccg_reference(times, clu, bin_size, half_bins, n_groups):
    """Brute-force O(n²) reference matching CCGHeart.c semantics exactly."""
    times = np.asarray(times, dtype=np.float64)
    clu = np.asarray(clu, dtype=np.int64)
    n_bins = 1 + 2 * half_bins
    furthest_edge = bin_size * (half_bins + 0.5)
    counts = np.zeros((n_bins, n_groups, n_groups), dtype=np.int64)
    for i in range(times.size):
        m1 = clu[i]
        if not (0 <= m1 < n_groups):
            continue
        t1 = times[i]
        for j in range(times.size):
            if i == j:
                continue
            m2 = clu[j]
            if not (0 <= m2 < n_groups):
                continue
            t2 = times[j]
            # Match C asymmetry: backward strict >, forward strict >=
            if j < i:
                if t1 - t2 > furthest_edge:
                    continue
            else:
                if t2 - t1 >= furthest_edge:
                    continue
            bin_idx = half_bins + int(np.floor(0.5 + (t2 - t1) / bin_size))
            if 0 <= bin_idx < n_bins:
                counts[bin_idx, m1, m2] += 1
    return counts


# ─────────────────────────────────────────────────────────────────────────── #
# Kernel parametrisation                                                       #
# ─────────────────────────────────────────────────────────────────────────── #

def _get_kernels():
    """Yield (label, counts_fn, pairs_fn) for every available kernel."""
    kernels: list[tuple[str, callable, callable]] = []
    # Pure-Python fallback always available
    from neurobox.analysis.spikes._ccg_python_fallback import (
        compute_ccg_counts as py_counts,
        compute_ccg_counts_with_pairs as py_pairs,
    )
    kernels.append(("python", py_counts, py_pairs))
    # Cython if compiled
    try:
        from neurobox.analysis.spikes._ccg_engine import (
            compute_ccg_counts as cy_counts,
            compute_ccg_counts_with_pairs as cy_pairs,
        )
        kernels.append(("cython", cy_counts, cy_pairs))
    except ImportError:
        pass
    return kernels


KERNELS = _get_kernels()


@pytest.fixture(params=KERNELS, ids=lambda k: k[0])
def kernel(request):
    """Yields (label, counts_fn, pairs_fn) for each available kernel."""
    return request.param


# ─────────────────────────────────────────────────────────────────────────── #
# Kernel tests — apply to both Cython and Python                              #
# ─────────────────────────────────────────────────────────────────────────── #

class TestKernels:
    """Both kernels produce identical output to the brute-force reference."""

    def test_simple_two_spikes(self, kernel):
        _, counts_fn, _ = kernel
        # Two spikes 50 samples apart, both in group 0
        times = np.array([100.0, 150.0])
        clu   = np.array([0, 0], dtype=np.int64)
        c = counts_fn(times, clu, bin_size=20.0, half_bins=10, n_groups=1)
        ref = _ccg_reference(times, clu, bin_size=20.0, half_bins=10, n_groups=1)
        np.testing.assert_array_equal(c, ref)

    def test_random_against_reference(self, kernel):
        _, counts_fn, _ = kernel
        rng = np.random.default_rng(0)
        # 200 spikes in 10 s at 20 kHz, 3 groups
        n = 200
        times = np.sort(rng.uniform(0, 200_000, size=n))
        clu = rng.integers(0, 3, size=n)
        c   = counts_fn(times, clu, bin_size=20.0, half_bins=25, n_groups=3)
        ref = _ccg_reference(times, clu, bin_size=20.0, half_bins=25, n_groups=3)
        np.testing.assert_array_equal(c, ref)

    def test_pairs_count_matches_total_counts(self, kernel):
        _, counts_fn, pairs_fn = kernel
        rng = np.random.default_rng(1)
        n = 100
        times = np.sort(rng.uniform(0, 100_000, size=n))
        clu = rng.integers(0, 2, size=n)
        c1 = counts_fn(times, clu, bin_size=20.0, half_bins=15, n_groups=2)
        c2, pairs = pairs_fn(times, clu, bin_size=20.0, half_bins=15, n_groups=2)
        np.testing.assert_array_equal(c1, c2)
        # Total counts should equal number of pairs
        assert pairs.shape[0] == c2.sum()
        # Pairs are valid indices
        assert (pairs >= 0).all() and (pairs < n).all()

    def test_self_pair_excluded(self, kernel):
        """Lag-0 bin should NOT contain spike pairs of (i, i)."""
        _, counts_fn, _ = kernel
        # Single spike: nothing to histogram
        c = counts_fn(np.array([100.0]),
                      np.array([0], dtype=np.int64),
                      bin_size=20.0, half_bins=5, n_groups=1)
        assert c.sum() == 0

    def test_invalid_inputs_raise(self, kernel):
        _, counts_fn, _ = kernel
        with pytest.raises(ValueError):
            counts_fn(np.array([1.0, 2.0]),
                      np.array([0], dtype=np.int64),  # mismatched length
                      bin_size=1.0, half_bins=5, n_groups=1)
        with pytest.raises(ValueError):
            counts_fn(np.array([1.0]), np.array([0], dtype=np.int64),
                      bin_size=-1.0, half_bins=5, n_groups=1)
        with pytest.raises(ValueError):
            counts_fn(np.array([1.0]), np.array([0], dtype=np.int64),
                      bin_size=1.0, half_bins=-1, n_groups=1)


class TestKernelEquivalence:
    """If both Cython and Python kernels are available, they produce
    bit-for-bit identical output."""

    @pytest.mark.skipif(len(KERNELS) < 2, reason="Cython kernel not built")
    def test_kernels_identical(self):
        py = next(k for k in KERNELS if k[0] == "python")
        cy = next(k for k in KERNELS if k[0] == "cython")
        rng = np.random.default_rng(42)
        for trial in range(5):
            n = rng.integers(50, 500)
            times = np.sort(rng.uniform(0, 1_000_000, size=n))
            clu = rng.integers(0, 4, size=n).astype(np.int64)
            c_py = py[1](times, clu, 50.0, 20, 4)
            c_cy = cy[1](times, clu, 50.0, 20, 4)
            np.testing.assert_array_equal(c_py, c_cy,
                err_msg=f"kernels diverged on trial {trial} (n={n})")


# ─────────────────────────────────────────────────────────────────────────── #
# High-level ccg() wrapper                                                     #
# ─────────────────────────────────────────────────────────────────────────── #

class TestCcgWrapper:

    def test_auto_correlogram_period(self):
        """Auto-CCG of a regular spike train has peaks at multiples of period."""
        from neurobox.analysis.spikes import ccg
        # 100 Hz spike train at 20 kHz: period 200 samples
        times = np.arange(100, 200_000, 200, dtype=np.float64)
        groups = np.ones(times.size, dtype=int)
        result = ccg(times, groups, bin_size=20, half_bins=15,
                     sample_rate=20000.0, normalization="count")
        # Auto-CCG: peaks at lag 0 (centre = each spike, but no self-pair),
        # ±200 samples (10 ms = 10 bins), ±400 samples (20 ms), etc.
        nz_bins = np.flatnonzero(result.ccg[:, 0, 0])
        nz_lags_ms = result.t[nz_bins]
        # Should see multiples of 10 ms within ±15 ms
        # half_bins=15 → window is ±15 ms.  10 ms peaks → bins at -10, +10 ms only
        # (no lag 0, no ±20 ms within window).  Bin centres are integer ms here.
        assert set(nz_lags_ms.astype(int).tolist()) == {-10, 10}

    def test_sign_convention(self):
        """ccg[:, A, B] at positive lag = B after A."""
        from neurobox.analysis.spikes import ccg
        # A at 100, 300, …; B at 150, 350, … (B is 50 samples = +2.5 ms after A)
        times_a = np.arange(100, 200_000, 200, dtype=np.float64)
        times_b = np.arange(150, 200_000, 200, dtype=np.float64)
        times = np.concatenate([times_a, times_b])
        groups = np.concatenate([np.ones(times_a.size, dtype=int),
                                 2 * np.ones(times_b.size, dtype=int)])
        result = ccg(times, groups, bin_size=20, half_bins=10,
                     sample_rate=20000.0, group_subset=[1, 2],
                     normalization="count")
        # ccg[:, 0, 1] is "B relative to A" — B is 2.5 ms after A → peak at +3 ms.
        # Also peak at -7 ms (the previous B fires 7.5 ms before next A).
        nz_ab = result.t[np.flatnonzero(result.ccg[:, 0, 1])]
        assert 3 in nz_ab.astype(int).tolist()
        assert -7 in nz_ab.astype(int).tolist()
        # ccg[:, 1, 0] should be the time-reversed version
        np.testing.assert_array_equal(
            result.ccg[:, 1, 0], result.ccg[::-1, 0, 1]
        )

    def test_normalization_count(self):
        from neurobox.analysis.spikes import ccg
        rng = np.random.default_rng(0)
        times = np.sort(rng.uniform(0, 100_000, size=100))
        groups = rng.integers(1, 3, size=100)
        result = ccg(times, groups, bin_size=20, half_bins=10,
                     sample_rate=20000.0, group_subset=[1, 2],
                     normalization="count")
        assert result.ccg.dtype == np.int64
        assert result.normalization == "count"
        assert result.axis_unit == "(Spikes)"

    def test_normalization_hz_correct_factor(self):
        """Hz mode: ccg[:, i, j] = counts[:, i, j] * (Fs / (bin_size * n_i))."""
        from neurobox.analysis.spikes import ccg
        rng = np.random.default_rng(0)
        n = 200
        times = np.sort(rng.uniform(0, 200_000, size=n))
        groups = rng.integers(1, 3, size=n)
        result_count = ccg(times, groups, bin_size=20, half_bins=10,
                           sample_rate=20000.0, group_subset=[1, 2],
                           normalization="count")
        result_hz = ccg(times, groups, bin_size=20, half_bins=10,
                        sample_rate=20000.0, group_subset=[1, 2],
                        normalization="hz")
        # Verify Hz scaling on auto: ccg_hz[:, 0, 0] = ccg_count[:, 0, 0] * Fs / (bin_size * n_0)
        n_0 = result_hz.n_spikes_per_group[0]
        if n_0 > 0:
            expected = result_count.ccg[:, 0, 0] * (20000.0 / (20 * n_0))
            np.testing.assert_allclose(result_hz.ccg[:, 0, 0], expected, rtol=1e-12)

    def test_empty_train_returns_zeros(self):
        from neurobox.analysis.spikes import ccg
        result = ccg(np.array([]), np.array([], dtype=int), bin_size=20,
                     half_bins=10, sample_rate=20000.0,
                     group_subset=[1, 2], normalization="hz")
        assert result.ccg.shape == (21, 2, 2)
        assert (result.ccg == 0).all()
        # Empty case still gives a valid t axis
        assert result.t.size == 21

    def test_pairs_returned_when_requested(self):
        from neurobox.analysis.spikes import ccg
        rng = np.random.default_rng(0)
        times = np.sort(rng.uniform(0, 100_000, size=50))
        groups = rng.integers(1, 3, size=50)
        result = ccg(times, groups, bin_size=20, half_bins=10,
                     sample_rate=20000.0, group_subset=[1, 2],
                     normalization="count", return_pairs=True)
        assert result.pairs.ndim == 2
        assert result.pairs.shape[1] == 2
        # Pairs index back into the ORIGINAL (unsorted) times array
        if result.pairs.size > 0:
            assert (result.pairs >= 0).all()
            assert (result.pairs < 50).all()

    def test_unsorted_input_is_sorted_internally(self):
        """The wrapper must sort by time before invoking the kernel."""
        from neurobox.analysis.spikes import ccg
        times_sorted   = np.array([100.0, 200.0, 350.0, 500.0])
        groups_sorted  = np.array([1, 2, 1, 2])
        # Shuffle
        perm = np.array([2, 0, 3, 1])
        result_sorted = ccg(times_sorted, groups_sorted, bin_size=20,
                            half_bins=10, sample_rate=20000.0,
                            group_subset=[1, 2], normalization="count")
        result_shuffled = ccg(times_sorted[perm], groups_sorted[perm],
                              bin_size=20, half_bins=10, sample_rate=20000.0,
                              group_subset=[1, 2], normalization="count")
        np.testing.assert_array_equal(result_sorted.ccg, result_shuffled.ccg)

    def test_unknown_normalization_raises(self):
        from neurobox.analysis.spikes import ccg
        times = np.array([100.0, 200.0])
        groups = np.array([1, 2])
        with pytest.raises(ValueError, match="normalization"):
            ccg(times, groups, bin_size=20, half_bins=10,
                normalization="bogus")

    def test_epochs_filter(self):
        """Spikes outside epochs should be excluded."""
        from neurobox.analysis.spikes import ccg
        times = np.array([100.0, 200.0, 5000.0, 5100.0, 10_000.0])
        groups = np.array([1, 1, 1, 1, 1])
        # Epoch keeping only the middle two spikes
        epochs = np.array([[4500.0, 5500.0]])
        result = ccg(times, groups, bin_size=20, half_bins=10,
                     sample_rate=20000.0, group_subset=[1],
                     normalization="count", epochs=epochs)
        # Only 2 spikes (indices 2 and 3) are inside, separated by 100 samples → ±5 ms.
        # Auto-CCG: 2 entries (one for each centre)
        # For centre at 5000, other at 5100: bin = 10 + round(100/20) = 15 (>0)
        # For centre at 5100, other at 5000: bin = 10 + round(-100/20) = 5 (<0)
        nz = np.flatnonzero(result.ccg[:, 0, 0])
        assert set(nz.tolist()) == {5, 15}


# ─────────────────────────────────────────────────────────────────────────── #
# trains_to_ccg                                                                #
# ─────────────────────────────────────────────────────────────────────────── #

class TestTrainsToCcg:

    def test_basic_three_trains(self):
        from neurobox.analysis.spikes import trains_to_ccg
        rng = np.random.default_rng(0)
        trains = [
            np.sort(rng.uniform(0, 100_000, size=100)),
            np.sort(rng.uniform(0, 100_000, size=80)),
            np.sort(rng.uniform(0, 100_000, size=120)),
        ]
        result = trains_to_ccg(trains, bin_size_ms=1.0, half_bins=20,
                               sample_rate=20000.0, normalization="scale")
        assert result.ccg.shape == (41, 3, 3)
        assert result.normalization == "scale"
        # bin_size_samples = ceil(1.0 ms * 20000 Hz / 1000) = 20
        assert result.bin_size_samples == 20

    def test_empty_trains_dropped(self):
        from neurobox.analysis.spikes import trains_to_ccg
        # First train empty → dropped
        trains = [
            np.array([]),
            np.array([100.0, 200.0, 300.0]),
            np.array([150.0, 250.0]),
        ]
        result = trains_to_ccg(trains, bin_size_ms=1.0, half_bins=10,
                               sample_rate=20000.0)
        # Should have 2 groups left, not 3
        assert result.ccg.shape == (21, 2, 2)
        assert result.n_groups == 2

    def test_all_empty_raises(self):
        from neurobox.analysis.spikes import trains_to_ccg
        with pytest.raises(ValueError, match="all trains are empty"):
            trains_to_ccg([np.array([])], bin_size_ms=1.0, half_bins=10)

    def test_period_filter(self):
        from neurobox.analysis.spikes import trains_to_ccg
        # 4 spikes; period only catches the middle two
        trains = [np.array([100.0, 5000.0, 5100.0, 10_000.0])]
        period = np.array([[4500.0, 5500.0]])
        result = trains_to_ccg(trains, bin_size_ms=1.0, half_bins=10,
                               sample_rate=20000.0, period=period,
                               normalization="count")
        # 2 spikes survive → auto-CCG with 2 entries
        assert result.ccg[:, 0, 0].sum() == 2


# ─────────────────────────────────────────────────────────────────────────── #
# is_compiled                                                                  #
# ─────────────────────────────────────────────────────────────────────────── #

def test_is_compiled_returns_bool():
    from neurobox.analysis.spikes import is_compiled
    assert isinstance(is_compiled(), bool)
