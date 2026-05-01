"""Tests for neurobox.analysis.lfp.ranges."""

from __future__ import annotations

import numpy as np
import pytest


# ─────────────────────────────────────────────────────────────────────────── #
# join_ranges                                                                  #
# ─────────────────────────────────────────────────────────────────────────── #

class TestJoinRanges:

    def test_disjoint_pair(self):
        from neurobox.analysis.lfp import join_ranges
        A = np.array([[0, 5]])
        B = np.array([[10, 15]])
        out = join_ranges(A, B)
        np.testing.assert_array_equal(out, [[0, 5], [10, 15]])

    def test_overlapping_pair(self):
        from neurobox.analysis.lfp import join_ranges
        A = np.array([[0, 5], [10, 15]])
        B = np.array([[3, 7], [20, 25]])
        out = join_ranges(A, B)
        np.testing.assert_array_equal(out, [[0, 7], [10, 15], [20, 25]])

    def test_touching_ranges_merge(self):
        from neurobox.analysis.lfp import join_ranges
        A = np.array([[0, 5]])
        B = np.array([[5, 10]])
        out = join_ranges(A, B)
        # Touching at single point → merged
        np.testing.assert_array_equal(out, [[0, 10]])

    def test_three_arrays(self):
        from neurobox.analysis.lfp import join_ranges
        A = np.array([[0, 2]])
        B = np.array([[4, 6]])
        C = np.array([[1, 5]])  # bridges A and B
        out = join_ranges(A, B, C)
        np.testing.assert_array_equal(out, [[0, 6]])

    def test_empty_inputs(self):
        from neurobox.analysis.lfp import join_ranges
        empty = np.empty((0, 2))
        A = np.array([[0, 5]])
        out = join_ranges(empty, A)
        np.testing.assert_array_equal(out, [[0, 5]])
        out = join_ranges(empty, empty)
        assert out.shape == (0, 2)
        out = join_ranges()
        assert out.shape == (0, 2)

    def test_invalid_range_raises(self):
        from neurobox.analysis.lfp import join_ranges
        with pytest.raises(ValueError, match="start ≤ stop"):
            join_ranges(np.array([[5, 3]]))

    def test_canonical_output(self):
        """Output is sorted, non-overlapping, no zero-length."""
        from neurobox.analysis.lfp import join_ranges
        A = np.array([[10, 15], [0, 5], [3, 4]])  # unsorted, contains overlap
        out = join_ranges(A)
        assert np.all(np.diff(out[:, 0]) >= 0), "starts must be sorted"
        assert np.all(out[:, 1] > out[:, 0]),   "no zero-length"
        # Adjacent intervals must not overlap
        if out.shape[0] > 1:
            assert np.all(out[:-1, 1] <= out[1:, 0])


# ─────────────────────────────────────────────────────────────────────────── #
# intersect_ranges                                                             #
# ─────────────────────────────────────────────────────────────────────────── #

class TestIntersectRanges:

    def test_basic_intersection(self):
        from neurobox.analysis.lfp import intersect_ranges
        A = np.array([[0, 10], [20, 30]])
        B = np.array([[5, 25]])
        out = intersect_ranges(A, B)
        np.testing.assert_array_equal(out, [[5, 10], [20, 25]])

    def test_no_overlap(self):
        from neurobox.analysis.lfp import intersect_ranges
        A = np.array([[0, 5]])
        B = np.array([[10, 15]])
        out = intersect_ranges(A, B)
        assert out.shape == (0, 2)

    def test_one_contains_other(self):
        from neurobox.analysis.lfp import intersect_ranges
        A = np.array([[0, 100]])
        B = np.array([[20, 30], [40, 50]])
        out = intersect_ranges(A, B)
        np.testing.assert_array_equal(out, [[20, 30], [40, 50]])

    def test_identity(self):
        from neurobox.analysis.lfp import intersect_ranges
        A = np.array([[0, 10], [20, 30]])
        out = intersect_ranges(A, A)
        np.testing.assert_array_equal(out, A)

    def test_empty_input(self):
        from neurobox.analysis.lfp import intersect_ranges
        out = intersect_ranges(np.empty((0, 2)), np.array([[0, 10]]))
        assert out.shape == (0, 2)


# ─────────────────────────────────────────────────────────────────────────── #
# subtract_ranges                                                              #
# ─────────────────────────────────────────────────────────────────────────── #

class TestSubtractRanges:

    def test_holes_in_middle(self):
        from neurobox.analysis.lfp import subtract_ranges
        A = np.array([[0, 10]])
        B = np.array([[3, 5], [7, 8]])
        out = subtract_ranges(A, B)
        np.testing.assert_array_equal(out, [[0, 3], [5, 7], [8, 10]])

    def test_b_extends_outside_a(self):
        from neurobox.analysis.lfp import subtract_ranges
        A = np.array([[5, 10]])
        B = np.array([[0, 7], [9, 15]])
        out = subtract_ranges(A, B)
        np.testing.assert_array_equal(out, [[7, 9]])

    def test_b_subset_of_a(self):
        from neurobox.analysis.lfp import subtract_ranges
        A = np.array([[0, 10]])
        B = np.array([[3, 7]])
        out = subtract_ranges(A, B)
        np.testing.assert_array_equal(out, [[0, 3], [7, 10]])

    def test_disjoint(self):
        from neurobox.analysis.lfp import subtract_ranges
        A = np.array([[0, 5]])
        B = np.array([[10, 20]])
        out = subtract_ranges(A, B)
        np.testing.assert_array_equal(out, [[0, 5]])

    def test_empty_b_returns_a(self):
        """The labbox MATLAB version crashes on empty B; ours doesn't."""
        from neurobox.analysis.lfp import subtract_ranges
        A = np.array([[0, 10]])
        out = subtract_ranges(A, np.empty((0, 2)))
        np.testing.assert_array_equal(out, A)

    def test_full_subtraction_yields_empty(self):
        from neurobox.analysis.lfp import subtract_ranges
        A = np.array([[0, 10]])
        B = np.array([[0, 10]])
        out = subtract_ranges(A, B)
        assert out.shape == (0, 2)

    def test_multi_a_multi_b(self):
        from neurobox.analysis.lfp import subtract_ranges
        A = np.array([[0, 10], [20, 30]])
        B = np.array([[5, 8], [22, 28]])
        out = subtract_ranges(A, B)
        np.testing.assert_array_equal(out, [[0, 5], [8, 10], [20, 22], [28, 30]])


# ─────────────────────────────────────────────────────────────────────────── #
# complement_ranges                                                            #
# ─────────────────────────────────────────────────────────────────────────── #

class TestComplementRanges:

    def test_bounded_extent(self):
        from neurobox.analysis.lfp import complement_ranges
        A = np.array([[5, 10], [20, 30]])
        out = complement_ranges(A, extent=(0, 40))
        np.testing.assert_array_equal(out, [[0, 5], [10, 20], [30, 40]])

    def test_unbounded(self):
        from neurobox.analysis.lfp import complement_ranges
        A = np.array([[5, 10]])
        out = complement_ranges(A)
        assert out[0, 0] == -np.inf
        assert out[-1, 1] == np.inf

    def test_empty_ranges_returns_full_extent(self):
        from neurobox.analysis.lfp import complement_ranges
        out = complement_ranges(np.empty((0, 2)), extent=(0, 100))
        np.testing.assert_array_equal(out, [[0, 100]])

    def test_complement_then_complement(self):
        """A == complement(complement(A)) over a bounded extent."""
        from neurobox.analysis.lfp import complement_ranges
        A = np.array([[5, 10], [20, 30]])
        ext = (0, 40)
        roundtrip = complement_ranges(complement_ranges(A, ext), ext)
        np.testing.assert_array_equal(roundtrip, A)


# ─────────────────────────────────────────────────────────────────────────── #
# Cross-checks: sweep-line consistency                                         #
# ─────────────────────────────────────────────────────────────────────────── #

class TestRangesIdentities:
    """Test set-theoretic identities between the four operations."""

    @pytest.fixture
    def random_pair(self):
        rng = np.random.default_rng(0)
        def _gen(n):
            x = np.sort(rng.uniform(0, 100, size=n * 2)).reshape(n, 2)
            return np.column_stack([np.min(x, axis=1), np.max(x, axis=1)])
        return _gen(8), _gen(8)

    def test_de_morgan(self, random_pair):
        """A ∪ B == complement(complement(A) ∩ complement(B)) over a bound."""
        from neurobox.analysis.lfp import (
            join_ranges, intersect_ranges, complement_ranges,
        )
        A, B = random_pair
        ext = (-10, 110)
        lhs = join_ranges(A, B)
        rhs = complement_ranges(
            intersect_ranges(
                complement_ranges(A, ext),
                complement_ranges(B, ext),
            ),
            ext,
        )
        # Both sides should agree within the extent
        # (truncate lhs to the extent for comparison)
        from neurobox.analysis.lfp import intersect_ranges as ir
        lhs_in = ir(lhs, np.array([[ext[0], ext[1]]]))
        np.testing.assert_allclose(lhs_in, rhs)

    def test_subtract_via_intersect_complement(self, random_pair):
        """A − B == A ∩ complement(B)."""
        from neurobox.analysis.lfp import (
            subtract_ranges, intersect_ranges, complement_ranges,
        )
        A, B = random_pair
        lhs = subtract_ranges(A, B)
        rhs = intersect_ranges(A, complement_ranges(B))
        np.testing.assert_allclose(lhs, rhs)

    def test_a_minus_a_is_empty(self, random_pair):
        from neurobox.analysis.lfp import subtract_ranges
        A, _ = random_pair
        out = subtract_ranges(A, A)
        assert out.shape == (0, 2)

    def test_intersect_commutative(self, random_pair):
        from neurobox.analysis.lfp import intersect_ranges
        A, B = random_pair
        np.testing.assert_array_equal(
            intersect_ranges(A, B), intersect_ranges(B, A)
        )

    def test_join_commutative(self, random_pair):
        from neurobox.analysis.lfp import join_ranges
        A, B = random_pair
        np.testing.assert_array_equal(join_ranges(A, B), join_ranges(B, A))
