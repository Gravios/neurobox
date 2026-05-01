"""Tests for neurobox.config.standards."""

from __future__ import annotations

import pytest


# ─────────────────────────────────────────────────────────────────────────── #
# Mazes                                                                        #
# ─────────────────────────────────────────────────────────────────────────── #

class TestMazes:

    def test_load_mazes_count(self):
        from neurobox.config import load_mazes
        mazes = load_mazes()
        assert len(mazes) == 23

    def test_known_codes(self):
        """A few sentinel mazes we expect to be present."""
        from neurobox.config import load_mazes
        mazes = load_mazes()
        for code in ("cof", "chr", "rov", "vrr", "lin", "tm"):
            assert code in mazes, f"{code!r} missing from maze table"

    def test_cof_geometry(self):
        from neurobox.config import get_maze
        cof = get_maze("cof")
        assert cof.shape == "circle"
        # MATLAB: visual {[-500,500;-500,500;0,300]}, comp z extends to 360
        assert cof.visual_extent_xyz_mm == (
            (-500.0, 500.0), (-500.0, 500.0), (0.0, 300.0)
        )
        assert cof.computational_extent_xyz_mm == (
            (-500.0, 500.0), (-500.0, 500.0), (0.0, 360.0)
        )

    def test_rov_geometry(self):
        """rov: rectangle with different visual / comp extents."""
        from neurobox.config import get_maze
        rov = get_maze("rov")
        assert rov.shape == "rectangle"
        assert rov.visual_extent_xyz_mm == (
            (-800.0, 800.0), (-500.0, 500.0), (0.0, 300.0)
        )
        # Comp extent is smaller — defines the "valid" maze area
        assert rov.computational_extent_xyz_mm == (
            (-600.0, 600.0), (-350.0, 350.0), (0.0, 300.0)
        )

    def test_vrr_y_axis_extended(self):
        """vrr is the asymmetric VR maze: y wider than x."""
        from neurobox.config import get_maze
        vrr = get_maze("vrr")
        assert vrr.visual_extent("x") == (-500.0, 500.0)
        assert vrr.visual_extent("y") == (-1000.0, 1000.0)

    def test_axis_lookup(self):
        from neurobox.config import get_maze
        cof = get_maze("cof")
        # x/y/z accepted (case-insensitive)
        assert cof.visual_extent("x") == (-500.0, 500.0)
        assert cof.visual_extent("X") == (-500.0, 500.0)
        assert cof.visual_extent("z") == (0.0, 300.0)
        # numeric strings also accepted
        assert cof.visual_extent("0") == cof.visual_extent("x")

    def test_axis_unknown_raises(self):
        from neurobox.config import get_maze
        cof = get_maze("cof")
        with pytest.raises(ValueError, match="Unknown axis"):
            cof.visual_extent("w")

    def test_unknown_maze_raises(self):
        from neurobox.config import get_maze
        with pytest.raises(KeyError, match="Unknown maze code"):
            get_maze("xyz")

    def test_immutability(self):
        from neurobox.config import get_maze
        from dataclasses import FrozenInstanceError
        cof = get_maze("cof")
        with pytest.raises(FrozenInstanceError):
            cof.shape = "square"

    def test_extents_are_tuples(self):
        """Tuples (not lists) so MazeInfo stays hashable / immutable."""
        from neurobox.config import get_maze
        cof = get_maze("cof")
        assert isinstance(cof.visual_extent_xyz_mm, tuple)
        assert all(isinstance(row, tuple) for row in cof.visual_extent_xyz_mm)


# ─────────────────────────────────────────────────────────────────────────── #
# Markers                                                                      #
# ─────────────────────────────────────────────────────────────────────────── #

class TestMarkers:

    def test_marker_count(self):
        from neurobox.config import load_markers
        assert len(load_markers()) == 24

    def test_well_known_markers_present(self):
        from neurobox.config import load_markers
        markers = load_markers()
        for name in (
            "hip_right", "pelvis_root", "spine_lower",
            "spine_upper", "head_back", "head_front",
            "Head", "Body",
        ):
            assert name in markers, f"{name!r} missing from canonical markers"

    def test_marker_order_stable(self):
        """First markers match MATLAB MTAMarkers ordering exactly."""
        from neurobox.config import load_markers
        markers = load_markers()
        assert markers[0] == "hip_right"
        assert markers[1] == "pelvis_root"
        assert markers[2] == "hip_left"
        assert markers[3] == "spine_lower"


# ─────────────────────────────────────────────────────────────────────────── #
# Marker connections                                                           #
# ─────────────────────────────────────────────────────────────────────────── #

class TestMarkerConnections:

    def test_connection_count(self):
        from neurobox.config import load_marker_connections
        # Original MATLAB has 23 pairs
        assert len(load_marker_connections()) == 23

    def test_known_edges_present(self):
        from neurobox.config import load_marker_connections
        edges = set(tuple(c) for c in load_marker_connections())
        # Head triangle
        assert ("head_front", "head_left") in edges
        assert ("head_front", "head_right") in edges
        assert ("head_right", "head_left") in edges
        # Spine chain
        assert ("spine_upper", "spine_middle") in edges
        assert ("spine_middle", "pelvis_root") in edges
        # Hip connections
        assert ("pelvis_root", "hip_right") in edges
        assert ("hip_left", "hip_right") in edges


# ─────────────────────────────────────────────────────────────────────────── #
# make_standard_model                                                          #
# ─────────────────────────────────────────────────────────────────────────── #

class TestMakeStandardModel:

    def test_default_returns_all_markers(self):
        from neurobox.config import (
            load_markers, load_marker_connections, make_standard_model,
        )
        m = make_standard_model()
        assert m.markers == load_markers()
        # Note: the MATLAB MTAMarkers list and MTAMarkerConnections list
        # are not perfectly consistent — one connection involves
        # 'tail_proximal' which isn't in MTAMarkers.  The default model
        # therefore yields 22 connections (23 source edges minus 1
        # filtered out for the missing endpoint).
        all_edges = load_marker_connections()
        marker_set = set(load_markers())
        expected = sum(
            1 for a, b in all_edges
            if a in marker_set and b in marker_set
        )
        assert len(m.connections) == expected

    def test_subset_filters_connections(self):
        """When requesting a subset, edges with absent endpoints are dropped."""
        from neurobox.config import make_standard_model
        m = make_standard_model([
            "head_back", "head_left", "head_front", "head_right",
        ])
        # All edges should reference only these four markers
        marker_set = set(m.markers)
        for a, b in m.connections:
            assert a in marker_set
            assert b in marker_set
        # Triangle of head edges, plus head_back edges to the others = 5 edges
        # head_front-left, head_front-right, head_right-left, head_left-back, head_right-back
        assert len(m.connections) == 5

    def test_subset_preserves_marker_order(self):
        """The model's marker list keeps the user-supplied order."""
        from neurobox.config import make_standard_model
        order = ["head_right", "head_left", "head_back", "head_front"]
        m = make_standard_model(order)
        assert m.markers == order

    def test_only_present_connections_false(self):
        """With only_present_connections=False, all edges are returned."""
        from neurobox.config import make_standard_model
        m = make_standard_model(
            ["head_back"], only_present_connections=False
        )
        # All 23 edges, even ones with markers not in the list
        assert len(m.connections) == 23

    def test_custom_marker_passthrough(self):
        """User-defined marker names not in the canonical list are kept."""
        from neurobox.config import make_standard_model
        m = make_standard_model([
            "head_back", "spine_upper", "custom_widget",
        ])
        assert "custom_widget" in m.markers
        # Edges from the standard list that connect head_back ↔ spine_upper
        # might or might not appear; just verify no edge mentions custom_widget
        for a, b in m.connections:
            assert a != "custom_widget"
            assert b != "custom_widget"


# ─────────────────────────────────────────────────────────────────────────── #
# Caching                                                                      #
# ─────────────────────────────────────────────────────────────────────────── #

class TestCaching:

    def test_load_mazes_returns_fresh_dict(self):
        """load_mazes() returns a fresh dict each call (callers may mutate)."""
        from neurobox.config import load_mazes
        a = load_mazes()
        b = load_mazes()
        assert a is not b
        a.pop("cof")
        # b is unaffected
        assert "cof" in b

    def test_load_markers_returns_fresh_list(self):
        from neurobox.config import load_markers
        a = load_markers()
        b = load_markers()
        # Fresh copies — caller can mutate without affecting next call
        a.append("test")
        b2 = load_markers()
        assert "test" not in b2
