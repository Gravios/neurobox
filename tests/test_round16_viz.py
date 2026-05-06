"""Tests for round-16 — neurobox.viz layout / plot / state helpers."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

# All viz tests need matplotlib but never an interactive backend.
# matplotlib is an OPTIONAL dependency (extras: [viz]); skip the
# entire module gracefully if it isn't installed.
matplotlib = pytest.importorskip("matplotlib", exc_type=ImportError)
matplotlib.use("Agg")
import matplotlib.pyplot as plt    # noqa: E402

# Import under test
from neurobox.viz import (        # noqa: E402
    setup_figure, setup_axes, setup_axes_named, place_subplot,
    clear_axes_labels, clear_axes_ticks, clear_fax, empty_axis, linkax,
    parse_inkscape_layout,
    NBFigure, Layout,
    imagescnan, error_ellipse, apply_colorbar,
    circular_arrow, draw_arrow, sbar,
    plot_stc, plot_stcs, plot_state_durations, plot_features_with_stc,
)
from neurobox.dtype import NBEpoch, NBStateCollection      # noqa: E402


# ─────────────────────────────────────────────────────────────────────── #
# Layout                                                                     #
# ─────────────────────────────────────────────────────────────────────── #

class TestSetupFigure:
    def teardown_method(self):
        plt.close("all")

    def test_a4_portrait_dimensions(self):
        nbfig = setup_figure(format="A4", layout="portrait")
        assert nbfig.opts.page.width  == pytest.approx(21.0)
        assert nbfig.opts.page.height == pytest.approx(29.7)
        # Figure size in inches
        w_in, h_in = nbfig.figure.get_size_inches()
        assert w_in == pytest.approx(21.0 / 2.54, rel=1e-6)
        assert h_in == pytest.approx(29.7 / 2.54, rel=1e-6)

    def test_a4_landscape_swaps(self):
        nbfig = setup_figure(format="A4", layout="landscape")
        assert nbfig.opts.page.width  == pytest.approx(29.7)
        assert nbfig.opts.page.height == pytest.approx(21.0)

    def test_a3_a2(self):
        a3 = setup_figure(format="A3")
        assert a3.opts.page.width  == pytest.approx(29.7)
        assert a3.opts.page.height == pytest.approx(42.0)
        a2 = setup_figure(format="A2")
        assert a2.opts.page.width  == pytest.approx(42.0)
        assert a2.opts.page.height == pytest.approx(59.4)

    def test_1080p_uses_pixels(self):
        nbfig = setup_figure(format="1080p")
        assert nbfig.opts.page.units == "pixels"
        assert nbfig.opts.page.width  == 1920
        assert nbfig.opts.page.height == 1080

    def test_invalid_format(self):
        with pytest.raises(ValueError, match="format must be"):
            setup_figure(format="A5")

    def test_invalid_layout(self):
        with pytest.raises(ValueError, match="layout must be"):
            setup_figure(format="A4", layout="upside_down")

    def test_xpos_ypos_grids(self):
        nbfig = setup_figure(
            format="A4", layout="portrait",
            margin_left=2.0, margin_top=2.0,
            subplot_width=2.0, subplot_height=2.0,
            subplot_padding_horizontal=0.5,
            subplot_padding_vertical=0.5,
        )
        # xpos starts at margin_left=2.0, step 2.5
        assert nbfig.opts.page.xpos[0] == pytest.approx(2.0)
        assert nbfig.opts.page.xpos[1] == pytest.approx(4.5)
        # ypos descends (top of page first)
        assert nbfig.opts.page.ypos[0] > nbfig.opts.page.ypos[1]


class TestSetupAxes:
    def teardown_method(self):
        plt.close("all")

    def test_creates_axes(self):
        nbfig = setup_figure(format="A4")
        sax = setup_axes(nbfig, yind=4, yoffset=0.0, xind=2, xoffset=0.0)
        assert sax is not None
        # Bbox should be inside the figure (positive in figure-fraction)
        x, y, w, h = sax.get_position().bounds
        assert 0 <= x <= 1 and 0 <= y <= 1
        assert w > 0 and h > 0

    def test_reuse_returns_same_axes(self):
        nbfig = setup_figure(format="A4")
        a = setup_axes(nbfig, yind=4, yoffset=0.0, xind=2, xoffset=0.0)
        b = setup_axes(nbfig, yind=4, yoffset=0.0, xind=2, xoffset=0.0)
        assert a is b

    def test_different_args_different_axes(self):
        nbfig = setup_figure(format="A4")
        a = setup_axes(nbfig, yind=4, yoffset=0.0, xind=2, xoffset=0.0)
        b = setup_axes(nbfig, yind=4, yoffset=0.0, xind=3, xoffset=0.0)
        assert a is not b
        assert id(a.get_position()) != id(b.get_position())

    def test_global_offset_shifts_position(self):
        nbfig = setup_figure(format="A4")
        a = setup_axes(nbfig, yind=4, yoffset=0.0, xind=2, xoffset=0.0)
        b = setup_axes(nbfig, yind=4, yoffset=0.0, xind=2, xoffset=0.0,
                        g_xoffset=2.0)
        ax_a, ay_a, _, _ = a.get_position().bounds
        ax_b, ay_b, _, _ = b.get_position().bounds
        # b should be shifted right relative to a by 2 cm / page_width
        assert ax_b > ax_a

    def test_scale_changes_size(self):
        nbfig = setup_figure(format="A4")
        a = setup_axes(nbfig, yind=4, yoffset=0.0, xind=2, xoffset=0.0)
        b = setup_axes(nbfig, yind=4, yoffset=0.0, xind=2, xoffset=0.0,
                        wscale=2.0, hscale=2.0)
        wa, ha = a.get_position().width, a.get_position().height
        wb, hb = b.get_position().width, b.get_position().height
        assert wb == pytest.approx(2.0 * wa)
        assert hb == pytest.approx(2.0 * ha)

    def test_index_validation(self):
        nbfig = setup_figure(format="A4")
        with pytest.raises(IndexError):
            setup_axes(nbfig, yind=99, yoffset=0.0, xind=2, xoffset=0.0)
        with pytest.raises(IndexError):
            setup_axes(nbfig, yind=4, yoffset=0.0, xind=99, xoffset=0.0)

    def test_recovers_nbfig_from_bare_figure(self):
        nbfig = setup_figure(format="A4")
        sax = setup_axes(nbfig.figure, yind=4, yoffset=0.0, xind=2, xoffset=0.0)
        assert sax is not None

    def test_raises_on_bare_figure_without_setup(self):
        fig = plt.figure()
        with pytest.raises(RuntimeError, match="setup_figure"):
            setup_axes(fig, yind=4, yoffset=0.0, xind=2, xoffset=0.0)

    def test_delete_existing(self):
        nbfig = setup_figure(format="A4")
        a = setup_axes(nbfig, yind=4, yoffset=0.0, xind=2, xoffset=0.0)
        n_axes_before = len(nbfig.figure.axes)
        b = setup_axes(nbfig, yind=4, yoffset=0.0, xind=2, xoffset=0.0,
                        delete_existing=True)
        # Same key but new instance
        assert a is not b
        # Total axes count should stay the same (one removed, one added)
        assert len(nbfig.figure.axes) == n_axes_before


class TestPlaceSubplot:
    def teardown_method(self):
        plt.close("all")

    def test_place_subplot_works(self):
        nbfig = setup_figure(format="A4")
        sax = place_subplot(nbfig, yind=4, yoffset=0.0, xind=2, xoffset=0.0,
                             scale_width=1.5, scale_height=1.5)
        x, y, w, h = sax.get_position().bounds
        assert w > 0 and h > 0


class TestClearHelpers:
    def teardown_method(self):
        plt.close("all")

    def test_clear_axes_labels(self):
        nbfig = setup_figure(format="A4")
        sax = setup_axes(nbfig, yind=4, yoffset=0.0, xind=2, xoffset=0.0)
        sax.plot([1, 2, 3], [4, 5, 6])
        clear_axes_labels(sax)
        # Labels emptied but ticks may remain
        assert all(t.get_text() == "" for t in sax.get_xticklabels())
        assert all(t.get_text() == "" for t in sax.get_yticklabels())

    def test_clear_axes_ticks(self):
        nbfig = setup_figure(format="A4")
        sax = setup_axes(nbfig, yind=4, yoffset=0.0, xind=2, xoffset=0.0)
        sax.plot([1, 2, 3], [4, 5, 6])
        clear_axes_ticks(sax)
        assert len(sax.get_xticks()) == 0
        assert len(sax.get_yticks()) == 0

    def test_empty_axis_sets_background(self):
        nbfig = setup_figure(format="A4")
        sax = setup_axes(nbfig, yind=4, yoffset=0.0, xind=2, xoffset=0.0)
        empty_axis(sax)
        # Default bg = (0.9, 0.9, 0.9)
        rgba = sax.get_facecolor()
        assert rgba[:3] == pytest.approx((0.9, 0.9, 0.9), abs=1e-2)

    def test_clear_fax(self):
        nbfig = setup_figure(format="A4")
        nbfig.fax.text(5.0, 5.0, "annotation")
        clear_fax(nbfig)
        assert len(nbfig.fax.texts) == 0


class TestLinkax:
    def teardown_method(self):
        plt.close("all")

    def test_linkax_xy(self):
        nbfig = setup_figure(format="A4")
        a = setup_axes(nbfig, yind=4, yoffset=0.0, xind=2, xoffset=0.0)
        b = setup_axes(nbfig, yind=4, yoffset=0.0, xind=3, xoffset=0.0)
        a.plot([1, 2, 3], [10, 20, 30])
        b.plot([1, 2, 3], [10, 20, 30])
        linkax(nbfig, mode="xy")
        # Setting xlim on a should propagate to b after linkage
        a.set_xlim(-5, 5)
        # In matplotlib sharex, b inherits via shared axis machinery
        # — exact check is that the shared-axis siblings are populated.
        assert b in a.get_shared_x_axes().get_siblings(a)

    def test_linkax_under_2_axes_is_noop(self):
        nbfig = setup_figure(format="A4")
        # No axes added; should not crash
        linkax(nbfig)
        a = setup_axes(nbfig, yind=4, yoffset=0.0, xind=2, xoffset=0.0)
        # Single axis; should not crash
        linkax(nbfig)


# ─────────────────────────────────────────────────────────────────────── #
# Inkscape layout                                                            #
# ─────────────────────────────────────────────────────────────────────── #

class TestInkscapeLayout:
    def teardown_method(self):
        plt.close("all")

    def test_parse_json(self, tmp_path):
        layout_data = {
            "body": {"x": 0, "y": 0, "width": 800.0, "height": 1000.0},
            "subplots": {
                "panel-A": {"x": 100.0, "y": 100.0,
                            "width": 200.0, "height": 200.0},
                "panel-B": {"x": 400.0, "y": 100.0,
                            "width": 200.0, "height": 200.0},
            },
        }
        f = tmp_path / "layout.json"
        f.write_text(json.dumps(layout_data))
        layout = parse_inkscape_layout(f)
        assert isinstance(layout, Layout)
        assert layout.body["width"] == 800.0
        assert "panel-A" in layout.subplots
        assert "panel-B" in layout.subplots

    def test_parse_json_with_filter(self, tmp_path):
        layout_data = {
            "body": {"width": 800.0, "height": 1000.0},
            "subplots": {
                "fig1-panel-A": {"x": 0, "y": 0, "width": 1, "height": 1},
                "fig2-panel-A": {"x": 0, "y": 0, "width": 1, "height": 1},
            },
        }
        f = tmp_path / "layout.json"
        f.write_text(json.dumps(layout_data))
        layout = parse_inkscape_layout(f, element_pattern="^fig1-")
        assert "fig1-panel-A" in layout.subplots
        assert "fig2-panel-A" not in layout.subplots

    def test_parse_csv_format(self, tmp_path):
        f = tmp_path / "layout.csv"
        f.write_text(
            "body,0,0,800,1000\n"
            "panel-A,100,100,200,200\n"
        )
        layout = parse_inkscape_layout(f)
        assert layout.body["width"] == 800.0
        assert "panel-A" in layout.subplots

    def test_setup_axes_named(self, tmp_path):
        layout_data = {
            "body": {"width": 800.0, "height": 1000.0},
            "subplots": {
                "p": {"x": 100.0, "y": 100.0,
                      "width": 200.0, "height": 200.0},
            },
        }
        f = tmp_path / "layout.json"
        f.write_text(json.dumps(layout_data))
        layout = parse_inkscape_layout(f)

        nbfig = setup_figure(format="A4")
        nbfig.layout = layout
        sax = setup_axes_named(nbfig, "p")
        x, y, w, h = sax.get_position().bounds
        # x = 100/800 = 0.125; w = 200/800 = 0.25
        assert x == pytest.approx(0.125)
        assert w == pytest.approx(0.25)

    def test_setup_axes_named_unknown_key_raises(self, tmp_path):
        nbfig = setup_figure(format="A4")
        nbfig.layout = Layout(
            body={"width": 100, "height": 100}, subplots={},
        )
        with pytest.raises(KeyError):
            setup_axes_named(nbfig, "missing")


# ─────────────────────────────────────────────────────────────────────── #
# Plots                                                                      #
# ─────────────────────────────────────────────────────────────────────── #

class TestImagescnan:
    def teardown_method(self):
        plt.close("all")

    def test_basic_array(self):
        fig, ax = plt.subplots()
        data = np.random.rand(10, 12)
        im, cb = imagescnan(data, ax=ax)
        assert im is not None
        assert cb is None

    def test_nan_values_handled(self):
        fig, ax = plt.subplots()
        data = np.random.rand(10, 12)
        data[5, 5] = np.nan
        im, _ = imagescnan(data, ax=ax, nan_rgb=(1, 0, 0))
        # No exception; check colormap has bad colour set
        assert im.cmap.get_bad()[:3] == pytest.approx((1.0, 0.0, 0.0))

    def test_with_axis_arrays(self):
        fig, ax = plt.subplots()
        x = np.linspace(0, 100, 12)
        y = np.linspace(0, 50, 10)
        data = np.outer(np.linspace(0, 1, 10), np.linspace(0, 1, 12))
        im, _ = imagescnan((x, y, data), ax=ax)
        # Check extent is set
        ext = im.get_extent()
        assert ext is not None

    def test_color_limits_sym(self):
        fig, ax = plt.subplots()
        data = np.random.standard_normal((10, 10))
        im, _ = imagescnan(data, ax=ax, color_limits="sym")
        vmin, vmax = im.get_clim()
        assert abs(vmin + vmax) < 1e-9   # symmetric

    def test_add_colorbar(self):
        fig, ax = plt.subplots()
        data = np.random.rand(10, 10)
        im, cb = imagescnan(data, ax=ax, add_colorbar=True)
        assert cb is not None


class TestErrorEllipse:
    def teardown_method(self):
        plt.close("all")

    def test_basic_2d(self):
        fig, ax = plt.subplots()
        cov = np.array([[1.0, 0.5], [0.5, 1.0]])
        line = error_ellipse(cov, ax=ax, conf=0.95, color="red")
        assert line is not None

    def test_centred_at_mu(self):
        fig, ax = plt.subplots()
        cov = np.eye(2)
        mu = np.array([5.0, 10.0])
        line = error_ellipse(cov, mu, ax=ax, conf=0.95)
        # Centroid of ellipse points should be near mu
        x, y = line.get_data()
        centroid = (np.mean(x), np.mean(y))
        assert centroid[0] == pytest.approx(5.0, abs=0.5)
        assert centroid[1] == pytest.approx(10.0, abs=0.5)

    def test_rejects_non_psd(self):
        fig, ax = plt.subplots()
        # Negative-definite matrix
        with pytest.raises(ValueError, match="positive definite"):
            error_ellipse(np.array([[-1.0, 0.0], [0.0, -1.0]]), ax=ax)

    def test_rejects_wrong_shape(self):
        fig, ax = plt.subplots()
        with pytest.raises(ValueError, match="2x2"):
            error_ellipse(np.eye(3), ax=ax)

    def test_rejects_bad_conf(self):
        fig, ax = plt.subplots()
        cov = np.eye(2)
        with pytest.raises(ValueError, match="conf"):
            error_ellipse(cov, ax=ax, conf=1.5)


class TestApplyColorbar:
    def teardown_method(self):
        plt.close("all")

    def test_eastoutside(self):
        nbfig = setup_figure(format="A4")
        sax = setup_axes(nbfig, yind=4, yoffset=0.0, xind=2, xoffset=0.0)
        data = np.random.rand(10, 10)
        imagescnan(data, ax=sax)
        cb = apply_colorbar(sax, location="eastoutside")
        # Colorbar should be to the right of sax
        sax_pos = sax.get_position()
        cb_pos  = cb.ax.get_position()
        assert cb_pos.x0 > sax_pos.x1

    def test_southoutside(self):
        nbfig = setup_figure(format="A4")
        sax = setup_axes(nbfig, yind=4, yoffset=0.0, xind=2, xoffset=0.0)
        data = np.random.rand(10, 10)
        imagescnan(data, ax=sax)
        cb = apply_colorbar(sax, location="southoutside")
        sax_pos = sax.get_position()
        cb_pos  = cb.ax.get_position()
        assert cb_pos.y1 < sax_pos.y0

    def test_unknown_location_raises(self):
        nbfig = setup_figure(format="A4")
        sax = setup_axes(nbfig, yind=4, yoffset=0.0, xind=2, xoffset=0.0)
        imagescnan(np.random.rand(5, 5), ax=sax)
        with pytest.raises(ValueError, match="location must be"):
            apply_colorbar(sax, location="northwest")

    def test_no_mappable_raises(self):
        nbfig = setup_figure(format="A4")
        sax = setup_axes(nbfig, yind=4, yoffset=0.0, xind=2, xoffset=0.0)
        sax.plot([1, 2, 3])  # no image / mappable
        with pytest.raises(RuntimeError, match="no image"):
            apply_colorbar(sax)


class TestArrowPrimitives:
    def teardown_method(self):
        plt.close("all")

    def test_circular_arrow_runs(self):
        fig, ax = plt.subplots()
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        arcs, arrows = circular_arrow(
            ax, radius=1.0, center=(0, 0),
            arrow_angle=0.0, angle=2.0, direction=1, color="b",
        )
        assert len(arcs) == 1
        assert len(arrows) == 1

    def test_circular_arrow_double_head(self):
        fig, ax = plt.subplots()
        ax.set_xlim(-2, 2); ax.set_ylim(-2, 2)
        arcs, arrows = circular_arrow(
            ax, radius=1.0, center=(0, 0),
            arrow_angle=0.0, angle=2.0, direction=2, color="b",
        )
        assert len(arrows) == 2

    def test_circular_arrow_invalid_direction(self):
        fig, ax = plt.subplots()
        with pytest.raises(ValueError, match="direction"):
            circular_arrow(ax, radius=1.0, center=(0, 0),
                            arrow_angle=0.0, angle=2.0, direction=99)

    def test_draw_arrow_straight(self):
        fig, ax = plt.subplots()
        a = draw_arrow(ax, (0, 0), (1, 1), style="straight")
        assert a is not None

    def test_draw_arrow_curve(self):
        fig, ax = plt.subplots()
        a = draw_arrow(ax, (0, 0), (1, 1), style="curve")
        assert a is not None

    def test_draw_arrow_invalid_style(self):
        fig, ax = plt.subplots()
        with pytest.raises(ValueError, match="style"):
            draw_arrow(ax, (0, 0), (1, 1), style="zigzag")


class TestSbar:
    def teardown_method(self):
        plt.close("all")

    def test_overlay_histogram(self):
        fig, ax = plt.subplots()
        rng = np.random.default_rng(0)
        feature = rng.standard_normal(1000)
        bins = np.linspace(-3, 3, 21)
        mask = feature > 0
        bg, fg = sbar(feature, bins, mask, ax=ax)
        # Background bars cover all data
        assert len(bg) == len(bins) - 1
        assert len(fg) == len(bins) - 1


# ─────────────────────────────────────────────────────────────────────── #
# State plotting                                                             #
# ─────────────────────────────────────────────────────────────────────── #

@pytest.fixture
def synthetic_stc():
    fs = 30.0
    def ep(starts, ends, label):
        return NBEpoch(
            data=np.column_stack([np.asarray(starts), np.asarray(ends)]),
            samplerate=fs,
            label=label,
        )
    stc = NBStateCollection()
    stc.add_state(ep([0, 1500], [500, 2500], "walk"))
    stc.add_state(ep([600], [800], "rear"))
    stc.add_state(ep([900], [1100], "turn"))
    stc.add_state(ep([1200], [1450], "pause"))
    stc.add_state(ep([2700], [2800], "groom"))
    stc.add_state(ep([2900], [3000], "sit"))
    stc.add_state(ep([0, 1300, 2500], [1000, 2200, 3000], "theta"))
    return stc, fs


class TestPlotStc:
    def teardown_method(self):
        plt.close("all")

    def test_basic_render(self, synthetic_stc):
        stc, fs = synthetic_stc
        fig, ax = plt.subplots()
        ax = plot_stc(
            stc, ax=ax, samplerate=fs, time_unit="seconds",
            states=["walk", "rear", "turn", "pause", "groom", "sit"],
        )
        # Should have added patches
        assert len(ax.patches) > 0

    def test_single_row_when_unstaggered(self, synthetic_stc):
        stc, fs = synthetic_stc
        fig, ax = plt.subplots()
        plot_stc(
            stc, ax=ax, samplerate=fs, staggered=False,
            states=["walk", "rear"],
        )
        # All patches should have the same y (overlay)
        ys = [p.get_y() for p in ax.patches]
        assert all(y == 0.0 for y in ys)

    def test_color_string_brgcmy(self, synthetic_stc):
        stc, fs = synthetic_stc
        fig, ax = plt.subplots()
        plot_stc(
            stc, ax=ax, samplerate=fs,
            states=["walk", "rear", "turn", "pause", "groom", "sit"],
            state_colors="brgcmy",
        )
        # No exception thrown


class TestPlotStateDurations:
    def teardown_method(self):
        plt.close("all")

    def test_basic_render(self, synthetic_stc):
        stc, fs = synthetic_stc
        fig, ax = plt.subplots()
        plot_state_durations(stc, ax=ax)
        # Top + bottom rows produce patches (top row alone = 6 patches
        # for 6 states; bottom row adds up to 12 more for theta-split)
        assert len(ax.patches) >= 6


class TestPlotFeaturesWithStc:
    def teardown_method(self):
        plt.close("all")

    def test_basic_render(self, synthetic_stc):
        stc, fs = synthetic_stc
        fig, ax = plt.subplots()
        T = 3000
        feat = np.cumsum(np.random.standard_normal(T)) / np.sqrt(T)
        plot_features_with_stc(
            feat, stc, samplerate=fs, ax=ax,
            states=["walk", "rear"],
        )
        # Feature line + state spans (axvspan adds Rectangles)
        assert len(ax.lines) >= 1
        assert any(p.__class__.__name__ == "Rectangle" for p in ax.patches)


class TestPlotStcs:
    def teardown_method(self):
        plt.close("all")

    def test_multiple_collections(self, synthetic_stc):
        stc, fs = synthetic_stc
        fig, ax = plt.subplots()
        plot_stcs(
            [stc, stc], ax=ax, samplerate=fs,
            states=["walk", "rear", "turn"],
        )
        # 2 collections × 3 states each, multiple periods → patches > 0
        assert len(ax.patches) > 0


# ─────────────────────────────────────────────────────────────────────── #
# Integration: render a small EgoProCode2D-style figure                      #
# ─────────────────────────────────────────────────────────────────────── #

class TestIntegration:
    def teardown_method(self):
        plt.close("all")

    def test_egoprocode2d_style_grid(self, tmp_path):
        """Reproduce an EgoProCode2D-style nested loop over phz × hba."""
        nbfig = setup_figure(format="A4", layout="portrait")

        n_phz, n_hba = 3, 3
        for phz_i in range(n_phz):
            for hba_i in range(n_hba):
                sax = setup_axes(
                    nbfig,
                    yind   = 5 + phz_i,
                    yoffset = -0.15,
                    xind   = 2 + hba_i,
                    xoffset = 0.15,
                )
                # Plot synthetic egocentric ratemap
                ratemap = np.random.rand(20, 20)
                ratemap[~np.isfinite(ratemap)] = 0
                imagescnan(ratemap, ax=sax, color_limits=(0, 1))
                clear_axes_ticks(sax)

        # Should have 9 axes
        assert len(nbfig.axes) == 9

        # And we can save vector PDF
        out = tmp_path / "ego_grid.pdf"
        nbfig.figure.savefig(out)
        assert out.exists()
        assert out.stat().st_size > 1000   # non-trivial PDF
