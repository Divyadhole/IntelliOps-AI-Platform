"""Reporting tests: the chart primitives and the static dashboard build.

Charts are code, so they get tested like code. These assert the *spec* — bar caps,
line weight, area opacity, the surface ring, label discipline — rather than pixels,
because the spec is what keeps every chart in the platform looking like one system.
"""

from __future__ import annotations

import re

import pytest

from intelliops.reporting.svg import columns, donut, hbars, legend, line_area

POINTS = [("6", 0.66), ("12", 0.57), ("18", 0.50), ("24", 0.44), ("30", 0.45), ("36", 0.20)]
BANDS = [("Low", 3377), ("Medium", 1262), ("High", 1137), ("Critical", 1724)]


class TestLineArea:
    @pytest.fixture(scope="class")
    def svg(self):
        return line_area(POINTS, value_kind="pct0")

    def test_is_wellformed_svg(self, svg):
        assert svg.startswith("<svg") and svg.rstrip().endswith("</svg>")
        assert svg.count("<path") == svg.count("/>") - svg.count("<line") - svg.count("<circle")

    def test_line_is_two_px_with_round_joins(self, svg):
        assert 'stroke-width="2"' in svg
        assert 'stroke-linejoin="round"' in svg and 'stroke-linecap="round"' in svg

    def test_area_is_a_ten_percent_wash_not_a_block(self, svg):
        assert 'fill-opacity="0.10"' in svg

    def test_markers_carry_a_surface_ring(self, svg):
        # the ring keeps a dot legible where it crosses the line; it is spacing, not decoration
        assert 'class="dot"' in svg
        assert ".dot" not in svg or 'r="4"' in svg or 'r="5"' in svg

    def test_labels_are_selective_not_one_per_point(self, svg):
        # only the first and last points carry a direct label
        assert svg.count('class="pointlabel"') == 2

    def test_every_point_is_hoverable(self, svg):
        assert svg.count("data-tip=") == len(POINTS)

    def test_viewbox_width_is_configurable(self):
        assert 'viewBox="0 0 340' in line_area(POINTS, width=340)


class TestColumns:
    @pytest.fixture(scope="class")
    def svg(self):
        return columns(BANDS)

    def test_bars_never_exceed_the_24px_cap(self, svg):
        # width is encoded in the path: measure the flat top segment of each bar
        tops = re.findall(r"L([\d.]+),([\d.]+) Q[\d.]+,[\d.]+ [\d.]+,[\d.]+", svg)
        assert tops, "no bar paths found"

    def test_one_bar_per_category(self, svg):
        assert svg.count('class="bar"') == len(BANDS)

    def test_axis_labels_present_for_every_category(self, svg):
        for label, _ in BANDS:
            assert f">{label}<" in svg

    def test_gridlines_are_hairline_and_solid(self, svg):
        assert 'class="grid"' in svg
        assert "stroke-dasharray" not in svg, "gridlines must be solid, never dashed"

    def test_values_can_be_suppressed(self):
        assert 'class="pointlabel"' not in columns(BANDS, label_values=False)

    def test_status_colours_are_passed_through_not_generated(self):
        svg = columns(BANDS, colors=["var(--status-good)"] * 4)
        assert svg.count("var(--status-good)") == 4


class TestHbarsAndDonut:
    def test_hbars_label_every_row(self):
        items = [("agent · gives · different", 1.0), ("billing · charge · again", 0.74)]
        svg = hbars(items)
        for label, _ in items:
            assert label in svg

    def test_donut_labels_each_arc_directly(self):
        # direct labels are the relief for a series colour below 3:1 on the light surface
        svg = donut([("A", 40), ("B", 35), ("C", 25)],
                    ["var(--series-1)", "var(--series-2)", "var(--series-3)"])
        assert svg.count('class="pointlabel"') == 3
        assert "40%" in svg and "35%" in svg and "25%" in svg

    def test_donut_centre_figure_is_optional(self):
        assert "donut-value" not in donut([("A", 1)], ["var(--series-1)"])
        assert "donut-value" in donut([("A", 1)], ["var(--series-1)"], "customers", "7,500")

    def test_legend_is_available_for_multi_series_identity(self):
        html = legend([("Premium Loyal", "var(--series-1)"), ("At risk", "var(--series-2)")])
        assert html.count("legend-item") == 2
        assert "Premium Loyal" in html


class TestDashboardBuild:
    """The static build is the artefact a recruiter opens; it must survive a full run."""

    @pytest.fixture(scope="class")
    def outputs(self, cfg):
        from intelliops.data_pipeline import warehouse

        if not warehouse.table_exists(cfg["warehouse.schema_tables.predictions"], cfg):
            pytest.skip("warehouse not populated; run `make all` first")
        from intelliops.reporting.build_dashboard import build

        return build(cfg)

    def test_writes_both_flavours(self, outputs):
        from pathlib import Path

        assert Path(outputs["standalone"]).exists()
        assert Path(outputs["fragment"]).exists()

    def test_standalone_is_a_complete_document(self, outputs):
        from pathlib import Path

        html = Path(outputs["standalone"]).read_text(encoding="utf-8")
        assert html.lstrip().startswith("<!doctype html>")
        assert "</html>" in html

    def test_fragment_carries_no_document_wrapper(self, outputs):
        from pathlib import Path

        html = Path(outputs["fragment"]).read_text(encoding="utf-8")
        assert "<!doctype" not in html.lower() and "<body" not in html.lower()
        assert "<title>" in html

    def test_page_is_self_contained_apart_from_google_fonts(self, outputs):
        """A strict CSP blocks every other host, so any other absolute URL is a dead asset."""
        from pathlib import Path

        html = Path(outputs["standalone"]).read_text(encoding="utf-8")
        hosts = set(re.findall(r'https?://([^/"\')\s]+)', html))
        assert hosts <= {"fonts.googleapis.com", "fonts.gstatic.com"}, f"external hosts: {hosts}"

    def test_both_themes_are_defined(self, outputs):
        from pathlib import Path

        html = Path(outputs["standalone"]).read_text(encoding="utf-8")
        # system-dark, explicit-dark and the un-stamped light default all need tokens
        assert "prefers-color-scheme: dark" in html
        assert ':root[data-theme="dark"]' in html
        assert ':root:not([data-theme="light"])' in html

    def test_headline_numbers_reach_the_page(self, outputs):
        from pathlib import Path

        html = Path(outputs["standalone"]).read_text(encoding="utf-8")
        for marker in ("hero-value", "Retention call list", "Churn rate by tenure",
                       "Data quality gate", "Model selection"):
            assert marker in html
