"""Tests for the row distance-metrics widget."""

from __future__ import annotations

import pytest

pytest.importorskip("PyQt6", reason="PyQt6 is required for AtomMapper GUI tests")
pytest.importorskip("pytestqt", reason="pytest-qt is required for AtomMapper GUI tests")

from AtomMapper.app.models import AtomPoint, AtomRow
from AtomMapper.app.plots import build_row_distance_metrics
from AtomMapper.app.row_metrics_widget import RowMetricsWidget


def _make_row_with_distances() -> AtomRow:
    return AtomRow(
        row_id="row-1",
        source_group_id="group-1",
        display_name="Row 1",
        points=(
            AtomPoint(
                row_id="row-1",
                image_id="image-1",
                source_group_id="group-1",
                point_index=0,
                x_px=0.0,
                y_px=0.0,
                point_id="point-1",
            ),
            AtomPoint(
                row_id="row-1",
                image_id="image-1",
                source_group_id="group-1",
                point_index=1,
                x_px=3.0,
                y_px=4.0,
                point_id="point-2",
            ),
            AtomPoint(
                row_id="row-1",
                image_id="image-1",
                source_group_id="group-1",
                point_index=2,
                x_px=9.0,
                y_px=12.0,
                point_id="point-3",
            ),
        ),
    )


def test_row_metrics_widget_initializes_with_placeholder(qtbot):
    widget = RowMetricsWidget()
    qtbot.addWidget(widget)

    assert widget.current_metrics is None
    assert widget.stack.currentWidget() is widget.placeholder_label
    assert "distance metrics" in widget.placeholder_label.text().lower()


def test_row_metrics_widget_renders_distance_summary(qtbot):
    widget = RowMetricsWidget()
    qtbot.addWidget(widget)

    metrics = build_row_distance_metrics(_make_row_with_distances())
    widget.set_metrics(metrics)

    assert widget.current_metrics == metrics
    assert widget.stack.currentWidget() is widget.metrics_panel
    assert widget.point_count_value.text() == "3"
    assert widget.distance_count_value.text() == "2"
    assert widget.mean_distance_value.text() == "7.500 px"
    assert widget.std_distance_value.text() == "2.500 px"
    assert widget.min_distance_value.text() == "5.000 px"
    assert widget.max_distance_value.text() == "10.000 px"


def test_row_metrics_widget_shows_empty_state_for_too_few_points(qtbot):
    widget = RowMetricsWidget()
    qtbot.addWidget(widget)

    row = AtomRow(
        row_id="row-1",
        source_group_id="group-1",
        display_name="Row 1",
        points=(
            AtomPoint(
                row_id="row-1",
                image_id="image-1",
                source_group_id="group-1",
                point_index=0,
                x_px=1.0,
                y_px=2.0,
                point_id="point-1",
            ),
        ),
    )
    widget.set_metrics(build_row_distance_metrics(row))

    assert widget.stack.currentWidget() is widget.placeholder_label
    assert "at least 2 points" in widget.placeholder_label.text().lower()
    assert "waiting for more data" in widget.info_label.text().lower()
