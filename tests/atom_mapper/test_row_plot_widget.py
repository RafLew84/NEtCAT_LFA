"""Tests for the selected-row plot widget."""

from __future__ import annotations

import pytest

pytest.importorskip("PyQt6", reason="PyQt6 is required for AtomMapper GUI tests")
pytest.importorskip("pytestqt", reason="pytest-qt is required for AtomMapper GUI tests")
pytest.importorskip("pyqtgraph", reason="pyqtgraph is required for AtomMapper plot widgets")

from AtomMapper.app.models import AtomPoint, AtomRow
from AtomMapper.app.plots import RowPlotMode, build_row_metric_series
from AtomMapper.app.row_plot_widget import RowPlotWidget


def _make_row_with_points() -> AtomRow:
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
                x_px=10.0,
                y_px=20.0,
                point_id="point-1",
            ),
            AtomPoint(
                row_id="row-1",
                image_id="image-1",
                source_group_id="group-1",
                point_index=1,
                x_px=12.5,
                y_px=21.5,
                point_id="point-2",
            ),
        ),
    )


def test_row_plot_widget_initializes_with_placeholder(qtbot):
    widget = RowPlotWidget()
    qtbot.addWidget(widget)

    assert widget.current_series is None
    assert widget.stack.currentWidget() is widget.placeholder_label
    assert "Select an atom row" in widget.placeholder_label.text()


def test_row_plot_widget_renders_basic_row_series(qtbot):
    widget = RowPlotWidget()
    qtbot.addWidget(widget)

    row = _make_row_with_points()
    series = build_row_metric_series(row, RowPlotMode.X_PX)
    widget.set_series(series)

    assert widget.current_series == series
    assert widget.stack.currentWidget() is widget.plot_widget
    plotted_x, plotted_y = widget.curve_item.getData()
    assert list(plotted_x) == pytest.approx([0.0, 1.0])
    assert list(plotted_y) == pytest.approx([10.0, 12.5])
    assert "Row 1" in widget.info_label.text()
    assert "2 samples" in widget.info_label.text()


def test_row_plot_widget_shows_empty_state_for_row_without_plottable_samples(qtbot):
    widget = RowPlotWidget()
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
                x_px=10.0,
                y_px=20.0,
                point_id="point-1",
            ),
        ),
    )
    distance_series = build_row_metric_series(row, RowPlotMode.DISTANCE_PX)
    widget.set_series(distance_series)

    assert widget.stack.currentWidget() is widget.placeholder_label
    assert "no plottable samples" in widget.placeholder_label.text().lower()
    assert "waiting for enough points" in widget.info_label.text().lower()
