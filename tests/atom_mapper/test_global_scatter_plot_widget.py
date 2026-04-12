"""Tests for the global scatter plot widget."""

from __future__ import annotations

import pytest

pytest.importorskip("PyQt6", reason="PyQt6 is required for AtomMapper GUI tests")
pytest.importorskip("pytestqt", reason="pytest-qt is required for AtomMapper GUI tests")
pytest.importorskip("pyqtgraph", reason="pyqtgraph is required for AtomMapper plot widgets")

from AtomMapper.app.global_scatter_plot_widget import GlobalScatterPlotWidget
from AtomMapper.app.models import AtomPoint, AtomRow
from AtomMapper.app.plots import PlotUnit, build_global_scatter_series


def _make_rows() -> tuple[AtomRow, AtomRow]:
    row_1 = AtomRow(
        row_id="row-1",
        source_group_id="group-1",
        display_name="Row 1",
        color_hex="#ff8800",
        points=(
            AtomPoint(
                row_id="row-1",
                image_id="image-1",
                source_group_id="group-1",
                point_index=0,
                x_px=10.0,
                y_px=20.0,
                x_nm=10.0,
                y_nm=20.0,
                point_id="point-1",
            ),
            AtomPoint(
                row_id="row-1",
                image_id="image-1",
                source_group_id="group-1",
                point_index=1,
                x_px=12.0,
                y_px=21.0,
                x_nm=12.0,
                y_nm=21.0,
                point_id="point-2",
            ),
        ),
    )
    row_2 = AtomRow(
        row_id="row-2",
        source_group_id="group-1",
        display_name="Row 2",
        points=(
            AtomPoint(
                row_id="row-2",
                image_id="image-2",
                source_group_id="group-1",
                point_index=0,
                x_px=30.0,
                y_px=40.0,
                x_nm=30.0,
                y_nm=40.0,
                point_id="point-3",
                manual_override=True,
                manual_override_source="drag",
            ),
        ),
    )
    return row_1, row_2


def test_global_scatter_plot_widget_initializes_with_placeholder(qtbot):
    widget = GlobalScatterPlotWidget()
    qtbot.addWidget(widget)

    assert widget.current_series is None
    assert widget.stack.currentWidget() is widget.placeholder_label
    assert widget.title_label.text() == "Global rows plot [px]"
    assert "global scatter plot" in widget.placeholder_label.text().lower()


def test_global_scatter_plot_widget_renders_grouped_rows(qtbot):
    widget = GlobalScatterPlotWidget()
    qtbot.addWidget(widget)

    rows = _make_rows()
    series = build_global_scatter_series(rows)
    widget.set_rows(rows)

    assert widget.current_series == series
    assert widget.stack.currentWidget() is widget.plot_widget
    assert len(widget.scatter_items) == 2
    assert widget.legend_item is not None
    assert len(widget.legend_item.items) == 2
    assert "2 rows" in widget.info_label.text()
    assert "3 points" in widget.info_label.text()


def test_global_scatter_plot_widget_handles_empty_series(qtbot):
    widget = GlobalScatterPlotWidget()
    qtbot.addWidget(widget)

    empty_series = build_global_scatter_series(())
    widget.set_series(empty_series)

    assert widget.stack.currentWidget() is widget.placeholder_label
    assert "no saved points" in widget.placeholder_label.text().lower()
    assert "waiting for saved points" in widget.info_label.text().lower()


def test_global_scatter_plot_widget_can_switch_to_nm(qtbot):
    widget = GlobalScatterPlotWidget()
    qtbot.addWidget(widget)

    widget.set_rows(_make_rows())
    unit_index = widget.unit_combo.findData(PlotUnit.NM)
    widget.unit_combo.setCurrentIndex(unit_index)

    assert widget.current_series is not None
    assert widget.current_series.unit is PlotUnit.NM
    assert widget.title_label.text() == "Global rows plot [nm]"
    assert widget.current_series.x_label == "x (nm)"
    assert widget.current_series.y_label == "y (nm)"
    assert widget.stack.currentWidget() is widget.plot_widget
    assert "nm" in widget.info_label.text()
