"""Tests for the row-disturbance summary widget."""

from __future__ import annotations

import pytest

pytest.importorskip("PyQt6", reason="PyQt6 is required for AtomMapper GUI tests")
pytest.importorskip("pytestqt", reason="pytest-qt is required for AtomMapper GUI tests")

from AtomMapper.app.models import AtomPoint, AtomRow
from AtomMapper.app.plots import PlotUnit
from AtomMapper.app.row_disturbance_widget import RowDisturbanceWidget


def _make_row_with_disturbance() -> AtomRow:
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
                x_nm=0.0,
                y_nm=0.0,
                point_id="point-0",
            ),
            AtomPoint(
                row_id="row-1",
                image_id="image-1",
                source_group_id="group-1",
                point_index=1,
                x_px=1.0,
                y_px=0.0,
                x_nm=0.1,
                y_nm=0.0,
                point_id="point-1",
            ),
            AtomPoint(
                row_id="row-1",
                image_id="image-1",
                source_group_id="group-1",
                point_index=2,
                x_px=2.0,
                y_px=0.0,
                x_nm=0.2,
                y_nm=0.0,
                point_id="point-2",
            ),
            AtomPoint(
                row_id="row-1",
                image_id="image-1",
                source_group_id="group-1",
                point_index=3,
                x_px=5.0,
                y_px=0.0,
                x_nm=0.5,
                y_nm=0.0,
                point_id="point-3",
            ),
            AtomPoint(
                row_id="row-1",
                image_id="image-1",
                source_group_id="group-1",
                point_index=4,
                x_px=6.0,
                y_px=0.0,
                x_nm=0.6,
                y_nm=0.0,
                point_id="point-4",
            ),
        ),
    )


def test_row_disturbance_widget_initializes_with_placeholder(qtbot):
    widget = RowDisturbanceWidget()
    qtbot.addWidget(widget)

    assert widget.current_row is None
    assert widget.current_series is None
    assert widget.current_unit is PlotUnit.PX
    assert widget.stack.currentWidget() is widget.placeholder_label
    assert "disturbance candidates" in widget.placeholder_label.text().lower()


def test_row_disturbance_widget_renders_candidate_summary(qtbot):
    widget = RowDisturbanceWidget()
    qtbot.addWidget(widget)

    widget.set_row(_make_row_with_disturbance(), unit=PlotUnit.PX)

    assert widget.current_series is not None
    assert widget.current_series.candidate_count >= 1
    assert widget.stack.currentWidget() is widget.summary_panel
    assert widget.title_label.text() == "Row disturbance candidates [px]"
    assert widget.sample_count_value.text() == "3"
    assert widget.candidate_count_value.text() == str(widget.current_series.candidate_count)
    assert widget.strongest_point_value.text() != "-"
    assert widget.spacing_threshold_value.text().endswith(" px")


def test_row_disturbance_widget_uses_nm_and_handles_missing_calibration(qtbot):
    widget = RowDisturbanceWidget()
    qtbot.addWidget(widget)

    widget.set_row(_make_row_with_disturbance(), unit=PlotUnit.NM)
    assert widget.current_series is not None
    assert widget.title_label.text() == "Row disturbance candidates [nm]"
    assert widget.transverse_threshold_value.text().endswith(" nm")

    uncalibrated_row = AtomRow(
        row_id="row-uncal",
        source_group_id="group-1",
        display_name="Uncal row",
        points=(
            AtomPoint(
                row_id="row-uncal",
                image_id="image-1",
                source_group_id="group-1",
                point_index=0,
                x_px=0.0,
                y_px=0.0,
                point_id="point-0",
            ),
            AtomPoint(
                row_id="row-uncal",
                image_id="image-1",
                source_group_id="group-1",
                point_index=1,
                x_px=1.0,
                y_px=0.0,
                point_id="point-1",
            ),
            AtomPoint(
                row_id="row-uncal",
                image_id="image-1",
                source_group_id="group-1",
                point_index=2,
                x_px=2.0,
                y_px=0.0,
                point_id="point-2",
            ),
        ),
    )
    widget.set_row(uncalibrated_row, unit=PlotUnit.NM)

    assert widget.current_series is None
    assert widget.stack.currentWidget() is widget.placeholder_label
    assert "calibrated points" in widget.placeholder_label.text().lower()
