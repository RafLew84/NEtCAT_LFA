"""Tests for AtomMapper plot-series helpers."""

from __future__ import annotations

import pytest

from AtomMapper.app.models import AtomPoint, AtomRow
from AtomMapper.app.plots import (
    RowPlotMode,
    build_global_scatter_series,
    build_row_distance_metrics,
    build_row_metric_series,
    sorted_row_points,
)


def _make_point(
    *,
    row_id: str,
    source_group_id: str,
    image_id: str,
    point_index: int,
    x_px: float,
    y_px: float,
    point_id: str,
    manual_override: bool = False,
    manual_override_source: str | None = None,
) -> AtomPoint:
    return AtomPoint(
        row_id=row_id,
        image_id=image_id,
        source_group_id=source_group_id,
        point_index=point_index,
        x_px=x_px,
        y_px=y_px,
        point_id=point_id,
        manual_override=manual_override,
        manual_override_source=manual_override_source,
    )


def test_sorted_row_points_orders_by_point_index_then_point_id():
    row = AtomRow(
        row_id="row-1",
        source_group_id="group-1",
        display_name="Row 1",
        points=(
            _make_point(
                row_id="row-1",
                source_group_id="group-1",
                image_id="image-1",
                point_index=2,
                x_px=8.0,
                y_px=9.0,
                point_id="point-b",
            ),
            _make_point(
                row_id="row-1",
                source_group_id="group-1",
                image_id="image-1",
                point_index=0,
                x_px=1.0,
                y_px=2.0,
                point_id="point-a",
            ),
            _make_point(
                row_id="row-1",
                source_group_id="group-1",
                image_id="image-1",
                point_index=2,
                x_px=7.0,
                y_px=8.0,
                point_id="point-a2",
            ),
        ),
    )

    sorted_points = sorted_row_points(row)

    assert [point.point_id for point in sorted_points] == ["point-a", "point-a2", "point-b"]


def test_build_row_metric_series_supports_x_and_y_modes():
    row = AtomRow(
        row_id="row-1",
        source_group_id="group-1",
        display_name="Row 1",
        points=(
            _make_point(
                row_id="row-1",
                source_group_id="group-1",
                image_id="image-1",
                point_index=0,
                x_px=10.0,
                y_px=20.0,
                point_id="point-1",
            ),
            _make_point(
                row_id="row-1",
                source_group_id="group-1",
                image_id="image-1",
                point_index=1,
                x_px=12.5,
                y_px=21.5,
                point_id="point-2",
            ),
        ),
    )

    x_series = build_row_metric_series(row, RowPlotMode.X_PX)
    y_series = build_row_metric_series(row, RowPlotMode.Y_PX)

    assert x_series.row_id == row.row_id
    assert x_series.x_label == "point index"
    assert x_series.y_label == "x (px)"
    assert [(sample.x_value, sample.y_value) for sample in x_series.samples] == [
        (0.0, 10.0),
        (1.0, 12.5),
    ]

    assert y_series.row_id == row.row_id
    assert y_series.y_label == "y (px)"
    assert [(sample.x_value, sample.y_value) for sample in y_series.samples] == [
        (0.0, 20.0),
        (1.0, 21.5),
    ]


def test_build_row_metric_series_distance_mode_uses_consecutive_point_spacing():
    row = AtomRow(
        row_id="row-1",
        source_group_id="group-1",
        display_name="Row 1",
        points=(
            _make_point(
                row_id="row-1",
                source_group_id="group-1",
                image_id="image-1",
                point_index=0,
                x_px=0.0,
                y_px=0.0,
                point_id="point-1",
            ),
            _make_point(
                row_id="row-1",
                source_group_id="group-1",
                image_id="image-1",
                point_index=1,
                x_px=3.0,
                y_px=4.0,
                point_id="point-2",
            ),
            _make_point(
                row_id="row-1",
                source_group_id="group-1",
                image_id="image-1",
                point_index=2,
                x_px=6.0,
                y_px=8.0,
                point_id="point-3",
            ),
        ),
    )

    series = build_row_metric_series(row, RowPlotMode.DISTANCE_PX)

    assert series.x_label == "segment index"
    assert series.y_label == "distance (px)"
    assert len(series.samples) == 2
    assert series.samples[0].x_value == pytest.approx(0.0)
    assert series.samples[0].y_value == pytest.approx(5.0)
    assert series.samples[1].x_value == pytest.approx(1.0)
    assert series.samples[1].y_value == pytest.approx(5.0)


def test_build_global_scatter_series_collects_all_rows_and_manual_status():
    row_1 = AtomRow(
        row_id="row-1",
        source_group_id="group-1",
        display_name="Alpha",
        color_hex="#ff0000",
        points=(
            _make_point(
                row_id="row-1",
                source_group_id="group-1",
                image_id="image-1",
                point_index=1,
                x_px=3.0,
                y_px=4.0,
                point_id="point-2",
                manual_override=True,
                manual_override_source="drag",
            ),
        ),
    )
    row_2 = AtomRow(
        row_id="row-2",
        source_group_id="group-1",
        display_name="Beta",
        color_hex="#00ff00",
        points=(
            _make_point(
                row_id="row-2",
                source_group_id="group-1",
                image_id="image-2",
                point_index=0,
                x_px=1.0,
                y_px=2.0,
                point_id="point-1",
            ),
        ),
    )

    series = build_global_scatter_series((row_2, row_1))

    assert series.x_label == "x (px)"
    assert series.y_label == "y (px)"
    assert [sample.row_display_name for sample in series.samples] == ["Alpha", "Beta"]
    assert series.samples[0].point_id == "point-2"
    assert series.samples[0].is_manual_override is True
    assert series.samples[0].color_hex == "#ff0000"
    assert series.samples[1].point_id == "point-1"
    assert series.samples[1].is_manual_override is False


def test_build_row_distance_metrics_returns_summary_statistics():
    row = AtomRow(
        row_id="row-1",
        source_group_id="group-1",
        display_name="Row 1",
        points=(
            _make_point(
                row_id="row-1",
                source_group_id="group-1",
                image_id="image-1",
                point_index=0,
                x_px=0.0,
                y_px=0.0,
                point_id="point-1",
            ),
            _make_point(
                row_id="row-1",
                source_group_id="group-1",
                image_id="image-1",
                point_index=1,
                x_px=3.0,
                y_px=4.0,
                point_id="point-2",
            ),
            _make_point(
                row_id="row-1",
                source_group_id="group-1",
                image_id="image-1",
                point_index=2,
                x_px=9.0,
                y_px=12.0,
                point_id="point-3",
            ),
        ),
    )

    metrics = build_row_distance_metrics(row)

    assert metrics.row_id == row.row_id
    assert metrics.point_count == 3
    assert metrics.distance_count == 2
    assert metrics.mean_distance_px == pytest.approx(7.5)
    assert metrics.std_distance_px == pytest.approx(2.5)
    assert metrics.min_distance_px == pytest.approx(5.0)
    assert metrics.max_distance_px == pytest.approx(10.0)


def test_build_row_distance_metrics_handles_rows_with_fewer_than_two_points():
    row = AtomRow(
        row_id="row-1",
        source_group_id="group-1",
        display_name="Row 1",
        points=(
            _make_point(
                row_id="row-1",
                source_group_id="group-1",
                image_id="image-1",
                point_index=0,
                x_px=1.0,
                y_px=2.0,
                point_id="point-1",
            ),
        ),
    )

    metrics = build_row_distance_metrics(row)

    assert metrics.point_count == 1
    assert metrics.distance_count == 0
    assert metrics.mean_distance_px is None
    assert metrics.std_distance_px is None
    assert metrics.min_distance_px is None
    assert metrics.max_distance_px is None
