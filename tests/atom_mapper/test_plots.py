"""Tests for AtomMapper plot-series helpers."""

from __future__ import annotations

import pytest

from AtomMapper.app.models import AtomPoint, AtomRow
from AtomMapper.app.plots import (
    PlotUnit,
    RowPlotMode,
    build_global_scatter_series,
    build_row_distance_metrics,
    build_row_geometry_metrics,
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
    x_nm: float | None = None,
    y_nm: float | None = None,
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
        x_nm=x_nm,
        y_nm=y_nm,
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


def test_build_row_metric_series_supports_nm_modes_and_skips_uncalibrated_samples():
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
                x_nm=1.0,
                y_nm=2.0,
                point_id="point-1",
            ),
            _make_point(
                row_id="row-1",
                source_group_id="group-1",
                image_id="image-1",
                point_index=1,
                x_px=12.5,
                y_px=21.5,
                x_nm=None,
                y_nm=2.15,
                point_id="point-2",
            ),
            _make_point(
                row_id="row-1",
                source_group_id="group-1",
                image_id="image-1",
                point_index=2,
                x_px=15.0,
                y_px=25.5,
                x_nm=1.5,
                y_nm=2.55,
                point_id="point-3",
            ),
        ),
    )

    x_series = build_row_metric_series(row, RowPlotMode.X_NM)
    y_series = build_row_metric_series(row, RowPlotMode.Y_NM)

    assert x_series.y_label == "x (nm)"
    assert [(sample.x_value, sample.y_value) for sample in x_series.samples] == [
        (0.0, 1.0),
        (2.0, 1.5),
    ]
    assert y_series.y_label == "y (nm)"
    assert [(sample.x_value, sample.y_value) for sample in y_series.samples] == [
        (0.0, 2.0),
        (1.0, 2.15),
        (2.0, 2.55),
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


def test_build_row_metric_series_distance_nm_uses_calibrated_segments_only():
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
                x_nm=0.0,
                y_nm=0.0,
                point_id="point-1",
            ),
            _make_point(
                row_id="row-1",
                source_group_id="group-1",
                image_id="image-1",
                point_index=1,
                x_px=3.0,
                y_px=4.0,
                x_nm=0.3,
                y_nm=0.4,
                point_id="point-2",
            ),
            _make_point(
                row_id="row-1",
                source_group_id="group-1",
                image_id="image-1",
                point_index=2,
                x_px=6.0,
                y_px=8.0,
                x_nm=None,
                y_nm=None,
                point_id="point-3",
            ),
        ),
    )

    series = build_row_metric_series(row, RowPlotMode.DISTANCE_NM)

    assert series.y_label == "distance (nm)"
    assert len(series.samples) == 1
    assert series.samples[0].x_value == pytest.approx(0.0)
    assert series.samples[0].y_value == pytest.approx(0.5)


def test_build_row_metric_series_supports_along_and_transverse_geometry_modes():
    row = AtomRow(
        row_id="row-geo",
        source_group_id="group-1",
        display_name="Geometry row",
        points=(
            _make_point(
                row_id="row-geo",
                source_group_id="group-1",
                image_id="image-1",
                point_index=0,
                x_px=0.0,
                y_px=0.0,
                point_id="point-1",
            ),
            _make_point(
                row_id="row-geo",
                source_group_id="group-1",
                image_id="image-1",
                point_index=1,
                x_px=2.0,
                y_px=0.5,
                point_id="point-2",
            ),
            _make_point(
                row_id="row-geo",
                source_group_id="group-1",
                image_id="image-1",
                point_index=2,
                x_px=4.0,
                y_px=1.0,
                point_id="point-3",
            ),
        ),
    )

    along_series = build_row_metric_series(row, RowPlotMode.ALONG_PX)
    transverse_series = build_row_metric_series(row, RowPlotMode.TRANSVERSE_PX)

    assert along_series.x_label == "point index"
    assert along_series.y_label == "along (px)"
    assert [sample.x_value for sample in along_series.samples] == pytest.approx([0.0, 1.0, 2.0])
    assert [sample.y_value for sample in along_series.samples] == pytest.approx(
        [-2.0615528, 0.0, 2.0615528],
        rel=1e-6,
    )

    assert transverse_series.x_label == "point index"
    assert transverse_series.y_label == "transverse (px)"
    assert [sample.y_value for sample in transverse_series.samples] == pytest.approx(
        [0.0, 0.0, 0.0],
        abs=1e-9,
    )


def test_build_row_metric_series_supports_spacing_along_nm_with_axis_sorting():
    row = AtomRow(
        row_id="row-spacing",
        source_group_id="group-1",
        display_name="Spacing row",
        points=(
            _make_point(
                row_id="row-spacing",
                source_group_id="group-1",
                image_id="image-1",
                point_index=9,
                x_px=4.0,
                y_px=0.0,
                x_nm=2.0,
                y_nm=0.0,
                point_id="point-right",
            ),
            _make_point(
                row_id="row-spacing",
                source_group_id="group-1",
                image_id="image-1",
                point_index=1,
                x_px=0.0,
                y_px=0.0,
                x_nm=0.0,
                y_nm=0.0,
                point_id="point-left",
            ),
            _make_point(
                row_id="row-spacing",
                source_group_id="group-1",
                image_id="image-1",
                point_index=7,
                x_px=2.0,
                y_px=0.0,
                x_nm=1.0,
                y_nm=0.0,
                point_id="point-center",
            ),
        ),
    )

    series = build_row_metric_series(row, RowPlotMode.SPACING_ALONG_NM)

    assert series.x_label == "segment index"
    assert series.y_label == "spacing along (nm)"
    assert [sample.x_value for sample in series.samples] == pytest.approx([0.0, 1.0])
    assert [sample.y_value for sample in series.samples] == pytest.approx([1.0, 1.0])


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
                x_nm=0.0,
                y_nm=0.0,
                point_id="point-1",
            ),
            _make_point(
                row_id="row-1",
                source_group_id="group-1",
                image_id="image-1",
                point_index=1,
                x_px=3.0,
                y_px=4.0,
                x_nm=0.3,
                y_nm=0.4,
                point_id="point-2",
            ),
            _make_point(
                row_id="row-1",
                source_group_id="group-1",
                image_id="image-1",
                point_index=2,
                x_px=9.0,
                y_px=12.0,
                x_nm=0.9,
                y_nm=1.2,
                point_id="point-3",
            ),
        ),
    )

    metrics = build_row_distance_metrics(row)

    assert metrics.row_id == row.row_id
    assert metrics.point_count == 3
    assert metrics.distance_count == 2
    assert metrics.distance_count_for_unit(PlotUnit.PX) == 2
    assert metrics.distance_count_for_unit(PlotUnit.NM) == 2
    assert metrics.mean_distance_px == pytest.approx(7.5)
    assert metrics.std_distance_px == pytest.approx(2.5)
    assert metrics.min_distance_px == pytest.approx(5.0)
    assert metrics.max_distance_px == pytest.approx(10.0)
    assert metrics.mean_distance_nm == pytest.approx(0.75)
    assert metrics.std_distance_nm == pytest.approx(0.25)
    assert metrics.min_distance_nm == pytest.approx(0.5)
    assert metrics.max_distance_nm == pytest.approx(1.0)


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
    assert metrics.distance_count_for_unit(PlotUnit.NM) == 0
    assert metrics.mean_distance_px is None
    assert metrics.std_distance_px is None
    assert metrics.min_distance_px is None
    assert metrics.max_distance_px is None
    assert metrics.mean_distance_nm is None
    assert metrics.std_distance_nm is None
    assert metrics.min_distance_nm is None
    assert metrics.max_distance_nm is None


def test_build_row_distance_metrics_leaves_nm_metrics_empty_without_calibrated_points():
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
                x_nm=0.0,
                y_nm=0.0,
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
        ),
    )

    metrics = build_row_distance_metrics(row)

    assert metrics.distance_count_for_unit(PlotUnit.PX) == 1
    assert metrics.distance_count_for_unit(PlotUnit.NM) == 0
    assert metrics.mean_distance_px == pytest.approx(5.0)
    assert metrics.mean_distance_nm is None


def test_build_row_geometry_metrics_returns_axis_and_spacing_statistics():
    row = AtomRow(
        row_id="row-geo",
        source_group_id="group-1",
        display_name="Geometry row",
        points=(
            _make_point(
                row_id="row-geo",
                source_group_id="group-1",
                image_id="image-1",
                point_index=0,
                x_px=0.0,
                y_px=0.0,
                x_nm=0.0,
                y_nm=0.0,
                point_id="point-1",
            ),
            _make_point(
                row_id="row-geo",
                source_group_id="group-1",
                image_id="image-1",
                point_index=1,
                x_px=3.0,
                y_px=4.0,
                x_nm=0.3,
                y_nm=0.4,
                point_id="point-2",
            ),
            _make_point(
                row_id="row-geo",
                source_group_id="group-1",
                image_id="image-1",
                point_index=2,
                x_px=9.0,
                y_px=12.0,
                x_nm=0.9,
                y_nm=1.2,
                point_id="point-3",
            ),
        ),
    )

    metrics = build_row_geometry_metrics(row)

    assert metrics.row_id == row.row_id
    assert metrics.point_count == 3
    assert metrics.fitted_point_count == 3
    assert metrics.spacing_count_for_unit(PlotUnit.PX) == 2
    assert metrics.spacing_count_for_unit(PlotUnit.NM) == 2
    assert metrics.axis_angle_deg_for_unit(PlotUnit.PX) == pytest.approx(53.130102, rel=1e-6)
    assert metrics.axis_angle_deg_for_unit(PlotUnit.NM) == pytest.approx(53.130102, rel=1e-6)
    assert metrics.transverse_rms_for_unit(PlotUnit.PX) == pytest.approx(0.0)
    assert metrics.transverse_rms_for_unit(PlotUnit.NM) == pytest.approx(0.0)
    assert metrics.mean_spacing_along_for_unit(PlotUnit.PX) == pytest.approx(7.5)
    assert metrics.std_spacing_along_for_unit(PlotUnit.PX) == pytest.approx(2.5)
    assert metrics.mean_spacing_along_for_unit(PlotUnit.NM) == pytest.approx(0.75)
    assert metrics.std_spacing_along_for_unit(PlotUnit.NM) == pytest.approx(0.25)


def test_build_row_geometry_metrics_handles_too_few_or_uncalibrated_points():
    short_row = AtomRow(
        row_id="row-short",
        source_group_id="group-1",
        display_name="Short row",
        points=(
            _make_point(
                row_id="row-short",
                source_group_id="group-1",
                image_id="image-1",
                point_index=0,
                x_px=1.0,
                y_px=2.0,
                point_id="point-1",
            ),
        ),
    )
    uncalibrated_row = AtomRow(
        row_id="row-uncal",
        source_group_id="group-1",
        display_name="Uncalibrated row",
        points=(
            _make_point(
                row_id="row-uncal",
                source_group_id="group-1",
                image_id="image-1",
                point_index=0,
                x_px=0.0,
                y_px=0.0,
                x_nm=0.0,
                y_nm=0.0,
                point_id="point-1",
            ),
            _make_point(
                row_id="row-uncal",
                source_group_id="group-1",
                image_id="image-1",
                point_index=1,
                x_px=3.0,
                y_px=4.0,
                point_id="point-2",
            ),
        ),
    )

    short_metrics = build_row_geometry_metrics(short_row)
    uncalibrated_metrics = build_row_geometry_metrics(uncalibrated_row)

    assert short_metrics.fitted_point_count == 0
    assert short_metrics.axis_angle_deg_for_unit(PlotUnit.PX) is None
    assert short_metrics.spacing_count_for_unit(PlotUnit.PX) == 0

    assert uncalibrated_metrics.fitted_point_count == 2
    assert uncalibrated_metrics.axis_angle_deg_for_unit(PlotUnit.PX) == pytest.approx(53.130102, rel=1e-6)
    assert uncalibrated_metrics.axis_angle_deg_for_unit(PlotUnit.NM) is None
    assert uncalibrated_metrics.spacing_count_for_unit(PlotUnit.PX) == 1
    assert uncalibrated_metrics.spacing_count_for_unit(PlotUnit.NM) == 0
