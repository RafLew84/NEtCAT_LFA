"""Tests for AtomMapper row-projection helpers."""

from __future__ import annotations

import pytest

from AtomMapper.app.models import AtomPoint, AtomRow
from AtomMapper.app.row_geometry import (
    RowGeometryUnit,
    RowProjectionSortMode,
    fit_row_geometry,
    project_row_points,
)


def _make_point(
    *,
    row_id: str,
    source_group_id: str,
    image_id: str,
    point_index: int,
    point_id: str,
    x_px: float,
    y_px: float,
    x_nm: float | None = None,
    y_nm: float | None = None,
    manual_override: bool = False,
) -> AtomPoint:
    return AtomPoint(
        row_id=row_id,
        source_group_id=source_group_id,
        image_id=image_id,
        point_index=point_index,
        point_id=point_id,
        x_px=x_px,
        y_px=y_px,
        x_nm=x_nm,
        y_nm=y_nm,
        manual_override=manual_override,
    )


def test_project_row_points_returns_px_samples_sorted_by_point_index_by_default():
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
                point_id="point-3",
                x_px=4.0,
                y_px=0.0,
            ),
            _make_point(
                row_id="row-1",
                source_group_id="group-1",
                image_id="image-1",
                point_index=0,
                point_id="point-1",
                x_px=0.0,
                y_px=0.0,
            ),
            _make_point(
                row_id="row-1",
                source_group_id="group-1",
                image_id="image-1",
                point_index=1,
                point_id="point-2",
                x_px=2.0,
                y_px=0.0,
            ),
        ),
    )

    series = project_row_points(row)

    assert series is not None
    assert series.unit is RowGeometryUnit.PX
    assert series.sort_mode is RowProjectionSortMode.POINT_INDEX
    assert [sample.point_id for sample in series.samples] == ["point-1", "point-2", "point-3"]
    assert [sample.along_value for sample in series.samples] == pytest.approx([-2.0, 0.0, 2.0])
    assert [sample.transverse_value for sample in series.samples] == pytest.approx([0.0, 0.0, 0.0])


def test_project_row_points_can_sort_by_projection_along_axis():
    row = AtomRow(
        row_id="row-2",
        source_group_id="group-1",
        display_name="Row 2",
        points=(
            _make_point(
                row_id="row-2",
                source_group_id="group-1",
                image_id="image-1",
                point_index=9,
                point_id="point-right",
                x_px=6.0,
                y_px=0.0,
            ),
            _make_point(
                row_id="row-2",
                source_group_id="group-1",
                image_id="image-1",
                point_index=1,
                point_id="point-center",
                x_px=3.0,
                y_px=0.0,
            ),
            _make_point(
                row_id="row-2",
                source_group_id="group-1",
                image_id="image-1",
                point_index=7,
                point_id="point-left",
                x_px=0.0,
                y_px=0.0,
            ),
        ),
    )

    series = project_row_points(row, sort_mode=RowProjectionSortMode.ALONG_AXIS)

    assert series is not None
    assert series.sort_mode is RowProjectionSortMode.ALONG_AXIS
    assert [sample.point_id for sample in series.samples] == [
        "point-left",
        "point-center",
        "point-right",
    ]
    assert [sample.along_value for sample in series.samples] == pytest.approx([-3.0, 0.0, 3.0])


def test_project_row_points_supports_nm_projection_when_calibrated():
    row = AtomRow(
        row_id="row-3",
        source_group_id="group-1",
        display_name="Row 3",
        points=(
            _make_point(
                row_id="row-3",
                source_group_id="group-1",
                image_id="image-1",
                point_index=0,
                point_id="point-1",
                x_px=0.0,
                y_px=0.0,
                x_nm=0.0,
                y_nm=0.0,
            ),
            _make_point(
                row_id="row-3",
                source_group_id="group-1",
                image_id="image-1",
                point_index=1,
                point_id="point-2",
                x_px=2.0,
                y_px=1.0,
                x_nm=1.0,
                y_nm=0.75,
                manual_override=True,
            ),
            _make_point(
                row_id="row-3",
                source_group_id="group-1",
                image_id="image-1",
                point_index=2,
                point_id="point-3",
                x_px=4.0,
                y_px=2.0,
                x_nm=2.0,
                y_nm=1.5,
            ),
        ),
    )

    series = project_row_points(row, unit=RowGeometryUnit.NM)

    assert series is not None
    assert series.unit is RowGeometryUnit.NM
    assert series.geometry.has_nm_geometry is True
    assert [sample.along_value for sample in series.samples] == pytest.approx([-1.25, 0.0, 1.25])
    assert [sample.transverse_value for sample in series.samples] == pytest.approx([0.0, 0.0, 0.0])
    assert series.samples[1].is_manual_override is True


def test_project_row_points_returns_none_for_missing_nm_or_mismatched_geometry():
    row = AtomRow(
        row_id="row-4",
        source_group_id="group-1",
        display_name="Row 4",
        points=(
            _make_point(
                row_id="row-4",
                source_group_id="group-1",
                image_id="image-1",
                point_index=0,
                point_id="point-1",
                x_px=0.0,
                y_px=0.0,
                x_nm=0.0,
                y_nm=0.0,
            ),
            _make_point(
                row_id="row-4",
                source_group_id="group-1",
                image_id="image-1",
                point_index=1,
                point_id="point-2",
                x_px=1.0,
                y_px=0.0,
                x_nm=None,
                y_nm=None,
            ),
        ),
    )

    assert project_row_points(row, unit=RowGeometryUnit.NM) is None

    other_row = AtomRow(
        row_id="row-other",
        source_group_id="group-1",
        display_name="Other row",
        points=(
            _make_point(
                row_id="row-other",
                source_group_id="group-1",
                image_id="image-1",
                point_index=0,
                point_id="other-point-1",
                x_px=0.0,
                y_px=0.0,
            ),
            _make_point(
                row_id="row-other",
                source_group_id="group-1",
                image_id="image-1",
                point_index=1,
                point_id="other-point-2",
                x_px=2.0,
                y_px=0.0,
            ),
        ),
    )

    other_geometry = fit_row_geometry(other_row)
    assert other_geometry is not None

    with pytest.raises(ValueError, match="geometry.row_id"):
        project_row_points(row, geometry=other_geometry)
