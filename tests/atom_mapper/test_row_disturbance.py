"""Tests for AtomMapper local row-disturbance helpers."""

from __future__ import annotations

import pytest

from AtomMapper.app.models import AtomPoint, AtomRow
from AtomMapper.app.row_geometry import RowGeometryUnit, build_row_disturbance_series


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
    )


def test_build_row_disturbance_series_detects_spacing_jump_candidates():
    row = AtomRow(
        row_id="row-spacing",
        source_group_id="group-1",
        display_name="Spacing row",
        points=(
            _make_point(
                row_id="row-spacing",
                source_group_id="group-1",
                image_id="image-1",
                point_index=0,
                point_id="point-0",
                x_px=0.0,
                y_px=0.0,
            ),
            _make_point(
                row_id="row-spacing",
                source_group_id="group-1",
                image_id="image-1",
                point_index=1,
                point_id="point-1",
                x_px=1.0,
                y_px=0.0,
            ),
            _make_point(
                row_id="row-spacing",
                source_group_id="group-1",
                image_id="image-1",
                point_index=2,
                point_id="point-2",
                x_px=2.0,
                y_px=0.0,
            ),
            _make_point(
                row_id="row-spacing",
                source_group_id="group-1",
                image_id="image-1",
                point_index=3,
                point_id="point-3",
                x_px=5.0,
                y_px=0.0,
            ),
            _make_point(
                row_id="row-spacing",
                source_group_id="group-1",
                image_id="image-1",
                point_index=4,
                point_id="point-4",
                x_px=6.0,
                y_px=0.0,
            ),
        ),
    )

    series = build_row_disturbance_series(
        row,
        spacing_jump_threshold=1.0,
        transverse_jump_threshold=999.0,
        direction_change_threshold_deg=180.0,
    )

    assert series is not None
    assert series.unit is RowGeometryUnit.PX
    assert series.candidate_count == 2
    assert [sample.point_id for sample in series.samples] == ["point-1", "point-2", "point-3"]
    assert [sample.spacing_jump_abs for sample in series.samples] == pytest.approx([0.0, 2.0, 2.0])
    assert [sample.is_candidate_spacing for sample in series.samples] == [False, True, True]
    assert all(sample.is_candidate_transverse is False for sample in series.samples)
    assert all(sample.is_candidate_direction is False for sample in series.samples)


def test_build_row_disturbance_series_detects_transverse_jump_candidates():
    row = AtomRow(
        row_id="row-transverse",
        source_group_id="group-1",
        display_name="Transverse row",
        points=(
            _make_point(
                row_id="row-transverse",
                source_group_id="group-1",
                image_id="image-1",
                point_index=0,
                point_id="point-0",
                x_px=0.0,
                y_px=0.0,
            ),
            _make_point(
                row_id="row-transverse",
                source_group_id="group-1",
                image_id="image-1",
                point_index=1,
                point_id="point-1",
                x_px=1.0,
                y_px=0.0,
            ),
            _make_point(
                row_id="row-transverse",
                source_group_id="group-1",
                image_id="image-1",
                point_index=2,
                point_id="point-2",
                x_px=2.0,
                y_px=0.0,
            ),
            _make_point(
                row_id="row-transverse",
                source_group_id="group-1",
                image_id="image-1",
                point_index=3,
                point_id="point-3",
                x_px=3.0,
                y_px=2.0,
            ),
            _make_point(
                row_id="row-transverse",
                source_group_id="group-1",
                image_id="image-1",
                point_index=4,
                point_id="point-4",
                x_px=4.0,
                y_px=2.0,
            ),
        ),
    )

    series = build_row_disturbance_series(
        row,
        spacing_jump_threshold=999.0,
        transverse_jump_threshold=0.5,
        direction_change_threshold_deg=180.0,
    )

    assert series is not None
    candidate_ids = {sample.point_id for sample in series.samples if sample.is_candidate_transverse}
    assert candidate_ids.issuperset({"point-2", "point-3"})
    assert series.candidate_count >= 2
    assert all(sample.is_candidate_spacing is False for sample in series.samples)


def test_build_row_disturbance_series_detects_local_direction_change_candidates():
    row = AtomRow(
        row_id="row-bend",
        source_group_id="group-1",
        display_name="Bend row",
        points=(
            _make_point(
                row_id="row-bend",
                source_group_id="group-1",
                image_id="image-1",
                point_index=0,
                point_id="point-0",
                x_px=0.0,
                y_px=0.0,
            ),
            _make_point(
                row_id="row-bend",
                source_group_id="group-1",
                image_id="image-1",
                point_index=1,
                point_id="point-1",
                x_px=1.0,
                y_px=0.0,
            ),
            _make_point(
                row_id="row-bend",
                source_group_id="group-1",
                image_id="image-1",
                point_index=2,
                point_id="point-2",
                x_px=2.0,
                y_px=0.0,
            ),
            _make_point(
                row_id="row-bend",
                source_group_id="group-1",
                image_id="image-1",
                point_index=3,
                point_id="point-3",
                x_px=2.0,
                y_px=1.0,
            ),
            _make_point(
                row_id="row-bend",
                source_group_id="group-1",
                image_id="image-1",
                point_index=4,
                point_id="point-4",
                x_px=2.0,
                y_px=2.0,
            ),
        ),
    )

    series = build_row_disturbance_series(
        row,
        spacing_jump_threshold=999.0,
        transverse_jump_threshold=999.0,
        direction_change_threshold_deg=30.0,
    )

    assert series is not None
    candidate_ids = {sample.point_id for sample in series.samples if sample.is_candidate_direction}
    assert candidate_ids == {"point-2"}
    assert max(sample.local_direction_change_deg for sample in series.samples) == pytest.approx(90.0)


def test_build_row_disturbance_series_supports_nm_and_returns_none_for_too_short_rows():
    row = AtomRow(
        row_id="row-nm",
        source_group_id="group-1",
        display_name="NM row",
        points=(
            _make_point(
                row_id="row-nm",
                source_group_id="group-1",
                image_id="image-1",
                point_index=0,
                point_id="point-0",
                x_px=0.0,
                y_px=0.0,
                x_nm=0.0,
                y_nm=0.0,
            ),
            _make_point(
                row_id="row-nm",
                source_group_id="group-1",
                image_id="image-1",
                point_index=1,
                point_id="point-1",
                x_px=2.0,
                y_px=0.0,
                x_nm=1.0,
                y_nm=0.0,
            ),
            _make_point(
                row_id="row-nm",
                source_group_id="group-1",
                image_id="image-1",
                point_index=2,
                point_id="point-2",
                x_px=4.0,
                y_px=0.0,
                x_nm=2.0,
                y_nm=0.0,
            ),
        ),
    )
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
                point_id="short-0",
                x_px=0.0,
                y_px=0.0,
            ),
            _make_point(
                row_id="row-short",
                source_group_id="group-1",
                image_id="image-1",
                point_index=1,
                point_id="short-1",
                x_px=1.0,
                y_px=0.0,
            ),
        ),
    )

    series = build_row_disturbance_series(
        row,
        unit=RowGeometryUnit.NM,
        spacing_jump_threshold=0.2,
        transverse_jump_threshold=0.2,
        direction_change_threshold_deg=5.0,
    )

    assert series is not None
    assert series.unit is RowGeometryUnit.NM
    assert len(series.samples) == 1
    assert series.samples[0].spacing_before == pytest.approx(1.0)
    assert series.samples[0].spacing_after == pytest.approx(1.0)
    assert build_row_disturbance_series(short_row) is None
