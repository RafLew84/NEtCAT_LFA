"""Tests for AtomMapper row-axis fitting helpers."""

from __future__ import annotations

import math

import pytest

from AtomMapper.app.models import AtomPoint, AtomRow
from AtomMapper.app.row_geometry import fit_row_geometry


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


def test_fit_row_geometry_returns_none_for_too_short_or_degenerate_rows():
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
                point_id="point-1",
                x_px=5.0,
                y_px=6.0,
            ),
        ),
    )
    degenerate_row = AtomRow(
        row_id="row-degenerate",
        source_group_id="group-1",
        display_name="Degenerate row",
        points=(
            _make_point(
                row_id="row-degenerate",
                source_group_id="group-1",
                image_id="image-1",
                point_index=0,
                point_id="point-1",
                x_px=2.0,
                y_px=2.0,
            ),
            _make_point(
                row_id="row-degenerate",
                source_group_id="group-1",
                image_id="image-1",
                point_index=1,
                point_id="point-2",
                x_px=2.0,
                y_px=2.0,
            ),
        ),
    )

    assert fit_row_geometry(short_row) is None
    assert fit_row_geometry(degenerate_row) is None


def test_fit_row_geometry_uses_svd_axis_and_orients_it_with_point_order():
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
                x_px=6.0,
                y_px=6.1,
            ),
            _make_point(
                row_id="row-1",
                source_group_id="group-1",
                image_id="image-1",
                point_index=0,
                point_id="point-1",
                x_px=0.0,
                y_px=0.1,
            ),
            _make_point(
                row_id="row-1",
                source_group_id="group-1",
                image_id="image-1",
                point_index=1,
                point_id="point-2",
                x_px=3.0,
                y_px=3.2,
            ),
        ),
    )

    geometry = fit_row_geometry(row)

    assert geometry is not None
    assert geometry.row_id == row.row_id
    assert geometry.fitted_point_count == 3
    assert geometry.has_nm_geometry is False
    assert geometry.reference_x_px == pytest.approx(3.0)
    assert geometry.reference_y_px == pytest.approx((0.1 + 3.2 + 6.1) / 3.0)
    assert geometry.direction_x_px > 0.0
    assert geometry.direction_y_px > 0.0
    assert math.hypot(geometry.direction_x_px, geometry.direction_y_px) == pytest.approx(1.0)
    assert geometry.span_length_px == pytest.approx(math.hypot(6.0, 6.0), rel=0.03)
    assert geometry.metadata["fit_method"] == "svd-pca"
    assert geometry.metadata["used_point_ids"] == ("point-1", "point-2", "point-3")


def test_fit_row_geometry_produces_nm_geometry_when_row_is_calibrated():
    row = AtomRow(
        row_id="row-2",
        source_group_id="group-1",
        display_name="Row 2",
        points=(
            _make_point(
                row_id="row-2",
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
                row_id="row-2",
                source_group_id="group-1",
                image_id="image-1",
                point_index=1,
                point_id="point-2",
                x_px=2.0,
                y_px=1.0,
                x_nm=1.0,
                y_nm=0.75,
            ),
            _make_point(
                row_id="row-2",
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

    geometry = fit_row_geometry(row)

    assert geometry is not None
    assert geometry.has_nm_geometry is True
    assert geometry.direction_x_px == pytest.approx(2.0 / math.sqrt(5.0))
    assert geometry.direction_y_px == pytest.approx(1.0 / math.sqrt(5.0))
    assert geometry.direction_x_nm == pytest.approx(0.8)
    assert geometry.direction_y_nm == pytest.approx(0.6)
    assert geometry.span_length_nm == pytest.approx(2.5)


def test_fit_row_geometry_marks_manual_override_in_metadata():
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
            ),
            _make_point(
                row_id="row-3",
                source_group_id="group-1",
                image_id="image-1",
                point_index=1,
                point_id="point-2",
                x_px=4.0,
                y_px=0.0,
                manual_override=True,
            ),
        ),
    )

    geometry = fit_row_geometry(row)

    assert geometry is not None
    assert geometry.metadata["uses_manual_override"] is True
