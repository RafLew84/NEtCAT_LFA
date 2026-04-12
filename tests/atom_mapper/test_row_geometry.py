"""Tests for AtomMapper row-geometry data contracts."""

from __future__ import annotations

import math

import pytest

from AtomMapper.app.row_geometry import RowGeometry


def test_row_geometry_normalizes_direction_vectors_and_roundtrips():
    geometry = RowGeometry(
        row_id="row-1",
        row_display_name="Row 1",
        point_count=5,
        fitted_point_count=4,
        reference_x_px=10.0,
        reference_y_px=20.0,
        direction_x_px=3.0,
        direction_y_px=4.0,
        span_length_px=15.0,
        reference_x_nm=1.0,
        reference_y_nm=2.0,
        direction_x_nm=6.0,
        direction_y_nm=8.0,
        span_length_nm=1.5,
        metadata={"method": "planned-fit"},
    )

    assert geometry.has_nm_geometry is True
    assert geometry.direction_x_px == pytest.approx(0.6)
    assert geometry.direction_y_px == pytest.approx(0.8)
    assert geometry.direction_x_nm == pytest.approx(0.6)
    assert geometry.direction_y_nm == pytest.approx(0.8)
    assert math.hypot(geometry.direction_x_px, geometry.direction_y_px) == pytest.approx(1.0)
    assert math.hypot(geometry.direction_x_nm, geometry.direction_y_nm) == pytest.approx(1.0)

    restored = RowGeometry.from_dict(geometry.to_dict())
    assert restored == geometry


def test_row_geometry_supports_px_only_contract():
    geometry = RowGeometry(
        row_id="row-2",
        row_display_name="Row 2",
        point_count=3,
        fitted_point_count=3,
        reference_x_px=5.0,
        reference_y_px=7.0,
        direction_x_px=1.0,
        direction_y_px=0.0,
        span_length_px=9.5,
    )

    assert geometry.has_nm_geometry is False
    assert geometry.reference_x_nm is None
    assert geometry.direction_x_nm is None
    assert geometry.span_length_nm is None


def test_row_geometry_rejects_partial_nm_payload():
    with pytest.raises(ValueError, match="all nm fields"):
        RowGeometry(
            row_id="row-3",
            row_display_name="Row 3",
            point_count=3,
            fitted_point_count=2,
            reference_x_px=0.0,
            reference_y_px=0.0,
            direction_x_px=1.0,
            direction_y_px=1.0,
            span_length_px=4.0,
            reference_x_nm=0.0,
            reference_y_nm=0.0,
            direction_x_nm=1.0,
            direction_y_nm=1.0,
        )


def test_row_geometry_rejects_invalid_counts_or_zero_direction():
    with pytest.raises(ValueError, match="point_count"):
        RowGeometry(
            row_id="row-4",
            row_display_name="Row 4",
            point_count=0,
            fitted_point_count=0,
            reference_x_px=0.0,
            reference_y_px=0.0,
            direction_x_px=1.0,
            direction_y_px=0.0,
            span_length_px=1.0,
        )

    with pytest.raises(ValueError, match="direction vector"):
        RowGeometry(
            row_id="row-4",
            row_display_name="Row 4",
            point_count=2,
            fitted_point_count=2,
            reference_x_px=0.0,
            reference_y_px=0.0,
            direction_x_px=0.0,
            direction_y_px=0.0,
            span_length_px=1.0,
        )
