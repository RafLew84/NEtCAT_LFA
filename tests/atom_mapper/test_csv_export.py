"""Tests for AtomMapper CSV export helpers."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from AtomMapper.app.csv_export import (
    POINT_EXPORT_FIELDNAMES,
    build_point_export_rows,
    describe_point_status,
    export_point_rows_to_csv,
)
from AtomMapper.app.models import AtomPoint, AtomRow, LoadedImage


def _make_loaded_image(name: str, width: int = 8, height: int = 6) -> LoadedImage:
    image_data = np.arange(width * height, dtype=float).reshape((height, width))
    return LoadedImage(
        source_path=str(Path("/tmp") / name),
        display_name=name,
        file_extension=Path(name).suffix.lower(),
        image_data=image_data,
        pixels_x=width,
        pixels_y=height,
        size_nm_x=float(width),
        size_nm_y=float(height),
        metadata={"image_type": "Topo"},
        raw_metadata={},
    )


def test_build_point_export_rows_includes_px_nm_fit_and_status_fields():
    original = _make_loaded_image("sample.stp")
    variant = original.derive_variant(variant_name="blur", image_data=original.image_data + 1.0)
    row = AtomRow(
        row_id="row-1",
        source_group_id=original.source_group_id,
        display_name="Row 1",
        points=(
            AtomPoint(
                row_id="row-1",
                image_id=original.image_id,
                source_group_id=original.source_group_id,
                point_index=0,
                x_px=1.0,
                y_px=2.0,
                x_nm=1.0,
                y_nm=2.0,
                point_id="point-1",
                fit_success=True,
                metadata={
                    "fit_model": "gaussian",
                    "fit_method": "gaussian_fit",
                    "fit_mask_active": False,
                    "fit_mask_pixel_count": None,
                },
            ),
            AtomPoint(
                row_id="row-1",
                image_id=variant.image_id,
                source_group_id=original.source_group_id,
                point_index=1,
                x_px=3.5,
                y_px=4.5,
                x_nm=3.5,
                y_nm=4.5,
                point_id="point-2",
                amplitude=17.0,
                sigma_x_px=1.1,
                sigma_y_px=1.2,
                position_std_x_px=0.31,
                position_std_y_px=0.42,
                position_std_x_nm=0.031,
                position_std_y_nm=0.042,
                theta_deg=15.0,
                offset=0.5,
                fit_success=False,
                fit_error_message="fallback",
                manual_override=True,
                manual_override_source="drag",
                metadata={
                    "fit_model": "voigt",
                    "fit_method": "voigt_fit",
                    "fit_mask_active": True,
                    "fit_mask_pixel_count": 37,
                    "position_uncertainty_status": "recomputed",
                    "position_uncertainty_method": "fit_covariance",
                    "position_uncertainty_reference": "original_fit_position",
                    "position_uncertainty_settings_source": "session_fallback",
                    "position_uncertainty_original_mask_missing": True,
                    "position_uncertainty_retry_status": "succeeded_at_constraint_boundary",
                    "position_uncertainty_retry_at_bound": True,
                    "position_uncertainty_covariance_condition": 1234.5,
                },
            ),
        ),
    )

    export_rows = build_point_export_rows((row,), (original, variant))

    assert len(export_rows) == 2
    assert tuple(export_rows[0].keys()) == POINT_EXPORT_FIELDNAMES
    assert export_rows[1]["image_name"] == variant.display_name
    assert export_rows[1]["image_variant"] == "blur"
    assert export_rows[1]["row_name"] == "Row 1"
    assert export_rows[1]["previous_point_id"] == "point-1"
    assert export_rows[1]["next_point_id"] == ""
    assert export_rows[1]["x_px"] == "3.500000"
    assert export_rows[1]["y_nm"] == "4.500000"
    assert export_rows[1]["distance_to_previous_px"] == "3.535534"
    assert export_rows[1]["distance_to_next_px"] == ""
    assert export_rows[1]["distance_to_previous_nm"] == "3.535534"
    assert export_rows[1]["distance_to_next_nm"] == ""
    assert export_rows[1]["amplitude"] == "17.000000"
    assert export_rows[1]["sigma_x_px"] == "1.100000"
    assert export_rows[1]["position_std_x_px"] == "0.310000"
    assert export_rows[1]["position_std_y_px"] == "0.420000"
    assert export_rows[1]["position_std_x_nm"] == "0.031000"
    assert export_rows[1]["position_std_y_nm"] == "0.042000"
    assert export_rows[1]["position_uncertainty_status"] == "recomputed"
    assert export_rows[1]["position_uncertainty_method"] == "fit_covariance"
    assert export_rows[1]["position_uncertainty_reference"] == "original_fit_position"
    assert export_rows[1]["position_uncertainty_settings_source"] == "session_fallback"
    assert export_rows[1]["position_uncertainty_original_mask_missing"] == "true"
    assert export_rows[1]["position_uncertainty_retry_status"] == (
        "succeeded_at_constraint_boundary"
    )
    assert export_rows[1]["position_uncertainty_retry_at_bound"] == "true"
    assert export_rows[1]["position_uncertainty_covariance_condition"] == "1234.500000"
    assert export_rows[1]["fit_success"] == "false"
    assert export_rows[1]["fit_model"] == "voigt"
    assert export_rows[1]["fit_method"] == "voigt_fit"
    assert export_rows[1]["fit_mask_active"] == "true"
    assert export_rows[1]["fit_mask_pixel_count"] == "37"
    assert export_rows[1]["manual_override"] == "true"
    assert export_rows[1]["manual_override_source"] == "drag"
    assert export_rows[1]["status"] == "manual (drag)"


def test_export_point_rows_to_csv_writes_header_and_rows(tmp_path: Path):
    image = _make_loaded_image("export.stp")
    row = AtomRow(
        row_id="row-1",
        source_group_id=image.source_group_id,
        display_name="Row 1",
        points=(
            AtomPoint(
                row_id="row-1",
                image_id=image.image_id,
                source_group_id=image.source_group_id,
                point_index=0,
                x_px=1.0,
                y_px=2.0,
                x_nm=1.0,
                y_nm=2.0,
                point_id="point-1",
                metadata={
                    "fit_method": "gaussian_fit",
                    "fit_mask_active": False,
                    "fit_mask_pixel_count": None,
                },
            ),
        ),
    )

    export_path = tmp_path / "points.csv"
    exported_count = export_point_rows_to_csv(export_path, (row,), (image,))

    assert exported_count == 1
    with export_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 1
    assert rows[0]["image_name"] == "export.stp"
    assert rows[0]["point_id"] == "point-1"
    assert rows[0]["previous_point_id"] == ""
    assert rows[0]["next_point_id"] == ""
    assert rows[0]["distance_to_previous_px"] == ""
    assert rows[0]["distance_to_next_px"] == ""
    assert rows[0]["distance_to_previous_nm"] == ""
    assert rows[0]["distance_to_next_nm"] == ""
    assert rows[0]["status"] == "fit"
    assert rows[0]["x_nm"] == "1.000000"
    assert rows[0]["fit_model"] == "gaussian"
    assert rows[0]["fit_method"] == "gaussian_fit"
    assert rows[0]["fit_mask_active"] == "false"
    assert rows[0]["fit_mask_pixel_count"] == ""
    assert rows[0]["manual_override"] == "false"


def test_describe_point_status_prefers_manual_override():
    point = AtomPoint(
        row_id="row-1",
        image_id="image-1",
        source_group_id="group-1",
        point_index=0,
        x_px=1.0,
        y_px=2.0,
        point_id="point-1",
        manual_override=True,
        manual_override_source="drag",
        fit_success=False,
        metadata={"fallback_used": True},
    )

    assert describe_point_status(point) == "manual (drag)"
