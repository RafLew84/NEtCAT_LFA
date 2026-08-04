"""Behavior tests for retrying implausibly large AtomMapper uncertainties."""

from __future__ import annotations

import numpy as np

from AtomMapper.app.fit_settings import FitSettingsState
from AtomMapper.app.models import AtomPoint, AtomRow, LoadedImage
from AtomMapper.app.position_uncertainty_retry import (
    has_very_large_position_uncertainty,
    retry_very_large_position_uncertainties,
)


def _make_gaussian_image() -> LoadedImage:
    y_grid, x_grid = np.mgrid[0:15, 0:15]
    image_data = 1.5 + 24.0 * np.exp(
        -(((y_grid - 7.2) ** 2) / (2.0 * 1.4**2) + ((x_grid - 6.7) ** 2) / (2.0 * 1.8**2))
    )
    return LoadedImage(
        source_path="/tmp/retry-uncertainty.stp",
        display_name="retry-uncertainty.stp",
        file_extension=".stp",
        image_data=image_data,
        pixels_x=15,
        pixels_y=15,
        size_nm_x=3.0,
        size_nm_y=4.5,
        metadata={"image_type": "Topo"},
        raw_metadata={},
    )


def _make_large_uncertainty_point(image: LoadedImage) -> AtomPoint:
    return AtomPoint(
        point_id="point-1",
        row_id="row-1",
        image_id=image.image_id,
        source_group_id=image.source_group_id,
        point_index=0,
        x_px=6.7,
        y_px=7.2,
        amplitude=24.0,
        sigma_x_px=1.8,
        sigma_y_px=1.4,
        offset=1.5,
        position_std_x_px=1000.0,
        position_std_y_px=500.0,
        metadata={
            "fit_model": "gaussian",
            "fit_method": "gaussian_fit",
            "roi_x": 2,
            "roi_y": 2,
            "roi_width": 11,
            "roi_height": 11,
            "fit_mask_active": True,
            "position_uncertainty_status": "recomputed",
            "position_uncertainty_original_mask_missing": True,
            "position_uncertainty_settings_source": "session_fallback",
        },
    )


def test_retry_very_large_uncertainty_uses_saved_fit_context_without_moving_point():
    image = _make_gaussian_image()
    point = _make_large_uncertainty_point(image)
    row = AtomRow(
        row_id="row-1",
        source_group_id=image.source_group_id,
        display_name="Row 1",
        points=(point,),
    )

    assert has_very_large_position_uncertainty(point) is True

    updated_rows, summary = retry_very_large_position_uncertainties(
        (row,),
        (image,),
        FitSettingsState(),
    )

    updated = updated_rows[0].points[0]
    assert (updated.x_px, updated.y_px) == (point.x_px, point.y_px)
    assert (updated.sigma_x_px, updated.sigma_y_px) == (
        point.sigma_x_px,
        point.sigma_y_px,
    )
    assert updated.position_std_x_px is not None
    assert updated.position_std_y_px is not None
    assert updated.position_std_x_px <= 11.0
    assert updated.position_std_y_px <= 11.0
    assert updated.metadata["position_uncertainty_status"] == "recomputed"
    assert updated.metadata["position_uncertainty_settings_source"] == "bounded_retry"
    assert updated.metadata["position_uncertainty_retry_status"] in {
        "succeeded",
        "succeeded_at_constraint_boundary",
    }
    assert summary.total_points == 1
    assert summary.detected_points == 1
    assert summary.successfully_retried_points == 1
    assert summary.still_unreliable_points == 0


def test_retry_clears_large_values_when_required_image_is_missing():
    image = _make_gaussian_image()
    point = _make_large_uncertainty_point(image)
    row = AtomRow(
        row_id="row-1",
        source_group_id=image.source_group_id,
        display_name="Row 1",
        points=(point,),
    )

    updated_rows, summary = retry_very_large_position_uncertainties(
        (row,),
        (),
        FitSettingsState(),
    )

    updated = updated_rows[0].points[0]
    assert updated.position_std_x_px is None
    assert updated.position_std_y_px is None
    assert updated.metadata["position_uncertainty_status"] == "unreliable_fit"
    assert updated.metadata["position_uncertainty_retry_status"] == "missing_image"
    assert summary.detected_points == 1
    assert summary.successfully_retried_points == 0
    assert summary.still_unreliable_points == 1


def test_large_uncertainty_detection_requires_value_larger_than_matching_roi():
    image = _make_gaussian_image()
    point = _make_large_uncertainty_point(image)
    boundary_point = AtomPoint.from_dict(
        {
            **point.to_dict(),
            "position_std_x_px": 11.0,
            "position_std_y_px": 11.0,
        }
    )

    assert has_very_large_position_uncertainty(boundary_point) is False
