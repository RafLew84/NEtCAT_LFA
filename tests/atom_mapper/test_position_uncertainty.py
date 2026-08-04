"""Behavior tests for recalculating saved AtomMapper position uncertainties."""

from __future__ import annotations

import numpy as np
import pytest

from AtomMapper.app.controller import AtomMapperController
from AtomMapper.app.fit_settings import CommonFitSettings, FitSettingsState
from AtomMapper.app.models import AtomPoint, AtomRow, LoadedImage
from AtomMapper.app.position_uncertainty import recalculate_position_uncertainties


def _make_gaussian_image() -> LoadedImage:
    y_grid, x_grid = np.mgrid[0:15, 0:15]
    image_data = 1.5 + 24.0 * np.exp(
        -(((y_grid - 7.2) ** 2) / (2.0 * 1.4**2) + ((x_grid - 6.7) ** 2) / (2.0 * 1.8**2))
    )
    return LoadedImage(
        source_path="/tmp/uncertainty.stp",
        display_name="uncertainty.stp",
        file_extension=".stp",
        image_data=image_data,
        pixels_x=15,
        pixels_y=15,
        size_nm_x=3.0,
        size_nm_y=4.5,
        metadata={"image_type": "Topo"},
        raw_metadata={},
    )


def test_recalculate_position_uncertainties_updates_saved_point_without_moving_it():
    image = _make_gaussian_image()
    point = AtomPoint(
        point_id="point-1",
        row_id="row-1",
        image_id=image.image_id,
        source_group_id=image.source_group_id,
        point_index=0,
        x_px=6.7,
        y_px=7.2,
        sigma_x_px=1.8,
        sigma_y_px=1.4,
        metadata={
            "fit_model": "gaussian",
            "fit_method": "gaussian_fit",
            "roi_x": 2,
            "roi_y": 2,
            "roi_width": 11,
            "roi_height": 11,
            "fit_mask_active": False,
        },
    )
    row = AtomRow(
        row_id="row-1",
        source_group_id=image.source_group_id,
        display_name="Row 1",
        points=(point,),
    )

    updated_rows, summary = recalculate_position_uncertainties(
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
    assert updated.position_std_x_px > 0.0
    assert updated.position_std_y_px > 0.0
    assert updated.position_std_x_nm == pytest.approx(updated.position_std_x_px * 0.2)
    assert updated.position_std_y_nm == pytest.approx(updated.position_std_y_px * 0.3)
    assert updated.metadata["position_uncertainty_status"] == "recomputed"
    assert updated.metadata["position_uncertainty_original_mask_missing"] is False
    assert updated.metadata["position_uncertainty_settings_source"] == "session_fallback"
    assert updated.metadata["position_uncertainty_reference"] == "saved_position"
    assert summary.total_points == 1
    assert summary.recomputed_points == 1
    assert summary.recomputed_without_original_mask == 0
    assert summary.failed_points == 0


def test_controller_recalculates_all_saved_position_uncertainties():
    image = _make_gaussian_image()
    controller = AtomMapperController()
    controller.set_loaded_images((image,))
    row = controller.create_row_for_active_source_group(display_name="Row 1")
    controller.add_point_to_row(
        AtomPoint(
            point_id="point-1",
            row_id=row.row_id,
            image_id=image.image_id,
            source_group_id=image.source_group_id,
            point_index=0,
            x_px=6.7,
            y_px=7.2,
            metadata={
                "fit_model": "gaussian",
                "fit_method": "gaussian_fit",
                "roi_x": 2,
                "roi_y": 2,
                "roi_width": 11,
                "roi_height": 11,
                "fit_mask_active": False,
            },
        )
    )

    summary = controller.recalculate_position_uncertainties(FitSettingsState())

    updated = controller.atom_rows[0].points[0]
    assert summary.recomputed_points == 1
    assert updated.position_std_x_px is not None
    assert updated.position_std_y_px is not None
    assert controller.active_row_id == row.row_id


def test_recalculation_flags_missing_mask_and_manual_position_reference():
    image = _make_gaussian_image()
    point = AtomPoint(
        point_id="point-1",
        row_id="row-1",
        image_id=image.image_id,
        source_group_id=image.source_group_id,
        point_index=0,
        x_px=7.1,
        y_px=7.0,
        manual_override=True,
        manual_override_source="drag",
        original_x_px=6.7,
        original_y_px=7.2,
        metadata={
            "fit_model": "gaussian",
            "fit_method": "gaussian_fit",
            "roi_x": 2,
            "roi_y": 2,
            "roi_width": 11,
            "roi_height": 11,
            "fit_mask_active": True,
            "fit_mask_pixel_count": 72,
        },
    )
    row = AtomRow(
        row_id="row-1",
        source_group_id=image.source_group_id,
        display_name="Row 1",
        points=(point,),
    )

    updated_rows, summary = recalculate_position_uncertainties(
        (row,),
        (image,),
        FitSettingsState(),
    )

    updated = updated_rows[0].points[0]
    assert updated.metadata["position_uncertainty_status"] == "recomputed"
    assert updated.metadata["position_uncertainty_original_mask_missing"] is True
    assert updated.metadata["position_uncertainty_settings_source"] == "session_fallback"
    assert updated.metadata["position_uncertainty_reference"] == "original_fit_position"
    assert updated.metadata["position_uncertainty_method"] in {
        "fit_covariance",
        "monte_carlo",
    }
    assert summary.recomputed_points == 1
    assert summary.recomputed_without_original_mask == 1


def test_recalculation_uses_point_fit_settings_snapshot(monkeypatch):
    image = _make_gaussian_image()
    point_settings = FitSettingsState(common=CommonFitSettings(max_nfev=3210))
    point = AtomPoint(
        point_id="point-1",
        row_id="row-1",
        image_id=image.image_id,
        source_group_id=image.source_group_id,
        point_index=0,
        x_px=6.7,
        y_px=7.2,
        metadata={
            "fit_model": "gaussian",
            "fit_settings": point_settings.to_dict(),
            "roi_x": 2,
            "roi_y": 2,
            "roi_width": 11,
            "roi_height": 11,
            "fit_mask_active": False,
        },
    )
    row = AtomRow(
        row_id="row-1",
        source_group_id=image.source_group_id,
        display_name="Row 1",
        points=(point,),
    )
    captured_max_nfev: list[int] = []

    from AtomMapper.app import position_uncertainty as uncertainty_module

    original_fit_local_peak = uncertainty_module.fit_local_peak

    def capture_request(request):
        assert request.fit_settings_state is not None
        captured_max_nfev.append(request.fit_settings_state.common.max_nfev)
        return original_fit_local_peak(request)

    monkeypatch.setattr(uncertainty_module, "fit_local_peak", capture_request)

    updated_rows, summary = recalculate_position_uncertainties(
        (row,),
        (image,),
        FitSettingsState(common=CommonFitSettings(max_nfev=9876)),
    )

    updated = updated_rows[0].points[0]
    assert captured_max_nfev == [3210]
    assert updated.metadata["position_uncertainty_status"] == "recomputed"
    assert updated.metadata["position_uncertainty_settings_source"] == "point_snapshot"
    assert summary.recomputed_points == 1


def test_failed_recalculation_clears_stale_uncertainty_method():
    point = AtomPoint(
        point_id="point-1",
        row_id="row-1",
        image_id="missing-image",
        source_group_id="group-1",
        point_index=0,
        x_px=6.7,
        y_px=7.2,
        position_std_x_px=0.1,
        position_std_y_px=0.2,
        metadata={"position_uncertainty_method": "fit_covariance"},
    )
    row = AtomRow(
        row_id="row-1",
        source_group_id="group-1",
        display_name="Row 1",
        points=(point,),
    )

    updated_rows, summary = recalculate_position_uncertainties(
        (row,),
        (),
        FitSettingsState(),
    )

    updated = updated_rows[0].points[0]
    assert updated.metadata["position_uncertainty_status"] == "missing_image"
    assert "position_uncertainty_method" not in updated.metadata
    assert updated.position_std_x_px is None
    assert updated.position_std_y_px is None
    assert summary.failed_points == 1
