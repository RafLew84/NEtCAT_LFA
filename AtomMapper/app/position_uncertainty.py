"""Recalculate localization uncertainty for points stored in AtomMapper projects."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np

from .fit_models import LocalFitModelType, LocalFitRequest
from .fit_settings import FitSettingsState
from .gaussian_fit import fit_local_peak
from .models import AtomPoint, AtomRow, LoadedImage, ROIState


@dataclass(frozen=True)
class PositionUncertaintyRecalculationSummary:
    """Counts describing one batch recalculation."""

    total_points: int
    recomputed_points: int
    recomputed_without_original_mask: int
    failed_points: int


def recalculate_position_uncertainties(
    rows: Sequence[AtomRow],
    images: Sequence[LoadedImage],
    fit_settings: FitSettingsState,
) -> tuple[tuple[AtomRow, ...], PositionUncertaintyRecalculationSummary]:
    """Return rows whose points carry freshly fitted position uncertainties."""

    images_by_id = {image.image_id: image for image in images}
    updated_rows: list[AtomRow] = []
    total_points = 0
    recomputed_points = 0
    recomputed_without_original_mask = 0
    failed_points = 0

    for row in rows:
        updated_points: list[AtomPoint] = []
        for point in row.points:
            total_points += 1
            updated_point = _recalculate_point_uncertainty(
                point,
                image=images_by_id.get(point.image_id),
                fit_settings=fit_settings,
            )
            status = updated_point.metadata.get("position_uncertainty_status")
            if status == "recomputed":
                recomputed_points += 1
                if updated_point.metadata.get("position_uncertainty_original_mask_missing"):
                    recomputed_without_original_mask += 1
            else:
                failed_points += 1
            updated_points.append(updated_point)
        updated_rows.append(replace(row, points=tuple(updated_points)))

    return (
        tuple(updated_rows),
        PositionUncertaintyRecalculationSummary(
            total_points=total_points,
            recomputed_points=recomputed_points,
            recomputed_without_original_mask=recomputed_without_original_mask,
            failed_points=failed_points,
        ),
    )


def _recalculate_point_uncertainty(
    point: AtomPoint,
    *,
    image: LoadedImage | None,
    fit_settings: FitSettingsState,
) -> AtomPoint:
    metadata = dict(point.metadata)
    if image is None:
        return _with_failure_status(point, metadata, "missing_image")

    try:
        roi = ROIState(
            x=int(metadata["roi_x"]),
            y=int(metadata["roi_y"]),
            width=int(metadata["roi_width"]),
            height=int(metadata["roi_height"]),
        )
        model = LocalFitModelType(metadata.get("fit_model", fit_settings.model.value))
    except (KeyError, TypeError, ValueError):
        return _with_failure_status(point, metadata, "missing_fit_context")

    point_fit_settings, settings_source = _fit_settings_for_point(metadata, fit_settings)
    point_fit_settings = point_fit_settings.with_model(model)

    image_data = np.asarray(image.image_data, dtype=float)
    x0 = max(0, roi.x)
    y0 = max(0, roi.y)
    x1 = min(image_data.shape[1], roi.x + roi.width)
    y1 = min(image_data.shape[0], roi.y + roi.height)
    if x1 <= x0 or y1 <= y0:
        return _with_failure_status(point, metadata, "invalid_roi")

    fit_result = fit_local_peak(
        LocalFitRequest(
            model=model,
            roi_patch=image_data[y0:y1, x0:x1].copy(),
            roi_origin_yx=(y0, x0),
            compute_uncertainty=True,
            fit_mask=None,
            fit_settings_state=point_fit_settings,
        )
    )
    if not fit_result.success or fit_result.center_std_yx is None:
        return _with_failure_status(point, metadata, "fit_uncertainty_unavailable")

    position_std_y_px, position_std_x_px = fit_result.center_std_yx
    calibration = image.physical_calibration
    position_std_x_nm = None
    position_std_y_nm = None
    if calibration is not None:
        position_std_x_nm = position_std_x_px * calibration.pixel_size_nm_x
        position_std_y_nm = position_std_y_px * calibration.pixel_size_nm_y

    missing_original_mask = bool(metadata.get("fit_mask_active"))
    metadata.pop("position_uncertainty_retry_status", None)
    metadata.pop("position_uncertainty_retry_at_bound", None)
    metadata.pop("position_uncertainty_covariance_condition", None)
    metadata["position_uncertainty_status"] = "recomputed"
    metadata["position_uncertainty_original_mask_missing"] = missing_original_mask
    metadata["position_uncertainty_settings_source"] = settings_source
    metadata["position_uncertainty_reference"] = (
        "original_fit_position" if point.manual_override else "saved_position"
    )
    metadata["position_uncertainty_method"] = position_uncertainty_method(fit_result.raw_result)

    return replace(
        point,
        position_std_x_px=position_std_x_px,
        position_std_y_px=position_std_y_px,
        position_std_x_nm=position_std_x_nm,
        position_std_y_nm=position_std_y_nm,
        metadata=metadata,
    )


def _with_failure_status(
    point: AtomPoint,
    metadata: dict[str, object],
    reason: str,
) -> AtomPoint:
    metadata.pop("position_uncertainty_method", None)
    metadata.pop("position_uncertainty_original_mask_missing", None)
    metadata.pop("position_uncertainty_settings_source", None)
    metadata.pop("position_uncertainty_retry_status", None)
    metadata.pop("position_uncertainty_retry_at_bound", None)
    metadata.pop("position_uncertainty_covariance_condition", None)
    metadata["position_uncertainty_status"] = reason
    metadata["position_uncertainty_reference"] = (
        "original_fit_position" if point.manual_override else "saved_position"
    )
    return replace(
        point,
        position_std_x_px=None,
        position_std_y_px=None,
        position_std_x_nm=None,
        position_std_y_nm=None,
        metadata=metadata,
    )


def _fit_settings_for_point(
    metadata: dict[str, object],
    session_fit_settings: FitSettingsState,
) -> tuple[FitSettingsState, str]:
    snapshot = metadata.get("fit_settings")
    if isinstance(snapshot, dict):
        try:
            return FitSettingsState.from_dict(snapshot), "point_snapshot"
        except (TypeError, ValueError):
            pass
    return session_fit_settings.normalized(), "session_fallback"


def position_uncertainty_method(raw_result: object | None) -> str:
    """Describe how a fit result produced its center uncertainty."""

    covariance = getattr(raw_result, "pcov", None)
    if covariance is not None:
        covariance_array = np.asarray(covariance, dtype=float)
        if covariance_array.ndim == 2 and covariance_array.shape[0] >= 3:
            center_diagonal = np.diag(covariance_array)[1:3]
            if np.all(np.isfinite(center_diagonal)) and np.all(center_diagonal >= 0.0):
                return "fit_covariance"
    return "monte_carlo"
