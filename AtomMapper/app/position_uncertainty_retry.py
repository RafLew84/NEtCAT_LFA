"""Constrained retry for implausibly large AtomMapper position uncertainties."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import pi, radians
from typing import Sequence

import numpy as np

from .fit_models import LocalFitModelType, LocalFitRequest, LocalPeakFitResult
from .fit_settings import FitSettingsState, ParameterBounds
from .gaussian_fit import fit_local_peak
from .models import AtomPoint, AtomRow, LoadedImage, ROIState
from .position_uncertainty import position_uncertainty_method

RETRY_CENTER_RADIUS_PX = 2.0
RETRY_MIN_WIDTH_PX = 0.3
RETRY_WIDTH_LOWER_SCALE = 0.5
RETRY_WIDTH_UPPER_SCALE = 1.5
RETRY_MAX_COVARIANCE_CONDITION = 1.0e10


@dataclass(frozen=True)
class PositionUncertaintyRetrySummary:
    """Counts describing one constrained retry batch."""

    total_points: int
    detected_points: int
    successfully_retried_points: int
    still_unreliable_points: int
    constraint_boundary_points: int


@dataclass(frozen=True)
class _RetryFitPlan:
    settings: FitSettingsState
    lower_bounds: tuple[float, ...]
    upper_bounds: tuple[float, ...]


def has_very_large_position_uncertainty(point: AtomPoint) -> bool:
    """Return whether a stored center uncertainty exceeds its matching ROI dimension."""

    try:
        roi_width = float(point.metadata["roi_width"])
        roi_height = float(point.metadata["roi_height"])
    except (KeyError, TypeError, ValueError):
        return False
    if roi_width <= 0.0 or roi_height <= 0.0:
        return False
    return bool(
        (point.position_std_x_px is not None and point.position_std_x_px > roi_width)
        or (point.position_std_y_px is not None and point.position_std_y_px > roi_height)
    )


def retry_very_large_position_uncertainties(
    rows: Sequence[AtomRow],
    images: Sequence[LoadedImage],
    fit_settings: FitSettingsState,
) -> tuple[tuple[AtomRow, ...], PositionUncertaintyRetrySummary]:
    """Retry only oversized uncertainties with bounds derived from the saved fit."""

    images_by_id = {image.image_id: image for image in images}
    updated_rows: list[AtomRow] = []
    total_points = 0
    detected_points = 0
    successfully_retried_points = 0
    still_unreliable_points = 0
    constraint_boundary_points = 0

    for row in rows:
        updated_points: list[AtomPoint] = []
        for point in row.points:
            total_points += 1
            if not has_very_large_position_uncertainty(point):
                updated_points.append(point)
                continue

            detected_points += 1
            updated = _retry_point_uncertainty(
                point,
                image=images_by_id.get(point.image_id),
                session_fit_settings=fit_settings,
            )
            if updated.metadata.get("position_uncertainty_retry_status") in {
                "succeeded",
                "succeeded_at_constraint_boundary",
            }:
                successfully_retried_points += 1
                if updated.metadata.get("position_uncertainty_retry_at_bound"):
                    constraint_boundary_points += 1
            else:
                still_unreliable_points += 1
            updated_points.append(updated)
        updated_rows.append(replace(row, points=tuple(updated_points)))

    return (
        tuple(updated_rows),
        PositionUncertaintyRetrySummary(
            total_points=total_points,
            detected_points=detected_points,
            successfully_retried_points=successfully_retried_points,
            still_unreliable_points=still_unreliable_points,
            constraint_boundary_points=constraint_boundary_points,
        ),
    )


def _retry_point_uncertainty(
    point: AtomPoint,
    *,
    image: LoadedImage | None,
    session_fit_settings: FitSettingsState,
) -> AtomPoint:
    metadata = dict(point.metadata)
    if image is None:
        return _with_unreliable_status(point, metadata, "missing_image")

    try:
        roi = ROIState(
            x=int(metadata["roi_x"]),
            y=int(metadata["roi_y"]),
            width=int(metadata["roi_width"]),
            height=int(metadata["roi_height"]),
        )
        model = LocalFitModelType(metadata.get("fit_model", session_fit_settings.model.value))
    except (KeyError, TypeError, ValueError):
        return _with_unreliable_status(point, metadata, "missing_fit_context")

    image_data = np.asarray(image.image_data, dtype=float)
    x0 = max(0, roi.x)
    y0 = max(0, roi.y)
    x1 = min(image_data.shape[1], roi.x + roi.width)
    y1 = min(image_data.shape[0], roi.y + roi.height)
    if x1 <= x0 or y1 <= y0:
        return _with_unreliable_status(point, metadata, "invalid_roi")
    patch = image_data[y0:y1, x0:x1].copy()

    try:
        plan = _build_retry_fit_plan(
            point,
            model=model,
            patch=patch,
            roi_origin_yx=(y0, x0),
            session_fit_settings=session_fit_settings,
        )
    except (TypeError, ValueError):
        return _with_unreliable_status(point, metadata, "invalid_saved_fit_context")

    fit_result = fit_local_peak(
        LocalFitRequest(
            model=model,
            roi_patch=patch,
            roi_origin_yx=(y0, x0),
            compute_uncertainty=True,
            fit_mask=None,
            fit_settings_state=plan.settings,
        )
    )
    covariance_condition = _covariance_condition(fit_result.raw_result)
    if not _is_acceptable_retry_result(
        fit_result,
        patch_shape=patch.shape,
        covariance_condition=covariance_condition,
    ):
        return _with_unreliable_status(
            point,
            metadata,
            "bounded_fit_unreliable",
            covariance_condition=covariance_condition,
        )

    assert fit_result.center_std_yx is not None
    position_std_y_px, position_std_x_px = fit_result.center_std_yx
    calibration = image.physical_calibration
    position_std_x_nm = None
    position_std_y_nm = None
    if calibration is not None:
        position_std_x_nm = position_std_x_px * calibration.pixel_size_nm_x
        position_std_y_nm = position_std_y_px * calibration.pixel_size_nm_y

    at_bound = _fit_reached_constraint_boundary(fit_result, plan)
    metadata["position_uncertainty_status"] = "recomputed"
    metadata["position_uncertainty_settings_source"] = "bounded_retry"
    metadata["position_uncertainty_retry_status"] = (
        "succeeded_at_constraint_boundary" if at_bound else "succeeded"
    )
    metadata["position_uncertainty_retry_at_bound"] = at_bound
    metadata["position_uncertainty_covariance_condition"] = covariance_condition
    metadata["position_uncertainty_original_mask_missing"] = bool(metadata.get("fit_mask_active"))
    metadata["position_uncertainty_reference"] = (
        "original_fit_position" if point.manual_override else "saved_position"
    )
    metadata["position_uncertainty_method"] = position_uncertainty_method(fit_result.raw_result)
    return replace(
        point,
        position_std_x_px=float(position_std_x_px),
        position_std_y_px=float(position_std_y_px),
        position_std_x_nm=position_std_x_nm,
        position_std_y_nm=position_std_y_nm,
        metadata=metadata,
    )


def _build_retry_fit_plan(
    point: AtomPoint,
    *,
    model: LocalFitModelType,
    patch: np.ndarray,
    roi_origin_yx: tuple[int, int],
    session_fit_settings: FitSettingsState,
) -> _RetryFitPlan:
    height, width = patch.shape
    origin_y, origin_x = roi_origin_yx
    center_y = float(np.clip(point.fit_y_px - origin_y, 0.0, height - 1.0))
    center_x = float(np.clip(point.fit_x_px - origin_x, 0.0, width - 1.0))
    center_y_bounds = _center_bounds(center_y, height)
    center_x_bounds = _center_bounds(center_x, width)

    patch_range = max(float(np.ptp(patch)), 1.0e-9)
    amplitude = _amplitude_guess(point.amplitude, patch_range)
    amplitude_bounds = _amplitude_bounds(amplitude, patch_range)
    offset_bounds = (
        float(np.min(patch) - patch_range),
        float(np.max(patch) + patch_range),
    )
    offset = float(
        np.clip(
            point.offset if point.offset is not None else np.min(patch),
            offset_bounds[0],
            offset_bounds[1],
        )
    )
    theta = radians(float(point.theta_deg or 0.0))
    theta_bounds = (theta - (pi / 4.0), theta + (pi / 4.0))

    base = _saved_or_session_fit_settings(point, session_fit_settings).with_model(model)
    common = replace(
        base.common,
        compute_uncertainty=True,
        use_custom_initial_guess=True,
        use_custom_bounds=True,
    )
    sigma_y = _width_guess(point.sigma_y_px, height)
    sigma_x = _width_guess(point.sigma_x_px, width)
    sigma_y_bounds = _width_bounds(sigma_y, height)
    sigma_x_bounds = _width_bounds(sigma_x, width)

    if model is LocalFitModelType.GAUSSIAN:
        active = replace(
            base.gaussian,
            amplitude_init=amplitude,
            center_y_init=center_y,
            center_x_init=center_x,
            sigma_y_init=sigma_y,
            sigma_x_init=sigma_x,
            theta_init_rad=theta,
            offset_init=offset,
            amplitude_bounds=ParameterBounds(*amplitude_bounds),
            center_y_bounds=ParameterBounds(*center_y_bounds),
            center_x_bounds=ParameterBounds(*center_x_bounds),
            sigma_y_bounds=ParameterBounds(*sigma_y_bounds),
            sigma_x_bounds=ParameterBounds(*sigma_x_bounds),
            theta_bounds_rad=ParameterBounds(*theta_bounds),
            offset_bounds=ParameterBounds(*offset_bounds),
        )
        return _RetryFitPlan(
            settings=replace(base, common=common, gaussian=active),
            lower_bounds=(
                amplitude_bounds[0],
                center_y_bounds[0],
                center_x_bounds[0],
                sigma_y_bounds[0],
                sigma_x_bounds[0],
                theta_bounds[0],
                offset_bounds[0],
            ),
            upper_bounds=(
                amplitude_bounds[1],
                center_y_bounds[1],
                center_x_bounds[1],
                sigma_y_bounds[1],
                sigma_x_bounds[1],
                theta_bounds[1],
                offset_bounds[1],
            ),
        )

    if model is LocalFitModelType.LORENTZIAN:
        active = replace(
            base.lorentzian,
            amplitude_init=amplitude,
            center_y_init=center_y,
            center_x_init=center_x,
            gamma_y_init=sigma_y,
            gamma_x_init=sigma_x,
            theta_init_rad=theta,
            offset_init=offset,
            amplitude_bounds=ParameterBounds(*amplitude_bounds),
            center_y_bounds=ParameterBounds(*center_y_bounds),
            center_x_bounds=ParameterBounds(*center_x_bounds),
            gamma_y_bounds=ParameterBounds(*sigma_y_bounds),
            gamma_x_bounds=ParameterBounds(*sigma_x_bounds),
            theta_bounds_rad=ParameterBounds(*theta_bounds),
            offset_bounds=ParameterBounds(*offset_bounds),
        )
        return _RetryFitPlan(
            settings=replace(base, common=common, lorentzian=active),
            lower_bounds=(
                amplitude_bounds[0],
                center_y_bounds[0],
                center_x_bounds[0],
                sigma_y_bounds[0],
                sigma_x_bounds[0],
                theta_bounds[0],
                offset_bounds[0],
            ),
            upper_bounds=(
                amplitude_bounds[1],
                center_y_bounds[1],
                center_x_bounds[1],
                sigma_y_bounds[1],
                sigma_x_bounds[1],
                theta_bounds[1],
                offset_bounds[1],
            ),
        )

    shape_parameters = point.metadata.get("fit_shape_parameters")
    shape_parameters = shape_parameters if isinstance(shape_parameters, dict) else {}
    gamma_y = _width_guess(shape_parameters.get("gamma_y"), height)
    gamma_x = _width_guess(shape_parameters.get("gamma_x"), width)
    gamma_y_bounds = _width_bounds(gamma_y, height)
    gamma_x_bounds = _width_bounds(gamma_x, width)
    active = replace(
        base.voigt,
        amplitude_init=amplitude,
        center_y_init=center_y,
        center_x_init=center_x,
        sigma_y_init=sigma_y,
        sigma_x_init=sigma_x,
        gamma_y_init=gamma_y,
        gamma_x_init=gamma_x,
        theta_init_rad=theta,
        offset_init=offset,
        amplitude_bounds=ParameterBounds(*amplitude_bounds),
        center_y_bounds=ParameterBounds(*center_y_bounds),
        center_x_bounds=ParameterBounds(*center_x_bounds),
        sigma_y_bounds=ParameterBounds(*sigma_y_bounds),
        sigma_x_bounds=ParameterBounds(*sigma_x_bounds),
        gamma_y_bounds=ParameterBounds(*gamma_y_bounds),
        gamma_x_bounds=ParameterBounds(*gamma_x_bounds),
        theta_bounds_rad=ParameterBounds(*theta_bounds),
        offset_bounds=ParameterBounds(*offset_bounds),
    )
    return _RetryFitPlan(
        settings=replace(base, common=common, voigt=active),
        lower_bounds=(
            amplitude_bounds[0],
            center_y_bounds[0],
            center_x_bounds[0],
            sigma_y_bounds[0],
            sigma_x_bounds[0],
            gamma_y_bounds[0],
            gamma_x_bounds[0],
            theta_bounds[0],
            offset_bounds[0],
        ),
        upper_bounds=(
            amplitude_bounds[1],
            center_y_bounds[1],
            center_x_bounds[1],
            sigma_y_bounds[1],
            sigma_x_bounds[1],
            gamma_y_bounds[1],
            gamma_x_bounds[1],
            theta_bounds[1],
            offset_bounds[1],
        ),
    )


def _saved_or_session_fit_settings(
    point: AtomPoint,
    session_fit_settings: FitSettingsState,
) -> FitSettingsState:
    snapshot = point.metadata.get("fit_settings")
    if isinstance(snapshot, dict):
        try:
            return FitSettingsState.from_dict(snapshot)
        except (AttributeError, TypeError, ValueError):
            pass
    return session_fit_settings.normalized()


def _center_bounds(center: float, size: int) -> tuple[float, float]:
    return (
        max(0.0, center - RETRY_CENTER_RADIUS_PX),
        min(float(size - 1), center + RETRY_CENTER_RADIUS_PX),
    )


def _width_guess(value: object, limit: int) -> float:
    try:
        number = abs(float(value))
    except (TypeError, ValueError):
        number = max(1.0, float(limit) / 4.0)
    if not np.isfinite(number) or number <= 0.0:
        number = max(1.0, float(limit) / 4.0)
    return float(np.clip(number, RETRY_MIN_WIDTH_PX, float(limit)))


def _width_bounds(width: float, limit: int) -> tuple[float, float]:
    return (
        max(RETRY_MIN_WIDTH_PX, width * RETRY_WIDTH_LOWER_SCALE),
        min(float(limit), width * RETRY_WIDTH_UPPER_SCALE),
    )


def _amplitude_guess(value: object, patch_range: float) -> float:
    try:
        amplitude = float(value)
    except (TypeError, ValueError):
        amplitude = patch_range
    if not np.isfinite(amplitude) or abs(amplitude) < 1.0e-12:
        amplitude = patch_range
    return amplitude


def _amplitude_bounds(amplitude: float, patch_range: float) -> tuple[float, float]:
    extent = max(patch_range * 5.0, abs(amplitude) * 2.0, 1.0e-9)
    if amplitude >= 0.0:
        return 0.0, extent
    return -extent, 0.0


def _covariance_condition(raw_result: object | None) -> float | None:
    covariance = getattr(raw_result, "pcov", None)
    if covariance is None:
        return None
    covariance_array = np.asarray(covariance, dtype=float)
    try:
        condition = float(np.linalg.cond(covariance_array))
    except np.linalg.LinAlgError:
        return float("inf")
    return condition


def _is_acceptable_retry_result(
    result: LocalPeakFitResult,
    *,
    patch_shape: tuple[int, ...],
    covariance_condition: float | None,
) -> bool:
    if (
        not result.success
        or result.center_patch_yx is None
        or result.center_std_yx is None
        or len(patch_shape) != 2
    ):
        return False
    height, width = patch_shape
    center_y, center_x = result.center_patch_yx
    std_y, std_x = result.center_std_yx
    values = (center_y, center_x, std_y, std_x)
    if not all(np.isfinite(value) for value in values):
        return False
    if not (0.0 <= center_y <= height - 1.0 and 0.0 <= center_x <= width - 1.0):
        return False
    if not (0.0 <= std_y <= height and 0.0 <= std_x <= width):
        return False
    if result.width_y is None or result.width_x is None:
        return False
    if not (0.0 < result.width_y <= height and 0.0 < result.width_x <= width):
        return False
    if covariance_condition is not None and (
        not np.isfinite(covariance_condition)
        or covariance_condition > RETRY_MAX_COVARIANCE_CONDITION
    ):
        return False
    return True


def _fit_reached_constraint_boundary(
    result: LocalPeakFitResult,
    plan: _RetryFitPlan,
) -> bool:
    parameters = getattr(result.raw_result, "popt", None)
    if parameters is None or len(parameters) != len(plan.lower_bounds):
        return False
    for value, lower, upper in zip(
        np.asarray(parameters, dtype=float),
        plan.lower_bounds,
        plan.upper_bounds,
    ):
        if np.isfinite(lower) and np.isclose(value, lower, rtol=1.0e-5, atol=1.0e-6):
            return True
        if np.isfinite(upper) and np.isclose(value, upper, rtol=1.0e-5, atol=1.0e-6):
            return True
    return False


def _with_unreliable_status(
    point: AtomPoint,
    metadata: dict[str, object],
    reason: str,
    *,
    covariance_condition: float | None = None,
) -> AtomPoint:
    metadata.pop("position_uncertainty_method", None)
    metadata["position_uncertainty_status"] = "unreliable_fit"
    metadata["position_uncertainty_settings_source"] = "bounded_retry"
    metadata["position_uncertainty_retry_status"] = reason
    metadata["position_uncertainty_retry_at_bound"] = False
    metadata["position_uncertainty_covariance_condition"] = covariance_condition
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
