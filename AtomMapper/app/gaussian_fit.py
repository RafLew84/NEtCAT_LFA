"""AtomMapper-specific adapter for local peak fitting on ROI patches."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from lfa.analysis.peak_fitting import (
    PeakRefinementResult,
    fit_2d_gaussian_on_patch,
    fit_2d_lorentzian_on_patch,
    fit_2d_voigt_on_patch,
    voigt_profile,
)
from .fit_settings import (
    CommonFitSettings,
    FitSettingsState,
    GaussianFitSettings,
    LorentzianFitSettings,
    ParameterBounds,
    VoigtFitSettings,
)
from .fit_models import LocalFitModelType, LocalFitRequest, LocalPeakFitResult


def _build_gaussian_model_patch(
    patch_shape: tuple[int, int],
    amplitude: float,
    center_patch_yx: Tuple[float, float],
    sigma_y: float,
    sigma_x: float,
    theta_rad: float,
    offset: float,
) -> np.ndarray:
    """Evaluate the fitted Gaussian model on the ROI patch grid."""

    rows, cols = patch_shape
    y_coords = np.arange(rows, dtype=float)
    x_coords = np.arange(cols, dtype=float)
    x_grid, y_grid = np.meshgrid(x_coords, y_coords)

    y0, x0 = center_patch_yx
    a = (np.cos(theta_rad) ** 2) / (2.0 * sigma_y**2) + (np.sin(theta_rad) ** 2) / (2.0 * sigma_x**2)
    b = -(np.sin(2.0 * theta_rad)) / (4.0 * sigma_y**2) + (np.sin(2.0 * theta_rad)) / (4.0 * sigma_x**2)
    c = (np.sin(theta_rad) ** 2) / (2.0 * sigma_y**2) + (np.cos(theta_rad) ** 2) / (2.0 * sigma_x**2)
    exponent = -(a * ((y_grid - y0) ** 2) + 2.0 * b * (y_grid - y0) * (x_grid - x0) + c * ((x_grid - x0) ** 2))
    return offset + amplitude * np.exp(exponent)


def _build_lorentzian_model_patch(
    patch_shape: tuple[int, int],
    amplitude: float,
    center_patch_yx: Tuple[float, float],
    gamma_y: float,
    gamma_x: float,
    theta_rad: float,
    offset: float,
) -> np.ndarray:
    """Evaluate the fitted Lorentzian model on the ROI patch grid."""

    rows, cols = patch_shape
    y_coords = np.arange(rows, dtype=float)
    x_coords = np.arange(cols, dtype=float)
    x_grid, y_grid = np.meshgrid(x_coords, y_coords)

    y0, x0 = center_patch_yx
    dy = y_grid - y0
    dx = x_grid - x0
    gamma_y = max(abs(float(gamma_y)), 1e-12)
    gamma_x = max(abs(float(gamma_x)), 1e-12)
    y_rot = np.cos(theta_rad) * dy + np.sin(theta_rad) * dx
    x_rot = -np.sin(theta_rad) * dy + np.cos(theta_rad) * dx
    q = (y_rot / gamma_y) ** 2 + (x_rot / gamma_x) ** 2
    return offset + amplitude / (1.0 + q)


def _build_voigt_model_patch(
    patch_shape: tuple[int, int],
    amplitude: float,
    center_patch_yx: Tuple[float, float],
    sigma_y: float,
    sigma_x: float,
    gamma_y: float,
    gamma_x: float,
    theta_rad: float,
    offset: float,
) -> np.ndarray:
    """Evaluate the fitted Voigt model on the ROI patch grid."""

    if voigt_profile is None:  # pragma: no cover
        raise RuntimeError("SciPy voigt_profile is not available.")

    rows, cols = patch_shape
    y_coords = np.arange(rows, dtype=float)
    x_coords = np.arange(cols, dtype=float)
    x_grid, y_grid = np.meshgrid(x_coords, y_coords)

    y0, x0 = center_patch_yx
    dy = y_grid - y0
    dx = x_grid - x0
    sigma_y = max(abs(float(sigma_y)), 1e-12)
    sigma_x = max(abs(float(sigma_x)), 1e-12)
    gamma_y = max(abs(float(gamma_y)), 1e-12)
    gamma_x = max(abs(float(gamma_x)), 1e-12)
    y_rot = np.cos(theta_rad) * dy + np.sin(theta_rad) * dx
    x_rot = -np.sin(theta_rad) * dy + np.cos(theta_rad) * dx
    profile_y = voigt_profile(y_rot, sigma_y, gamma_y)
    profile_x = voigt_profile(x_rot, sigma_x, gamma_x)
    center_y_norm = float(voigt_profile(np.array([0.0]), sigma_y, gamma_y)[0])
    center_x_norm = float(voigt_profile(np.array([0.0]), sigma_x, gamma_x)[0])
    center_y_norm = center_y_norm if np.isfinite(center_y_norm) and center_y_norm > 0.0 else 1.0
    center_x_norm = center_x_norm if np.isfinite(center_x_norm) and center_x_norm > 0.0 else 1.0
    return offset + amplitude * (profile_y / center_y_norm) * (profile_x / center_x_norm)


def _unsupported_model_result(
    request: LocalFitRequest,
    message: str,
    *,
    method: str = "model_unavailable",
) -> LocalPeakFitResult:
    """Build a normalized failure result for unsupported or invalid requests."""

    patch = request.normalized_patch()
    mask = request.normalized_mask()
    return LocalPeakFitResult(
        model=request.model,
        center_patch_yx=None,
        center_image_yx=None,
        center_std_yx=None,
        amplitude=None,
        width_y=None,
        width_x=None,
        theta_rad=None,
        offset=None,
        method=method,
        success=False,
        error_message=message,
        model_patch=None,
        roi_patch=patch.copy(),
        raw_result=None,
        fit_mask=None if mask is None else mask.copy(),
    )


def _resolve_fit_settings_state(request: LocalFitRequest) -> FitSettingsState:
    if request.fit_settings_state is None:
        return FitSettingsState()
    return request.fit_settings_state.normalized()


def _normalize_initial_guess_mask(
    patch: np.ndarray,
    mask: np.ndarray | None,
) -> np.ndarray | None:
    """Return a usable boolean mask for initial-guess statistics."""

    if mask is None:
        return None
    normalized_mask = np.asarray(mask, dtype=bool)
    if normalized_mask.shape != patch.shape or not normalized_mask.any():
        return None
    return normalized_mask


def _build_initial_guess_statistics(
    patch: np.ndarray,
    mask: np.ndarray | None,
) -> tuple[float, int, int, int, int, float]:
    """Derive initial-guess stats from either the full ROI or the masked subregion."""

    patch_array = np.asarray(patch, dtype=float)
    guess_mask = _normalize_initial_guess_mask(patch_array, mask)

    if guess_mask is None:
        finite_patch = patch_array[np.isfinite(patch_array)]
        selected_values = finite_patch if finite_patch.size > 0 else np.array([0.0], dtype=float)
        peak_source = np.where(np.isfinite(patch_array), patch_array, -np.inf)
        peak_y, peak_x = np.unravel_index(int(np.argmax(peak_source)), patch_array.shape)
        span_y, span_x = patch_array.shape
    else:
        selected_values = patch_array[guess_mask]
        finite_selected = selected_values[np.isfinite(selected_values)]
        selected_values = finite_selected if finite_selected.size > 0 else np.array([0.0], dtype=float)
        masked_peak_source = np.where(guess_mask & np.isfinite(patch_array), patch_array, -np.inf)
        peak_y, peak_x = np.unravel_index(int(np.argmax(masked_peak_source)), patch_array.shape)
        y_coords, x_coords = np.nonzero(guess_mask)
        span_y = int(y_coords.max() - y_coords.min() + 1)
        span_x = int(x_coords.max() - x_coords.min() + 1)

    peak_to_peak = float(np.ptp(selected_values))
    offset_guess = float(np.min(selected_values))
    return peak_to_peak, int(peak_y), int(peak_x), int(span_y), int(span_x), offset_guess


def _merge_gaussian_initial_params(
    default_p0: list[float],
    common_settings: CommonFitSettings,
    gaussian_settings: GaussianFitSettings,
) -> list[float]:
    if not common_settings.use_custom_initial_guess:
        return list(default_p0)

    p0 = list(default_p0)
    overrides = (
        (0, gaussian_settings.amplitude_init),
        (1, gaussian_settings.center_y_init),
        (2, gaussian_settings.center_x_init),
        (3, gaussian_settings.sigma_y_init),
        (4, gaussian_settings.sigma_x_init),
        (5, gaussian_settings.theta_init_rad),
        (6, gaussian_settings.offset_init),
    )
    for index, value in overrides:
        if value is not None:
            p0[index] = float(value)
    return p0


def _apply_bounds_to_parameter(
    bounds: ParameterBounds,
    lower: list[float],
    upper: list[float],
    index: int,
) -> None:
    normalized = bounds.normalized()
    if normalized.lower is not None:
        lower[index] = float(normalized.lower)
    if normalized.upper is not None:
        upper[index] = float(normalized.upper)


def _build_gaussian_parameter_bounds(
    common_settings: CommonFitSettings,
    gaussian_settings: GaussianFitSettings,
) -> tuple[list[float], list[float]] | None:
    if not common_settings.use_custom_bounds:
        return None

    lower = [float(-np.inf)] * 7
    upper = [float(np.inf)] * 7
    bounds_map = (
        (0, gaussian_settings.amplitude_bounds),
        (1, gaussian_settings.center_y_bounds),
        (2, gaussian_settings.center_x_bounds),
        (3, gaussian_settings.sigma_y_bounds),
        (4, gaussian_settings.sigma_x_bounds),
        (5, gaussian_settings.theta_bounds_rad),
        (6, gaussian_settings.offset_bounds),
    )
    any_active = False
    for index, bounds in bounds_map:
        if bounds.is_active:
            any_active = True
        _apply_bounds_to_parameter(bounds, lower, upper, index)
    return (lower, upper) if any_active else None


def _merge_lorentzian_initial_params(
    default_p0: list[float],
    common_settings: CommonFitSettings,
    lorentzian_settings: LorentzianFitSettings,
) -> list[float]:
    if not common_settings.use_custom_initial_guess:
        return list(default_p0)

    p0 = list(default_p0)
    overrides = (
        (0, lorentzian_settings.amplitude_init),
        (1, lorentzian_settings.center_y_init),
        (2, lorentzian_settings.center_x_init),
        (3, lorentzian_settings.gamma_y_init),
        (4, lorentzian_settings.gamma_x_init),
        (5, lorentzian_settings.theta_init_rad),
        (6, lorentzian_settings.offset_init),
    )
    for index, value in overrides:
        if value is not None:
            p0[index] = float(value)
    return p0


def _build_lorentzian_parameter_bounds(
    common_settings: CommonFitSettings,
    lorentzian_settings: LorentzianFitSettings,
) -> tuple[list[float], list[float]] | None:
    if not common_settings.use_custom_bounds:
        return None

    lower = [float(-np.inf)] * 7
    upper = [float(np.inf)] * 7
    bounds_map = (
        (0, lorentzian_settings.amplitude_bounds),
        (1, lorentzian_settings.center_y_bounds),
        (2, lorentzian_settings.center_x_bounds),
        (3, lorentzian_settings.gamma_y_bounds),
        (4, lorentzian_settings.gamma_x_bounds),
        (5, lorentzian_settings.theta_bounds_rad),
        (6, lorentzian_settings.offset_bounds),
    )
    any_active = False
    for index, bounds in bounds_map:
        if bounds.is_active:
            any_active = True
        _apply_bounds_to_parameter(bounds, lower, upper, index)
    return (lower, upper) if any_active else None


def _merge_voigt_initial_params(
    default_p0: list[float],
    common_settings: CommonFitSettings,
    voigt_settings: VoigtFitSettings,
) -> list[float]:
    if not common_settings.use_custom_initial_guess:
        return list(default_p0)

    p0 = list(default_p0)
    overrides = (
        (0, voigt_settings.amplitude_init),
        (1, voigt_settings.center_y_init),
        (2, voigt_settings.center_x_init),
        (3, voigt_settings.sigma_y_init),
        (4, voigt_settings.sigma_x_init),
        (5, voigt_settings.gamma_y_init),
        (6, voigt_settings.gamma_x_init),
        (7, voigt_settings.theta_init_rad),
        (8, voigt_settings.offset_init),
    )
    for index, value in overrides:
        if value is not None:
            p0[index] = float(value)
    return p0


def _build_voigt_parameter_bounds(
    common_settings: CommonFitSettings,
    voigt_settings: VoigtFitSettings,
) -> tuple[list[float], list[float]] | None:
    if not common_settings.use_custom_bounds:
        return None

    lower = [float(-np.inf)] * 9
    upper = [float(np.inf)] * 9
    bounds_map = (
        (0, voigt_settings.amplitude_bounds),
        (1, voigt_settings.center_y_bounds),
        (2, voigt_settings.center_x_bounds),
        (3, voigt_settings.sigma_y_bounds),
        (4, voigt_settings.sigma_x_bounds),
        (5, voigt_settings.gamma_y_bounds),
        (6, voigt_settings.gamma_x_bounds),
        (7, voigt_settings.theta_bounds_rad),
        (8, voigt_settings.offset_bounds),
    )
    any_active = False
    for index, bounds in bounds_map:
        if bounds.is_active:
            any_active = True
        _apply_bounds_to_parameter(bounds, lower, upper, index)
    return (lower, upper) if any_active else None


def _fit_gaussian_request(request: LocalFitRequest) -> LocalPeakFitResult:
    """Execute the current Gaussian backend through the common request/result API."""

    patch = request.normalized_patch()
    mask = request.normalized_mask()
    fit_settings_state = _resolve_fit_settings_state(request)
    common_settings = fit_settings_state.common
    gaussian_settings = fit_settings_state.gaussian
    if patch.ndim != 2:
        return LocalPeakFitResult(
            model=LocalFitModelType.GAUSSIAN,
            center_patch_yx=None,
            center_image_yx=None,
            center_std_yx=None,
            amplitude=None,
            width_y=None,
            width_x=None,
            theta_rad=None,
            offset=None,
            method="gaussian_fit",
            success=False,
            error_message=f"Expected a 2D ROI patch, got shape {patch.shape!r}.",
            model_patch=None,
            roi_patch=patch.copy(),
            raw_result=None,
            fit_mask=None if mask is None else mask.copy(),
        )

    peak_to_peak, peak_y, peak_x, span_y, span_x, offset_guess = _build_initial_guess_statistics(
        patch,
        mask,
    )
    sigma_y_guess = max(float(span_y) / 4.0, 1.0)
    sigma_x_guess = max(float(span_x) / 4.0, 1.0)
    default_p0 = [
        peak_to_peak,
        float(peak_y),
        float(peak_x),
        sigma_y_guess,
        sigma_x_guess,
        0.0,
        offset_guess,
    ]

    fit_result = fit_2d_gaussian_on_patch(
        patch,
        roi_origin_yx=request.roi_origin_yx,
        fit_mask=mask,
        initial_params=_merge_gaussian_initial_params(
            default_p0=default_p0,
            common_settings=common_settings,
            gaussian_settings=gaussian_settings,
        ),
        parameter_bounds=_build_gaussian_parameter_bounds(common_settings, gaussian_settings),
        max_nfev=common_settings.max_nfev,
        compute_uncertainty=bool(request.compute_uncertainty),
    )
    if fit_result is None:
        return LocalPeakFitResult(
            model=LocalFitModelType.GAUSSIAN,
            center_patch_yx=None,
            center_image_yx=None,
            center_std_yx=None,
            amplitude=None,
            width_y=None,
            width_x=None,
            theta_rad=None,
            offset=None,
            method="gaussian_fit",
            success=False,
            error_message="Gaussian fit could not be computed for the provided ROI patch.",
            model_patch=None,
            roi_patch=patch.copy(),
            raw_result=None,
            fit_mask=None if mask is None else mask.copy(),
        )

    center_image_yx = (float(fit_result.center[0]), float(fit_result.center[1]))
    center_patch_yx = (
        float(center_image_yx[0] - request.roi_origin_yx[0]),
        float(center_image_yx[1] - request.roi_origin_yx[1]),
    )
    sigma_yx = None
    if fit_result.center_std is not None:
        sigma_yx = (
            float(fit_result.center_std[0]),
            float(fit_result.center_std[1]),
        )

    amplitude = sigma_y = sigma_x = theta_rad = offset = None
    model_patch = None
    if fit_result.popt is not None and len(fit_result.popt) >= 7:
        amplitude = float(fit_result.popt[0])
        sigma_y = abs(float(fit_result.popt[3]))
        sigma_x = abs(float(fit_result.popt[4]))
        theta_rad = float(fit_result.popt[5])
        offset = float(fit_result.popt[6])
        if sigma_y != 0.0 and sigma_x != 0.0:
            model_patch = _build_gaussian_model_patch(
                patch.shape,
                amplitude,
                center_patch_yx,
                sigma_y,
                sigma_x,
                theta_rad,
                offset,
            )

    error_message = None
    if not fit_result.success:
        error_message = (
            "Gaussian fit did not converge; fallback estimate returned by "
            f"{fit_result.method}."
        )

    return LocalPeakFitResult(
        model=LocalFitModelType.GAUSSIAN,
        center_patch_yx=center_patch_yx,
        center_image_yx=center_image_yx,
        center_std_yx=sigma_yx,
        amplitude=amplitude,
        width_y=sigma_y,
        width_x=sigma_x,
        theta_rad=theta_rad,
        offset=offset,
        method=fit_result.method,
        success=bool(fit_result.success),
        error_message=error_message,
        model_patch=model_patch,
        roi_patch=fit_result.roi_patch.copy(),
        raw_result=fit_result,
        fit_mask=None if mask is None else mask.copy(),
        shape_parameters={},
    )


def _fit_lorentzian_request(request: LocalFitRequest) -> LocalPeakFitResult:
    """Execute the current Lorentzian backend through the common request/result API."""

    patch = request.normalized_patch()
    mask = request.normalized_mask()
    fit_settings_state = _resolve_fit_settings_state(request)
    common_settings = fit_settings_state.common
    lorentzian_settings = fit_settings_state.lorentzian
    if patch.ndim != 2:
        return LocalPeakFitResult(
            model=LocalFitModelType.LORENTZIAN,
            center_patch_yx=None,
            center_image_yx=None,
            center_std_yx=None,
            amplitude=None,
            width_y=None,
            width_x=None,
            theta_rad=None,
            offset=None,
            method="lorentzian_fit",
            success=False,
            error_message=f"Expected a 2D ROI patch, got shape {patch.shape!r}.",
            model_patch=None,
            roi_patch=patch.copy(),
            raw_result=None,
            fit_mask=None if mask is None else mask.copy(),
        )

    peak_to_peak, peak_y, peak_x, span_y, span_x, offset_guess = _build_initial_guess_statistics(
        patch,
        mask,
    )
    gamma_y_guess = max(float(span_y) / 4.0, 1.0)
    gamma_x_guess = max(float(span_x) / 4.0, 1.0)
    default_p0 = [
        peak_to_peak,
        float(peak_y),
        float(peak_x),
        gamma_y_guess,
        gamma_x_guess,
        0.0,
        offset_guess,
    ]

    fit_result = fit_2d_lorentzian_on_patch(
        patch,
        roi_origin_yx=request.roi_origin_yx,
        fit_mask=mask,
        initial_params=_merge_lorentzian_initial_params(
            default_p0=default_p0,
            common_settings=common_settings,
            lorentzian_settings=lorentzian_settings,
        ),
        parameter_bounds=_build_lorentzian_parameter_bounds(common_settings, lorentzian_settings),
        max_nfev=common_settings.max_nfev,
        compute_uncertainty=bool(request.compute_uncertainty),
    )
    if fit_result is None:
        return LocalPeakFitResult(
            model=LocalFitModelType.LORENTZIAN,
            center_patch_yx=None,
            center_image_yx=None,
            center_std_yx=None,
            amplitude=None,
            width_y=None,
            width_x=None,
            theta_rad=None,
            offset=None,
            method="lorentzian_fit",
            success=False,
            error_message="Lorentzian fit could not be computed for the provided ROI patch.",
            model_patch=None,
            roi_patch=patch.copy(),
            raw_result=None,
            fit_mask=None if mask is None else mask.copy(),
        )

    center_image_yx = (float(fit_result.center[0]), float(fit_result.center[1]))
    center_patch_yx = (
        float(center_image_yx[0] - request.roi_origin_yx[0]),
        float(center_image_yx[1] - request.roi_origin_yx[1]),
    )
    sigma_yx = None
    if fit_result.center_std is not None:
        sigma_yx = (
            float(fit_result.center_std[0]),
            float(fit_result.center_std[1]),
        )

    amplitude = gamma_y = gamma_x = theta_rad = offset = None
    model_patch = None
    if fit_result.popt is not None and len(fit_result.popt) >= 7:
        amplitude = float(fit_result.popt[0])
        gamma_y = abs(float(fit_result.popt[3]))
        gamma_x = abs(float(fit_result.popt[4]))
        theta_rad = float(fit_result.popt[5])
        offset = float(fit_result.popt[6])
        if gamma_y != 0.0 and gamma_x != 0.0:
            model_patch = _build_lorentzian_model_patch(
                patch.shape,
                amplitude,
                center_patch_yx,
                gamma_y,
                gamma_x,
                theta_rad,
                offset,
            )

    error_message = None
    if not fit_result.success:
        error_message = (
            "Lorentzian fit did not converge; fallback estimate returned by "
            f"{fit_result.method}."
        )

    return LocalPeakFitResult(
        model=LocalFitModelType.LORENTZIAN,
        center_patch_yx=center_patch_yx,
        center_image_yx=center_image_yx,
        center_std_yx=sigma_yx,
        amplitude=amplitude,
        width_y=gamma_y,
        width_x=gamma_x,
        theta_rad=theta_rad,
        offset=offset,
        method=fit_result.method,
        success=bool(fit_result.success),
        error_message=error_message,
        model_patch=model_patch,
        roi_patch=fit_result.roi_patch.copy(),
        raw_result=fit_result,
        fit_mask=None if mask is None else mask.copy(),
        shape_parameters={},
    )


def _fit_voigt_request(request: LocalFitRequest) -> LocalPeakFitResult:
    """Execute the current Voigt backend through the common request/result API."""

    patch = request.normalized_patch()
    mask = request.normalized_mask()
    fit_settings_state = _resolve_fit_settings_state(request)
    common_settings = fit_settings_state.common
    voigt_settings = fit_settings_state.voigt
    if patch.ndim != 2:
        return LocalPeakFitResult(
            model=LocalFitModelType.VOIGT,
            center_patch_yx=None,
            center_image_yx=None,
            center_std_yx=None,
            amplitude=None,
            width_y=None,
            width_x=None,
            theta_rad=None,
            offset=None,
            method="voigt_fit",
            success=False,
            error_message=f"Expected a 2D ROI patch, got shape {patch.shape!r}.",
            model_patch=None,
            roi_patch=patch.copy(),
            raw_result=None,
            fit_mask=None if mask is None else mask.copy(),
        )

    peak_to_peak, peak_y, peak_x, span_y, span_x, offset_guess = _build_initial_guess_statistics(
        patch,
        mask,
    )
    sigma_y_guess = max(float(span_y) / 5.0, 0.75)
    sigma_x_guess = max(float(span_x) / 5.0, 0.75)
    gamma_y_guess = max(float(span_y) / 6.0, 0.5)
    gamma_x_guess = max(float(span_x) / 6.0, 0.5)
    default_p0 = [
        peak_to_peak,
        float(peak_y),
        float(peak_x),
        sigma_y_guess,
        sigma_x_guess,
        gamma_y_guess,
        gamma_x_guess,
        0.0,
        offset_guess,
    ]

    fit_result = fit_2d_voigt_on_patch(
        patch,
        roi_origin_yx=request.roi_origin_yx,
        fit_mask=mask,
        initial_params=_merge_voigt_initial_params(
            default_p0=default_p0,
            common_settings=common_settings,
            voigt_settings=voigt_settings,
        ),
        parameter_bounds=_build_voigt_parameter_bounds(common_settings, voigt_settings),
        max_nfev=common_settings.max_nfev,
        compute_uncertainty=bool(request.compute_uncertainty),
    )
    if fit_result is None:
        return LocalPeakFitResult(
            model=LocalFitModelType.VOIGT,
            center_patch_yx=None,
            center_image_yx=None,
            center_std_yx=None,
            amplitude=None,
            width_y=None,
            width_x=None,
            theta_rad=None,
            offset=None,
            method="voigt_fit",
            success=False,
            error_message="Voigt fit could not be computed for the provided ROI patch.",
            model_patch=None,
            roi_patch=patch.copy(),
            raw_result=None,
            fit_mask=None if mask is None else mask.copy(),
        )

    center_image_yx = (float(fit_result.center[0]), float(fit_result.center[1]))
    center_patch_yx = (
        float(center_image_yx[0] - request.roi_origin_yx[0]),
        float(center_image_yx[1] - request.roi_origin_yx[1]),
    )
    sigma_yx = None
    if fit_result.center_std is not None:
        sigma_yx = (
            float(fit_result.center_std[0]),
            float(fit_result.center_std[1]),
        )

    amplitude = sigma_y = sigma_x = gamma_y = gamma_x = theta_rad = offset = None
    model_patch = None
    if fit_result.popt is not None and len(fit_result.popt) >= 9:
        amplitude = float(fit_result.popt[0])
        sigma_y = abs(float(fit_result.popt[3]))
        sigma_x = abs(float(fit_result.popt[4]))
        gamma_y = abs(float(fit_result.popt[5]))
        gamma_x = abs(float(fit_result.popt[6]))
        theta_rad = float(fit_result.popt[7])
        offset = float(fit_result.popt[8])
        if sigma_y != 0.0 and sigma_x != 0.0 and gamma_y != 0.0 and gamma_x != 0.0:
            model_patch = _build_voigt_model_patch(
                patch.shape,
                amplitude,
                center_patch_yx,
                sigma_y,
                sigma_x,
                gamma_y,
                gamma_x,
                theta_rad,
                offset,
            )

    error_message = None
    if not fit_result.success:
        error_message = (
            "Voigt fit did not converge; fallback estimate returned by "
            f"{fit_result.method}."
        )

    return LocalPeakFitResult(
        model=LocalFitModelType.VOIGT,
        center_patch_yx=center_patch_yx,
        center_image_yx=center_image_yx,
        center_std_yx=sigma_yx,
        amplitude=amplitude,
        width_y=sigma_y,
        width_x=sigma_x,
        theta_rad=theta_rad,
        offset=offset,
        method=fit_result.method,
        success=bool(fit_result.success),
        error_message=error_message,
        model_patch=model_patch,
        roi_patch=fit_result.roi_patch.copy(),
        raw_result=fit_result,
        fit_mask=None if mask is None else mask.copy(),
        shape_parameters={
            "gamma_y": gamma_y,
            "gamma_x": gamma_x,
        },
    )


def fit_local_peak(request: LocalFitRequest) -> LocalPeakFitResult:
    """
    Run the requested local model through the common AtomMapper fit interface.

    Gaussian, Lorentzian and Voigt are implemented. Unsupported model types still
    return a structured failure result so the GUI can rely on one stable
    request/result contract.
    """

    if request.model is LocalFitModelType.GAUSSIAN:
        return _fit_gaussian_request(request)
    if request.model is LocalFitModelType.LORENTZIAN:
        return _fit_lorentzian_request(request)
    if request.model is LocalFitModelType.VOIGT:
        return _fit_voigt_request(request)
    return _unsupported_model_result(
        request,
        f"Unsupported local fit model: {request.model!r}.",
    )


def fit_gaussian_to_roi_patch(
    roi_patch: np.ndarray,
    *,
    roi_origin_yx: Tuple[int, int] = (0, 0),
    compute_uncertainty: bool = True,
) -> LocalPeakFitResult:
    """
    Compatibility wrapper for the current Gaussian-only workflow.

    The adapter never raises on fit failure. Instead it returns a structured
    result with ``success=False`` and an ``error_message``.
    """

    return fit_local_peak(
        LocalFitRequest(
            model=LocalFitModelType.GAUSSIAN,
            roi_patch=roi_patch,
            roi_origin_yx=roi_origin_yx,
            compute_uncertainty=compute_uncertainty,
        )
    )


GaussianPatchFitResult = LocalPeakFitResult
