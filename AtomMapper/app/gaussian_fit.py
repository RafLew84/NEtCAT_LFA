"""AtomMapper-specific adapter for Gaussian fitting on ROI patches."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from lfa.analysis.peak_fitting import PeakRefinementResult, fit_2d_gaussian_on_patch


@dataclass
class GaussianPatchFitResult:
    """Normalized AtomMapper-facing result for a Gaussian fit on an ROI patch."""

    center_patch_yx: Optional[Tuple[float, float]]
    center_image_yx: Optional[Tuple[float, float]]
    center_std_yx: Optional[Tuple[float, float]]
    amplitude: Optional[float]
    sigma_y: Optional[float]
    sigma_x: Optional[float]
    theta_rad: Optional[float]
    offset: Optional[float]
    method: str
    success: bool
    error_message: Optional[str]
    roi_patch: np.ndarray = field(repr=False)
    raw_result: Optional[PeakRefinementResult] = field(default=None, repr=False)
    model_patch: Optional[np.ndarray] = field(default=None, repr=False)


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


def fit_gaussian_to_roi_patch(
    roi_patch: np.ndarray,
    *,
    roi_origin_yx: Tuple[int, int] = (0, 0),
    compute_uncertainty: bool = True,
) -> GaussianPatchFitResult:
    """
    Run the shared 2D Gaussian fit on a pre-extracted ROI patch.

    The adapter never raises on fit failure. Instead it returns a structured
    ``GaussianPatchFitResult`` with ``success=False`` and an ``error_message``.
    """
    patch = np.asarray(roi_patch, dtype=float)
    if patch.ndim != 2:
        return GaussianPatchFitResult(
            center_patch_yx=None,
            center_image_yx=None,
            center_std_yx=None,
            amplitude=None,
            sigma_y=None,
            sigma_x=None,
            theta_rad=None,
            offset=None,
            method="gaussian_fit",
            success=False,
            error_message=f"Expected a 2D ROI patch, got shape {patch.shape!r}.",
            model_patch=None,
            roi_patch=patch.copy(),
            raw_result=None,
        )

    fit_result = fit_2d_gaussian_on_patch(
        patch,
        roi_origin_yx=roi_origin_yx,
        compute_uncertainty=compute_uncertainty,
    )
    if fit_result is None:
        return GaussianPatchFitResult(
            center_patch_yx=None,
            center_image_yx=None,
            center_std_yx=None,
            amplitude=None,
            sigma_y=None,
            sigma_x=None,
            theta_rad=None,
            offset=None,
            method="gaussian_fit",
            success=False,
            error_message="Gaussian fit could not be computed for the provided ROI patch.",
            model_patch=None,
            roi_patch=patch.copy(),
            raw_result=None,
        )

    center_image_yx = (float(fit_result.center[0]), float(fit_result.center[1]))
    center_patch_yx = (
        float(center_image_yx[0] - roi_origin_yx[0]),
        float(center_image_yx[1] - roi_origin_yx[1]),
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

    return GaussianPatchFitResult(
        center_patch_yx=center_patch_yx,
        center_image_yx=center_image_yx,
        center_std_yx=sigma_yx,
        amplitude=amplitude,
        sigma_y=sigma_y,
        sigma_x=sigma_x,
        theta_rad=theta_rad,
        offset=offset,
        method=fit_result.method,
        success=bool(fit_result.success),
        error_message=error_message,
        model_patch=model_patch,
        roi_patch=fit_result.roi_patch.copy(),
        raw_result=fit_result,
    )
