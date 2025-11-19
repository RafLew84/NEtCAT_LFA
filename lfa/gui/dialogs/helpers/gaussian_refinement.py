from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from ....analysis.peak_fitting import (
    PeakRefinementResult,
    derive_gaussian_center_std,
    fit_2d_gaussian_in_roi,
)


@dataclass
class GaussianRefinementOutcome:
    """Result container returned by :func:`run_gaussian_refinement_for_roi`."""

    center_yx: Tuple[float, float]
    sigma_yx: Optional[Tuple[float, float]]
    covariance: Optional[np.ndarray]
    fit_result: Optional[PeakRefinementResult]
    used_preview: bool


def _clamp_center(
    value: int,
    radius: int,
    limit: int,
) -> int:
    if limit <= 0:
        return 0
    if limit <= 2 * radius:
        return int(np.clip(value, 0, limit - 1))
    return int(np.clip(value, radius, limit - 1 - radius))


def run_gaussian_refinement_for_roi(
    fft_magnitude_data: Optional[np.ndarray],
    center_yx: Tuple[int, int],
    roi_radius: int,
    *,
    preview_result: Optional[PeakRefinementResult] = None,
    require_uncertainty: bool = True,
) -> GaussianRefinementOutcome:
    """
    Refine a spot around ``center_yx`` using the shared Gaussian fit helper.

    Ensures the ROI stays within bounds, optionally reuses a preview result, and
    derives coordinate covariance only when requested.
    """
    if fft_magnitude_data is None or fft_magnitude_data.ndim != 2:
        return GaussianRefinementOutcome(
            center_yx=(float(center_yx[0]), float(center_yx[1])),
            sigma_yx=None,
            covariance=None,
            fit_result=None,
            used_preview=False,
        )

    rows, cols = fft_magnitude_data.shape
    if rows == 0 or cols == 0:
        return GaussianRefinementOutcome(
            center_yx=(float(center_yx[0]), float(center_yx[1])),
            sigma_yx=None,
            covariance=None,
            fit_result=None,
            used_preview=False,
        )

    radius = max(1, int(roi_radius))
    eff_center_y = _clamp_center(int(center_yx[0]), radius, rows)
    eff_center_x = _clamp_center(int(center_yx[1]), radius, cols)

    result = preview_result
    used_preview = False
    sigma: Optional[Tuple[float, float]] = None

    if result is not None:
        used_preview = True
        if require_uncertainty:
            sigma = derive_gaussian_center_std(result)
            if sigma is None:
                result = None
                used_preview = False

    if result is None:
        result = fit_2d_gaussian_in_roi(
            fft_magnitude_data,
            (eff_center_y, eff_center_x),
            radius,
            compute_uncertainty=require_uncertainty,
        )
        if result is None:
            return GaussianRefinementOutcome(
                center_yx=(float(center_yx[0]), float(center_yx[1])),
                sigma_yx=None,
                covariance=None,
                fit_result=None,
                used_preview=False,
            )
        if require_uncertainty:
            if result.center_std:
                sigma = (
                    float(result.center_std[0]),
                    float(result.center_std[1]),
                )
            else:
                sigma = derive_gaussian_center_std(result)

    center = (float(result.center[0]), float(result.center[1]))

    covariance = None
    if sigma is not None:
        covariance = np.array(
            [[sigma[0] ** 2, 0.0], [0.0, sigma[1] ** 2]],
            dtype=float,
        )

    return GaussianRefinementOutcome(
        center_yx=center,
        sigma_yx=sigma,
        covariance=covariance,
        fit_result=result,
        used_preview=used_preview,
    )
