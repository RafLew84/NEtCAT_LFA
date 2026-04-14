"""Tests for the common local fit-model contracts."""

from __future__ import annotations

import numpy as np

from AtomMapper.app.fit_models import LocalFitModelType, LocalFitRequest, LocalPeakFitResult


def test_local_fit_request_normalizes_patch_and_mask_without_mutating_source():
    patch = np.arange(9, dtype=np.float32).reshape(3, 3)
    mask = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]], dtype=np.uint8)
    request = LocalFitRequest(
        model=LocalFitModelType.GAUSSIAN,
        roi_patch=patch,
        fit_mask=mask,
    )

    normalized_patch = request.normalized_patch()
    normalized_mask = request.normalized_mask()

    assert normalized_patch.dtype == float
    assert np.array_equal(normalized_patch, patch)
    assert normalized_mask is not None
    assert normalized_mask.dtype == bool
    assert np.array_equal(normalized_mask, mask.astype(bool))


def test_local_peak_fit_result_exposes_backward_compatible_sigma_aliases():
    result = LocalPeakFitResult(
        model=LocalFitModelType.GAUSSIAN,
        center_patch_yx=(1.0, 2.0),
        center_image_yx=(3.0, 4.0),
        center_std_yx=None,
        amplitude=5.0,
        width_y=1.5,
        width_x=2.5,
        theta_rad=0.2,
        offset=0.8,
        method="gaussian_fit",
        success=True,
        error_message=None,
        roi_patch=np.ones((3, 3), dtype=float),
    )

    assert result.sigma_y == 1.5
    assert result.sigma_x == 2.5
