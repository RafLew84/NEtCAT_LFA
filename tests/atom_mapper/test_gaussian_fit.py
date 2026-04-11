"""Tests for the AtomMapper Gaussian-fit adapter."""

from __future__ import annotations

import numpy as np
import pytest

from lfa.analysis.peak_fitting import SCIPY_AVAILABLE

from AtomMapper.app.gaussian_fit import fit_gaussian_to_roi_patch


@pytest.mark.skipif(not SCIPY_AVAILABLE, reason="SciPy not available, skipping Gaussian fit tests")
def test_atommapper_gaussian_adapter_returns_patch_and_image_coordinates():
    rows, cols = 15, 17
    y_grid, x_grid = np.mgrid[0:rows, 0:cols]
    y0_true, x0_true = 7.3, 9.2
    amplitude = 12.0
    sigma_y, sigma_x = 2.0, 1.6
    offset = 1.25
    patch = offset + amplitude * np.exp(
        -(((y_grid - y0_true) ** 2) / (2.0 * sigma_y**2) + ((x_grid - x0_true) ** 2) / (2.0 * sigma_x**2))
    )

    result = fit_gaussian_to_roi_patch(patch, roi_origin_yx=(40, 60))

    assert result.success is True
    assert result.error_message is None
    assert result.method == "gaussian_fit"
    assert result.center_patch_yx is not None
    assert result.center_image_yx is not None
    assert np.isclose(result.center_patch_yx[0], y0_true, atol=0.1)
    assert np.isclose(result.center_patch_yx[1], x0_true, atol=0.1)
    assert np.isclose(result.center_image_yx[0], 40 + y0_true, atol=0.1)
    assert np.isclose(result.center_image_yx[1], 60 + x0_true, atol=0.1)
    assert result.center_std_yx is not None
    assert result.amplitude is not None
    assert result.sigma_y is not None
    assert result.sigma_x is not None
    assert result.model_patch is not None
    assert result.model_patch.shape == patch.shape


@pytest.mark.skipif(not SCIPY_AVAILABLE, reason="SciPy not available, skipping Gaussian fit tests")
def test_atommapper_gaussian_adapter_handles_flat_patch_without_crashing():
    patch = np.ones((9, 9), dtype=float)

    result = fit_gaussian_to_roi_patch(patch)

    assert result.success is False
    assert result.center_patch_yx is None
    assert result.center_image_yx is None
    assert result.error_message is not None
    assert result.model_patch is None


def test_atommapper_gaussian_adapter_rejects_non_2d_patch():
    patch = np.arange(10, dtype=float)

    result = fit_gaussian_to_roi_patch(patch)

    assert result.success is False
    assert result.error_message is not None
    assert "Expected a 2D ROI patch" in result.error_message
    assert result.model_patch is None
