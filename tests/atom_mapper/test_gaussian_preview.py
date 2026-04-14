"""Tests for the AtomMapper Gaussian-fit preview widget."""

from __future__ import annotations

import numpy as np
import pytest

from lfa.analysis.peak_fitting import SCIPY_AVAILABLE

pytest.importorskip("PyQt6", reason="PyQt6 is required for AtomMapper GUI tests")
pytest.importorskip("pytestqt", reason="pytest-qt is required for AtomMapper GUI tests")

from AtomMapper.app.gaussian_fit import fit_gaussian_to_roi_patch
from AtomMapper.app.gaussian_preview import GaussianFitPreviewWidget


@pytest.mark.skipif(not SCIPY_AVAILABLE, reason="SciPy not available, skipping Gaussian fit tests")
def test_gaussian_preview_renders_model_with_center_marker(qtbot):
    rows, cols = 15, 17
    y_grid, x_grid = np.mgrid[0:rows, 0:cols]
    patch = 2.0 + 15.0 * np.exp(
        -(((y_grid - 6.4) ** 2) / (2.0 * 1.7**2) + ((x_grid - 9.1) ** 2) / (2.0 * 1.9**2))
    )

    fit_result = fit_gaussian_to_roi_patch(patch, roi_origin_yx=(100, 150), compute_uncertainty=False)
    preview = GaussianFitPreviewWidget()
    qtbot.addWidget(preview)
    preview.set_fit_result(fit_result)

    assert preview.current_fit_result is not None
    assert preview.current_fit_result.success is True
    assert preview.current_model_patch is not None
    assert preview.preview_label.pixmap() is not None
    assert not preview.preview_label.pixmap().isNull()
    assert preview.title_label.text() == "Fit preview - Gaussian"
    assert "center y=" in preview.info_label.text()
    assert "sigma_y=" in preview.info_label.text()
    assert "mask=off" in preview.info_label.text()


@pytest.mark.skipif(not SCIPY_AVAILABLE, reason="SciPy not available, skipping Gaussian fit tests")
def test_gaussian_preview_shows_masked_roi_and_model_comparison(qtbot):
    rows, cols = 15, 17
    y_grid, x_grid = np.mgrid[0:rows, 0:cols]
    patch = 2.0 + 15.0 * np.exp(
        -(((y_grid - 6.4) ** 2) / (2.0 * 1.7**2) + ((x_grid - 9.1) ** 2) / (2.0 * 1.9**2))
    )

    fit_result = fit_gaussian_to_roi_patch(patch, roi_origin_yx=(100, 150), compute_uncertainty=False)
    assert fit_result.fit_mask is None
    fit_result.fit_mask = np.zeros_like(patch, dtype=bool)
    fit_result.fit_mask[3:12, 4:13] = True

    preview = GaussianFitPreviewWidget()
    qtbot.addWidget(preview)
    preview.set_fit_result(fit_result)

    assert preview.preview_label.pixmap() is not None
    assert not preview.preview_label.pixmap().isNull()
    assert preview.title_label.text() == "Fit preview - Gaussian"
    assert "mask=" in preview.info_label.text()
    assert "px" in preview.info_label.text()


def test_gaussian_preview_handles_missing_model(qtbot):
    preview = GaussianFitPreviewWidget()
    qtbot.addWidget(preview)
    preview.set_fit_result(None)

    assert preview.current_fit_result is None
    assert preview.preview_label.text() == "Fit preview will appear here."
