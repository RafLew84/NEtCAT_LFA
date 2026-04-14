"""Tests for the bridge between the pyqtgraph viewport and existing preview widgets."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PyQt6", reason="PyQt6 is required for AtomMapper GUI tests")
pytest.importorskip("pytestqt", reason="pytest-qt is required for AtomMapper GUI tests")
pytest.importorskip("pyqtgraph", reason="pyqtgraph is required for the 2B viewport refactor")

from AtomMapper.app.gaussian_preview import GaussianFitPreviewWidget
from AtomMapper.app.fit_models import LocalFitModelType
from AtomMapper.app.fit_settings import FitSettingsState
from AtomMapper.app.models import LoadedImage, ROIState
from AtomMapper.app.polygon_mask import PolygonMaskState
from AtomMapper.app.pyqtgraph_image_view import PyQtGraphSTMViewport
from AtomMapper.app.pyqtgraph_preview_bridge import PyQtGraphPreviewBridge
from AtomMapper.app.roi_preview import ROIPreviewWidget


def _make_loaded_image(name: str, image_data: np.ndarray) -> LoadedImage:
    return LoadedImage(
        source_path=f"/tmp/{name}",
        display_name=name,
        file_extension=".stp",
        image_data=image_data,
        pixels_x=image_data.shape[1],
        pixels_y=image_data.shape[0],
        size_nm_x=float(image_data.shape[1]),
        size_nm_y=float(image_data.shape[0]),
        metadata={"image_type": "Topo"},
        raw_metadata={},
    )


def _make_gaussian_image(name: str, size: int = 40, *, amplitude: float = 20.0, offset: float = 1.0) -> LoadedImage:
    image_data = np.full((size, size), offset, dtype=float)
    patch_half = 6
    center = size // 2
    y_grid, x_grid = np.mgrid[-patch_half : patch_half + 1, -patch_half : patch_half + 1]
    gaussian_patch = amplitude * np.exp(-((y_grid**2) / (2.0 * 1.6**2) + (x_grid**2) / (2.0 * 1.8**2)))
    image_data[
        center - patch_half : center + patch_half + 1,
        center - patch_half : center + patch_half + 1,
    ] += gaussian_patch
    return _make_loaded_image(name, image_data)


def _make_lorentzian_image(name: str, size: int = 40, *, amplitude: float = 20.0, offset: float = 1.0) -> LoadedImage:
    image_data = np.full((size, size), offset, dtype=float)
    patch_half = 6
    center = size // 2
    y_grid, x_grid = np.mgrid[-patch_half : patch_half + 1, -patch_half : patch_half + 1]
    theta = 0.1
    y_rot = np.cos(theta) * y_grid + np.sin(theta) * x_grid
    x_rot = -np.sin(theta) * y_grid + np.cos(theta) * x_grid
    lorentzian_patch = amplitude / (1.0 + (y_rot / 1.6) ** 2 + (x_rot / 1.9) ** 2)
    image_data[
        center - patch_half : center + patch_half + 1,
        center - patch_half : center + patch_half + 1,
    ] += lorentzian_patch
    return _make_loaded_image(name, image_data)


def _make_voigt_image(name: str, size: int = 40, *, amplitude: float = 20.0, offset: float = 1.0) -> LoadedImage:
    from scipy.special import voigt_profile

    image_data = np.full((size, size), offset, dtype=float)
    patch_half = 6
    center = size // 2
    y_grid, x_grid = np.mgrid[-patch_half : patch_half + 1, -patch_half : patch_half + 1]
    theta = 0.08
    y_rot = np.cos(theta) * y_grid + np.sin(theta) * x_grid
    x_rot = -np.sin(theta) * y_grid + np.cos(theta) * x_grid
    profile_y = voigt_profile(y_rot, 1.2, 0.7)
    profile_x = voigt_profile(x_rot, 1.8, 1.1)
    profile_y /= float(voigt_profile(np.array([0.0]), 1.2, 0.7)[0])
    profile_x /= float(voigt_profile(np.array([0.0]), 1.8, 1.1)[0])
    voigt_patch = amplitude * profile_y * profile_x
    image_data[
        center - patch_half : center + patch_half + 1,
        center - patch_half : center + patch_half + 1,
    ] += voigt_patch
    return _make_loaded_image(name, image_data)


def _make_bridge(qtbot) -> tuple[PyQtGraphSTMViewport, ROIPreviewWidget, GaussianFitPreviewWidget, PyQtGraphPreviewBridge]:
    viewport = PyQtGraphSTMViewport()
    roi_preview = ROIPreviewWidget()
    gaussian_preview = GaussianFitPreviewWidget()
    bridge = PyQtGraphPreviewBridge(viewport, roi_preview, gaussian_preview)
    qtbot.addWidget(viewport)
    qtbot.addWidget(roi_preview)
    qtbot.addWidget(gaussian_preview)
    return viewport, roi_preview, gaussian_preview, bridge


def test_pyqtgraph_preview_bridge_updates_roi_and_gaussian_previews(qtbot):
    viewport, roi_preview, gaussian_preview, bridge = _make_bridge(qtbot)
    image = _make_gaussian_image("bridge-gauss.stp", size=40)

    bridge.set_loaded_image(image)
    bridge.set_roi_state(ROIState(x=14, y=14, width=12, height=12))

    assert viewport.current_loaded_image == image
    assert roi_preview.current_loaded_image == image
    assert roi_preview.current_patch_data is not None
    assert roi_preview.current_patch_data.shape == (12, 12)
    assert gaussian_preview.current_fit_result is not None
    assert gaussian_preview.current_fit_result.success is True
    assert gaussian_preview.current_fit_result.model_patch is not None


def test_pyqtgraph_preview_bridge_updates_previews_when_viewport_roi_changes(qtbot):
    viewport, roi_preview, gaussian_preview, bridge = _make_bridge(qtbot)
    image = _make_gaussian_image("bridge-edit.stp", size=40)

    bridge.set_loaded_image(image)
    bridge.set_roi_state(ROIState(x=14, y=14, width=12, height=12))

    emitted: list[ROIState] = []
    bridge.roi_state_edited.connect(emitted.append)

    viewport.roi_item.setPos((0.0, 0.0))
    viewport.roi_item.setSize((8.0, 8.0))

    qtbot.waitUntil(lambda: len(emitted) >= 2)
    assert bridge.current_roi_state == emitted[-1]
    assert roi_preview.current_patch_data is not None
    assert roi_preview.current_patch_data.shape == (8, 8)
    assert gaussian_preview.current_fit_result is not None
    assert gaussian_preview.current_fit_result.success is False


def test_pyqtgraph_preview_bridge_recomputes_previews_for_active_image_change(qtbot):
    viewport, roi_preview, gaussian_preview, bridge = _make_bridge(qtbot)
    first = _make_gaussian_image("bridge-first.stp", size=40, amplitude=20.0, offset=1.0)
    second = _make_gaussian_image("bridge-second.stp", size=52, amplitude=12.0, offset=3.0)

    bridge.set_loaded_image(first)
    bridge.set_roi_state(ROIState(x=14, y=14, width=12, height=12))
    first_patch = np.array(roi_preview.current_patch_data, copy=True)

    bridge.set_loaded_image(second)
    bridge.set_roi_state(ROIState(x=20, y=20, width=12, height=12))

    assert viewport.current_loaded_image == second
    assert roi_preview.current_loaded_image == second
    assert roi_preview.current_patch_data is not None
    assert not np.array_equal(roi_preview.current_patch_data, first_patch)
    assert gaussian_preview.current_fit_result is not None
    assert gaussian_preview.current_fit_result.success is True


def test_pyqtgraph_preview_bridge_is_unchanged_by_histogram_level_changes(qtbot):
    viewport, roi_preview, gaussian_preview, bridge = _make_bridge(qtbot)
    image = _make_gaussian_image("bridge-gamma.stp", size=40)

    bridge.set_loaded_image(image)
    bridge.set_roi_state(ROIState(x=14, y=14, width=12, height=12))

    patch_before = np.array(roi_preview.current_patch_data, copy=True)
    fit_before = gaussian_preview.current_fit_result
    assert fit_before is not None
    center_before = fit_before.center_patch_yx

    viewport.image_view.setLevels(min=1.5, max=18.0)

    assert np.array_equal(roi_preview.current_patch_data, patch_before)
    assert gaussian_preview.current_fit_result is fit_before
    assert gaussian_preview.current_fit_result.center_patch_yx == center_before


def test_pyqtgraph_preview_bridge_refreshes_fit_when_model_selection_changes(qtbot):
    viewport, roi_preview, gaussian_preview, bridge = _make_bridge(qtbot)
    image = _make_lorentzian_image("bridge-model-switch.stp", size=40)

    bridge.set_loaded_image(image)
    bridge.set_roi_state(ROIState(x=14, y=14, width=12, height=12))

    assert gaussian_preview.current_fit_result is not None
    assert gaussian_preview.current_fit_result.model is LocalFitModelType.GAUSSIAN
    assert gaussian_preview.current_fit_result.success is True

    bridge.set_fit_settings_state(FitSettingsState(model=LocalFitModelType.LORENTZIAN))

    assert gaussian_preview.current_fit_result is not None
    assert gaussian_preview.current_fit_result.model is LocalFitModelType.LORENTZIAN
    assert gaussian_preview.current_fit_result.success is True
    assert "Lorentzian" in gaussian_preview.title_label.text()


def test_pyqtgraph_preview_bridge_supports_voigt_model_preview(qtbot):
    viewport, roi_preview, gaussian_preview, bridge = _make_bridge(qtbot)
    image = _make_voigt_image("bridge-voigt.stp", size=40)

    bridge.set_loaded_image(image)
    bridge.set_roi_state(ROIState(x=14, y=14, width=12, height=12))
    bridge.set_fit_settings_state(FitSettingsState(model=LocalFitModelType.VOIGT))

    assert gaussian_preview.current_fit_result is not None
    assert gaussian_preview.current_fit_result.model is LocalFitModelType.VOIGT
    assert gaussian_preview.current_fit_result.success is True
    assert gaussian_preview.current_fit_result.model_patch is not None
    assert gaussian_preview.current_fit_result.shape_parameters["gamma_y"] is not None
    assert gaussian_preview.current_fit_result.shape_parameters["gamma_x"] is not None
    assert "Voigt" in gaussian_preview.title_label.text()


def test_pyqtgraph_preview_bridge_passes_polygon_mask_to_current_fit(qtbot):
    viewport, roi_preview, gaussian_preview, bridge = _make_bridge(qtbot)
    image = _make_gaussian_image("bridge-mask.stp", size=40)

    bridge.set_loaded_image(image)
    bridge.set_roi_state(ROIState(x=14, y=14, width=12, height=12))
    bridge.set_polygon_mask_state(
        PolygonMaskState(
            vertices_xy=(
                (17.0, 17.0),
                (23.0, 17.0),
                (23.0, 23.0),
                (17.0, 23.0),
            )
        )
    )

    fit_result = bridge.compute_current_fit_result()

    assert fit_result is not None
    assert fit_result.fit_mask is not None
    assert int(fit_result.fit_mask.sum()) > 0
    assert fit_result.raw_result is not None
    assert fit_result.raw_result.metadata["fit_mask_pixel_count"] == int(fit_result.fit_mask.sum())
    assert gaussian_preview.current_fit_result is not None
    assert gaussian_preview.current_fit_result.fit_mask is not None
