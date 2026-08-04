"""End-to-end GUI smoke tests for the AtomMapper foundation workflow."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QCheckBox, QDialog, QLineEdit, QMessageBox, QSpinBox

from AtomMapper.app.controller import AtomMapperController
from AtomMapper.app.fit_models import LocalFitModelType
from AtomMapper.app.main import create_main_window
from AtomMapper.app.main_window import AtomMapperMainWindow
from AtomMapper.app.models import AtomPoint, LoadedImage, ROIState
from AtomMapper.app.plots import PlotUnit, RowPlotMode
from AtomMapper.app.polygon_mask import PolygonMaskState
from AtomMapper.app.preprocessing import is_bm3d_available
from AtomMapper.app.preprocessing_dialog import PreprocessingDialog
from AtomMapper.app.preprocessing_state import (
    BlurParameters,
    BM3DParameters,
    FlipParameters,
    NonLocalMeansParameters,
    PreprocessingMethod,
    PreprocessingState,
    RotateParameters,
)
from AtomMapper.app.session_io import build_session_from_runtime, save_session_to_file
from AtomMapper.app.session_model import ATOMMAPPER_SESSION_VERSION, SessionViewState

pytest.importorskip("PyQt6", reason="PyQt6 is required for AtomMapper GUI tests")
pytest.importorskip("pytestqt", reason="pytest-qt is required for AtomMapper GUI tests")

PROJECT_ROOT = Path(__file__).resolve().parents[2]


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


def _make_gaussian_image(
    name: str,
    size: int = 40,
    *,
    amplitude: float = 20.0,
    offset: float = 1.0,
) -> LoadedImage:
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


def _make_lorentzian_image(
    name: str,
    size: int = 40,
    *,
    amplitude: float = 20.0,
    offset: float = 1.0,
) -> LoadedImage:
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


def _make_voigt_image(
    name: str,
    size: int = 40,
    *,
    amplitude: float = 20.0,
    offset: float = 1.0,
) -> LoadedImage:
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


def _make_overlapping_lorentzian_image(
    name: str,
    size: int = 48,
    *,
    offset: float = 1.0,
    primary_amplitude: float = 16.0,
    secondary_amplitude: float = 28.0,
) -> LoadedImage:
    y_grid, x_grid = np.mgrid[0:size, 0:size].astype(float)

    def lorentzian_peak(
        center_y: float,
        center_x: float,
        amplitude: float,
        gamma_y: float,
        gamma_x: float,
        theta: float,
    ) -> np.ndarray:
        y_shift = y_grid - center_y
        x_shift = x_grid - center_x
        y_rot = np.cos(theta) * y_shift + np.sin(theta) * x_shift
        x_rot = -np.sin(theta) * y_shift + np.cos(theta) * x_shift
        return amplitude / (1.0 + (y_rot / gamma_y) ** 2 + (x_rot / gamma_x) ** 2)

    center = size / 2.0
    image_data = np.full((size, size), offset, dtype=float)
    image_data += lorentzian_peak(
        center,
        center - 4.0,
        primary_amplitude,
        1.4,
        1.7,
        0.08,
    )
    image_data += lorentzian_peak(
        center,
        center + 4.0,
        secondary_amplitude,
        1.6,
        1.9,
        -0.05,
    )
    return _make_loaded_image(name, image_data)


def test_main_window_loads_sample_file_via_button(qtbot, monkeypatch):
    window = create_main_window()
    qtbot.addWidget(window)

    sample_path = str(PROJECT_ROOT / "data" / "8343.stp")

    monkeypatch.setattr(
        "AtomMapper.app.main_window.QFileDialog.getOpenFileNames",
        lambda *args, **kwargs: ([sample_path], "STM files (*.stp *.s94)"),
    )

    window.load_files_action.trigger()

    assert len(window.controller.loaded_images) == 1
    assert window.file_list_widget.count() == 1
    assert window.file_list_widget.item(0).text() == "8343.stp"
    assert window.controller.active_image is not None
    assert window.controller.active_image.display_name == "8343.stp"
    assert window.image_viewport.current_loaded_image is not None
    assert window.image_viewport.current_loaded_image.display_name == "8343.stp"
    assert window.image_viewport.image_item is not None
    assert window.image_viewport.image_item.image is not None
    assert window.image_viewport.stack.currentWidget() == window.image_viewport.plot_widget
    assert "nm" in window.active_image_label.text()
    assert "nm/px" in window.active_image_label.toolTip()
    assert window.preview_bridge.current_roi_patch_data is not None
    assert window.gaussian_fit_preview.current_fit_result is not None
    assert "Loaded 1 STM file." in window.statusBar().currentMessage()
    assert window.file_list_hint_label.text() == "1 STM file loaded."


def test_main_window_reports_load_errors(qtbot, monkeypatch, tmp_path: Path):
    window = create_main_window()
    qtbot.addWidget(window)

    bad_path = str(tmp_path / "bad.txt")
    Path(bad_path).write_text("dummy", encoding="utf-8")

    monkeypatch.setattr(
        "AtomMapper.app.main_window.QFileDialog.getOpenFileNames",
        lambda *args, **kwargs: ([bad_path], "STM files (*.stp *.s94)"),
    )

    captured: dict[str, str] = {}

    def fake_warning(parent, title, text):
        captured["title"] = title
        captured["text"] = text
        return QMessageBox.StandardButton.Ok

    monkeypatch.setattr("AtomMapper.app.main_window.QMessageBox.warning", fake_warning)

    window.load_files_action.trigger()

    assert len(window.controller.loaded_images) == 0
    assert window.file_list_widget.count() == 0
    assert captured["title"] == "AtomMapper - Load Error"
    assert "Could not load one or more STM files" in captured["text"]
    assert "bad.txt" in captured["text"]
    assert "Some files failed to load." in window.statusBar().currentMessage()


def test_main_window_updates_current_roi_patch_when_roi_changes(qtbot):
    controller = AtomMapperController()
    image = _make_loaded_image(
        "roi-sync.stp",
        np.arange(400, dtype=float).reshape((20, 20)),
    )
    controller.set_loaded_images([image])

    window = AtomMapperMainWindow(controller=controller)
    qtbot.addWidget(window)

    initial_patch = window.preview_bridge.current_roi_patch_data
    assert initial_patch is not None

    clamped_roi = controller.update_active_roi_state(ROIState(x=2, y=3, width=6, height=5))

    updated_patch = window.preview_bridge.current_roi_patch_data
    assert updated_patch is not None
    assert updated_patch.shape == (clamped_roi.height, clamped_roi.width)
    assert np.array_equal(
        updated_patch,
        image.image_data[
            clamped_roi.y : clamped_roi.y + clamped_roi.height,
            clamped_roi.x : clamped_roi.x + clamped_roi.width,
        ],
    )
    assert not np.array_equal(initial_patch, updated_patch)


def test_main_window_updates_gaussian_fit_preview_when_roi_changes(qtbot):
    controller = AtomMapperController()
    image = _make_gaussian_image("gauss.stp", size=40)
    controller.set_loaded_images([image])

    window = AtomMapperMainWindow(controller=controller)
    qtbot.addWidget(window)

    initial_fit = window.gaussian_fit_preview.current_fit_result
    assert initial_fit is not None
    assert initial_fit.success is True
    assert initial_fit.model_patch is not None
    assert window.gaussian_fit_preview.preview_label.pixmap() is not None

    controller.update_active_roi_state(ROIState(x=0, y=0, width=8, height=8))

    updated_fit = window.gaussian_fit_preview.current_fit_result
    assert updated_fit is not None
    assert updated_fit.success is False
    assert updated_fit.model_patch is None
    assert window.gaussian_fit_preview.preview_label.text() == (
        "Gaussian fit is unavailable for the current ROI."
    )


def test_main_window_checkbox_controls_gaussian_fit_preview(qtbot):
    controller = AtomMapperController()
    image = _make_gaussian_image("gauss-toggle.stp", size=40)
    controller.set_loaded_images([image])

    window = AtomMapperMainWindow(controller=controller)
    qtbot.addWidget(window)

    assert window.show_gaussian_fit_checkbox.isChecked()
    assert not window.gaussian_fit_preview.isHidden()
    assert window.gaussian_fit_preview.current_fit_result is not None
    assert window.gaussian_fit_preview.current_fit_result.success is True

    qtbot.mouseClick(window.show_gaussian_fit_checkbox, Qt.MouseButton.LeftButton)

    assert not window.show_gaussian_fit_checkbox.isChecked()
    assert window.gaussian_fit_preview.isHidden()
    assert window.gaussian_fit_preview.current_fit_result is None

    controller.update_active_roi_state(ROIState(x=0, y=0, width=8, height=8))
    assert window.gaussian_fit_preview.current_fit_result is None

    qtbot.mouseClick(window.show_gaussian_fit_checkbox, Qt.MouseButton.LeftButton)

    assert window.show_gaussian_fit_checkbox.isChecked()
    assert not window.gaussian_fit_preview.isHidden()
    assert window.gaussian_fit_preview.current_fit_result is not None
    assert window.gaussian_fit_preview.current_fit_result.success is False


def test_fit_settings_dock_is_non_modal_and_syncs_context(qtbot):
    controller = AtomMapperController()
    first = _make_gaussian_image("fit-first.stp", size=40)
    second = _make_gaussian_image("fit-second.stp", size=44, amplitude=16.0, offset=2.0)
    controller.set_loaded_images([first, second])

    window = AtomMapperMainWindow(controller=controller)
    qtbot.addWidget(window)

    assert window.fit_settings_dock.isHidden() is True

    window.fit_settings_action.trigger()

    assert window.fit_settings_dock.isHidden() is False
    assert "fit-first.stp" in window.fit_settings_panel.context_label.text()

    model_index = window.fit_settings_panel.model_combo.findData(LocalFitModelType.VOIGT)
    window.fit_settings_panel.model_combo.setCurrentIndex(model_index)

    max_nfev_spinbox = window.fit_settings_panel.findChild(QSpinBox, "atommapper_fit_common_max_nfev_spinbox")
    assert max_nfev_spinbox is not None
    max_nfev_spinbox.setValue(4200)

    window.file_list_widget.setCurrentRow(1)
    assert controller.active_image == second
    assert "fit-second.stp" in window.fit_settings_panel.context_label.text()
    assert window.fit_settings_state.model is LocalFitModelType.VOIGT
    assert window.fit_settings_state.common.max_nfev == 4200

    controller.update_active_roi_state(ROIState(x=3, y=4, width=7, height=8))
    assert "ROI: x=3, y=4, 7x8 px" in window.fit_settings_panel.context_label.text()

    window.fit_settings_dock.close()
    assert window.fit_settings_dock.isHidden() is True

    window.fit_settings_action.trigger()
    assert window.fit_settings_dock.isHidden() is False


def test_fit_settings_panel_refreshes_preview_and_add_point_uses_same_gaussian_settings(qtbot):
    controller = AtomMapperController()
    image = _make_gaussian_image("fit-settings-gaussian.stp", size=40, amplitude=22.0, offset=1.0)
    controller.set_loaded_images([image])

    window = AtomMapperMainWindow(controller=controller)
    qtbot.addWidget(window)
    qtbot.mouseClick(window.new_row_button, Qt.MouseButton.LeftButton)

    before_result = window.gaussian_fit_preview.current_fit_result
    assert before_result is not None
    assert before_result.success is True

    window.fit_settings_action.trigger()
    model_index = window.fit_settings_panel.model_combo.findData(LocalFitModelType.GAUSSIAN)
    window.fit_settings_panel.model_combo.setCurrentIndex(model_index)

    max_nfev_spinbox = window.fit_settings_panel.findChild(QSpinBox, "atommapper_fit_common_max_nfev_spinbox")
    assert max_nfev_spinbox is not None
    max_nfev_spinbox.setValue(2800)

    custom_initial_guess_checkbox = window.fit_settings_panel.findChild(
        QCheckBox,
        "atommapper_fit_common_use_custom_initial_guess_checkbox",
    )
    assert custom_initial_guess_checkbox is not None
    custom_initial_guess_checkbox.setChecked(True)

    compute_uncertainty_checkbox = window.fit_settings_panel.findChild(
        QCheckBox,
        "atommapper_fit_common_compute_uncertainty_checkbox",
    )
    assert compute_uncertainty_checkbox is not None
    compute_uncertainty_checkbox.setChecked(True)

    gaussian_sigma_y_edit = window.fit_settings_panel.findChild(
        QLineEdit,
        "atommapper_fit_gaussian_sigma_y_init_lineedit",
    )
    assert gaussian_sigma_y_edit is not None
    gaussian_sigma_y_edit.setText("1.1")
    gaussian_sigma_y_edit.editingFinished.emit()

    assert window.fit_settings_state.common.max_nfev == 2800
    assert window.fit_settings_state.common.use_custom_initial_guess is True
    assert window.fit_settings_state.common.compute_uncertainty is True
    assert window.fit_settings_state.gaussian.sigma_y_init == 1.1

    refreshed_result = window.preview_bridge.compute_current_fit_result()
    assert refreshed_result is not None
    assert refreshed_result.success is True
    assert refreshed_result.raw_result is not None
    assert refreshed_result.raw_result.metadata["max_nfev"] == 2800
    assert refreshed_result.raw_result.metadata["initial_params"][3] == pytest.approx(1.1)

    qtbot.mouseClick(window.add_point_button, Qt.MouseButton.LeftButton)

    active_row = controller.active_row
    assert active_row is not None
    point = active_row.points[0]
    assert point.fit_success is True
    assert point.metadata["fallback_used"] is False
    assert point.metadata["fit_method"] == "gaussian_fit"
    assert point.position_std_x_px is not None
    assert point.position_std_y_px is not None
    assert point.position_std_x_nm is not None
    assert point.position_std_y_nm is not None
    assert point.metadata["position_uncertainty_status"] == "computed"
    assert point.metadata["position_uncertainty_settings_source"] == "point_snapshot"
    assert point.metadata["fit_settings"] == window.fit_settings_state.to_dict()
    assert "from Gaussian fit" in window.workflow_status_label.text()


def test_fit_settings_panel_refreshes_preview_and_add_point_uses_same_lorentzian_settings(qtbot):
    controller = AtomMapperController()
    image = _make_lorentzian_image("fit-settings-lorentzian.stp", size=40, amplitude=24.0, offset=1.5)
    controller.set_loaded_images([image])

    window = AtomMapperMainWindow(controller=controller)
    qtbot.addWidget(window)
    qtbot.mouseClick(window.new_row_button, Qt.MouseButton.LeftButton)

    window.fit_settings_action.trigger()
    model_index = window.fit_settings_panel.model_combo.findData(LocalFitModelType.LORENTZIAN)
    window.fit_settings_panel.model_combo.setCurrentIndex(model_index)

    max_nfev_spinbox = window.fit_settings_panel.findChild(QSpinBox, "atommapper_fit_common_max_nfev_spinbox")
    assert max_nfev_spinbox is not None
    max_nfev_spinbox.setValue(3300)

    custom_initial_guess_checkbox = window.fit_settings_panel.findChild(
        QCheckBox,
        "atommapper_fit_common_use_custom_initial_guess_checkbox",
    )
    assert custom_initial_guess_checkbox is not None
    custom_initial_guess_checkbox.setChecked(True)

    lorentzian_gamma_y_edit = window.fit_settings_panel.findChild(
        QLineEdit,
        "atommapper_fit_lorentzian_gamma_y_init_lineedit",
    )
    assert lorentzian_gamma_y_edit is not None
    lorentzian_gamma_y_edit.setText("1.3")
    lorentzian_gamma_y_edit.editingFinished.emit()

    assert window.fit_settings_state.model is LocalFitModelType.LORENTZIAN
    assert window.fit_settings_state.common.max_nfev == 3300
    assert window.fit_settings_state.common.use_custom_initial_guess is True
    assert window.fit_settings_state.lorentzian.gamma_y_init == 1.3

    refreshed_result = window.preview_bridge.compute_current_fit_result()
    assert refreshed_result is not None
    assert refreshed_result.model is LocalFitModelType.LORENTZIAN
    assert refreshed_result.success is True
    assert refreshed_result.raw_result is not None
    assert refreshed_result.raw_result.metadata["max_nfev"] == 3300
    assert refreshed_result.raw_result.metadata["initial_params"][3] == pytest.approx(1.3)
    assert "Lorentzian" in window.gaussian_fit_preview.title_label.text()

    qtbot.mouseClick(window.add_point_button, Qt.MouseButton.LeftButton)

    active_row = controller.active_row
    assert active_row is not None
    point = active_row.points[0]
    assert point.fit_success is True
    assert point.metadata["fallback_used"] is False
    assert point.metadata["fit_method"] == "lorentzian_fit"
    assert "from Lorentzian fit" in window.workflow_status_label.text()


def test_fit_settings_panel_refreshes_preview_and_add_point_uses_same_voigt_settings(qtbot):
    controller = AtomMapperController()
    image = _make_voigt_image("fit-settings-voigt.stp", size=40, amplitude=26.0, offset=1.2)
    controller.set_loaded_images([image])

    window = AtomMapperMainWindow(controller=controller)
    qtbot.addWidget(window)
    qtbot.mouseClick(window.new_row_button, Qt.MouseButton.LeftButton)

    window.fit_settings_action.trigger()
    model_index = window.fit_settings_panel.model_combo.findData(LocalFitModelType.VOIGT)
    window.fit_settings_panel.model_combo.setCurrentIndex(model_index)

    max_nfev_spinbox = window.fit_settings_panel.findChild(QSpinBox, "atommapper_fit_common_max_nfev_spinbox")
    assert max_nfev_spinbox is not None
    max_nfev_spinbox.setValue(3600)

    custom_initial_guess_checkbox = window.fit_settings_panel.findChild(
        QCheckBox,
        "atommapper_fit_common_use_custom_initial_guess_checkbox",
    )
    assert custom_initial_guess_checkbox is not None
    custom_initial_guess_checkbox.setChecked(True)

    voigt_gamma_y_edit = window.fit_settings_panel.findChild(
        QLineEdit,
        "atommapper_fit_voigt_gamma_y_init_lineedit",
    )
    assert voigt_gamma_y_edit is not None
    voigt_gamma_y_edit.setText("0.9")
    voigt_gamma_y_edit.editingFinished.emit()

    assert window.fit_settings_state.model is LocalFitModelType.VOIGT
    assert window.fit_settings_state.common.max_nfev == 3600
    assert window.fit_settings_state.common.use_custom_initial_guess is True
    assert window.fit_settings_state.voigt.gamma_y_init == 0.9

    refreshed_result = window.preview_bridge.compute_current_fit_result()
    assert refreshed_result is not None
    assert refreshed_result.model is LocalFitModelType.VOIGT
    assert refreshed_result.success is True
    assert refreshed_result.raw_result is not None
    assert refreshed_result.raw_result.metadata["max_nfev"] == 3600
    assert refreshed_result.raw_result.metadata["initial_params"][5] == pytest.approx(0.9)
    assert refreshed_result.shape_parameters["gamma_y"] is not None
    assert "Voigt" in window.gaussian_fit_preview.title_label.text()

    qtbot.mouseClick(window.add_point_button, Qt.MouseButton.LeftButton)

    active_row = controller.active_row
    assert active_row is not None
    point = active_row.points[0]
    assert point.fit_success is True
    assert point.metadata["fallback_used"] is False
    assert point.metadata["fit_method"] == "voigt_fit"
    assert point.metadata["fit_shape_parameters"]["gamma_y"] is not None
    assert point.metadata["fit_shape_parameters"]["gamma_x"] is not None
    assert "from Voigt fit" in window.workflow_status_label.text()


def test_main_window_end_to_end_roi_fit_workflow(qtbot, monkeypatch):
    window = create_main_window()
    qtbot.addWidget(window)

    first_path = "/tmp/fake-first.stp"
    second_path = "/tmp/fake-second.stp"
    fake_images = {
        first_path: _make_gaussian_image("fake-first.stp", size=40, amplitude=20.0, offset=1.0),
        second_path: _make_gaussian_image("fake-second.stp", size=52, amplitude=12.0, offset=3.0),
    }

    monkeypatch.setattr(
        "AtomMapper.app.main_window.QFileDialog.getOpenFileNames",
        lambda *args, **kwargs: ([first_path, second_path], "STM files (*.stp *.s94)"),
    )
    monkeypatch.setattr(
        "AtomMapper.app.controller.load_loaded_image",
        lambda file_path: fake_images[str(file_path)],
    )

    window.load_files_action.trigger()

    assert window.file_list_widget.count() == 2
    assert window.controller.active_image is not None
    assert window.controller.active_image.display_name == "fake-first.stp"
    assert window.image_viewport.image_item is not None
    assert window.image_viewport.image_item.image is not None
    assert window.preview_bridge.current_roi_patch_data is not None
    assert window.gaussian_fit_preview.current_fit_result is not None
    assert window.gaussian_fit_preview.current_fit_result.success is True
    assert "Gaussian center" in window.workflow_status_label.text()

    first_patch = np.array(window.preview_bridge.current_roi_patch_data, copy=True)

    window.file_list_widget.setCurrentRow(1)

    assert window.controller.active_image is not None
    assert window.controller.active_image.display_name == "fake-second.stp"
    assert window.image_viewport.current_loaded_image is not None
    assert window.image_viewport.current_loaded_image.display_name == "fake-second.stp"
    assert window.preview_bridge.current_roi_patch_data is not None
    assert not np.array_equal(window.preview_bridge.current_roi_patch_data, first_patch)
    assert window.gaussian_fit_preview.current_fit_result is not None
    assert window.statusBar().currentMessage() == "Selected fake-second.stp."

    qtbot.mouseClick(window.show_gaussian_fit_checkbox, Qt.MouseButton.LeftButton)
    assert window.gaussian_fit_preview.isHidden()
    assert "hidden" in window.workflow_status_label.text()

    qtbot.mouseClick(window.show_gaussian_fit_checkbox, Qt.MouseButton.LeftButton)
    assert not window.gaussian_fit_preview.isHidden()
    assert window.gaussian_fit_preview.current_fit_result is not None
    assert "Gaussian center" in window.workflow_status_label.text()


def test_main_window_groups_variants_under_original_and_keeps_selection_stable(qtbot):
    controller = AtomMapperController()
    first = _make_gaussian_image("first.stp", size=40, amplitude=18.0, offset=1.0)
    second = _make_gaussian_image("second.stp", size=44, amplitude=15.0, offset=2.0)
    controller.set_loaded_images([first, second])

    window = AtomMapperMainWindow(controller=controller)
    qtbot.addWidget(window)

    variant = controller.create_blur_variant_for_active_image(sigma_px=1.2, make_active=False)

    assert window.file_list_widget.count() == 3
    assert window.file_list_widget.item(0).text() == "first.stp"
    assert window.file_list_widget.item(1).text() == f"  - {variant.display_name}"
    assert window.file_list_widget.item(1).toolTip().startswith("Variant: blur")
    assert window.file_list_widget.item(2).text() == "second.stp"

    window.file_list_widget.setCurrentRow(1)
    assert window.controller.active_image == variant
    assert window.image_viewport.current_loaded_image == variant

    window.file_list_widget.setCurrentRow(2)
    assert window.controller.active_image == second
    assert window.statusBar().currentMessage() == "Selected second.stp."


def test_main_window_syncs_active_point_selection_between_table_and_overlay(qtbot):
    controller = AtomMapperController()
    image = _make_gaussian_image("point-select.stp", size=40, amplitude=18.0, offset=1.0)
    controller.set_loaded_images([image])
    row = controller.create_row_for_active_source_group(display_name="Row 1")

    first_point = AtomPoint(
        row_id=row.row_id,
        image_id=image.image_id,
        source_group_id=image.source_group_id,
        point_index=0,
        x_px=10.0,
        y_px=11.0,
        point_id="point-1",
    )
    second_point = AtomPoint(
        row_id=row.row_id,
        image_id=image.image_id,
        source_group_id=image.source_group_id,
        point_index=1,
        x_px=18.0,
        y_px=19.0,
        point_id="point-2",
    )
    controller.add_point_to_row(first_point)
    controller.add_point_to_row(second_point)

    window = AtomMapperMainWindow(controller=controller)
    qtbot.addWidget(window)

    window.points_table_widget.setCurrentCell(1, 0)
    qtbot.waitUntil(lambda: window.image_viewport.current_active_point_id == "point-2")

    spots = {
        spot.data()["point_id"]: spot
        for spot in window.image_viewport.point_scatter_item.points()
    }
    assert spots["point-2"].size() > spots["point-1"].size()

    window.image_viewport._on_point_scatter_clicked(
        window.image_viewport.point_scatter_item,
        [spots["point-1"]],
        None,
    )
    qtbot.waitUntil(
        lambda: (
            window.points_table_widget.currentItem() is not None
            and window.points_table_widget.currentItem().data(Qt.ItemDataRole.UserRole) == "point-1"
        )
    )

    assert window.image_viewport.current_active_point_id == "point-1"


def test_main_window_delete_point_removes_selected_point_from_table_and_overlay(qtbot):
    controller = AtomMapperController()
    image = _make_gaussian_image("point-delete.stp", size=40, amplitude=18.0, offset=1.0)
    controller.set_loaded_images([image])
    row = controller.create_row_for_active_source_group(display_name="Row 1")

    first_point = AtomPoint(
        row_id=row.row_id,
        image_id=image.image_id,
        source_group_id=image.source_group_id,
        point_index=0,
        x_px=10.0,
        y_px=11.0,
        point_id="point-1",
    )
    second_point = AtomPoint(
        row_id=row.row_id,
        image_id=image.image_id,
        source_group_id=image.source_group_id,
        point_index=1,
        x_px=18.0,
        y_px=19.0,
        point_id="point-2",
    )
    controller.add_point_to_row(first_point)
    controller.add_point_to_row(second_point)

    window = AtomMapperMainWindow(controller=controller)
    qtbot.addWidget(window)

    window.points_table_widget.setCurrentCell(1, 0)
    qtbot.waitUntil(lambda: window.image_viewport.current_active_point_id == "point-2")

    assert window.delete_point_button.isEnabled()

    qtbot.mouseClick(window.delete_point_button, Qt.MouseButton.LeftButton)

    qtbot.waitUntil(lambda: window.points_table_widget.rowCount() == 1)
    assert controller.active_row is not None
    assert controller.active_row.point_count == 1
    assert controller.active_row.points[0].point_id == "point-1"
    assert len(window.image_viewport.point_scatter_item.points()) == 1
    assert window.image_viewport.current_active_point_id is None
    assert window.points_table_widget.currentItem() is None
    assert not window.delete_point_button.isEnabled()
    assert window.statusBar().currentMessage() == "Deleted point 1 from Row 1."
    assert "selection cleared" in window.workflow_status_label.text()


def test_main_window_delete_point_without_selection_does_not_crash(qtbot):
    controller = AtomMapperController()
    image = _make_gaussian_image("point-delete-none.stp", size=40, amplitude=18.0, offset=1.0)
    controller.set_loaded_images([image])
    controller.create_row_for_active_source_group(display_name="Row 1")

    window = AtomMapperMainWindow(controller=controller)
    qtbot.addWidget(window)

    assert not window.delete_point_button.isEnabled()

    window._delete_active_point()

    assert window.statusBar().currentMessage() == "Select a saved point before deleting it."
    assert "no saved point selected" in window.workflow_status_label.text()


def test_main_window_drag_move_updates_point_position_and_marks_manual_override(qtbot):
    controller = AtomMapperController()
    image = _make_gaussian_image("point-drag.stp", size=40, amplitude=18.0, offset=1.0)
    controller.set_loaded_images([image])
    row = controller.create_row_for_active_source_group(display_name="Row 1")

    point = AtomPoint(
        row_id=row.row_id,
        image_id=image.image_id,
        source_group_id=image.source_group_id,
        point_index=0,
        x_px=10.0,
        y_px=11.0,
        point_id="point-1",
    )
    controller.add_point_to_row(point)

    window = AtomMapperMainWindow(controller=controller)
    qtbot.addWidget(window)

    window.points_table_widget.setCurrentCell(0, 0)
    qtbot.waitUntil(lambda: window.image_viewport.current_active_point_id == "point-1")

    window.image_viewport.active_point_target.setPos((16.5, 17.25))
    window.image_viewport._on_active_point_target_move_finished()

    qtbot.waitUntil(
        lambda: controller.active_row is not None
        and controller.active_row.points[0].manual_override is True
    )

    moved_point = controller.active_row.points[0]
    assert moved_point.x_px == pytest.approx(16.5)
    assert moved_point.y_px == pytest.approx(17.25)
    assert moved_point.x_nm == pytest.approx(16.5)
    assert moved_point.y_nm == pytest.approx(17.25)
    assert moved_point.manual_override is True
    assert moved_point.manual_override_source == "drag"
    assert moved_point.fit_x_px == pytest.approx(10.0)
    assert moved_point.fit_y_px == pytest.approx(11.0)
    assert window.image_viewport.current_active_point_id == "point-1"
    assert window.points_table_widget.currentItem() is not None
    assert window.points_table_widget.currentItem().data(Qt.ItemDataRole.UserRole) == "point-1"
    assert window.points_table_widget.item(0, 2).text() == "16.500"
    assert window.points_table_widget.item(0, 3).text() == "17.250"
    assert window.points_table_widget.item(0, 11).text() == "manual (drag)"
    assert window.statusBar().currentMessage() == "Moved point 0 in Row 1 to x=16.50, y=17.25."
    assert "selection preserved" in window.workflow_status_label.text()


def test_main_window_add_point_inserts_after_selected_point_in_active_row(qtbot):
    controller = AtomMapperController()
    image = _make_gaussian_image("point-insert.stp", size=40, amplitude=18.0, offset=1.0)
    controller.set_loaded_images([image])
    row = controller.create_row_for_active_source_group(display_name="Row 1")

    first_point = AtomPoint(
        row_id=row.row_id,
        image_id=image.image_id,
        source_group_id=image.source_group_id,
        point_index=0,
        x_px=10.0,
        y_px=10.0,
        point_id="point-1",
    )
    second_point = AtomPoint(
        row_id=row.row_id,
        image_id=image.image_id,
        source_group_id=image.source_group_id,
        point_index=1,
        x_px=30.0,
        y_px=30.0,
        point_id="point-2",
    )
    controller.add_point_to_row(first_point)
    controller.add_point_to_row(second_point)

    window = AtomMapperMainWindow(controller=controller)
    qtbot.addWidget(window)

    window.points_table_widget.setCurrentCell(0, 0)
    qtbot.waitUntil(lambda: window.image_viewport.current_active_point_id == "point-1")

    controller.update_active_roi_state(ROIState(x=14, y=14, width=8, height=8))
    qtbot.mouseClick(window.add_point_button, Qt.MouseButton.LeftButton)

    qtbot.waitUntil(lambda: controller.active_row is not None and controller.active_row.point_count == 3)
    active_row = controller.active_row
    assert active_row is not None
    assert [point.point_index for point in active_row.points] == [0, 1, 2]
    assert [point.point_id for point in active_row.points][:1] == ["point-1"]
    assert active_row.points[2].point_id == "point-2"
    inserted_point = active_row.points[1]
    assert inserted_point.point_id not in {"point-1", "point-2"}
    assert "inserted point 1" in window.workflow_status_label.text().lower()
    assert "after point 0" in window.workflow_status_label.text().lower()
    assert window.points_table_widget.item(0, 1).text() == "0"
    assert window.points_table_widget.item(1, 1).text() == "1"
    assert window.points_table_widget.item(2, 1).text() == "2"


def test_main_window_move_point_buttons_reorder_table_and_row_plot(qtbot):
    controller = AtomMapperController()
    image = _make_gaussian_image("point-reorder.stp", size=40, amplitude=18.0, offset=1.0)
    controller.set_loaded_images([image])
    row = controller.create_row_for_active_source_group(display_name="Row 1")

    for point_index, (point_id, x_px) in enumerate((("point-1", 10.0), ("point-2", 20.0), ("point-3", 30.0))):
        controller.add_point_to_row(
            AtomPoint(
                row_id=row.row_id,
                image_id=image.image_id,
                source_group_id=image.source_group_id,
                point_index=point_index,
                x_px=x_px,
                y_px=10.0,
                point_id=point_id,
            )
        )

    window = AtomMapperMainWindow(controller=controller)
    qtbot.addWidget(window)

    window.points_table_widget.setCurrentCell(2, 0)
    qtbot.waitUntil(lambda: window.image_viewport.current_active_point_id == "point-3")
    assert window.move_point_up_button.isEnabled() is True
    assert window.move_point_down_button.isEnabled() is False

    qtbot.mouseClick(window.move_point_up_button, Qt.MouseButton.LeftButton)

    qtbot.waitUntil(
        lambda: controller.active_row is not None
        and [point.point_id for point in controller.active_row.points] == ["point-1", "point-3", "point-2"]
    )

    active_row = controller.active_row
    assert active_row is not None
    assert [point.point_index for point in active_row.points] == [0, 1, 2]
    assert window.points_table_widget.item(0, 1).text() == "0"
    assert window.points_table_widget.item(1, 1).text() == "1"
    assert window.points_table_widget.item(2, 1).text() == "2"
    assert window.points_table_widget.item(1, 0).text() == "Row 1"
    assert window.points_table_widget.currentItem() is not None
    assert window.points_table_widget.currentItem().data(Qt.ItemDataRole.UserRole) == "point-3"
    assert window.row_plot_widget.current_series is not None
    assert [sample.y_value for sample in window.row_plot_widget.current_series.samples] == [10.0, 30.0, 20.0]
    assert "table and plots refreshed" in window.workflow_status_label.text().lower()


def test_main_window_end_to_end_stage3a_point_editing_workflow(qtbot):
    controller = AtomMapperController()
    image = _make_gaussian_image("stage3a-e2e.stp", size=40, amplitude=20.0, offset=1.0)
    controller.set_loaded_images([image])

    window = AtomMapperMainWindow(controller=controller)
    qtbot.addWidget(window)

    qtbot.mouseClick(window.new_row_button, Qt.MouseButton.LeftButton)
    active_row = controller.active_row
    assert active_row is not None

    controller.update_active_roi_state(ROIState(x=8, y=8, width=12, height=12))
    qtbot.mouseClick(window.add_point_button, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: window.points_table_widget.rowCount() == 1)

    controller.update_active_roi_state(ROIState(x=16, y=16, width=12, height=12))
    qtbot.mouseClick(window.add_point_button, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: window.points_table_widget.rowCount() == 2)

    active_row = controller.active_row
    assert active_row is not None
    first_point_id = active_row.points[0].point_id
    second_point_id = active_row.points[1].point_id
    assert len(window.image_viewport.point_scatter_item.points()) == 2

    window.points_table_widget.setCurrentCell(1, 0)
    qtbot.waitUntil(lambda: window.image_viewport.current_active_point_id == second_point_id)

    window.image_viewport.active_point_target.setPos((24.5, 23.75))
    window.image_viewport._on_active_point_target_move_finished()
    qtbot.waitUntil(
        lambda: (
            controller.active_row is not None
            and any(
                point.point_id == second_point_id and point.manual_override
                for point in controller.active_row.points
            )
        )
    )

    moved_point = next(point for point in controller.active_row.points if point.point_id == second_point_id)
    assert moved_point.manual_override is True
    assert moved_point.manual_override_source == "drag"
    assert window.points_table_widget.item(1, 11).text() == "manual (drag)"

    window.points_table_widget.setCurrentCell(0, 0)
    qtbot.waitUntil(lambda: window.image_viewport.current_active_point_id == first_point_id)
    qtbot.mouseClick(window.delete_point_button, Qt.MouseButton.LeftButton)

    qtbot.waitUntil(lambda: window.points_table_widget.rowCount() == 1)
    active_row = controller.active_row
    assert active_row is not None
    assert active_row.point_count == 1
    remaining_point = active_row.points[0]
    assert remaining_point.point_id == second_point_id
    assert remaining_point.manual_override is True
    assert len(window.image_viewport.point_scatter_item.points()) == 1


def test_main_window_exports_active_family_points_to_csv(qtbot, monkeypatch, tmp_path: Path):
    controller = AtomMapperController()
    original = _make_gaussian_image("export-family.stp", size=40, amplitude=18.0, offset=1.0)
    controller.set_loaded_images([original])
    variant = controller.create_blur_variant_for_active_image(sigma_px=1.1, make_active=True)
    row = controller.create_row_for_active_source_group(display_name="Row 1")

    original_point = AtomPoint(
        row_id=row.row_id,
        image_id=original.image_id,
        source_group_id=original.source_group_id,
        point_index=0,
        x_px=10.0,
        y_px=11.0,
        x_nm=10.0,
        y_nm=11.0,
        point_id="point-1",
        fit_success=True,
        metadata={
            "fit_method": "gaussian_fit",
            "fit_mask_active": False,
            "fit_mask_pixel_count": None,
        },
    )
    variant_point = AtomPoint(
        row_id=row.row_id,
        image_id=variant.image_id,
        source_group_id=original.source_group_id,
        point_index=1,
        x_px=18.5,
        y_px=19.5,
        x_nm=18.5,
        y_nm=19.5,
        point_id="point-2",
        fit_success=False,
        manual_override=True,
        manual_override_source="drag",
        metadata={
            "fit_model": "lorentzian",
            "fit_method": "lorentzian_fit",
            "fit_mask_active": True,
            "fit_mask_pixel_count": 28,
        },
    )
    controller.add_point_to_row(original_point)
    controller.add_point_to_row(variant_point)

    window = AtomMapperMainWindow(controller=controller)
    qtbot.addWidget(window)

    export_path = tmp_path / "family_points.csv"
    monkeypatch.setattr(
        "AtomMapper.app.main_window.QFileDialog.getSaveFileName",
        lambda *args, **kwargs: (str(export_path), "CSV files (*.csv)"),
    )

    window.export_csv_action.trigger()

    assert export_path.exists()
    with export_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 2
    assert rows[0]["image_name"] == original.display_name
    assert rows[0]["image_variant"] == "original"
    assert rows[0]["point_id"] == "point-1"
    assert rows[0]["previous_point_id"] == ""
    assert rows[0]["next_point_id"] == "point-2"
    assert rows[0]["distance_to_previous_px"] == ""
    assert rows[0]["distance_to_next_px"] == "12.020815"
    assert rows[0]["distance_to_previous_nm"] == ""
    assert rows[0]["distance_to_next_nm"] == "12.020815"
    assert rows[0]["status"] == "fit"
    assert rows[0]["fit_model"] == "gaussian"
    assert rows[0]["fit_method"] == "gaussian_fit"
    assert rows[0]["fit_mask_active"] == "false"
    assert rows[1]["image_name"] == variant.display_name
    assert rows[1]["image_variant"] == "blur"
    assert rows[1]["point_id"] == "point-2"
    assert rows[1]["previous_point_id"] == "point-1"
    assert rows[1]["next_point_id"] == ""
    assert rows[1]["distance_to_previous_px"] == "12.020815"
    assert rows[1]["distance_to_next_px"] == ""
    assert rows[1]["distance_to_previous_nm"] == "12.020815"
    assert rows[1]["distance_to_next_nm"] == ""
    assert rows[1]["manual_override"] == "true"
    assert rows[1]["status"] == "manual (drag)"
    assert rows[1]["x_nm"] == "18.500000"
    assert rows[1]["fit_model"] == "lorentzian"
    assert rows[1]["fit_method"] == "lorentzian_fit"
    assert rows[1]["fit_mask_active"] == "true"
    assert rows[1]["fit_mask_pixel_count"] == "28"
    assert window.statusBar().currentMessage() == "Exported 2 points to family_points.csv."
    assert "exported 2 points to CSV" in window.workflow_status_label.text()
    assert window.image_viewport.current_active_point_id is None
    assert window.points_table_widget.currentItem() is None
    assert window.points_table_widget.item(0, 11).text() == "fit"
    assert window.points_table_widget.item(1, 11).text() == "manual (drag)"


def test_main_window_saves_project_session_to_file(qtbot, monkeypatch, tmp_path: Path):
    controller = AtomMapperController()
    original = _make_gaussian_image("session-save.stp", size=40, amplitude=18.0, offset=1.0)
    controller.set_loaded_images([original])
    controller.update_active_roi_state(ROIState(x=12, y=13, width=10, height=10))
    row = controller.create_row_for_active_source_group(display_name="Row 1")
    controller.add_point_to_row(
        AtomPoint(
            row_id=row.row_id,
            image_id=original.image_id,
            source_group_id=original.source_group_id,
            point_index=0,
            x_px=14.0,
            y_px=15.0,
            point_id="point-1",
        )
    )

    window = AtomMapperMainWindow(controller=controller)
    qtbot.addWidget(window)
    window.points_table_widget.setCurrentCell(0, 0)
    qtbot.waitUntil(lambda: window.image_viewport.current_active_point_id == "point-1")
    window.show_gaussian_fit_checkbox.setChecked(False)
    window.row_plot_widget.metric_combo.setCurrentIndex(
        window.row_plot_widget.metric_combo.findData(RowPlotMode.DISTANCE_PX)
    )
    window.row_metrics_widget.unit_combo.setCurrentIndex(
        window.row_metrics_widget.unit_combo.findData(PlotUnit.NM)
    )
    window.global_scatter_plot_widget.unit_combo.setCurrentIndex(
        window.global_scatter_plot_widget.unit_combo.findData(PlotUnit.NM)
    )
    window.polygon_mask_action.trigger()
    assert window.image_viewport.add_polygon_mask_vertex(12.0, 13.0) is True
    assert window.image_viewport.add_polygon_mask_vertex(22.0, 13.0) is True
    assert window.image_viewport.add_polygon_mask_vertex(22.0, 23.0) is True
    assert window.image_viewport.add_polygon_mask_vertex(12.0, 23.0) is True
    assert window.image_viewport.finish_polygon_mask() is True
    qtbot.waitUntil(lambda: window.image_viewport.current_polygon_mask_state is not None)

    session_path = tmp_path / "session-save.atommapper_proj"
    monkeypatch.setattr(
        "AtomMapper.app.main_window.QFileDialog.getSaveFileName",
        lambda *args, **kwargs: (str(session_path), "AtomMapper project (*.atommapper_proj)"),
    )

    window.save_session_action.trigger()

    assert session_path.exists()
    with session_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    assert payload["version"] == ATOMMAPPER_SESSION_VERSION
    assert payload["active_image_id"] == original.image_id
    assert payload["roi_states_by_image_id"][original.image_id] == {
        "x": 12,
        "y": 13,
        "width": 10,
        "height": 10,
    }
    assert payload["active_row_id_by_source_group"][original.source_group_id] == row.row_id
    assert payload["active_point_id_by_source_group"][original.source_group_id] == "point-1"
    assert payload["view_state"]["show_gaussian_fit"] is False
    assert payload["view_state"]["row_plot_mode"] == "distance_px"
    assert payload["view_state"]["row_plot_unit"] == "px"
    assert payload["view_state"]["row_metrics_unit"] == "nm"
    assert payload["view_state"]["global_scatter_unit"] == "nm"
    assert payload["view_state"]["active_polygon_mask"] is not None
    assert len(payload["view_state"]["active_polygon_mask"]["vertices_xy"]) == 4
    assert len(payload["rows"]) == 1
    assert payload["rows"][0]["points"][0]["point_id"] == "point-1"
    assert window.statusBar().currentMessage() == "Saved session to session-save.atommapper_proj."
    assert "saved project session" in window.workflow_status_label.text()


def test_main_window_loads_project_session_from_file(qtbot, monkeypatch, tmp_path: Path):
    original = _make_gaussian_image("session-load.stp", size=40, amplitude=18.0, offset=1.0)
    source_controller = AtomMapperController()
    source_controller.set_loaded_images([original])
    source_controller.update_active_roi_state(ROIState(x=12, y=13, width=10, height=10))
    row = source_controller.create_row_for_active_source_group(display_name="Row 1")
    source_controller.add_point_to_row(
        AtomPoint(
            row_id=row.row_id,
            image_id=original.image_id,
            source_group_id=original.source_group_id,
            point_index=0,
            x_px=14.0,
            y_px=15.0,
            point_id="point-1",
        )
    )
    variant = source_controller.create_blur_variant_for_active_image(sigma_px=1.25, make_active=True)
    source_controller.update_active_roi_state(ROIState(x=9, y=10, width=11, height=12))
    source_controller.add_point_to_row(
        AtomPoint(
            row_id=row.row_id,
            image_id=variant.image_id,
            source_group_id=variant.source_group_id,
            point_index=1,
            x_px=19.0,
            y_px=21.0,
            point_id="point-2",
        )
    )
    session = build_session_from_runtime(
        source_controller,
        active_point_id_by_source_group={variant.source_group_id: "point-2"},
        view_state=SessionViewState(
            show_gaussian_fit=False,
            row_plot_mode=RowPlotMode.DISTANCE_NM,
            row_plot_unit=PlotUnit.NM,
            row_metrics_unit=PlotUnit.NM,
            global_scatter_unit=PlotUnit.NM,
            active_polygon_mask=PolygonMaskState(
                vertices_xy=((10.0, 11.0), (20.0, 11.0), (20.0, 21.0), (10.0, 21.0))
            ),
        ),
    )
    session_path = tmp_path / "session-load.atommapper_proj"
    save_session_to_file(session_path, session)

    target_controller = AtomMapperController()
    target_controller.set_loaded_images([_make_gaussian_image("other.stp", size=24)])
    window = AtomMapperMainWindow(controller=target_controller)
    qtbot.addWidget(window)

    monkeypatch.setattr(
        "AtomMapper.app.main_window.QFileDialog.getOpenFileName",
        lambda *args, **kwargs: (str(session_path), "AtomMapper project (*.atommapper_proj)"),
    )

    window.load_session_action.trigger()

    qtbot.waitUntil(
        lambda: (
            window.controller.active_image is not None
            and window.controller.active_image.image_id == variant.image_id
            and window.points_table_widget.rowCount() == 2
        )
    )

    assert len(window.controller.loaded_images) == 2
    assert window.controller.active_image is not None
    assert window.controller.active_image.image_id == variant.image_id
    assert window.controller.active_roi_state == ROIState(x=9, y=10, width=11, height=12)
    assert window.controller.active_row is not None
    assert window.controller.active_row.row_id == row.row_id
    assert window.image_viewport.current_active_point_id == "point-2"
    assert window.points_table_widget.currentItem() is not None
    assert window.points_table_widget.currentItem().data(Qt.ItemDataRole.UserRole) == "point-2"
    assert window.points_table_widget.item(0, 11).text() == "fit"
    assert window.points_table_widget.item(1, 11).text() == "fit"
    assert window.show_gaussian_fit_checkbox.isChecked() is False
    assert window.gaussian_fit_preview.isHidden()
    assert window.row_plot_widget.current_mode is RowPlotMode.DISTANCE_NM
    assert window.row_plot_widget.current_unit is PlotUnit.NM
    assert window.row_metrics_widget.current_unit is PlotUnit.NM
    assert window.global_scatter_plot_widget.current_unit is PlotUnit.NM
    assert window.preview_bridge.current_polygon_mask_state is not None
    assert window.image_viewport.current_polygon_mask_state is not None
    assert len(window.image_viewport.current_polygon_mask_state.vertices_xy) == 4
    assert window.clear_polygon_mask_action.isEnabled() is True
    assert len(window.image_viewport.point_scatter_item.points()) == 2
    assert window.file_list_widget.count() == 2
    assert window.statusBar().currentMessage() == "Loaded session from session-load.atommapper_proj."
    assert "loaded project session" in window.workflow_status_label.text()


def test_tools_action_recalculates_uncertainties_and_refreshes_saved_points_table(qtbot):
    controller = AtomMapperController()
    image = _make_gaussian_image("uncertainty-action.stp", size=40)
    controller.set_loaded_images([image])
    row = controller.create_row_for_active_source_group(display_name="Row 1")
    controller.add_point_to_row(
        AtomPoint(
            point_id="point-1",
            row_id=row.row_id,
            image_id=image.image_id,
            source_group_id=image.source_group_id,
            point_index=0,
            x_px=20.0,
            y_px=20.0,
            sigma_x_px=1.8,
            sigma_y_px=1.6,
            metadata={
                "fit_model": "gaussian",
                "fit_method": "gaussian_fit",
                "roi_x": 14,
                "roi_y": 14,
                "roi_width": 13,
                "roi_height": 13,
                "fit_mask_active": False,
            },
        )
    )
    window = AtomMapperMainWindow(controller=controller)
    qtbot.addWidget(window)

    tools_menu = next(
        action.menu() for action in window.menuBar().actions() if action.text() == "Tools"
    )
    assert tools_menu is not None
    assert window.export_csv_action in tools_menu.actions()
    assert window.export_csv_action.text() == "Export results table to CSV"
    assert window.recalculate_position_uncertainties_action in tools_menu.actions()
    assert window.recalculate_position_uncertainties_action.isEnabled() is True

    window.recalculate_position_uncertainties_action.trigger()

    updated = controller.atom_rows[0].points[0]
    assert updated.position_std_x_px is not None
    assert updated.position_std_y_px is not None
    assert window.points_table_widget.horizontalHeaderItem(6).text() == "position_std_x_px"
    assert window.points_table_widget.horizontalHeaderItem(7).text() == "position_std_y_px"
    assert window.points_table_widget.horizontalHeaderItem(8).text() == "position_std_x_nm"
    assert window.points_table_widget.horizontalHeaderItem(9).text() == "position_std_y_nm"
    assert window.points_table_widget.item(0, 6).text()
    assert window.points_table_widget.item(0, 7).text()
    assert window.points_table_widget.item(0, 8).text()
    assert window.points_table_widget.item(0, 9).text()
    assert window.points_table_widget.item(0, 10).text() == "recomputed (session settings)"
    assert window.points_table_widget.item(0, 11).text() == "fit"
    assert "Recalculated position uncertainties for 1 point" in (
        window.statusBar().currentMessage()
    )


def test_main_window_can_open_preprocessing_dialog_for_active_image(qtbot):
    controller = AtomMapperController()
    original = _make_gaussian_image("origin.stp", size=40, amplitude=18.0, offset=1.0)
    controller.set_loaded_images([original])

    window = AtomMapperMainWindow(controller=controller)
    qtbot.addWidget(window)

    opened: dict[str, object] = {}

    class FakeDialog:
        def __init__(self, loaded_image, parent):
            opened["image"] = loaded_image
            opened["parent"] = parent

        def exec(self):
            return int(QDialog.DialogCode.Rejected)

    window._preprocessing_dialog_class = FakeDialog

    assert window.preprocessing_action.isEnabled()

    window.preprocessing_action.trigger()

    assert opened["image"] == original
    assert opened["parent"] == window
    assert len(window.controller.loaded_images) == 1
    assert window.controller.active_image == original
    assert window.statusBar().currentMessage() == "Preprocessing cancelled."
    assert "closed without changes" in window.workflow_status_label.text()


def test_main_window_real_preprocessing_dialog_cancel_keeps_state(qtbot):
    controller = AtomMapperController()
    original = _make_gaussian_image("origin.stp", size=40, amplitude=18.0, offset=1.0)
    controller.set_loaded_images([original])

    window = AtomMapperMainWindow(controller=controller)
    qtbot.addWidget(window)

    class AutoCancelDialog(PreprocessingDialog):
        def __init__(self, loaded_image, parent):
            super().__init__(loaded_image, parent)
            QTimer.singleShot(0, self.reject)

    window._preprocessing_dialog_class = AutoCancelDialog

    window.preprocessing_action.trigger()

    assert len(window.controller.loaded_images) == 1
    assert window.controller.active_image == original
    assert window.statusBar().currentMessage() == "Preprocessing cancelled."
    assert "closed without changes" in window.workflow_status_label.text()


def test_main_window_applies_blur_variant_from_preprocessing_dialog(qtbot):
    controller = AtomMapperController()
    original = _make_gaussian_image("origin.stp", size=40, amplitude=18.0, offset=1.0)
    controller.set_loaded_images([original])

    window = AtomMapperMainWindow(controller=controller)
    qtbot.addWidget(window)

    class FakeDialog:
        def __init__(self, loaded_image, parent):
            self.preprocessing_state = PreprocessingState(
                method=PreprocessingMethod.BLUR,
                blur=BlurParameters(sigma_px=1.75),
            )

        def exec(self):
            return int(QDialog.DialogCode.Accepted)

    window._preprocessing_dialog_class = FakeDialog

    window.preprocessing_action.trigger()

    assert len(window.controller.loaded_images) == 2
    variant = window.controller.active_image
    assert variant is not None
    assert variant is not original
    assert variant.variant_name == "blur"
    assert variant.metadata["preprocess"] == "blur"
    assert variant.metadata["blur_sigma_px"] == pytest.approx(1.75)
    assert window.file_list_widget.count() == 2
    assert window.file_list_widget.item(1).text() == f"  - {variant.display_name}"
    assert "Created blur variant" in window.statusBar().currentMessage()
    assert "created blur variant" in window.workflow_status_label.text()


def test_main_window_real_preprocessing_dialog_apply_blur_creates_variant(qtbot):
    controller = AtomMapperController()
    original = _make_gaussian_image("origin.stp", size=40, amplitude=18.0, offset=1.0)
    controller.set_loaded_images([original])

    window = AtomMapperMainWindow(controller=controller)
    qtbot.addWidget(window)

    class AutoApplyBlurDialog(PreprocessingDialog):
        def __init__(self, loaded_image, parent):
            super().__init__(loaded_image, parent)
            self.blur_sigma_spinbox.setValue(1.35)
            QTimer.singleShot(0, self.apply_button.click)

    window._preprocessing_dialog_class = AutoApplyBlurDialog

    window.preprocessing_action.trigger()

    assert len(window.controller.loaded_images) == 2
    variant = window.controller.active_image
    assert variant is not None
    assert variant is not original
    assert variant.variant_name == "blur"
    assert variant.metadata["preprocess"] == "blur"
    assert variant.metadata["blur_sigma_px"] == pytest.approx(1.35)
    assert window.file_list_widget.item(1).text() == f"  - {variant.display_name}"
    assert "Created blur variant" in window.statusBar().currentMessage()
    assert "created blur variant" in window.workflow_status_label.text()


def test_main_window_applies_nlm_variant_from_preprocessing_dialog(qtbot):
    controller = AtomMapperController()
    original = _make_gaussian_image("origin.stp", size=40, amplitude=18.0, offset=1.0)
    controller.set_loaded_images([original])

    window = AtomMapperMainWindow(controller=controller)
    qtbot.addWidget(window)

    class FakeDialog:
        def __init__(self, loaded_image, parent):
            self.preprocessing_state = PreprocessingState(
                method=PreprocessingMethod.NLM,
                blur=BlurParameters(),
                nlm=NonLocalMeansParameters(
                    h=0.18,
                    patch_size=7,
                    patch_distance=8,
                    fast_mode=False,
                ),
            )

        def exec(self):
            return int(QDialog.DialogCode.Accepted)

    window._preprocessing_dialog_class = FakeDialog

    window.preprocessing_action.trigger()

    assert len(window.controller.loaded_images) == 2
    variant = window.controller.active_image
    assert variant is not None
    assert variant is not original
    assert variant.variant_name == "nlm"
    assert variant.metadata["preprocess"] == "nlm"
    assert variant.metadata["nlm_h"] == pytest.approx(0.18)
    assert variant.metadata["nlm_patch_size"] == 7
    assert variant.metadata["nlm_patch_distance"] == 8
    assert variant.metadata["nlm_fast_mode"] is False
    assert window.file_list_widget.count() == 2
    assert window.file_list_widget.item(1).text() == f"  - {variant.display_name}"
    assert "Created nlm variant" in window.statusBar().currentMessage()
    assert "created nlm variant" in window.workflow_status_label.text()


def test_main_window_applies_bm3d_variant_from_preprocessing_dialog(qtbot):
    if not is_bm3d_available():
        pytest.skip("bm3d package not available in test environment")

    controller = AtomMapperController()
    original = _make_gaussian_image("origin.stp", size=24, amplitude=18.0, offset=1.0)
    controller.set_loaded_images([original])

    window = AtomMapperMainWindow(controller=controller)
    qtbot.addWidget(window)

    class FakeDialog:
        def __init__(self, loaded_image, parent):
            self.preprocessing_state = PreprocessingState(
                method=PreprocessingMethod.BM3D,
                bm3d=BM3DParameters(sigma_psd=0.07, stage="all_stages"),
            )

        def exec(self):
            return int(QDialog.DialogCode.Accepted)

    window._preprocessing_dialog_class = FakeDialog

    window.preprocessing_action.trigger()

    assert len(window.controller.loaded_images) == 2
    variant = window.controller.active_image
    assert variant is not None
    assert variant is not original
    assert variant.variant_name == "bm3d"
    assert variant.metadata["preprocess"] == "bm3d"
    assert variant.metadata["bm3d_sigma_psd"] == pytest.approx(0.07)
    assert variant.metadata["bm3d_stage"] == "all_stages"
    assert window.file_list_widget.count() == 2
    assert window.file_list_widget.item(1).text() == f"  - {variant.display_name}"
    assert "Created bm3d variant" in window.statusBar().currentMessage()
    assert "created bm3d variant" in window.workflow_status_label.text()


def test_main_window_applies_rotate_variant_from_preprocessing_dialog(qtbot):
    controller = AtomMapperController()
    original = _make_gaussian_image("origin.stp", size=24, amplitude=18.0, offset=1.0)
    controller.set_loaded_images([original])

    window = AtomMapperMainWindow(controller=controller)
    qtbot.addWidget(window)

    class FakeDialog:
        def __init__(self, loaded_image, parent):
            self.preprocessing_state = PreprocessingState(
                method=PreprocessingMethod.ROTATE,
                rotate=RotateParameters(quarter_turns=1),
            )

        def exec(self):
            return int(QDialog.DialogCode.Accepted)

    window._preprocessing_dialog_class = FakeDialog

    window.preprocessing_action.trigger()

    assert len(window.controller.loaded_images) == 2
    variant = window.controller.active_image
    assert variant is not None
    assert variant is not original
    assert variant.variant_name == "rotate-90"
    assert variant.metadata["preprocess"] == "rotate"
    assert variant.metadata["rotate_quarter_turns"] == 1
    assert variant.pixels_x == original.pixels_y
    assert variant.pixels_y == original.pixels_x
    assert "Created rotate-90 variant" in window.statusBar().currentMessage()
    assert "created rotate-90 variant" in window.workflow_status_label.text()


def test_main_window_applies_flip_variant_from_preprocessing_dialog(qtbot):
    controller = AtomMapperController()
    original = _make_gaussian_image("origin.stp", size=24, amplitude=18.0, offset=1.0)
    controller.set_loaded_images([original])

    window = AtomMapperMainWindow(controller=controller)
    qtbot.addWidget(window)

    class FakeDialog:
        def __init__(self, loaded_image, parent):
            self.preprocessing_state = PreprocessingState(
                method=PreprocessingMethod.FLIP,
                flip=FlipParameters(flip_x=False, flip_y=True),
            )

        def exec(self):
            return int(QDialog.DialogCode.Accepted)

    window._preprocessing_dialog_class = FakeDialog

    window.preprocessing_action.trigger()

    assert len(window.controller.loaded_images) == 2
    variant = window.controller.active_image
    assert variant is not None
    assert variant is not original
    assert variant.variant_name == "flip-y"
    assert variant.metadata["preprocess"] == "flip"
    assert variant.metadata["flip_x"] is False
    assert variant.metadata["flip_y"] is True
    assert "Created flip-y variant" in window.statusBar().currentMessage()
    assert "created flip-y variant" in window.workflow_status_label.text()


def test_main_window_row_panel_can_create_select_and_delete_rows(qtbot):
    controller = AtomMapperController()
    original = _make_gaussian_image("rows.stp", size=40, amplitude=18.0, offset=1.0)
    controller.set_loaded_images([original])

    window = AtomMapperMainWindow(controller=controller)
    qtbot.addWidget(window)

    assert window.row_list_widget.count() == 0
    assert window.active_row_label.text() == "Active row: none"
    assert window.new_row_button.isEnabled()
    assert not window.delete_row_button.isEnabled()

    qtbot.mouseClick(window.new_row_button, Qt.MouseButton.LeftButton)

    first_row = controller.active_row
    assert first_row is not None
    assert len(controller.atom_rows) == 1
    assert window.row_list_widget.count() == 1
    assert window.row_list_widget.item(0).text() == "Row 1 (0 points)"
    assert "Row 1" in window.active_row_label.text()
    assert window.delete_row_button.isEnabled()
    assert window.statusBar().currentMessage() == "Created Row 1."

    qtbot.mouseClick(window.new_row_button, Qt.MouseButton.LeftButton)

    second_row = controller.active_row
    assert second_row is not None
    assert second_row.row_id != first_row.row_id
    assert len(controller.atom_rows) == 2
    assert window.row_list_widget.count() == 2
    assert window.row_list_widget.item(1).text() == "Row 2 (0 points)"
    assert "Row 2" in window.active_row_label.text()

    window.row_list_widget.setCurrentRow(0)

    assert controller.active_row is not None
    assert controller.active_row.row_id == first_row.row_id
    assert "Row 1" in window.active_row_label.text()
    assert window.statusBar().currentMessage() == "Selected Row 1."

    qtbot.mouseClick(window.delete_row_button, Qt.MouseButton.LeftButton)

    assert len(controller.atom_rows) == 1
    assert controller.active_row is not None
    assert controller.active_row.row_id == second_row.row_id
    assert window.row_list_widget.count() == 1
    assert window.row_list_widget.item(0).text() == "Row 2 (0 points)"
    assert "Row 2" in window.active_row_label.text()
    assert window.statusBar().currentMessage() == "Deleted Row 1 (0 points)."


def test_main_window_add_point_uses_gaussian_fit_for_active_row(qtbot):
    controller = AtomMapperController()
    image = _make_gaussian_image("point-fit.stp", size=40, amplitude=22.0, offset=1.0)
    controller.set_loaded_images([image])

    window = AtomMapperMainWindow(controller=controller)
    qtbot.addWidget(window)

    qtbot.mouseClick(window.new_row_button, Qt.MouseButton.LeftButton)
    assert controller.active_row is not None

    qtbot.mouseClick(window.add_point_button, Qt.MouseButton.LeftButton)

    active_row = controller.active_row
    assert active_row is not None
    assert active_row.point_count == 1
    point = active_row.points[0]
    assert point.image_id == image.image_id
    assert point.source_group_id == image.source_group_id
    assert point.fit_success is True
    assert point.metadata["fallback_used"] is False
    assert point.metadata["fit_method"] == "gaussian_fit"
    assert point.amplitude is not None
    assert point.sigma_x_px is not None
    assert point.sigma_y_px is not None
    assert point.offset is not None
    assert window.recalculate_position_uncertainties_action.isEnabled() is True
    assert window.row_list_widget.item(0).text() == "Row 1 (1 point)"
    assert "from Gaussian fit" in window.workflow_status_label.text()
    assert window.image_viewport.point_scatter_item is not None
    assert len(window.image_viewport.point_scatter_item.points()) == 1
    first_spot = window.image_viewport.point_scatter_item.points()[0]
    assert (first_spot.pos().x(), first_spot.pos().y()) == pytest.approx(
        (point.x_px, point.y_px)
    )


def test_main_window_polygon_mask_limits_fit_and_is_saved_with_point_metadata(qtbot):
    controller = AtomMapperController()
    image = _make_gaussian_image("point-mask.stp", size=40, amplitude=22.0, offset=1.0)
    controller.set_loaded_images([image])

    window = AtomMapperMainWindow(controller=controller)
    qtbot.addWidget(window)

    qtbot.mouseClick(window.new_row_button, Qt.MouseButton.LeftButton)
    window.polygon_mask_action.trigger()

    assert window.image_viewport.add_polygon_mask_vertex(16.0, 16.0) is True
    assert window.image_viewport.add_polygon_mask_vertex(24.0, 16.0) is True
    assert window.image_viewport.add_polygon_mask_vertex(24.0, 24.0) is True
    assert window.image_viewport.add_polygon_mask_vertex(16.0, 24.0) is True
    assert window.image_viewport.finish_polygon_mask() is True
    qtbot.waitUntil(lambda: window.image_viewport.current_polygon_mask_state is not None)

    fit_result = window.preview_bridge.compute_current_fit_result()
    assert fit_result is not None
    assert fit_result.fit_mask is not None
    assert int(fit_result.fit_mask.sum()) > 0
    assert fit_result.raw_result is not None
    assert fit_result.raw_result.metadata["fit_mask_pixel_count"] == int(fit_result.fit_mask.sum())

    qtbot.mouseClick(window.add_point_button, Qt.MouseButton.LeftButton)

    active_row = controller.active_row
    assert active_row is not None
    point = active_row.points[0]
    assert point.metadata["fit_mask_active"] is True
    assert point.metadata["fit_mask_pixel_count"] == int(fit_result.fit_mask.sum())

    window.clear_polygon_mask_action.trigger()

    assert window.preview_bridge.current_polygon_mask_state is None
    assert window.image_viewport.current_polygon_mask_state is None


def test_main_window_end_to_end_stage7_fit_model_mask_workflow_with_session_restore(
    qtbot, monkeypatch, tmp_path: Path
):
    controller = AtomMapperController()
    image = _make_overlapping_lorentzian_image("stage7-overlap.stp")
    controller.set_loaded_images([image])
    controller.update_active_roi_state(ROIState(x=14, y=16, width=20, height=16))

    window = AtomMapperMainWindow(controller=controller)
    qtbot.addWidget(window)
    qtbot.mouseClick(window.new_row_button, Qt.MouseButton.LeftButton)
    window.fit_settings_action.trigger()

    model_index = window.fit_settings_panel.model_combo.findData(LocalFitModelType.LORENTZIAN)
    window.fit_settings_panel.model_combo.setCurrentIndex(model_index)

    max_nfev_spinbox = window.fit_settings_panel.findChild(
        QSpinBox,
        "atommapper_fit_common_max_nfev_spinbox",
    )
    assert max_nfev_spinbox is not None
    max_nfev_spinbox.setValue(3400)

    custom_initial_guess_checkbox = window.fit_settings_panel.findChild(
        QCheckBox,
        "atommapper_fit_common_use_custom_initial_guess_checkbox",
    )
    assert custom_initial_guess_checkbox is not None
    custom_initial_guess_checkbox.setChecked(True)

    gamma_x_edit = window.fit_settings_panel.findChild(
        QLineEdit,
        "atommapper_fit_lorentzian_gamma_x_init_lineedit",
    )
    assert gamma_x_edit is not None
    gamma_x_edit.setText("1.8")
    gamma_x_edit.editingFinished.emit()

    assert window.fit_settings_state.model is LocalFitModelType.LORENTZIAN
    assert window.fit_settings_state.common.max_nfev == 3400
    assert window.fit_settings_state.lorentzian.gamma_x_init == 1.8
    assert window.fit_settings_dock.isHidden() is False

    unmasked_result = window.preview_bridge.compute_current_fit_result()
    assert unmasked_result is not None
    assert unmasked_result.success is True
    assert unmasked_result.center_image_yx is not None
    assert unmasked_result.center_image_yx[1] > 24.5

    window.polygon_mask_action.trigger()
    assert window.image_viewport.add_polygon_mask_vertex(15.5, 18.0) is True
    assert window.image_viewport.add_polygon_mask_vertex(24.0, 18.0) is True
    assert window.image_viewport.add_polygon_mask_vertex(24.0, 30.0) is True
    assert window.image_viewport.add_polygon_mask_vertex(15.5, 30.0) is True
    assert window.image_viewport.finish_polygon_mask() is True
    qtbot.waitUntil(lambda: window.image_viewport.current_polygon_mask_state is not None)

    masked_result = window.preview_bridge.compute_current_fit_result()
    assert masked_result is not None
    assert masked_result.model is LocalFitModelType.LORENTZIAN
    assert masked_result.success is True
    assert masked_result.center_image_yx is not None
    assert masked_result.fit_mask is not None
    assert masked_result.center_image_yx[1] < 24.0
    assert "Lorentzian" in window.gaussian_fit_preview.title_label.text()
    assert "mask=" in window.gaussian_fit_preview.info_label.text()

    qtbot.mouseClick(window.add_point_button, Qt.MouseButton.LeftButton)

    active_row = controller.active_row
    assert active_row is not None
    point = active_row.points[0]
    assert point.fit_success is True
    assert point.metadata["fit_model"] == "lorentzian"
    assert point.metadata["fit_method"] == "lorentzian_fit"
    assert point.metadata["fit_mask_active"] is True
    assert int(point.metadata["fit_mask_pixel_count"]) > 0
    assert "from Lorentzian fit" in window.workflow_status_label.text()

    session_path = tmp_path / "stage7-overlap.atommapper_proj"
    monkeypatch.setattr(
        "AtomMapper.app.main_window.QFileDialog.getSaveFileName",
        lambda *args, **kwargs: (str(session_path), "AtomMapper project (*.atommapper_proj)"),
    )
    window.save_session_action.trigger()
    assert session_path.exists()

    restored_controller = AtomMapperController()
    restored_window = AtomMapperMainWindow(controller=restored_controller)
    qtbot.addWidget(restored_window)
    monkeypatch.setattr(
        "AtomMapper.app.main_window.QFileDialog.getOpenFileName",
        lambda *args, **kwargs: (str(session_path), "AtomMapper project (*.atommapper_proj)"),
    )
    restored_window.load_session_action.trigger()

    qtbot.waitUntil(
        lambda: (
            restored_window.controller.active_image is not None
            and restored_window.preview_bridge.current_polygon_mask_state is not None
            and restored_window.controller.active_row is not None
            and restored_window.controller.active_row.point_count == 1
        )
    )

    assert restored_window.fit_settings_state.model is LocalFitModelType.LORENTZIAN
    assert restored_window.fit_settings_state.common.max_nfev == 3400
    assert restored_window.fit_settings_state.lorentzian.gamma_x_init == 1.8
    assert restored_window.preview_bridge.current_polygon_mask_state is not None
    assert restored_window.image_viewport.current_polygon_mask_state is not None
    assert restored_window.gaussian_fit_preview.current_fit_result is not None
    assert restored_window.gaussian_fit_preview.current_fit_result.model is LocalFitModelType.LORENTZIAN
    assert restored_window.gaussian_fit_preview.current_fit_result.fit_mask is not None
    assert restored_window.gaussian_fit_preview.current_fit_result.center_image_yx is not None
    assert restored_window.gaussian_fit_preview.current_fit_result.center_image_yx[1] < 24.0
    assert restored_window.points_table_widget.rowCount() == 1
    assert restored_window.points_table_widget.item(0, 11).text() == "fit"
    assert "loaded project session" in restored_window.workflow_status_label.text()


def test_main_window_add_point_falls_back_to_roi_center_when_fit_is_unavailable(qtbot):
    controller = AtomMapperController()
    image = _make_loaded_image(
        "point-fallback.stp",
        np.zeros((20, 20), dtype=float),
    )
    controller.set_loaded_images([image])

    window = AtomMapperMainWindow(controller=controller)
    qtbot.addWidget(window)
    qtbot.mouseClick(window.new_row_button, Qt.MouseButton.LeftButton)

    controller.update_active_roi_state(ROIState(x=2, y=3, width=4, height=4))

    qtbot.mouseClick(window.add_point_button, Qt.MouseButton.LeftButton)

    active_row = controller.active_row
    assert active_row is not None
    assert active_row.point_count == 1
    point = active_row.points[0]
    assert point.fit_success is False
    assert point.metadata["fallback_used"] is True
    assert str(point.metadata["fit_method"]).endswith("_fallback")
    assert point.metadata["roi_x"] == 2
    assert point.metadata["roi_y"] == 3
    assert point.x_px == pytest.approx(4.0)
    assert point.y_px == pytest.approx(5.0)
    assert point.x_nm == pytest.approx(4.0)
    assert point.y_nm == pytest.approx(5.0)
    assert "ROI center fallback" in window.workflow_status_label.text()
    assert window.row_list_widget.item(0).text() == "Row 1 (1 point)"


def test_main_window_add_point_without_active_row_does_not_crash(qtbot):
    controller = AtomMapperController()
    image = _make_gaussian_image("point-no-row.stp", size=40, amplitude=18.0, offset=1.0)
    controller.set_loaded_images([image])

    window = AtomMapperMainWindow(controller=controller)
    qtbot.addWidget(window)

    window._add_point_from_current_roi()

    assert controller.active_row is None
    assert "Create or select an atom row" in window.statusBar().currentMessage()
    assert "no active row selected" in window.workflow_status_label.text()


def test_main_window_points_table_refreshes_after_add_and_row_delete(qtbot):
    controller = AtomMapperController()
    image = _make_gaussian_image("points-table.stp", size=40, amplitude=20.0, offset=1.0)
    controller.set_loaded_images([image])

    window = AtomMapperMainWindow(controller=controller)
    qtbot.addWidget(window)

    assert window.points_table_widget.rowCount() == 0
    assert "No saved points" in window.points_table_hint_label.text()

    qtbot.mouseClick(window.new_row_button, Qt.MouseButton.LeftButton)
    qtbot.mouseClick(window.add_point_button, Qt.MouseButton.LeftButton)

    assert window.points_table_widget.rowCount() == 1
    assert window.points_table_widget.item(0, 0).text() == "Row 1"
    assert window.points_table_widget.item(0, 1).text() == "0"
    assert window.points_table_widget.item(0, 2).text() != "-"
    assert window.points_table_widget.item(0, 3).text() != "-"
    assert window.points_table_widget.item(0, 4).text() != "-"
    assert window.points_table_widget.item(0, 5).text() != "-"
    assert "Showing 1 saved point" in window.points_table_hint_label.text()

    qtbot.mouseClick(window.delete_row_button, Qt.MouseButton.LeftButton)

    assert window.points_table_widget.rowCount() == 0
    assert "No saved points" in window.points_table_hint_label.text()
    assert window.statusBar().currentMessage() == "Deleted Row 1 (1 point)."


def test_main_window_analysis_dock_refreshes_plots_and_metrics(qtbot):
    controller = AtomMapperController()
    image = _make_gaussian_image("analysis-dock.stp", size=40, amplitude=20.0, offset=1.0)
    controller.set_loaded_images([image])

    window = AtomMapperMainWindow(controller=controller)
    qtbot.addWidget(window)

    assert window.analysis_dock.widget() == window.analysis_dock_content
    assert window.row_plot_widget.current_row is None
    assert window.global_scatter_plot_widget.current_series is not None
    assert len(window.global_scatter_plot_widget.current_series.samples) == 0
    assert window.row_metrics_widget.current_metrics is None

    qtbot.mouseClick(window.new_row_button, Qt.MouseButton.LeftButton)
    active_row = controller.active_row
    assert active_row is not None
    assert window.row_plot_widget.current_row is not None
    assert window.row_plot_widget.current_row.row_id == active_row.row_id
    assert window.row_plot_widget.stack.currentWidget() is window.row_plot_widget.placeholder_label
    assert window.row_metrics_widget.current_metrics is not None
    assert window.row_metrics_widget.stack.currentWidget() is window.row_metrics_widget.placeholder_label

    controller.update_active_roi_state(ROIState(x=10, y=10, width=12, height=12))
    qtbot.mouseClick(window.add_point_button, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: window.points_table_widget.rowCount() == 1)

    assert window.row_plot_widget.current_row is not None
    assert window.row_plot_widget.current_row.point_count == 1
    assert window.row_plot_widget.stack.currentWidget() is window.row_plot_widget.plot_widget
    assert window.global_scatter_plot_widget.current_series is not None
    assert len(window.global_scatter_plot_widget.current_series.samples) == 1
    assert window.global_scatter_plot_widget.stack.currentWidget() is window.global_scatter_plot_widget.plot_widget
    assert window.row_metrics_widget.current_metrics is not None
    assert window.row_metrics_widget.current_metrics.distance_count == 0
    assert window.row_metrics_widget.stack.currentWidget() is window.row_metrics_widget.placeholder_label

    controller.update_active_roi_state(ROIState(x=18, y=18, width=12, height=12))
    qtbot.mouseClick(window.add_point_button, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: window.points_table_widget.rowCount() == 2)

    assert window.row_plot_widget.current_row is not None
    assert window.row_plot_widget.current_row.point_count == 2
    assert window.global_scatter_plot_widget.current_series is not None
    assert len(window.global_scatter_plot_widget.current_series.samples) == 2
    assert window.row_metrics_widget.current_metrics is not None
    assert window.row_metrics_widget.current_metrics.distance_count == 1
    assert window.row_metrics_widget.stack.currentWidget() is window.row_metrics_widget.metrics_panel

    window.global_scatter_plot_widget.unit_combo.setCurrentIndex(
        window.global_scatter_plot_widget.unit_combo.findData(PlotUnit.NM)
    )
    qtbot.waitUntil(
        lambda: (
            window.global_scatter_plot_widget.current_series is not None
            and window.global_scatter_plot_widget.current_series.unit is PlotUnit.NM
        )
    )
    assert window.global_scatter_plot_widget.current_series.x_label == "x (nm)"
    assert window.global_scatter_plot_widget.current_series.y_label == "y (nm)"


def test_main_window_stage6_refreshes_row_geometry_overlay_and_disturbance_panel(qtbot):
    controller = AtomMapperController()
    image = _make_gaussian_image("stage6-overlay.stp", size=40, amplitude=20.0, offset=1.0)
    controller.set_loaded_images([image])

    window = AtomMapperMainWindow(controller=controller)
    qtbot.addWidget(window)

    qtbot.mouseClick(window.new_row_button, Qt.MouseButton.LeftButton)
    active_row = controller.active_row
    assert active_row is not None

    for point_index, x_value in enumerate((0.0, 1.0, 2.0, 5.0, 6.0)):
        controller.add_point_to_row(
            AtomPoint(
                row_id=active_row.row_id,
                image_id=image.image_id,
                source_group_id=image.source_group_id,
                point_index=point_index,
                x_px=x_value,
                y_px=0.0,
                point_id=f"point-{point_index}",
            )
        )

    qtbot.waitUntil(
        lambda: (
            window.row_disturbance_widget.current_series is not None
            and window.image_viewport.row_axis_item is not None
            and window.image_viewport.row_axis_item.isVisible()
        )
    )

    assert window.row_disturbance_widget.stack.currentWidget() is window.row_disturbance_widget.summary_panel
    assert window.row_disturbance_widget.current_series is not None
    assert window.row_disturbance_widget.current_series.candidate_count >= 1
    assert window.image_viewport.row_disturbance_scatter_item is not None
    assert len(window.image_viewport.row_disturbance_scatter_item.points()) >= 1
    assert "candidate" in window.active_row_label.text()

    qtbot.mouseClick(window.delete_row_button, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(
        lambda: (
            window.row_disturbance_widget.current_series is None
            and window.image_viewport.row_axis_item is not None
            and not window.image_viewport.row_axis_item.isVisible()
        )
    )

    assert window.row_disturbance_widget.stack.currentWidget() is window.row_disturbance_widget.placeholder_label
    assert window.image_viewport.row_disturbance_scatter_item is not None
    assert not window.image_viewport.row_disturbance_scatter_item.isVisible()


def test_main_window_end_to_end_stage6_geometry_workflow_with_session_restore(
    qtbot, monkeypatch, tmp_path: Path
):
    controller = AtomMapperController()
    image = _make_gaussian_image("stage6-session.stp", size=40, amplitude=20.0, offset=1.0)
    controller.set_loaded_images([image])

    window = AtomMapperMainWindow(controller=controller)
    qtbot.addWidget(window)

    qtbot.mouseClick(window.new_row_button, Qt.MouseButton.LeftButton)
    active_row = controller.active_row
    assert active_row is not None

    for point_index, x_value in enumerate((0.0, 1.0, 2.0, 5.0, 6.0)):
        controller.add_point_to_row(
            AtomPoint(
                row_id=active_row.row_id,
                image_id=image.image_id,
                source_group_id=image.source_group_id,
                point_index=point_index,
                x_px=x_value,
                y_px=0.0,
                point_id=f"point-{point_index}",
            )
        )

    qtbot.waitUntil(
        lambda: (
            window.row_disturbance_widget.current_series is not None
            and window.image_viewport.row_axis_item is not None
            and window.image_viewport.row_axis_item.isVisible()
        )
    )

    window.row_plot_widget.unit_combo.setCurrentIndex(
        window.row_plot_widget.unit_combo.findData(PlotUnit.NM)
    )
    window.row_plot_widget.metric_combo.setCurrentIndex(
        window.row_plot_widget.metric_combo.findData(RowPlotMode.ALONG_PX)
    )
    qtbot.waitUntil(
        lambda: (
            window.row_plot_widget.current_series is not None
            and window.row_plot_widget.current_series.mode is RowPlotMode.ALONG_NM
        )
    )

    window.row_metrics_widget.unit_combo.setCurrentIndex(
        window.row_metrics_widget.unit_combo.findData(PlotUnit.NM)
    )
    qtbot.waitUntil(
        lambda: (
            window.row_disturbance_widget.current_series is not None
            and window.row_disturbance_widget.current_unit is PlotUnit.NM
            and window.row_disturbance_widget.current_series.unit.value == "nm"
        )
    )

    point_lookup = {point.point_id: point for point in controller.active_row.points}
    move_source_point = point_lookup["point-2"]
    window._handle_viewport_point_move_requested(
        {
            "row_id": active_row.row_id,
            "point_id": move_source_point.point_id,
            "image_id": move_source_point.image_id,
            "x_px": 2.0,
            "y_px": 1.5,
            "source": "drag",
        }
    )
    qtbot.waitUntil(
        lambda: (
            window.points_table_widget.rowCount() == 5
            and any(
                window.points_table_widget.item(row_index, 11).text() == "manual (drag)"
                for row_index in range(window.points_table_widget.rowCount())
            )
        )
    )

    assert window.row_plot_widget.current_mode is RowPlotMode.ALONG_NM
    assert window.row_metrics_widget.current_unit is PlotUnit.NM
    assert window.row_disturbance_widget.current_unit is PlotUnit.NM
    assert window.global_scatter_plot_widget.current_unit is PlotUnit.PX
    assert "candidate" in window.active_row_label.text() or "no local candidates" in window.active_row_label.text()

    session_path = tmp_path / "stage6-session.atommapper_proj"
    monkeypatch.setattr(
        "AtomMapper.app.main_window.QFileDialog.getSaveFileName",
        lambda *args, **kwargs: (str(session_path), "AtomMapper project (*.atommapper_proj)"),
    )
    window.save_session_action.trigger()
    assert session_path.exists()

    restored_controller = AtomMapperController()
    restored_window = AtomMapperMainWindow(controller=restored_controller)
    qtbot.addWidget(restored_window)

    monkeypatch.setattr(
        "AtomMapper.app.main_window.QFileDialog.getOpenFileName",
        lambda *args, **kwargs: (str(session_path), "AtomMapper project (*.atommapper_proj)"),
    )
    restored_window.load_session_action.trigger()

    qtbot.waitUntil(
        lambda: (
            restored_window.controller.active_row is not None
            and restored_window.row_plot_widget.current_series is not None
            and restored_window.row_disturbance_widget.current_series is not None
            and restored_window.points_table_widget.rowCount() == 5
        )
    )

    assert restored_window.row_plot_widget.current_mode is RowPlotMode.ALONG_NM
    assert restored_window.row_plot_widget.current_unit is PlotUnit.NM
    assert restored_window.row_metrics_widget.current_unit is PlotUnit.NM
    assert restored_window.row_disturbance_widget.current_unit is PlotUnit.NM
    assert restored_window.image_viewport.row_axis_item is not None
    assert restored_window.image_viewport.row_axis_item.isVisible() is True
    assert restored_window.image_viewport.row_disturbance_scatter_item is not None
    assert len(restored_window.image_viewport.row_disturbance_scatter_item.points()) >= 1
    assert any(
        restored_window.points_table_widget.item(row_index, 11).text() == "manual (drag)"
        for row_index in range(restored_window.points_table_widget.rowCount())
    )
    assert "candidate" in restored_window.active_row_label.text() or "no local candidates" in restored_window.active_row_label.text()


def test_main_window_end_to_end_stage4_analysis_workflow(qtbot):
    controller = AtomMapperController()
    image = _make_gaussian_image("stage4-e2e.stp", size=40, amplitude=20.0, offset=1.0)
    controller.set_loaded_images([image])

    window = AtomMapperMainWindow(controller=controller)
    qtbot.addWidget(window)

    qtbot.mouseClick(window.new_row_button, Qt.MouseButton.LeftButton)
    first_row = controller.active_row
    assert first_row is not None

    controller.update_active_roi_state(ROIState(x=10, y=10, width=12, height=12))
    qtbot.mouseClick(window.add_point_button, Qt.MouseButton.LeftButton)
    controller.update_active_roi_state(ROIState(x=18, y=18, width=12, height=12))
    qtbot.mouseClick(window.add_point_button, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: window.points_table_widget.rowCount() == 2)

    assert window.row_plot_widget.current_row is not None
    assert window.row_plot_widget.current_row.row_id == first_row.row_id
    assert window.row_plot_widget.current_mode is RowPlotMode.X_PX
    assert window.row_plot_widget.current_series is not None
    assert window.row_plot_widget.current_series.mode is RowPlotMode.X_PX
    assert window.global_scatter_plot_widget.current_series is not None
    assert len(window.global_scatter_plot_widget.current_series.samples) == 2
    assert window.row_metrics_widget.current_metrics is not None
    assert window.row_metrics_widget.current_metrics.distance_count == 1
    assert window.row_metrics_widget.stack.currentWidget() is window.row_metrics_widget.metrics_panel

    window.row_plot_widget.metric_combo.setCurrentIndex(
        window.row_plot_widget.metric_combo.findData(RowPlotMode.DISTANCE_PX)
    )
    qtbot.waitUntil(
        lambda: (
            window.row_plot_widget.current_series is not None
            and window.row_plot_widget.current_series.mode is RowPlotMode.DISTANCE_PX
        )
    )
    plotted_x, plotted_y = window.row_plot_widget.curve_item.getData()
    assert list(plotted_x) == pytest.approx([0.0])
    assert len(plotted_y) == 1

    qtbot.mouseClick(window.new_row_button, Qt.MouseButton.LeftButton)
    second_row = controller.active_row
    assert second_row is not None
    assert second_row.row_id != first_row.row_id
    controller.update_active_roi_state(ROIState(x=24, y=24, width=10, height=10))
    qtbot.mouseClick(window.add_point_button, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: window.points_table_widget.rowCount() == 3)

    assert window.global_scatter_plot_widget.current_series is not None
    assert len(window.global_scatter_plot_widget.current_series.samples) == 3
    assert len(window.global_scatter_plot_widget.scatter_items) == 2
    assert window.row_plot_widget.current_row is not None
    assert window.row_plot_widget.current_row.row_id == second_row.row_id
    assert window.row_plot_widget.current_mode is RowPlotMode.DISTANCE_PX
    assert window.row_plot_widget.stack.currentWidget() is window.row_plot_widget.placeholder_label
    assert window.row_metrics_widget.current_metrics is not None
    assert window.row_metrics_widget.current_metrics.distance_count == 0
    assert window.row_metrics_widget.stack.currentWidget() is window.row_metrics_widget.placeholder_label

    window.global_scatter_plot_widget.unit_combo.setCurrentIndex(
        window.global_scatter_plot_widget.unit_combo.findData(PlotUnit.NM)
    )
    qtbot.waitUntil(
        lambda: (
            window.global_scatter_plot_widget.current_series is not None
            and window.global_scatter_plot_widget.current_series.unit is PlotUnit.NM
        )
    )
    assert window.global_scatter_plot_widget.current_unit is PlotUnit.NM
    assert window.global_scatter_plot_widget.current_series.x_label == "x (nm)"
    assert window.global_scatter_plot_widget.current_series.y_label == "y (nm)"

    window.row_list_widget.setCurrentRow(0)
    qtbot.waitUntil(
        lambda: controller.active_row is not None and controller.active_row.row_id == first_row.row_id
    )

    assert window.row_plot_widget.current_row is not None
    assert window.row_plot_widget.current_row.row_id == first_row.row_id
    assert window.row_plot_widget.current_mode is RowPlotMode.DISTANCE_PX
    assert window.row_plot_widget.current_series is not None
    assert window.row_plot_widget.current_series.mode is RowPlotMode.DISTANCE_PX
    assert window.row_plot_widget.stack.currentWidget() is window.row_plot_widget.plot_widget
    assert window.global_scatter_plot_widget.current_unit is PlotUnit.NM
    assert window.row_metrics_widget.current_metrics is not None
    assert window.row_metrics_widget.current_metrics.distance_count == 1
    assert window.row_metrics_widget.stack.currentWidget() is window.row_metrics_widget.metrics_panel
    assert "Showing 3 saved points" in window.points_table_hint_label.text()


def test_main_window_end_to_end_stage3_workflow_across_variant_and_row_switch(qtbot):
    controller = AtomMapperController()
    original = _make_gaussian_image("stage3-e2e.stp", size=40, amplitude=20.0, offset=1.0)
    controller.set_loaded_images([original])

    window = AtomMapperMainWindow(controller=controller)
    qtbot.addWidget(window)

    qtbot.mouseClick(window.new_row_button, Qt.MouseButton.LeftButton)
    first_row = controller.active_row
    assert first_row is not None

    controller.update_active_roi_state(ROIState(x=14, y=14, width=12, height=12))
    qtbot.mouseClick(window.add_point_button, Qt.MouseButton.LeftButton)

    first_row_after_first_point = controller.active_row
    assert first_row_after_first_point is not None
    assert first_row_after_first_point.row_id == first_row.row_id
    assert first_row_after_first_point.point_count == 1
    assert window.points_table_widget.rowCount() == 1

    variant = controller.create_blur_variant_for_active_image(sigma_px=1.2, make_active=False)
    assert window.file_list_widget.count() == 2
    window.file_list_widget.setCurrentRow(1)

    assert controller.active_image == variant
    assert controller.active_row is not None
    assert controller.active_row.row_id == first_row.row_id
    assert len(window.image_viewport.point_scatter_item.points()) == 1

    qtbot.mouseClick(window.add_point_button, Qt.MouseButton.LeftButton)

    first_row_after_second_point = controller.active_row
    assert first_row_after_second_point is not None
    assert first_row_after_second_point.row_id == first_row.row_id
    assert first_row_after_second_point.point_count == 2
    assert {point.image_id for point in first_row_after_second_point.points} == {
        original.image_id,
        variant.image_id,
    }
    assert window.points_table_widget.rowCount() == 2
    assert "Showing 2 saved points" in window.points_table_hint_label.text()
    assert len(window.image_viewport.point_scatter_item.points()) == 2

    qtbot.mouseClick(window.new_row_button, Qt.MouseButton.LeftButton)
    second_row = controller.active_row
    assert second_row is not None
    assert second_row.row_id != first_row.row_id
    assert second_row.point_count == 0

    qtbot.mouseClick(window.add_point_button, Qt.MouseButton.LeftButton)

    second_row_after_point = controller.active_row
    assert second_row_after_point is not None
    assert second_row_after_point.row_id == second_row.row_id
    assert second_row_after_point.point_count == 1
    assert window.points_table_widget.rowCount() == 3

    window.row_list_widget.setCurrentRow(0)
    assert controller.active_row is not None
    assert controller.active_row.row_id == first_row.row_id

    qtbot.mouseClick(window.add_point_button, Qt.MouseButton.LeftButton)

    first_row_final = controller.active_row
    assert first_row_final is not None
    assert first_row_final.row_id == first_row.row_id
    assert first_row_final.point_count == 3
    assert window.points_table_widget.rowCount() == 4
    assert "Added point 2 to Row 1" in window.statusBar().currentMessage()
    assert "from Gaussian fit" in window.workflow_status_label.text()


def test_main_window_reports_bm3d_failure_without_crashing(qtbot, monkeypatch):
    controller = AtomMapperController()
    original = _make_gaussian_image("origin.stp", size=24, amplitude=18.0, offset=1.0)
    controller.set_loaded_images([original])

    window = AtomMapperMainWindow(controller=controller)
    qtbot.addWidget(window)

    class FakeDialog:
        def __init__(self, loaded_image, parent):
            self.preprocessing_state = PreprocessingState(
                method=PreprocessingMethod.BM3D,
                bm3d=BM3DParameters(sigma_psd=0.07, stage="all_stages"),
            )

        def exec(self):
            return int(QDialog.DialogCode.Accepted)

    window._preprocessing_dialog_class = FakeDialog
    monkeypatch.setattr(
        window.controller,
        "create_bm3d_variant_for_active_image",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("BM3D missing")),
    )

    captured: dict[str, str] = {}

    def fake_warning(parent, title, text):
        captured["title"] = title
        captured["text"] = text
        return QMessageBox.StandardButton.Ok

    monkeypatch.setattr("AtomMapper.app.main_window.QMessageBox.warning", fake_warning)

    window.preprocessing_action.trigger()

    assert len(window.controller.loaded_images) == 1
    assert captured["title"] == "AtomMapper - Preprocessing Error"
    assert "Could not create BM3D variant" in captured["text"]
    assert "BM3D preprocessing failed." in window.statusBar().currentMessage()
    assert "BM3D preprocessing failed." in window.workflow_status_label.text()


def test_main_window_does_not_open_preprocessing_dialog_without_active_image(qtbot):
    window = create_main_window()
    qtbot.addWidget(window)

    assert window.preprocessing_action.isEnabled() is False

    window._open_preprocessing_dialog()

    assert window.statusBar().currentMessage() == "Select an STM image before preprocessing."
    assert "before opening preprocessing" in window.workflow_status_label.text()
