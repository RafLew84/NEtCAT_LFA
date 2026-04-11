"""End-to-end GUI smoke tests for the AtomMapper foundation workflow."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtWidgets import QDialog, QMessageBox

from AtomMapper.app.controller import AtomMapperController
from AtomMapper.app.main import create_main_window
from AtomMapper.app.main_window import AtomMapperMainWindow
from AtomMapper.app.models import LoadedImage, ROIState
from AtomMapper.app.preprocessing import is_bm3d_available
from AtomMapper.app.preprocessing_dialog import PreprocessingDialog
from AtomMapper.app.preprocessing_state import (
    BM3DParameters,
    BlurParameters,
    NonLocalMeansParameters,
    PreprocessingMethod,
    PreprocessingState,
)

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


def test_main_window_loads_sample_file_via_button(qtbot, monkeypatch):
    window = create_main_window()
    qtbot.addWidget(window)

    sample_path = str(PROJECT_ROOT / "data" / "8343.stp")

    monkeypatch.setattr(
        "AtomMapper.app.main_window.QFileDialog.getOpenFileNames",
        lambda *args, **kwargs: ([sample_path], "STM files (*.stp *.s94)"),
    )

    qtbot.mouseClick(window.load_button, Qt.MouseButton.LeftButton)

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
    assert window.roi_preview.current_loaded_image is not None
    assert window.roi_preview.current_patch_data is not None
    assert window.roi_preview.preview_label.pixmap() is not None
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

    qtbot.mouseClick(window.load_button, Qt.MouseButton.LeftButton)

    assert len(window.controller.loaded_images) == 0
    assert window.file_list_widget.count() == 0
    assert captured["title"] == "AtomMapper - Load Error"
    assert "Could not load one or more STM files" in captured["text"]
    assert "bad.txt" in captured["text"]
    assert "Some files failed to load." in window.statusBar().currentMessage()


def test_main_window_updates_roi_preview_when_roi_changes(qtbot):
    controller = AtomMapperController()
    image = _make_loaded_image(
        "roi-sync.stp",
        np.arange(400, dtype=float).reshape((20, 20)),
    )
    controller.set_loaded_images([image])

    window = AtomMapperMainWindow(controller=controller)
    qtbot.addWidget(window)

    initial_patch = window.roi_preview.current_patch_data
    assert initial_patch is not None

    clamped_roi = controller.update_active_roi_state(ROIState(x=2, y=3, width=6, height=5))

    updated_patch = window.roi_preview.current_patch_data
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

    qtbot.mouseClick(window.load_button, Qt.MouseButton.LeftButton)

    assert window.file_list_widget.count() == 2
    assert window.controller.active_image is not None
    assert window.controller.active_image.display_name == "fake-first.stp"
    assert window.image_viewport.image_item is not None
    assert window.image_viewport.image_item.image is not None
    assert window.roi_preview.current_patch_data is not None
    assert window.gaussian_fit_preview.current_fit_result is not None
    assert window.gaussian_fit_preview.current_fit_result.success is True
    assert "Gaussian center" in window.workflow_status_label.text()

    first_patch = np.array(window.roi_preview.current_patch_data, copy=True)

    window.file_list_widget.setCurrentRow(1)

    assert window.controller.active_image is not None
    assert window.controller.active_image.display_name == "fake-second.stp"
    assert window.roi_preview.current_loaded_image is not None
    assert window.roi_preview.current_loaded_image.display_name == "fake-second.stp"
    assert window.roi_preview.current_patch_data is not None
    assert not np.array_equal(window.roi_preview.current_patch_data, first_patch)
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
    assert window.roi_preview.current_loaded_image == variant

    window.file_list_widget.setCurrentRow(2)
    assert window.controller.active_image == second
    assert window.statusBar().currentMessage() == "Selected second.stp."


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

    assert window.preprocessing_button.isEnabled()

    qtbot.mouseClick(window.preprocessing_button, Qt.MouseButton.LeftButton)

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

    qtbot.mouseClick(window.preprocessing_button, Qt.MouseButton.LeftButton)

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

    qtbot.mouseClick(window.preprocessing_button, Qt.MouseButton.LeftButton)

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

    qtbot.mouseClick(window.preprocessing_button, Qt.MouseButton.LeftButton)

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

    qtbot.mouseClick(window.preprocessing_button, Qt.MouseButton.LeftButton)

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

    qtbot.mouseClick(window.preprocessing_button, Qt.MouseButton.LeftButton)

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

    qtbot.mouseClick(window.preprocessing_button, Qt.MouseButton.LeftButton)

    assert len(window.controller.loaded_images) == 1
    assert captured["title"] == "AtomMapper - Preprocessing Error"
    assert "Could not create BM3D variant" in captured["text"]
    assert "BM3D preprocessing failed." in window.statusBar().currentMessage()
    assert "BM3D preprocessing failed." in window.workflow_status_label.text()


def test_main_window_does_not_open_preprocessing_dialog_without_active_image(qtbot):
    window = create_main_window()
    qtbot.addWidget(window)

    assert window.preprocessing_button.isEnabled() is False

    window._open_preprocessing_dialog()

    assert window.statusBar().currentMessage() == "Select an STM image before preprocessing."
    assert "before opening preprocessing" in window.workflow_status_label.text()
