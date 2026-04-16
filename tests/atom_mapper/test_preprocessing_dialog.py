"""Tests for the AtomMapper preprocessing dialog skeleton."""

from __future__ import annotations

import numpy as np
import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QDialogButtonBox

from AtomMapper.app.preprocessing_dialog import PreprocessingDialog
from AtomMapper.app.models import LoadedImage
from AtomMapper.app.preprocessing import is_bm3d_available
from AtomMapper.app.preprocessing_state import (
    BM3DParameters,
    BlurParameters,
    FlipParameters,
    PreprocessingMethod,
    PreprocessingPreviewRequest,
    PreprocessingState,
    RotateParameters,
)

pytest.importorskip("PyQt6", reason="PyQt6 is required for AtomMapper GUI tests")
pytest.importorskip("pytestqt", reason="pytest-qt is required for AtomMapper GUI tests")


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


def test_preprocessing_dialog_shows_original_preview_and_placeholders(qtbot):
    image = _make_loaded_image("sample.stp", np.arange(400, dtype=float).reshape((20, 20)))
    dialog = PreprocessingDialog(image)
    qtbot.addWidget(dialog)

    assert "sample.stp" in dialog.windowTitle()
    assert dialog.header_label.text() == "Preprocessing: sample.stp"
    assert dialog.original_preview_label.pixmap() is not None
    assert dialog.processed_preview.image_label.pixmap() is not None
    assert "Blur preview ready. sigma_px=1.00" in dialog.processed_preview_label.text()
    assert dialog.preprocessing_state.method is PreprocessingMethod.BLUR
    assert "Active method: Blur" in dialog.parameters_placeholder_label.text()
    assert dialog.original_preview.current_viewport == dialog.processed_preview.current_viewport
    assert dialog.original_preview.current_patch_shape == dialog.processed_preview.current_patch_shape
    assert dialog.method_combo.count() == 5
    assert dialog.method_combo.itemData(0) is PreprocessingMethod.BLUR
    assert dialog.method_combo.itemData(1) is PreprocessingMethod.NLM
    assert dialog.method_combo.itemData(2) is PreprocessingMethod.BM3D
    assert dialog.method_combo.itemData(3) is PreprocessingMethod.ROTATE
    assert dialog.method_combo.itemData(4) is PreprocessingMethod.FLIP
    assert dialog.apply_button is not None
    assert dialog.apply_button.isEnabled() is True


def test_preprocessing_dialog_cancel_rejects_without_side_effects(qtbot):
    image = _make_loaded_image("sample.stp", np.arange(400, dtype=float).reshape((20, 20)))
    dialog = PreprocessingDialog(image)
    qtbot.addWidget(dialog)
    dialog.show()

    qtbot.mouseClick(
        dialog.button_box.button(QDialogButtonBox.StandardButton.Cancel),
        Qt.MouseButton.LeftButton,
    )

    assert dialog.result() == int(dialog.DialogCode.Rejected)


def test_preprocessing_dialog_apply_button_accepts_when_enabled(qtbot):
    image = _make_loaded_image("sample.stp", np.arange(400, dtype=float).reshape((20, 20)))
    dialog = PreprocessingDialog(image)
    qtbot.addWidget(dialog)
    dialog.show()

    assert dialog.apply_button is not None
    assert dialog.apply_button.isEnabled() is True

    qtbot.mouseClick(
        dialog.button_box.button(QDialogButtonBox.StandardButton.Apply),
        Qt.MouseButton.LeftButton,
    )

    assert dialog.result() == int(dialog.DialogCode.Accepted)


def test_preprocessing_dialog_updates_shared_state_and_request(qtbot):
    image = _make_loaded_image("sample.stp", np.arange(400, dtype=float).reshape((20, 20)))
    dialog = PreprocessingDialog(image)
    qtbot.addWidget(dialog)

    dialog.method_combo.setCurrentIndex(1)

    assert dialog.preprocessing_state.method is PreprocessingMethod.NLM
    assert "Active method: Non-local means" in dialog.parameters_placeholder_label.text()

    request = dialog.current_preview_request()

    assert isinstance(request, PreprocessingPreviewRequest)
    assert request.image_id == image.image_id
    assert request.source_group_id == image.source_group_id
    assert request.state.method is PreprocessingMethod.NLM
    assert request.viewport is not None
    assert request.viewport.width == image.pixels_x
    assert request.viewport.height == image.pixels_y


def test_preprocessing_dialog_refreshes_only_processed_preview_on_method_change(qtbot):
    image = _make_loaded_image("sample.stp", np.arange(400, dtype=float).reshape((20, 20)))
    dialog = PreprocessingDialog(image)
    qtbot.addWidget(dialog)

    original_render_count = dialog.original_preview.render_count
    processed_render_count = dialog.processed_preview.render_count

    dialog.method_combo.setCurrentIndex(2)

    assert dialog.original_preview.render_count == original_render_count
    assert dialog.processed_preview.render_count == processed_render_count + 1
    assert dialog.original_preview.current_viewport == dialog.processed_preview.current_viewport
    if is_bm3d_available():
        assert "BM3D preview ready." in dialog.processed_preview_label.text()
        assert dialog.apply_button.isEnabled() is True
    else:
        assert "BM3D backend unavailable." in dialog.processed_preview_label.text()
        assert dialog.apply_button.isEnabled() is False


def test_preprocessing_dialog_blur_sigma_updates_live_preview(qtbot):
    image_data = np.zeros((25, 25), dtype=float)
    image_data[12, 12] = 10.0
    image = _make_loaded_image("sample.stp", image_data)
    dialog = PreprocessingDialog(image)
    qtbot.addWidget(dialog)

    initial_processed_render_count = dialog.processed_preview.render_count
    initial_original_render_count = dialog.original_preview.render_count
    initial_preview = dialog.latest_preview_result

    assert initial_preview is not None
    assert initial_preview.success is True
    assert "sigma_px=1.00" in dialog.processed_preview_label.text()
    assert dialog.apply_button.isEnabled() is True

    dialog.blur_sigma_spinbox.setValue(2.5)

    assert dialog.preprocessing_state == PreprocessingState(
        method=PreprocessingMethod.BLUR,
        blur=BlurParameters(sigma_px=2.5, mode="nearest"),
    ).normalized()
    assert dialog.original_preview.render_count == initial_original_render_count
    assert dialog.processed_preview.render_count == initial_processed_render_count + 1
    assert dialog.latest_preview_result is not None
    assert dialog.latest_preview_result.success is True
    assert dialog.latest_preview_result.processed_image is not None
    assert "sigma_px=2.50" in dialog.processed_preview_label.text()


def test_preprocessing_dialog_nlm_parameters_update_live_preview(qtbot):
    rng = np.random.default_rng(12345)
    image_data = np.zeros((25, 25), dtype=float)
    image_data[12, 12] = 10.0
    image_data = image_data + rng.normal(0.0, 0.08, size=image_data.shape)
    image = _make_loaded_image("sample.stp", image_data)
    dialog = PreprocessingDialog(image)
    qtbot.addWidget(dialog)

    initial_original_render_count = dialog.original_preview.render_count
    initial_processed_render_count = dialog.processed_preview.render_count

    dialog.method_combo.setCurrentIndex(1)

    assert dialog.preprocessing_state.method is PreprocessingMethod.NLM
    assert dialog.apply_button.isEnabled() is True
    assert dialog.nlm_h_spinbox.isEnabled() is True
    assert dialog.latest_preview_result is not None
    assert dialog.latest_preview_result.success is True
    assert "NLM preview ready." in dialog.processed_preview_label.text()

    dialog.nlm_h_spinbox.setValue(0.2)
    dialog.nlm_patch_size_spinbox.setValue(7)
    dialog.nlm_patch_distance_spinbox.setValue(8)
    dialog.nlm_fast_mode_checkbox.setChecked(False)

    assert dialog.preprocessing_state.nlm.h == pytest.approx(0.2)
    assert dialog.preprocessing_state.nlm.patch_size == 7
    assert dialog.preprocessing_state.nlm.patch_distance == 8
    assert dialog.preprocessing_state.nlm.fast_mode is False
    assert dialog.original_preview.render_count == initial_original_render_count
    assert dialog.processed_preview.render_count >= initial_processed_render_count + 2
    assert dialog.original_preview.current_viewport == dialog.processed_preview.current_viewport
    assert "h=0.200" in dialog.processed_preview_label.text()
    assert "patch=7" in dialog.processed_preview_label.text()


def test_preprocessing_dialog_bm3d_updates_live_preview_when_available(qtbot):
    if not is_bm3d_available():
        pytest.skip("bm3d package not available in test environment")

    rng = np.random.default_rng(12345)
    image_data = np.zeros((16, 16), dtype=float)
    image_data[8, 8] = 10.0
    image_data = image_data + rng.normal(0.0, 0.05, size=image_data.shape)
    image = _make_loaded_image("sample.stp", image_data)
    dialog = PreprocessingDialog(image)
    qtbot.addWidget(dialog)

    dialog.method_combo.setCurrentIndex(2)

    assert dialog.preprocessing_state.method is PreprocessingMethod.BM3D
    assert dialog.bm3d_sigma_spinbox.isEnabled() is True
    assert dialog.apply_button.isEnabled() is True
    assert "BM3D backend available." in dialog.bm3d_availability_label.text()
    assert dialog.latest_preview_result is not None
    assert dialog.latest_preview_result.success is True
    assert "BM3D preview ready." in dialog.processed_preview_label.text()

    dialog.bm3d_sigma_spinbox.setValue(0.08)

    assert dialog.preprocessing_state.bm3d == BM3DParameters(sigma_psd=0.08, stage="all_stages").normalized()
    assert dialog.latest_preview_result is not None
    assert dialog.latest_preview_result.success is True
    assert "sigma_psd=0.080" in dialog.processed_preview_label.text()


def test_preprocessing_dialog_disables_bm3d_when_backend_missing(qtbot, monkeypatch):
    monkeypatch.setattr("AtomMapper.app.preprocessing_dialog.is_bm3d_available", lambda: False)

    image = _make_loaded_image("sample.stp", np.arange(400, dtype=float).reshape((20, 20)))
    dialog = PreprocessingDialog(image)
    qtbot.addWidget(dialog)

    bm3d_index = dialog.method_combo.findData(PreprocessingMethod.BM3D)
    assert bm3d_index >= 0
    bm3d_item = dialog.method_combo.model().item(bm3d_index)
    assert bm3d_item is not None
    assert bm3d_item.isEnabled() is False
    assert "unavailable" in dialog.bm3d_availability_label.text().lower()
    assert dialog.method_combo.itemData(bm3d_index, Qt.ItemDataRole.ToolTipRole).startswith(
        "BM3D backend unavailable."
    )


def test_preprocessing_dialog_rotate_parameters_update_live_preview(qtbot):
    image_data = np.arange(15, dtype=float).reshape((3, 5))
    image = _make_loaded_image("rotate.stp", image_data)
    dialog = PreprocessingDialog(image)
    qtbot.addWidget(dialog)

    dialog.method_combo.setCurrentIndex(3)

    assert dialog.preprocessing_state.method is PreprocessingMethod.ROTATE
    assert dialog.rotate_turns_combo.isEnabled() is True
    assert dialog.latest_preview_result is not None
    assert dialog.latest_preview_result.success is True
    assert dialog.latest_preview_result.processed_image is not None
    assert dialog.latest_preview_result.processed_image.shape == (5, 3)

    dialog.rotate_turns_combo.setCurrentIndex(2)

    assert dialog.preprocessing_state.rotate == RotateParameters(quarter_turns=3).normalized()
    assert dialog.latest_preview_result is not None
    assert dialog.latest_preview_result.success is True
    assert dialog.latest_preview_result.processed_image is not None
    assert dialog.latest_preview_result.processed_image.shape == (5, 3)
    assert "angle=270° CCW" in dialog.processed_preview_label.text()


def test_preprocessing_dialog_flip_parameters_update_live_preview(qtbot):
    image_data = np.arange(16, dtype=float).reshape((4, 4))
    image = _make_loaded_image("flip.stp", image_data)
    dialog = PreprocessingDialog(image)
    qtbot.addWidget(dialog)

    dialog.method_combo.setCurrentIndex(4)

    assert dialog.preprocessing_state.method is PreprocessingMethod.FLIP
    assert dialog.flip_x_checkbox.isEnabled() is True
    assert dialog.flip_y_checkbox.isEnabled() is True
    assert dialog.latest_preview_result is not None
    assert dialog.latest_preview_result.success is True

    dialog.flip_x_checkbox.setChecked(False)
    dialog.flip_y_checkbox.setChecked(True)

    assert dialog.preprocessing_state.flip == FlipParameters(flip_x=False, flip_y=True).normalized()
    assert dialog.latest_preview_result is not None
    assert dialog.latest_preview_result.success is True
    assert "axes=Y" in dialog.processed_preview_label.text()
