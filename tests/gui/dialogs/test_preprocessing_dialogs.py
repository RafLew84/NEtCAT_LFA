import numpy as np
import pytest
from PyQt6.QtCore import Qt

pytest.importorskip("PyQt6", reason="PyQt6 is required for preprocessing dialog tests")
pytest.importorskip("pytestqt", reason="pytest-qt is required for qtbot fixture")

from lfa.gui.dialogs.preprocessing import (
    BM3DDialog,
    GaussianBlurDialog,
    MedianFilterDialog,
    NLMeansDialog,
    PlaneLevelingDialog,
)
from lfa.gui.dialogs.preprocessing.config import PREPROCESSING_CONFIG


@pytest.fixture
def sample_image():
    return np.zeros((16, 16), dtype=np.float32)


def test_gaussian_blur_defaults_and_roi(qtbot, sample_image):
    dialog = GaussianBlurDialog(sample_image)
    qtbot.addWidget(dialog)
    dialog.show()

    sigma_cfg = PREPROCESSING_CONFIG["gaussian_blur"]["sigma"]
    assert dialog.sigma_slider.value() == sigma_cfg["default"]

    params = dialog._get_current_parameters()
    assert params["sigma"] == pytest.approx(sigma_cfg["default"] * sigma_cfg["scale"])

    assert not dialog.roi.isVisible()
    qtbot.mouseClick(dialog.apply_to_roi_only_checkbox, Qt.MouseButton.LeftButton)
    assert dialog.roi.isVisible()


def test_nlmeans_parameter_collection(qtbot, sample_image):
    dialog = NLMeansDialog(sample_image)
    qtbot.addWidget(dialog)
    dialog.show()

    cfg = PREPROCESSING_CONFIG["nlmeans"]
    params = dialog._get_current_parameters()

    assert params["patch_size"] == cfg["patch_size"]["default"]
    assert params["patch_distance"] == cfg["patch_distance"]["default"]
    assert params["fast_mode"] is cfg["fast_mode"]["default"]
    assert params["sigma"] == pytest.approx(cfg["sigma"]["fallback"])


def test_plane_leveling_mode_switch(qtbot, sample_image):
    dialog = PlaneLevelingDialog(sample_image)
    qtbot.addWidget(dialog)
    dialog.show()

    assert dialog.rb_whole.isChecked()
    assert not dialog.roi.isVisible()

    qtbot.mouseClick(dialog.rb_roi, Qt.MouseButton.LeftButton)
    assert dialog.roi.isVisible()
    assert dialog.apply_to_roi_only_checkbox.isVisible()

    qtbot.mouseClick(dialog.rb_3pt, Qt.MouseButton.LeftButton)
    assert dialog.points_groupbox.isVisible()
    dialog._selected_points = [(0, 0)]
    dialog._clear_points()
    assert dialog._selected_points == []


def test_median_mode_toggles_constant_controls(qtbot, sample_image):
    dialog = MedianFilterDialog(sample_image)
    qtbot.addWidget(dialog)
    dialog.show()

    assert not dialog.cval_label.isVisible()

    dialog.mode_combobox.setCurrentText("constant")
    assert dialog.cval_label.isVisible()
    assert dialog.cval_spinbox.isVisible()


def test_bm3d_update_preview(monkeypatch, qtbot, sample_image):
    dialog = BM3DDialog(sample_image)
    qtbot.addWidget(dialog)
    dialog.show()

    def fake_bm3d(image, sigma_psd=0.05):
        return np.ones_like(image)

    monkeypatch.setattr(
        "lfa.gui.dialogs.preprocessing.denoising.denoise_bm3d_lfa",
        fake_bm3d,
    )

    dialog._update_preview()
    assert np.all(dialog.preview_data == 1.0)
