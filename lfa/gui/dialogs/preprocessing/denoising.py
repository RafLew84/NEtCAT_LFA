"""Denoising preprocessing dialogs (NL-Means and BM3D)."""

import logging
from typing import Any, Dict, Optional

import numpy as np

from .base import (
    BasePreprocessingDialog,
    QCheckBox,
    QComboBox,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
    Qt,
    pyqtSlot,
    RectROI,
    pg,
    ImageItem,
)
from .config import PREPROCESSING_CONFIG

try:
    from lfa.preprocessing.denoising import denoise_nlmeans_skimage, denoise_bm3d_lfa
except ImportError:  # pragma: no cover - import fallback
    logging.error("Could not import denoising functions (NLMeans/BM3D).")

    def denoise_nlmeans_skimage(**kwargs):
        return kwargs.get("image")

    def denoise_bm3d_lfa(**kwargs):
        return kwargs.get("image")

logger = logging.getLogger(__name__)


NLMEANS_CFG = PREPROCESSING_CONFIG["nlmeans"]
BM3D_CFG = PREPROCESSING_CONFIG["bm3d"]


class NLMeansDialog(BasePreprocessingDialog):
    """Dialog for applying Non-Local Means denoising (skimage implementation)."""

    def __init__(self, original_data: np.ndarray, parent=None):
        if original_data is None:
            raise ValueError("Original data cannot be None")
        super().__init__("NL-Means Denoising (skimage)", parent)
        self.original_data = original_data.astype(np.float32)
        self.preview_data = self.original_data.copy()

        self.setWindowTitle(f"{self.operation_name} Settings")
        self.setMinimumSize(900, 600)
        current_flags = self.windowFlags()
        self.setWindowFlags(
            current_flags
            | Qt.WindowType.WindowMinimizeButtonHint
            | Qt.WindowType.WindowMaximizeButtonHint
        )

        main_layout = QVBoxLayout(self)
        top_layout = QHBoxLayout()
        controls_area_layout = QVBoxLayout()
        bottom_layout = QHBoxLayout()

        pg.setConfigOption("background", "w")
        pg.setConfigOption("foreground", "k")
        self.win = pg.GraphicsLayoutWidget()
        self.plot_original = self.win.addPlot(row=0, col=0, title="Original", name="plot_orig")
        self.img_original = ImageItem()
        self.plot_original.addItem(self.img_original)
        self.plot_original.hideAxis("left")
        self.plot_original.hideAxis("bottom")
        self.plot_original.setAspectLocked(True)

        self.plot_processed = self.win.addPlot(row=0, col=1, title="Preview", name="plot_proc")
        self.img_processed = ImageItem()
        self.plot_processed.addItem(self.img_processed)
        self.plot_processed.hideAxis("left")
        self.plot_processed.hideAxis("bottom")
        self.plot_processed.setAspectLocked(True)
        self.plot_processed.vb.setXLink(self.plot_original.vb)
        self.plot_processed.vb.setYLink(self.plot_original.vb)
        self.plot_original.vb.invertY(True)
        self.plot_processed.vb.invertY(True)
        top_layout.addWidget(self.win, stretch=3)

        controls_panel = QWidget()
        controls_panel.setMaximumWidth(250)
        controls_panel.setLayout(controls_area_layout)

        parameter_widget_container = QWidget()
        parameter_layout = QVBoxLayout(parameter_widget_container)
        parameter_layout.setContentsMargins(0, 0, 0, 0)
        self._create_parameter_controls(parameter_layout)
        controls_area_layout.addWidget(parameter_widget_container)

        controls_area_layout.addWidget(
            QFrame(frameShape=QFrame.Shape.HLine, frameShadow=QFrame.Shadow.Sunken)
        )

        self.apply_to_roi_only_checkbox = QCheckBox("Apply only to ROI area")
        self.apply_to_roi_only_checkbox.setChecked(False)
        self.live_preview_checkbox = QCheckBox("Live Preview")
        self.live_preview_checkbox.setChecked(True)
        controls_area_layout.addWidget(self.apply_to_roi_only_checkbox)
        controls_area_layout.addWidget(self.live_preview_checkbox)

        controls_area_layout.addWidget(
            QFrame(frameShape=QFrame.Shape.HLine, frameShadow=QFrame.Shadow.Sunken)
        )

        self.roi_info_label = QLabel("ROI: Not selected")
        controls_area_layout.addWidget(self.roi_info_label)

        height, width = self.original_data.shape
        roi_w, roi_h = width // 4, height // 4
        roi_x = width // 2 - roi_w // 2
        roi_y = height // 2 - roi_h // 2
        self.roi = RectROI(
            pos=(roi_x, roi_y),
            size=(roi_w, roi_h),
            pen=pg.mkPen("b", width=2),
            translateSnap=True,
            scaleSnap=True,
        )
        self.plot_original.addItem(self.roi)

        controls_area_layout.addStretch()
        top_layout.addWidget(controls_panel, stretch=1)

        self.button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.button_box.button(QDialogButtonBox.StandardButton.Ok).setText("Apply Changes")
        bottom_layout.addWidget(self.button_box)

        main_layout.addLayout(top_layout)
        main_layout.addLayout(bottom_layout)

        self._initialize_common_behavior()

    def _create_parameter_controls(self, layout: QVBoxLayout) -> None:
        sigma_layout = QHBoxLayout()
        sigma_layout.addWidget(QLabel("Noise sigma:"))
        sigma_cfg = NLMEANS_CFG["sigma"]
        self.sigma_spinbox = QDoubleSpinBox()
        self.sigma_spinbox.setDecimals(sigma_cfg["decimals"])
        self.sigma_spinbox.setRange(sigma_cfg["min"], sigma_cfg["max"])
        estimated_sigma = float(np.std(self.original_data) / sigma_cfg["estimate_divisor"])
        self.sigma_spinbox.setValue(max(sigma_cfg["fallback"], estimated_sigma))
        self.sigma_spinbox.setSingleStep(sigma_cfg["single_step"])
        sigma_layout.addWidget(self.sigma_spinbox)
        layout.addLayout(sigma_layout)

        h_mult_layout = QHBoxLayout()
        h_mult_layout.addWidget(QLabel("H multiplier:"))
        h_cfg = NLMEANS_CFG["h_multiplier"]
        self.h_mult_spinbox = QDoubleSpinBox()
        self.h_mult_spinbox.setDecimals(h_cfg["decimals"])
        self.h_mult_spinbox.setRange(h_cfg["min"], h_cfg["max"])
        self.h_mult_spinbox.setValue(h_cfg["default"])
        self.h_mult_spinbox.setSingleStep(h_cfg["single_step"])
        h_mult_layout.addWidget(self.h_mult_spinbox)
        layout.addLayout(h_mult_layout)

        patch_size_layout = QHBoxLayout()
        patch_size_layout.addWidget(QLabel("Patch size:"))
        patch_cfg = NLMEANS_CFG["patch_size"]
        self.patch_size_spinbox = QSpinBox()
        self.patch_size_spinbox.setRange(patch_cfg["min"], patch_cfg["max"])
        self.patch_size_spinbox.setSingleStep(patch_cfg["single_step"])
        self.patch_size_spinbox.setValue(patch_cfg["default"])
        patch_size_layout.addWidget(self.patch_size_spinbox)
        layout.addLayout(patch_size_layout)

        patch_dist_layout = QHBoxLayout()
        patch_dist_layout.addWidget(QLabel("Patch distance:"))
        patch_dist_cfg = NLMEANS_CFG["patch_distance"]
        self.patch_dist_spinbox = QSpinBox()
        self.patch_dist_spinbox.setRange(patch_dist_cfg["min"], patch_dist_cfg["max"])
        self.patch_dist_spinbox.setSingleStep(patch_dist_cfg["single_step"])
        self.patch_dist_spinbox.setValue(patch_dist_cfg["default"])
        patch_dist_layout.addWidget(self.patch_dist_spinbox)
        layout.addLayout(patch_dist_layout)

        self.fast_mode_checkbox = QCheckBox("Use Fast Mode")
        self.fast_mode_checkbox.setChecked(NLMEANS_CFG["fast_mode"]["default"])
        layout.addWidget(self.fast_mode_checkbox)

        self.sigma_spinbox.valueChanged.connect(self._on_parameter_or_preview_changed)
        self.h_mult_spinbox.valueChanged.connect(self._on_parameter_or_preview_changed)
        self.patch_size_spinbox.valueChanged.connect(self._on_parameter_or_preview_changed)
        self.patch_dist_spinbox.valueChanged.connect(self._on_parameter_or_preview_changed)
        self.fast_mode_checkbox.stateChanged.connect(self._on_parameter_or_preview_changed)

    def _get_current_parameters(self) -> Dict[str, Any]:
        return {
            "sigma": self.sigma_spinbox.value(),
            "h_param_mult": self.h_mult_spinbox.value(),
            "patch_size": self.patch_size_spinbox.value(),
            "patch_distance": self.patch_dist_spinbox.value(),
            "fast_mode": self.fast_mode_checkbox.isChecked(),
            "apply_roi_only": self.apply_to_roi_only_checkbox.isChecked(),
        }

    def _apply_operation(self, image: np.ndarray, params: Dict[str, Any]) -> Optional[np.ndarray]:
        logger.debug("NL-Means parameters: %s", params)
        try:
            processed = denoise_nlmeans_skimage(
                image,
                sigma=params.get("sigma", 0.01),
                h_param_mult=params.get("h_param_mult", 1.0),
                patch_size=params.get("patch_size", 7),
                patch_distance=params.get("patch_distance", 11),
                fast_mode=params.get("fast_mode", True),
            )
        except Exception as exc:  # pragma: no cover
            logger.exception("NL-Means denoising failed", exc_info=exc)
            return None
        return self._apply_with_optional_roi(
            image,
            processed,
            params.get("apply_roi_only", False),
        )


class BM3DDialog(BasePreprocessingDialog):
    """Dialog for BM3D denoising with optional ROI application."""

    def __init__(self, original_data: np.ndarray, parent=None):
        if original_data is None:
            raise ValueError("Original data cannot be None")
        super().__init__("BM3D Denoising", parent)
        self.original_data = original_data.astype(np.float32)
        self.preview_data = self.original_data.copy()

        self.setWindowTitle(f"{self.operation_name} Settings")
        self.setMinimumSize(900, 550)
        current_flags = self.windowFlags()
        self.setWindowFlags(
            current_flags
            | Qt.WindowType.WindowMinimizeButtonHint
            | Qt.WindowType.WindowMaximizeButtonHint
        )

        main_layout = QVBoxLayout(self)
        top_layout = QHBoxLayout()
        controls_area_layout = QVBoxLayout()
        bottom_layout = QHBoxLayout()

        pg.setConfigOption("background", "w")
        pg.setConfigOption("foreground", "k")
        self.win = pg.GraphicsLayoutWidget()
        self.plot_original = self.win.addPlot(row=0, col=0, title="Original", name="plot_orig")
        self.img_original = ImageItem()
        self.plot_original.addItem(self.img_original)
        self.plot_original.hideAxis("left")
        self.plot_original.hideAxis("bottom")
        self.plot_original.setAspectLocked(True)

        self.plot_processed = self.win.addPlot(row=0, col=1, title="Preview", name="plot_proc")
        self.img_processed = ImageItem()
        self.plot_processed.addItem(self.img_processed)
        self.plot_processed.hideAxis("left")
        self.plot_processed.hideAxis("bottom")
        self.plot_processed.setAspectLocked(True)
        self.plot_processed.vb.setXLink(self.plot_original.vb)
        self.plot_processed.vb.setYLink(self.plot_original.vb)
        self.plot_original.vb.invertY(True)
        self.plot_processed.vb.invertY(True)
        top_layout.addWidget(self.win, stretch=3)

        controls_panel = QWidget()
        controls_panel.setMaximumWidth(250)
        controls_panel.setLayout(controls_area_layout)

        parameter_widget_container = QWidget()
        parameter_layout = QVBoxLayout(parameter_widget_container)
        parameter_layout.setContentsMargins(0, 0, 0, 0)
        self._create_parameter_controls(parameter_layout)
        controls_area_layout.addWidget(parameter_widget_container)

        controls_area_layout.addWidget(
            QFrame(frameShape=QFrame.Shape.HLine, frameShadow=QFrame.Shadow.Sunken)
        )

        self.apply_to_roi_only_checkbox = QCheckBox("Apply only to ROI area")
        self.apply_to_roi_only_checkbox.setChecked(False)
        controls_area_layout.addWidget(self.apply_to_roi_only_checkbox)

        self.live_preview_checkbox = QCheckBox("Live Preview")
        self.live_preview_checkbox.setChecked(False)
        self.live_preview_checkbox.setToolTip("BM3D is expensive; live preview is disabled by default.")
        controls_area_layout.addWidget(self.live_preview_checkbox)

        self.update_preview_button = QPushButton("Update Preview")
        controls_area_layout.addWidget(self.update_preview_button)

        controls_area_layout.addWidget(
            QFrame(frameShape=QFrame.Shape.HLine, frameShadow=QFrame.Shadow.Sunken)
        )

        self.roi_info_label = QLabel("ROI: Not selected")
        controls_area_layout.addWidget(self.roi_info_label)

        height, width = self.original_data.shape
        roi_w, roi_h = width // 4, height // 4
        roi_x = width // 2 - roi_w // 2
        roi_y = height // 2 - roi_h // 2
        self.roi = RectROI(
            pos=(roi_x, roi_y),
            size=(roi_w, roi_h),
            pen=pg.mkPen("r", width=2),
            translateSnap=True,
            scaleSnap=True,
        )
        self.plot_original.addItem(self.roi)

        controls_area_layout.addStretch()
        top_layout.addWidget(controls_panel, stretch=1)

        self.button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.button_box.button(QDialogButtonBox.StandardButton.Ok).setText("Apply Changes")
        bottom_layout.addWidget(self.button_box)

        main_layout.addLayout(top_layout)
        main_layout.addLayout(bottom_layout)

        self._initialize_common_behavior()
        self.update_preview_button.clicked.connect(self._update_preview)

    def _create_parameter_controls(self, layout: QVBoxLayout) -> None:
        sigma_layout = QHBoxLayout()
        sigma_layout.addWidget(QLabel("Noise sigma [0-1]:"))
        sigma_cfg = BM3D_CFG["sigma"]
        self.sigma_spinbox = QDoubleSpinBox()
        self.sigma_spinbox.setDecimals(sigma_cfg["decimals"])
        self.sigma_spinbox.setRange(sigma_cfg["min"], sigma_cfg["max"])
        self.sigma_spinbox.setValue(sigma_cfg["default"])
        self.sigma_spinbox.setSingleStep(sigma_cfg["single_step"])
        self.sigma_spinbox.setToolTip("Estimated noise standard deviation relative to the [0, 1] data range.")
        sigma_layout.addWidget(self.sigma_spinbox)
        layout.addLayout(sigma_layout)

        self.sigma_spinbox.valueChanged.connect(self._on_parameter_or_preview_changed)

    def _get_current_parameters(self) -> Dict[str, Any]:
        return {
            "sigma_psd": self.sigma_spinbox.value(),
            "apply_roi_only": self.apply_to_roi_only_checkbox.isChecked(),
        }

    def _apply_operation(self, image: np.ndarray, params: Dict[str, Any]) -> Optional[np.ndarray]:
        sigma = params.get("sigma_psd", 0.05)
        apply_roi_only = params.get("apply_roi_only", False)
        logger.debug("BM3D params: sigma=%s, ROI Only=%s", sigma, apply_roi_only)
        try:
            processed = denoise_bm3d_lfa(image, sigma_psd=sigma)
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("BM3D denoising failed", exc_info=exc)
            return None
        return self._apply_with_optional_roi(image, processed, apply_roi_only)

    def _handle_parameter_widget_change(self) -> None:
        is_roi_mode = self.apply_to_roi_only_checkbox.isChecked()
        self.roi.setVisible(is_roi_mode)
        self.roi_info_label.setVisible(is_roi_mode)

    @pyqtSlot()
    def _on_parameter_or_preview_changed(self) -> None:
        super()._on_parameter_or_preview_changed()

    @pyqtSlot()
    def _on_roi_changed(self) -> None:
        super()._on_roi_changed()

    @pyqtSlot()
    def _update_preview(self) -> None:
        params = self._get_current_parameters()
        logger.debug("Manually updating BM3D preview. Params: %s", params)
        try:
            self.preview_data = self._apply_operation(self.original_data, params)
            if self.preview_data is None:
                self.preview_data = self.original_data.copy()
            self.update_preview_view()
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Error during BM3D preview update", exc_info=exc)
            QMessageBox.warning(self, "Preview Error", f"Could not update preview:\n{exc}")


__all__ = ["NLMeansDialog", "BM3DDialog"]
