"""Gaussian blur and sharpening preprocessing dialogs."""

import logging
from typing import Any, Dict, Optional

import numpy as np

from .base import (
    BasePreprocessingDialog,
    ImageItem,
    QCheckBox,
    QDialogButtonBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QSlider,
    Qt,
    QVBoxLayout,
    QWidget,
    RectROI,
    pg,
)
from .config import PREPROCESSING_CONFIG

try:
    from lfa.preprocessing.filtering import gaussian_blur, gaussian_sharpen_unsharp_mask
except ImportError:  # pragma: no cover
    logging.error("Could not import filtering functions (gaussian_blur/gaussian_sharpen_unsharp_mask).")

    def gaussian_blur(image, sigma):
        return image

    def gaussian_sharpen_unsharp_mask(image, radius, amount):
        return image

logger = logging.getLogger(__name__)


BLUR_CFG = PREPROCESSING_CONFIG["gaussian_blur"]
SHARPEN_CFG = PREPROCESSING_CONFIG["gaussian_sharpen"]


class GaussianBlurDialog(BasePreprocessingDialog):
    """Dialog for applying Gaussian blur with ROI-aware preview."""

    def __init__(self, original_data: np.ndarray, parent=None):
        if original_data is None:
            raise ValueError("Original data cannot be None")
        super().__init__("Gaussian Blur", parent)
        self.original_data = original_data.astype(np.float32)
        self.preview_data = self.original_data.copy()

        self.setWindowTitle(f"{self.operation_name} Settings")
        self.setMinimumSize(900, 500)
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
            pen=pg.mkPen("y", width=2),
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
        sigma_layout.addWidget(QLabel("Sigma:"))
        sigma_cfg = BLUR_CFG["sigma"]
        self.sigma_slider = QSlider(Qt.Orientation.Horizontal)
        self.sigma_slider.setRange(sigma_cfg["slider_min"], sigma_cfg["slider_max"])
        self.sigma_slider.setTickInterval(sigma_cfg["tick_interval"])
        self.sigma_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.sigma_slider.setValue(sigma_cfg["default"])
        sigma_layout.addWidget(self.sigma_slider)
        self.sigma_value_label = QLabel(f"{sigma_cfg['default'] * sigma_cfg['scale']:.1f}")
        sigma_layout.addWidget(self.sigma_value_label)
        layout.addLayout(sigma_layout)

        self.sigma_slider.valueChanged.connect(self._on_parameter_or_preview_changed)

    def _handle_parameter_widget_change(self) -> None:
        sigma = self.sigma_slider.value() * BLUR_CFG["sigma"]["scale"]
        self.sigma_value_label.setText(f"{sigma:.1f}")

    def _get_current_parameters(self) -> Dict[str, Any]:
        sigma = self.sigma_slider.value() * BLUR_CFG["sigma"]["scale"]
        return {
            "sigma": round(sigma, 2),
            "apply_roi_only": self.apply_to_roi_only_checkbox.isChecked(),
        }

    def _apply_operation(self, image: np.ndarray, params: Dict[str, Any]) -> Optional[np.ndarray]:
        sigma = params.get("sigma", 0.0)
        apply_roi_only = params.get("apply_roi_only", False)
        logger.debug("Applying Gaussian Blur. Sigma=%s, ROI Only=%s", sigma, apply_roi_only)
        try:
            processed = gaussian_blur(image, sigma)
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Gaussian Blur failed", exc_info=exc)
            return None
        return self._apply_with_optional_roi(image, processed, apply_roi_only)


class GaussianSharpeningDialog(BasePreprocessingDialog):
    """Dialog for Gaussian unsharp masking with ROI support."""

    def __init__(self, original_data: np.ndarray, parent=None):
        if original_data is None:
            raise ValueError("Original data cannot be None")
        super().__init__("Gaussian Sharpening", parent)
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
            pen=pg.mkPen("c", width=2),
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
        radius_layout = QHBoxLayout()
        radius_layout.addWidget(QLabel("Radius:"))
        radius_cfg = SHARPEN_CFG["radius"]
        self.radius_slider = QSlider(Qt.Orientation.Horizontal)
        self.radius_slider.setRange(radius_cfg["slider_min"], radius_cfg["slider_max"])
        self.radius_slider.setTickInterval(radius_cfg["tick_interval"])
        self.radius_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.radius_slider.setValue(radius_cfg["default"])
        radius_layout.addWidget(self.radius_slider)
        self.radius_value_label = QLabel(
            f"{radius_cfg['default'] * radius_cfg['scale']:.1f}"
        )
        radius_layout.addWidget(self.radius_value_label)
        layout.addLayout(radius_layout)

        amount_layout = QHBoxLayout()
        amount_layout.addWidget(QLabel("Amount:"))
        amount_cfg = SHARPEN_CFG["amount"]
        self.amount_slider = QSlider(Qt.Orientation.Horizontal)
        self.amount_slider.setRange(amount_cfg["slider_min"], amount_cfg["slider_max"])
        self.amount_slider.setTickInterval(amount_cfg["tick_interval"])
        self.amount_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.amount_slider.setValue(amount_cfg["default"])
        amount_layout.addWidget(self.amount_slider)
        self.amount_value_label = QLabel(
            f"{amount_cfg['default'] * amount_cfg['scale']:.1f}"
        )
        amount_layout.addWidget(self.amount_value_label)
        layout.addLayout(amount_layout)

        self.radius_slider.valueChanged.connect(self._on_parameter_or_preview_changed)
        self.amount_slider.valueChanged.connect(self._on_parameter_or_preview_changed)

    def _handle_parameter_widget_change(self) -> None:
        radius = self.radius_slider.value() * SHARPEN_CFG["radius"]["scale"]
        amount = self.amount_slider.value() * SHARPEN_CFG["amount"]["scale"]
        self.radius_value_label.setText(f"{radius:.1f}")
        self.amount_value_label.setText(f"{amount:.1f}")

    def _get_current_parameters(self) -> Dict[str, Any]:
        radius = self.radius_slider.value() * SHARPEN_CFG["radius"]["scale"]
        amount = self.amount_slider.value() * SHARPEN_CFG["amount"]["scale"]
        return {
            "radius": round(radius, 2),
            "amount": round(amount, 2),
            "apply_roi_only": self.apply_to_roi_only_checkbox.isChecked(),
        }

    def _apply_operation(self, image: np.ndarray, params: Dict[str, Any]) -> Optional[np.ndarray]:
        radius = params.get("radius", 1.0)
        amount = params.get("amount", 1.0)
        apply_roi_only = params.get("apply_roi_only", False)
        logger.debug(
            "Applying Gaussian sharpening. Radius=%s, Amount=%s, ROI Only=%s",
            radius,
            amount,
            apply_roi_only,
        )
        try:
            processed = gaussian_sharpen_unsharp_mask(image, radius, amount)
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Gaussian sharpening failed", exc_info=exc)
            return None
        return self._apply_with_optional_roi(image, processed, apply_roi_only)


__all__ = ["GaussianBlurDialog", "GaussianSharpeningDialog"]
