"""Median filter preprocessing dialog."""

import logging
from typing import Any, Dict, Optional

import numpy as np

from .base import (
    BasePreprocessingDialog,
    ImageItem,
    QCheckBox,
    QComboBox,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    Qt,
    QVBoxLayout,
    QWidget,
    RectROI,
    pg,
)
from .config import PREPROCESSING_CONFIG

try:
    from lfa.preprocessing.filtering import median_filter_lfa
except ImportError:  # pragma: no cover
    logging.error("Could not import median_filter_lfa.")

    def median_filter_lfa(image, size, mode="reflect", cval=0.0):
        return image

logger = logging.getLogger(__name__)


MEDIAN_CFG = PREPROCESSING_CONFIG["median"]


class MedianFilterDialog(BasePreprocessingDialog):
    """Dialog for applying a median filter with optional ROI restriction."""

    def __init__(self, original_data: np.ndarray, parent=None):
        if original_data is None:
            raise ValueError("Original data cannot be None")
        super().__init__("Median Filter", parent)
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
        self._update_cval_visibility()

    def _create_parameter_controls(self, layout: QVBoxLayout) -> None:
        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("Kernel size:"))
        kernel_cfg = MEDIAN_CFG["kernel"]
        self.size_spinbox = QSpinBox()
        self.size_spinbox.setRange(kernel_cfg["min"], kernel_cfg["max"])
        self.size_spinbox.setSingleStep(kernel_cfg["single_step"])
        self.size_spinbox.setValue(kernel_cfg["default"])
        size_layout.addWidget(self.size_spinbox)
        layout.addLayout(size_layout)

        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Mode:"))
        self.mode_combobox = QComboBox()
        self.valid_modes = MEDIAN_CFG["mode"]["options"]
        self.mode_combobox.addItems(self.valid_modes)
        self.mode_combobox.setCurrentText(MEDIAN_CFG["mode"]["default"])
        mode_layout.addWidget(self.mode_combobox)
        layout.addLayout(mode_layout)

        cval_layout = QHBoxLayout()
        self.cval_label = QLabel("Constant value:")
        cval_cfg = MEDIAN_CFG["constant_value"]
        self.cval_spinbox = QDoubleSpinBox()
        self.cval_spinbox.setRange(cval_cfg["min"], cval_cfg["max"])
        self.cval_spinbox.setDecimals(cval_cfg["decimals"])
        self.cval_spinbox.setSingleStep(cval_cfg["single_step"])
        self.cval_spinbox.setValue(cval_cfg["default"])
        cval_layout.addWidget(self.cval_label)
        cval_layout.addWidget(self.cval_spinbox)
        layout.addLayout(cval_layout)

        self.size_spinbox.valueChanged.connect(self._on_parameter_or_preview_changed)
        self.mode_combobox.currentIndexChanged.connect(self._on_mode_combobox_changed)
        self.cval_spinbox.valueChanged.connect(self._on_parameter_or_preview_changed)

    def _on_mode_combobox_changed(self, index: int) -> None:
        self._update_cval_visibility()
        self._on_parameter_or_preview_changed()

    def _update_cval_visibility(self) -> None:
        is_constant_mode = self.mode_combobox.currentText() == "constant"
        self.cval_label.setVisible(is_constant_mode)
        self.cval_spinbox.setVisible(is_constant_mode)

    def _get_current_parameters(self) -> Dict[str, Any]:
        return {
            "size": self.size_spinbox.value(),
            "mode": self.mode_combobox.currentText(),
            "cval": self.cval_spinbox.value(),
            "apply_roi_only": self.apply_to_roi_only_checkbox.isChecked(),
        }

    def _apply_operation(self, image: np.ndarray, params: Dict[str, Any]) -> Optional[np.ndarray]:
        size = params.get("size", 3)
        mode = params.get("mode", "reflect")
        cval = params.get("cval", 0.0)
        apply_roi_only = params.get("apply_roi_only", False)
        logger.debug("Median filter params: size=%s, mode=%s, cval=%s, ROI Only=%s", size, mode, cval, apply_roi_only)
        try:
            processed = median_filter_lfa(image, size=size, mode=mode, cval=cval)
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Median filter failed", exc_info=exc)
            return None
        return self._apply_with_optional_roi(image, processed, apply_roi_only)


__all__ = ["MedianFilterDialog"]
