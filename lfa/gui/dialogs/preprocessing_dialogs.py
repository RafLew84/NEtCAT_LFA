# lfa/gui/dialogs/preprocessing_dialogs.py
import logging
import numpy as np
from typing import Optional, Tuple, Dict, Any, List

try:
    from PyQt6.QtWidgets import (
        QDialog, QVBoxLayout, QHBoxLayout, QSlider, QLabel, QCheckBox,
        QDialogButtonBox, QWidget, QSizePolicy, QSpacerItem, QFrame, QMessageBox, 
        QGroupBox, QRadioButton, QPushButton, QComboBox, QSpinBox, QDoubleSpinBox
    )
    from PyQt6.QtGui import QIntValidator
    from PyQt6.QtCore import Qt, pyqtSlot
    import pyqtgraph as pg
    from pyqtgraph import PlotItem, RectROI, ROI, ImageItem
except ImportError as e:
    logging.critical(f"Failed to import necessary Qt or pyqtgraph modules: {e}")
    raise

# Import image processing functions
try:
    from lfa.preprocessing.filtering import gaussian_blur, median_filter_lfa, gaussian_sharpen_unsharp_mask
except ImportError:
    logging.error("Could not import gaussian_blur function.")
    def gaussian_blur(image, sigma): return image 
    def median_filter_lfa(image, size, mode='reflect', cval=0.0): return image 
    def gaussian_sharpen_unsharp_mask(image, radius, amount): return image

try:
    from lfa.preprocessing.leveling import fit_plane, fit_plane_3pts, level_by_plane
except ImportError:
    logging.error("Could not import fit_plane, fit_plane_3pts, or level_by_plane functions.")
    def fit_plane(*args, **kwargs): return None
    def fit_plane_3pts(*args, **kwargs): return None
    def level_by_plane(*args, **kwargs): return None

try: 
    from lfa.preprocessing.denoising import denoise_nlmeans_skimage, denoise_bm3d_lfa
except ImportError: 
    logging.error("Could not import denoising functions."); 
    def denoise_nlmeans_skimage(**kwargs): return kwargs.get('image')
    def denoise_bm3d_lfa(**kwargs): return kwargs.get('image')

logger = logging.getLogger(__name__)

class BasePreprocessingDialog(QDialog):
    """
    Shared functionality for preprocessing dialogs.

    Subclasses are responsible for:
      * creating the PyQtGraph views and associated controls
      * setting ``self.original_data`` and ``self.preview_data`` before calling
        :meth:`_initialize_common_behavior`
      * implementing :meth:`_get_current_parameters` and :meth:`_apply_operation`
    """

    def __init__(self, operation_name: str, parent=None):
        super().__init__(parent)
        self.operation_name = operation_name
        self.original_data: Optional[np.ndarray] = None
        self.preview_data: Optional[np.ndarray] = None
        self._final_processed_data: Optional[np.ndarray] = None
        self._final_params: Dict[str, Any] = {}
        self._final_is_roi_applied_only: bool = False
        self._manage_roi_with_checkbox: bool = True

    # ------------------------------------------------------------------ public helpers
    def _initialize_common_behavior(self) -> None:
        """
        Call once subclasses have constructed the UI widgets.
        Connects the shared signals/slots, updates the initial ROI label and
        triggers the first preview.
        """
        if getattr(self, "apply_to_roi_only_checkbox", None) is not None:
            self.apply_to_roi_only_checkbox.stateChanged.connect(self._on_parameter_or_preview_changed)
        if getattr(self, "live_preview_checkbox", None) is not None:
            self.live_preview_checkbox.stateChanged.connect(self._on_parameter_or_preview_changed)
        if getattr(self, "roi", None) is not None:
            self.roi.sigRegionChanged.connect(self._on_roi_changed)
        if getattr(self, "button_box", None) is not None:
            self.button_box.accepted.connect(self.accept)
            self.button_box.rejected.connect(self.reject)

        self._update_roi_visibility()
        self._update_roi_label()
        self.update_original_view()
        self._update_preview()
        logger.debug("%s: common preprocessing behaviour initialised.", self.operation_name)

    # ------------------------------------------------------------------ overridable hooks
    def _handle_parameter_widget_change(self) -> None:
        """Hook for subclasses to update parameter labels when widgets change."""
        return

    # ------------------------------------------------------------------ shared slots/logic
    def _on_parameter_or_preview_changed(self) -> None:
        self._handle_parameter_widget_change()
        if self._manage_roi_with_checkbox:
            self._update_roi_visibility()
        if getattr(self, "live_preview_checkbox", None) is not None and self.live_preview_checkbox.isChecked():
            self._update_preview()

    def _update_roi_visibility(self) -> None:
        if getattr(self, "roi", None) is None or getattr(self, "apply_to_roi_only_checkbox", None) is None:
            return
        is_roi_mode = self.apply_to_roi_only_checkbox.isChecked()
        self.roi.setVisible(is_roi_mode)
        if getattr(self, "roi_info_label", None) is not None:
            self.roi_info_label.setVisible(is_roi_mode)

    def _on_roi_changed(self) -> None:
        self._update_roi_label()
        if (
            getattr(self, "apply_to_roi_only_checkbox", None) is not None
            and self.apply_to_roi_only_checkbox.isChecked()
            and getattr(self, "live_preview_checkbox", None) is not None
            and self.live_preview_checkbox.isChecked()
        ):
            self._update_preview()

    def _update_roi_label(self) -> None:
        if getattr(self, "roi_info_label", None) is None or getattr(self, "roi", None) is None:
            return
        if not self.roi.isVisible():
            self.roi_info_label.setText("ROI: Not selected")
            return
        pos = self.roi.pos()
        size = self.roi.size()
        self.roi_info_label.setText(
            f"ROI: ({pos.x():.1f}, {pos.y():.1f}) Size: ({size.x():.1f}, {size.y():.1f})"
        )

    def _get_roi_slice(self) -> Optional[Tuple[slice, slice]]:
        if getattr(self, "roi", None) is None or not self.roi.isVisible():
            return None
        size = self.roi.size()
        if not (size.x() > 0 and size.y() > 0):
            return None
        pos = self.roi.pos()
        height, width = self.original_data.shape if self.original_data is not None else (0, 0)
        x0, y0 = int(round(pos.x())), int(round(pos.y()))
        w, h = int(round(size.x())), int(round(size.y()))
        x1, y1 = min(x0 + w, width), min(y0 + h, height)
        x0, y0 = max(0, x0), max(0, y0)
        if x1 > x0 and y1 > y0:
            return slice(y0, y1), slice(x0, x1)
        logger.warning("%s: invalid ROI dimensions.", self.operation_name)
        return None

    def _apply_with_optional_roi(
        self,
        original: np.ndarray,
        processed: Optional[np.ndarray],
        apply_roi_only: bool,
    ) -> Optional[np.ndarray]:
        """
        Merge ``processed`` with ``original`` when ROI-only mode is enabled.

        Args:
            original: Source image that should remain untouched outside the ROI.
            processed: Full-frame processed image. May be ``None`` if the operation failed.
            apply_roi_only: Whether the result should only replace the selected ROI.

        Returns:
            A numpy array ready to display/apply, or ``None`` if ``processed`` is ``None``.
        """
        if processed is None:
            return None
        if not apply_roi_only:
            return processed
        roi_slice = self._get_roi_slice()
        if roi_slice is None:
            logger.warning(
                "%s: ROI-only requested but ROI is invalid; returning original data.",
                self.operation_name,
            )
            return original
        result = original.copy()
        result[roi_slice] = processed[roi_slice]
        return result

    def _update_preview(self) -> None:
        if self.original_data is None:
            return
        params = self._get_current_parameters()
        logger.debug("%s: updating preview with params=%s", self.operation_name, params)
        preview = None
        try:
            preview = self._apply_operation(self.original_data, params)
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("%s: error during preview computation", self.operation_name, exc_info=exc)
        if preview is None:
            preview = self.original_data.copy()
        self.preview_data = preview
        self.update_preview_view()

    def update_original_view(self) -> None:
        if self.original_data is not None and getattr(self, "img_original", None):
            self.img_original.setImage(self.original_data.T)
            if getattr(self, "plot_original", None):
                self.plot_original.autoRange()

    def update_preview_view(self) -> None:
        if getattr(self, "img_processed", None) is None:
            return
        if self.preview_data is not None:
            self.img_processed.setImage(self.preview_data.T)
        else:
            self.img_processed.clear()

    # ------------------------------------------------------------------ dialog lifecycle
    def accept(self) -> None:
        params = self._get_current_parameters()
        self._final_params = params
        self._final_is_roi_applied_only = params.get("apply_roi_only", False)
        logger.info("%s: applying operation (ROI only=%s)", self.operation_name, self._final_is_roi_applied_only)
        try:
            result = self._apply_operation(self.original_data, params)
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("%s: error computing final result", self.operation_name, exc_info=exc)
            result = None
        if result is None:
            QMessageBox.critical(self, "Error", f"{self.operation_name} failed. See logs for details.")
            self._final_processed_data = None
            self._final_is_roi_applied_only = False
            super().reject()
            return
        if np.allclose(result, self.original_data):
            logger.info("%s: data unchanged; dialog will be rejected.", self.operation_name)
            self._final_processed_data = None
            super().reject()
            return
        self._final_processed_data = result
        super().accept()

    def reject(self) -> None:
        logger.info("%s dialog rejected.", self.operation_name)
        self._final_processed_data = None
        super().reject()

    # ------------------------------------------------------------------ result helpers
    def get_processed_data(self) -> Optional[np.ndarray]:
        if self._final_processed_data is None:
            return None
        return self._final_processed_data.copy()

    def get_parameters(self) -> Dict[str, Any]:
        return self._final_params if self._final_params else self._get_current_parameters()

    def was_roi_applied_only(self) -> bool:
        return self._final_is_roi_applied_only

    def get_final_roi_slice(self) -> Optional[Tuple[slice, slice]]:
        return self._get_roi_slice() if self._final_is_roi_applied_only else None

    # ------------------------------------------------------------------ abstract interface
    def _get_current_parameters(self) -> Dict[str, Any]:
        """Return the current parameter dictionary."""
        raise NotImplementedError("Subclasses must implement _get_current_parameters()")

    def _apply_operation(self, image: np.ndarray, params: Dict[str, Any]) -> Optional[np.ndarray]:
        """Execute the underlying image processing operation."""
        raise NotImplementedError("Subclasses must implement _apply_operation()")



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
        self.sigma_slider = QSlider(Qt.Orientation.Horizontal)
        self.sigma_slider.setRange(0, 100)
        self.sigma_slider.setTickInterval(10)
        self.sigma_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.sigma_slider.setValue(0)
        sigma_layout.addWidget(self.sigma_slider)
        self.sigma_value_label = QLabel("0.0")
        sigma_layout.addWidget(self.sigma_value_label)
        layout.addLayout(sigma_layout)

        self.sigma_slider.valueChanged.connect(self._on_parameter_or_preview_changed)

    def _handle_parameter_widget_change(self) -> None:
        sigma = self.sigma_slider.value() / 10.0
        self.sigma_value_label.setText(f"{sigma:.1f}")

    def _get_current_parameters(self) -> Dict[str, Any]:
        sigma = self.sigma_slider.value() / 10.0
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
        self.radius_slider = QSlider(Qt.Orientation.Horizontal)
        self.radius_slider.setRange(0, 100)  # maps to 0.0 - 10.0
        self.radius_slider.setTickInterval(10)
        self.radius_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.radius_slider.setValue(10)
        radius_layout.addWidget(self.radius_slider)
        self.radius_value_label = QLabel("1.0")
        radius_layout.addWidget(self.radius_value_label)
        layout.addLayout(radius_layout)

        amount_layout = QHBoxLayout()
        amount_layout.addWidget(QLabel("Amount:"))
        self.amount_slider = QSlider(Qt.Orientation.Horizontal)
        self.amount_slider.setRange(0, 50)  # maps to 0.0 - 5.0
        self.amount_slider.setTickInterval(5)
        self.amount_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.amount_slider.setValue(10)
        amount_layout.addWidget(self.amount_slider)
        self.amount_value_label = QLabel("1.0")
        amount_layout.addWidget(self.amount_value_label)
        layout.addLayout(amount_layout)

        self.radius_slider.valueChanged.connect(self._on_parameter_or_preview_changed)
        self.amount_slider.valueChanged.connect(self._on_parameter_or_preview_changed)

    def _handle_parameter_widget_change(self) -> None:
        radius = self.radius_slider.value() / 10.0
        amount = self.amount_slider.value() / 10.0
        self.radius_value_label.setText(f"{radius:.1f}")
        self.amount_value_label.setText(f"{amount:.1f}")

    def _get_current_parameters(self) -> Dict[str, Any]:
        radius = self.radius_slider.value() / 10.0
        amount = self.amount_slider.value() / 10.0
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



class NLMeansDialog(BasePreprocessingDialog):
    """Dialog for scikit-image Non-Local Means denoising with ROI support."""

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
        sigma_layout.addWidget(QLabel("Sigma:"))
        self.sigma_spinbox = QDoubleSpinBox()
        self.sigma_spinbox.setDecimals(4)
        self.sigma_spinbox.setRange(0.0001, 1_000_000.0)
        estimated_sigma = float(np.std(self.original_data) / 10.0)
        self.sigma_spinbox.setValue(max(0.001, estimated_sigma))
        self.sigma_spinbox.setSingleStep(0.01)
        sigma_layout.addWidget(self.sigma_spinbox)
        layout.addLayout(sigma_layout)

        h_mult_layout = QHBoxLayout()
        h_mult_layout.addWidget(QLabel("H multiplier:"))
        self.h_mult_spinbox = QDoubleSpinBox()
        self.h_mult_spinbox.setDecimals(2)
        self.h_mult_spinbox.setRange(0.1, 5.0)
        self.h_mult_spinbox.setValue(1.0)
        self.h_mult_spinbox.setSingleStep(0.05)
        h_mult_layout.addWidget(self.h_mult_spinbox)
        layout.addLayout(h_mult_layout)

        patch_size_layout = QHBoxLayout()
        patch_size_layout.addWidget(QLabel("Patch size:"))
        self.patch_size_spinbox = QSpinBox()
        self.patch_size_spinbox.setRange(3, 21)
        self.patch_size_spinbox.setSingleStep(2)
        self.patch_size_spinbox.setValue(7)
        patch_size_layout.addWidget(self.patch_size_spinbox)
        layout.addLayout(patch_size_layout)

        patch_dist_layout = QHBoxLayout()
        patch_dist_layout.addWidget(QLabel("Patch distance:"))
        self.patch_dist_spinbox = QSpinBox()
        self.patch_dist_spinbox.setRange(1, 100)
        self.patch_dist_spinbox.setValue(11)
        patch_dist_layout.addWidget(self.patch_dist_spinbox)
        layout.addLayout(patch_dist_layout)

        self.fast_mode_checkbox = QCheckBox("Use Fast Mode")
        self.fast_mode_checkbox.setChecked(True)
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
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("NL-Means denoising failed", exc_info=exc)
            return None
        return self._apply_with_optional_roi(
            image,
            processed,
            params.get("apply_roi_only", False),
        )



class PlaneLevelingDialog(BasePreprocessingDialog):
    """Dialog for plane leveling with whole, ROI, and 3-point modes."""

    def __init__(self, original_data: np.ndarray, parent=None):
        if original_data is None:
            raise ValueError("Original data cannot be None")
        super().__init__("Plane Leveling", parent)
        self.original_data = original_data.astype(np.float32)
        self.preview_data = self.original_data.copy()

        self._is_selecting_points = False
        self._selected_points: List[Tuple[int, int]] = []
        self._mouse_click_connection = None

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
        controls_panel.setMaximumWidth(260)
        controls_panel.setLayout(controls_area_layout)

        parameter_widget_container = QWidget()
        parameter_layout = QVBoxLayout(parameter_widget_container)
        parameter_layout.setContentsMargins(0, 0, 0, 0)
        self._create_parameter_controls(parameter_layout)
        controls_area_layout.addWidget(parameter_widget_container)

        controls_area_layout.addWidget(
            QFrame(frameShape=QFrame.Shape.HLine, frameShadow=QFrame.Shadow.Sunken)
        )

        self.apply_to_roi_only_checkbox = QCheckBox("Apply plane only to ROI area")
        self.apply_to_roi_only_checkbox.setChecked(False)
        controls_area_layout.addWidget(self.apply_to_roi_only_checkbox)

        self.live_preview_checkbox = QCheckBox("Live Preview")
        self.live_preview_checkbox.setChecked(True)
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
            pen=pg.mkPen("g", width=2),
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

        # The base class should not auto-manage ROI visibility; the mode handler handles it.
        self._manage_roi_with_checkbox = False
        self._initialize_common_behavior()
        self._update_mode_ui()

    def _create_parameter_controls(self, layout: QVBoxLayout) -> None:
        mode_groupbox = QGroupBox("Leveling Mode")
        mode_layout = QVBoxLayout(mode_groupbox)

        self.rb_whole = QRadioButton("Whole image")
        self.rb_roi = QRadioButton("Fit plane to ROI")
        self.rb_3pt = QRadioButton("Fit plane through 3 points")
        self.rb_whole.setChecked(True)

        for rb in (self.rb_whole, self.rb_roi, self.rb_3pt):
            mode_layout.addWidget(rb)
            rb.toggled.connect(self._on_mode_changed)

        layout.addWidget(mode_groupbox)

        self.points_groupbox = QGroupBox("3-point selection")
        points_layout = QVBoxLayout(self.points_groupbox)
        self.point_coords_labels: List[QLabel] = []
        for idx in range(3):
            label = QLabel(f"P{idx + 1}: -")
            points_layout.addWidget(label)
            self.point_coords_labels.append(label)

        buttons_row = QHBoxLayout()
        self.select_points_button = QPushButton("Select")
        self.select_points_button.clicked.connect(self._toggle_point_selection)
        buttons_row.addWidget(self.select_points_button)

        self.clear_points_button = QPushButton("Clear")
        self.clear_points_button.setEnabled(False)
        self.clear_points_button.clicked.connect(self._clear_points)
        buttons_row.addWidget(self.clear_points_button)

        points_layout.addLayout(buttons_row)
        layout.addWidget(self.points_groupbox)

    def _update_mode_ui(self) -> None:
        is_roi_mode = self.rb_roi.isChecked()
        is_3pt_mode = self.rb_3pt.isChecked()

        self.roi.setVisible(is_roi_mode)
        self.roi_info_label.setVisible(is_roi_mode)
        self.apply_to_roi_only_checkbox.setVisible(is_roi_mode)
        if not is_roi_mode:
            self.apply_to_roi_only_checkbox.setChecked(False)

        self.points_groupbox.setVisible(is_3pt_mode)
        if not is_3pt_mode:
            self._toggle_point_selection(force_off=True)
            self._clear_points(trigger_preview=False)

    def _on_mode_changed(self) -> None:
        self._update_mode_ui()
        if self.live_preview_checkbox.isChecked():
            self._update_preview()

    def _toggle_point_selection(self, force_off: bool = False) -> None:
        enable = not force_off and not self._is_selecting_points
        if enable:
            self._is_selecting_points = True
            self.select_points_button.setText("Stop selecting")
            vb = self.plot_original.getViewBox()
            scene = vb.scene() if vb is not None else None
            if scene is not None and hasattr(scene, "sigMouseClicked"):
                self._mouse_click_connection = scene.sigMouseClicked.connect(self._handle_mouse_click)
        else:
            if self._mouse_click_connection is not None:
                try:
                    vb = self.plot_original.getViewBox()
                    scene = vb.scene() if vb is not None else None
                    if scene is not None and hasattr(scene, "sigMouseClicked"):
                        scene.sigMouseClicked.disconnect(self._mouse_click_connection)
                except Exception as exc:  # pragma: no cover - defensive
                    logger.warning("PlaneLeveling: failed to disconnect point selection: %s", exc)
                self._mouse_click_connection = None
            self._is_selecting_points = False
            self.select_points_button.setText("Select")

    def _clear_points(self, trigger_preview: bool = True) -> None:
        self._selected_points.clear()
        for idx, label in enumerate(self.point_coords_labels, start=1):
            label.setText(f"P{idx}: -")
        self.clear_points_button.setEnabled(False)
        if trigger_preview and self.live_preview_checkbox.isChecked():
            self._update_preview()

    def _handle_mouse_click(self, event) -> None:
        if not self._is_selecting_points or len(self._selected_points) >= 3:
            return
        vb = self.plot_original.getViewBox()
        if vb is None:
            return
        pos_data = vb.mapSceneToView(event.scenePos())
        x = int(round(pos_data.x()))
        y = int(round(pos_data.y()))
        height, width = self.original_data.shape
        if 0 <= x < width and 0 <= y < height:
            self._selected_points.append((x, y))
            idx = len(self._selected_points)
            self.point_coords_labels[idx - 1].setText(f"P{idx}: ({x}, {y})")
            self.clear_points_button.setEnabled(True)
            logger.info("PlaneLeveling: point %s added at (%s, %s)", idx, x, y)
            if len(self._selected_points) == 3:
                self._toggle_point_selection(force_off=True)
                if self.live_preview_checkbox.isChecked():
                    self._update_preview()
        else:
            logger.warning("PlaneLeveling: click (%s, %s) outside bounds", x, y)

    def _get_current_parameters(self) -> Dict[str, Any]:
        mode = "whole"
        if self.rb_roi.isChecked():
            mode = "roi"
        elif self.rb_3pt.isChecked():
            mode = "3pt"

        params: Dict[str, Any] = {"mode": mode, "apply_roi_only": False}
        if mode == "roi":
            params["apply_roi_only"] = self.apply_to_roi_only_checkbox.isChecked()
        elif mode == "3pt":
            params["points"] = self._selected_points.copy()
        return params

    def _calculate_leveled_image(self, image_in: np.ndarray, params: Dict[str, Any]) -> Optional[np.ndarray]:
        mode = params.get("mode", "whole")
        apply_roi_only = params.get("apply_roi_only", False)
        roi_slice = None

        try:
            if mode == "whole":
                fitted_plane = fit_plane(image_in, roi_slice=None)
            elif mode == "roi":
                roi_slice = self._get_roi_slice()
                if roi_slice is None:
                    logger.warning("PlaneLeveling: ROI mode selected but ROI invalid.")
                    return image_in
                fitted_plane = fit_plane(image_in, roi_slice=roi_slice)
            elif mode == "3pt":
                points = params.get("points", [])
                if len(points) != 3:
                    logger.warning("PlaneLeveling: 3pt mode without 3 points.")
                    return image_in
                fitted_plane = fit_plane_3pts(image_in, points)
            else:
                logger.error("PlaneLeveling: unknown mode %s", mode)
                return image_in
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("PlaneLeveling: plane fitting failed", exc_info=exc)
            return None

        if fitted_plane is None:
            return None

        if mode == "roi" and apply_roi_only and roi_slice is not None:
            leveled = image_in.copy()
            leveled[roi_slice] = image_in[roi_slice] - fitted_plane[roi_slice]
            return leveled

        return image_in - fitted_plane

    def _apply_operation(self, image: np.ndarray, params: Dict[str, Any]) -> Optional[np.ndarray]:
        return self._calculate_leveled_image(image, params)

    def accept(self) -> None:
        params = self._get_current_parameters()
        if params.get("mode") == "3pt" and len(params.get("points", [])) != 3:
            QMessageBox.warning(self, "Missing Points", "Please select exactly 3 points.")
            return
        result = self._calculate_leveled_image(self.original_data, params)
        if result is None:
            QMessageBox.critical(self, "Error", "Plane leveling failed. See logs for details.")
            self._final_processed_data = None
            self._final_is_roi_applied_only = False
            super().reject()
            return
        if np.allclose(result, self.original_data):
            logger.info("PlaneLeveling: data unchanged; rejecting dialog.")
            self._final_processed_data = None
            super().reject()
            return
        self._final_processed_data = result
        self._final_params = params
        self._final_is_roi_applied_only = params.get("apply_roi_only", False)
        super().accept()

    def reject(self) -> None:
        self._toggle_point_selection(force_off=True)
        super().reject()

    def _on_roi_changed(self) -> None:
        super()._on_roi_changed()
        if self.rb_roi.isChecked() and self.live_preview_checkbox.isChecked():
            self._update_preview()

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
        self.sigma_spinbox = QDoubleSpinBox()
        self.sigma_spinbox.setDecimals(4)
        self.sigma_spinbox.setRange(0.0001, 1.0)
        self.sigma_spinbox.setValue(0.05)
        self.sigma_spinbox.setSingleStep(0.005)
        self.sigma_spinbox.setToolTip("Estimated noise standard deviation relative to [0, 1] range.")
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
        logger.debug("BM3D parameters: sigma=%s, ROI Only=%s", sigma, apply_roi_only)
        try:
            processed = denoise_bm3d_lfa(image, sigma_psd=sigma)
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("BM3D denoising failed", exc_info=exc)
            return None
        return self._apply_with_optional_roi(image, processed, apply_roi_only)

    def _handle_parameter_widget_change(self) -> None:
        roi_enabled = self.apply_to_roi_only_checkbox.isChecked()
        self.roi.setVisible(roi_enabled)
        self.roi_info_label.setVisible(roi_enabled)


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
        controls_area_layout.addWidget(self.apply_to_roi_only_checkbox)

        self.live_preview_checkbox = QCheckBox("Live Preview")
        self.live_preview_checkbox.setChecked(True)
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
        self.size_spinbox = QSpinBox()
        self.size_spinbox.setRange(1, 31)
        self.size_spinbox.setSingleStep(2)
        self.size_spinbox.setValue(3)
        size_layout.addWidget(self.size_spinbox)
        layout.addLayout(size_layout)

        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Mode:"))
        self.mode_combobox = QComboBox()
        self.valid_modes = ["reflect", "constant", "nearest", "mirror", "wrap"]
        self.mode_combobox.addItems(self.valid_modes)
        self.mode_combobox.setCurrentText("reflect")
        mode_layout.addWidget(self.mode_combobox)
        layout.addLayout(mode_layout)

        cval_layout = QHBoxLayout()
        self.cval_label = QLabel("Constant value:")
        self.cval_spinbox = QDoubleSpinBox()
        self.cval_spinbox.setRange(-1_000_000.0, 1_000_000.0)
        self.cval_spinbox.setDecimals(3)
        self.cval_spinbox.setValue(0.0)
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


