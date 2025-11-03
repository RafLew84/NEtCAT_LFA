"""Shared base dialog infrastructure for preprocessing operations."""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from PyQt6.QtWidgets import (
        QCheckBox,
        QDialog,
        QDialogButtonBox,
        QFrame,
        QLabel,
        QMessageBox,
        QPushButton,
        QRadioButton,
        QSizePolicy,
        QSpacerItem,
        QSpinBox,
        QDoubleSpinBox,
        QSlider,
        QVBoxLayout,
        QHBoxLayout,
        QWidget,
        QGroupBox,
        QComboBox,
    )
    from PyQt6.QtGui import QIntValidator
    from PyQt6.QtCore import Qt, pyqtSlot
    import pyqtgraph as pg
    from pyqtgraph import ImageItem, PlotItem, RectROI, ROI
except ImportError as exc:  # pragma: no cover
    logging.critical("Failed to import required Qt/pyqtgraph modules: %s", exc)
    raise

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
        except Exception as exc:  # pragma: no cover
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
        except Exception as exc:  # pragma: no cover
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

    # ------------------------------------------------------------------ abstract-ish interface
    def _get_current_parameters(self) -> Dict[str, Any]:
        """Return the current parameter dictionary."""
        raise NotImplementedError("Subclasses must implement _get_current_parameters()")

    def _apply_operation(self, image: np.ndarray, params: Dict[str, Any]) -> Optional[np.ndarray]:
        """Execute the underlying image processing operation."""
        raise NotImplementedError("Subclasses must implement _apply_operation()")


__all__ = [
    "BasePreprocessingDialog",
    "pg",
    "RectROI",
    "ROI",
    "ImageItem",
    "QCheckBox",
    "QDialogButtonBox",
    "QFrame",
    "QLabel",
    "QMessageBox",
    "QPushButton",
    "QRadioButton",
    "QSizePolicy",
    "QSpacerItem",
    "QSpinBox",
    "QDoubleSpinBox",
    "QSlider",
    "QVBoxLayout",
    "QHBoxLayout",
    "QWidget",
    "QGroupBox",
    "QComboBox",
    "QIntValidator",
    "Qt",
    "pyqtSlot",
    "np",
]
