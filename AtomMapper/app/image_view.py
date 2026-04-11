"""Image viewport widget for displaying STM images in AtomMapper."""

from __future__ import annotations

from typing import Optional

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap, QResizeEvent, QWheelEvent
from PyQt6.QtWidgets import QLabel, QScrollArea, QVBoxLayout, QWidget

from .models import LoadedImage


class _ZoomScrollArea(QScrollArea):
    """Scroll area that delegates mouse-wheel zoom to the parent viewport."""

    def __init__(self, viewport_widget: "STMImageViewport", parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._viewport_widget = viewport_widget

    def wheelEvent(self, event: QWheelEvent) -> None:
        if self._viewport_widget.current_loaded_image is None:
            super().wheelEvent(event)
            return
        self._viewport_widget.handle_wheel_delta(event.angleDelta().y())
        event.accept()


class STMImageViewport(QWidget):
    """Simple grayscale viewport for the currently selected STM image."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.current_loaded_image: Optional[LoadedImage] = None
        self._base_pixmap: Optional[QPixmap] = None
        self._zoom_factor: float = 1.0

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.title_label = QLabel("STM image")
        self.title_label.setStyleSheet("font-size: 16px; font-weight: 600;")

        self.image_label = QLabel("No image selected")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet(
            "border: 1px solid palette(mid); background: palette(base); padding: 12px;"
        )

        self.scroll_area = _ZoomScrollArea(self, self)
        self.scroll_area.setWidget(self.image_label)
        self.scroll_area.setWidgetResizable(False)
        self.scroll_area.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.scroll_area.setMinimumHeight(420)
        self.scroll_area.setStyleSheet("border: 1px solid palette(mid);")

        self.info_label = QLabel("Load or select an STM image to display it here.")
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.info_label.setStyleSheet("font-size: 12px; color: palette(mid);")

        layout.addWidget(self.title_label)
        layout.addWidget(self.scroll_area, 1)
        layout.addWidget(self.info_label)

    @property
    def zoom_factor(self) -> float:
        """Return the user-controlled zoom multiplier."""

        return self._zoom_factor

    def set_loaded_image(self, loaded_image: Optional[LoadedImage]) -> None:
        """Render the provided image or show an empty-state placeholder."""

        self.current_loaded_image = loaded_image
        self._zoom_factor = 1.0
        if loaded_image is None:
            self._base_pixmap = None
            self.image_label.clear()
            self.image_label.setText("No image selected")
            self.image_label.adjustSize()
            self.info_label.setText("Load or select an STM image to display it here.")
            return

        self._base_pixmap = self._build_pixmap(loaded_image.image_data)
        self._apply_scaled_pixmap()
        self._update_info_label()

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self._apply_scaled_pixmap()

    def handle_wheel_delta(self, delta_y: int) -> None:
        """Adjust zoom based on the mouse-wheel delta."""

        if self._base_pixmap is None or delta_y == 0:
            return
        scale_step = 1.15 if delta_y > 0 else 1.0 / 1.15
        self._zoom_factor = max(0.25, min(self._zoom_factor * scale_step, 20.0))
        self._apply_scaled_pixmap()
        self._update_info_label()

    def _update_info_label(self) -> None:
        if self.current_loaded_image is None:
            self.info_label.setText("Load or select an STM image to display it here.")
            return

        self.info_label.setText(
            f"{self.current_loaded_image.display_name} | "
            f"{self.current_loaded_image.pixels_x}x{self.current_loaded_image.pixels_y} px | "
            f"{self.current_loaded_image.size_nm_x:.3f} x {self.current_loaded_image.size_nm_y:.3f} nm | "
            f"zoom {self._zoom_factor:.2f}x"
        )

    def _apply_scaled_pixmap(self) -> None:
        if self._base_pixmap is None:
            return

        fit_scale = self._compute_fit_scale()
        scale = fit_scale * self._zoom_factor
        scaled = self._base_pixmap.scaled(
            max(1, int(round(self._base_pixmap.width() * scale))),
            max(1, int(round(self._base_pixmap.height() * scale))),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.image_label.setPixmap(scaled)
        self.image_label.resize(scaled.size())
        self.image_label.setMinimumSize(scaled.size())

    def _compute_fit_scale(self) -> float:
        if self._base_pixmap is None:
            return 1.0

        available_width = max(1, self.scroll_area.viewport().width() - 8)
        available_height = max(1, self.scroll_area.viewport().height() - 8)
        pixmap_width = max(1, self._base_pixmap.width())
        pixmap_height = max(1, self._base_pixmap.height())
        return min(available_width / pixmap_width, available_height / pixmap_height)

    @staticmethod
    def _build_pixmap(image_data: np.ndarray) -> QPixmap:
        image_array = np.asarray(image_data, dtype=float)
        if image_array.ndim != 2:
            raise ValueError(f"Expected 2D image data, got shape {image_array.shape!r}.")

        finite_mask = np.isfinite(image_array)
        if not finite_mask.any():
            normalized = np.zeros(image_array.shape, dtype=np.uint8)
        else:
            finite_values = image_array[finite_mask]
            min_value = float(finite_values.min())
            max_value = float(finite_values.max())
            if max_value - min_value < 1e-12:
                normalized_float = np.zeros(image_array.shape, dtype=float)
            else:
                normalized_float = (image_array - min_value) / (max_value - min_value)
                normalized_float = np.clip(normalized_float, 0.0, 1.0)
            normalized_float = np.where(finite_mask, normalized_float, 0.0)
            normalized = np.round(normalized_float * 255.0).astype(np.uint8)

        height, width = normalized.shape
        bytes_per_line = normalized.strides[0]
        qimage = QImage(
            normalized.data,
            width,
            height,
            bytes_per_line,
            QImage.Format.Format_Grayscale8,
        )
        return QPixmap.fromImage(qimage.copy())
