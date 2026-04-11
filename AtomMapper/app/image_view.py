"""Legacy QLabel/QScrollArea STM viewport kept only as a fallback backend for tests."""

from __future__ import annotations

from typing import Optional

import numpy as np
from PyQt6.QtCore import QPoint, QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import (
    QImage,
    QMouseEvent,
    QPixmap,
    QResizeEvent,
    QWheelEvent,
)
from PyQt6.QtWidgets import QLabel, QRubberBand, QScrollArea, QVBoxLayout, QWidget

from .models import LoadedImage, ROIState


class _ZoomScrollArea(QScrollArea):
    """Scroll area that delegates mouse-wheel zoom to the parent viewport."""

    def __init__(self, viewport_widget: "STMImageViewport", parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._viewport_widget = viewport_widget

    def wheelEvent(self, event: QWheelEvent) -> None:
        if self._viewport_widget.current_loaded_image is None:
            super().wheelEvent(event)
            return
        self._viewport_widget.handle_wheel_delta(event.angleDelta().y(), event.position())
        event.accept()


class _ROIImageLabel(QLabel):
    """Pixmap label that draws and edits a rectangular ROI."""

    roi_state_edited = pyqtSignal(object)
    pan_delta_requested = pyqtSignal(object)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.current_loaded_image: Optional[LoadedImage] = None
        self.current_roi_state: Optional[ROIState] = None
        self._drag_mode: Optional[str] = None
        self._drag_start_pos: Optional[QPoint] = None
        self._drag_origin_roi: Optional[ROIState] = None
        self._handle_size = 10.0
        self._roi_band = QRubberBand(QRubberBand.Shape.Rectangle, self)
        self._roi_band.setStyleSheet(
            "border: 2px solid rgb(255, 180, 40); background: transparent;"
        )
        self._roi_band.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self._resize_handle = QWidget(self)
        self._resize_handle.setStyleSheet("background: rgb(255, 180, 40);")
        self._resize_handle.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self._resize_handle.hide()

    def set_loaded_image(self, loaded_image: Optional[LoadedImage]) -> None:
        self.current_loaded_image = loaded_image
        self.update_overlay_geometry()

    def set_roi_state(self, roi_state: Optional[ROIState]) -> None:
        self.current_roi_state = roi_state
        self.update_overlay_geometry()

    def roi_display_rect(self) -> Optional[QRectF]:
        image = self.current_loaded_image
        roi = self.current_roi_state
        if image is None or roi is None or self.width() <= 0 or self.height() <= 0:
            return None

        scale_x = self.width() / max(1, image.pixels_x)
        scale_y = self.height() / max(1, image.pixels_y)
        return QRectF(
            roi.x * scale_x,
            roi.y * scale_y,
            roi.width * scale_x,
            roi.height * scale_y,
        )

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.update_overlay_geometry()

    def update_overlay_geometry(self) -> None:
        roi_rect = self.roi_display_rect()
        if roi_rect is None:
            self._roi_band.hide()
            self._resize_handle.hide()
            return

        rect = roi_rect.toAlignedRect()
        self._roi_band.setGeometry(rect)
        self._roi_band.show()
        handle_size = int(round(self._handle_size))
        self._resize_handle.setGeometry(
            rect.right() - handle_size + 1,
            rect.bottom() - handle_size + 1,
            handle_size,
            handle_size,
        )
        self._resize_handle.show()

    def _clamped_event_pos(self, event: QMouseEvent) -> QPoint:
        """Clamp mouse positions to the real image-label bounds."""

        max_x = max(0, self.width() - 1)
        max_y = max(0, self.height() - 1)
        x = min(max(0, event.pos().x()), max_x)
        y = min(max(0, event.pos().y()), max_y)
        return QPoint(x, y)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.MiddleButton:
            self._drag_mode = "pan"
            self._drag_start_pos = event.pos()
            self._drag_origin_roi = None
            event.accept()
            return

        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return

        roi_rect = self.roi_display_rect()
        if roi_rect is None:
            self._drag_mode = "pan"
            self._drag_start_pos = event.pos()
            self._drag_origin_roi = None
            event.accept()
            return

        handle_rect = QRectF(
            roi_rect.right() - self._handle_size,
            roi_rect.bottom() - self._handle_size,
            self._handle_size,
            self._handle_size,
        )

        if handle_rect.contains(event.position()):
            self._drag_mode = "resize"
        elif roi_rect.contains(event.position()):
            self._drag_mode = "move"
        else:
            self._drag_mode = "pan"
            self._drag_start_pos = self._clamped_event_pos(event)
            self._drag_origin_roi = None
            event.accept()
            return

        self._drag_start_pos = self._clamped_event_pos(event)
        self._drag_origin_roi = self.current_roi_state
        event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if (
            self._drag_mode is None
        ):
            super().mouseMoveEvent(event)
            return

        if self._drag_mode == "pan":
            if self._drag_start_pos is None:
                super().mouseMoveEvent(event)
                return
            current_pos = self._clamped_event_pos(event)
            delta = current_pos - self._drag_start_pos
            self._drag_start_pos = current_pos
            self.pan_delta_requested.emit(delta)
            event.accept()
            return

        if (
            self._drag_start_pos is None
            or self._drag_origin_roi is None
            or self.current_loaded_image is None
        ):
            super().mouseMoveEvent(event)
            return

        scale_x = max(1e-9, self.width() / max(1, self.current_loaded_image.pixels_x))
        scale_y = max(1e-9, self.height() / max(1, self.current_loaded_image.pixels_y))
        current_pos = self._clamped_event_pos(event)
        delta = current_pos - self._drag_start_pos
        dx = int(round(delta.x() / scale_x))
        dy = int(round(delta.y() / scale_y))

        if self._drag_mode == "move":
            updated = ROIState(
                x=self._drag_origin_roi.x + dx,
                y=self._drag_origin_roi.y + dy,
                width=self._drag_origin_roi.width,
                height=self._drag_origin_roi.height,
            )
        else:
            updated = ROIState(
                x=self._drag_origin_roi.x,
                y=self._drag_origin_roi.y,
                width=self._drag_origin_roi.width + dx,
                height=self._drag_origin_roi.height + dy,
            )

        updated = updated.clamped(
            self.current_loaded_image.pixels_x,
            self.current_loaded_image.pixels_y,
        )
        self.current_roi_state = updated
        self.roi_state_edited.emit(updated)
        self.update()
        event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self._drag_mode = None
        self._drag_start_pos = None
        self._drag_origin_roi = None
        super().mouseReleaseEvent(event)


class STMImageViewport(QWidget):
    """Simple grayscale viewport for the currently selected STM image."""

    roi_state_edited = pyqtSignal(object)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.current_loaded_image: Optional[LoadedImage] = None
        self.current_roi_state: Optional[ROIState] = None
        self._base_pixmap: Optional[QPixmap] = None
        self._zoom_factor: float = 1.0

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.title_label = QLabel("STM image")
        self.title_label.setStyleSheet("font-size: 16px; font-weight: 600;")

        self.image_label = _ROIImageLabel("No image selected")
        self.image_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
        )
        self.image_label.setContentsMargins(0, 0, 0, 0)
        self.image_label.setMargin(0)
        self.image_label.setScaledContents(False)
        self.image_label.setStyleSheet("background: palette(base);")

        self.scroll_area = _ZoomScrollArea(self, self)
        self.scroll_area.setWidget(self.image_label)
        self.scroll_area.setWidgetResizable(False)
        self.scroll_area.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
        )
        self.scroll_area.setMinimumHeight(420)
        self.scroll_area.setStyleSheet("border: 1px solid palette(mid);")

        self.info_label = QLabel("Load or select an STM image to display it here.")
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.info_label.setStyleSheet("font-size: 12px; color: palette(mid);")

        layout.addWidget(self.title_label)
        layout.addWidget(self.scroll_area, 1)
        layout.addWidget(self.info_label)
        self.image_label.roi_state_edited.connect(self._on_roi_state_edited)
        self.image_label.pan_delta_requested.connect(self._pan_view)

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
            self.current_roi_state = None
            self.image_label.clear()
            self.image_label.setText("No image selected")
            self.image_label.adjustSize()
            self.image_label.setFixedSize(self.image_label.sizeHint())
            self.image_label.set_loaded_image(None)
            self.image_label.set_roi_state(None)
            self.info_label.setText("Load or select an STM image to display it here.")
            return

        self._base_pixmap = self._build_pixmap(loaded_image.image_data)
        self.image_label.set_loaded_image(loaded_image)
        self._apply_scaled_pixmap()
        self._update_info_label()

    def set_roi_state(self, roi_state: Optional[ROIState]) -> None:
        """Update the ROI overlay shown on top of the image."""

        self.current_roi_state = roi_state
        self.image_label.set_roi_state(roi_state)
        self._update_info_label()

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self._apply_scaled_pixmap()

    def handle_wheel_delta(self, delta_y: int, anchor_pos: Optional[QPointF] = None) -> None:
        """Adjust zoom based on the mouse-wheel delta."""

        if self._base_pixmap is None or delta_y == 0:
            return
        old_width = max(1, self.image_label.width())
        old_height = max(1, self.image_label.height())
        horizontal_bar = self.scroll_area.horizontalScrollBar()
        vertical_bar = self.scroll_area.verticalScrollBar()
        content_anchor_x = None
        content_anchor_y = None
        if anchor_pos is not None:
            content_anchor_x = float(horizontal_bar.value()) + float(anchor_pos.x())
            content_anchor_y = float(vertical_bar.value()) + float(anchor_pos.y())
            content_anchor_x = min(max(0.0, content_anchor_x), float(old_width))
            content_anchor_y = min(max(0.0, content_anchor_y), float(old_height))

        scale_step = 1.15 if delta_y > 0 else 1.0 / 1.15
        self._zoom_factor = max(0.25, min(self._zoom_factor * scale_step, 20.0))
        self._apply_scaled_pixmap()
        if anchor_pos is not None and content_anchor_x is not None and content_anchor_y is not None:
            new_width = max(1, self.image_label.width())
            new_height = max(1, self.image_label.height())
            relative_x = content_anchor_x / float(old_width)
            relative_y = content_anchor_y / float(old_height)
            new_content_x = relative_x * float(new_width)
            new_content_y = relative_y * float(new_height)
            horizontal_bar.setValue(int(round(new_content_x - float(anchor_pos.x()))))
            vertical_bar.setValue(int(round(new_content_y - float(anchor_pos.y()))))
        self._update_info_label()

    def _update_info_label(self) -> None:
        if self.current_loaded_image is None:
            self.info_label.setText("Load or select an STM image to display it here.")
            return

        roi_suffix = ""
        if self.current_roi_state is not None:
            roi_suffix = (
                f" | ROI x={self.current_roi_state.x} y={self.current_roi_state.y} "
                f"w={self.current_roi_state.width} h={self.current_roi_state.height}"
            )

        self.info_label.setText(
            f"{self.current_loaded_image.display_name} | "
            f"{self.current_loaded_image.pixels_x}x{self.current_loaded_image.pixels_y} px | "
            f"{self.current_loaded_image.size_nm_x:.3f} x {self.current_loaded_image.size_nm_y:.3f} nm | "
            f"zoom {self._zoom_factor:.2f}x"
            f"{roi_suffix}"
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
        self.image_label.setFixedSize(scaled.size())
        self.image_label.update()

    def _compute_fit_scale(self) -> float:
        if self._base_pixmap is None:
            return 1.0

        available_width = max(1, self.scroll_area.viewport().width())
        available_height = max(1, self.scroll_area.viewport().height())
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

    def _on_roi_state_edited(self, roi_state: ROIState) -> None:
        self.current_roi_state = roi_state
        self._update_info_label()
        self.roi_state_edited.emit(roi_state)

    def _pan_view(self, delta: QPoint) -> None:
        self.scroll_area.horizontalScrollBar().setValue(
            self.scroll_area.horizontalScrollBar().value() - delta.x()
        )
        self.scroll_area.verticalScrollBar().setValue(
            self.scroll_area.verticalScrollBar().value() - delta.y()
        )
