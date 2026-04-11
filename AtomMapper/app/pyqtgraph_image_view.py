"""pyqtgraph ImageView-based STM viewport aligned with the LFA main-image control."""

from __future__ import annotations

from typing import Optional

import numpy as np

from PyQt6.QtCore import QRectF, Qt, pyqtSignal
from PyQt6.QtWidgets import QLabel, QStackedWidget, QVBoxLayout, QWidget

from .models import LoadedImage, ROIState

try:
    import pyqtgraph as pg
except ImportError:  # pragma: no cover - exercised only in missing-dependency environments
    pg = None


class PyQtGraphSTMViewport(QWidget):
    """STM viewport backed by pyqtgraph.ImageView, mirroring the LFA display control."""

    roi_state_edited = pyqtSignal(object)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.current_loaded_image: Optional[LoadedImage] = None
        self.current_roi_state: Optional[ROIState] = None
        self.backend_available: bool = pg is not None
        self._syncing_roi_overlay = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.title_label = QLabel("STM image")
        self.title_label.setStyleSheet("font-size: 16px; font-weight: 600;")

        self.stack = QStackedWidget(self)

        self.placeholder_label = QLabel("No image selected")
        self.placeholder_label.setObjectName("atommapper_pg_view_placeholder")
        self.placeholder_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.placeholder_label.setWordWrap(True)
        self.placeholder_label.setStyleSheet(
            "border: 1px solid palette(mid); color: palette(mid); padding: 24px;"
        )

        self.image_view = None
        self.plot_widget = None
        self.image_item = None
        self.histogram_widget = None
        self.roi_item = None
        if self.backend_available:
            self.image_view = pg.ImageView(self)
            self.image_view.setObjectName("atommapper_pg_image_view")
            self.plot_widget = self.image_view
            self.image_item = self.image_view.getImageItem()
            self.histogram_widget = self.image_view.getHistogramWidget()
            self.view_box.invertY(True)

            self.roi_item = pg.RectROI(
                [0.0, 0.0],
                [4.0, 4.0],
                pen=pg.mkPen(color=(255, 180, 40), width=2),
                movable=True,
                rotatable=False,
                resizable=True,
            )
            self.roi_item.hide()
            self.roi_item.sigRegionChanged.connect(self._on_roi_item_changed)
            self.view_box.addItem(self.roi_item)
            self.stack.addWidget(self.image_view)

        self.info_label = QLabel(
            "pyqtgraph ImageView viewport ready. Histogram/LUT controls match the LFA-style image display."
        )
        self.info_label.setWordWrap(True)
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.info_label.setStyleSheet("font-size: 12px; color: palette(mid);")

        self.stack.addWidget(self.placeholder_label)
        self._show_placeholder("No image selected")

        layout.addWidget(self.title_label)
        layout.addWidget(self.stack, 1)
        layout.addWidget(self.info_label)

    @property
    def view_box(self):
        """Return the active pyqtgraph view box when the backend is available."""

        if self.image_view is None:
            return None
        return self.image_view.getView()

    def set_loaded_image(self, loaded_image: Optional[LoadedImage]) -> None:
        """Store the active image and toggle between placeholder and backend canvas."""

        self.current_loaded_image = loaded_image
        if loaded_image is None:
            self._clear_image_view()
            self.current_roi_state = None
            self._show_placeholder("No image selected")
            return

        if not self.backend_available or self.image_view is None or self.image_item is None:
            self._show_placeholder("pyqtgraph backend is not available.")
            return

        image_array = np.asarray(loaded_image.image_data, dtype=np.float32)
        if image_array.ndim != 2:
            raise ValueError(f"Expected 2D image data, got shape {image_array.shape!r}.")

        self.image_item.setImage(image_array.T, autoLevels=True)
        self.reset_view()
        self._update_roi_overlay_from_state()
        self.stack.setCurrentWidget(self.image_view)
        self.info_label.setText(
            f"{loaded_image.display_name} | "
            f"{loaded_image.pixels_x}x{loaded_image.pixels_y} px | "
            "rendered via pyqtgraph.ImageView using the same image-display flow as LFA."
        )

    def set_roi_state(self, roi_state: Optional[ROIState]) -> None:
        """Update the ROI overlay shown on top of the image."""

        self.current_roi_state = roi_state
        self._update_roi_overlay_from_state()

    def reset_view(self) -> None:
        """Reset the visible range to the full image bounds."""

        if self.current_loaded_image is None or self.view_box is None:
            return
        self.view_box.autoRange()

    def _show_placeholder(self, message: str) -> None:
        self.placeholder_label.setText(message)
        self.stack.setCurrentWidget(self.placeholder_label)
        self.info_label.setText(
            "pyqtgraph ImageView viewport ready. Histogram/LUT controls match the LFA-style image display."
        )

    def _update_roi_overlay_from_state(self) -> None:
        if self.roi_item is None:
            return

        image = self.current_loaded_image
        roi = self.current_roi_state
        if image is None or roi is None:
            self.roi_item.hide()
            return

        clamped_roi = roi.clamped(image_width=image.pixels_x, image_height=image.pixels_y)
        if clamped_roi != roi:
            self.current_roi_state = clamped_roi

        self._syncing_roi_overlay = True
        try:
            self.roi_item.setPos((float(clamped_roi.x), float(clamped_roi.y)), update=False)
            self.roi_item.setSize((float(clamped_roi.width), float(clamped_roi.height)), update=False)
            self.roi_item.show()
        finally:
            self._syncing_roi_overlay = False

    def _on_roi_item_changed(self) -> None:
        if self._syncing_roi_overlay or self.roi_item is None or self.current_loaded_image is None:
            return

        image = self.current_loaded_image
        x_value = float(self.roi_item.pos().x())
        y_value = float(self.roi_item.pos().y())
        width_value = float(self.roi_item.size().x())
        height_value = float(self.roi_item.size().y())

        roi_state = ROIState(
            x=int(round(x_value)),
            y=int(round(y_value)),
            width=int(round(width_value)),
            height=int(round(height_value)),
        ).clamped(image_width=image.pixels_x, image_height=image.pixels_y)

        self.current_roi_state = roi_state
        self._update_roi_overlay_from_state()
        self.roi_state_edited.emit(roi_state)

    def image_bounding_rect(self) -> Optional[QRectF]:
        """Return the current image bounding rectangle in image coordinates."""

        if self.image_item is None:
            return None
        rect = self.image_item.boundingRect()
        if rect.isNull():
            return None
        return rect

    def closeEvent(self, event) -> None:  # pragma: no cover - exercised indirectly in GUI teardown
        self._clear_image_view(detach=True)
        super().closeEvent(event)

    def _clear_image_view(self, detach: bool = False) -> None:
        """Clear image/ROI state and optionally detach graphics items during teardown."""

        if self.roi_item is not None:
            try:
                self.roi_item.hide()
            except Exception:
                pass
            if detach:
                try:
                    self.roi_item.setParentItem(None)
                except Exception:
                    pass
        if self.image_view is not None:
            try:
                self.image_view.clear()
            except Exception:
                pass
            if detach:
                try:
                    self.image_view.close()
                except Exception:
                    pass
