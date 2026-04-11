"""Live ROI preview widget for AtomMapper."""

from __future__ import annotations

from typing import Optional

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget

from .image_utils import build_grayscale_pixmap, extract_roi_patch
from .models import LoadedImage, ROIState


class ROIPreviewWidget(QWidget):
    """Render the current ROI crop from the active STM image."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.current_loaded_image: Optional[LoadedImage] = None
        self.current_roi_state: Optional[ROIState] = None
        self.current_patch_data: Optional[np.ndarray] = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.title_label = QLabel("ROI preview")
        self.title_label.setStyleSheet("font-size: 16px; font-weight: 600;")

        self.preview_label = QLabel("ROI preview will appear here.")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setStyleSheet(
            "border: 1px solid palette(mid); background: palette(base); padding: 12px;"
        )
        self.preview_label.setMinimumHeight(220)

        self.info_label = QLabel("Load an STM image and adjust ROI to preview the crop.")
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.info_label.setStyleSheet("font-size: 12px; color: palette(mid);")

        layout.addWidget(self.title_label)
        layout.addWidget(self.preview_label)
        layout.addWidget(self.info_label)

    def set_loaded_image(self, loaded_image: Optional[LoadedImage]) -> None:
        """Store the active image and refresh the visible ROI crop."""

        self.current_loaded_image = loaded_image
        self._refresh_preview()

    def set_roi_state(self, roi_state: Optional[ROIState]) -> None:
        """Store the active ROI and refresh the visible ROI crop."""

        self.current_roi_state = roi_state
        self._refresh_preview()

    def _refresh_preview(self) -> None:
        image = self.current_loaded_image
        roi = self.current_roi_state
        if image is None:
            self.current_patch_data = None
            self.preview_label.clear()
            self.preview_label.setText("ROI preview will appear here.")
            self.info_label.setText("Load an STM image and adjust ROI to preview the crop.")
            return

        if roi is None:
            self.current_patch_data = None
            self.preview_label.clear()
            self.preview_label.setText("No ROI defined for the active image.")
            self.info_label.setText(f"{image.display_name} | waiting for ROI geometry.")
            return

        patch = extract_roi_patch(image.image_data, roi)
        if patch is None:
            self.current_patch_data = None
            self.preview_label.clear()
            self.preview_label.setText("ROI is outside the image bounds.")
            self.info_label.setText(
                f"{image.display_name} | ROI x={roi.x} y={roi.y} w={roi.width} h={roi.height} "
                "does not intersect the image."
            )
            return

        self.current_patch_data = patch
        pixmap = build_grayscale_pixmap(patch)
        scaled = pixmap.scaled(
            320,
            220,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.preview_label.setPixmap(scaled)
        self.info_label.setText(
            f"{image.display_name} | ROI x={roi.x} y={roi.y} w={roi.width} h={roi.height} | "
            f"patch {patch.shape[1]}x{patch.shape[0]} px"
        )
