"""Shared preview widgets for the AtomMapper preprocessing dialog."""

from __future__ import annotations

from typing import Optional

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget

from .image_utils import build_grayscale_pixmap
from .preprocessing_state import PreviewViewport


def _extract_viewport_patch(
    image_data: np.ndarray,
    viewport: Optional[PreviewViewport],
) -> tuple[np.ndarray, PreviewViewport]:
    """Return the image patch matching ``viewport`` and the effective viewport."""

    image_array = np.asarray(image_data, dtype=float)
    if image_array.ndim != 2:
        raise ValueError(f"Expected 2D image data, got shape {image_array.shape!r}.")

    full_viewport = PreviewViewport(
        x=0,
        y=0,
        width=int(image_array.shape[1]),
        height=int(image_array.shape[0]),
    )
    normalized = viewport.normalized() if viewport is not None else full_viewport
    if normalized.width <= 0 or normalized.height <= 0:
        normalized = full_viewport

    x0 = max(0, min(normalized.x, image_array.shape[1] - 1))
    y0 = max(0, min(normalized.y, image_array.shape[0] - 1))
    x1 = max(x0 + 1, min(image_array.shape[1], x0 + normalized.width))
    y1 = max(y0 + 1, min(image_array.shape[0], y0 + normalized.height))

    effective = PreviewViewport(
        x=x0,
        y=y0,
        width=x1 - x0,
        height=y1 - y0,
    )
    return np.array(image_array[y0:y1, x0:x1], copy=True), effective


class PreprocessingImagePreview(QWidget):
    """Preview panel that renders the same normalized viewport each time."""

    def __init__(self, placeholder_text: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.render_count = 0
        self.current_viewport = PreviewViewport()
        self.current_patch_shape: tuple[int, int] = (0, 0)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(320, 240)
        self.image_label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self.image_label.setStyleSheet(
            "border: 1px solid palette(mid); background: palette(base);"
        )

        self.status_label = QLabel(placeholder_text, self)
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("font-size: 12px; color: palette(mid);")

        layout.addWidget(self.image_label, 1)
        layout.addWidget(self.status_label)

    def set_preview_image(
        self,
        image_data: np.ndarray,
        *,
        viewport: Optional[PreviewViewport] = None,
        status_text: str = "",
    ) -> None:
        """Render ``image_data`` using the provided preview viewport."""

        patch, effective_viewport = _extract_viewport_patch(image_data, viewport)
        pixmap = build_grayscale_pixmap(patch)
        scaled = pixmap.scaled(
            480,
            360,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.image_label.setPixmap(scaled)
        self.status_label.setText(status_text)
        self.current_viewport = effective_viewport
        self.current_patch_shape = tuple(int(value) for value in patch.shape)
        self.render_count += 1
