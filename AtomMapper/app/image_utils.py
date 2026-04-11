"""Shared image helpers for AtomMapper preview widgets."""

from __future__ import annotations

from typing import Optional

import numpy as np
from PyQt6.QtGui import QImage, QPixmap

from .models import ROIState


def extract_roi_patch(image_data: np.ndarray, roi: ROIState) -> Optional[np.ndarray]:
    """Return a copy of the image crop described by ``roi``."""

    image_array = np.asarray(image_data)
    if image_array.ndim != 2:
        raise ValueError(f"Expected 2D image data, got shape {image_array.shape!r}.")

    x0 = max(0, int(roi.x))
    y0 = max(0, int(roi.y))
    x1 = min(image_array.shape[1], int(roi.x + roi.width))
    y1 = min(image_array.shape[0], int(roi.y + roi.height))
    if x1 <= x0 or y1 <= y0:
        return None
    return np.array(image_array[y0:y1, x0:x1], copy=True)


def build_grayscale_pixmap(image_data: np.ndarray) -> QPixmap:
    """Convert a 2D numeric array into an 8-bit grayscale pixmap."""

    image_array = np.asarray(image_data, dtype=float)
    finite_mask = np.isfinite(image_array)
    if not finite_mask.any():
        normalized = np.zeros(image_array.shape, dtype=np.uint8)
    else:
        finite_values = image_array[finite_mask]
        min_value = float(finite_values.min())
        max_value = float(finite_values.max())
        if max_value <= min_value:
            normalized = np.zeros(image_array.shape, dtype=np.uint8)
        else:
            scaled = (image_array - min_value) / (max_value - min_value)
            scaled = np.clip(scaled, 0.0, 1.0)
            normalized = np.zeros(image_array.shape, dtype=np.uint8)
            normalized[finite_mask] = np.round(scaled[finite_mask] * 255.0).astype(np.uint8)

    height, width = normalized.shape
    bytes_per_line = width
    qimage = QImage(
        normalized.data,
        width,
        height,
        bytes_per_line,
        QImage.Format.Format_Grayscale8,
    ).copy()
    return QPixmap.fromImage(qimage)
