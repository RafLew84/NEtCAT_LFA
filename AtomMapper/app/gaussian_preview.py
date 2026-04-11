"""Preview widget for the fitted 2D Gaussian model."""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget

from .gaussian_fit import GaussianPatchFitResult
from .image_utils import build_grayscale_pixmap


class GaussianFitPreviewWidget(QWidget):
    """Render the fitted Gaussian model and its center marker."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.current_fit_result: Optional[GaussianPatchFitResult] = None
        self.current_model_patch = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.title_label = QLabel("Gaussian fit preview")
        self.title_label.setStyleSheet("font-size: 16px; font-weight: 600;")

        self.preview_label = QLabel("Gaussian-fit preview will appear here.")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setStyleSheet(
            "border: 1px solid palette(mid); background: palette(base); padding: 12px;"
        )
        self.preview_label.setMinimumHeight(220)

        self.info_label = QLabel("Move ROI onto a local maximum to preview the fitted model.")
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.info_label.setStyleSheet("font-size: 12px; color: palette(mid);")

        layout.addWidget(self.title_label)
        layout.addWidget(self.preview_label)
        layout.addWidget(self.info_label)

    def set_fit_result(self, fit_result: Optional[GaussianPatchFitResult]) -> None:
        """Update the currently displayed Gaussian model preview."""

        self.current_fit_result = fit_result
        self.current_model_patch = None if fit_result is None else fit_result.model_patch
        self._refresh_preview()

    def _refresh_preview(self) -> None:
        fit_result = self.current_fit_result
        if fit_result is None:
            self.preview_label.clear()
            self.preview_label.setText("Gaussian-fit preview will appear here.")
            self.info_label.setText("Move ROI onto a local maximum to preview the fitted model.")
            return

        if fit_result.model_patch is None or fit_result.center_patch_yx is None:
            self.preview_label.clear()
            self.preview_label.setText("Gaussian fit is unavailable for the current ROI.")
            self.info_label.setText(
                fit_result.error_message
                or "The current ROI does not contain a stable Gaussian fit."
            )
            return

        pixmap = build_grayscale_pixmap(fit_result.model_patch)
        scaled = pixmap.scaled(
            320,
            220,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        marked = self._draw_center_marker(scaled, fit_result.center_patch_yx, fit_result.model_patch.shape)
        self.preview_label.setPixmap(marked)

        info_parts = [
            f"center y={fit_result.center_patch_yx[0]:.2f} x={fit_result.center_patch_yx[1]:.2f}",
            f"amp={fit_result.amplitude:.2f}" if fit_result.amplitude is not None else None,
            (
                f"sigma_y={fit_result.sigma_y:.2f} sigma_x={fit_result.sigma_x:.2f}"
                if fit_result.sigma_y is not None and fit_result.sigma_x is not None
                else None
            ),
        ]
        self.info_label.setText(" | ".join(part for part in info_parts if part))

    @staticmethod
    def _draw_center_marker(
        pixmap: QPixmap,
        center_patch_yx: tuple[float, float],
        patch_shape: tuple[int, int],
    ) -> QPixmap:
        marked = QPixmap(pixmap)
        painter = QPainter(marked)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        pen = QPen(QColor(255, 100, 100))
        pen.setWidth(2)
        painter.setPen(pen)

        rows, cols = patch_shape
        scale_x = marked.width() / max(1, cols)
        scale_y = marked.height() / max(1, rows)
        center_y, center_x = center_patch_yx
        cx = float(center_x) * scale_x
        cy = float(center_y) * scale_y
        half = 8.0
        painter.drawLine(int(round(cx - half)), int(round(cy)), int(round(cx + half)), int(round(cy)))
        painter.drawLine(int(round(cx)), int(round(cy - half)), int(round(cx)), int(round(cy + half)))
        painter.end()
        return marked
