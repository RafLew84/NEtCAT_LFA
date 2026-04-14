"""Preview widget for the currently selected local-fit model."""

from __future__ import annotations

from typing import Optional

import numpy as np

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget

from .fit_models import LocalPeakFitResult
from .image_utils import build_grayscale_pixmap


class GaussianFitPreviewWidget(QWidget):
    """Render the active fit model together with ROI/mask context."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.current_fit_result: Optional[LocalPeakFitResult] = None
        self.current_model_patch = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.title_label = QLabel("Fit preview")
        self.title_label.setStyleSheet("font-size: 16px; font-weight: 600;")

        self.preview_label = QLabel("Fit preview will appear here.")
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

    def set_fit_result(self, fit_result: Optional[LocalPeakFitResult]) -> None:
        """Update the currently displayed local-model preview."""

        self.current_fit_result = fit_result
        self.current_model_patch = None if fit_result is None else fit_result.model_patch
        self._refresh_preview()

    def _refresh_preview(self) -> None:
        fit_result = self.current_fit_result
        if fit_result is None:
            self.title_label.setText("Fit preview")
            self.preview_label.clear()
            self.preview_label.setText("Fit preview will appear here.")
            self.info_label.setText("Move ROI onto a local maximum to preview the fitted model.")
            return

        model_label = fit_result.model.value.capitalize()
        self.title_label.setText(f"Fit preview - {model_label}")

        if fit_result.model_patch is None or fit_result.center_patch_yx is None:
            self.preview_label.clear()
            self.preview_label.setText(f"{model_label} fit is unavailable for the current ROI.")
            info_text = fit_result.error_message or f"The current ROI does not contain a stable {model_label} fit."
            if fit_result.fit_mask is not None:
                info_text += f" Mask active on {int(fit_result.fit_mask.sum())} px."
            self.info_label.setText(info_text)
            return

        comparison_pixmap = self._build_comparison_pixmap(
            roi_patch=fit_result.roi_patch,
            model_patch=fit_result.model_patch,
            center_patch_yx=fit_result.center_patch_yx,
            fit_mask=fit_result.fit_mask,
        )
        self.preview_label.clear()
        self.preview_label.setPixmap(comparison_pixmap)

        info_parts = [
            f"center y={fit_result.center_patch_yx[0]:.2f} x={fit_result.center_patch_yx[1]:.2f}",
            f"amp={fit_result.amplitude:.2f}" if fit_result.amplitude is not None else None,
            self._format_width_info(fit_result),
            (
                f"mask={int(fit_result.fit_mask.sum())} px"
                if fit_result.fit_mask is not None
                else "mask=off"
            ),
        ]
        self.info_label.setText(" | ".join(part for part in info_parts if part))

    @staticmethod
    def _format_width_info(fit_result: LocalPeakFitResult) -> str | None:
        if fit_result.sigma_y is None or fit_result.sigma_x is None:
            return None
        if fit_result.model.value == "gaussian":
            return f"sigma_y={fit_result.sigma_y:.2f} sigma_x={fit_result.sigma_x:.2f}"
        if fit_result.model.value == "lorentzian":
            return f"gamma_y={fit_result.sigma_y:.2f} gamma_x={fit_result.sigma_x:.2f}"
        if fit_result.model.value == "voigt":
            gamma_y = fit_result.shape_parameters.get("gamma_y")
            gamma_x = fit_result.shape_parameters.get("gamma_x")
            if gamma_y is not None and gamma_x is not None:
                return (
                    f"sigma_y={fit_result.sigma_y:.2f} sigma_x={fit_result.sigma_x:.2f}"
                    f" | gamma_y={float(gamma_y):.2f} gamma_x={float(gamma_x):.2f}"
                )
        return f"width_y={fit_result.sigma_y:.2f} width_x={fit_result.sigma_x:.2f}"

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

    @staticmethod
    def _apply_fit_mask_to_roi_patch(roi_patch, fit_mask):
        patch = np.asarray(roi_patch, dtype=float)
        if fit_mask is None:
            return patch

        mask = np.asarray(fit_mask, dtype=bool)
        if mask.shape != patch.shape:
            return patch
        if not mask.any():
            return patch

        masked_patch = np.array(patch, copy=True)
        finite_values = patch[np.isfinite(patch)]
        if finite_values.size == 0:
            masked_patch[~mask] = 0.0
            return masked_patch

        min_value = float(finite_values.min())
        max_value = float(finite_values.max())
        range_value = max(max_value - min_value, 1.0)
        masked_patch[~mask] = min_value - (0.25 * range_value)
        return masked_patch

    def _build_comparison_pixmap(
        self,
        *,
        roi_patch,
        model_patch,
        center_patch_yx: tuple[float, float],
        fit_mask,
    ) -> QPixmap:
        """Render a side-by-side comparison of ROI/mask and fitted model."""

        display_roi_patch = self._apply_fit_mask_to_roi_patch(roi_patch, fit_mask)
        roi_pixmap = build_grayscale_pixmap(display_roi_patch).scaled(
            140,
            170,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        roi_pixmap = self._draw_center_marker(roi_pixmap, center_patch_yx, np.asarray(roi_patch).shape)

        model_pixmap = build_grayscale_pixmap(model_patch).scaled(
            140,
            170,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        model_pixmap = self._draw_center_marker(model_pixmap, center_patch_yx, np.asarray(model_patch).shape)

        canvas = QPixmap(320, 220)
        canvas.fill(self.palette().base().color())

        painter = QPainter(canvas)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(QPen(QColor(140, 140, 140), 1))
        painter.drawRect(0, 0, canvas.width() - 1, canvas.height() - 1)
        painter.drawLine(canvas.width() // 2, 28, canvas.width() // 2, canvas.height() - 12)

        title_pen = QPen(self.palette().text().color())
        painter.setPen(title_pen)
        left_caption = "Masked ROI" if fit_mask is not None else "ROI patch"
        painter.drawText(16, 18, left_caption)
        painter.drawText((canvas.width() // 2) + 16, 18, "Fit model")

        top_y = 34
        left_x = 10 + ((canvas.width() // 2 - 20 - roi_pixmap.width()) // 2)
        right_x = (canvas.width() // 2) + 10 + ((canvas.width() // 2 - 20 - model_pixmap.width()) // 2)
        painter.drawPixmap(left_x, top_y + ((170 - roi_pixmap.height()) // 2), roi_pixmap)
        painter.drawPixmap(right_x, top_y + ((170 - model_pixmap.height()) // 2), model_pixmap)
        painter.end()
        return canvas
