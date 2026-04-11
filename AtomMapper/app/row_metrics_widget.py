"""Distance-metrics widget for the currently selected atom row."""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QFormLayout, QLabel, QStackedWidget, QVBoxLayout, QWidget

from .plots import RowDistanceMetrics


class RowMetricsWidget(QWidget):
    """Render basic distance statistics for the currently selected row."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.current_metrics: Optional[RowDistanceMetrics] = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.title_label = QLabel("Row distance metrics")
        self.title_label.setStyleSheet("font-size: 16px; font-weight: 600;")

        self.stack = QStackedWidget(self)

        self.placeholder_label = QLabel("Select an atom row to display distance metrics.")
        self.placeholder_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.placeholder_label.setWordWrap(True)
        self.placeholder_label.setStyleSheet(
            "border: 1px solid palette(mid); color: palette(mid); padding: 24px;"
        )

        self.metrics_panel = QWidget(self)
        form_layout = QFormLayout(self.metrics_panel)
        form_layout.setContentsMargins(12, 12, 12, 12)
        form_layout.setHorizontalSpacing(16)
        form_layout.setVerticalSpacing(8)

        self.point_count_value = QLabel("-")
        self.distance_count_value = QLabel("-")
        self.mean_distance_value = QLabel("-")
        self.std_distance_value = QLabel("-")
        self.min_distance_value = QLabel("-")
        self.max_distance_value = QLabel("-")

        form_layout.addRow("Points", self.point_count_value)
        form_layout.addRow("Segments", self.distance_count_value)
        form_layout.addRow("Mean distance", self.mean_distance_value)
        form_layout.addRow("Std distance", self.std_distance_value)
        form_layout.addRow("Min distance", self.min_distance_value)
        form_layout.addRow("Max distance", self.max_distance_value)

        self.stack.addWidget(self.metrics_panel)
        self.stack.addWidget(self.placeholder_label)

        self.info_label = QLabel("Load points and select an atom row to inspect distances.")
        self.info_label.setWordWrap(True)
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.info_label.setStyleSheet("font-size: 12px; color: palette(mid);")

        layout.addWidget(self.title_label)
        layout.addWidget(self.stack, 1)
        layout.addWidget(self.info_label)

        self._show_placeholder("Select an atom row to display distance metrics.")

    def set_metrics(self, metrics: Optional[RowDistanceMetrics]) -> None:
        """Set the current metrics payload and refresh the visible state."""

        self.current_metrics = metrics
        self._refresh_view()

    def _show_placeholder(self, message: str) -> None:
        self.placeholder_label.setText(message)
        self.stack.setCurrentWidget(self.placeholder_label)

    def _refresh_view(self) -> None:
        metrics = self.current_metrics
        if metrics is None:
            self._show_placeholder("Select an atom row to display distance metrics.")
            self.info_label.setText("Load points and select an atom row to inspect distances.")
            return

        if metrics.distance_count <= 0:
            self._show_placeholder(
                f"{metrics.row_display_name} needs at least 2 points to compute distances."
            )
            self.info_label.setText(
                f"{metrics.row_display_name} | {metrics.point_count} point(s) | waiting for more data."
            )
            return

        self.point_count_value.setText(str(metrics.point_count))
        self.distance_count_value.setText(str(metrics.distance_count))
        self.mean_distance_value.setText(self._format_px(metrics.mean_distance_px))
        self.std_distance_value.setText(self._format_px(metrics.std_distance_px))
        self.min_distance_value.setText(self._format_px(metrics.min_distance_px))
        self.max_distance_value.setText(self._format_px(metrics.max_distance_px))
        self.stack.setCurrentWidget(self.metrics_panel)
        self.info_label.setText(
            f"{metrics.row_display_name} | {metrics.point_count} points | {metrics.distance_count} segments"
        )

    @staticmethod
    def _format_px(value: float | None) -> str:
        if value is None:
            return "-"
        return f"{value:.3f} px"
