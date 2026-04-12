"""Distance-metrics widget for the currently selected atom row."""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from .plots import PlotUnit, RowDistanceMetrics


class RowMetricsWidget(QWidget):
    """Render basic distance statistics for the currently selected row."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.current_metrics: Optional[RowDistanceMetrics] = None
        self.current_unit = PlotUnit.PX

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.title_label = QLabel("Row distance metrics")
        self.title_label.setStyleSheet("font-size: 16px; font-weight: 600;")
        self.unit_combo = QComboBox(self)
        self.unit_combo.setObjectName("atommapper_row_metrics_unit_combo")
        self.unit_combo.addItem("px", PlotUnit.PX)
        self.unit_combo.addItem("nm", PlotUnit.NM)
        self.unit_combo.currentIndexChanged.connect(self._on_unit_changed)

        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(8)
        header_layout.addWidget(self.title_label, 1)
        header_layout.addWidget(self.unit_combo)

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

        layout.addLayout(header_layout)
        layout.addWidget(self.stack, 1)
        layout.addWidget(self.info_label)

        self._update_title_label()
        self._show_placeholder("Select an atom row to display distance metrics.")

    def set_metrics(self, metrics: Optional[RowDistanceMetrics]) -> None:
        """Set the current metrics payload and refresh the visible state."""

        self.current_metrics = metrics
        self._refresh_view()

    def _show_placeholder(self, message: str) -> None:
        self.placeholder_label.setText(message)
        self.stack.setCurrentWidget(self.placeholder_label)

    def _update_title_label(self) -> None:
        self.title_label.setText(f"Row distance metrics [{self.current_unit.value}]")

    def _on_unit_changed(self, index: int) -> None:
        if index < 0:
            return
        selected_unit = self.unit_combo.itemData(index)
        if not isinstance(selected_unit, PlotUnit):
            return
        self.current_unit = selected_unit
        self._refresh_view()

    def _refresh_view(self) -> None:
        self._update_title_label()
        metrics = self.current_metrics
        if metrics is None:
            self._show_placeholder("Select an atom row to display distance metrics.")
            self.info_label.setText("Load points and select an atom row to inspect distances.")
            return

        distance_count = metrics.distance_count_for_unit(self.current_unit)
        if distance_count <= 0:
            if self.current_unit is PlotUnit.NM:
                self._show_placeholder(
                    f"{metrics.row_display_name} needs calibrated points to compute distances in nm."
                )
                self.info_label.setText(
                    f"{metrics.row_display_name} | {metrics.point_count} point(s) | waiting for calibrated data."
                )
            else:
                self._show_placeholder(
                    f"{metrics.row_display_name} needs at least 2 points to compute distances."
                )
                self.info_label.setText(
                    f"{metrics.row_display_name} | {metrics.point_count} point(s) | waiting for more data."
                )
            return

        self.point_count_value.setText(str(metrics.point_count))
        self.distance_count_value.setText(str(distance_count))
        self.mean_distance_value.setText(
            self._format_value(metrics.mean_distance_for_unit(self.current_unit), self.current_unit)
        )
        self.std_distance_value.setText(
            self._format_value(metrics.std_distance_for_unit(self.current_unit), self.current_unit)
        )
        self.min_distance_value.setText(
            self._format_value(metrics.min_distance_for_unit(self.current_unit), self.current_unit)
        )
        self.max_distance_value.setText(
            self._format_value(metrics.max_distance_for_unit(self.current_unit), self.current_unit)
        )
        self.stack.setCurrentWidget(self.metrics_panel)
        self.info_label.setText(
            f"{metrics.row_display_name} | {metrics.point_count} points | {distance_count} segments | {self.current_unit.value}"
        )

    @staticmethod
    def _format_value(value: float | None, unit: PlotUnit) -> str:
        if value is None:
            return "-"
        return f"{value:.3f} {unit.value}"
