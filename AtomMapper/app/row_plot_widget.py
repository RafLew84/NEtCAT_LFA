"""Row-plot widget for the selected atom row."""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QStackedWidget, QVBoxLayout, QWidget

from .models import AtomRow
from .plots import RowMetricSeries, RowPlotMode, build_row_metric_series

try:
    import pyqtgraph as pg
except ImportError:  # pragma: no cover - exercised only in missing-dependency environments
    pg = None


class RowPlotWidget(QWidget):
    """Render a basic pyqtgraph plot for the currently selected atom row."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.current_row: Optional[AtomRow] = None
        self.current_series: Optional[RowMetricSeries] = None
        self.current_mode = RowPlotMode.X_PX
        self.backend_available = pg is not None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(8)

        self.title_label = QLabel("Selected row plot")
        self.title_label.setStyleSheet("font-size: 16px; font-weight: 600;")
        self.metric_combo = QComboBox(self)
        self.metric_combo.setObjectName("atommapper_row_plot_metric_combo")
        self.metric_combo.addItem("x(i)", RowPlotMode.X_PX)
        self.metric_combo.addItem("y(i)", RowPlotMode.Y_PX)
        self.metric_combo.addItem("distance(i,i+1)", RowPlotMode.DISTANCE_PX)
        self.metric_combo.setToolTip("Select the metric displayed for the active row.")
        self.metric_combo.currentIndexChanged.connect(self._on_metric_changed)
        self.metric_combo.setEnabled(False)

        header_layout.addWidget(self.title_label, 1)
        header_layout.addWidget(self.metric_combo)

        self.stack = QStackedWidget(self)

        self.placeholder_label = QLabel("Select an atom row to display a plot.")
        self.placeholder_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.placeholder_label.setWordWrap(True)
        self.placeholder_label.setStyleSheet(
            "border: 1px solid palette(mid); color: palette(mid); padding: 24px;"
        )

        self.plot_widget = None
        self.plot_item = None
        self.curve_item = None
        if self.backend_available:
            self.plot_widget = pg.PlotWidget(self)
            self.plot_widget.setObjectName("atommapper_row_plot_widget")
            self.plot_item = self.plot_widget.getPlotItem()
            self.plot_item.showGrid(x=True, y=True, alpha=0.25)
            self.plot_item.setMenuEnabled(False)
            self.curve_item = self.plot_item.plot(
                [],
                [],
                pen=pg.mkPen(color=(100, 180, 255), width=2),
                symbol="o",
                symbolSize=8,
                symbolBrush=pg.mkBrush(100, 180, 255, 220),
                symbolPen=pg.mkPen(color=(30, 30, 30, 220), width=1),
            )
            self.stack.addWidget(self.plot_widget)

        self.stack.addWidget(self.placeholder_label)
        self.info_label = QLabel("Load points and select an atom row to inspect x(i) or y(i).")
        self.info_label.setWordWrap(True)
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.info_label.setStyleSheet("font-size: 12px; color: palette(mid);")

        layout.addLayout(header_layout)
        layout.addWidget(self.stack, 1)
        layout.addWidget(self.info_label)

        self._show_placeholder("Select an atom row to display a plot.")

    def set_row(self, row: Optional[AtomRow]) -> None:
        """Set the current AtomRow and rebuild the plot for the selected metric."""

        self.current_row = row
        self.metric_combo.setEnabled(row is not None)
        self._rebuild_series_from_row()

    def set_series(self, series: Optional[RowMetricSeries]) -> None:
        """Set a prebuilt row series directly.

        This compatibility path keeps the widget usable before full main-window
        integration. Metric switching requires ``set_row(...)`` because the
        widget then owns the source data needed to rebuild the series.
        """

        self.current_row = None
        self.current_series = series
        self.metric_combo.setEnabled(False)
        if series is not None:
            self.current_mode = series.mode
        self._sync_metric_combo()
        self._refresh_view()

    def _show_placeholder(self, message: str) -> None:
        self.placeholder_label.setText(message)
        self.stack.setCurrentWidget(self.placeholder_label)

    def _sync_metric_combo(self) -> None:
        combo_index = self.metric_combo.findData(self.current_mode)
        if combo_index < 0:
            return
        self.metric_combo.blockSignals(True)
        self.metric_combo.setCurrentIndex(combo_index)
        self.metric_combo.blockSignals(False)

    def _rebuild_series_from_row(self) -> None:
        if self.current_row is None:
            self.current_series = None
            self._sync_metric_combo()
            self._refresh_view()
            return

        self.current_series = build_row_metric_series(self.current_row, self.current_mode)
        self._sync_metric_combo()
        self._refresh_view()

    def _on_metric_changed(self, index: int) -> None:
        if index < 0:
            return
        selected_mode = self.metric_combo.itemData(index)
        if not isinstance(selected_mode, RowPlotMode):
            return
        self.current_mode = selected_mode
        if self.current_row is None:
            return
        self._rebuild_series_from_row()

    def _refresh_view(self) -> None:
        series = self.current_series
        if series is None:
            self._show_placeholder("Select an atom row to display a plot.")
            self.info_label.setText(
                "Load points and select an atom row to inspect x(i), y(i), or distance(i,i+1)."
            )
            return

        if not self.backend_available or self.plot_widget is None or self.curve_item is None:
            self._show_placeholder("pyqtgraph backend is not available.")
            self.info_label.setText("Row plot backend is unavailable in the current environment.")
            return

        if not series.samples:
            self.curve_item.setData([], [])
            self._show_placeholder(
                f"{series.row_display_name} has no plottable samples for {series.mode.value}."
            )
            self.info_label.setText(
                f"{series.row_display_name} | {series.mode.value} | waiting for enough points."
            )
            return

        x_values = [sample.x_value for sample in series.samples]
        y_values = [sample.y_value for sample in series.samples]
        self.curve_item.setData(x_values, y_values)
        self.plot_item.setLabel("bottom", series.x_label)
        self.plot_item.setLabel("left", series.y_label)
        self.plot_item.setTitle(f"{series.row_display_name} | {series.mode.value}")
        self.plot_widget.enableAutoRange()
        self.stack.setCurrentWidget(self.plot_widget)
        noun = "sample" if len(series.samples) == 1 else "samples"
        self.info_label.setText(
            f"{series.row_display_name} | {series.mode.value} | {len(series.samples)} {noun}"
        )
