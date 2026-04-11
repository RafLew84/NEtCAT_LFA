"""Row-plot widget for the selected atom row."""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QLabel, QStackedWidget, QVBoxLayout, QWidget

from .plots import RowMetricSeries

try:
    import pyqtgraph as pg
except ImportError:  # pragma: no cover - exercised only in missing-dependency environments
    pg = None


class RowPlotWidget(QWidget):
    """Render a basic pyqtgraph plot for the currently selected atom row."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.current_series: Optional[RowMetricSeries] = None
        self.backend_available = pg is not None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.title_label = QLabel("Selected row plot")
        self.title_label.setStyleSheet("font-size: 16px; font-weight: 600;")

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

        layout.addWidget(self.title_label)
        layout.addWidget(self.stack, 1)
        layout.addWidget(self.info_label)

        self._show_placeholder("Select an atom row to display a plot.")

    def set_series(self, series: Optional[RowMetricSeries]) -> None:
        """Set the current row series and refresh the plot/placeholder state."""

        self.current_series = series
        self._refresh_view()

    def _show_placeholder(self, message: str) -> None:
        self.placeholder_label.setText(message)
        self.stack.setCurrentWidget(self.placeholder_label)

    def _refresh_view(self) -> None:
        series = self.current_series
        if series is None:
            self._show_placeholder("Select an atom row to display a plot.")
            self.info_label.setText("Load points and select an atom row to inspect x(i) or y(i).")
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
