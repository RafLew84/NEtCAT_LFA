"""Global scatter-plot widget for all rows in the active image family."""

from __future__ import annotations

from collections import OrderedDict
from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QStackedWidget, QVBoxLayout, QWidget

from .models import AtomRow
from .plots import GlobalScatterSample, GlobalScatterSeries, PlotUnit, build_global_scatter_series

try:
    import pyqtgraph as pg
except ImportError:  # pragma: no cover - exercised only in missing-dependency environments
    pg = None


class GlobalScatterPlotWidget(QWidget):
    """Render a scatter plot of all points stored in the active image family."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.current_rows: tuple[AtomRow, ...] = ()
        self.current_series: Optional[GlobalScatterSeries] = None
        self.current_unit = PlotUnit.PX
        self.backend_available = pg is not None
        self.scatter_items: list[object] = []
        self.legend_item = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(8)

        self.title_label = QLabel("Global rows plot")
        self.title_label.setStyleSheet("font-size: 16px; font-weight: 600;")
        self.unit_combo = QComboBox(self)
        self.unit_combo.setObjectName("atommapper_global_scatter_unit_combo")
        self.unit_combo.addItem("px", PlotUnit.PX)
        self.unit_combo.addItem("nm", PlotUnit.NM)
        self.unit_combo.setToolTip("Select the unit displayed for the global rows plot.")
        self.unit_combo.currentIndexChanged.connect(self._on_unit_changed)

        header_layout.addWidget(self.title_label, 1)
        header_layout.addWidget(self.unit_combo)

        self.stack = QStackedWidget(self)
        self.placeholder_label = QLabel("Add saved points to display the global scatter plot.")
        self.placeholder_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.placeholder_label.setWordWrap(True)
        self.placeholder_label.setStyleSheet(
            "border: 1px solid palette(mid); color: palette(mid); padding: 24px;"
        )

        self.plot_widget = None
        self.plot_item = None
        if self.backend_available:
            self.plot_widget = pg.PlotWidget(self)
            self.plot_widget.setObjectName("atommapper_global_scatter_plot_widget")
            self.plot_item = self.plot_widget.getPlotItem()
            self.plot_item.showGrid(x=True, y=True, alpha=0.25)
            self.plot_item.setMenuEnabled(False)
            self.plot_item.setLabel("bottom", "x (px)")
            self.plot_item.setLabel("left", "y (px)")
            self.legend_item = self.plot_item.addLegend(offset=(8, 8))
            self.stack.addWidget(self.plot_widget)

        self.stack.addWidget(self.placeholder_label)
        self.info_label = QLabel("Load points to inspect the global geometry of all rows.")
        self.info_label.setWordWrap(True)
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.info_label.setStyleSheet("font-size: 12px; color: palette(mid);")

        layout.addLayout(header_layout)
        layout.addWidget(self.stack, 1)
        layout.addWidget(self.info_label)

        self._update_title_label()
        self._show_placeholder("Add saved points to display the global scatter plot.")

    def set_rows(self, rows: tuple[AtomRow, ...] | list[AtomRow]) -> None:
        """Set source rows and rebuild the scatter series in the selected unit."""

        self.current_rows = tuple(rows)
        self.unit_combo.setEnabled(True)
        self._rebuild_series_from_rows()

    def set_series(self, series: Optional[GlobalScatterSeries]) -> None:
        """Set the current global scatter series and refresh the widget."""

        self.current_rows = ()
        self.current_series = series
        if series is not None:
            self.current_unit = series.unit
        self.unit_combo.setEnabled(False)
        self._sync_unit_combo()
        self._refresh_view()

    def _show_placeholder(self, message: str) -> None:
        self.placeholder_label.setText(message)
        self.stack.setCurrentWidget(self.placeholder_label)

    def _update_title_label(self) -> None:
        self.title_label.setText(f"Global rows plot [{self.current_unit.value}]")

    def _sync_unit_combo(self) -> None:
        combo_index = self.unit_combo.findData(self.current_unit)
        if combo_index < 0:
            return
        self.unit_combo.blockSignals(True)
        self.unit_combo.setCurrentIndex(combo_index)
        self.unit_combo.blockSignals(False)

    def _rebuild_series_from_rows(self) -> None:
        self.current_series = build_global_scatter_series(self.current_rows, unit=self.current_unit)
        self._sync_unit_combo()
        self._refresh_view()

    def _on_unit_changed(self, index: int) -> None:
        if index < 0:
            return
        selected_unit = self.unit_combo.itemData(index)
        if not isinstance(selected_unit, PlotUnit):
            return
        self.current_unit = selected_unit
        if not self.current_rows:
            self.current_series = build_global_scatter_series((), unit=self.current_unit)
            self._refresh_view()
            return
        self._rebuild_series_from_rows()

    def _clear_scatter_items(self) -> None:
        if self.plot_item is None:
            self.scatter_items = []
            return
        for item in self.scatter_items:
            self.plot_item.removeItem(item)
        self.scatter_items = []
        if self.legend_item is not None:
            self.legend_item.clear()

    def _refresh_view(self) -> None:
        self._update_title_label()
        series = self.current_series
        if series is None:
            self._clear_scatter_items()
            self._show_placeholder("Add saved points to display the global scatter plot.")
            self.info_label.setText("Load points to inspect the global geometry of all rows.")
            return

        if not self.backend_available or self.plot_widget is None or self.plot_item is None:
            self._show_placeholder("pyqtgraph backend is not available.")
            self.info_label.setText("Global scatter backend is unavailable in the current environment.")
            return

        self._clear_scatter_items()
        if not series.samples:
            self._show_placeholder("The active image family has no saved points yet.")
            self.info_label.setText("Global scatter is waiting for saved points.")
            return

        grouped_samples: "OrderedDict[str, list[GlobalScatterSample]]" = OrderedDict()
        for sample in series.samples:
            grouped_samples.setdefault(sample.row_id, []).append(sample)

        for index, grouped in enumerate(grouped_samples.values()):
            reference_sample = grouped[0]
            color = self._resolve_row_color(reference_sample, index)
            scatter_item = self.plot_item.plot(
                [sample.x_value for sample in grouped],
                [sample.y_value for sample in grouped],
                pen=None,
                symbol="o",
                symbolSize=9,
                symbolBrush=pg.mkBrush(color),
                symbolPen=pg.mkPen(color=QColor(30, 30, 30, 220), width=1),
                name=reference_sample.row_display_name,
            )
            self.scatter_items.append(scatter_item)

        self.plot_item.setTitle(f"All rows | x-y scatter | {series.unit.value}")
        self.plot_item.setLabel("bottom", series.x_label)
        self.plot_item.setLabel("left", series.y_label)
        self.plot_widget.enableAutoRange()
        self.stack.setCurrentWidget(self.plot_widget)

        row_count = len(grouped_samples)
        point_count = len(series.samples)
        row_noun = "row" if row_count == 1 else "rows"
        point_noun = "point" if point_count == 1 else "points"
        self.info_label.setText(
            f"{row_count} {row_noun} | {point_count} {point_noun} | {series.unit.value}"
        )

    def _resolve_row_color(self, sample: GlobalScatterSample, index: int) -> QColor:
        if sample.color_hex:
            return pg.mkColor(sample.color_hex)
        return pg.intColor(index, hues=max(6, len(self.scatter_items) + 2))
