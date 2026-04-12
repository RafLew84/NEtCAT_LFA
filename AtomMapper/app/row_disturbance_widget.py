"""Disturbance-summary widget for the currently selected atom row."""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QFormLayout, QLabel, QStackedWidget, QVBoxLayout, QWidget

from .models import AtomRow
from .plots import PlotUnit
from .row_geometry import RowDisturbanceSeries, RowGeometryUnit, build_row_disturbance_series


class RowDisturbanceWidget(QWidget):
    """Render a compact summary of local disturbance candidates for the active row."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.current_row: Optional[AtomRow] = None
        self.current_series: Optional[RowDisturbanceSeries] = None
        self.current_unit = PlotUnit.PX

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.title_label = QLabel("Row disturbance candidates [px]")
        self.title_label.setStyleSheet("font-size: 16px; font-weight: 600;")

        self.stack = QStackedWidget(self)

        self.placeholder_label = QLabel("Select an atom row to inspect local disturbance candidates.")
        self.placeholder_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.placeholder_label.setWordWrap(True)
        self.placeholder_label.setStyleSheet(
            "border: 1px solid palette(mid); color: palette(mid); padding: 24px;"
        )

        self.summary_panel = QWidget(self)
        form_layout = QFormLayout(self.summary_panel)
        form_layout.setContentsMargins(12, 12, 12, 12)
        form_layout.setHorizontalSpacing(16)
        form_layout.setVerticalSpacing(8)

        self.sample_count_value = QLabel("-")
        self.candidate_count_value = QLabel("-")
        self.strongest_point_value = QLabel("-")
        self.strongest_score_value = QLabel("-")
        self.strongest_flags_value = QLabel("-")
        self.spacing_threshold_value = QLabel("-")
        self.transverse_threshold_value = QLabel("-")
        self.direction_threshold_value = QLabel("-")

        form_layout.addRow("Interior samples", self.sample_count_value)
        form_layout.addRow("Candidate count", self.candidate_count_value)
        form_layout.addRow("Strongest point", self.strongest_point_value)
        form_layout.addRow("Strongest score", self.strongest_score_value)
        form_layout.addRow("Strongest markers", self.strongest_flags_value)
        form_layout.addRow("Spacing jump threshold", self.spacing_threshold_value)
        form_layout.addRow("Transverse jump threshold", self.transverse_threshold_value)
        form_layout.addRow("Direction threshold", self.direction_threshold_value)

        self.stack.addWidget(self.summary_panel)
        self.stack.addWidget(self.placeholder_label)

        self.info_label = QLabel("Load points and select an atom row to inspect local disturbances.")
        self.info_label.setWordWrap(True)
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.info_label.setStyleSheet("font-size: 12px; color: palette(mid);")

        layout.addWidget(self.title_label)
        layout.addWidget(self.stack, 1)
        layout.addWidget(self.info_label)

        self._show_placeholder("Select an atom row to inspect local disturbance candidates.")

    def set_row(self, row: Optional[AtomRow], *, unit: PlotUnit = PlotUnit.PX) -> None:
        """Rebuild the disturbance summary from the active row and selected unit."""

        self.current_row = row
        self.current_unit = unit
        if row is None:
            self.current_series = None
        else:
            self.current_series = build_row_disturbance_series(
                row,
                unit=self._to_geometry_unit(unit),
            )
        self._refresh_view()

    def set_series(
        self,
        series: Optional[RowDisturbanceSeries],
        *,
        row: Optional[AtomRow] = None,
        unit: PlotUnit | None = None,
    ) -> None:
        """Set a prebuilt series directly for tests or compatibility paths."""

        self.current_row = row
        self.current_series = series
        if unit is not None:
            self.current_unit = unit
        elif series is not None:
            self.current_unit = PlotUnit.NM if series.unit is RowGeometryUnit.NM else PlotUnit.PX
        self._refresh_view()

    def _show_placeholder(self, message: str) -> None:
        self.title_label.setText(f"Row disturbance candidates [{self.current_unit.value}]")
        self.placeholder_label.setText(message)
        self.stack.setCurrentWidget(self.placeholder_label)

    def _refresh_view(self) -> None:
        self.title_label.setText(f"Row disturbance candidates [{self.current_unit.value}]")
        series = self.current_series
        row = self.current_row
        if series is None:
            if row is None:
                self._show_placeholder("Select an atom row to inspect local disturbance candidates.")
                self.info_label.setText(
                    "Load points and select an atom row to inspect local disturbances."
                )
                return
            if self.current_unit is PlotUnit.NM:
                self._show_placeholder(
                    f"{row.display_name} needs calibrated points to inspect disturbance candidates in nm."
                )
                self.info_label.setText(
                    f"{row.display_name} | {row.point_count} points | waiting for calibrated data."
                )
                return
            self._show_placeholder(
                f"{row.display_name} needs at least 3 points to inspect disturbance candidates."
            )
            self.info_label.setText(
                f"{row.display_name} | {row.point_count} points | waiting for more data."
            )
            return

        strongest_sample = max(series.samples, key=lambda sample: sample.candidate_score, default=None)
        self.sample_count_value.setText(str(len(series.samples)))
        self.candidate_count_value.setText(str(series.candidate_count))

        if strongest_sample is None:
            self.strongest_point_value.setText("none")
            self.strongest_score_value.setText("0.000")
            self.strongest_flags_value.setText("-")
        else:
            self.strongest_point_value.setText(
                f"{strongest_sample.point_index} ({strongest_sample.point_id})"
            )
            self.strongest_score_value.setText(f"{strongest_sample.candidate_score:.3f}")
            self.strongest_flags_value.setText(self._format_flags(strongest_sample))

        self.spacing_threshold_value.setText(
            self._format_value(series.spacing_jump_threshold, self.current_unit)
        )
        self.transverse_threshold_value.setText(
            self._format_value(series.transverse_jump_threshold, self.current_unit)
        )
        self.direction_threshold_value.setText(
            f"{series.direction_change_threshold_deg:.2f} deg"
        )
        self.stack.setCurrentWidget(self.summary_panel)
        self.info_label.setText(
            f"{series.row_display_name} | {series.candidate_count} candidate(s) / "
            f"{len(series.samples)} interior sample(s) | {self.current_unit.value}"
        )

    @staticmethod
    def _to_geometry_unit(unit: PlotUnit) -> RowGeometryUnit:
        return RowGeometryUnit.NM if unit is PlotUnit.NM else RowGeometryUnit.PX

    @staticmethod
    def _format_value(value: float, unit: PlotUnit) -> str:
        return f"{value:.3f} {unit.value}"

    @staticmethod
    def _format_flags(sample) -> str:
        flags: list[str] = []
        if sample.is_candidate_spacing:
            flags.append("spacing")
        if sample.is_candidate_transverse:
            flags.append("transverse")
        if sample.is_candidate_direction:
            flags.append("direction")
        if not flags:
            return "none"
        return ", ".join(flags)
