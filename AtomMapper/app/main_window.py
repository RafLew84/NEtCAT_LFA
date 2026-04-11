"""Main window for the AtomMapper application."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDockWidget,
    QFileDialog,
    QDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .controller import AtomMapperController
from .gaussian_preview import GaussianFitPreviewWidget
from .io import SUPPORTED_STM_EXTENSIONS
from .models import AtomPoint
from .pyqtgraph_image_view import PyQtGraphSTMViewport
from .pyqtgraph_preview_bridge import PyQtGraphPreviewBridge
from .preprocessing_dialog import PreprocessingDialog
from .preprocessing_state import PreprocessingMethod
from .roi_preview import ROIPreviewWidget

logger = logging.getLogger(__name__)


class AtomMapperMainWindow(QMainWindow):
    """Bootstrap window with loaded-file state and selection list."""

    def __init__(self, controller: AtomMapperController | None = None) -> None:
        super().__init__()
        self.controller = controller or AtomMapperController(self)

        self.setWindowTitle("AtomMapper")
        self.resize(1200, 800)

        central = QWidget(self)
        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(24, 24, 24, 24)
        root_layout.setSpacing(16)

        left_panel = QWidget(central)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10)

        left_title = QLabel("Loaded STM files")
        left_title.setStyleSheet("font-size: 16px; font-weight: 600;")

        self.load_button = QPushButton("Load STM Files...")
        self.preprocessing_button = QPushButton("Preprocessing")
        self.preprocessing_button.setObjectName("atommapper_preprocessing_button")
        self.file_list_hint_label = QLabel("No STM files loaded. Use 'Load STM Files...' to start.")
        self.file_list_hint_label.setWordWrap(True)
        self.file_list_hint_label.setStyleSheet("font-size: 12px; color: palette(mid);")
        self.file_list_widget = QListWidget()
        self.file_list_widget.setObjectName("atommapper_file_list")
        rows_title = QLabel("Atom rows")
        rows_title.setStyleSheet("font-size: 16px; font-weight: 600;")
        self.active_row_label = QLabel("Active row: none")
        self.active_row_label.setWordWrap(True)
        self.active_row_label.setStyleSheet("font-size: 12px; color: palette(mid);")
        self.new_row_button = QPushButton("New Row")
        self.new_row_button.setObjectName("atommapper_new_row_button")
        self.delete_row_button = QPushButton("Delete Row")
        self.delete_row_button.setObjectName("atommapper_delete_row_button")
        self.add_point_button = QPushButton("Add Point")
        self.add_point_button.setObjectName("atommapper_add_point_button")
        self.delete_point_button = QPushButton("Delete Point")
        self.delete_point_button.setObjectName("atommapper_delete_point_button")
        row_button_panel = QWidget(left_panel)
        row_button_layout = QHBoxLayout(row_button_panel)
        row_button_layout.setContentsMargins(0, 0, 0, 0)
        row_button_layout.setSpacing(8)
        row_button_layout.addWidget(self.new_row_button)
        row_button_layout.addWidget(self.delete_row_button)
        row_button_layout.addWidget(self.add_point_button)
        row_button_layout.addWidget(self.delete_point_button)
        self.row_list_hint_label = QLabel("Load or select an STM image to manage rows.")
        self.row_list_hint_label.setWordWrap(True)
        self.row_list_hint_label.setStyleSheet("font-size: 12px; color: palette(mid);")
        self.row_list_widget = QListWidget()
        self.row_list_widget.setObjectName("atommapper_row_list")

        left_layout.addWidget(left_title)
        left_layout.addWidget(self.load_button)
        left_layout.addWidget(self.preprocessing_button)
        left_layout.addWidget(self.file_list_hint_label)
        left_layout.addWidget(self.file_list_widget, 1)
        left_layout.addWidget(rows_title)
        left_layout.addWidget(self.active_row_label)
        left_layout.addWidget(row_button_panel)
        left_layout.addWidget(self.row_list_hint_label)
        left_layout.addWidget(self.row_list_widget, 1)
        left_panel.setMinimumWidth(280)
        left_panel.setMaximumWidth(360)

        right_panel = QWidget(central)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(12)

        title = QLabel("AtomMapper")
        title.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        title.setStyleSheet("font-size: 24px; font-weight: 600;")

        subtitle = QLabel(
            "Standalone STM helper for atom localization, row tracking, and Gaussian ROI fitting."
        )
        subtitle.setWordWrap(True)
        subtitle.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)

        self.active_image_label = QLabel("Active image: none")
        self.active_image_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        self.active_image_label.setStyleSheet("font-size: 13px; color: palette(mid);")
        self.show_gaussian_fit_checkbox = QCheckBox("Show Gaussian Fit", right_panel)
        self.show_gaussian_fit_checkbox.setChecked(True)
        self.workflow_status_label = QLabel("Workflow status: load an STM image to begin.")
        self.workflow_status_label.setWordWrap(True)
        self.workflow_status_label.setStyleSheet("font-size: 12px; color: palette(mid);")

        self.image_viewport = PyQtGraphSTMViewport(right_panel)
        self.roi_preview = ROIPreviewWidget(right_panel)
        self.gaussian_fit_preview = GaussianFitPreviewWidget(right_panel)
        self.preview_bridge = PyQtGraphPreviewBridge(
            self.image_viewport,
            self.roi_preview,
            self.gaussian_fit_preview,
            self,
        )
        points_title = QLabel("Saved points")
        points_title.setStyleSheet("font-size: 16px; font-weight: 600;")
        self.points_table_hint_label = QLabel(
            "Saved points for the selected STM file family will appear here."
        )
        self.points_table_hint_label.setWordWrap(True)
        self.points_table_hint_label.setStyleSheet("font-size: 12px; color: palette(mid);")
        self.points_table_widget = QTableWidget(0, 7, right_panel)
        self.points_table_widget.setObjectName("atommapper_points_table")
        self.points_table_widget.setHorizontalHeaderLabels(
            ["row", "index", "x_px", "y_px", "sigma_x", "sigma_y", "status"]
        )
        self.points_table_widget.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.points_table_widget.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.points_table_widget.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.points_table_widget.verticalHeader().setVisible(False)
        self.points_table_widget.setAlternatingRowColors(True)
        self.points_table_widget.setMinimumHeight(220)
        self.points_table_widget.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.points_table_widget.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.points_table_widget.horizontalHeader().setStretchLastSection(True)
        self.saved_points_panel = QWidget(right_panel)
        self.saved_points_panel.setObjectName("atommapper_saved_points_panel")
        saved_points_layout = QVBoxLayout(self.saved_points_panel)
        saved_points_layout.setContentsMargins(0, 0, 0, 0)
        saved_points_layout.setSpacing(8)
        saved_points_layout.addWidget(points_title)
        saved_points_layout.addWidget(self.points_table_hint_label)
        saved_points_layout.addWidget(self.points_table_widget)

        self.analysis_grid_panel = QWidget(right_panel)
        self.analysis_grid_panel.setObjectName("atommapper_analysis_grid_panel")
        analysis_grid_layout = QGridLayout(self.analysis_grid_panel)
        analysis_grid_layout.setContentsMargins(0, 0, 0, 0)
        analysis_grid_layout.setHorizontalSpacing(12)
        analysis_grid_layout.setVerticalSpacing(12)
        analysis_grid_layout.addWidget(self.roi_preview, 0, 0)
        analysis_grid_layout.addWidget(self.gaussian_fit_preview, 1, 0)
        analysis_grid_layout.addWidget(self.image_viewport, 0, 1, 2, 1)
        analysis_grid_layout.setColumnStretch(0, 2)
        analysis_grid_layout.setColumnStretch(1, 3)
        analysis_grid_layout.setRowStretch(0, 1)
        analysis_grid_layout.setRowStretch(1, 1)

        right_layout.addWidget(title)
        right_layout.addWidget(subtitle)
        right_layout.addWidget(self.active_image_label)
        right_layout.addWidget(self.show_gaussian_fit_checkbox)
        right_layout.addWidget(self.workflow_status_label)
        right_layout.addWidget(self.analysis_grid_panel, 1)

        root_layout.addWidget(left_panel)
        root_layout.addWidget(right_panel, 1)

        self.setCentralWidget(central)
        self.analysis_dock = QDockWidget("Analysis", self)
        self.analysis_dock.setObjectName("atommapper_analysis_dock")
        self.analysis_dock.setAllowedAreas(
            Qt.DockWidgetArea.BottomDockWidgetArea
            | Qt.DockWidgetArea.LeftDockWidgetArea
            | Qt.DockWidgetArea.RightDockWidgetArea
        )
        self.analysis_dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        self.analysis_dock.setWidget(self.saved_points_panel)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.analysis_dock)
        self._preprocessing_dialog_class = PreprocessingDialog
        self._active_point_id_by_source_group: dict[str, str] = {}
        self.statusBar().showMessage("Ready. Load an STM file to begin.", 5000)
        self._connect_signals()
        self._refresh_file_list()
        self._refresh_row_list()
        self._refresh_points_table()
        self._refresh_image_point_overlay()
        self._update_active_image_label(self.controller.active_image)
        self._update_preprocess_controls(self.controller.active_image)
        self._update_active_row_label(self.controller.active_row)
        self._update_row_controls(self.controller.active_image, self.controller.active_row)
        self._update_point_controls()
        self._handle_active_image_changed(self.controller.active_image)

    def _connect_signals(self) -> None:
        self.load_button.clicked.connect(self._open_file_dialog)
        self.preprocessing_button.clicked.connect(self._open_preprocessing_dialog)
        self.new_row_button.clicked.connect(self._create_new_row)
        self.delete_row_button.clicked.connect(self._delete_active_row)
        self.add_point_button.clicked.connect(self._add_point_from_current_roi)
        self.delete_point_button.clicked.connect(self._delete_active_point)
        self.file_list_widget.currentRowChanged.connect(self._on_current_row_changed)
        self.row_list_widget.currentRowChanged.connect(self._on_current_row_changed_for_rows)
        self.points_table_widget.itemSelectionChanged.connect(self._on_points_table_selection_changed)
        self.controller.loaded_images_changed.connect(self._refresh_file_list)
        self.controller.loaded_images_changed.connect(self._refresh_row_list)
        self.controller.active_image_changed.connect(self._update_active_image_label)
        self.controller.active_image_changed.connect(self._update_preprocess_controls)
        self.controller.active_image_changed.connect(self._handle_active_image_changed)
        self.controller.active_image_changed.connect(self._handle_active_image_changed_for_rows)
        self.controller.active_image_changed.connect(self._refresh_points_table)
        self.controller.active_image_changed.connect(self._refresh_image_point_overlay)
        self.controller.roi_state_changed.connect(self._handle_roi_state_changed)
        self.controller.rows_changed.connect(self._refresh_row_list)
        self.controller.rows_changed.connect(self._handle_rows_changed)
        self.controller.rows_changed.connect(self._refresh_points_table)
        self.controller.rows_changed.connect(self._refresh_image_point_overlay)
        self.controller.active_row_changed.connect(self._handle_active_row_changed)
        self.controller.row_points_changed.connect(self._handle_row_points_changed)
        self.show_gaussian_fit_checkbox.stateChanged.connect(self._on_show_gaussian_fit_changed)
        self.preview_bridge.roi_state_edited.connect(self.controller.update_active_roi_state)
        self.image_viewport.point_selected.connect(self._handle_viewport_point_selected)
        self.image_viewport.point_move_requested.connect(self._handle_viewport_point_move_requested)

    def _build_file_dialog_filter(self) -> str:
        patterns = " ".join(f"*{suffix}" for suffix in sorted(SUPPORTED_STM_EXTENSIONS))
        return f"STM files ({patterns})"

    def _open_file_dialog(self) -> None:
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Open STM Files",
            "",
            self._build_file_dialog_filter(),
        )
        self._load_files_from_paths(file_paths)

    def _load_files_from_paths(self, file_paths: list[str]) -> None:
        if not file_paths:
            self.statusBar().showMessage("Load cancelled.", 3000)
            return

        loaded_count = 0
        error_messages: list[str] = []

        for file_path in file_paths:
            try:
                self.controller.load_files([file_path])
                loaded_count += 1
            except Exception as exc:  # pragma: no cover - covered via monkeypatched GUI test
                logger.exception("Failed to load STM file '%s': %s", file_path, exc)
                error_messages.append(f"{Path(file_path).name}: {exc}")

        if loaded_count:
            noun = "file" if loaded_count == 1 else "files"
            self.statusBar().showMessage(f"Loaded {loaded_count} STM {noun}.", 5000)

        if error_messages:
            self.statusBar().showMessage("Some files failed to load.", 7000)
            QMessageBox.warning(
                self,
                "AtomMapper - Load Error",
                "Could not load one or more STM files:\n\n" + "\n".join(error_messages),
            )

    def _refresh_file_list(self) -> None:
        self.file_list_widget.blockSignals(True)
        self.file_list_widget.clear()

        display_entries = self._build_file_list_entries()
        active_image = self.controller.active_image
        active_row = None

        for row_index, (controller_index, image) in enumerate(display_entries):
            item = QListWidgetItem(self._format_file_list_label(image))
            item.setToolTip(self._build_file_list_tooltip(image))
            item.setData(Qt.ItemDataRole.UserRole, controller_index)
            self.file_list_widget.addItem(item)
            if active_image is not None and image.image_id == active_image.image_id:
                active_row = row_index

        if active_row is not None and 0 <= active_row < self.file_list_widget.count():
            self.file_list_widget.setCurrentRow(active_row)

        image_count = self.file_list_widget.count()
        self.file_list_widget.setEnabled(image_count > 0)
        if image_count == 0:
            self.file_list_hint_label.setText("No STM files loaded. Use 'Load STM Files...' to start.")
        else:
            noun = "file" if image_count == 1 else "files"
            self.file_list_hint_label.setText(f"{image_count} STM {noun} loaded.")

        self.file_list_widget.blockSignals(False)

    def _on_current_row_changed(self, row: int) -> None:
        if row < 0:
            return
        item = self.file_list_widget.item(row)
        if item is None:
            return
        controller_index = item.data(Qt.ItemDataRole.UserRole)
        if controller_index is None:
            return
        image = self.controller.select_image(int(controller_index))
        if image is not None:
            self.statusBar().showMessage(f"Selected {image.display_name}.", 3000)

    def _refresh_row_list(self, *_args: Any) -> None:
        self.row_list_widget.blockSignals(True)
        self.row_list_widget.clear()

        active_group = self.controller.active_source_group_id
        active_row = self.controller.active_row
        rows = self.controller.rows_for_source_group(active_group) if active_group is not None else ()
        active_index = None

        for row_index, row in enumerate(rows):
            item = QListWidgetItem(self._format_row_list_label(row))
            item.setToolTip(self._build_row_list_tooltip(row))
            item.setData(Qt.ItemDataRole.UserRole, row.row_id)
            self.row_list_widget.addItem(item)
            if active_row is not None and row.row_id == active_row.row_id:
                active_index = row_index

        if active_index is not None and 0 <= active_index < self.row_list_widget.count():
            self.row_list_widget.setCurrentRow(active_index)

        row_count = self.row_list_widget.count()
        self.row_list_widget.setEnabled(row_count > 0)
        if active_group is None:
            self.row_list_hint_label.setText("Load or select an STM image to manage rows.")
        elif row_count == 0:
            self.row_list_hint_label.setText(
                "No rows for the selected STM file family. Use 'New Row' to create one."
            )
        else:
            noun = "row" if row_count == 1 else "rows"
            self.row_list_hint_label.setText(f"{row_count} atom {noun} for the selected STM file family.")

        self.row_list_widget.blockSignals(False)

    def _on_current_row_changed_for_rows(self, row_index: int) -> None:
        if row_index < 0:
            return
        item = self.row_list_widget.item(row_index)
        if item is None:
            return
        row_id = item.data(Qt.ItemDataRole.UserRole)
        if not row_id:
            return
        selected_row = self.controller.select_row(str(row_id))
        if selected_row is not None:
            self.statusBar().showMessage(f"Selected {selected_row.display_name}.", 3000)

    def _refresh_points_table(self, *_args: Any) -> None:
        self.points_table_widget.blockSignals(True)
        self.points_table_widget.setRowCount(0)
        active_group = self.controller.active_source_group_id
        if active_group is None:
            self.points_table_widget.clearSelection()
            self.points_table_hint_label.setText(
                "Load or select an STM image to inspect saved points."
            )
            self.points_table_widget.blockSignals(False)
            self._update_point_controls()
            return

        rows = self.controller.rows_for_source_group(active_group)
        display_points: list[tuple[Any, Any]] = []
        for row in rows:
            for point in row.points:
                display_points.append((row, point))

        display_points.sort(key=lambda item: (item[0].display_name.lower(), item[1].point_index, item[1].point_id))
        self._normalize_active_point_selection(display_points)
        self.points_table_widget.setRowCount(len(display_points))

        active_point_id = self._active_point_id_for_current_group()
        active_row_index: int | None = None
        for row_index, (row, point) in enumerate(display_points):
            self._set_points_table_item(row_index, 0, row.display_name, point_id=point.point_id)
            self._set_points_table_item(row_index, 1, str(point.point_index), point_id=point.point_id)
            self._set_points_table_item(row_index, 2, f"{point.x_px:.3f}", point_id=point.point_id)
            self._set_points_table_item(row_index, 3, f"{point.y_px:.3f}", point_id=point.point_id)
            self._set_points_table_item(
                row_index,
                4,
                self._format_optional_float(point.sigma_x_px),
                point_id=point.point_id,
            )
            self._set_points_table_item(
                row_index,
                5,
                self._format_optional_float(point.sigma_y_px),
                point_id=point.point_id,
            )
            self._set_points_table_item(
                row_index,
                6,
                self._format_point_status(point),
                point_id=point.point_id,
            )
            if point.point_id == active_point_id:
                active_row_index = row_index

        self.points_table_widget.resizeColumnsToContents()
        if active_row_index is not None:
            self.points_table_widget.selectRow(active_row_index)
            self.points_table_widget.setCurrentCell(active_row_index, 0)
        else:
            self.points_table_widget.clearSelection()

        if not display_points:
            self.points_table_hint_label.setText(
                "No saved points for the selected STM file family yet."
            )
        else:
            noun = "point" if len(display_points) == 1 else "points"
            self.points_table_hint_label.setText(
                f"Showing {len(display_points)} saved {noun} for the selected STM file family."
            )
        self.points_table_widget.blockSignals(False)
        self._update_point_controls()

    def _refresh_image_point_overlay(self, *_args: Any) -> None:
        active_group = self.controller.active_source_group_id
        rows = self.controller.rows_for_source_group(active_group) if active_group is not None else ()
        active_image = self.controller.active_image
        self._normalize_active_point_selection(
            [(row, point) for row in rows for point in row.points]
        )
        self.image_viewport.set_atom_rows(
            rows,
            active_row_id=self.controller.active_row_id,
            active_image_id=None if active_image is None else active_image.image_id,
            active_point_id=self._active_point_id_for_current_group(),
        )

    def _build_file_list_entries(self) -> list[tuple[int, Any]]:
        groups: dict[str, list[tuple[int, Any]]] = {}
        group_order: list[str] = []

        for controller_index, image in enumerate(self.controller.loaded_images):
            if image.source_group_id not in groups:
                groups[image.source_group_id] = []
                group_order.append(image.source_group_id)
            groups[image.source_group_id].append((controller_index, image))

        ordered_entries: list[tuple[int, Any]] = []
        for source_group_id in group_order:
            group_entries = groups[source_group_id]
            originals = [entry for entry in group_entries if entry[1].is_original]
            variants = [entry for entry in group_entries if not entry[1].is_original]
            ordered_entries.extend(originals + variants)
        return ordered_entries

    @staticmethod
    def _format_file_list_label(image: Any) -> str:
        if image.is_original:
            return image.display_name
        return f"  - {image.display_name}"

    @staticmethod
    def _build_file_list_tooltip(image: Any) -> str:
        if image.is_original:
            return f"Original image\nPath: {image.source_path}"
        return (
            f"Variant: {image.variant_name}\n"
            f"Parent image id: {image.parent_image_id}\n"
            f"Path: {image.source_path}"
        )

    @staticmethod
    def _format_row_list_label(row: Any) -> str:
        noun = "point" if row.point_count == 1 else "points"
        return f"{row.display_name} ({row.point_count} {noun})"

    @staticmethod
    def _build_row_list_tooltip(row: Any) -> str:
        return (
            f"Row id: {row.row_id}\n"
            f"Source family: {row.source_group_id}\n"
            f"Points: {row.point_count}"
        )

    @staticmethod
    def _format_optional_float(value: Any) -> str:
        if value is None:
            return "-"
        return f"{float(value):.3f}"

    @staticmethod
    def _format_point_status(point: AtomPoint) -> str:
        if point.manual_override:
            source = point.manual_override_source or "manual"
            return f"manual ({source})"
        if point.fit_success:
            return "fit"
        if point.metadata.get("fallback_used"):
            return "fallback"
        return "stored"

    def _set_points_table_item(self, row: int, column: int, text: str, *, point_id: str | None = None) -> None:
        item = QTableWidgetItem(text)
        item.setTextAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            if column > 0
            else Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        if point_id is not None:
            item.setData(Qt.ItemDataRole.UserRole, point_id)
        self.points_table_widget.setItem(row, column, item)

    def _active_point_id_for_current_group(self) -> str | None:
        active_group = self.controller.active_source_group_id
        if active_group is None:
            return None
        return self._active_point_id_by_source_group.get(active_group)

    def _active_point_for_current_group(self) -> AtomPoint | None:
        active_point_id = self._active_point_id_for_current_group()
        if active_point_id is None:
            return None

        active_group = self.controller.active_source_group_id
        if active_group is None:
            return None

        for row in self.controller.rows_for_source_group(active_group):
            for point in row.points:
                if point.point_id == active_point_id:
                    return point
        return None

    def _set_active_point_for_current_group(self, point_id: str | None) -> None:
        active_group = self.controller.active_source_group_id
        if active_group is None:
            return

        normalized_point_id = None if point_id is None else str(point_id).strip() or None
        current_point_id = self._active_point_id_by_source_group.get(active_group)
        if normalized_point_id == current_point_id:
            return

        if normalized_point_id is None:
            self._active_point_id_by_source_group.pop(active_group, None)
        else:
            self._active_point_id_by_source_group[active_group] = normalized_point_id

        self._refresh_points_table()
        self._refresh_image_point_overlay()
        self._update_point_controls()

    def _normalize_active_point_selection(self, display_points: list[tuple[Any, AtomPoint]]) -> None:
        active_group = self.controller.active_source_group_id
        if active_group is None:
            return
        active_point_id = self._active_point_id_by_source_group.get(active_group)
        if active_point_id is None:
            return
        if any(point.point_id == active_point_id for _row, point in display_points):
            return
        self._active_point_id_by_source_group.pop(active_group, None)
        self._update_point_controls()

    def _update_active_image_label(self, active_image: Any) -> None:
        if active_image is None:
            self.active_image_label.setText("Active image: none")
            return
        self.active_image_label.setText(
            f"Active image: {active_image.display_name} "
            f"({active_image.pixels_x}x{active_image.pixels_y} px)"
        )

    def _update_preprocess_controls(self, active_image: Any) -> None:
        has_image = active_image is not None
        self.preprocessing_button.setEnabled(has_image)
        if has_image:
            self.preprocessing_button.setToolTip(
                f"Open preprocessing dialog for {active_image.display_name}."
            )
        else:
            self.preprocessing_button.setToolTip("Load or select an STM image first.")

    def _update_active_row_label(self, active_row: Any) -> None:
        if active_row is None:
            self.active_row_label.setText("Active row: none")
            return
        noun = "point" if active_row.point_count == 1 else "points"
        self.active_row_label.setText(
            f"Active row: {active_row.display_name} ({active_row.point_count} {noun})"
        )

    def _update_row_controls(self, active_image: Any, active_row: Any) -> None:
        has_image = active_image is not None
        has_row = active_row is not None
        self.new_row_button.setEnabled(has_image)
        self.delete_row_button.setEnabled(has_row)
        self.add_point_button.setEnabled(has_image and has_row)
        if has_image:
            self.new_row_button.setToolTip("Create a new atom row for the selected STM file family.")
        else:
            self.new_row_button.setToolTip("Load or select an STM image first.")
        if has_row:
            self.delete_row_button.setToolTip(f"Delete the active row '{active_row.display_name}'.")
            self.add_point_button.setToolTip(
                f"Store the current ROI center in '{active_row.display_name}' using Gaussian fit or ROI fallback."
            )
        else:
            self.delete_row_button.setToolTip("Select an atom row first.")
            self.add_point_button.setToolTip("Create or select an atom row first.")

    def _update_point_controls(self) -> None:
        active_point = self._active_point_for_current_group()
        has_point = active_point is not None
        self.delete_point_button.setEnabled(has_point)
        if has_point:
            self.delete_point_button.setToolTip(
                f"Delete selected point {active_point.point_index} from the current STM file family."
            )
        else:
            self.delete_point_button.setToolTip("Select a saved point in the table or on the image first.")

    def _open_preprocessing_dialog(self) -> None:
        active_image = self.controller.active_image
        if active_image is None:
            self.statusBar().showMessage("Select an STM image before preprocessing.", 5000)
            self.workflow_status_label.setText(
                "Workflow status: select an STM image before opening preprocessing."
            )
            return

        dialog = self._preprocessing_dialog_class(active_image, self)
        self.statusBar().showMessage(
            f"Opened preprocessing dialog for {active_image.display_name}.",
            3000,
        )
        self.analysis_grid_panel.setUpdatesEnabled(False)
        self.image_viewport.setUpdatesEnabled(False)
        try:
            result = dialog.exec()
        finally:
            self.image_viewport.setUpdatesEnabled(True)
            self.analysis_grid_panel.setUpdatesEnabled(True)
            self.image_viewport.update()
            self.analysis_grid_panel.update()
        if result == int(QDialog.DialogCode.Accepted):
            self._apply_preprocessing_dialog_result(dialog)
        else:
            self.statusBar().showMessage("Preprocessing cancelled.", 3000)
            self.workflow_status_label.setText(
                "Workflow status: preprocessing dialog closed without changes."
            )

    def _apply_preprocessing_dialog_result(self, dialog: Any) -> None:
        """Create a preprocessing variant from an accepted dialog."""

        state = getattr(dialog, "preprocessing_state", None)
        if state is None:
            self.statusBar().showMessage("Preprocessing dialog accepted without a valid state.", 4000)
            self.workflow_status_label.setText(
                "Workflow status: preprocessing dialog accepted, but no preprocessing state was provided."
            )
            return

        try:
            if state.method is PreprocessingMethod.BLUR:
                variant = self.controller.create_blur_variant_for_active_image(
                    sigma_px=state.blur.sigma_px,
                    make_active=True,
                )
                status_suffix = f"sigma {state.blur.sigma_px:.2f}px"
            elif state.method is PreprocessingMethod.NLM:
                variant = self.controller.create_nlm_variant_for_active_image(
                    h=state.nlm.h,
                    patch_size=state.nlm.patch_size,
                    patch_distance=state.nlm.patch_distance,
                    fast_mode=state.nlm.fast_mode,
                    make_active=True,
                )
                status_suffix = (
                    f"h {state.nlm.h:.3f}, patch {state.nlm.patch_size}, "
                    f"distance {state.nlm.patch_distance}, fast={state.nlm.fast_mode}"
                )
            elif state.method is PreprocessingMethod.BM3D:
                variant = self.controller.create_bm3d_variant_for_active_image(
                    sigma_psd=state.bm3d.sigma_psd,
                    stage=state.bm3d.stage,
                    make_active=True,
                )
                status_suffix = f"sigma_psd {state.bm3d.sigma_psd:.3f}, stage {state.bm3d.stage}"
            else:
                self.statusBar().showMessage(f"{state.method.label} apply is not available yet.", 4000)
                self.workflow_status_label.setText(
                    f"Workflow status: {state.method.label} is selected, but this apply path is not enabled in the current step."
                )
                return
        except Exception as exc:  # pragma: no cover - GUI error path
            logger.exception("Failed to apply %s preprocessing: %s", state.method.value, exc)
            QMessageBox.warning(
                self,
                "AtomMapper - Preprocessing Error",
                f"Could not create {state.method.label} variant:\n\n{exc}",
            )
            self.statusBar().showMessage(f"{state.method.label} preprocessing failed.", 5000)
            self.workflow_status_label.setText(
                f"Workflow status: {state.method.label} preprocessing failed."
            )
            return

        self.statusBar().showMessage(
            f"Created {variant.variant_name} variant '{variant.display_name}' ({status_suffix}).",
            5000,
        )
        self.workflow_status_label.setText(
            f"Workflow status: created {variant.variant_name} variant '{variant.display_name}' with {status_suffix}."
        )

    def _create_new_row(self) -> None:
        active_image = self.controller.active_image
        if active_image is None:
            self.statusBar().showMessage("Select an STM image before creating a row.", 4000)
            self.workflow_status_label.setText(
                "Workflow status: select an STM image before creating an atom row."
            )
            return

        next_index = len(self.controller.rows_for_source_group(active_image.source_group_id)) + 1
        row = self.controller.create_row_for_active_source_group(display_name=f"Row {next_index}")
        self.statusBar().showMessage(f"Created {row.display_name}.", 3000)
        self.workflow_status_label.setText(
            f"Workflow status: active row set to {row.display_name}."
        )

    def _delete_active_row(self) -> None:
        active_row = self.controller.active_row
        if active_row is None:
            self.statusBar().showMessage("Select an atom row before deleting it.", 4000)
            self.workflow_status_label.setText(
                "Workflow status: no active row selected for deletion."
            )
            return

        removed_row = self.controller.remove_row(active_row.row_id)
        if removed_row is None:
            return
        noun = "point" if removed_row.point_count == 1 else "points"
        self.statusBar().showMessage(
            f"Deleted {removed_row.display_name} ({removed_row.point_count} {noun}).",
            3000,
        )
        self.workflow_status_label.setText(
            f"Workflow status: deleted {removed_row.display_name} with {removed_row.point_count} {noun}."
        )

    def _add_point_from_current_roi(self) -> None:
        active_image = self.controller.active_image
        active_row = self.controller.active_row
        roi = self.controller.active_roi_state

        if active_image is None or roi is None:
            self.statusBar().showMessage("Select an STM image with a valid ROI before adding a point.", 4000)
            self.workflow_status_label.setText(
                "Workflow status: select an STM image and ROI before adding a point."
            )
            return

        if active_row is None:
            self.statusBar().showMessage("Create or select an atom row before adding a point.", 4000)
            self.workflow_status_label.setText(
                "Workflow status: no active row selected for point capture."
            )
            return

        fit_result = self.preview_bridge.compute_current_fit_result()
        fallback_used = False
        fit_error_message = None
        x_px: float
        y_px: float
        sigma_x_px = None
        sigma_y_px = None
        amplitude = None
        theta_deg = None
        offset = None
        fit_success = False
        fit_method = "roi_center_fallback"

        if fit_result is not None and fit_result.center_image_yx is not None:
            y_px = float(fit_result.center_image_yx[0])
            x_px = float(fit_result.center_image_yx[1])
            amplitude = fit_result.amplitude
            sigma_x_px = fit_result.sigma_x
            sigma_y_px = fit_result.sigma_y
            theta_deg = None if fit_result.theta_rad is None else math.degrees(float(fit_result.theta_rad))
            offset = fit_result.offset
            fit_success = bool(fit_result.success)
            fit_method = fit_result.method
            fit_error_message = fit_result.error_message
            if fit_result.center_std_yx is not None:
                sigma_y_px = fit_result.center_std_yx[0] if fit_result.sigma_y is None else sigma_y_px
                sigma_x_px = fit_result.center_std_yx[1] if fit_result.sigma_x is None else sigma_x_px
        else:
            fallback_used = True
            x_px = roi.x + (roi.width / 2.0)
            y_px = roi.y + (roi.height / 2.0)
            fit_error_message = "Gaussian fit unavailable; ROI center fallback used."

        if fit_result is not None and fit_result.center_image_yx is None:
            fallback_used = True
            x_px = roi.x + (roi.width / 2.0)
            y_px = roi.y + (roi.height / 2.0)
            fit_success = False
            fit_method = f"{fit_result.method}_fallback"
            fit_error_message = fit_result.error_message or "Gaussian fit unavailable; ROI center fallback used."

        point = AtomPoint(
            row_id=active_row.row_id,
            image_id=active_image.image_id,
            source_group_id=active_image.source_group_id,
            point_index=active_row.next_point_index,
            x_px=x_px,
            y_px=y_px,
            amplitude=amplitude,
            sigma_x_px=sigma_x_px,
            sigma_y_px=sigma_y_px,
            theta_deg=theta_deg,
            offset=offset,
            fit_success=fit_success,
            fit_error_message=fit_error_message,
            metadata={
                "fit_method": fit_method,
                "roi_x": roi.x,
                "roi_y": roi.y,
                "roi_width": roi.width,
                "roi_height": roi.height,
                "fallback_used": fallback_used,
            },
        )
        updated_row = self.controller.add_point_to_row(point)
        self.statusBar().showMessage(
            f"Added point {point.point_index} to {updated_row.display_name} at x={point.x_px:.2f}, y={point.y_px:.2f}.",
            4000,
        )
        if fallback_used:
            self.workflow_status_label.setText(
                f"Workflow status: added point {point.point_index} to {updated_row.display_name} using ROI center fallback."
            )
        else:
            self.workflow_status_label.setText(
                f"Workflow status: added point {point.point_index} to {updated_row.display_name} from Gaussian fit."
            )

    def _delete_active_point(self) -> None:
        active_point = self._active_point_for_current_group()
        if active_point is None:
            self.statusBar().showMessage("Select a saved point before deleting it.", 4000)
            self.workflow_status_label.setText(
                "Workflow status: no saved point selected for deletion."
            )
            return

        row = next(
            (
                candidate_row
                for candidate_row in self.controller.rows_for_source_group(active_point.source_group_id)
                if candidate_row.row_id == active_point.row_id
            ),
            None,
        )
        row_name = row.display_name if row is not None else active_point.row_id
        removed_point_index = active_point.point_index
        updated_row = self.controller.remove_point_from_row(active_point.row_id, active_point.point_id)
        self._set_active_point_for_current_group(None)
        self.statusBar().showMessage(
            f"Deleted point {removed_point_index} from {row_name}.",
            4000,
        )
        noun = "point" if updated_row.point_count == 1 else "points"
        self.workflow_status_label.setText(
            f"Workflow status: deleted point {removed_point_index} from {row_name}. Active point selection cleared; {updated_row.point_count} {noun} remain."
        )

    def _handle_active_image_changed_for_rows(self, active_image: Any) -> None:
        self._refresh_row_list()
        self._update_row_controls(active_image, self.controller.active_row)

    def _handle_rows_changed(self) -> None:
        self._refresh_row_list()
        self._update_row_controls(self.controller.active_image, self.controller.active_row)

    def _handle_active_row_changed(self, active_row: Any) -> None:
        self._update_active_row_label(active_row)
        self._refresh_row_list()
        self._refresh_points_table()
        self._refresh_image_point_overlay()
        self._update_row_controls(self.controller.active_image, active_row)
        self._update_point_controls()

    def _handle_row_points_changed(self, _updated_row: Any) -> None:
        self._refresh_row_list()
        self._refresh_points_table()
        self._refresh_image_point_overlay()
        self._update_active_row_label(self.controller.active_row)
        self._update_point_controls()

    def _on_points_table_selection_changed(self) -> None:
        selected_items = self.points_table_widget.selectedItems()
        if not selected_items:
            self._set_active_point_for_current_group(None)
            return

        point_id = selected_items[0].data(Qt.ItemDataRole.UserRole)
        if point_id:
            self._set_active_point_for_current_group(str(point_id))

    def _handle_viewport_point_selected(self, payload: Any) -> None:
        if not isinstance(payload, dict):
            return
        point_id = payload.get("point_id")
        if point_id:
            self._set_active_point_for_current_group(str(point_id))

    def _handle_viewport_point_move_requested(self, payload: Any) -> None:
        if not isinstance(payload, dict):
            return

        row_id = payload.get("row_id")
        point_id = payload.get("point_id")
        x_px = payload.get("x_px")
        y_px = payload.get("y_px")
        source = payload.get("source", "drag")
        if not row_id or not point_id or x_px is None or y_px is None:
            return

        try:
            updated_row = self.controller.move_point_in_row(
                row_id=str(row_id),
                point_id=str(point_id),
                x_px=float(x_px),
                y_px=float(y_px),
                source=str(source),
            )
        except Exception as exc:  # pragma: no cover - GUI error path
            logger.exception("Failed to move point '%s': %s", point_id, exc)
            self.statusBar().showMessage("Manual point move failed.", 4000)
            self.workflow_status_label.setText(
                "Workflow status: manual point correction failed."
            )
            self._refresh_image_point_overlay()
            return

        moved_point = next(
            (point for point in updated_row.points if point.point_id == str(point_id)),
            None,
        )
        self._set_active_point_for_current_group(str(point_id))
        if moved_point is None:
            return

        self.statusBar().showMessage(
            f"Moved point {moved_point.point_index} in {updated_row.display_name} to x={moved_point.x_px:.2f}, y={moved_point.y_px:.2f}.",
            4000,
        )
        self.workflow_status_label.setText(
            f"Workflow status: manually corrected point {moved_point.point_index} in {updated_row.display_name}; selection preserved and point marked as manual."
        )

    def _handle_active_image_changed(self, active_image: Any) -> None:
        self.preview_bridge.set_loaded_image(active_image)
        self.preview_bridge.set_roi_state(self.controller.active_roi_state)
        self._sync_gaussian_preview_visibility()
        self._update_workflow_status()

    def _handle_roi_state_changed(self, roi_state: Any) -> None:
        self.preview_bridge.set_roi_state(roi_state)
        self._sync_gaussian_preview_visibility()
        self._update_workflow_status()

    def _update_workflow_status(self) -> None:
        image = self.controller.active_image
        roi = self.controller.active_roi_state

        if image is None:
            self.workflow_status_label.setText("Workflow status: load an STM image to begin.")
            return

        if roi is None:
            self.workflow_status_label.setText(
                f"Workflow status: {image.display_name} loaded. Waiting for ROI geometry."
            )
            return

        if not self.show_gaussian_fit_checkbox.isChecked():
            self.workflow_status_label.setText(
                f"Workflow status: ROI preview active for {image.display_name}. Gaussian fit preview hidden."
            )
            return

        patch = self.roi_preview.current_patch_data
        if patch is None:
            self.workflow_status_label.setText(
                f"Workflow status: ROI for {image.display_name} is outside image bounds."
            )
            return

        fit_result = self.gaussian_fit_preview.current_fit_result
        if fit_result is None:
            self.workflow_status_label.setText(
                "Workflow status: "
                f"ROI {patch.shape[1]}x{patch.shape[0]} px ready. "
                "Gaussian fit preview is waiting for refresh."
            )
            return

        if fit_result.success and fit_result.center_patch_yx is not None:
            self.workflow_status_label.setText(
                "Workflow status: "
                f"ROI {patch.shape[1]}x{patch.shape[0]} px ready. "
                f"Gaussian center y={fit_result.center_patch_yx[0]:.2f} "
                f"x={fit_result.center_patch_yx[1]:.2f}."
            )
        else:
            self.workflow_status_label.setText(
                "Workflow status: "
                f"ROI {patch.shape[1]}x{patch.shape[0]} px ready. "
                f"{fit_result.error_message or 'Gaussian fit unavailable.'}"
            )

    def _sync_gaussian_preview_visibility(self) -> None:
        is_visible = self.show_gaussian_fit_checkbox.isChecked()
        self.gaussian_fit_preview.setVisible(is_visible)
        if not is_visible:
            self.gaussian_fit_preview.set_fit_result(None)

    def _on_show_gaussian_fit_changed(self, state: int) -> None:
        is_visible = state == int(Qt.CheckState.Checked.value)
        message = "Gaussian fit preview shown." if is_visible else "Gaussian fit preview hidden."
        self.statusBar().showMessage(message, 3000)
        if is_visible:
            self.preview_bridge.set_loaded_image(self.controller.active_image)
            self.preview_bridge.set_roi_state(self.controller.active_roi_state)
        self._sync_gaussian_preview_visibility()
        self._update_workflow_status()
