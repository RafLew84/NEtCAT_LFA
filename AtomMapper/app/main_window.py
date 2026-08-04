"""Main window for the AtomMapper application."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDialog,
    QDockWidget,
    QFileDialog,
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
from .csv_export import describe_point_status, export_point_rows_to_csv
from .fit_settings import FitSettingsState
from .fit_settings_panel import FitSettingsPanelWidget
from .gaussian_preview import GaussianFitPreviewWidget
from .global_scatter_plot_widget import GlobalScatterPlotWidget
from .image_utils import extract_roi_patch
from .io import SUPPORTED_STM_EXTENSIONS
from .models import AtomPoint
from .plots import PlotUnit, build_row_geometry_metrics
from .position_uncertainty import position_uncertainty_method
from .preprocessing_dialog import PreprocessingDialog
from .preprocessing_state import PreprocessingMethod
from .pyqtgraph_image_view import PyQtGraphSTMViewport
from .pyqtgraph_preview_bridge import PyQtGraphPreviewBridge
from .row_disturbance_widget import RowDisturbanceWidget
from .row_geometry import RowGeometryUnit, build_row_disturbance_series, fit_row_geometry
from .row_metrics_widget import RowMetricsWidget
from .row_plot_widget import RowPlotWidget
from .session_io import build_session_from_runtime, load_session_from_file, save_session_to_file
from .session_model import SessionViewState

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

        self.load_button = QPushButton("Load STM Files...", left_panel)
        self.preprocessing_button = QPushButton("Preprocessing", left_panel)
        self.preprocessing_button.setObjectName("atommapper_preprocessing_button")
        self.fit_settings_button = QPushButton("Fit Settings", left_panel)
        self.fit_settings_button.setObjectName("atommapper_fit_settings_button")
        self.polygon_mask_button = QPushButton("Polygon Mask", left_panel)
        self.polygon_mask_button.setObjectName("atommapper_polygon_mask_button")
        self.polygon_mask_button.setCheckable(True)
        self.clear_polygon_mask_button = QPushButton("Clear Mask", left_panel)
        self.clear_polygon_mask_button.setObjectName("atommapper_clear_polygon_mask_button")
        self.export_csv_button = QPushButton("Export CSV", left_panel)
        self.export_csv_button.setObjectName("atommapper_export_csv_button")
        self.save_session_button = QPushButton("Save Session", left_panel)
        self.save_session_button.setObjectName("atommapper_save_session_button")
        self.load_session_button = QPushButton("Load Session", left_panel)
        self.load_session_button.setObjectName("atommapper_load_session_button")
        for hidden_action_button in (
            self.load_button,
            self.preprocessing_button,
            self.fit_settings_button,
            self.polygon_mask_button,
            self.clear_polygon_mask_button,
            self.export_csv_button,
            self.save_session_button,
            self.load_session_button,
        ):
            hidden_action_button.hide()

        self.load_files_action = QAction("Load STM Files...", self)
        self.load_files_action.setObjectName("atommapper_load_files_action")
        self.preprocessing_action = QAction("Preprocessing", self)
        self.preprocessing_action.setObjectName("atommapper_preprocessing_action")
        self.fit_settings_action = QAction("Fit Settings", self)
        self.fit_settings_action.setObjectName("atommapper_fit_settings_action")
        self.polygon_mask_action = QAction("Polygon Mask", self)
        self.polygon_mask_action.setObjectName("atommapper_polygon_mask_action")
        self.polygon_mask_action.setCheckable(True)
        self.clear_polygon_mask_action = QAction("Clear Mask", self)
        self.clear_polygon_mask_action.setObjectName("atommapper_clear_polygon_mask_action")
        self.export_csv_action = QAction("Export results table to CSV", self)
        self.export_csv_action.setObjectName("atommapper_export_csv_action")
        self.save_session_action = QAction("Save Session", self)
        self.save_session_action.setObjectName("atommapper_save_session_action")
        self.load_session_action = QAction("Load Session", self)
        self.load_session_action.setObjectName("atommapper_load_session_action")
        self.recalculate_position_uncertainties_action = QAction(
            "Recalculate position uncertainties",
            self,
        )
        self.recalculate_position_uncertainties_action.setObjectName(
            "atommapper_recalculate_position_uncertainties_action"
        )
        self.file_list_hint_label = QLabel("No STM files loaded. Use File > Load STM Files... to start.")
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
        self.move_point_up_button = QPushButton("Move Up")
        self.move_point_up_button.setObjectName("atommapper_move_point_up_button")
        self.move_point_down_button = QPushButton("Move Down")
        self.move_point_down_button.setObjectName("atommapper_move_point_down_button")
        row_button_panel = QWidget(left_panel)
        row_button_layout = QGridLayout(row_button_panel)
        row_button_layout.setContentsMargins(0, 0, 0, 0)
        row_button_layout.setSpacing(8)
        row_button_layout.addWidget(self.new_row_button, 0, 0)
        row_button_layout.addWidget(self.delete_row_button, 0, 1)
        row_button_layout.addWidget(self.add_point_button, 0, 2)
        row_button_layout.addWidget(self.move_point_up_button, 1, 0)
        row_button_layout.addWidget(self.move_point_down_button, 1, 1)
        row_button_layout.addWidget(self.delete_point_button, 1, 2)
        self.row_list_hint_label = QLabel("Load or select an STM image to manage rows.")
        self.row_list_hint_label.setWordWrap(True)
        self.row_list_hint_label.setStyleSheet("font-size: 12px; color: palette(mid);")
        self.row_list_widget = QListWidget()
        self.row_list_widget.setObjectName("atommapper_row_list")

        left_layout.addWidget(left_title)
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
        self.show_gaussian_fit_checkbox = QCheckBox("Show Fit Preview", right_panel)
        self.show_gaussian_fit_checkbox.setChecked(True)
        self.workflow_status_label = QLabel("Workflow status: load an STM image to begin.")
        self.workflow_status_label.setWordWrap(True)
        self.workflow_status_label.setStyleSheet("font-size: 12px; color: palette(mid);")

        self.image_viewport = PyQtGraphSTMViewport(right_panel)
        self.gaussian_fit_preview = GaussianFitPreviewWidget(right_panel)
        self.preview_bridge = PyQtGraphPreviewBridge(
            self.image_viewport,
            None,
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
        self.points_table_widget = QTableWidget(0, 12, right_panel)
        self.points_table_widget.setObjectName("atommapper_points_table")
        self.points_table_widget.setHorizontalHeaderLabels(
            [
                "row",
                "index",
                "x_px",
                "y_px",
                "sigma_x",
                "sigma_y",
                "position_std_x_px",
                "position_std_y_px",
                "position_std_x_nm",
                "position_std_y_nm",
                "uncertainty",
                "status",
            ]
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

        self.row_plot_widget = RowPlotWidget(right_panel)
        self.row_plot_widget.setObjectName("atommapper_row_plot_widget_container")
        self.global_scatter_plot_widget = GlobalScatterPlotWidget(right_panel)
        self.global_scatter_plot_widget.setObjectName("atommapper_global_scatter_widget_container")
        self.row_metrics_widget = RowMetricsWidget(right_panel)
        self.row_metrics_widget.setObjectName("atommapper_row_metrics_widget_container")
        self.row_disturbance_widget = RowDisturbanceWidget(right_panel)
        self.row_disturbance_widget.setObjectName("atommapper_row_disturbance_widget_container")

        self.analysis_grid_panel = QWidget(right_panel)
        self.analysis_grid_panel.setObjectName("atommapper_analysis_grid_panel")
        analysis_grid_layout = QGridLayout(self.analysis_grid_panel)
        analysis_grid_layout.setContentsMargins(0, 0, 0, 0)
        analysis_grid_layout.setHorizontalSpacing(12)
        analysis_grid_layout.setVerticalSpacing(12)
        analysis_grid_layout.addWidget(self.gaussian_fit_preview, 0, 0)
        analysis_grid_layout.addWidget(self.image_viewport, 0, 1)
        analysis_grid_layout.setColumnStretch(0, 2)
        analysis_grid_layout.setColumnStretch(1, 3)
        analysis_grid_layout.setRowStretch(0, 1)

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
        self.analysis_dock_content = QWidget(self)
        self.analysis_dock_content.setObjectName("atommapper_analysis_dock_content")
        analysis_dock_layout = QGridLayout(self.analysis_dock_content)
        analysis_dock_layout.setContentsMargins(12, 12, 12, 12)
        analysis_dock_layout.setHorizontalSpacing(12)
        analysis_dock_layout.setVerticalSpacing(12)
        analysis_dock_layout.addWidget(self.saved_points_panel, 0, 0)
        analysis_dock_layout.addWidget(self.row_plot_widget, 0, 1)
        analysis_dock_layout.addWidget(self.global_scatter_plot_widget, 1, 0)
        analysis_dock_layout.addWidget(self.row_metrics_widget, 1, 1)
        analysis_dock_layout.addWidget(self.row_disturbance_widget, 2, 0, 1, 2)
        analysis_dock_layout.setColumnStretch(0, 1)
        analysis_dock_layout.setColumnStretch(1, 1)
        analysis_dock_layout.setRowStretch(0, 1)
        analysis_dock_layout.setRowStretch(1, 1)
        analysis_dock_layout.setRowStretch(2, 1)
        self.analysis_dock.setWidget(self.analysis_dock_content)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.analysis_dock)
        self.fit_settings_state = FitSettingsState()
        self.fit_settings_panel = FitSettingsPanelWidget(self)
        self.fit_settings_panel.setObjectName("atommapper_fit_settings_panel_widget")
        self.fit_settings_panel.set_fit_settings_state(self.fit_settings_state)
        self.preview_bridge.set_fit_settings_state(self.fit_settings_state)
        self.fit_settings_dock = QDockWidget("Fit Settings", self)
        self.fit_settings_dock.setObjectName("atommapper_fit_settings_dock")
        self.fit_settings_dock.setAllowedAreas(
            Qt.DockWidgetArea.RightDockWidgetArea
            | Qt.DockWidgetArea.LeftDockWidgetArea
        )
        self.fit_settings_dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetClosable
            | QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        self.fit_settings_dock.setWidget(self.fit_settings_panel)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.fit_settings_dock)
        self.fit_settings_dock.hide()
        file_menu = self.menuBar().addMenu("File")
        file_menu.addAction(self.load_files_action)
        file_menu.addSeparator()
        file_menu.addAction(self.export_csv_action)
        file_menu.addSeparator()
        file_menu.addAction(self.save_session_action)
        file_menu.addAction(self.load_session_action)

        tools_menu = self.menuBar().addMenu("Tools")
        tools_menu.addAction(self.preprocessing_action)
        tools_menu.addAction(self.fit_settings_action)
        tools_menu.addSeparator()
        tools_menu.addAction(self.recalculate_position_uncertainties_action)
        tools_menu.addAction(self.export_csv_action)
        tools_menu.addSeparator()
        tools_menu.addAction(self.polygon_mask_action)
        tools_menu.addAction(self.clear_polygon_mask_action)

        view_menu = self.menuBar().addMenu("View")
        analysis_toggle_action = self.analysis_dock.toggleViewAction()
        analysis_toggle_action.setText("Analysis")
        fit_settings_toggle_action = self.fit_settings_dock.toggleViewAction()
        fit_settings_toggle_action.setText("Fit Settings Dock")
        view_menu.addAction(analysis_toggle_action)
        view_menu.addAction(fit_settings_toggle_action)
        self._preprocessing_dialog_class = PreprocessingDialog
        self._active_point_id_by_source_group: dict[str, str] = {}
        self.statusBar().showMessage("Ready. Load an STM file to begin.", 5000)
        self._connect_signals()
        self._refresh_file_list()
        self._refresh_row_list()
        self._refresh_points_table()
        self._refresh_image_point_overlay()
        self._refresh_analysis_widgets()
        self._update_active_image_label(self.controller.active_image)
        self._update_preprocess_controls(self.controller.active_image)
        self._update_export_controls(self.controller.active_image)
        self._update_position_uncertainty_controls()
        self._update_polygon_mask_controls(
            self.controller.active_image,
            self.controller.active_roi_state,
        )
        self._update_fit_settings_context()
        self._update_active_row_label(self.controller.active_row)
        self._update_row_controls(self.controller.active_image, self.controller.active_row)
        self._update_point_controls()
        self._handle_active_image_changed(self.controller.active_image)

    def _connect_signals(self) -> None:
        self.load_button.clicked.connect(self._open_file_dialog)
        self.load_files_action.triggered.connect(self._open_file_dialog)
        self.preprocessing_button.clicked.connect(self._open_preprocessing_dialog)
        self.preprocessing_action.triggered.connect(self._open_preprocessing_dialog)
        self.fit_settings_button.clicked.connect(self._open_fit_settings_dock)
        self.fit_settings_action.triggered.connect(self._open_fit_settings_dock)
        self.polygon_mask_button.toggled.connect(self._on_polygon_mask_toggled)
        self.polygon_mask_action.toggled.connect(self._on_polygon_mask_toggled)
        self.clear_polygon_mask_button.clicked.connect(self._clear_polygon_mask)
        self.clear_polygon_mask_action.triggered.connect(self._clear_polygon_mask)
        self.export_csv_button.clicked.connect(self._export_points_csv)
        self.export_csv_action.triggered.connect(self._export_points_csv)
        self.save_session_button.clicked.connect(self._save_session_to_project_file)
        self.save_session_action.triggered.connect(self._save_session_to_project_file)
        self.load_session_button.clicked.connect(self._load_session_from_project_file)
        self.load_session_action.triggered.connect(self._load_session_from_project_file)
        self.recalculate_position_uncertainties_action.triggered.connect(
            self._recalculate_position_uncertainties
        )
        self.new_row_button.clicked.connect(self._create_new_row)
        self.delete_row_button.clicked.connect(self._delete_active_row)
        self.add_point_button.clicked.connect(self._add_point_from_current_roi)
        self.move_point_up_button.clicked.connect(lambda: self._move_active_point_in_table(-1))
        self.move_point_down_button.clicked.connect(lambda: self._move_active_point_in_table(1))
        self.delete_point_button.clicked.connect(self._delete_active_point)
        self.file_list_widget.currentRowChanged.connect(self._on_current_row_changed)
        self.row_list_widget.currentRowChanged.connect(self._on_current_row_changed_for_rows)
        self.points_table_widget.itemSelectionChanged.connect(self._on_points_table_selection_changed)
        self.controller.loaded_images_changed.connect(self._refresh_file_list)
        self.controller.loaded_images_changed.connect(self._refresh_row_list)
        self.controller.active_image_changed.connect(self._update_active_image_label)
        self.controller.active_image_changed.connect(self._update_preprocess_controls)
        self.controller.active_image_changed.connect(self._update_export_controls)
        self.controller.active_image_changed.connect(lambda *_args: self._update_fit_settings_context())
        self.controller.active_image_changed.connect(self._handle_active_image_changed)
        self.controller.active_image_changed.connect(self._handle_active_image_changed_for_rows)
        self.controller.active_image_changed.connect(self._refresh_points_table)
        self.controller.active_image_changed.connect(self._refresh_image_point_overlay)
        self.controller.roi_state_changed.connect(self._handle_roi_state_changed)
        self.controller.roi_state_changed.connect(lambda *_args: self._update_fit_settings_context())
        self.controller.rows_changed.connect(self._refresh_row_list)
        self.controller.rows_changed.connect(self._handle_rows_changed)
        self.controller.rows_changed.connect(self._refresh_points_table)
        self.controller.rows_changed.connect(self._refresh_image_point_overlay)
        self.controller.rows_changed.connect(self._update_position_uncertainty_controls)
        self.controller.active_row_changed.connect(self._handle_active_row_changed)
        self.controller.row_points_changed.connect(self._handle_row_points_changed)
        self.controller.row_points_changed.connect(self._update_position_uncertainty_controls)
        self.show_gaussian_fit_checkbox.stateChanged.connect(self._on_show_gaussian_fit_changed)
        self.row_metrics_widget.unit_combo.currentIndexChanged.connect(
            self._on_row_geometry_unit_changed
        )
        self.fit_settings_panel.fit_settings_changed.connect(self._handle_fit_settings_changed)
        self.preview_bridge.roi_state_edited.connect(self.controller.update_active_roi_state)
        self.image_viewport.point_selected.connect(self._handle_viewport_point_selected)
        self.image_viewport.point_move_requested.connect(self._handle_viewport_point_move_requested)
        self.image_viewport.polygon_mask_state_changed.connect(self._handle_polygon_mask_state_changed)

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
            self.file_list_hint_label.setText("No STM files loaded. Use File > Load STM Files... to start.")
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
                self._format_optional_float(point.position_std_x_px),
                point_id=point.point_id,
            )
            self._set_points_table_item(
                row_index,
                7,
                self._format_optional_float(point.position_std_y_px),
                point_id=point.point_id,
            )
            self._set_points_table_item(
                row_index,
                8,
                self._format_optional_float(point.position_std_x_nm),
                point_id=point.point_id,
            )
            self._set_points_table_item(
                row_index,
                9,
                self._format_optional_float(point.position_std_y_nm),
                point_id=point.point_id,
            )
            self._set_points_table_item(
                row_index,
                10,
                self._format_position_uncertainty_status(point),
                point_id=point.point_id,
            )
            self._set_points_table_item(
                row_index,
                11,
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

    def _refresh_row_plot_widget(self) -> None:
        self.row_plot_widget.set_row(self.controller.active_row)

    def _refresh_global_scatter_plot_widget(self) -> None:
        active_group = self.controller.active_source_group_id
        if active_group is None:
            self.global_scatter_plot_widget.set_rows(())
            return
        rows = self.controller.rows_for_source_group(active_group)
        self.global_scatter_plot_widget.set_rows(rows)

    def _refresh_row_metrics_widget(self) -> None:
        active_row = self.controller.active_row
        if active_row is None:
            self.row_metrics_widget.set_metrics(None)
            return
        self.row_metrics_widget.set_metrics(build_row_geometry_metrics(active_row))

    def _refresh_row_geometry_overlay(self) -> None:
        active_image = self.controller.active_image
        active_row = self.controller.active_row
        if active_image is None or active_row is None:
            self.image_viewport.set_row_geometry_overlay(None)
            return

        geometry = fit_row_geometry(active_row)
        disturbance_markers: list[dict[str, object]] = []
        if geometry is not None:
            disturbance_series = build_row_disturbance_series(
                active_row,
                geometry=geometry,
                unit=RowGeometryUnit.PX,
            )
            point_lookup = {point.point_id: point for point in active_row.points}
            if disturbance_series is not None:
                for sample in disturbance_series.samples:
                    if not sample.is_candidate:
                        continue
                    point = point_lookup.get(sample.point_id)
                    if point is None:
                        continue
                    disturbance_markers.append(
                        {
                            "point_id": sample.point_id,
                            "row_id": active_row.row_id,
                            "x_px": float(point.x_px),
                            "y_px": float(point.y_px),
                            "score": float(sample.candidate_score),
                        }
                    )

        self.image_viewport.set_row_geometry_overlay(
            geometry,
            disturbance_markers=disturbance_markers,
        )

    def _refresh_row_disturbance_widget(self) -> None:
        active_row = self.controller.active_row
        if self.row_metrics_widget.current_unit is PlotUnit.NM:
            self.row_disturbance_widget.set_row(active_row, unit=PlotUnit.NM)
            return
        self.row_disturbance_widget.set_row(active_row, unit=PlotUnit.PX)

    def _refresh_analysis_widgets(self) -> None:
        self._refresh_row_plot_widget()
        self._refresh_global_scatter_plot_widget()
        self._refresh_row_metrics_widget()
        self._refresh_row_disturbance_widget()
        self._refresh_row_geometry_overlay()

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
        return describe_point_status(point)

    @staticmethod
    def _format_position_uncertainty_status(point: AtomPoint) -> str:
        status = str(point.metadata.get("position_uncertainty_status") or "")
        if not status:
            return ""
        qualifiers: list[str] = []
        if point.metadata.get("position_uncertainty_original_mask_missing"):
            qualifiers.append("original mask unavailable")
        if point.metadata.get("position_uncertainty_settings_source") == "session_fallback":
            qualifiers.append("session settings")
        if not qualifiers:
            return status
        return f"{status} ({', '.join(qualifiers)})"

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

    def _active_point_context(self) -> tuple[Any, AtomPoint, int] | None:
        """Return the active point together with its row and current order position."""

        active_point = self._active_point_for_current_group()
        if active_point is None:
            return None

        active_group = self.controller.active_source_group_id
        if active_group is None:
            return None

        row = next(
            (
                candidate_row
                for candidate_row in self.controller.rows_for_source_group(active_group)
                if candidate_row.row_id == active_point.row_id
            ),
            None,
        )
        if row is None:
            return None

        ordered_points = sorted(
            row.points,
            key=lambda point: (point.point_index, point.point_id),
        )
        point_position = next(
            (index for index, point in enumerate(ordered_points) if point.point_id == active_point.point_id),
            None,
        )
        if point_position is None:
            return None
        return (row, active_point, point_position)

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
        self._update_row_controls(self.controller.active_image, self.controller.active_row)

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
        self._update_row_controls(self.controller.active_image, self.controller.active_row)

    def _update_active_image_label(self, active_image: Any) -> None:
        if active_image is None:
            self.active_image_label.setText("Active image: none")
            self.active_image_label.setToolTip("No active STM image selected.")
            return
        calibration_summary = active_image.calibration_summary
        if calibration_summary is None:
            label_text = (
                f"Active image: {active_image.display_name} "
                f"({active_image.pixels_x}x{active_image.pixels_y} px | uncalibrated)"
            )
            tooltip_text = (
                f"{active_image.display_name}\n"
                f"Pixels: {active_image.pixels_x} x {active_image.pixels_y}\n"
                "Physical calibration unavailable for this image."
            )
        else:
            label_text = (
                f"Active image: {active_image.display_name} "
                f"({active_image.pixels_x}x{active_image.pixels_y} px | {active_image.size_nm_x:.3f} x {active_image.size_nm_y:.3f} nm)"
            )
            tooltip_text = (
                f"{active_image.display_name}\n"
                f"Pixels: {active_image.pixels_x} x {active_image.pixels_y}\n"
                f"Calibration: {calibration_summary}"
            )
        self.active_image_label.setText(label_text)
        self.active_image_label.setToolTip(tooltip_text)

    def _update_preprocess_controls(self, active_image: Any) -> None:
        has_image = active_image is not None
        self.preprocessing_button.setEnabled(has_image)
        self.preprocessing_action.setEnabled(has_image)
        if has_image:
            tooltip = f"Open preprocessing dialog for {active_image.display_name}."
        else:
            tooltip = "Load or select an STM image first."
        self.preprocessing_button.setToolTip(tooltip)
        self.preprocessing_action.setStatusTip(tooltip)

    def _update_export_controls(self, active_image: Any) -> None:
        has_image = active_image is not None
        self.export_csv_button.setEnabled(has_image)
        self.export_csv_action.setEnabled(has_image)
        if has_image:
            export_tooltip = (
                f"Export saved points for the active STM file family of {active_image.display_name}."
            )
        else:
            export_tooltip = "Load or select an STM image first."
        self.export_csv_button.setToolTip(export_tooltip)
        self.export_csv_action.setStatusTip(export_tooltip)
        self.save_session_button.setEnabled(True)
        self.save_session_action.setEnabled(True)
        save_session_tooltip = "Save the current AtomMapper project state to a .atommapper_proj file."
        self.save_session_button.setToolTip(save_session_tooltip)
        self.save_session_action.setStatusTip(save_session_tooltip)
        self.load_session_button.setEnabled(True)
        self.load_session_action.setEnabled(True)
        load_session_tooltip = "Load an AtomMapper project state from a .atommapper_proj file."
        self.load_session_button.setToolTip(load_session_tooltip)
        self.load_session_action.setStatusTip(load_session_tooltip)
        self.fit_settings_button.setEnabled(True)
        self.fit_settings_action.setEnabled(True)
        fit_settings_tooltip = (
            "Open the non-modal fit-settings dock to edit the local peak model and its parameters."
        )
        self.fit_settings_button.setToolTip(fit_settings_tooltip)
        self.fit_settings_action.setStatusTip(fit_settings_tooltip)

    def _update_position_uncertainty_controls(self, *_args: Any) -> None:
        point_count = sum(row.point_count for row in self.controller.atom_rows)
        has_points = point_count > 0
        self.recalculate_position_uncertainties_action.setEnabled(has_points)
        if has_points:
            tooltip = (
                "Re-fit saved point ROIs and calculate localization uncertainty for "
                f"{point_count} saved point(s)."
            )
        else:
            tooltip = "Add or load saved atom points before recalculating uncertainties."
        self.recalculate_position_uncertainties_action.setStatusTip(tooltip)

    def _recalculate_position_uncertainties(self) -> None:
        summary = self.controller.recalculate_position_uncertainties(self.fit_settings_state)
        point_noun = "point" if summary.total_points == 1 else "points"
        if summary.recomputed_points == summary.total_points:
            message = (
                f"Recalculated position uncertainties for {summary.total_points} {point_noun}."
            )
        else:
            message = (
                "Recalculated position uncertainties for "
                f"{summary.recomputed_points} of {summary.total_points} {point_noun}."
            )
        if summary.recomputed_without_original_mask:
            message += (
                f" {summary.recomputed_without_original_mask} result(s) were recalculated "
                "without the original mask."
            )
        if summary.failed_points:
            message += f" {summary.failed_points} point(s) could not be recalculated."
        self.statusBar().showMessage(message, 10000)
        self.workflow_status_label.setText(f"Workflow status: {message}")

    def _update_polygon_mask_controls(self, active_image: Any, roi_state: Any) -> None:
        has_image = active_image is not None
        has_roi = roi_state is not None
        has_mask = self.image_viewport.has_polygon_mask_or_draft
        can_draw = has_image and has_roi

        self.polygon_mask_button.setEnabled(can_draw)
        self.polygon_mask_action.setEnabled(can_draw)
        self.clear_polygon_mask_button.setEnabled(has_mask or self.polygon_mask_button.isChecked())
        self.clear_polygon_mask_action.setEnabled(has_mask or self.polygon_mask_button.isChecked())

        if can_draw:
            polygon_tooltip = (
                "Draw a polygon inside the active ROI. Click to add vertices, double click to close."
            )
        elif has_image:
            polygon_tooltip = "Define or select an ROI before drawing a polygon mask."
        else:
            polygon_tooltip = "Load or select an STM image first."
        self.polygon_mask_button.setToolTip(polygon_tooltip)
        self.polygon_mask_action.setStatusTip(polygon_tooltip)

        if has_mask:
            clear_mask_tooltip = "Clear the active polygon mask used for local fitting."
        else:
            clear_mask_tooltip = "No polygon mask is currently active."
        self.clear_polygon_mask_button.setToolTip(clear_mask_tooltip)
        self.clear_polygon_mask_action.setStatusTip(clear_mask_tooltip)

    def _set_polygon_mask_toggle_checked(self, checked: bool) -> None:
        self.polygon_mask_button.blockSignals(True)
        self.polygon_mask_button.setChecked(checked)
        self.polygon_mask_button.blockSignals(False)
        self.polygon_mask_action.blockSignals(True)
        self.polygon_mask_action.setChecked(checked)
        self.polygon_mask_action.blockSignals(False)

    def _update_fit_settings_context(self) -> None:
        self.fit_settings_panel.set_context(
            self.controller.active_image,
            self.controller.active_roi_state,
        )

    def _open_fit_settings_dock(self) -> None:
        self.fit_settings_dock.show()
        self.fit_settings_dock.raise_()
        self.statusBar().showMessage(
            "Fit Settings dock opened. It stays non-modal while you work on the image.",
            3000,
        )

    def _handle_fit_settings_changed(self, state: FitSettingsState) -> None:
        self.fit_settings_state = state.normalized()
        self.preview_bridge.set_fit_settings_state(self.fit_settings_state)
        self._update_workflow_status()

    def _on_polygon_mask_toggled(self, checked: bool) -> None:
        self._set_polygon_mask_toggle_checked(checked)
        if checked:
            if self.controller.active_image is None or self.controller.active_roi_state is None:
                self._set_polygon_mask_toggle_checked(False)
                self.statusBar().showMessage("Select an STM image and ROI before drawing a polygon mask.", 4000)
                self.workflow_status_label.setText(
                    "Workflow status: select an STM image and ROI before drawing a polygon mask."
                )
                self._update_polygon_mask_controls(
                    self.controller.active_image,
                    self.controller.active_roi_state,
                )
                return

            self.preview_bridge.set_polygon_mask_state(None)
            self.image_viewport.clear_polygon_mask(emit_signal=False)
            self.image_viewport.set_polygon_mask_drawing_enabled(True)
            self.statusBar().showMessage(
                "Polygon mask mode enabled. Click inside ROI to add vertices; double click to close.",
                5000,
            )
            self.workflow_status_label.setText(
                "Workflow status: polygon mask mode active. Click inside ROI to add vertices; double click to close."
            )
        else:
            self.image_viewport.set_polygon_mask_drawing_enabled(False)
            if self.image_viewport.current_polygon_mask_state is None:
                self.statusBar().showMessage("Polygon mask drawing cancelled.", 3000)

        self._update_polygon_mask_controls(
            self.controller.active_image,
            self.controller.active_roi_state,
        )

    def _handle_polygon_mask_state_changed(self, state: Any) -> None:
        self.preview_bridge.set_polygon_mask_state(state)
        if state is not None:
            self._set_polygon_mask_toggle_checked(False)
            self.image_viewport.set_polygon_mask_drawing_enabled(False)
            self.statusBar().showMessage("Polygon mask applied to the current ROI fit.", 4000)
        else:
            self.statusBar().showMessage("Polygon mask cleared.", 3000)

        self._update_polygon_mask_controls(
            self.controller.active_image,
            self.controller.active_roi_state,
        )
        self._update_workflow_status()

    def _clear_polygon_mask(self) -> None:
        self._set_polygon_mask_toggle_checked(False)
        self.image_viewport.set_polygon_mask_drawing_enabled(False)
        self.image_viewport.clear_polygon_mask()
        self._update_polygon_mask_controls(
            self.controller.active_image,
            self.controller.active_roi_state,
        )

    def _update_active_row_label(self, active_row: Any) -> None:
        if active_row is None:
            self.active_row_label.setText("Active row: none")
            self.active_row_label.setToolTip("No active atom row selected.")
            return
        noun = "point" if active_row.point_count == 1 else "points"
        geometry = fit_row_geometry(active_row)
        disturbance_series = None
        if geometry is not None:
            disturbance_series = build_row_disturbance_series(
                active_row,
                geometry=geometry,
                unit=RowGeometryUnit.PX,
            )

        if geometry is None:
            summary = "geometry pending"
            tooltip = (
                f"{active_row.display_name}\n"
                f"Points: {active_row.point_count}\n"
                "Need at least 2 points to fit a stable row axis."
            )
        else:
            axis_angle_deg = math.degrees(
                math.atan2(float(geometry.direction_y_px), float(geometry.direction_x_px))
            )
            if disturbance_series is None:
                summary = "axis fitted"
                tooltip = (
                    f"{active_row.display_name}\n"
                    f"Points: {active_row.point_count}\n"
                    f"Axis angle: {axis_angle_deg:.2f} deg\n"
                    "Need at least 3 points to inspect local disturbances."
                )
            else:
                candidate_noun = (
                    "candidate" if disturbance_series.candidate_count == 1 else "candidates"
                )
                summary = (
                    "no local candidates"
                    if disturbance_series.candidate_count == 0
                    else f"{disturbance_series.candidate_count} {candidate_noun}"
                )
                tooltip = (
                    f"{active_row.display_name}\n"
                    f"Points: {active_row.point_count}\n"
                    f"Axis angle: {axis_angle_deg:.2f} deg\n"
                    f"Local disturbance candidates: {disturbance_series.candidate_count}\n"
                    f"Interior samples: {len(disturbance_series.samples)}"
                )

        self.active_row_label.setText(
            f"Active row: {active_row.display_name} ({active_row.point_count} {noun} | {summary})"
        )
        self.active_row_label.setToolTip(tooltip)

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
            active_point_context = self._active_point_context()
            if active_point_context is not None and active_point_context[0].row_id == active_row.row_id:
                _, selected_point, _ = active_point_context
                self.add_point_button.setToolTip(
                    f"Store the current ROI point in '{active_row.display_name}' after selected point {selected_point.point_index}."
                )
            else:
                self.add_point_button.setToolTip(
                    f"Store the current ROI point at the end of '{active_row.display_name}'."
                )
        else:
            self.delete_row_button.setToolTip("Select an atom row first.")
            self.add_point_button.setToolTip("Create or select an atom row first.")

    def _update_point_controls(self) -> None:
        active_point_context = self._active_point_context()
        has_point = active_point_context is not None
        self.delete_point_button.setEnabled(has_point)
        self.move_point_up_button.setEnabled(False)
        self.move_point_down_button.setEnabled(False)
        if has_point:
            row, active_point, point_position = active_point_context
            self.delete_point_button.setToolTip(
                f"Delete selected point {active_point.point_index} from the current STM file family."
            )
            can_move_up = point_position > 0
            can_move_down = point_position < row.point_count - 1
            self.move_point_up_button.setEnabled(can_move_up)
            self.move_point_down_button.setEnabled(can_move_down)
            if can_move_up:
                self.move_point_up_button.setToolTip(
                    f"Move selected point {active_point.point_index} one position earlier in {row.display_name}."
                )
            else:
                self.move_point_up_button.setToolTip("Selected point is already first in its row.")
            if can_move_down:
                self.move_point_down_button.setToolTip(
                    f"Move selected point {active_point.point_index} one position later in {row.display_name}."
                )
            else:
                self.move_point_down_button.setToolTip("Selected point is already last in its row.")
        else:
            self.delete_point_button.setToolTip("Select a saved point in the table or on the image first.")
            self.move_point_up_button.setToolTip("Select a saved point in the table or on the image first.")
            self.move_point_down_button.setToolTip("Select a saved point in the table or on the image first.")

    def _on_row_geometry_unit_changed(self, _index: int) -> None:
        self._refresh_row_disturbance_widget()

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
        self.analysis_dock.setUpdatesEnabled(False)
        self.analysis_dock_content.setUpdatesEnabled(False)
        try:
            result = dialog.exec()
        finally:
            self.analysis_dock_content.setUpdatesEnabled(True)
            self.analysis_dock.setUpdatesEnabled(True)
            self.image_viewport.setUpdatesEnabled(True)
            self.analysis_grid_panel.setUpdatesEnabled(True)
            self.image_viewport.update()
            self.analysis_grid_panel.update()
            self.analysis_dock_content.update()
            self.analysis_dock.update()
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
            elif state.method is PreprocessingMethod.ROTATE:
                variant = self.controller.create_rotate_variant_for_active_image(
                    quarter_turns=state.rotate.quarter_turns,
                    make_active=True,
                )
                status_suffix = f"angle {state.rotate.angle_deg}° CCW"
            elif state.method is PreprocessingMethod.FLIP:
                variant = self.controller.create_flip_variant_for_active_image(
                    flip_x=state.flip.flip_x,
                    flip_y=state.flip.flip_y,
                    make_active=True,
                )
                axes = []
                if state.flip.flip_x:
                    axes.append("X")
                if state.flip.flip_y:
                    axes.append("Y")
                status_suffix = f"axes {'+'.join(axes)}"
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

    def _export_points_csv(self) -> None:
        active_image = self.controller.active_image
        active_group = self.controller.active_source_group_id
        if active_image is None or active_group is None:
            self.statusBar().showMessage("Select an STM image before exporting CSV.", 4000)
            self.workflow_status_label.setText(
                "Workflow status: select an STM image before exporting saved points."
            )
            return

        default_name = f"{Path(active_image.source_path).stem}_points.csv"
        export_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Points to CSV",
            default_name,
            "CSV files (*.csv)",
        )
        if not export_path:
            self.statusBar().showMessage("CSV export cancelled.", 3000)
            self.workflow_status_label.setText(
                "Workflow status: CSV export dialog closed without saving."
            )
            return

        rows = self.controller.rows_for_source_group(active_group)
        try:
            exported_count = export_point_rows_to_csv(
                export_path,
                rows,
                self.controller.loaded_images,
            )
        except Exception as exc:  # pragma: no cover - GUI error path
            logger.exception("Failed to export CSV '%s': %s", export_path, exc)
            QMessageBox.warning(
                self,
                "AtomMapper - CSV Export Error",
                f"Could not export CSV:\n\n{exc}",
            )
            self.statusBar().showMessage("CSV export failed.", 5000)
            self.workflow_status_label.setText("Workflow status: CSV export failed.")
            return

        noun = "point" if exported_count == 1 else "points"
        export_name = Path(export_path).name
        self.statusBar().showMessage(
            f"Exported {exported_count} {noun} to {export_name}.",
            5000,
        )
        self.workflow_status_label.setText(
            f"Workflow status: exported {exported_count} {noun} to CSV '{export_name}'."
        )

    def _build_session_view_state(self) -> SessionViewState:
        """Build the serializable UI/view state captured in a saved project."""

        return SessionViewState(
            show_gaussian_fit=self.show_gaussian_fit_checkbox.isChecked(),
            row_plot_mode=self.row_plot_widget.current_mode,
            row_plot_unit=self.row_plot_widget.current_unit,
            row_metrics_unit=self.row_metrics_widget.current_unit,
            global_scatter_unit=self.global_scatter_plot_widget.current_unit,
            active_polygon_mask=self.image_viewport.current_polygon_mask_state,
        )

    def _save_session_to_project_file(self) -> None:
        active_image = self.controller.active_image
        default_name = "atommapper_session.atommapper_proj"
        if active_image is not None:
            default_name = f"{Path(active_image.source_path).stem}.atommapper_proj"

        session_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save AtomMapper Session",
            default_name,
            "AtomMapper project (*.atommapper_proj);;JSON files (*.json)",
        )
        if not session_path:
            self.statusBar().showMessage("Session save cancelled.", 3000)
            self.workflow_status_label.setText(
                "Workflow status: session-save dialog closed without writing a project file."
            )
            return

        try:
            session = build_session_from_runtime(
                self.controller,
                active_point_id_by_source_group=self._active_point_id_by_source_group,
                fit_settings=self.fit_settings_state,
                view_state=self._build_session_view_state(),
            )
            saved_path = save_session_to_file(session_path, session)
        except Exception as exc:  # pragma: no cover - GUI error path
            logger.exception("Failed to save session '%s': %s", session_path, exc)
            QMessageBox.warning(
                self,
                "AtomMapper - Save Session Error",
                f"Could not save the project session:\n\n{exc}",
            )
            self.statusBar().showMessage("Session save failed.", 5000)
            self.workflow_status_label.setText(
                "Workflow status: saving the project session failed."
            )
            return

        saved_name = Path(saved_path).name
        self.statusBar().showMessage(f"Saved session to {saved_name}.", 5000)
        self.workflow_status_label.setText(
            f"Workflow status: saved project session to '{saved_name}'."
        )

    def _apply_session_view_state(self, view_state: SessionViewState) -> None:
        """Restore persisted GUI state after a session load."""

        self.show_gaussian_fit_checkbox.blockSignals(True)
        self.show_gaussian_fit_checkbox.setChecked(view_state.show_gaussian_fit)
        self.show_gaussian_fit_checkbox.blockSignals(False)
        self._sync_gaussian_preview_visibility()

        unit_index = self.row_plot_widget.unit_combo.findData(view_state.row_plot_unit)
        if unit_index >= 0:
            self.row_plot_widget.unit_combo.setCurrentIndex(unit_index)

        metric_index = self.row_plot_widget.metric_combo.findData(
            self.row_plot_widget._metric_base_mode(view_state.row_plot_mode)
        )
        if metric_index >= 0:
            self.row_plot_widget.metric_combo.setCurrentIndex(metric_index)

        metrics_unit_index = self.row_metrics_widget.unit_combo.findData(
            view_state.row_metrics_unit
        )
        if metrics_unit_index >= 0:
            self.row_metrics_widget.unit_combo.setCurrentIndex(metrics_unit_index)

        global_scatter_unit_index = self.global_scatter_plot_widget.unit_combo.findData(
            view_state.global_scatter_unit
        )
        if global_scatter_unit_index >= 0:
            self.global_scatter_plot_widget.unit_combo.setCurrentIndex(global_scatter_unit_index)

        self.preview_bridge.set_polygon_mask_state(view_state.active_polygon_mask)
        self._set_polygon_mask_toggle_checked(False)
        self.image_viewport.set_polygon_mask_drawing_enabled(False)
        self._update_polygon_mask_controls(self.controller.active_image, self.controller.active_roi_state)

        self._refresh_analysis_widgets()
        self._update_workflow_status()

    def _apply_fit_settings_state(self, fit_settings_state: FitSettingsState) -> None:
        """Restore persisted fit-settings state after a session load."""

        self.fit_settings_state = fit_settings_state.normalized()
        self.fit_settings_panel.set_fit_settings_state(self.fit_settings_state)
        self.preview_bridge.set_fit_settings_state(self.fit_settings_state)

    def _load_session_from_project_file(self) -> None:
        session_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load AtomMapper Session",
            "",
            "AtomMapper project (*.atommapper_proj);;JSON files (*.json)",
        )
        if not session_path:
            self.statusBar().showMessage("Session load cancelled.", 3000)
            self.workflow_status_label.setText(
                "Workflow status: session-load dialog closed without opening a project file."
            )
            return

        try:
            session = load_session_from_file(session_path)
            self._active_point_id_by_source_group = {}
            self.controller.restore_from_session(session)
            self._active_point_id_by_source_group = dict(
                session.active_point_id_by_source_group
            )
            self._apply_fit_settings_state(session.fit_settings)
            self._apply_session_view_state(session.view_state)
            self._refresh_file_list()
            self._refresh_row_list()
            self._refresh_points_table()
            self._refresh_image_point_overlay()
            self._refresh_analysis_widgets()
            self._update_fit_settings_context()
            self._update_point_controls()
        except Exception as exc:  # pragma: no cover - GUI error path
            logger.exception("Failed to load session '%s': %s", session_path, exc)
            QMessageBox.warning(
                self,
                "AtomMapper - Load Session Error",
                f"Could not load the project session:\n\n{exc}",
            )
            self.statusBar().showMessage("Session load failed.", 5000)
            self.workflow_status_label.setText(
                "Workflow status: loading the project session failed."
            )
            return

        loaded_name = Path(session_path).name
        self.statusBar().showMessage(f"Loaded session from {loaded_name}.", 5000)
        self.workflow_status_label.setText(
            f"Workflow status: loaded project session from '{loaded_name}'."
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
        position_std_x_px = None
        position_std_y_px = None
        position_std_x_nm = None
        position_std_y_nm = None
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
                position_std_y_px = float(fit_result.center_std_yx[0])
                position_std_x_px = float(fit_result.center_std_yx[1])
                calibration = active_image.physical_calibration
                if calibration is not None:
                    position_std_x_nm = (
                        position_std_x_px * calibration.pixel_size_nm_x
                    )
                    position_std_y_nm = (
                        position_std_y_px * calibration.pixel_size_nm_y
                    )
        else:
            fallback_used = True
            x_px = roi.x + (roi.width / 2.0)
            y_px = roi.y + (roi.height / 2.0)
            fit_error_message = (
                f"{self.fit_settings_state.model.value.capitalize()} fit unavailable; ROI center fallback used."
            )

        if fit_result is not None and fit_result.center_image_yx is None:
            fallback_used = True
            x_px = roi.x + (roi.width / 2.0)
            y_px = roi.y + (roi.height / 2.0)
            fit_success = False
            fit_method = f"{fit_result.method}_fallback"
            fit_error_message = (
                fit_result.error_message
                or f"{fit_result.model.value.capitalize()} fit unavailable; ROI center fallback used."
            )

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
            position_std_x_px=position_std_x_px,
            position_std_y_px=position_std_y_px,
            position_std_x_nm=position_std_x_nm,
            position_std_y_nm=position_std_y_nm,
            theta_deg=theta_deg,
            offset=offset,
            fit_success=fit_success,
            fit_error_message=fit_error_message,
            metadata={
                "fit_model": self.fit_settings_state.model.value,
                "fit_method": fit_method,
                "roi_x": roi.x,
                "roi_y": roi.y,
                "roi_width": roi.width,
                "roi_height": roi.height,
                "fallback_used": fallback_used,
                "fit_mask_active": bool(fit_result is not None and fit_result.fit_mask is not None),
                "fit_mask_pixel_count": (
                    None
                    if fit_result is None or fit_result.fit_mask is None
                    else int(fit_result.fit_mask.sum())
                ),
                "fit_settings": self.fit_settings_state.to_dict(),
                "position_uncertainty_status": (
                    "computed" if position_std_x_px is not None else None
                ),
                "position_uncertainty_method": (
                    None
                    if fit_result is None or fit_result.center_std_yx is None
                    else position_uncertainty_method(fit_result.raw_result)
                ),
                "position_uncertainty_reference": (
                    "saved_position" if position_std_x_px is not None else None
                ),
                "position_uncertainty_settings_source": (
                    "point_snapshot" if position_std_x_px is not None else None
                ),
                "position_uncertainty_original_mask_missing": False,
                "fit_shape_parameters": {}
                if fit_result is None
                else {
                    key: value
                    for key, value in fit_result.shape_parameters.items()
                    if value is not None
                },
            },
        )
        insert_index = None
        inserted_after_point = None
        active_point_context = self._active_point_context()
        if active_point_context is not None:
            selected_row, selected_point, selected_position = active_point_context
            if selected_row.row_id == active_row.row_id:
                insert_index = selected_position + 1
                inserted_after_point = selected_point

        updated_row = self.controller.add_point_to_row(point, insert_index=insert_index)
        stored_point = next(
            (candidate_point for candidate_point in updated_row.points if candidate_point.point_id == point.point_id),
            point,
        )
        self.statusBar().showMessage(
            (
                f"Inserted point {stored_point.point_index} into {updated_row.display_name} at "
                f"x={stored_point.x_px:.2f}, y={stored_point.y_px:.2f}."
                if inserted_after_point is not None
                else f"Added point {stored_point.point_index} to {updated_row.display_name} at "
                f"x={stored_point.x_px:.2f}, y={stored_point.y_px:.2f}."
            ),
            4000,
        )
        if fallback_used:
            self.workflow_status_label.setText(
                (
                    f"Workflow status: inserted point {stored_point.point_index} into {updated_row.display_name} "
                    f"after point {inserted_after_point.point_index} using ROI center fallback."
                    if inserted_after_point is not None
                    else f"Workflow status: added point {stored_point.point_index} to {updated_row.display_name} "
                    "using ROI center fallback."
                )
            )
        else:
            self.workflow_status_label.setText(
                (
                    f"Workflow status: inserted point {stored_point.point_index} into {updated_row.display_name} "
                    f"after point {inserted_after_point.point_index} from {self.fit_settings_state.model.value.capitalize()} fit."
                    if inserted_after_point is not None
                    else f"Workflow status: added point {stored_point.point_index} to {updated_row.display_name} "
                    f"from {self.fit_settings_state.model.value.capitalize()} fit."
                )
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

    def _move_active_point_in_table(self, step: int) -> None:
        active_point_context = self._active_point_context()
        if active_point_context is None:
            self.statusBar().showMessage("Select a saved point before changing its position.", 4000)
            self.workflow_status_label.setText(
                "Workflow status: no saved point selected for row reordering."
            )
            return

        row, active_point, point_position = active_point_context
        target_index = point_position + int(step)
        if target_index < 0 or target_index >= row.point_count:
            direction_label = "first" if target_index < 0 else "last"
            self.statusBar().showMessage(
                f"Selected point is already {direction_label} in {row.display_name}.",
                3000,
            )
            self.workflow_status_label.setText(
                f"Workflow status: selected point is already {direction_label} in {row.display_name}."
            )
            return

        try:
            updated_row = self.controller.reorder_point_in_row(
                row_id=row.row_id,
                point_id=active_point.point_id,
                target_index=target_index,
            )
        except Exception as exc:  # pragma: no cover - GUI error path
            logger.exception("Failed to reorder point '%s': %s", active_point.point_id, exc)
            self.statusBar().showMessage("Point reorder failed.", 4000)
            self.workflow_status_label.setText(
                "Workflow status: point reorder failed."
            )
            return

        moved_point = next(
            (point for point in updated_row.points if point.point_id == active_point.point_id),
            None,
        )
        self._set_active_point_for_current_group(active_point.point_id)
        if moved_point is None:
            return

        direction_label = "up" if step < 0 else "down"
        self.statusBar().showMessage(
            f"Moved point {moved_point.point_index} {direction_label} in {updated_row.display_name}.",
            4000,
        )
        self.workflow_status_label.setText(
            f"Workflow status: moved point to position {moved_point.point_index} in {updated_row.display_name}; table and plots refreshed."
        )

    def _handle_active_image_changed_for_rows(self, active_image: Any) -> None:
        self._refresh_row_list()
        self._update_row_controls(active_image, self.controller.active_row)
        self._refresh_analysis_widgets()

    def _handle_rows_changed(self) -> None:
        self._refresh_row_list()
        self._update_row_controls(self.controller.active_image, self.controller.active_row)
        self._refresh_analysis_widgets()

    def _handle_active_row_changed(self, active_row: Any) -> None:
        self._update_active_row_label(active_row)
        self._refresh_row_list()
        self._refresh_points_table()
        self._refresh_image_point_overlay()
        self._refresh_analysis_widgets()
        self._update_row_controls(self.controller.active_image, active_row)
        self._update_point_controls()

    def _handle_row_points_changed(self, _updated_row: Any) -> None:
        self._refresh_row_list()
        self._refresh_points_table()
        self._refresh_image_point_overlay()
        self._refresh_analysis_widgets()
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
        self.preview_bridge.set_polygon_mask_state(None)
        self.preview_bridge.set_roi_state(self.controller.active_roi_state)
        self._set_polygon_mask_toggle_checked(False)
        self.image_viewport.set_polygon_mask_drawing_enabled(False)
        self._update_polygon_mask_controls(active_image, self.controller.active_roi_state)
        self._sync_gaussian_preview_visibility()
        self._refresh_analysis_widgets()
        self._update_workflow_status()

    def _handle_roi_state_changed(self, roi_state: Any) -> None:
        self.preview_bridge.set_roi_state(roi_state)
        self._update_polygon_mask_controls(self.controller.active_image, roi_state)
        self._sync_gaussian_preview_visibility()
        self._update_workflow_status()

    def _update_workflow_status(self) -> None:
        image = self.controller.active_image
        roi = self.controller.active_roi_state
        current_model_label = self.fit_settings_state.model.value.capitalize()

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
                f"Workflow status: image view active for {image.display_name}. Fit preview hidden."
            )
            return

        patch = self.preview_bridge.current_roi_patch_data
        if patch is None and image is not None and roi is not None:
            patch = extract_roi_patch(image.image_data, roi)
        if patch is None:
            self.workflow_status_label.setText(
                f"Workflow status: ROI for {image.display_name} is outside image bounds."
            )
            return

        fit_result = self.gaussian_fit_preview.current_fit_result
        mask_suffix = ""
        if fit_result is not None and fit_result.fit_mask is not None:
            mask_suffix = f" Polygon mask active ({int(fit_result.fit_mask.sum())} px)."
        if fit_result is None:
            self.workflow_status_label.setText(
                "Workflow status: "
                f"ROI {patch.shape[1]}x{patch.shape[0]} px ready. "
                f"{current_model_label} fit preview is waiting for refresh.{mask_suffix}"
            )
            return

        if fit_result.success and fit_result.center_patch_yx is not None:
            self.workflow_status_label.setText(
                "Workflow status: "
                f"ROI {patch.shape[1]}x{patch.shape[0]} px ready. "
                f"{fit_result.model.value.capitalize()} center y={fit_result.center_patch_yx[0]:.2f} "
                f"x={fit_result.center_patch_yx[1]:.2f}.{mask_suffix}"
            )
        else:
            self.workflow_status_label.setText(
                "Workflow status: "
                f"ROI {patch.shape[1]}x{patch.shape[0]} px ready. "
                f"{fit_result.error_message or f'{fit_result.model.value.capitalize()} fit unavailable.'}"
                f"{mask_suffix}"
            )

    def _sync_gaussian_preview_visibility(self) -> None:
        is_visible = self.show_gaussian_fit_checkbox.isChecked()
        self.gaussian_fit_preview.setVisible(is_visible)
        if not is_visible:
            self.gaussian_fit_preview.set_fit_result(None)

    def _on_show_gaussian_fit_changed(self, state: int) -> None:
        is_visible = state == int(Qt.CheckState.Checked.value)
        message = "Fit preview shown." if is_visible else "Fit preview hidden."
        self.statusBar().showMessage(message, 3000)
        if is_visible:
            self.preview_bridge.set_loaded_image(self.controller.active_image)
            self.preview_bridge.set_roi_state(self.controller.active_roi_state)
        self._sync_gaussian_preview_visibility()
        self._update_workflow_status()
