"""Main window for the AtomMapper application."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .controller import AtomMapperController
from .gaussian_fit import fit_gaussian_to_roi_patch
from .gaussian_preview import GaussianFitPreviewWidget
from .image_view import STMImageViewport
from .image_utils import extract_roi_patch
from .io import SUPPORTED_STM_EXTENSIONS
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
        self.file_list_hint_label = QLabel("No STM files loaded. Use 'Load STM Files...' to start.")
        self.file_list_hint_label.setWordWrap(True)
        self.file_list_hint_label.setStyleSheet("font-size: 12px; color: palette(mid);")
        self.file_list_widget = QListWidget()
        self.file_list_widget.setObjectName("atommapper_file_list")

        left_layout.addWidget(left_title)
        left_layout.addWidget(self.load_button)
        left_layout.addWidget(self.file_list_hint_label)
        left_layout.addWidget(self.file_list_widget, 1)
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

        self.image_viewport = STMImageViewport(right_panel)
        self.roi_preview = ROIPreviewWidget(right_panel)
        self.gaussian_fit_preview = GaussianFitPreviewWidget(right_panel)
        previews_panel = QWidget(right_panel)
        previews_layout = QHBoxLayout(previews_panel)
        previews_layout.setContentsMargins(0, 0, 0, 0)
        previews_layout.setSpacing(12)
        previews_layout.addWidget(self.roi_preview, 1)
        previews_layout.addWidget(self.gaussian_fit_preview, 1)

        right_layout.addWidget(title)
        right_layout.addWidget(subtitle)
        right_layout.addWidget(self.active_image_label)
        right_layout.addWidget(self.show_gaussian_fit_checkbox)
        right_layout.addWidget(self.workflow_status_label)
        right_layout.addWidget(self.image_viewport, 1)
        right_layout.addWidget(previews_panel)

        root_layout.addWidget(left_panel)
        root_layout.addWidget(right_panel, 1)

        self.setCentralWidget(central)
        self.statusBar().showMessage("Ready. Load an STM file to begin.", 5000)
        self._connect_signals()
        self._refresh_file_list()
        self._update_active_image_label(self.controller.active_image)
        self.image_viewport.set_loaded_image(self.controller.active_image)
        self.image_viewport.set_roi_state(self.controller.active_roi_state)
        self.roi_preview.set_loaded_image(self.controller.active_image)
        self.roi_preview.set_roi_state(self.controller.active_roi_state)
        self._update_gaussian_fit_preview()

    def _connect_signals(self) -> None:
        self.load_button.clicked.connect(self._open_file_dialog)
        self.file_list_widget.currentRowChanged.connect(self._on_current_row_changed)
        self.controller.loaded_images_changed.connect(self._refresh_file_list)
        self.controller.active_image_changed.connect(self._update_active_image_label)
        self.controller.active_image_changed.connect(self.image_viewport.set_loaded_image)
        self.controller.active_image_changed.connect(self.roi_preview.set_loaded_image)
        self.controller.active_image_changed.connect(self._update_gaussian_fit_preview)
        self.controller.roi_state_changed.connect(self.image_viewport.set_roi_state)
        self.controller.roi_state_changed.connect(self.roi_preview.set_roi_state)
        self.controller.roi_state_changed.connect(self._update_gaussian_fit_preview)
        self.show_gaussian_fit_checkbox.stateChanged.connect(self._on_show_gaussian_fit_changed)
        self.image_viewport.roi_state_edited.connect(self.controller.update_active_roi_state)

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

        for image in self.controller.loaded_images:
            item = QListWidgetItem(image.display_name)
            item.setToolTip(image.source_path)
            self.file_list_widget.addItem(item)

        active_index = self.controller.active_image_index
        if active_index is not None and 0 <= active_index < self.file_list_widget.count():
            self.file_list_widget.setCurrentRow(active_index)

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
        image = self.controller.select_image(row)
        if image is not None:
            self.statusBar().showMessage(f"Selected {image.display_name}.", 3000)

    def _update_active_image_label(self, active_image: Any) -> None:
        if active_image is None:
            self.active_image_label.setText("Active image: none")
            return
        self.active_image_label.setText(
            f"Active image: {active_image.display_name} "
            f"({active_image.pixels_x}x{active_image.pixels_y} px)"
        )

    def _update_gaussian_fit_preview(self, *_args: Any) -> None:
        image = self.controller.active_image
        roi = self.controller.active_roi_state

        if image is None:
            self.gaussian_fit_preview.set_fit_result(None)
            self.workflow_status_label.setText("Workflow status: load an STM image to begin.")
            return

        if roi is None:
            self.gaussian_fit_preview.set_fit_result(None)
            self.workflow_status_label.setText(
                f"Workflow status: {image.display_name} loaded. Waiting for ROI geometry."
            )
            return

        if not self.show_gaussian_fit_checkbox.isChecked():
            self.gaussian_fit_preview.set_fit_result(None)
            self.workflow_status_label.setText(
                f"Workflow status: ROI preview active for {image.display_name}. Gaussian fit preview hidden."
            )
            return

        patch = extract_roi_patch(image.image_data, roi)
        if patch is None:
            self.gaussian_fit_preview.set_fit_result(None)
            self.workflow_status_label.setText(
                f"Workflow status: ROI for {image.display_name} is outside image bounds."
            )
            return

        fit_result = fit_gaussian_to_roi_patch(
            patch,
            roi_origin_yx=(roi.y, roi.x),
            compute_uncertainty=False,
        )
        self.gaussian_fit_preview.set_fit_result(fit_result)
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

    def _on_show_gaussian_fit_changed(self, state: int) -> None:
        is_visible = state == int(Qt.CheckState.Checked.value)
        self.gaussian_fit_preview.setVisible(is_visible)
        message = "Gaussian fit preview shown." if is_visible else "Gaussian fit preview hidden."
        self.statusBar().showMessage(message, 3000)
        self._update_gaussian_fit_preview()
