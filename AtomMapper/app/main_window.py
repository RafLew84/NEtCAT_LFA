"""Main window for the AtomMapper application."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
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
from .image_view import STMImageViewport
from .io import SUPPORTED_STM_EXTENSIONS

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

        self.image_viewport = STMImageViewport(right_panel)

        right_layout.addWidget(title)
        right_layout.addWidget(subtitle)
        right_layout.addWidget(self.active_image_label)
        right_layout.addWidget(self.image_viewport, 1)

        root_layout.addWidget(left_panel)
        root_layout.addWidget(right_panel, 1)

        self.setCentralWidget(central)
        self.statusBar().showMessage("Ready. Load an STM file to begin.", 5000)
        self._connect_signals()
        self._refresh_file_list()
        self._update_active_image_label(self.controller.active_image)
        self.image_viewport.set_loaded_image(self.controller.active_image)

    def _connect_signals(self) -> None:
        self.load_button.clicked.connect(self._open_file_dialog)
        self.file_list_widget.currentRowChanged.connect(self._on_current_row_changed)
        self.controller.loaded_images_changed.connect(self._refresh_file_list)
        self.controller.active_image_changed.connect(self._update_active_image_label)
        self.controller.active_image_changed.connect(self.image_viewport.set_loaded_image)

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
        self.controller.select_image(row)

    def _update_active_image_label(self, active_image: Any) -> None:
        if active_image is None:
            self.active_image_label.setText("Active image: none")
            return
        self.active_image_label.setText(
            f"Active image: {active_image.display_name} "
            f"({active_image.pixels_x}x{active_image.pixels_y} px)"
        )
