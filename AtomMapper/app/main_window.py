"""Main window for the AtomMapper application."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QDialog,
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
from .gaussian_preview import GaussianFitPreviewWidget
from .io import SUPPORTED_STM_EXTENSIONS
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

        left_layout.addWidget(left_title)
        left_layout.addWidget(self.load_button)
        left_layout.addWidget(self.preprocessing_button)
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

        self.image_viewport = PyQtGraphSTMViewport(right_panel)
        self.roi_preview = ROIPreviewWidget(right_panel)
        self.gaussian_fit_preview = GaussianFitPreviewWidget(right_panel)
        self.preview_bridge = PyQtGraphPreviewBridge(
            self.image_viewport,
            self.roi_preview,
            self.gaussian_fit_preview,
            self,
        )
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
        self._preprocessing_dialog_class = PreprocessingDialog
        self.statusBar().showMessage("Ready. Load an STM file to begin.", 5000)
        self._connect_signals()
        self._refresh_file_list()
        self._update_active_image_label(self.controller.active_image)
        self._update_preprocess_controls(self.controller.active_image)
        self._handle_active_image_changed(self.controller.active_image)

    def _connect_signals(self) -> None:
        self.load_button.clicked.connect(self._open_file_dialog)
        self.preprocessing_button.clicked.connect(self._open_preprocessing_dialog)
        self.file_list_widget.currentRowChanged.connect(self._on_current_row_changed)
        self.controller.loaded_images_changed.connect(self._refresh_file_list)
        self.controller.active_image_changed.connect(self._update_active_image_label)
        self.controller.active_image_changed.connect(self._update_preprocess_controls)
        self.controller.active_image_changed.connect(self._handle_active_image_changed)
        self.controller.roi_state_changed.connect(self._handle_roi_state_changed)
        self.show_gaussian_fit_checkbox.stateChanged.connect(self._on_show_gaussian_fit_changed)
        self.preview_bridge.roi_state_edited.connect(self.controller.update_active_roi_state)

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
        result = dialog.exec()
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
