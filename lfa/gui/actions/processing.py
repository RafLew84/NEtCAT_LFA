"""Helpers for launching preprocessing dialogs from the main window."""

from __future__ import annotations

import logging
from typing import Callable, Optional

from PyQt6.QtWidgets import QDialog, QMessageBox

logger = logging.getLogger(__name__)


class ProcessingDialogLauncher:
    """Encapsulates the common flow for preprocessing dialogs."""

    def __init__(self, main_window, app_controller) -> None:
        self._window = main_window
        self._controller = app_controller

    def launch(
        self,
        dialog_class,
        controller_method_name: str,
        *,
        dialog_specific_checks: Optional[Callable[[], bool]] = None,
    ) -> None:
        """Open the given dialog and dispatch the result to the controller."""
        if not self._controller:
            logger.error("ProcessingDialogLauncher: AppController unavailable.")
            return

        current_node_info = self._controller.get_current_node_info_for_dialogs()
        if not current_node_info:
            QMessageBox.warning(
                self._window,
                "No Image",
                "No data loaded or selected in history to process.",
            )
            return

        (
            parent_id,
            parent_data_type,
            image_data_copy,
            source_image_id,
            source_label,
        ) = current_node_info

        if not dialog_class:
            name = getattr(dialog_class, "__name__", "Dialog")
            QMessageBox.critical(self._window, "Error", f"{name} is not available.")
            return

        if dialog_specific_checks is not None and not dialog_specific_checks():
            return

        dialog = dialog_class(image_data_copy, parent=self._window)
        dialog.source_image_id = source_image_id
        dialog.source_image_label = source_label
        if source_label and source_label not in dialog.windowTitle():
            dialog.setWindowTitle(f"{dialog.windowTitle()} [{source_label}]")

        if dialog.exec() != QDialog.DialogCode.Accepted:
            op_display_name = getattr(dialog, "operation_name", controller_method_name)
            self._status(f"{op_display_name} cancelled.")
            logger.info("%s dialog cancelled.", op_display_name)
            return

        processed_data = dialog.get_processed_data()
        if processed_data is None:
            logger.warning(
                "%s accepted, but no processed data returned.",
                getattr(dialog, "__class__", type(dialog)).__name__,
            )
            self._status("Operation cancelled or no changes made.")
            return

        params = dialog.get_parameters()
        was_roi_only = dialog.was_roi_applied_only()
        roi_slice = dialog.get_final_roi_slice() if was_roi_only else None

        controller_method = getattr(self._controller, controller_method_name, None)
        if not callable(controller_method):
            logger.error(
                "ProcessingDialogLauncher: method %s not found on controller.",
                controller_method_name,
            )
            self._status(f"Error applying {controller_method_name}.")
            return

        controller_method(
            parent_node_id=parent_id,
            parent_data_type=parent_data_type,
            processed_data=processed_data,
            params=params,
            source_roi_slice=roi_slice,
        )

        op_display_name = getattr(dialog, "operation_name", controller_method_name)
        logger.info("%s applied successfully.", op_display_name)
        self._status(f"{op_display_name} applied.")

    def _status(self, message: str, timeout_ms: int = 3000) -> None:
        """Show a message on the main window status bar if available."""
        status_bar = getattr(self._window, "statusBar", None)
        if callable(status_bar):
            status_bar().showMessage(message, timeout_ms)
