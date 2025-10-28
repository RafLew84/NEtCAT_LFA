import logging
from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QMenu, QMessageBox


logger = logging.getLogger(__name__)


class HistoryContextMenu:
    """Encapsulates context-menu handling for the history list."""

    def __init__(self, main_window) -> None:
        self.main_window = main_window

    # ------------------------------------------------------------------ API
    def show_menu(self, position) -> None:
        history_manager = getattr(self.main_window, "history_manager", None)
        if not history_manager:
            return

        list_widget = self.main_window.history_list_widget
        item = list_widget.itemAt(position)
        if item is None:
            return

        node_id = item.data(Qt.ItemDataRole.UserRole)
        if not node_id:
            return

        node = history_manager.get_node_by_id(node_id)
        if not node:
            return

        menu = QMenu(self.main_window)
        is_root = node.parent_id is None or node.operation_name == "Original"

        delete_step_action = None
        if not is_root:
            delete_step_action = menu.addAction("Delete Step")

        original_image_id = item.data(Qt.ItemDataRole.UserRole + 2) or node.original_image_id
        root_node = history_manager.get_root_node_for_node(node.node_id)
        original_label = None
        if root_node:
            original_image_id = root_node.original_image_id
            original_label = root_node.parameters.get("original_label")
        if original_label is None:
            original_label = node.parameters.get("original_label")

        delete_original_action = None
        label_text = None
        if original_image_id:
            label_text = original_label or f"Original Image {original_image_id}"
            delete_original_action = menu.addAction(f"Delete {label_text}...")

        if not menu.actions():
            return

        chosen_action = menu.exec(list_widget.mapToGlobal(position))
        if not chosen_action:
            return

        if delete_step_action and chosen_action == delete_step_action:
            self._delete_history_step(node_id)
        elif delete_original_action and chosen_action == delete_original_action and original_image_id:
            self._delete_original_image(original_image_id, node_id, label_text)

    # ------------------------------------------------------------------ Helpers
    def _delete_history_step(self, node_id: str) -> None:
        app_controller = getattr(self.main_window, "app_controller", None)
        if not (app_controller and node_id):
            return

        confirmation = QMessageBox.question(
            self.main_window,
            "Delete Step",
            "Are you sure you want to delete this history step?\nThis action cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if confirmation != QMessageBox.StandardButton.Yes:
            return

        try:
            success = app_controller.delete_history_step(node_id)
        except AttributeError:
            logger.error("AppController.delete_history_step is not available.")
            success = False

        if success:
            self.main_window.statusBar().showMessage("History step deleted.", 3000)
        else:
            QMessageBox.warning(self.main_window, "Delete Failed", "Could not delete the selected history step.")

    def _delete_original_image(self, image_id: str, node_id: str, label_text: Optional[str]) -> None:
        app_controller = getattr(self.main_window, "app_controller", None)
        if not app_controller:
            return

        display_label = label_text or "this original image"
        confirmation = QMessageBox.question(
            self.main_window,
            "Delete Original Image",
            f"Delete {display_label} and all of its derived history steps?\nThis action cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if confirmation != QMessageBox.StandardButton.Yes:
            return

        try:
            success = app_controller.delete_original_image(image_id, node_id=node_id)
        except TypeError:
            success = app_controller.delete_original_image(image_id)  # type: ignore[arg-type]
        except AttributeError:
            logger.error("AppController.delete_original_image is not available.")
            success = False

        if success:
            self.main_window.statusBar().showMessage(f"{display_label} deleted.", 3000)
        else:
            QMessageBox.warning(self.main_window, "Delete Failed", f"Could not delete {display_label}.")
