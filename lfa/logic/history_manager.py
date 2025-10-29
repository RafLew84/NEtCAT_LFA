import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, TYPE_CHECKING

from PyQt6.QtCore import QObject, Qt, pyqtSignal
from PyQt6.QtWidgets import QListWidget, QListWidgetItem

from ..core.data_models import OriginalImageRecord
from ..core.history import HistoryNode
from .history_backend import HistoryBackend

if TYPE_CHECKING:  # pragma: no cover
    from .session_state import HistoryState


@dataclass(frozen=True)
class ActiveNodeChangedEvent:
    """Represents an update to the currently active history node."""

    node_id: Optional[str]
    node: Optional[HistoryNode]
    reason: str


@dataclass(frozen=True)
class OriginalImageEvent:
    """Describes mutations involving original image roots."""

    image_id: str
    record: Optional[OriginalImageRecord]


logger = logging.getLogger(__name__)


class HistoryManager(QObject):
    """
    Qt-aware wrapper around the pure HistoryBackend.

    Signals
    -------
    active_node_changed(ActiveNodeChangedEvent)
        Emitted whenever the current selection changes, alongside a reason
        describing why (manual selection, clear, forced update, etc.).
    original_image_added(OriginalImageEvent)
        Fired after a new original image has been registered in the backend.
    original_image_removed(OriginalImageEvent)
        Fired after an original image (and its branch) is removed.
    history_structure_changed()
        Broadcast whenever the order or content of the list widget may have
        changed and views should refresh any cached indices.
    current_node_changed(HistoryNode | None)
        Legacy signal kept for backwards compatibility. New code should prefer
        ``active_node_changed`` to access the structured event payload.
    """

    current_node_changed = pyqtSignal(object)
    active_node_changed = pyqtSignal(object)
    original_image_added = pyqtSignal(object)
    original_image_removed = pyqtSignal(object)
    history_cleared = pyqtSignal()
    history_structure_changed = pyqtSignal()

    def __init__(self, history_list_widget: QListWidget, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self.history_list_widget = history_list_widget
        self.backend = HistoryBackend()
        logger.debug("HistoryManager initialized.")

    # ------------------------------------------------------------------ Compatibility accessors
    @property
    def history(self) -> Dict[str, HistoryNode]:
        return self.backend.history

    @history.setter
    def history(self, value: Dict[str, HistoryNode]) -> None:
        self.backend.history = value

    @property
    def current_node_id(self) -> Optional[str]:
        return self.backend.current_node_id

    @current_node_id.setter
    def current_node_id(self, value: Optional[str]) -> None:
        self.backend.current_node_id = value

    @property
    def original_images(self) -> Dict[str, OriginalImageRecord]:
        return self.backend.original_images

    @original_images.setter
    def original_images(self, value: Dict[str, OriginalImageRecord]) -> None:
        self.backend.original_images = value

    @property
    def _original_order(self) -> List[str]:
        return self.backend.original_order

    @_original_order.setter
    def _original_order(self, value: List[str]) -> None:
        self.backend.original_order = list(value)

    @property
    def _root_nodes_by_image_id(self) -> Dict[str, str]:
        return self.backend.root_nodes_by_image_id

    @_root_nodes_by_image_id.setter
    def _root_nodes_by_image_id(self, value: Dict[str, str]) -> None:
        self.backend.root_nodes_by_image_id = value

    # ------------------------------------------------------------------ Core operations
    def clear_history(self) -> None:
        """Clear all history data and notify listeners about the reset."""
        self.backend.clear()
        self.history_list_widget.clear()
        logger.info("History cleared by HistoryManager.")
        self.history_structure_changed.emit()
        self.history_cleared.emit()
        self._emit_active_node_changed(reason="cleared")

    def refresh_widget(self) -> None:
        """Re-populate the QListWidget based on the backend's current state."""
        current_id = self.backend.current_node_id
        self.history_list_widget.blockSignals(True)
        self.history_list_widget.clear()

        for image_id in self.backend.original_order:
            nodes = [
                node for node in self.backend.history.values()
                if node.original_image_id == image_id
            ]
            if not nodes:
                continue
            nodes.sort(key=lambda n: (0 if n.parent_id is None or n.operation_name == "Original" else 1, n.timestamp))
            for node in nodes:
                self.history_list_widget.addItem(self._create_item_for_node(node))

        legacy_nodes = [node for node in self.backend.history.values() if not node.original_image_id]
        legacy_nodes.sort(key=lambda n: n.timestamp)
        for node in legacy_nodes:
            self.history_list_widget.addItem(self._create_item_for_node(node))

        self.history_list_widget.blockSignals(False)
        self.set_current_node_by_id(current_id, emit_signal=False)
        self.history_structure_changed.emit()

    def register_original_image(self, record: OriginalImageRecord) -> None:
        """Register a new original image and emit the structured event."""
        self.backend.register_original_image(record)
        logger.debug("Registered original image record: %s", record.display_name)
        self.original_image_added.emit(OriginalImageEvent(image_id=record.image_id, record=record))
        self.history_structure_changed.emit()

    def unregister_original_image(self, image_id: str) -> None:
        """Remove an original image branch and notify observers."""
        existing = self.backend.get_original_image_record(image_id)
        self.backend.unregister_original_image(image_id)
        self.original_image_removed.emit(OriginalImageEvent(image_id=image_id, record=existing))
        self.history_structure_changed.emit()

    def get_original_image_record(self, image_id: str) -> Optional[OriginalImageRecord]:
        return self.backend.get_original_image_record(image_id)

    def iter_original_image_ids(self) -> List[str]:
        return list(self.backend.iter_original_image_ids())

    def get_next_original_display_name(self) -> str:
        return self.backend.get_next_original_display_name()

    def rebuild_indexes(self) -> None:
        """Rebuild internal indexes that connect history nodes with original images."""
        self.backend.rebuild_indexes()

    # ------------------------------------------------------------------ Session IO helpers
    def export_session_state(self) -> "HistoryState":
        from .session_state import HistoryState

        return HistoryState.from_history_manager(self)

    def load_session_state(self, state: "HistoryState") -> None:
        from .session_state import HistoryState

        if not isinstance(state, HistoryState):
            raise TypeError("Expected HistoryState when loading session data.")
        state.apply_to(self)
        self.set_current_node_by_id(self.current_node_id, emit_signal=True, force_signal=True)

    def add_node(self, node: HistoryNode) -> Optional[QListWidgetItem]:
        """Insert a node into the backend and mirrored list widget."""
        if not node or not node.node_id:
            logger.error("HistoryManager: Attempted to add invalid or null node to history.")
            return None
        if node.node_id in self.backend.history:
            logger.warning("HistoryManager: Node with ID %s already exists. Not adding.", node.node_id)
            for i in range(self.history_list_widget.count()):
                item = self.history_list_widget.item(i)
                if item.data(Qt.ItemDataRole.UserRole) == node.node_id:
                    return item
            return None

        self.backend.add_node(node)
        item = self._create_item_for_node(node)
        insert_row = self._determine_insertion_row(node)
        self.history_list_widget.insertItem(insert_row, item)
        logger.debug("HistoryManager added node: '%s' (ID: %s) at row %d", node.get_display_text(), node.node_id, insert_row)
        self.history_structure_changed.emit()
        return item

    def get_current_node(self) -> Optional[HistoryNode]:
        node_id = self.backend.current_node_id
        return self.backend.history.get(node_id) if node_id else None

    def get_node_by_id(self, node_id: str) -> Optional[HistoryNode]:
        return self.backend.history.get(node_id)

    def set_current_node_by_id(
        self,
        node_id: Optional[str],
        emit_signal: bool = True,
        force_signal: bool = False,
    ) -> None:
        previous_node_id = self.backend.current_node_id
        try:
            self.backend.set_current_node(node_id)
        except KeyError:
            logger.error("HistoryManager: Cannot set current node to ID '%s' - not found in history.", node_id)
            if emit_signal:
                self._emit_active_node_changed(reason="invalid-selection")
            return

        self.history_list_widget.blockSignals(True)
        self._update_selection(self.backend.current_node_id)
        self.history_list_widget.blockSignals(False)

        if not emit_signal:
            return

        if force_signal:
            self._emit_active_node_changed(reason="forced")
            return

        if previous_node_id != self.backend.current_node_id:
            self._emit_active_node_changed(reason="selection-changed")
        elif self.backend.current_node_id is None and previous_node_id is not None:
            self._emit_active_node_changed(reason="selection-cleared")

    def get_root_node_for_node(self, node_id: Optional[str]) -> Optional[HistoryNode]:
        return self.backend.get_root_node_for_node(node_id)

    def delete_node_branch(self, node_id: str) -> Optional[Dict[str, Optional[str]]]:
        """Delete a node and its descendants, keeping selection consistent."""
        try:
            result = self.backend.delete_node_branch(node_id)
        except KeyError:
            logger.warning("HistoryManager.delete_node_branch: Node %s not found.", node_id)
            return None

        self.refresh_widget()
        self.set_current_node_by_id(
            result.get("new_current_node_id"),
            emit_signal=True,
            force_signal=True,
        )
        return result

    def delete_original_image_branch(self, image_id: str) -> Optional[Dict[str, Optional[str]]]:
        """Remove an entire original image branch and re-evaluate selection."""
        try:
            result = self.backend.delete_original_image_branch(image_id)
        except KeyError:
            logger.warning("HistoryManager.delete_original_image_branch: Original image %s not found.", image_id)
            return None
        self.refresh_widget()
        self.set_current_node_by_id(
            result.get("new_current_node_id"),
            emit_signal=True,
            force_signal=True,
        )
        return result

    # ------------------------------------------------------------------ Helper methods
    def _emit_active_node_changed(self, reason: str) -> None:
        """Emit the rich event payload plus the legacy signal for compatibility."""
        payload = ActiveNodeChangedEvent(
            node_id=self.backend.current_node_id,
            node=self.get_current_node(),
            reason=reason,
        )
        self.active_node_changed.emit(payload)
        self.current_node_changed.emit(payload.node)

    def _create_item_for_node(self, node: HistoryNode) -> QListWidgetItem:
        item = QListWidgetItem(node.get_display_text())
        item.setData(Qt.ItemDataRole.UserRole, node.node_id)
        if node.original_image_id:
            item.setData(Qt.ItemDataRole.UserRole + 2, node.original_image_id)
        if node.operation_name == "Original":
            font = item.font()
            font.setBold(True)
            item.setFont(font)
            label = node.parameters.get("original_label", "Original Image")
            tooltip = f"Original STM image ({label})"
            filename = node.parameters.get("filename")
            if filename:
                tooltip += f"\nFile: {filename}"
            item.setToolTip(tooltip)
        else:
            label = node.parameters.get("source_image_label")
            if label:
                item.setToolTip(f"Derived from {label}")
        return item

    def _determine_insertion_row(self, node: HistoryNode) -> int:
        count = self.history_list_widget.count()
        if not node.original_image_id or count == 0:
            return count
        last_row = -1
        for row in range(count):
            item = self.history_list_widget.item(row)
            if not item:
                continue
            existing_id = item.data(Qt.ItemDataRole.UserRole)
            if not existing_id:
                continue
            existing_node = self.backend.history.get(existing_id)
            if existing_node and existing_node.original_image_id == node.original_image_id:
                last_row = row
        return last_row + 1 if last_row >= 0 else count

    def _update_selection(self, node_id: Optional[str]) -> None:
        if node_id is None:
            self.history_list_widget.clearSelection()
            self.history_list_widget.setCurrentItem(None)
            return

        found_item = None
        for i in range(self.history_list_widget.count()):
            item = self.history_list_widget.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == node_id:
                if self.history_list_widget.currentItem() is not item:
                    item.setSelected(True)
                    self.history_list_widget.setCurrentItem(item)
                found_item = item
                break
        if not found_item:
            self.history_list_widget.clearSelection()
