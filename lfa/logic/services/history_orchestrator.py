from __future__ import annotations

import logging
from typing import Any, Optional, Tuple

from ...core.history import HistoryNode
from ...core.data_models import OriginalImageRecord
from ..history_manager import HistoryManager

logger = logging.getLogger(__name__)


class HistoryOrchestrator:
    """
    Lightweight façade over :class:`HistoryManager` that exposes common
    read/write helpers used by the controller and dialog layer.

    Public API
    ----------
    get_current_node() -> HistoryNode | None
        Return the currently selected history node.
    get_current_image_data_copy() -> ndarray | None
        Defensive copy of the active node's image data (or ``None`` when unavailable).
    get_current_node_info_for_dialogs() -> tuple | None
        Convenience tuple ``(node_id, data_type, image_copy, original_id, original_label)``.
    add_node_and_select(node: HistoryNode)
        Append a node to history and make it active.
    clear_history()
        Remove all entries and reset the underlying widget.
    refresh_widget()
        Re-synchronise the QListWidget representation with backend data.
    delete_node_branch(node_id) -> dict | None
        Delete a node (and descendants) from history.
    delete_original_image_branch(image_id) -> dict | None
        Remove an entire original image branch.
    register_original_image(record)
        Store a new original image record.
    get_root_node(node_id) -> HistoryNode | None
        Resolve the root node owning the provided node id.
    get_next_original_display_name() -> str
        Delegates to HistoryManager helper for user-friendly labels.

    These helpers are intentionally side-effect light to keep them re-usable
    from GUI presenters or future services.
    """

    def __init__(self, history_manager: HistoryManager) -> None:
        self._history = history_manager

    # ------------------------------------------------------------------ Basic queries
    def get_current_node(self) -> Optional[HistoryNode]:
        """Return the currently selected history node."""
        return self._history.get_current_node() if self._history else None

    def get_current_image_data_copy(self) -> Optional[Any]:
        """
        Return a defensive copy of the active node's image data.
        Logs a debug message when no data is present.
        """
        node = self.get_current_node()
        if node and node.image_data is not None:
            return node.image_data.copy()
        logger.debug("HistoryOrchestrator: No image data available on the current node.")
        return None

    def get_original_record(self, image_id: Optional[str]) -> Optional[OriginalImageRecord]:
        """Fetch the original image record for the given id."""
        if not image_id:
            return None
        return self._history.get_original_image_record(image_id)

    def get_current_node_info_for_dialogs(
        self,
    ) -> Optional[Tuple[str, str, Any, Optional[str], Optional[str]]]:
        """
        Provide dialog-friendly data about the active node.

        Returns
        -------
        tuple | None
            (node_id, data_type, image_data_copy, original_image_id, original_label).
        """
        node = self.get_current_node()
        if node and node.image_data is not None:
            source_image_id = node.original_image_id
            source_label = None
            if source_image_id:
                record = self.get_original_record(source_image_id)
                if record:
                    source_label = record.display_name
            return (
                node.node_id,
                node.data_type,
                node.image_data.copy(),
                source_image_id,
                source_label,
            )
        return None

    # ------------------------------------------------------------------ Mutations
    def add_node_and_select(self, node: HistoryNode) -> None:
        """
        Insert a node into the history and make it active.
        """
        self._history.add_node(node)
        self._history.set_current_node_by_id(node.node_id)
        logger.info("HistoryOrchestrator: Added node '%s' (id=%s).", node.operation_name, node.node_id)

    # ------------------------------------------------------------------ Mutations / utilities
    def register_original_image(self, record: OriginalImageRecord) -> None:
        self._history.register_original_image(record)

    def clear_history(self) -> None:
        self._history.clear_history()

    def refresh_widget(self) -> None:
        self._history.refresh_widget()

    def delete_node_branch(self, node_id: str):
        return self._history.delete_node_branch(node_id)

    def delete_original_image_branch(self, image_id: str):
        return self._history.delete_original_image_branch(image_id)

    def set_current_node(self, node_id: Optional[str]) -> None:
        self._history.set_current_node_by_id(node_id)

    def get_root_node(self, node_id: Optional[str]) -> Optional[HistoryNode]:
        return self._history.get_root_node_for_node(node_id)

    def get_node_by_id(self, node_id: str) -> Optional[HistoryNode]:
        return self._history.get_node_by_id(node_id)

    def get_next_original_display_name(self) -> str:
        return self._history.get_next_original_display_name()
