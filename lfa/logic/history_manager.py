# lfa/logic/history_manager.py
import logging
from typing import Dict, Optional, Any # 'Any' może być potrzebne później dla parametrów
from PyQt6.QtWidgets import QListWidget, QListWidgetItem # Dodaj potrzebne importy Qt
from PyQt6.QtCore import Qt, pyqtSignal, QObject
# Użyj ścieżki względnej, jeśli HistoryNode jest w lfa.core
# Zakładając, że history_manager.py jest w lfa/gui/, a HistoryNode w lfa/core/
from ..core.history import HistoryNode

logger = logging.getLogger(__name__)

class HistoryManager(QObject):
    """
    Manages the history of operations, including adding nodes,
    tracking the current node, and interacting with the history list widget.
    """
    # Sygnał emitowany, gdy zmieni się bieżący węzeł historii.
    # Przekazuje nowy bieżący węzeł (obiekt HistoryNode lub None).
    current_node_changed = pyqtSignal(object) 

    def __init__(self, history_list_widget: QListWidget, parent: Optional[QObject] = None):
        """
        Initializes the HistoryManager.

        Args:
            history_list_widget (QListWidget): The QListWidget from the main window
                                               used to display history items.
            parent (Optional[QObject]): Parent object for Qt's memory management.
        """
        super().__init__(parent)
        self.history: Dict[str, HistoryNode] = {}
        self.current_node_id: Optional[str] = None
        self.history_list_widget = history_list_widget

        logger.debug("HistoryManager initialized.")

    def clear_history(self):
        """Clears the entire history and updates the list widget."""
        self.history.clear()
        self.current_node_id = None
        self.history_list_widget.clear() # HistoryManager jest odpowiedzialny za widget
        logger.info("History cleared by HistoryManager.")
        # Emituj sygnał, że bieżący węzeł to None, aby MainWindow mogło zareagować
        # (np. wyczyścić widok obrazu, zaktualizować stan akcji)
        self.current_node_changed.emit(None)

    def add_node(self, node: HistoryNode) -> Optional[QListWidgetItem]:
        """
        Adds a new node to the history and to the QListWidget.

        Args:
            node (HistoryNode): The history node to add.

        Returns:
            Optional[QListWidgetItem]: The QListWidgetItem added to the list, or None on failure.
        """
        if not node or not node.node_id:
            logger.error("HistoryManager: Attempted to add invalid or null node to history.")
            return None
        if node.node_id in self.history:
            logger.warning(f"HistoryManager: Node with ID {node.node_id} already exists. Not adding.")
            # Można by zwrócić istniejący item, jeśli to pożądane
            for i in range(self.history_list_widget.count()):
                item = self.history_list_widget.item(i)
                if item.data(Qt.ItemDataRole.UserRole) == node.node_id:
                    return item
            return None

        self.history[node.node_id] = node
        item = QListWidgetItem(node.get_display_text())
        item.setData(Qt.ItemDataRole.UserRole, node.node_id)
        self.history_list_widget.addItem(item)
        logger.debug(f"HistoryManager added node: '{node.get_display_text()}' (ID: {node.node_id})")
        return item

    def get_current_node(self) -> Optional[HistoryNode]:
        """
        Returns the current active HistoryNode object.

        Returns:
            Optional[HistoryNode]: The current node, or None if no node is active.
        """
        if self.current_node_id:
            return self.history.get(self.current_node_id)
        return None

    def get_node_by_id(self, node_id: str) -> Optional[HistoryNode]:
        """
        Retrieves a specific HistoryNode by its ID.

        Args:
            node_id (str): The ID of the node to retrieve.

        Returns:
            Optional[HistoryNode]: The found node, or None if not found.
        """
        if not node_id:
            return None
        return self.history.get(node_id)
    
    def set_current_node_by_id(self, node_id: Optional[str], emit_signal: bool = True):
        """
        Sets the current active node by its ID.

        Updates the internal current_node_id, synchronizes the selection
        in the QListWidget, and optionally emits the current_node_changed signal.

        Args:
            node_id (Optional[str]): The ID of the node to set as current,
                                     or None to clear the current selection.
            emit_signal (bool): If True, the current_node_changed signal is emitted.
                                Defaults to True.
        """
        previous_node_id = self.current_node_id

        if node_id is not None and node_id not in self.history:
            logger.error(f"HistoryManager: Cannot set current node to ID '{node_id}' - not found in history.")
            # Optionally, revert to previous_node_id or set to None if previous was also invalid
            if previous_node_id not in self.history:
                self.current_node_id = None
            else:
                self.current_node_id = previous_node_id # Revert to last valid known ID
                # No need to update widget if reverting, as it should reflect previous_node_id
                if emit_signal:
                    self.current_node_changed.emit(self.get_current_node())
                return # Exit early as we don't want to change widget selection to an invalid one
        else:
            self.current_node_id = node_id

        if previous_node_id != self.current_node_id: # Log only if ID actually changes
            logger.info(f"HistoryManager: Current history node ID set to: {self.current_node_id}")

        # Synchronize QListWidget selection
        self.history_list_widget.blockSignals(True) # Prevent feedback loop
        found_item = None
        for i in range(self.history_list_widget.count()):
            item = self.history_list_widget.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == self.current_node_id:
                # Only set if not already current to avoid unnecessary widget operations
                if self.history_list_widget.currentItem() is not item:
                    item.setSelected(True)
                    self.history_list_widget.setCurrentItem(item)
                found_item = item
                break
        
        if not found_item and self.current_node_id is None:
            # If node_id is None, clear selection in the list widget
            self.history_list_widget.clearSelection()
            self.history_list_widget.setCurrentItem(None)
            
        self.history_list_widget.blockSignals(False)

        # Emit signal only if the node ID has actually changed and emit_signal is true
        if emit_signal and previous_node_id != self.current_node_id:
            self.current_node_changed.emit(self.get_current_node())
        elif emit_signal and self.current_node_id is None and previous_node_id is not None:
            # Also emit if we are clearing the selection
            self.current_node_changed.emit(None)

    def get_root_node_for_node(self, node_id: Optional[str]) -> Optional[HistoryNode]:
        """
        Traces back from the given node_id to find the root node (Original).

        Args:
            node_id (Optional[str]): The ID of the starting node.
                                     If None, returns None.

        Returns:
            Optional[HistoryNode]: The root node (typically named "Original"),
                                   or the oldest ancestor found if a typical root isn't reached,
                                   or None if the starting node is invalid or history is broken.
        """
        if not node_id:
            logger.debug("get_root_node_for_node: Called with no node_id.")
            return None
        
        current_node = self.get_node_by_id(node_id)
        if not current_node:
            logger.warning(f"get_root_node_for_node: Starting node {node_id} not found in history.")
            return None

        visited_ids = {current_node.node_id} # To prevent cycles

        # Iterate backwards, max 100 steps for safety against malformed history
        for _ in range(100): 
            if not current_node.parent_id: # No parent means it's a root
                logger.debug(f"get_root_node_for_node: Found root (no parent_id) for {node_id}: {current_node.node_id}")
                return current_node
            
            if current_node.operation_name == "Original": # Explicit "Original" node is a root
                logger.debug(f"get_root_node_for_node: Found 'Original' node for {node_id}: {current_node.node_id}")
                return current_node

            if current_node.parent_id not in self.history:
                logger.warning(f"get_root_node_for_node: Parent ID {current_node.parent_id} for node {current_node.node_id} not found in history. Returning current node as oldest.")
                return current_node # Return current node as it's the oldest we can trace

            # Safety check for cycles
            if current_node.parent_id in visited_ids:
                logger.error(f"Cycle detected in history trace for node {node_id} at parent {current_node.parent_id}. Aborting root search.")
                return current_node # Return current node to prevent infinite loop

            parent_node = self.history.get(current_node.parent_id)
            if not parent_node: # Should have been caught by `parent_id not in self.history`
                logger.error(f"get_root_node_for_node: Inconsistency - Parent node {current_node.parent_id} was in history keys but get returned None.")
                return current_node 
            
            current_node = parent_node
            visited_ids.add(current_node.node_id)
        
        # If loop finished due to iteration limit without finding a definitive root
        logger.warning(f"get_root_node_for_node: Reached iteration limit for {node_id}. Returning oldest ancestor found: {current_node.node_id}")
        return current_node