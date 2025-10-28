# lfa/logic/history_manager.py
import logging
from typing import Dict, Optional, Any, List, Set
from PyQt6.QtWidgets import QListWidget, QListWidgetItem
from PyQt6.QtCore import Qt, pyqtSignal, QObject
from ..core.history import HistoryNode
from ..core.data_models import OriginalImageRecord
import uuid

logger = logging.getLogger(__name__)

class HistoryManager(QObject):
    """
    Manages the history of operations, including adding nodes,
    tracking the current node, and interacting with the history list widget.
    """
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
        self.original_images: Dict[str, OriginalImageRecord] = {}
        self._original_order: List[str] = [] # Keeps insertion order of original images
        self._root_nodes_by_image_id: Dict[str, str] = {}

        logger.debug("HistoryManager initialized.")

    def clear_history(self):
        """Clears the entire history and updates the list widget."""
        self.history.clear()
        self.current_node_id = None
        self.history_list_widget.clear()
        self.original_images.clear()
        self._original_order.clear()
        self._root_nodes_by_image_id.clear()
        logger.info("History cleared by HistoryManager.")
        self.current_node_changed.emit(None)

    def _create_item_for_node(self, node: HistoryNode) -> QListWidgetItem:
        """Create a view item for a history node."""
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
        """Determine list row for inserting a node to keep grouping."""
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
            existing_node = self.history.get(existing_id)
            if existing_node and existing_node.original_image_id == node.original_image_id:
                last_row = row
        return last_row + 1 if last_row >= 0 else count

    def _collect_descendant_ids(self, node_id: str) -> Set[str]:
        """
        Collects the IDs of the node and all of its descendants.
        """
        descendants: Set[str] = set()
        stack = [node_id]
        nodes_snapshot = list(self.history.values())

        while stack:
            current_id = stack.pop()
            if current_id in descendants:
                continue
            node = self.history.get(current_id)
            if not node:
                continue
            descendants.add(current_id)
            for candidate in nodes_snapshot:
                if candidate.parent_id == current_id:
                    stack.append(candidate.node_id)

        return descendants

    def refresh_widget(self):
        """Rebuild the QListWidget from the current history tree."""
        current_id = self.current_node_id
        self.history_list_widget.blockSignals(True)
        self.history_list_widget.clear()

        # Render grouped by original image order
        for image_id in self._original_order:
            nodes = [
                node for node in self.history.values()
                if node.original_image_id == image_id
            ]
            if not nodes:
                continue
            nodes.sort(key=lambda n: (0 if n.parent_id is None or n.operation_name == "Original" else 1, n.timestamp))
            for node in nodes:
                item = self._create_item_for_node(node)
                self.history_list_widget.addItem(item)

        legacy_nodes = [node for node in self.history.values() if not node.original_image_id]
        legacy_nodes.sort(key=lambda n: n.timestamp)
        for node in legacy_nodes:
            item = self._create_item_for_node(node)
            self.history_list_widget.addItem(item)

        self.history_list_widget.blockSignals(False)
        if current_id and current_id in self.history:
            self.set_current_node_by_id(current_id, emit_signal=False)
        else:
            self.set_current_node_by_id(None, emit_signal=False)

    def register_original_image(self, record: OriginalImageRecord):
        """
        Registers metadata for an original STM image.
        """
        if record.image_id not in self.original_images:
            self._original_order.append(record.image_id)
        self.original_images[record.image_id] = record
        logger.debug("Registered original image record: %s", record.display_name)

    def unregister_original_image(self, image_id: str):
        """
        Removes the original image record and associated root mapping.
        """
        if image_id in self.original_images:
            del self.original_images[image_id]
        if image_id in self._root_nodes_by_image_id:
            del self._root_nodes_by_image_id[image_id]
        if image_id in self._original_order:
            self._original_order.remove(image_id)

    def get_original_image_record(self, image_id: str) -> Optional[OriginalImageRecord]:
        """
        Retrieves the registered record for an original image.
        """
        return self.original_images.get(image_id)

    def iter_original_image_ids(self) -> List[str]:
        """
        Returns original image IDs in registration order.
        """
        return list(self._original_order)

    def _assign_original_image_for_node(self, node: HistoryNode):
        """
        Ensures the node has an original_image_id assigned.
        """
        if node.original_image_id:
            return

        # If the node has a parent, inherit the parent's original image id
        if node.parent_id and node.parent_id in self.history:
            parent = self.history[node.parent_id]
            node.original_image_id = parent.original_image_id
            return

        legacy_id = str(uuid.uuid4())
        node.original_image_id = legacy_id
        display_name = node.parameters.get("original_label") or self._generate_default_display_name()
        node.parameters["original_label"] = display_name
        legacy_record = OriginalImageRecord(
            image_id=legacy_id,
            display_name=display_name,
            stm_image=None,
            source_path=node.parameters.get("filename"),
            extra_metadata=dict(node.parameters)
        )
        self.register_original_image(legacy_record)
        self._root_nodes_by_image_id[legacy_id] = node.node_id

    def _generate_default_display_name(self) -> str:
        """
        Generates the next display name for an original image.
        """
        next_index = len(self._original_order) + 1
        return f"Original Image {next_index}"

    def get_next_original_display_name(self) -> str:
        """
        Returns a display label for a newly added original image without mutating state.
        """
        return self._generate_default_display_name()

    def _finalize_root_registration(self, node: HistoryNode):
        """
        Records root node mapping and ensures original image metadata exists.
        """
        if not node.original_image_id:
            self._assign_original_image_for_node(node)

        image_id = node.original_image_id
        if image_id is None:
            return

        # Ensure registration exists
        if image_id not in self.original_images:
            display_name = node.parameters.get("original_label") or self._generate_default_display_name()
            node.parameters["original_label"] = display_name
            node.parameters["source_image_label"] = display_name
            record = OriginalImageRecord(
                image_id=image_id,
                display_name=display_name,
                stm_image=None,
                source_path=node.parameters.get("filename"),
                extra_metadata=dict(node.parameters)
            )
            self.register_original_image(record)
        else:
            record = self.original_images[image_id]
            node.parameters.setdefault("original_label", record.display_name)
            node.parameters.setdefault("source_image_label", record.display_name)
            record.extra_metadata = dict(node.parameters)

        self._root_nodes_by_image_id[image_id] = node.node_id

    def rebuild_indexes(self):
        """
        Rebuilds the mapping between original images and history roots.
        Useful after loading history from disk.
        """
        self._root_nodes_by_image_id.clear()
        known_ids = set(self.original_images.keys())
        for node in self.history.values():
            if node.parent_id and node.original_image_id is None:
                parent = self.history.get(node.parent_id)
                if parent:
                    node.original_image_id = parent.original_image_id

        root_candidates = [
            node for node in self.history.values()
            if node.parent_id is None or node.operation_name == "Original"
        ]
        root_candidates.sort(key=lambda n: n.timestamp)

        self._original_order = []
        for node in root_candidates:
            if node.parent_id is not None and node.operation_name != "Original":
                continue
            if node.original_image_id is None:
                node.original_image_id = str(uuid.uuid4())
            image_id = node.original_image_id

            display_name = node.parameters.get("original_label")
            if not display_name:
                display_name = self._generate_default_display_name()
                node.parameters["original_label"] = display_name

            if image_id not in self.original_images or image_id not in known_ids:
                record = OriginalImageRecord(
                    image_id=image_id,
                    display_name=display_name,
                    stm_image=None,
                    source_path=node.parameters.get("filename"),
                    extra_metadata=dict(node.parameters)
                )
                self.original_images[image_id] = record
            else:
                record = self.original_images[image_id]
                record.extra_metadata = dict(node.parameters)
            self._original_order.append(image_id)
            self._root_nodes_by_image_id[image_id] = node.node_id

        for node in self.history.values():
            if not node.original_image_id:
                continue
            record = self.original_images.get(node.original_image_id)
            if not record:
                continue
            node.parameters.setdefault("source_image_label", record.display_name)
            if node.operation_name == "Original":
                node.parameters.setdefault("original_label", record.display_name)

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
            # Optionally return the existing item if desired
            for i in range(self.history_list_widget.count()):
                item = self.history_list_widget.item(i)
                if item.data(Qt.ItemDataRole.UserRole) == node.node_id:
                    return item
            return None

        # Ensure original image linkage propagates
        self._assign_original_image_for_node(node)

        self.history[node.node_id] = node

        if node.parent_id is None or node.operation_name == "Original":
            self._finalize_root_registration(node)

        item = self._create_item_for_node(node)
        insert_row = self._determine_insertion_row(node)
        self.history_list_widget.insertItem(insert_row, item)
        logger.debug(
            "HistoryManager added node: '%s' (ID: %s) at row %d",
            node.get_display_text(),
            node.node_id,
            insert_row
        )
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
            if previous_node_id not in self.history:
                self.current_node_id = None
            else:
                self.current_node_id = previous_node_id # Revert to last valid known ID
                if emit_signal:
                    self.current_node_changed.emit(self.get_current_node())
                return 
        else:
            self.current_node_id = node_id

        if previous_node_id != self.current_node_id: # Log only if ID actually changes
            logger.info(f"HistoryManager: Current history node ID set to: {self.current_node_id}")

        self.history_list_widget.blockSignals(True) # Prevent feedback loop
        found_item = None
        for i in range(self.history_list_widget.count()):
            item = self.history_list_widget.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == self.current_node_id:
                if self.history_list_widget.currentItem() is not item:
                    item.setSelected(True)
                    self.history_list_widget.setCurrentItem(item)
                found_item = item
                break
        
        if not found_item and self.current_node_id is None:
            self.history_list_widget.clearSelection()
            self.history_list_widget.setCurrentItem(None)
            
        self.history_list_widget.blockSignals(False)

        if emit_signal and previous_node_id != self.current_node_id:
            self.current_node_changed.emit(self.get_current_node())
        elif emit_signal and self.current_node_id is None and previous_node_id is not None:
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

        if current_node.original_image_id:
            root_id = self._root_nodes_by_image_id.get(current_node.original_image_id)
            if root_id:
                root_node = self.history.get(root_id)
                if root_node:
                    return root_node

        visited_ids = {current_node.node_id}

        for _ in range(100): 
            if not current_node.parent_id: # No parent means it's a root
                logger.debug(f"get_root_node_for_node: Found root (no parent_id) for {node_id}: {current_node.node_id}")
                return current_node
            
            if current_node.operation_name == "Original": # Explicit "Original" node is a root
                logger.debug(f"get_root_node_for_node: Found 'Original' node for {node_id}: {current_node.node_id}")
                return current_node

            if current_node.parent_id not in self.history:
                logger.warning(f"get_root_node_for_node: Parent ID {current_node.parent_id} for node {current_node.node_id} not found in history. Returning current node as oldest.")
                return current_node 

            if current_node.parent_id in visited_ids:
                logger.error(f"Cycle detected in history trace for node {node_id} at parent {current_node.parent_id}. Aborting root search.")
                return current_node

            parent_node = self.history.get(current_node.parent_id)
            if not parent_node: # Should have been caught by `parent_id not in self.history`
                logger.error(f"get_root_node_for_node: Inconsistency - Parent node {current_node.parent_id} was in history keys but get returned None.")
                return current_node 
            
            current_node = parent_node
            visited_ids.add(current_node.node_id)
        
        logger.warning(f"get_root_node_for_node: Reached iteration limit for {node_id}. Returning oldest ancestor found: {current_node.node_id}")
        return current_node

    def delete_node_branch(self, node_id: str) -> Optional[Dict[str, Any]]:
        """
        Deletes the specified node and all of its descendants from history.

        Returns a dictionary with deletion details or None if deletion failed.
        """
        if not node_id:
            logger.warning("HistoryManager.delete_node_branch called with empty node_id.")
            return None

        node = self.history.get(node_id)
        if not node:
            logger.warning("HistoryManager.delete_node_branch: Node %s not found.", node_id)
            return None

        descendants = self._collect_descendant_ids(node_id)
        if not descendants:
            logger.warning("HistoryManager.delete_node_branch: No descendants found for node %s.", node_id)
            return None

        parent_id = node.parent_id
        original_image_id = node.original_image_id
        is_root = node.operation_name == "Original" or node.parent_id is None

        logger.info(
            "HistoryManager: Deleting node %s (%s) with %d descendant(s).",
            node_id,
            node.operation_name,
            len(descendants) - 1,
        )

        for removed_id in descendants:
            self.history.pop(removed_id, None)

        if is_root and original_image_id:
            self.unregister_original_image(original_image_id)

        # Remove root mapping entries pointing to deleted nodes (legacy safety)
        for image_key, root_node_id in list(self._root_nodes_by_image_id.items()):
            if root_node_id in descendants:
                self._root_nodes_by_image_id.pop(image_key, None)

        new_current_id = self.current_node_id
        if new_current_id in descendants:
            new_current_id = None

        if new_current_id is None:
            if parent_id and parent_id not in descendants and parent_id in self.history:
                new_current_id = parent_id
            else:
                candidate_id = None
                if original_image_id:
                    ordered_nodes = sorted(self.history.values(), key=lambda n: n.timestamp)
                    for remaining in ordered_nodes:
                        if remaining.original_image_id == original_image_id and remaining.node_id not in descendants:
                            candidate_id = remaining.node_id
                            break
                if candidate_id is None and self.history:
                    candidate_id = min(self.history.values(), key=lambda n: n.timestamp).node_id
                new_current_id = candidate_id

        self.current_node_id = new_current_id
        self.refresh_widget()
        self.set_current_node_by_id(new_current_id, emit_signal=True)

        return {
            "deleted_node_ids": descendants,
            "removed_original_image_id": original_image_id if is_root else None,
            "new_current_node_id": new_current_id,
        }

    def delete_original_image_branch(self, image_id: str) -> Optional[Dict[str, Any]]:
        """
        Deletes the entire history associated with the specified original image.
        """
        if not image_id:
            logger.warning("HistoryManager.delete_original_image_branch called with empty image_id.")
            return None

        root_node_id = self._root_nodes_by_image_id.get(image_id)
        if root_node_id:
            return self.delete_node_branch(root_node_id)

        # Fallback for legacy sessions without root mapping
        for node in self.history.values():
            if node.original_image_id == image_id and (node.parent_id is None or node.operation_name == "Original"):
                return self.delete_node_branch(node.node_id)

        logger.warning("HistoryManager.delete_original_image_branch: Original image %s not found.", image_id)
        return None
