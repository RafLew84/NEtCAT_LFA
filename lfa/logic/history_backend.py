from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

from ..core.data_models import OriginalImageRecord
from ..core.history import HistoryNode

logger = logging.getLogger(__name__)


@dataclass
class HistoryBackend:
    """Qt-free storage and bookkeeping for HistoryNodes."""

    history: Dict[str, HistoryNode] = field(default_factory=dict)
    current_node_id: Optional[str] = None
    original_images: Dict[str, OriginalImageRecord] = field(default_factory=dict)
    original_order: List[str] = field(default_factory=list)
    root_nodes_by_image_id: Dict[str, str] = field(default_factory=dict)

    def clear(self) -> None:
        self.history.clear()
        self.current_node_id = None
        self.original_images.clear()
        self.original_order.clear()
        self.root_nodes_by_image_id.clear()

    # ------------------------------------------------------------------ Original images
    def register_original_image(self, record: OriginalImageRecord) -> None:
        if record.image_id not in self.original_images:
            self.original_order.append(record.image_id)
        self.original_images[record.image_id] = record

    def unregister_original_image(self, image_id: str) -> None:
        self.original_images.pop(image_id, None)
        self.root_nodes_by_image_id.pop(image_id, None)
        if image_id in self.original_order:
            self.original_order.remove(image_id)

    def get_original_image_record(self, image_id: str) -> Optional[OriginalImageRecord]:
        return self.original_images.get(image_id)

    def iter_original_image_ids(self) -> Iterable[str]:
        return list(self.original_order)

    # ------------------------------------------------------------------ History nodes
    def add_node(self, node: HistoryNode) -> None:
        if not node.original_image_id:
            self._assign_original_image_for_node(node)

        self.history[node.node_id] = node
        if node.parent_id is None or node.operation_name == "Original":
            self._finalise_root_registration(node)

    def _assign_original_image_for_node(self, node: HistoryNode) -> None:
        if node.original_image_id:
            return
        if node.parent_id and node.parent_id in self.history:
            node.original_image_id = self.history[node.parent_id].original_image_id
            return
        legacy_id = str(uuid.uuid4())
        display_name = node.parameters.get("original_label") or self._generate_default_display_name()
        node.parameters["original_label"] = display_name
        node.original_image_id = legacy_id
        legacy_record = OriginalImageRecord(
            image_id=legacy_id,
            display_name=display_name,
            stm_image=None,
            source_path=node.parameters.get("filename"),
            extra_metadata=dict(node.parameters),
        )
        self.register_original_image(legacy_record)
        self.root_nodes_by_image_id[legacy_id] = node.node_id

    def _generate_default_display_name(self) -> str:
        next_index = len(self.original_order) + 1
        return f"Original Image {next_index}"

    def get_next_original_display_name(self) -> str:
        return self._generate_default_display_name()

    def _finalise_root_registration(self, node: HistoryNode) -> None:
        if not node.original_image_id:
            self._assign_original_image_for_node(node)
        image_id = node.original_image_id
        if image_id is None:
            return
        if image_id not in self.original_images:
            display_name = node.parameters.get("original_label") or self._generate_default_display_name()
            node.parameters["original_label"] = display_name
            node.parameters["source_image_label"] = display_name
            record = OriginalImageRecord(
                image_id=image_id,
                display_name=display_name,
                stm_image=None,
                source_path=node.parameters.get("filename"),
                extra_metadata=dict(node.parameters),
            )
            self.register_original_image(record)
        else:
            record = self.original_images[image_id]
            node.parameters.setdefault("original_label", record.display_name)
            node.parameters.setdefault("source_image_label", record.display_name)
            record.extra_metadata = dict(node.parameters)
        self.root_nodes_by_image_id[image_id] = node.node_id

    # ------------------------------------------------------------------ Utilities
    def rebuild_indexes(self) -> None:
        self.root_nodes_by_image_id.clear()
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

        self.original_order = []
        for node in root_candidates:
            if node.parent_id is not None and node.operation_name != "Original":
                continue
            if node.original_image_id is None:
                node.original_image_id = str(uuid.uuid4())
            image_id = node.original_image_id
            display_name = node.parameters.get("original_label") or self._generate_default_display_name()
            node.parameters["original_label"] = display_name

            if image_id not in self.original_images or image_id not in known_ids:
                record = OriginalImageRecord(
                    image_id=image_id,
                    display_name=display_name,
                    stm_image=None,
                    source_path=node.parameters.get("filename"),
                    extra_metadata=dict(node.parameters),
                )
                self.original_images[image_id] = record
            else:
                record = self.original_images[image_id]
                record.extra_metadata = dict(node.parameters)
            self.original_order.append(image_id)
            self.root_nodes_by_image_id[image_id] = node.node_id

        for node in self.history.values():
            if not node.original_image_id:
                continue
            record = self.original_images.get(node.original_image_id)
            if not record:
                continue
            node.parameters.setdefault("source_image_label", record.display_name)
            if node.operation_name == "Original":
                node.parameters.setdefault("original_label", record.display_name)

    def get_node(self, node_id: str) -> Optional[HistoryNode]:
        return self.history.get(node_id)

    def get_root_node_for_node(self, node_id: Optional[str]) -> Optional[HistoryNode]:
        if not node_id:
            return None
        current_node = self.history.get(node_id)
        if not current_node:
            return None
        if current_node.original_image_id:
            root_id = self.root_nodes_by_image_id.get(current_node.original_image_id)
            if root_id:
                return self.history.get(root_id)
        visited_ids = {current_node.node_id}
        for _ in range(100):
            if not current_node.parent_id:
                return current_node
            if current_node.operation_name == "Original":
                return current_node
            parent = self.history.get(current_node.parent_id)
            if not parent or parent.node_id in visited_ids:
                return current_node
            current_node = parent
            visited_ids.add(current_node.node_id)
        return current_node

    def delete_node_branch(self, node_id: str) -> Dict[str, Optional[str]]:
        node = self.history.get(node_id)
        if not node:
            raise KeyError(f"Node {node_id} not found in history.")

        descendants = self._collect_descendants(node_id)
        parent_id = node.parent_id
        original_image_id = node.original_image_id
        is_root = node.operation_name == "Original" or node.parent_id is None

        for removed_id in descendants:
            self.history.pop(removed_id, None)

        if is_root and original_image_id:
            self.unregister_original_image(original_image_id)

        for image_key, root_node_id in list(self.root_nodes_by_image_id.items()):
            if root_node_id in descendants:
                self.root_nodes_by_image_id.pop(image_key, None)

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
        return {
            "deleted_node_ids": descendants,
            "removed_original_image_id": original_image_id if is_root else None,
            "new_current_node_id": new_current_id,
        }

    def delete_original_image_branch(self, image_id: str) -> Dict[str, Optional[str]]:
        root_node_id = self.root_nodes_by_image_id.get(image_id)
        if root_node_id:
            return self.delete_node_branch(root_node_id)

        for node in self.history.values():
            if node.original_image_id == image_id and (node.parent_id is None or node.operation_name == "Original"):
                return self.delete_node_branch(node.node_id)
        raise KeyError(f"Original image {image_id} not found.")

    def _collect_descendants(self, node_id: str) -> List[str]:
        descendants: List[str] = []
        stack = [node_id]
        while stack:
            current_id = stack.pop()
            node = self.history.get(current_id)
            if not node:
                continue
            descendants.append(current_id)
            for candidate in self.history.values():
                if candidate.parent_id == current_id:
                    stack.append(candidate.node_id)
        return descendants

    def set_current_node(self, node_id: Optional[str]) -> None:
        if node_id is not None and node_id not in self.history:
            raise KeyError(f"Node {node_id} not found in history.")
        self.current_node_id = node_id
