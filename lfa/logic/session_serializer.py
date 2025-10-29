from __future__ import annotations

import logging
import os
import pickle
import uuid
from typing import Any, Dict

from ..core.data_models import OriginalImageRecord
from .session_migrations import CURRENT_SESSION_VERSION, migrate_payload
from .session_state import ControllerState, HistoryState, SessionState

logger = logging.getLogger(__name__)


class SessionSerializer:
    """Handles serialisation and restoration of controller + history state."""

    FORMAT_VERSION = CURRENT_SESSION_VERSION

    def __init__(self, history_manager) -> None:
        self.history_manager = history_manager

    # ------------------------------------------------------------------ Saving
    def build_session_state(self, controller) -> SessionState:
        if hasattr(controller, "export_session_state"):
            try:
                controller_state = controller.export_session_state()
                if not isinstance(controller_state, ControllerState):
                    raise TypeError
            except Exception:  # pragma: no cover - fallback
                controller_state = ControllerState.from_controller(controller)
        else:
            controller_state = ControllerState.from_controller(controller)

        if hasattr(self.history_manager, "export_session_state"):
            try:
                history_state = self.history_manager.export_session_state()
                if not isinstance(history_state, HistoryState):
                    raise TypeError
            except Exception:  # pragma: no cover - fallback
                history_state = HistoryState.from_history_manager(self.history_manager)
        else:
            history_state = HistoryState.from_history_manager(self.history_manager)

        return SessionState(
            format_version=self.FORMAT_VERSION,
            controller=controller_state,
            history=history_state,
        )

    def build_session_payload(self, controller) -> Dict[str, Any]:
        return self.build_session_state(controller).to_payload()

    @staticmethod
    def dump_to_file(file_path: str, session_state: SessionState | Dict[str, Any]) -> None:
        payload = session_state.to_payload() if isinstance(session_state, SessionState) else session_state
        with open(file_path, "wb") as handle:
            pickle.dump(payload, handle)

    # ------------------------------------------------------------------ Loading
    @classmethod
    def load_from_file(cls, file_path: str) -> SessionState:
        with open(file_path, "rb") as handle:
            payload = pickle.load(handle)

        if not isinstance(payload, dict):
            raise ValueError("Unsupported session file format.")

        migrated = migrate_payload(payload)
        return SessionState.from_payload(migrated)

    def restore_session(self, controller, session_state: SessionState | Dict[str, Any]) -> None:
        if isinstance(session_state, dict):
            migrated = migrate_payload(session_state)
            session_state = SessionState.from_payload(migrated)

        logger.debug("Applying controller state from session...")
        if hasattr(controller, "load_session_state"):
            controller.load_session_state(session_state.controller)
        else:
            session_state.controller.apply_to(controller)

        logger.debug("Applying history state from session...")
        if hasattr(self.history_manager, "load_session_state"):
            self.history_manager.load_session_state(session_state.history)
        else:
            session_state.history.apply_to(self.history_manager)

        self._ensure_history_nodes_have_ids()

        file_name = os.path.basename(controller.original_file_path or "Loaded Session")
        controller.file_loaded_successfully.emit(file_name)
        controller.adsorbate_sets_structure_changed.emit()
        controller.substrate_definition_changed.emit()
        controller.substrate_transform_results_updated.emit()
        controller.superstructure_periodicity_results_updated.emit(
            controller.superstructure_periodicity_results
        )

    # ------------------------------------------------------------------ Helpers
    def _ensure_history_nodes_have_ids(self) -> None:
        if not getattr(self.history_manager, "history", None):
            return

        get_root = getattr(self.history_manager, "get_root_node_for_node", None)
        for node in self.history_manager.history.values():
            if getattr(node, "original_image_id", None):
                continue

            root_node = get_root(node.node_id) if callable(get_root) else None
            target_node = root_node or node

            if getattr(target_node, "original_image_id", None) is None:
                display_name = target_node.parameters.get("original_label")
                if not display_name and hasattr(self.history_manager, "get_next_original_display_name"):
                    display_name = self.history_manager.get_next_original_display_name()
                display_name = display_name or "Original Image"

                image_id = str(uuid.uuid4())
                record = OriginalImageRecord(
                    image_id=image_id,
                    display_name=display_name,
                    stm_image=None,
                    source_path=target_node.parameters.get("filename"),
                    extra_metadata=dict(target_node.parameters),
                )
                if hasattr(self.history_manager, "register_original_image"):
                    self.history_manager.register_original_image(record)
                if hasattr(self.history_manager, "_root_nodes_by_image_id"):
                    self.history_manager._root_nodes_by_image_id[image_id] = target_node.node_id
                target_node.parameters.setdefault("original_label", display_name)
                target_node.original_image_id = image_id

            node.original_image_id = target_node.original_image_id
