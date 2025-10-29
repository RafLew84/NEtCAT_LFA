from __future__ import annotations

import logging
import os
from typing import Optional

from PyQt6.QtWidgets import QFileDialog, QMessageBox

from ...core.constants import ADSORBATE_LATTICE_TYPE_UNKNOWN, PREDEFINED_SUBSTRATE_NONE
from ...core.data_models import OriginalImageRecord
from ...core.history import HistoryNode
from ..session_serializer import SessionSerializer
from .history_orchestrator import HistoryOrchestrator

logger = logging.getLogger(__name__)


class SessionService:
    """
    Handles high-level session lifecycle operations (save/load/reset) so
    that :class:`AppController` focuses on orchestration rather than I/O.
    """

    def __init__(
        self,
        controller: "AppController",
        history_service: HistoryOrchestrator,
    ) -> None:
        self._controller = controller
        self._history = history_service

    # ------------------------------------------------------------------ Save/Load
    def save_session(self) -> None:
        controller = self._controller
        if not self._history.get_current_node():
            logger.warning("Attempted to save an empty session. Aborted.")
            QMessageBox.information(None, "Save cancelled", "No active analysis is available to save.")
            return

        if controller.original_file_path:
            base_name = os.path.basename(controller.original_file_path)
            suggested_name = os.path.splitext(base_name)[0] + ".lfa_proj"
        else:
            suggested_name = "analysis.lfa_proj"

        files_filter = "LFA Project Files (*.lfa_proj);;All Files (*)"
        file_path, _ = QFileDialog.getSaveFileName(None, "Save analysis session", suggested_name, files_filter)
        if not file_path:
            logger.info("Session save was cancelled by the user.")
            return

        logger.debug("Collecting session data for serialization...")
        session_state = controller.session_serializer.build_session_state(controller)
        try:
            SessionSerializer.dump_to_file(file_path, session_state)
            logger.info("Analysis session saved successfully at: %s", file_path)
            QMessageBox.information(
                None,
                "Saved",
                f"Sesja zostala pomyslnie zapisana w pliku:\n{os.path.basename(file_path)}",
            )
        except Exception as exc:  # pragma: no cover - GUI feedback path
            logger.exception("Critical error while saving the session file: %s", exc)
            QMessageBox.critical(None, "Save error", f"Wystapil blad podczas zapisu pliku:\n{exc}")

    def load_session(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            None, "Load analysis session", "", "LFA Project Files (*.lfa_proj);;All Files (*)"
        )
        if not file_path:
            logger.info("Session load was cancelled by the user.")
            return

        try:
            session_state = SessionSerializer.load_from_file(file_path)
        except Exception as exc:  # pragma: no cover - GUI feedback path
            logger.exception("Critical error while loading the session file: %s", exc)
            return

        logger.info("Restoring controller state from session file...")
        self._history.clear_history()
        self._controller.clear_all_spot_data()

        try:
            controller = self._controller
            controller.session_serializer.restore_session(controller, session_state)
        except ValueError as exc:
            logger.warning("Attempted to load an incompatible session: %s", exc)
            QMessageBox.warning(None, "Version error", str(exc))
        except Exception as exc:  # pragma: no cover - GUI feedback path
            logger.exception("Failed to restore session: %s", exc)
            QMessageBox.critical(None, "Load error", f"Could not load session:\n{exc}")

    # ------------------------------------------------------------------ Reset
    def reset_session(self) -> None:
        controller = self._controller
        logger.info("SessionService: Resetting current analysis session.")

        self._history.clear_history()
        controller.clear_all_spot_data()

        controller.original_file_path = None
        controller.reference_ideal_substrate_spots_px.clear()
        controller.custom_lattice_info = None
        controller.last_selected_substrate = PREDEFINED_SUBSTRATE_NONE
        controller.current_substrate_a_surf = None
        controller.current_substrate_type = None
        controller.current_substrate_name = PREDEFINED_SUBSTRATE_NONE
        controller.substrate_definition_name = PREDEFINED_SUBSTRATE_NONE
        controller.substrate_lattice_type = None
        controller.substrate_a_surf = None
        controller.substrate_F_m2i = None
        controller.substrate_t_m2i = None
        controller.substrate_transform_analysis_m2i = None
        controller.displayable_fitted_substrate_spots_on_fft.clear()
        controller.show_ideal_lattice = True
        controller.current_fft_data_shape = None
        controller.user_selected_substrate_spots.clear()
        controller.substrate_visual_offset_nm = (0.0, 0.0)
        controller.adsorbate_visual_offsets_nm = {0: (0.0, 0.0)}

        self._history.refresh_widget()

        controller.substrate_definition_changed.emit()
        controller.substrate_transform_results_updated.emit()
        controller.substrate_real_space_params_updated.emit({})
        controller.adsorbate_sets_structure_changed.emit()
        controller.adsorbate_set_updated.emit(0)
        controller.adsorbate_real_space_params_updated.emit(0, {})
        controller.adsorbate_expected_type_updated.emit(0, ADSORBATE_LATTICE_TYPE_UNKNOWN)
        controller.superstructure_periodicity_results_updated.emit(None)
        controller.spot_lists_updated.emit()

        logger.info("SessionService: Session reset complete.")

    # ------------------------------------------------------------------ Helpers for load_file
    def register_new_original(self, record: OriginalImageRecord, root_node: HistoryNode) -> None:
        """
        Register a freshly loaded STM image as an original record/node.
        """
        self._history.register_original_image(record)
        self._history.add_node_and_select(root_node)
