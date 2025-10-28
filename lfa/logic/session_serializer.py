from __future__ import annotations

import logging
import os
import pickle
import uuid
from typing import Any, Dict, List, Optional

from ..core.data_models import OriginalImageRecord

logger = logging.getLogger(__name__)


class SessionSerializer:
    """Handles serialisation and restoration of controller + history state."""

    FORMAT_VERSION = "1.1"

    def __init__(self, history_manager) -> None:
        self.history_manager = history_manager

    # ------------------------------------------------------------------ Saving
    def build_session_payload(self, controller) -> Dict[str, Any]:
        """Collect controller and history state for persistence."""
        controller_state = self._collect_controller_state(controller)
        history_payload = self._build_history_payload()
        return {
            "format_version": self.FORMAT_VERSION,
            "history_data": history_payload,
            "controller_state": controller_state,
        }

    def _collect_controller_state(self, controller) -> Dict[str, Any]:
        return {
            "original_file_path": controller.original_file_path,
            "spot_selection_mode": controller.spot_selection_mode,
            "spot_refinement_method": controller.spot_refinement_method,
            "refinement_roi_size": controller.refinement_roi_size,
            "user_selected_substrate_spots": controller.user_selected_substrate_spots,
            "substrate_lattice_type": controller.substrate_lattice_type,
            "substrate_a_surf": controller.substrate_a_surf,
            "substrate_definition_name": controller.substrate_definition_name,
            "substrate_F_m2i": controller.substrate_F_m2i,
            "substrate_t_m2i": controller.substrate_t_m2i,
            "substrate_transform_analysis_m2i": controller.substrate_transform_analysis_m2i,
            "displayable_fitted_substrate_spots_on_fft": controller.displayable_fitted_substrate_spots_on_fft,
            "substrate_spot_pairs": [
                {"raw": raw, "transformed": transformed}
                for raw, transformed in controller.substrate_spot_pairs
            ],
            "show_substrate_raw_spots": controller.show_substrate_raw_spots,
            "show_substrate_transformed_spots": controller.show_substrate_transformed_spots,
            "show_adsorbate_raw_spots": controller.show_adsorbate_raw_spots,
            "show_adsorbate_transformed_spots": controller.show_adsorbate_transformed_spots,
            "show_substrate_spots_markers": controller.show_substrate_spots_markers,
            "show_adsorbate_spots_markers": controller.show_adsorbate_spots_markers,
            "substrate_real_space_results": controller.substrate_real_space_results,
            "adsorbate_spot_sets": controller.adsorbate_spot_sets,
            "corrected_adsorbate_spot_sets": controller.corrected_adsorbate_spot_sets,
            "current_adsorbate_set_index": controller.current_adsorbate_set_index,
            "adsorbate_real_space_results": controller.adsorbate_real_space_results,
            "adsorbate_expected_lattice_types": controller.adsorbate_expected_lattice_types,
            "superstructure_periodicity_results": controller.superstructure_periodicity_results,
            "substrate_visual_offset_nm": controller.substrate_visual_offset_nm,
            "adsorbate_visual_offsets_nm": controller.adsorbate_visual_offsets_nm,
            "adsorbate_spot_pairs": {
                index: [
                    {"raw": raw, "transformed": transformed}
                    for raw, transformed in pairs
                ]
                for index, pairs in controller.adsorbate_spot_pairs.items()
            },
        }

    def _build_history_payload(self) -> Dict[str, Any]:
        original_images_payload: List[Dict[str, Any]] = []
        original_order_payload: List[str] = []
        if hasattr(self.history_manager, "iter_original_image_ids"):
            for image_id in self.history_manager.iter_original_image_ids():
                record = self.history_manager.get_original_image_record(image_id)
                if not record:
                    continue
                original_images_payload.append(
                    {
                        "image_id": image_id,
                        "display_name": record.display_name,
                        "source_path": record.source_path,
                        "extra_metadata": record.extra_metadata,
                    }
                )
                original_order_payload.append(image_id)

        return {
            "tree": self.history_manager.history,
            "current_node_id": self.history_manager.current_node_id,
            "original_images": original_images_payload,
            "original_order": original_order_payload,
        }

    @staticmethod
    def dump_to_file(file_path: str, payload: Dict[str, Any]) -> None:
        with open(file_path, "wb") as handle:
            pickle.dump(payload, handle)

    # ------------------------------------------------------------------ Loading
    @staticmethod
    def load_from_file(file_path: str) -> Dict[str, Any]:
        with open(file_path, "rb") as handle:
            return pickle.load(handle)

    def restore_session(self, controller, session_data: Dict[str, Any]) -> None:
        """Apply a session payload to the controller and history manager."""
        format_version = session_data.get("format_version", "1.0")
        if format_version not in {"1.0", self.FORMAT_VERSION}:
            raise ValueError(f"The session file uses an unsupported format version: {format_version}")

        controller_state = session_data.get("controller_state", {}) or {}
        history_data = session_data.get("history_data", {}) or {}

        self._restore_controller_state(controller, controller_state)
        self._restore_history_state(controller, history_data, format_version)

    def _restore_controller_state(self, controller, state: Dict[str, Any]) -> None:
        if 'domain_wall_analysis_results' in state and 'superstructure_periodicity_results' not in state:
            state = dict(state)
            state['superstructure_periodicity_results'] = state.get('domain_wall_analysis_results')

        controller.original_file_path = state.get("original_file_path")
        controller.spot_selection_mode = state.get("spot_selection_mode", controller.spot_selection_mode)
        controller.spot_refinement_method = state.get("spot_refinement_method", controller.spot_refinement_method)
        controller.refinement_roi_size = state.get("refinement_roi_size", controller.refinement_roi_size)
        controller.user_selected_substrate_spots = state.get("user_selected_substrate_spots", [])
        controller.substrate_lattice_type = state.get("substrate_lattice_type")
        controller.substrate_a_surf = state.get("substrate_a_surf")
        controller.substrate_definition_name = state.get("substrate_definition_name", controller.substrate_definition_name)
        controller.substrate_F_m2i = state.get("substrate_F_m2i")
        controller.substrate_t_m2i = state.get("substrate_t_m2i")
        controller.substrate_transform_analysis_m2i = state.get("substrate_transform_analysis_m2i")
        controller.displayable_fitted_substrate_spots_on_fft = state.get(
            "displayable_fitted_substrate_spots_on_fft", []
        )

        substrate_pairs = state.get("substrate_spot_pairs") or []
        controller.substrate_spot_pairs = [
            (tuple(pair.get("raw", (0.0, 0.0))), tuple(pair.get("transformed", (0.0, 0.0))))
            for pair in substrate_pairs
        ]

        controller.show_substrate_raw_spots = state.get("show_substrate_raw_spots", True)
        controller.show_substrate_transformed_spots = state.get("show_substrate_transformed_spots", True)
        controller.show_adsorbate_raw_spots = state.get("show_adsorbate_raw_spots", True)
        controller.show_adsorbate_transformed_spots = state.get("show_adsorbate_transformed_spots", True)
        controller.show_substrate_spots_markers = state.get("show_substrate_spots_markers", True)
        controller.show_adsorbate_spots_markers = state.get("show_adsorbate_spots_markers", True)
        controller.substrate_real_space_results = state.get("substrate_real_space_results")
        controller.adsorbate_spot_sets = state.get("adsorbate_spot_sets", [[]])
        controller.corrected_adsorbate_spot_sets = state.get("corrected_adsorbate_spot_sets", [[]])
        controller.current_adsorbate_set_index = state.get("current_adsorbate_set_index", 0)
        controller.adsorbate_real_space_results = state.get("adsorbate_real_space_results", {})
        controller.adsorbate_expected_lattice_types = state.get(
            "adsorbate_expected_lattice_types", {0: "Unknown"}
        )
        controller.superstructure_periodicity_results = state.get("superstructure_periodicity_results")
        controller.substrate_visual_offset_nm = _normalise_offset(state.get("substrate_visual_offset_nm", (0.0, 0.0)))

        ads_offsets = state.get("adsorbate_visual_offsets_nm", {})
        controller.adsorbate_visual_offsets_nm = {
            int(k): _normalise_offset(v)
            for k, v in ads_offsets.items()
        } if isinstance(ads_offsets, dict) else {0: (0.0, 0.0)}

        raw_adsorbate_pairs = state.get("adsorbate_spot_pairs") or {}
        converted_adsorbate_pairs: Dict[int, List[Any]] = {}
        if isinstance(raw_adsorbate_pairs, dict):
            for index_key, pairs in raw_adsorbate_pairs.items():
                try:
                    index = int(index_key)
                except (TypeError, ValueError):
                    continue
                converted_adsorbate_pairs[index] = [
                    (tuple(pair.get("raw", (0.0, 0.0))), tuple(pair.get("transformed", (0.0, 0.0))))
                    for pair in pairs
                ]
        controller.adsorbate_spot_pairs = converted_adsorbate_pairs or {0: []}

        controller.set_substrate_raw_visibility(controller.show_substrate_raw_spots)
        controller.set_substrate_transformed_visibility(controller.show_substrate_transformed_spots)
        controller.set_adsorbate_raw_visibility(controller.show_adsorbate_raw_spots)
        controller.set_adsorbate_transformed_visibility(controller.show_adsorbate_transformed_spots)

    def _restore_history_state(self, controller, history_data: Dict[str, Any], format_version: str) -> None:
        self.history_manager.history = history_data.get("tree", {}) or {}

        if hasattr(self.history_manager, "original_images"):
            self.history_manager.original_images.clear()
        if hasattr(self.history_manager, "_original_order"):
            self.history_manager._original_order = []
        if hasattr(self.history_manager, "_root_nodes_by_image_id"):
            self.history_manager._root_nodes_by_image_id.clear()

        if format_version == "1.1":
            stored_records = history_data.get("original_images", []) or []
            for record_data in stored_records:
                image_id = record_data.get("image_id") or str(uuid.uuid4())
                record = OriginalImageRecord(
                    image_id=image_id,
                    display_name=record_data.get("display_name", "Original Image"),
                    stm_image=None,
                    source_path=record_data.get("source_path"),
                    extra_metadata=record_data.get("extra_metadata", {}),
                )
                if hasattr(self.history_manager, "register_original_image"):
                    self.history_manager.register_original_image(record)

        if hasattr(self.history_manager, "rebuild_indexes"):
            self.history_manager.rebuild_indexes()

        if format_version == "1.1":
            stored_order = history_data.get("original_order", []) or []
            filtered_order = [
                img_id for img_id in stored_order
                if img_id in getattr(self.history_manager, "original_images", {})
            ]
            for img_id in getattr(self.history_manager, "original_images", {}):
                if img_id not in filtered_order:
                    filtered_order.append(img_id)
            if hasattr(self.history_manager, "_original_order"):
                self.history_manager._original_order = filtered_order

        if hasattr(self.history_manager, "refresh_widget"):
            self.history_manager.refresh_widget()

        self._ensure_history_nodes_have_ids()

        current_id = history_data.get("current_node_id")
        self.history_manager.set_current_node_by_id(current_id, emit_signal=False)

        file_name = os.path.basename(controller.original_file_path or "Loaded Session")
        controller.file_loaded_successfully.emit(file_name)
        controller.adsorbate_sets_structure_changed.emit()
        controller.substrate_definition_changed.emit()
        controller.substrate_transform_results_updated.emit()
        controller.superstructure_periodicity_results_updated.emit(
            controller.superstructure_periodicity_results
        )

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


def _normalise_offset(offset: Optional[Any]) -> tuple[float, float]:
    if isinstance(offset, (list, tuple)) and len(offset) == 2:
        return float(offset[0]), float(offset[1])
    return (0.0, 0.0)

