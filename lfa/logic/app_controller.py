# lfa/logic/app_controller.py
"""
Central controller for the LFA application.
Manages application state and coordinates operations between UI and backend modules.
"""
import logging
import os
import pickle
import uuid
import numpy as np

from typing import Optional, List, Tuple, Dict, Any

from PyQt6.QtCore import QObject, pyqtSignal

from ..core.data_models import STMImage, OriginalImageRecord 
from ..io.factory import load_stm_file  
from ..core.history import HistoryNode  
from ..gui.dialogs.substrate_spot_dialog import PREDEFINED_SUBSTRATE_NONE, PREDEFINED_SUBSTRATE_CUSTOM

from PyQt6.QtWidgets import QFileDialog, QMessageBox
from PyQt6.QtCore import Qt


logger = logging.getLogger(__name__)

SPOT_SELECTION_SUBSTRATE = "Substrate"
SPOT_SELECTION_ADSORBATE = "Adsorbate"

REFINEMENT_DIRECT_CLICK = "Direct Click"
REFINEMENT_MAX_PIXEL = "Max Pixel"
REFINEMENT_GAUSSIAN_FIT = "2D Gaussian Fit"

ADSORBATE_LATTICE_TYPE_UNKNOWN = "Unknown"
ADSORBATE_LATTICE_TYPE_HEXAGONAL = "Hexagonal"
ADSORBATE_LATTICE_TYPE_SQUARE = "Square"

MAX_SUBSTRATE_SPOTS = 8

try:
    from ..analysis.lattice import (
        get_real_space_lattice_parameters, 
        calculate_real_space_vectors_from_g, 
        convert_g_vector_px_to_nm_inv, 
        select_adsorbate_reciprocal_basis_vectors_px,
        LATTICE_TYPE_HEXAGONAL, LATTICE_TYPE_SQUARE, 
    )
    LATTICE_ANALYSIS_FUNCTIONS_AVAILABLE = True
except ImportError: # pragma: no cover
    logging.error("AppController: Could not import lattice analysis functions.")
    LATTICE_ANALYSIS_FUNCTIONS_AVAILABLE = False
    def get_real_space_lattice_parameters(*args, **kwargs): return None

class AppController(QObject):
    """
    Manages the core application state and logic, acting as a bridge
    between the GUI (MainWindow) and the backend processing/analysis modules.
    """

    file_loaded_successfully = pyqtSignal(str)
    file_loading_failed = pyqtSignal(str)   

    spot_lists_updated = pyqtSignal()
    adsorbate_set_updated = pyqtSignal(int)
    spot_selection_parameters_changed = pyqtSignal()
    adsorbate_sets_structure_changed = pyqtSignal()
    substrate_transform_results_updated = pyqtSignal()
    superstructure_periodicity_results_updated = pyqtSignal(object)

    substrate_definition_changed = pyqtSignal()

    substrate_real_space_params_updated = pyqtSignal(dict)
    adsorbate_real_space_params_updated = pyqtSignal(int, dict)

    adsorbate_expected_type_updated = pyqtSignal(int, str)

    substrate_raw_visibility_updated = pyqtSignal(bool)
    substrate_transformed_visibility_updated = pyqtSignal(bool)
    adsorbate_raw_visibility_updated = pyqtSignal(bool)
    adsorbate_transformed_visibility_updated = pyqtSignal(bool)

    def __init__(self, history_manager, parent: Optional[QObject] = None):
        """
        Initializes the AppController.

        Args:
            history_manager (HistoryManager): Instance of the history manager.
            parent (Optional[QObject]): Parent object for Qt memory management.
        """
        super().__init__(parent)
        
        self.history_manager = history_manager 

        self.original_file_path: Optional[str] = None

        self.substrate_spots: List[Tuple[float, float]] = []
        self.adsorbate_spot_sets: List[List[Tuple[float, float]]] = [[]] 
        self.corrected_adsorbate_spot_sets: List[List[Tuple[float, float]]] = [[]]
        self.current_adsorbate_set_index: int = 0

        self.spot_selection_mode: str = "Substrate"  
        self.spot_refinement_method: str = "Direct Click"
        self.refinement_roi_size: int = 5 
        self.reference_ideal_substrate_spots_px: List[Tuple[float, float]] = []

        self.custom_lattice_info: Optional[Dict[str, Any]] = None 
        self.last_selected_substrate: str = "None"
        
        self.show_ideal_lattice: bool = True 

        self.current_substrate_a_surf: Optional[float] = None 
        self.current_substrate_type: Optional[str] = None  
        self.current_substrate_name: str = PREDEFINED_SUBSTRATE_NONE

        self.user_selected_substrate_spots: List[Tuple[float, float]] = [] 

        self.substrate_lattice_type: Optional[str] = None
        self.substrate_a_surf: Optional[float] = None
        self.substrate_definition_name: str = PREDEFINED_SUBSTRATE_NONE 

        self.substrate_F_m2i: Optional[np.ndarray] = None 
        self.substrate_t_m2i: Optional[np.ndarray] = None 
        self.substrate_transform_analysis_m2i: Optional[Dict[str, Any]] = None
        
        self.displayable_fitted_substrate_spots_on_fft: List[Tuple[float, float]] = []
        self.substrate_spot_pairs: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []

        self.show_fitted_substrate_spots: bool = True
        self.show_substrate_raw_spots: bool = True
        self.show_substrate_transformed_spots: bool = True
        self.show_adsorbate_raw_spots: bool = True
        self.show_adsorbate_transformed_spots: bool = True
        self.show_substrate_spots_markers: bool = True
        self.show_adsorbate_spots_markers: bool = True
        self.adsorbate_spot_pairs: Dict[int, List[Tuple[Tuple[float, float], Tuple[float, float]]]] = {0: []}

        self.current_fft_data_shape: Optional[Tuple[int, int]] = None
        self.substrate_real_space_results: Optional[Dict[str, Any]] = None
        self.adsorbate_real_space_results: Dict[int, Dict[str, Any]] = {}
        self.adsorbate_expected_lattice_types: Dict[int, str] = {0: ADSORBATE_LATTICE_TYPE_UNKNOWN}

        self.superstructure_periodicity_results: Optional[Dict[str, Any]] = None

        logger.info("AppController initialized.")

    def save_analysis_session(self):
        if not self.history_manager.get_current_node():
            logger.warning("Attempted to save an empty session. Aborted.")
            QMessageBox.information(None, "Save cancelled", "No active analysis is available to save.")
            return

        if self.original_file_path:
            base_name = os.path.basename(self.original_file_path)
            suggested_name = os.path.splitext(base_name)[0] + ".lfa_proj"
        else:
            suggested_name = "analysis.lfa_proj"

        file_path, _ = QFileDialog.getSaveFileName(
            None, "Save analysis session", suggested_name, "LFA Project Files (*.lfa_proj);;All Files (*)"
        )

        if not file_path:
            logger.info("Session save was cancelled by the user.")
            return

        logger.debug("Collecting session data for serialization...")
        
        controller_state_to_save = {
            'original_file_path': self.original_file_path,
            'spot_selection_mode': self.spot_selection_mode,
            'spot_refinement_method': self.spot_refinement_method,
            'refinement_roi_size': self.refinement_roi_size,
            'user_selected_substrate_spots': self.user_selected_substrate_spots,
            'substrate_lattice_type': self.substrate_lattice_type,
            'substrate_a_surf': self.substrate_a_surf,
            'substrate_definition_name': self.substrate_definition_name,
            'substrate_F_m2i': self.substrate_F_m2i,
            'substrate_t_m2i': self.substrate_t_m2i,
            'substrate_transform_analysis_m2i': self.substrate_transform_analysis_m2i,
            'displayable_fitted_substrate_spots_on_fft': self.displayable_fitted_substrate_spots_on_fft,
            'substrate_spot_pairs': [
                {'raw': raw, 'transformed': transformed}
                for raw, transformed in self.substrate_spot_pairs
            ],
            'show_substrate_raw_spots': self.show_substrate_raw_spots,
            'show_substrate_transformed_spots': self.show_substrate_transformed_spots,
            'show_adsorbate_raw_spots': self.show_adsorbate_raw_spots,
            'show_adsorbate_transformed_spots': self.show_adsorbate_transformed_spots,
            'show_substrate_spots_markers': self.show_substrate_spots_markers,
            'show_adsorbate_spots_markers': self.show_adsorbate_spots_markers,
            'substrate_real_space_results': self.substrate_real_space_results,
            'adsorbate_spot_sets': self.adsorbate_spot_sets,
            'corrected_adsorbate_spot_sets': self.corrected_adsorbate_spot_sets,
            'current_adsorbate_set_index': self.current_adsorbate_set_index,
            'adsorbate_real_space_results': self.adsorbate_real_space_results,
            'adsorbate_expected_lattice_types': self.adsorbate_expected_lattice_types,
            'superstructure_periodicity_results': self.superstructure_periodicity_results,
            'adsorbate_spot_pairs': {
                index: [
                    {'raw': raw, 'transformed': transformed}
                    for raw, transformed in pairs
                ]
                for index, pairs in self.adsorbate_spot_pairs.items()
            },
        }

        original_images_payload = []
        original_order_payload: List[str] = []
        if hasattr(self.history_manager, "iter_original_image_ids"):
            for image_id in self.history_manager.iter_original_image_ids():
                record = self.history_manager.get_original_image_record(image_id)
                if not record:
                    continue
                original_images_payload.append({
                    "image_id": image_id,
                    "display_name": record.display_name,
                    "source_path": record.source_path,
                    "extra_metadata": record.extra_metadata,
                })
                original_order_payload.append(image_id)
        session_data = {
            "format_version": "1.1",
            "history_data": {
                "tree": self.history_manager.history,
                "current_node_id": self.history_manager.current_node_id,
                "original_images": original_images_payload,
                "original_order": original_order_payload,
            },
            "controller_state": controller_state_to_save
        }

        try:
            with open(file_path, 'wb') as f:
                pickle.dump(session_data, f)
            logger.info(f"Analysis session saved successfully at: {file_path}")
            QMessageBox.information(None, "Saved", f"Sesja została pomyślnie zapisana w pliku:\n{os.path.basename(file_path)}")
        except Exception as e:
            logger.exception(f"Critical error while saving the session file: {e}")
            QMessageBox.critical(None, "Save error", f"Wystąpił błąd podczas zapisu pliku:\n{e}")

    def load_analysis_session(self):
        file_path, _ = QFileDialog.getOpenFileName(
            None, "Load analysis session", "", "LFA Project Files (*.lfa_proj);;All Files (*)"
        )

        if not file_path:
            logger.info("Session load was cancelled by the user.")
            return

        # 1. Load session data from file
        try:
            with open(file_path, 'rb') as f:
                session_data = pickle.load(f)

            format_version = session_data.get("format_version", "1.0")
            if format_version not in {"1.0", "1.1"}:
                logger.warning("Attempted to load a session file with an incompatible format version: %s", format_version)
                QMessageBox.warning(None, "Version error", "The session file uses an unsupported format version.")
                return

        except Exception as e:
            logger.exception(f"Critical error while loading the session file: {e}")
            return

        logger.info("Restoring controller state from session file...")

        self.history_manager.clear_history()
        self.clear_all_spot_data()

        loaded_state = session_data.get("controller_state", {})
        legacy_superstructure_results = loaded_state.get("domain_wall_analysis_results")
        if legacy_superstructure_results is not None and "superstructure_periodicity_results" not in loaded_state:
            logger.info("Migrating legacy domain wall analysis results to superstructure periodicity results.")
            self.superstructure_periodicity_results = legacy_superstructure_results

        for key, value in loaded_state.items():
            if key == "domain_wall_analysis_results":
                continue
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                logger.warning(f"Attribute '{key}' from the session file no longer exists on AppController.")

        raw_substrate_pairs = loaded_state.get("substrate_spot_pairs")
        if raw_substrate_pairs is not None:
            self.substrate_spot_pairs = [
                (tuple(pair.get("raw", (0.0, 0.0))), tuple(pair.get("transformed", (0.0, 0.0))))
                for pair in raw_substrate_pairs
            ]
        else:
            self.substrate_spot_pairs = []

        raw_substrate_pairs = loaded_state.get("substrate_spot_pairs") or []
        self.substrate_spot_pairs = [
            (tuple(pair.get("raw", (0.0, 0.0))), tuple(pair.get("transformed", (0.0, 0.0))))
            for pair in raw_substrate_pairs
        ]

        raw_adsorbate_pairs = loaded_state.get("adsorbate_spot_pairs")
        converted_adsorbate_pairs: Dict[int, List[Tuple[Tuple[float, float], Tuple[float, float]]]] = {}
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
        self.adsorbate_spot_pairs = converted_adsorbate_pairs

        if not self.adsorbate_spot_pairs:
            self.adsorbate_spot_pairs = {i: [] for i in range(len(self.adsorbate_spot_sets))}
        else:
            for idx in range(len(self.adsorbate_spot_sets)):
                self.adsorbate_spot_pairs.setdefault(idx, [])

        self.set_substrate_raw_visibility(getattr(self, "show_substrate_raw_spots", True))
        self.set_substrate_transformed_visibility(getattr(self, "show_substrate_transformed_spots", True))
        self.set_adsorbate_raw_visibility(getattr(self, "show_adsorbate_raw_spots", True))
        self.set_adsorbate_transformed_visibility(getattr(self, "show_adsorbate_transformed_spots", True))

        format_version = session_data.get("format_version", "1.0")
        history_data = session_data.get("history_data", {})

        self.history_manager.history = history_data.get("tree", {}) or {}

        # Rebuild original image registry
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

        # Ensure nodes are linked to latest registry (also handles legacy sessions)
        if hasattr(self.history_manager, "rebuild_indexes"):
            self.history_manager.rebuild_indexes()

        if format_version == "1.1":
            stored_order = history_data.get("original_order", []) or []
            filtered_order = [img_id for img_id in stored_order if img_id in getattr(self.history_manager, "original_images", {})]
            for img_id in getattr(self.history_manager, "original_images", {}):
                if img_id not in filtered_order:
                    filtered_order.append(img_id)
            if hasattr(self.history_manager, "_original_order"):
                self.history_manager._original_order = filtered_order

        if hasattr(self.history_manager, "refresh_widget"):
            self.history_manager.refresh_widget()

        current_id = history_data.get("current_node_id")
        self.history_manager.set_current_node_by_id(current_id, emit_signal=False)

        logger.info(f"Session loaded successfully. Refreshing user interface...")
        
        self.file_loaded_successfully.emit(os.path.basename(self.original_file_path or "Loaded Session"))
        self.adsorbate_sets_structure_changed.emit()
        self.substrate_definition_changed.emit()
        self.substrate_transform_results_updated.emit()
        self.superstructure_periodicity_results_updated.emit(self.superstructure_periodicity_results)
        
        self.history_manager.current_node_changed.emit(self.history_manager.get_current_node())

    def get_current_image_data_for_processing(self) -> Optional[Any]:
        """Gets the image data from the current history node for processing."""
        current_node = self.history_manager.get_current_node()
        if current_node and current_node.image_data is not None:
            return current_node.image_data.copy()
        logger.warning("AppController: No current image data available for processing.")
        return None

    def get_current_node_info_for_dialogs(self) -> Optional[Tuple[str, str, Any, Optional[str], Optional[str]]]:
        """
        Returns information about the current node needed to open dialogs.
        Returns: Tuple (node_id, node_data_type, image_data_copy, original_image_id, original_image_label) or None.
        """
        current_node = self.history_manager.get_current_node()
        if current_node and current_node.image_data is not None:
            source_image_id = current_node.original_image_id
            source_label = None
            if source_image_id:
                record = self.history_manager.get_original_image_record(source_image_id)
                if record:
                    source_label = record.display_name
            return (
                current_node.node_id,
                current_node.data_type,
                current_node.image_data.copy(),
                source_image_id,
                source_label,
            )
        return None
    
    def load_metadata_into_session(self):
        """
        Load metadata from a selected .stp file and attach it to the
        root history node so that a later save is possible.
        """
        root_node = self.history_manager.get_root_node_for_node(self.history_manager.current_node_id)
        if not root_node:
            QMessageBox.warning(None, "Error", "Could not find the root node in the active history.")
            return

        file_path, _ = QFileDialog.getOpenFileName(
            None, "Select the original STP file to load metadata", "", "Omicron STP Files (*.stp)"
        )
        if not file_path:
            return

        try:
            # Use the reader but only extract the header
            temp_stm_image = load_stm_file(file_path)
            if temp_stm_image and temp_stm_image.raw_header:
                # Update the `raw_header` within the root node parameters
                root_node.parameters["raw_header"] = temp_stm_image.raw_header
                # Update core fields if they were missing
                if "size_nm_x" not in root_node.parameters:
                    root_node.parameters["size_nm_x"] = temp_stm_image.size_nm_x
                    root_node.parameters["size_nm_y"] = temp_stm_image.size_nm_y

                logger.info(f"Successfully loaded and attached metadata from file: {os.path.basename(file_path)}")
                QMessageBox.information(None, "Success", "Metadata has been loaded. You can now save an .stp file.")
            else:
                raise ValueError("The selected file did not contain valid metadata.")
        except Exception as e:
            logger.exception(f"Error while loading metadata: {e}")
            QMessageBox.critical(None, "Error", f"Couldnt load metadata:\n{e}")
    
    def reset_session(self):
        """
        Clears all loaded data, history, and analysis results so the user can start fresh.
        """
        logger.info("AppController: Resetting current analysis session.")
        self.history_manager.clear_history()
        self.clear_all_spot_data()

        self.original_file_path = None
        self.reference_ideal_substrate_spots_px.clear()
        self.custom_lattice_info = None
        self.last_selected_substrate = PREDEFINED_SUBSTRATE_NONE
        self.current_substrate_a_surf = None
        self.current_substrate_type = None
        self.current_substrate_name = PREDEFINED_SUBSTRATE_NONE
        self.substrate_definition_name = PREDEFINED_SUBSTRATE_NONE
        self.substrate_lattice_type = None
        self.substrate_a_surf = None
        self.substrate_F_m2i = None
        self.substrate_t_m2i = None
        self.substrate_transform_analysis_m2i = None
        self.displayable_fitted_substrate_spots_on_fft.clear()
        self.show_ideal_lattice = True
        self.current_fft_data_shape = None
        self.user_selected_substrate_spots.clear()

        self.history_manager.refresh_widget()

        self.substrate_definition_changed.emit()
        self.substrate_transform_results_updated.emit()
        self.substrate_real_space_params_updated.emit({})
        self.adsorbate_sets_structure_changed.emit()
        self.adsorbate_set_updated.emit(0)
        self.adsorbate_real_space_params_updated.emit(0, {})
        self.adsorbate_expected_type_updated.emit(0, ADSORBATE_LATTICE_TYPE_UNKNOWN)
        self.superstructure_periodicity_results_updated.emit(None)
        self.spot_lists_updated.emit()
        logger.info("AppController: Session reset complete.")

    def load_file(self, file_path: str):
        """
        Loads an STM file, creates an initial history node, and registers it as a new
        original image without discarding previously loaded data. Emits signals
        indicating success or failure.
        """
        logger.info(f"AppController: Attempting to load file: {file_path}")
        try:
            stm_image_obj: Optional[STMImage] = load_stm_file(file_path)

            if not (stm_image_obj and stm_image_obj.data is not None):
                logger.error(f"AppController: Failed to load valid data from file: {file_path}")
                self.file_loading_failed.emit(f"Could not load valid data from file: {file_path}")
                return

            self.original_file_path = file_path  # Track the most recently loaded source

            display_name = self.history_manager.get_next_original_display_name()
            root_params = {
                "raw_header": stm_image_obj.raw_header,
                "filename": os.path.basename(file_path),
                "source_path": file_path,
                "pixels_x": stm_image_obj.pixels_x,
                "pixels_y": stm_image_obj.pixels_y,
                "size_nm_x": stm_image_obj.size_nm_x,
                "size_nm_y": stm_image_obj.size_nm_y,
                "bias_v": stm_image_obj.bias_v,
                "setpoint_a": stm_image_obj.setpoint_a,
                "scan_angle_deg": stm_image_obj.scan_angle_deg,
                "scan_speed_nm_s": stm_image_obj.scan_speed_nm_s,
                "z_nm_per_raw": stm_image_obj.z_nm_per_raw,
                "image_type": stm_image_obj.image_type,
                "original_label": display_name,
                "source_image_label": display_name,
            }

            record = OriginalImageRecord(
                display_name=display_name,
                stm_image=stm_image_obj,
                source_path=file_path,
                extra_metadata=dict(root_params),
            )
            self.history_manager.register_original_image(record)
            record.extra_metadata["source_image_id"] = record.image_id
            root_params["source_image_id"] = record.image_id

            root_node = HistoryNode(
                operation_name="Original",
                image_data=stm_image_obj.data.copy(),
                parameters=root_params,
                data_type="STM",
                original_image_id=record.image_id,
            )

            self.history_manager.add_node(root_node)
            self.history_manager.set_current_node_by_id(root_node.node_id)

            logger.info(
                "AppController: File '%s' registered as %s (node_id=%s).",
                os.path.basename(file_path),
                display_name,
                root_node.node_id,
            )
            self.file_loaded_successfully.emit(f"{display_name} - {os.path.basename(file_path)}")

        except FileNotFoundError: # pragma: no cover
            logger.error(f"AppController: File not found: {file_path}")
            self.file_loading_failed.emit(f"File not found: {file_path}")
        except ValueError as ve: # pragma: no cover
            logger.error(f"AppController: Value error while loading file {file_path}: {ve}")
            self.file_loading_failed.emit(f"Format error in file {file_path}: {ve}")
        except Exception as e: # pragma: no cover
            logger.exception(f"AppController: An unexpected error occurred while loading file {file_path}: {e}")
            self.file_loading_failed.emit(f"Unexpected error loading file: {e}")

    def update_substrate_analysis_results(self, results: Dict[str, Any]):
        """
        Updates the substrate state based on the results from SubstrateSpotSelectionDialog.
        """
        new_user_spots = results.get("spots", [])
        new_lattice_type = results.get("lattice_type")
        new_a_surf = results.get("a_surf")
        new_def_name = results.get("substrate_definition", PREDEFINED_SUBSTRATE_NONE)
        
        new_F_m2i = results.get("transformation_F_m2i")
        new_t_m2i = results.get("translation_t_m2i")
        new_analysis_m2i = results.get("transform_analysis_m2i")
        new_displayable_fitted_spots = results.get("displayable_fitted_spots", [])

        spots_changed = (self.user_selected_substrate_spots != new_user_spots)
        def_changed = (self.substrate_lattice_type != new_lattice_type or
                       self.substrate_a_surf != new_a_surf or
                       self.substrate_definition_name != new_def_name)
        transform_changed = (not np.array_equal(self.substrate_F_m2i, new_F_m2i) or
                             not np.array_equal(self.substrate_t_m2i, new_t_m2i) or
                             self.displayable_fitted_substrate_spots_on_fft != new_displayable_fitted_spots)

        self.user_selected_substrate_spots = list(new_user_spots)
        self.substrate_lattice_type = new_lattice_type
        self.substrate_a_surf = new_a_surf
        self.substrate_definition_name = new_def_name
        
        self.substrate_F_m2i = new_F_m2i
        self.substrate_t_m2i = new_t_m2i
        self.substrate_transform_analysis_m2i = new_analysis_m2i
        self.displayable_fitted_substrate_spots_on_fft = list(new_displayable_fitted_spots)
        if new_displayable_fitted_spots and new_user_spots:
            pair_count = min(len(new_user_spots), len(new_displayable_fitted_spots))
            self.substrate_spot_pairs = [
                (tuple(new_user_spots[i]), tuple(new_displayable_fitted_spots[i]))
                for i in range(pair_count)
            ]
        else:
            self.substrate_spot_pairs = []

        if new_def_name == PREDEFINED_SUBSTRATE_CUSTOM:
            self.custom_lattice_info = {"type": new_lattice_type, "a_surf": new_a_surf, "name": "Custom (Dialog)"}
            self.last_selected_substrate = PREDEFINED_SUBSTRATE_CUSTOM
        elif new_def_name != PREDEFINED_SUBSTRATE_NONE:
            self.custom_lattice_info = None
            self.last_selected_substrate = new_def_name
        else:
            self.custom_lattice_info = None
            self.last_selected_substrate = PREDEFINED_SUBSTRATE_NONE

        new_ideal_ref_spots = results.get("ideal_substrate_spots_px_for_reference", [])
        self.reference_ideal_substrate_spots_px = list(new_ideal_ref_spots)
        logger.info(f"AppController: Updated reference ideal substrate spots count: {len(self.reference_ideal_substrate_spots_px)}")

        logger.info(f"AppController: Substrate analysis results updated. Spots: {len(self.user_selected_substrate_spots)}. "
                    f"Transform F: {'Set' if self.substrate_F_m2i is not None else 'None'}. "
                    f"Displayable fitted spots: {len(self.displayable_fitted_substrate_spots_on_fft)}")
        
        self.substrate_real_space_results = None
        self.substrate_real_space_params_updated.emit({})

        if spots_changed:
            self.spot_lists_updated.emit()
        if def_changed:
            self.substrate_definition_changed.emit()
        if transform_changed or spots_changed or def_changed:
            self.substrate_transform_results_updated.emit()

    def calculate_and_store_substrate_real_params(self):
        """
        Calculates real-space lattice parameters for the substrate based on
        the FITTED substrate spots (ideal lattice points transformed to match user clicks).
        Stores them and emits a signal upon completion.
        """
        if not LATTICE_ANALYSIS_FUNCTIONS_AVAILABLE:
            logger.error("Lattice analysis functions (get_real_space_lattice_parameters) not available.")
            self.substrate_real_space_params_updated.emit({"error": "Lattice functions missing."})
            return

        logger.info("AppController: Attempting to calculate substrate real space parameters from fitted spots.")

        if not (self.substrate_lattice_type and self.substrate_a_surf and self.substrate_a_surf > 0):
            logger.warning("Substrate definition (type/a_surf) used for fitting not set. Cannot calculate real space params.")
            self.substrate_real_space_params_updated.emit({"error": "Substrate definition for fit missing."})
            return
        
        if not self.displayable_fitted_substrate_spots_on_fft:
            logger.warning("Fitted substrate spots (in FFT px) are not available.")
            self.substrate_real_space_params_updated.emit({"error": "Fitted substrate spots missing."})
            return

        if not self.current_fft_data_shape:
            logger.warning("Current FFT data shape not available. Cannot calculate real space params.")
            self.substrate_real_space_params_updated.emit({"error": "FFT shape missing."})
            return

        current_node = self.history_manager.get_current_node()
        if not current_node:
            logger.warning("No current history node available for Lx/Ly."); self.substrate_real_space_params_updated.emit({"error": "No active node."}); return
        
        root_node = self.history_manager.get_root_node_for_node(current_node.node_id)
        if not (root_node and root_node.operation_name == "Original" and root_node.parameters):
            logger.warning("Could not get Original node parameters for Lx/Ly."); self.substrate_real_space_params_updated.emit({"error": "Original node params missing."}); return
        
        Lx_nm = root_node.parameters.get("size_nm_x")
        Ly_nm = root_node.parameters.get("size_nm_y")

        if not (Lx_nm and Ly_nm and Lx_nm > 0 and Ly_nm > 0):
            logger.warning("Invalid Lx/Ly from Original node."); self.substrate_real_space_params_updated.emit({"error": "Invalid Lx/Ly."}); return

        fft_rows_ky, fft_cols_kx = self.current_fft_data_shape
        center_kx_px = fft_cols_kx / 2.0
        center_ky_px = fft_rows_ky / 2.0

        fitted_g_vectors_relative_px = [
            (kx_abs - center_kx_px, ky_abs - center_ky_px)
            for kx_abs, ky_abs in self.displayable_fitted_substrate_spots_on_fft
        ]

        expected_spot_count = 0
        if self.substrate_lattice_type == LATTICE_TYPE_HEXAGONAL: expected_spot_count = 6
        elif self.substrate_lattice_type == LATTICE_TYPE_SQUARE: expected_spot_count = 4
        
        if len(fitted_g_vectors_relative_px) != expected_spot_count and expected_spot_count > 0:
            logger.warning(f"Incorrect number of fitted spots ({len(fitted_g_vectors_relative_px)}) "
                           f"for {self.substrate_lattice_type} (expected {expected_spot_count}). Cannot calculate real space params.")
            self.substrate_real_space_params_updated.emit({"error": f"Need {expected_spot_count} fitted spots."})
            return
        if expected_spot_count == 0 and len(fitted_g_vectors_relative_px) < 2 :
             logger.warning(f"Not enough fitted spots to define basis for unknown lattice type.")
             self.substrate_real_space_params_updated.emit({"error": "Need >=2 fitted spots."})
             return


        print(f"displayable_fitted_substrate_spots_on_fft: {self.displayable_fitted_substrate_spots_on_fft}")
        print(f"fitted_g_vectors_relative_px: {fitted_g_vectors_relative_px}")
        results = get_real_space_lattice_parameters(
            selected_g_vectors_relative_px=fitted_g_vectors_relative_px,
            lattice_type=self.substrate_lattice_type,
            Lx_nm=Lx_nm,
            Ly_nm=Ly_nm,
            fft_shape_cols_kx=fft_cols_kx,
            fft_shape_rows_ky=fft_rows_ky
        )

        if results:
            self.substrate_real_space_results = results
            logger.info(f"Successfully calculated substrate real space parameters (from fitted spots): {results}")
            self.substrate_real_space_params_updated.emit(results)
        else: # pragma: no cover
            self.substrate_real_space_results = None
            logger.warning("Failed to calculate substrate real space parameters from fitted spots.")
            self.substrate_real_space_params_updated.emit({"error": "Calculation failed in lattice module."})

    
    def calculate_and_store_adsorbate_real_params(self, set_index: int):
        """
        Calculates real-space lattice parameters for the specified adsorbate set
        using its corrected spots, and stores them. Emits a signal.
        """
        if not LATTICE_ANALYSIS_FUNCTIONS_AVAILABLE: # pragma: no cover
            logger.error("Lattice analysis functions not available for adsorbate.")
            self.adsorbate_real_space_params_updated.emit(set_index, {"error": "Lattice functions missing."})
            return

        logger.info(f"AppController: Attempting to calculate real space params for adsorbate set {set_index}.")

        if not (0 <= set_index < len(self.corrected_adsorbate_spot_sets)): # pragma: no cover
            logger.warning(f"Invalid set_index {set_index} for adsorbate real space params.")
            self.adsorbate_real_space_params_updated.emit(set_index, {"error": "Invalid set index."})
            return
        
        corrected_spots_ideal_px_abs = self.corrected_adsorbate_spot_sets[set_index]
        
        expected_ads_type = self.adsorbate_expected_lattice_types.get(set_index, ADSORBATE_LATTICE_TYPE_UNKNOWN)
        
        num_corrected_spots = len(corrected_spots_ideal_px_abs)

        if num_corrected_spots < 2:
            logger.warning(f"Not enough corrected adsorbate spots (need >=2, got {num_corrected_spots}) for set {set_index}.")
            self.adsorbate_real_space_params_updated.emit(set_index, {"error": f"Need >= 2 corrected spots, got {num_corrected_spots}."})
            return

        if not self.current_fft_data_shape: # pragma: no cover
            logger.warning("Current FFT data shape not available."); self.adsorbate_real_space_params_updated.emit(set_index, {"error": "FFT shape missing."}); return

        root_node = self.history_manager.get_root_node_for_node(self.history_manager.get_current_node().node_id) # type: ignore
        if not (root_node and root_node.parameters): 
            logger.warning("Cannot get Lx, Ly for adsorbate.")
            self.adsorbate_real_space_params_updated.emit(set_index, {"error": "Original node params missing."})
            return # pragma: no cover
        Lx_nm = root_node.parameters.get("size_nm_x")
        Ly_nm = root_node.parameters.get("size_nm_y")
        if not (Lx_nm and Ly_nm and Lx_nm > 0 and Ly_nm > 0): 
            logger.warning("Invalid Lx/Ly.")
            self.adsorbate_real_space_params_updated.emit(set_index, {"error": "Invalid Lx/Ly."})
            return # pragma: no cover
        
        fft_rows_ky, fft_cols_kx = self.current_fft_data_shape
        center_kx_ideal_px = fft_cols_kx / 2.0
        center_ky_ideal_px = fft_rows_ky / 2.0
        
        g_vectors_adsorbate_relative_px = [
            (spot_abs_kx - center_kx_ideal_px, spot_abs_ky - center_ky_ideal_px)
            for spot_abs_kx, spot_abs_ky in corrected_spots_ideal_px_abs
        ]

        basis_g_ads_px = select_adsorbate_reciprocal_basis_vectors_px(
            corrected_g_vectors_relative_px=g_vectors_adsorbate_relative_px,
            expected_lattice_type=expected_ads_type
        )

        if basis_g_ads_px is None:
            logger.warning(f"Failed to select basis g-vectors for adsorbate set {set_index} with type '{expected_ads_type}'.")
            self.adsorbate_real_space_params_updated.emit(set_index, {"error": "Basis g-vector selection failed."})
            return
        g1_ads_px, g2_ads_px = basis_g_ads_px
            
        g1_ads_nm_inv = convert_g_vector_px_to_nm_inv(g1_ads_px, Lx_nm, Ly_nm, fft_cols_kx, fft_rows_ky)
        g2_ads_nm_inv = convert_g_vector_px_to_nm_inv(g2_ads_px, Lx_nm, Ly_nm, fft_cols_kx, fft_rows_ky)

        if g1_ads_nm_inv is None or g2_ads_nm_inv is None: # pragma: no cover
            self.adsorbate_real_space_params_updated.emit(set_index, {"error": "g-vector conversion to nm^-1 failed."}); return

        real_space_vecs_ads = calculate_real_space_vectors_from_g(g1_ads_nm_inv, g2_ads_nm_inv)
        if real_space_vecs_ads is None: # pragma: no cover
            self.adsorbate_real_space_params_updated.emit(set_index, {"error": "Real space vector calculation failed (g-vectors likely collinear)."}); return
        a1_ads_vec_nm, a2_ads_vec_nm = real_space_vecs_ads

        a1_ads_mag_nm = np.linalg.norm(a1_ads_vec_nm)
        a2_ads_mag_nm = np.linalg.norm(a2_ads_vec_nm)
        
        if a1_ads_mag_nm < 1e-9 or a2_ads_mag_nm < 1e-9: # pragma: no cover
            logger.warning(f"Calculated real space vectors for adsorbate set {set_index} have zero or near-zero magnitude.")
            self.adsorbate_real_space_params_updated.emit(set_index, {"error": "Calculated real vectors too short."}); return

        dot_product_ads = np.dot(a1_ads_vec_nm, a2_ads_vec_nm)
        cos_alpha_ads = np.clip(dot_product_ads / (a1_ads_mag_nm * a2_ads_mag_nm), -1.0, 1.0)
        alpha_ads_deg = np.degrees(np.arccos(cos_alpha_ads))

        results = {
            "a1_nm": a1_ads_mag_nm, "a2_nm": a2_ads_mag_nm, "alpha_deg": alpha_ads_deg,
            "a1_vec_nm": a1_ads_vec_nm, "a2_vec_nm": a2_ads_vec_nm,
            "g1_vec_px_ideal_sys": g1_ads_px,
            "g2_vec_px_ideal_sys": g2_ads_px,
            "g1_vec_nm_inv": g1_ads_nm_inv, 
            "g2_vec_nm_inv": g2_ads_nm_inv,
            "source_corrected_spots_ideal_px": corrected_spots_ideal_px_abs
        }
        logger.info(f"Adsorbate set {set_index} real space params: a1={a1_ads_mag_nm:.3f}nm, a2={a2_ads_mag_nm:.3f}nm, alpha={alpha_ads_deg:.2f}deg")
        self.adsorbate_real_space_results[set_index] = results
        self.adsorbate_real_space_params_updated.emit(set_index, results)


    def add_operation_to_history(self,
                                 parent_node_id: str,
                                 op_name: str,
                                 params: Dict[str, Any],
                                 processed_data: np.ndarray,
                                 data_type: str,
                                 source_roi_slice: Optional[Tuple[slice, slice]] = None):
        """
        Creates a new history node for the performed operation and adds it to the manager.
        """
        if processed_data is None:
            logger.warning(f"AppController: No processed data provided for operation '{op_name}'. Node not added.")
            return

        parent_node = self.history_manager.get_node_by_id(parent_node_id)
        original_params = dict(params) if params else {}

        if parent_node and parent_node.image_data is not None:
            if np.array_equal(processed_data, parent_node.image_data) and \
               original_params == parent_node.parameters.get(op_name, {}):
                logger.info(f"AppController: Data for '{op_name}' has not changed. Node not added.")
                return

        params = dict(original_params)
        source_image_id: Optional[str] = None
        source_image_label: Optional[str] = None
        if parent_node:
            source_image_id = parent_node.original_image_id
            if source_image_id:
                record = self.history_manager.get_original_image_record(source_image_id)
                if record:
                    source_image_label = record.display_name
        if source_image_id:
            params.setdefault("source_image_id", source_image_id)
        if source_image_label:
            params.setdefault("source_image_label", source_image_label)

        new_node = HistoryNode(
            parent_id=parent_node_id,
            operation_name=op_name,
            parameters=params,
            image_data=processed_data,
            data_type=data_type,
            source_roi_slice=source_roi_slice,
            original_image_id=source_image_id
        )

        self.history_manager.add_node(new_node)
        self.history_manager.set_current_node_by_id(new_node.node_id)
        logger.info(f"AppController: Added '{op_name}' node (ID: {new_node.node_id}) to history.")

    def apply_gaussian_blur(self, parent_node_id: str, parent_data_type: str,
                            processed_data: np.ndarray, params: Dict[str, Any],
                            source_roi_slice: Optional[Tuple[slice, slice]] = None):
        self.add_operation_to_history(parent_node_id, "Gaussian Blur", params, processed_data, parent_data_type, source_roi_slice)

    def apply_gaussian_sharpening(self, parent_node_id: str, parent_data_type: str,
                                  processed_data: np.ndarray, params: Dict[str, Any],
                                  source_roi_slice: Optional[Tuple[slice, slice]] = None):
        self.add_operation_to_history(parent_node_id, "Gaussian Sharpening", params, processed_data, parent_data_type, source_roi_slice)
    
    def apply_stm_transform(self, parent_node_id: str, processed_data: np.ndarray, params: Dict[str, Any]):
        """Add the transformed STM image as a new step in history."""
        self.add_operation_to_history(
            parent_node_id=parent_node_id,
            op_name="STM Transform",
            params=params,
            processed_data=processed_data,
            data_type="STM" # Result remains an STM image
        )

    def apply_plane_leveling(self, parent_node_id: str, parent_data_type: str,
                             processed_data: np.ndarray, params: Dict[str, Any],
                             source_roi_slice: Optional[Tuple[slice, slice]] = None):
        """Applies plane leveling to the processed data."""
        self.add_operation_to_history(parent_node_id, "Plane Leveling", params, processed_data, parent_data_type, source_roi_slice)

    def apply_median_filter(self, parent_node_id: str, parent_data_type: str,
                            processed_data: np.ndarray, params: Dict[str, Any],
                            source_roi_slice: Optional[Tuple[slice, slice]] = None):
        """Applies median filter to the processed data."""
        self.add_operation_to_history(parent_node_id, "Median Filter", params, processed_data, parent_data_type, source_roi_slice)

    def apply_nlmeans_denoising(self, parent_node_id: str, parent_data_type: str,
                                processed_data: np.ndarray, params: Dict[str, Any],
                                source_roi_slice: Optional[Tuple[slice, slice]] = None):
        """Applies NL-Means denoising to the processed data."""
        self.add_operation_to_history(parent_node_id, "NL-Means", params, processed_data, parent_data_type, source_roi_slice)

    def apply_bm3d_denoising(self, parent_node_id: str, parent_data_type: str,
                             processed_data: np.ndarray, params: Dict[str, Any],
                             source_roi_slice: Optional[Tuple[slice, slice]] = None):
        """Applies BM3D denoising to the processed data."""
        self.add_operation_to_history(parent_node_id, "BM3D", params, processed_data, parent_data_type, source_roi_slice)

    def calculate_fft_operation(self, parent_node_id: str,
                                processed_fft_data: np.ndarray,
                                complex_fft_data: Optional[np.ndarray],
                                params: Dict[str, Any],
                                source_roi_slice: Optional[Tuple[slice, slice]] = None):
        """Calculates the FFT of the current image data and stores it in the history."""
        if processed_fft_data is not None:
            self.current_fft_data_shape = processed_fft_data.shape
            logger.info(f"AppController: Stored current FFT data shape: {self.current_fft_data_shape}")
        else: # pragma: no cover
            self.current_fft_data_shape = None
            logger.warning("AppController: FFT data is None, cannot store shape.")

        params = dict(params) if params else {}
        parent_node = self.history_manager.get_node_by_id(parent_node_id)
        source_image_id: Optional[str] = None
        source_image_label: Optional[str] = None
        if parent_node:
            source_image_id = parent_node.original_image_id
            if source_image_id:
                record = self.history_manager.get_original_image_record(source_image_id)
                if record:
                    source_image_label = record.display_name
        if source_image_id:
            params.setdefault("source_image_id", source_image_id)
        if source_image_label:
            params.setdefault("source_image_label", source_image_label)

            
        # self.add_operation_to_history(parent_node_id, "FFT", params, processed_fft_data, "FFT", source_roi_slice)
        new_node = HistoryNode(
            parent_id=parent_node_id,
            operation_name="FFT",
            parameters=params,
            image_data=processed_fft_data,
            data_type="FFT",
            complex_fft_data=complex_fft_data,
            source_roi_slice=source_roi_slice,
            original_image_id=source_image_id
        )
        self.history_manager.add_node(new_node)
        self.history_manager.set_current_node_by_id(new_node.node_id)
        self.substrate_real_space_results = None
        self.substrate_real_space_params_updated.emit({})
        self.adsorbate_real_space_results.clear()
        if hasattr(self, 'adsorbate_real_space_params_updated'): self.adsorbate_real_space_params_updated.emit(self.current_adsorbate_set_index, {})

    def set_spot_selection_mode(self, mode: str):
        """Sets the spot selection mode (Substrate/Adsorbate)."""
        if mode in [SPOT_SELECTION_SUBSTRATE, SPOT_SELECTION_ADSORBATE]:
            if self.spot_selection_mode != mode:
                self.spot_selection_mode = mode
                logger.info(f"Spot selection mode set to: {self.spot_selection_mode}")
                self.spot_selection_parameters_changed.emit()
        else:
            logger.warning(f"Attempted to set invalid spot selection mode: {mode}")

    def set_spot_refinement_method(self, method: str):
        """Sets the spot refinement method."""
        if method in [REFINEMENT_DIRECT_CLICK, REFINEMENT_MAX_PIXEL, REFINEMENT_GAUSSIAN_FIT]:
            if self.spot_refinement_method != method:
                self.spot_refinement_method = method
                logger.info(f"Spot refinement method set to: {self.spot_refinement_method}")
                self.spot_selection_parameters_changed.emit()
        else:
            logger.warning(f"Attempted to set invalid spot refinement method: {method}")

    def set_refinement_roi_size(self, size: int):
        """Sets the ROI size for spot refinement."""
        if isinstance(size, int) and 3 <= size <= 21 and size % 2 != 0: # Example validation
            if self.refinement_roi_size != size:
                self.refinement_roi_size = size
                logger.info(f"Refinement ROI size set to: {self.refinement_roi_size}")
                self.spot_selection_parameters_changed.emit()
        else:
            logger.warning(f"Attempted to set invalid refinement ROI size: {size}")

    def set_substrate_raw_visibility(self, is_visible: bool) -> None:
        """Stores substrate raw spot visibility and informs listeners."""
        if self.show_substrate_raw_spots != is_visible:
            logger.debug("AppController: Substrate raw visibility changed to %s", is_visible)
        self.show_substrate_raw_spots = is_visible
        self.substrate_raw_visibility_updated.emit(is_visible)

    def set_substrate_transformed_visibility(self, is_visible: bool) -> None:
        """Stores substrate transformed spot visibility and informs listeners."""
        if self.show_substrate_transformed_spots != is_visible:
            logger.debug("AppController: Substrate transformed visibility changed to %s", is_visible)
        self.show_substrate_transformed_spots = is_visible
        self.substrate_transformed_visibility_updated.emit(is_visible)

    def set_adsorbate_raw_visibility(self, is_visible: bool) -> None:
        """Stores adsorbate raw spot visibility and informs listeners."""
        if self.show_adsorbate_raw_spots != is_visible:
            logger.debug("AppController: Adsorbate raw visibility changed to %s", is_visible)
        self.show_adsorbate_raw_spots = is_visible
        self.adsorbate_raw_visibility_updated.emit(is_visible)

    def set_adsorbate_transformed_visibility(self, is_visible: bool) -> None:
        """Stores adsorbate transformed spot visibility and informs listeners."""
        if self.show_adsorbate_transformed_spots != is_visible:
            logger.debug("AppController: Adsorbate transformed visibility changed to %s", is_visible)
        self.show_adsorbate_transformed_spots = is_visible
        self.adsorbate_transformed_visibility_updated.emit(is_visible)

    def clear_last_adsorbate_spot(self):
        """Removes the last added spot from the current adsorbate set."""
        if self.spot_selection_mode == SPOT_SELECTION_ADSORBATE and \
           0 <= self.current_adsorbate_set_index < len(self.adsorbate_spot_sets):
            current_set = self.adsorbate_spot_sets[self.current_adsorbate_set_index]
            if current_set:
                removed_point = current_set.pop()
                if self.current_adsorbate_set_index < len(self.corrected_adsorbate_spot_sets) and self.corrected_adsorbate_spot_sets[self.current_adsorbate_set_index]:
                    self.corrected_adsorbate_spot_sets[self.current_adsorbate_set_index].pop()
                if self.current_adsorbate_set_index in self.adsorbate_spot_pairs and self.adsorbate_spot_pairs[self.current_adsorbate_set_index]:
                    self.adsorbate_spot_pairs[self.current_adsorbate_set_index].pop()
                logger.info(f"Removed last adsorbate spot {removed_point} from set {self.current_adsorbate_set_index}.")
                self.spot_lists_updated.emit()
            else: logger.debug("No adsorbate spots in current set to clear.") # pragma: no cover
        else: logger.debug("Not in adsorbate mode or invalid set index for clear_last_adsorbate_spot.") # pragma: no cover

    def reselect_current_adsorbate_set(self):
        """Clears all spots from the current adsorbate set."""
        if self.spot_selection_mode == SPOT_SELECTION_ADSORBATE and \
            0 <= self.current_adsorbate_set_index < len(self.adsorbate_spot_sets):
            if self.adsorbate_spot_sets[self.current_adsorbate_set_index]:
                self.adsorbate_spot_sets[self.current_adsorbate_set_index].clear()
                if 0 <= self.current_adsorbate_set_index < len(self.corrected_adsorbate_spot_sets):
                    self.corrected_adsorbate_spot_sets[self.current_adsorbate_set_index].clear()
                self.adsorbate_spot_pairs[self.current_adsorbate_set_index] = []
                logger.info(f"Cleared all spots from adsorbate set {self.current_adsorbate_set_index}.")
                self.spot_lists_updated.emit()
        else: logger.debug("Not in adsorbate mode or invalid set index for reselect_current_adsorbate_set.") # pragma: no cover

    def set_expected_adsorbate_lattice_type(self, set_index: int, lattice_type: str):
        """Sets the expected lattice type for a given adsorbate set."""
        valid_types = [ADSORBATE_LATTICE_TYPE_UNKNOWN, ADSORBATE_LATTICE_TYPE_HEXAGONAL, ADSORBATE_LATTICE_TYPE_SQUARE]
        if not (0 <= set_index < len(self.adsorbate_spot_sets)) or lattice_type not in valid_types:
            logger.warning(f"AppController: Invalid set_index {set_index} or lattice_type '{lattice_type}' for adsorbate.")
            return
        
        if self.adsorbate_expected_lattice_types.get(set_index) != lattice_type:
            self.adsorbate_expected_lattice_types[set_index] = lattice_type
            logger.info(f"AppController: Expected lattice type for adsorbate set {set_index} set to '{lattice_type}'.")
            if set_index in self.adsorbate_real_space_results:
                del self.adsorbate_real_space_results[set_index]
                if hasattr(self, 'adsorbate_real_space_params_updated'): self.adsorbate_real_space_params_updated.emit(set_index, {}) 
            
            self.adsorbate_expected_type_updated.emit(set_index, lattice_type)

    def clear_all_adsorbate_sets(self):
        """Clears all adsorbate sets and resets to one empty set."""
        if self.adsorbate_spot_sets != [[]] or self.current_adsorbate_set_index != 0 :
            self.adsorbate_spot_sets = [[]]
            self.corrected_adsorbate_spot_sets = [[]]
            self.current_adsorbate_set_index = 0
            self.adsorbate_expected_lattice_types = {0: ADSORBATE_LATTICE_TYPE_UNKNOWN}
            logger.info("All adsorbate spot sets cleared. Reset to one empty set.")
            self.adsorbate_sets_structure_changed.emit()
            if hasattr(self, 'adsorbate_expected_type_updated'): self.adsorbate_expected_type_updated.emit(0, ADSORBATE_LATTICE_TYPE_UNKNOWN)
            self.adsorbate_set_updated.emit(0)
        else:
            logger.debug("No adsorbate sets to clear or already in default state.")

    def update_adsorbate_set_results(self, 
                                     set_index: int, 
                                     raw_spots: List[Tuple[float, float]], 
                                     corrected_spots_ideal_system: List[Tuple[float, float]]):
        """
        Updates the raw and corrected spots for a given adsorbate set.
        """
        if not (0 <= set_index < len(self.adsorbate_spot_sets)):
            logger.error(f"AppController: Invalid set_index {set_index} for updating adsorbate spots.")
            return

        while len(self.corrected_adsorbate_spot_sets) <= set_index:
            self.corrected_adsorbate_spot_sets.append([])

        self.adsorbate_spot_pairs.setdefault(set_index, [])

        raw_changed = self.adsorbate_spot_sets[set_index] != raw_spots
        corrected_changed = self.corrected_adsorbate_spot_sets[set_index] != corrected_spots_ideal_system

        if raw_changed:
            self.adsorbate_spot_sets[set_index] = list(raw_spots)
            logger.info(f"AppController: Updated raw adsorbate spots for set {set_index}. Count: {len(raw_spots)}")
        
        if corrected_changed:
            self.corrected_adsorbate_spot_sets[set_index] = list(corrected_spots_ideal_system)
            logger.info(f"AppController: Updated corrected adsorbate spots (ideal sys) for set {set_index}. Count: {len(corrected_spots_ideal_system)}")

        pair_count = min(len(self.adsorbate_spot_sets[set_index]), len(self.corrected_adsorbate_spot_sets[set_index]))
        if pair_count > 0:
            self.adsorbate_spot_pairs[set_index] = [
                (tuple(self.adsorbate_spot_sets[set_index][i]), tuple(self.corrected_adsorbate_spot_sets[set_index][i]))
                for i in range(pair_count)
            ]
        else:
            self.adsorbate_spot_pairs[set_index] = []

        if raw_changed or corrected_changed:
            self.adsorbate_set_updated.emit(set_index)
            if raw_changed and hasattr(self, 'spot_lists_updated'):
                 self.spot_lists_updated.emit()

        if set_index in self.adsorbate_real_space_results:
            del self.adsorbate_real_space_results[set_index]
            self.adsorbate_real_space_params_updated.emit(set_index, {})
            if hasattr(self, 'adsorbate_real_space_params_updated'): self.adsorbate_real_space_params_updated.emit(set_index, {})

    def update_superstructure_periodicity_results(self, results: Optional[Dict[str, Any]]):
        """Updates and stores the superstructure periodicity analysis results."""
        self.superstructure_periodicity_results = results
        logger.info(f"AppController: Updated superstructure periodicity results: {results}")
        self.superstructure_periodicity_results_updated.emit(self.superstructure_periodicity_results)

    def add_new_node_to_history(self, new_node: HistoryNode):
        """
        Add a prepared history node and make it the current entry.
        """
        if self.history_manager:
            self.history_manager.add_node(new_node)
            # Promote the newly added node to active
            self.history_manager.set_current_node_by_id(new_node.node_id)
            logger.info(f"AppController: Added new node '{new_node.operation_name}' to history.")

    def add_new_adsorbate_set(self):
        """Adds a new, empty adsorbate spot set and sets it as current."""
        self.adsorbate_spot_sets.append([])
        self.corrected_adsorbate_spot_sets.append([])
        self.current_adsorbate_set_index = len(self.adsorbate_spot_sets) - 1
        new_set_index = len(self.adsorbate_spot_sets) - 1
        self.adsorbate_spot_pairs.setdefault(new_set_index, [])
        logger.info(f"Added new adsorbate set. Index: {self.current_adsorbate_set_index}")
        last_selected_type_in_panel = ADSORBATE_LATTICE_TYPE_UNKNOWN
        self.adsorbate_expected_lattice_types[new_set_index] = last_selected_type_in_panel
        self.spot_lists_updated.emit()
        self.adsorbate_sets_structure_changed.emit()
        if hasattr(self, 'adsorbate_expected_type_updated'): self.adsorbate_expected_type_updated.emit(new_set_index, last_selected_type_in_panel)

    def set_current_adsorbate_set_by_index(self, index: int):
        """Sets the current adsorbate set based on the index."""
        if 0 <= index < len(self.adsorbate_spot_sets):
            if self.current_adsorbate_set_index != index:
                self.current_adsorbate_set_index = index
                logger.info(f"Current adsorbate set changed to index: {index}")
                self.spot_selection_parameters_changed.emit()
        else:
            logger.warning(f"Attempted to set invalid adsorbate set index: {index}") # pragma: no cover

    def clear_all_spot_data(self):
        """Clears all spot data."""
        changed = False
        if self.substrate_spots:
            self.substrate_spots.clear()
            changed = True
        if self.adsorbate_spot_sets != [[]] or self.current_adsorbate_set_index != 0:
            self.adsorbate_spot_sets = [[]]
            self.corrected_adsorbate_spot_sets = [[]]
            self.current_adsorbate_set_index = 0
            changed = True
        
        self.user_selected_substrate_spots.clear()
        self.adsorbate_real_space_results.clear()
        self.substrate_F_m2i = None
        self.substrate_t_m2i = None
        self.substrate_transform_analysis_m2i = None
        self.displayable_fitted_substrate_spots_on_fft.clear()
        self.substrate_spot_pairs.clear()
        self.substrate_real_space_results = None
        self.adsorbate_expected_lattice_types = {0: ADSORBATE_LATTICE_TYPE_UNKNOWN}
        self.adsorbate_spot_pairs = {0: []}
        self.superstructure_periodicity_results = None
        self.superstructure_periodicity_results_updated.emit(None)

        self.set_substrate_raw_visibility(True)
        self.set_substrate_transformed_visibility(True)
        self.set_adsorbate_raw_visibility(True)
        self.set_adsorbate_transformed_visibility(True)

        if hasattr(self, 'substrate_real_space_params_updated'): self.substrate_real_space_params_updated.emit({})
        if hasattr(self, 'spot_lists_updated'): self.spot_lists_updated.emit()
        if hasattr(self, 'adsorbate_sets_structure_changed'): self.adsorbate_sets_structure_changed.emit()
        if hasattr(self, 'substrate_transform_results_updated'): self.substrate_transform_results_updated.emit()
        if hasattr(self, 'adsorbate_real_space_params_updated'): self.adsorbate_real_space_params_updated.emit(0, {})
        if hasattr(self, 'adsorbate_expected_type_updated'): self.adsorbate_expected_type_updated.emit(0, ADSORBATE_LATTICE_TYPE_UNKNOWN)
        if changed and hasattr(self, 'adsorbate_set_updated'): self.adsorbate_set_updated.emit(0)
        logger.debug("All spot data and substrate transform results cleared.")
        
        if changed:
            logger.debug("All spot data cleared by clear_all_spot_data.")
            self.spot_lists_updated.emit()
            self.adsorbate_sets_structure_changed.emit()
        else:
            logger.debug("No spot data to clear or already in default state.")

