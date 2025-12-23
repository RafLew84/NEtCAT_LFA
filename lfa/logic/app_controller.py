# lfa/logic/app_controller.py
"""
Central controller for the LFA application.
Manages application state and coordinates operations between UI and backend modules.
"""
import csv
import json
import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import QApplication, QFileDialog, QMessageBox

from ..core.constants import (
    ADSORBATE_LATTICE_TYPE_UNKNOWN,
    LATTICE_TYPE_CUSTOM,
    LATTICE_TYPE_HEXAGONAL,
    LATTICE_TYPE_SQUARE,
    PREDEFINED_SUBSTRATE_CUSTOM,
    PREDEFINED_SUBSTRATE_NONE,
    REFINEMENT_DIRECT_CLICK,
    REFINEMENT_GAUSSIAN_FIT,
    REFINEMENT_LOCAL_DFT,
    REFINEMENT_MAX_PIXEL,
    REFINEMENT_PARABOLA_3X3,
    SPOT_SELECTION_ADSORBATE,
    SPOT_SELECTION_SUBSTRATE,
)
from ..core.data_models import OriginalImageRecord, STMImage
from ..core.history import HistoryNode
from ..io.factory import load_stm_file
from .reporting import (
    build_real_space_json,
    build_real_space_records,
    build_real_space_summary,
)
from .services import AnalysisExecutor, HistoryOrchestrator, SessionService, SpotSetService
from .session_serializer import SessionSerializer

if TYPE_CHECKING:  # pragma: no cover
    from .session_state import ControllerState


logger = logging.getLogger(__name__)

MAX_SUBSTRATE_SPOTS = 8

@dataclass(frozen=True)
class FFTPanelState:
    fft_active: bool
    edit_substrate_enabled: bool = False
    edit_adsorbate_enabled: bool = False
    reselect_adsorbate_enabled: bool = False
    clear_all_adsorbate_sets_enabled: bool = False
    can_calculate_substrate_rs: bool = False
    can_calculate_adsorbate_rs: bool = False

try:
    from ..analysis.lattice import (
        apply_k_resolution_floor_to_covariance,
        augment_covariance_with_calibration,
        calculate_real_space_vectors_from_g,
        compute_real_space_metric_uncertainty,
        convert_g_vector_px_to_nm_inv,
        get_real_space_lattice_parameters,
        select_adsorbate_reciprocal_basis_vectors_px,
    )
    LATTICE_ANALYSIS_FUNCTIONS_AVAILABLE = True
except ImportError: 
    logging.error("AppController: Could not import lattice analysis functions.")
    LATTICE_ANALYSIS_FUNCTIONS_AVAILABLE = False
    def get_real_space_lattice_parameters(*args, **kwargs): return None
    def augment_covariance_with_calibration(*args, **kwargs): return None
    def apply_k_resolution_floor_to_covariance(*args, **kwargs): return None

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
        if self.history_manager and hasattr(self.history_manager, "active_node_changed"):
            self.history_manager.active_node_changed.connect(self._on_history_active_node_changed)
        self.history_service = HistoryOrchestrator(history_manager)

        self.original_file_path: Optional[str] = None

        self.substrate_spots: List[Tuple[float, float]] = []
        self.adsorbate_spot_sets: List[List[Tuple[float, float]]] = [[]] 
        self.corrected_adsorbate_spot_sets: List[List[Tuple[float, float]]] = [[]]
        self.adsorbate_spot_covariance_sets: List[List[Optional[np.ndarray]]] = [[]]
        self.corrected_adsorbate_covariance_sets: List[List[Optional[np.ndarray]]] = [[]]
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
        self.user_selected_substrate_covariances: List[Optional[np.ndarray]] = []

        self.substrate_lattice_type: Optional[str] = None
        self.substrate_a_surf: Optional[float] = None
        self.substrate_definition_name: str = PREDEFINED_SUBSTRATE_NONE 

        self.substrate_F_m2i: Optional[np.ndarray] = None 
        self.substrate_t_m2i: Optional[np.ndarray] = None 
        self.substrate_transform_analysis_m2i: Optional[Dict[str, Any]] = None
        
        self.displayable_fitted_substrate_spots_on_fft: List[Tuple[float, float]] = []
        self.fitted_substrate_spot_covariances: List[Optional[np.ndarray]] = []
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
        self.substrate_visual_offset_nm: Tuple[float, float] = (0.0, 0.0)
        self.adsorbate_visual_offsets_nm: Dict[int, Tuple[float, float]] = {0: (0.0, 0.0)}
        self.pixel_calibration_sigma_nm: Tuple[float, float] = (0.0, 0.0)

        self.superstructure_periodicity_results: Optional[Dict[str, Any]] = None
        self.session_serializer = SessionSerializer(self.history_manager)
        self.session_service = SessionService(self, self.history_service)
        self.spot_service = SpotSetService(self, ADSORBATE_LATTICE_TYPE_UNKNOWN)
        self.analysis_executor = AnalysisExecutor(self)


        logger.info("AppController initialized.")

    def save_analysis_session(self):
        self.session_service.save_session()

    def load_analysis_session(self):
        self.session_service.load_session()

    def delete_history_step(self, node_id: str) -> bool:
        """
        Deletes a single history node (and its descendants) from the history tree.
        """
        if not node_id or not self.history_manager:
            logger.warning("AppController.delete_history_step called with invalid parameters.")
            return False

        previous_node = self.history_service.get_current_node()
        previous_original_id = previous_node.original_image_id if previous_node else None

        result = self.history_service.delete_node_branch(node_id)
        if not result:
            logger.warning("AppController.delete_history_step: HistoryManager reported failure.")
            return False

        self._handle_history_deletion_result(result, previous_original_id)
        return True

    def delete_original_image(self, image_id: str, node_id: Optional[str] = None) -> bool:
        """
        Deletes an entire original image branch from the history.
        """
        if not image_id or not self.history_manager:
            logger.warning("AppController.delete_original_image called with invalid parameters.")
            return False

        previous_node = self.history_service.get_current_node()
        previous_original_id = previous_node.original_image_id if previous_node else None

        result = None
        if node_id:
            node = self.history_service.get_node_by_id(node_id)
            if node and node.original_image_id == image_id and (node.parent_id is None or node.operation_name == "Original"):
                result = self.history_service.delete_node_branch(node_id)
        if result is None:
            result = self.history_service.delete_original_image_branch(image_id)

        if not result:
            logger.warning("AppController.delete_original_image: HistoryManager reported failure.")
            return False

        self._handle_history_deletion_result(result, previous_original_id)
        return True

    def _handle_history_deletion_result(self, result: Dict[str, Any], previous_original_id: Optional[str]):
        """
        Applies controller-level housekeeping after history nodes have been deleted.
        """
        removed_image_id = result.get("removed_original_image_id")
        new_current_id = result.get("new_current_node_id")

        if removed_image_id is not None or new_current_id is None:
            logger.info("AppController: Clearing spot data after history deletion (original removed or no nodes remain).")
            self.clear_all_spot_data()
            return

        current_node = self.history_service.get_current_node()
        current_original_id = current_node.original_image_id if current_node else None

        if previous_original_id and current_original_id != previous_original_id:
            logger.info("AppController: Original image context changed after deletion; clearing spot data.")
            self.clear_all_spot_data()

    def get_current_image_data_for_processing(self) -> Optional[Any]:
        """Gets the image data from the current history node for processing."""
        image_copy = self.history_service.get_current_image_data_copy()
        if image_copy is not None:
            return image_copy
        logger.warning("AppController: No current image data available for processing.")
        return None

    def get_current_node_info_for_dialogs(self) -> Optional[Tuple[str, str, Any, Optional[str], Optional[str]]]:
        """
        Returns information about the current node needed to open dialogs.
        Returns: Tuple (node_id, node_data_type, image_data_copy, original_image_id, original_image_label) or None.
        """
        return self.history_service.get_current_node_info_for_dialogs()
    
    def load_metadata_into_session(self):
        """
        Load metadata from a selected .stp file and attach it to the
        root history node so that a later save is possible.
        """
        root_node = self.history_service.get_root_node(self.history_manager.current_node_id)
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
                root_node.parameters["raw_header"] = temp_stm_image.raw_header
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
        self.session_service.reset_session()

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

            self.original_file_path = file_path

            display_name = self.history_service.get_next_original_display_name()
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
            record.extra_metadata["source_image_id"] = record.image_id
            root_params["source_image_id"] = record.image_id

            root_node = HistoryNode(
                operation_name="Original",
                image_data=stm_image_obj.data.copy(),
                parameters=root_params,
                data_type="STM",
                original_image_id=record.image_id,
            )

            self.session_service.register_new_original(record, root_node)

            logger.info(
                "AppController: File '%s' registered as %s (node_id=%s).",
                os.path.basename(file_path),
                display_name,
                root_node.node_id,
            )
            self.file_loaded_successfully.emit(f"{display_name} - {os.path.basename(file_path)}")

        except FileNotFoundError:
            logger.error(f"AppController: File not found: {file_path}")
            self.file_loading_failed.emit(f"File not found: {file_path}")
        except ValueError as ve:
            logger.error(f"AppController: Value error while loading file {file_path}: {ve}")
            self.file_loading_failed.emit(f"Format error in file {file_path}: {ve}")
        except Exception as e:
            logger.exception(f"AppController: An unexpected error occurred while loading file {file_path}: {e}")
            self.file_loading_failed.emit(f"Unexpected error loading file: {e}")

    def update_substrate_analysis_results(self, results: Dict[str, Any]):
        """
        Updates the substrate state based on the results from SubstrateSpotSelectionDialog.
        """
        def _normalise_covariances(
            source_list: Optional[List[Optional[np.ndarray]]],
            expected_len: int,
        ) -> List[Optional[np.ndarray]]:
            covariances: List[Optional[np.ndarray]] = []
            if source_list:
                for cov in source_list:
                    if cov is None:
                        covariances.append(None)
                    else:
                        covariances.append(np.array(cov, dtype=float))
            if len(covariances) < expected_len:
                covariances.extend([None] * (expected_len - len(covariances)))
            elif len(covariances) > expected_len:
                covariances = covariances[:expected_len]
            return covariances

        new_user_spots = results.get("spots", [])
        new_lattice_type = results.get("lattice_type")
        new_a_surf = results.get("a_surf")
        new_def_name = results.get("substrate_definition", PREDEFINED_SUBSTRATE_NONE)
        
        new_F_m2i = results.get("transformation_F_m2i")
        new_t_m2i = results.get("translation_t_m2i")
        new_analysis_m2i = results.get("transform_analysis_m2i")
        new_displayable_fitted_spots = results.get("displayable_fitted_spots", [])
        new_custom_definition = results.get("custom_definition")
        new_spot_covariances = results.get("spot_covariances", [])
        new_fitted_covariances = results.get("fitted_spot_covariances", [])

        if new_lattice_type == LATTICE_TYPE_CUSTOM and isinstance(new_custom_definition, dict):
            new_a_length = new_custom_definition.get("a_length_nm")
            if new_a_length:
                new_a_surf = new_a_length

        previous_custom_info = self.custom_lattice_info

        spots_changed = (self.user_selected_substrate_spots != new_user_spots)
        def_changed = (self.substrate_lattice_type != new_lattice_type or
                       self.substrate_a_surf != new_a_surf or
                       self.substrate_definition_name != new_def_name)
        if new_def_name == PREDEFINED_SUBSTRATE_CUSTOM and new_lattice_type == LATTICE_TYPE_CUSTOM:
            def_changed = def_changed or previous_custom_info != new_custom_definition
        transform_changed = (not np.array_equal(self.substrate_F_m2i, new_F_m2i) or
                             not np.array_equal(self.substrate_t_m2i, new_t_m2i) or
                             self.displayable_fitted_substrate_spots_on_fft != new_displayable_fitted_spots)

        self.user_selected_substrate_spots = list(new_user_spots)
        self.user_selected_substrate_covariances = _normalise_covariances(
            new_spot_covariances,
            len(self.user_selected_substrate_spots),
        )
        self.substrate_lattice_type = new_lattice_type
        self.substrate_a_surf = new_a_surf
        self.substrate_definition_name = new_def_name
        
        self.substrate_F_m2i = new_F_m2i
        self.substrate_t_m2i = new_t_m2i
        self.substrate_transform_analysis_m2i = new_analysis_m2i
        self.displayable_fitted_substrate_spots_on_fft = list(new_displayable_fitted_spots)
        self.fitted_substrate_spot_covariances = _normalise_covariances(
            new_fitted_covariances,
            len(self.displayable_fitted_substrate_spots_on_fft),
        )
        if new_displayable_fitted_spots and new_user_spots:
            pair_count = min(len(new_user_spots), len(new_displayable_fitted_spots))
            self.substrate_spot_pairs = [
                (tuple(new_user_spots[i]), tuple(new_displayable_fitted_spots[i]))
                for i in range(pair_count)
            ]
        else:
            self.substrate_spot_pairs = []

        if new_def_name == PREDEFINED_SUBSTRATE_CUSTOM:
            if new_lattice_type == LATTICE_TYPE_CUSTOM and isinstance(new_custom_definition, dict):
                self.custom_lattice_info = dict(new_custom_definition)
                self.custom_lattice_info.setdefault("name", "Manual Definition")
                self.custom_lattice_info.setdefault("source", "User Defined")
            elif new_a_surf:
                self.custom_lattice_info = {"type": new_lattice_type, "a_surf": new_a_surf, "name": "Custom (Dialog)"}
            else:
                self.custom_lattice_info = None
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

    def _ensure_metric_sigma_fields(self, result: Dict[str, Any]) -> None:
        """
        Populate scalar sigma fields (a1/a2/alpha) when missing.

        When the metric covariance is absent but g-vector covariances are supplied, the method
        derives the covariance (and sigmas) by invoking ``compute_real_space_metric_uncertainty``.
        """
        if not isinstance(result, dict):
            return

        covariance = result.get("real_space_metric_covariance")
        if covariance is None:
            g1_cov = result.get("g1_vec_cov_nm_inv")
            g2_cov = result.get("g2_vec_cov_nm_inv")
            g1_vec = result.get("g1_vec_nm_inv")
            g2_vec = result.get("g2_vec_nm_inv")

            try:
                g1_cov_arr = np.asarray(g1_cov, dtype=float) if g1_cov is not None else None
                g2_cov_arr = np.asarray(g2_cov, dtype=float) if g2_cov is not None else None
                g1_vec_arr = np.asarray(g1_vec, dtype=float)
                g2_vec_arr = np.asarray(g2_vec, dtype=float)
            except (TypeError, ValueError):
                g1_cov_arr = g2_cov_arr = None
            else:
                if (
                    g1_cov_arr is not None
                    and g2_cov_arr is not None
                    and g1_cov_arr.shape == (2, 2)
                    and g2_cov_arr.shape == (2, 2)
                    and g1_vec_arr.shape == (2,)
                    and g2_vec_arr.shape == (2,)
                ):
                    combined_cov = np.zeros((4, 4), dtype=float)
                    combined_cov[:2, :2] = g1_cov_arr
                    combined_cov[2:, 2:] = g2_cov_arr
                    try:
                        propagation = compute_real_space_metric_uncertainty(
                            (float(g1_vec_arr[0]), float(g1_vec_arr[1])),
                            (float(g2_vec_arr[0]), float(g2_vec_arr[1])),
                            combined_cov,
                        )
                    except ValueError:
                        propagation = None
                    if propagation is not None:
                        covariance = propagation.covariance
                        result["real_space_metric_covariance"] = covariance

        if covariance is None:
            return

        try:
            cov_arr = np.asarray(covariance, dtype=float)
        except (TypeError, ValueError):
            return

        if cov_arr.ndim != 2 or cov_arr.shape[0] < 3 or cov_arr.shape[1] < 3:
            return

        diag_values = np.clip(np.diag(cov_arr)[:3], 0.0, None)
        keys = ("a1_nm_sigma", "a2_nm_sigma", "alpha_deg_sigma")
        for idx, key in enumerate(keys):
            if result.get(key) is not None:
                continue
            variance = diag_values[idx]
            result[key] = float(np.sqrt(variance)) if variance >= 0.0 else None
        for key in keys:
            if result.get(key) is None:
                result[key] = 0.0

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

        if self.substrate_lattice_type == LATTICE_TYPE_CUSTOM:
            if not isinstance(self.custom_lattice_info, dict):
                logger.warning("Custom lattice definition missing; cannot calculate real space params.")
                self.substrate_real_space_params_updated.emit({"error": "Custom lattice definition missing."})
                return
        elif not (self.substrate_lattice_type and self.substrate_a_surf and self.substrate_a_surf > 0):
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

        fitted_covariances_relative_px: Optional[List[Optional[np.ndarray]]] = None
        if self.fitted_substrate_spot_covariances:
            cov_list = [
                np.array(cov, dtype=float) if cov is not None else None
                for cov in self.fitted_substrate_spot_covariances
            ]
            if len(cov_list) < len(fitted_g_vectors_relative_px):
                cov_list.extend([None] * (len(fitted_g_vectors_relative_px) - len(cov_list)))
            elif len(cov_list) > len(fitted_g_vectors_relative_px):
                cov_list = cov_list[: len(fitted_g_vectors_relative_px)]
            fitted_covariances_relative_px = cov_list

        expected_spot_count = 0
        if self.substrate_lattice_type == LATTICE_TYPE_HEXAGONAL: expected_spot_count = 6
        elif self.substrate_lattice_type == LATTICE_TYPE_SQUARE: expected_spot_count = 4
        
        if len(fitted_g_vectors_relative_px) != expected_spot_count and expected_spot_count > 0:
            logger.warning(f"Incorrect number of fitted spots ({len(fitted_g_vectors_relative_px)}) "
                           f"for {self.substrate_lattice_type} (expected {expected_spot_count}). Cannot calculate real space params.")
            self.substrate_real_space_params_updated.emit({"error": f"Need {expected_spot_count} fitted spots."})
            return
        if expected_spot_count == 0 and len(fitted_g_vectors_relative_px) < 2 :
             logger.warning("Not enough fitted spots to define basis for non-standard lattice type.")
             self.substrate_real_space_params_updated.emit({"error": "Need >=2 fitted spots."})
             return


        print(f"displayable_fitted_substrate_spots_on_fft: {self.displayable_fitted_substrate_spots_on_fft}")
        print(f"fitted_g_vectors_relative_px: {fitted_g_vectors_relative_px}")
        sigma_Lx_nm, sigma_Ly_nm = self._resolve_pixel_calibration_uncertainty(root_node)

        results = get_real_space_lattice_parameters(
            selected_g_vectors_relative_px=fitted_g_vectors_relative_px,
            lattice_type=self.substrate_lattice_type,
            Lx_nm=Lx_nm,
            Ly_nm=Ly_nm,
            fft_shape_cols_kx=fft_cols_kx,
            fft_shape_rows_ky=fft_rows_ky,
            selected_g_vector_covariances_px=fitted_covariances_relative_px,
            Lx_sigma_nm=sigma_Lx_nm,
            Ly_sigma_nm=sigma_Ly_nm,
        )

        if results:
            self._ensure_metric_sigma_fields(results)
            self.substrate_real_space_results = results
            logger.info(f"Successfully calculated substrate real space parameters (from fitted spots): {results}")
            self.substrate_real_space_params_updated.emit(results)
        else:
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

        if not (0 <= set_index < len(self.corrected_adsorbate_spot_sets)):
            logger.warning(f"Invalid set_index {set_index} for adsorbate real space params.")
            self.adsorbate_real_space_params_updated.emit(set_index, {"error": "Invalid set index."})
            return
        
        corrected_spots_ideal_px_abs = self.corrected_adsorbate_spot_sets[set_index]
        corrected_covariances_px = []
        if set_index < len(self.corrected_adsorbate_covariance_sets):
            raw_cov_list = self.corrected_adsorbate_covariance_sets[set_index]
            corrected_covariances_px = [
                np.array(cov, dtype=float) if cov is not None else None
                for cov in raw_cov_list
            ]
        else:
            corrected_covariances_px = []
        
        expected_ads_type = self.adsorbate_expected_lattice_types.get(set_index, ADSORBATE_LATTICE_TYPE_UNKNOWN)
        
        num_corrected_spots = len(corrected_spots_ideal_px_abs)

        if num_corrected_spots < 2:
            logger.warning(f"Not enough corrected adsorbate spots (need >=2, got {num_corrected_spots}) for set {set_index}.")
            self.adsorbate_real_space_params_updated.emit(set_index, {"error": f"Need >= 2 corrected spots, got {num_corrected_spots}."})
            return

        if not self.current_fft_data_shape:
            logger.warning("Current FFT data shape not available."); self.adsorbate_real_space_params_updated.emit(set_index, {"error": "FFT shape missing."}); return

        root_node = self.history_manager.get_root_node_for_node(self.history_manager.get_current_node().node_id) # type: ignore
        if not (root_node and root_node.parameters): 
            logger.warning("Cannot get Lx, Ly for adsorbate.")
            self.adsorbate_real_space_params_updated.emit(set_index, {"error": "Original node params missing."})
            return
        Lx_nm = root_node.parameters.get("size_nm_x")
        Ly_nm = root_node.parameters.get("size_nm_y")
        if not (Lx_nm and Ly_nm and Lx_nm > 0 and Ly_nm > 0): 
            logger.warning("Invalid Lx/Ly.")
            self.adsorbate_real_space_params_updated.emit(set_index, {"error": "Invalid Lx/Ly."})
            return
        sigma_Lx_nm, sigma_Ly_nm = self._resolve_pixel_calibration_uncertainty(root_node)
        
        fft_rows_ky, fft_cols_kx = self.current_fft_data_shape
        center_kx_ideal_px = fft_cols_kx / 2.0
        center_ky_ideal_px = fft_rows_ky / 2.0
        g_vectors_adsorbate_relative_px = [
            (spot_abs_kx - center_kx_ideal_px, spot_abs_ky - center_ky_ideal_px)
            for spot_abs_kx, spot_abs_ky in corrected_spots_ideal_px_abs
        ]
        g_vector_covariances_adsorbate_px: List[Optional[np.ndarray]] = []
        if corrected_covariances_px:
            cov_list = corrected_covariances_px
            if len(cov_list) < len(g_vectors_adsorbate_relative_px):
                cov_list.extend([None] * (len(g_vectors_adsorbate_relative_px) - len(cov_list)))
            elif len(cov_list) > len(g_vectors_adsorbate_relative_px):
                cov_list = cov_list[: len(g_vectors_adsorbate_relative_px)]
            g_vector_covariances_adsorbate_px = cov_list
        else:
            g_vector_covariances_adsorbate_px = [None] * len(g_vectors_adsorbate_relative_px)

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

        if g1_ads_nm_inv is None or g2_ads_nm_inv is None:
            self.adsorbate_real_space_params_updated.emit(set_index, {"error": "g-vector conversion to nm^-1 failed."}); return

        real_space_vecs_ads = calculate_real_space_vectors_from_g(g1_ads_nm_inv, g2_ads_nm_inv)
        if real_space_vecs_ads is None:
            self.adsorbate_real_space_params_updated.emit(set_index, {"error": "Real space vector calculation failed (g-vectors likely collinear)."}); return
        a1_ads_vec_nm, a2_ads_vec_nm = real_space_vecs_ads

        a1_ads_mag_nm = np.linalg.norm(a1_ads_vec_nm)
        a2_ads_mag_nm = np.linalg.norm(a2_ads_vec_nm)
        
        if a1_ads_mag_nm < 1e-9 or a2_ads_mag_nm < 1e-9:
            logger.warning(f"Calculated real space vectors for adsorbate set {set_index} have zero or near-zero magnitude.")
            self.adsorbate_real_space_params_updated.emit(set_index, {"error": "Calculated real vectors too short."}); return

        dot_product_ads = np.dot(a1_ads_vec_nm, a2_ads_vec_nm)
        cos_alpha_ads = np.clip(dot_product_ads / (a1_ads_mag_nm * a2_ads_mag_nm), -1.0, 1.0)
        alpha_ads_deg = np.degrees(np.arccos(cos_alpha_ads))

        def _estimate_covariance_for_basis(
            basis_vec: Tuple[float, float],
        ) -> Optional[np.ndarray]:
            basis_arr = np.array(basis_vec, dtype=float)
            norm_basis = np.linalg.norm(basis_arr)
            if norm_basis < 1e-9:
                return None
            best_idx = None
            best_cos = -np.inf
            for idx, orig_vec in enumerate(g_vectors_adsorbate_relative_px):
                norm_orig = np.linalg.norm(orig_vec)
                if norm_orig < 1e-9:
                    continue
                cos_val = abs(np.dot(basis_arr, orig_vec) / (norm_basis * norm_orig))
                if cos_val > best_cos:
                    best_cos = cos_val
                    best_idx = idx
            if best_idx is None:
                return None
            base_cov = (
                g_vector_covariances_adsorbate_px[best_idx]
                if best_idx < len(g_vector_covariances_adsorbate_px)
                else None
            )
            if base_cov is None:
                return None
            norm_orig = np.linalg.norm(g_vectors_adsorbate_relative_px[best_idx])
            if norm_orig < 1e-9:
                return None
            scale = norm_basis / norm_orig
            return (scale ** 2) * base_cov

        g1_cov_ads_px = _estimate_covariance_for_basis(g1_ads_px)
        g2_cov_ads_px = _estimate_covariance_for_basis(g2_ads_px)
        if g2_cov_ads_px is None and g1_cov_ads_px is not None:
            g1_arr = np.array(g1_ads_px, dtype=float)
            g2_arr = np.array(g2_ads_px, dtype=float)
            norm_g1 = np.linalg.norm(g1_arr)
            norm_g2 = np.linalg.norm(g2_arr)
            if norm_g1 > 1e-9 and norm_g2 > 1e-9:
                theta = np.arctan2(g2_arr[1], g2_arr[0]) - np.arctan2(g1_arr[1], g1_arr[0])
                rotation_matrix = np.array(
                    [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
                    dtype=float,
                )
                scale = norm_g2 / norm_g1
                transform = scale * rotation_matrix
                g2_cov_ads_px = transform @ g1_cov_ads_px @ transform.T

        scale_matrix = np.array([[1.0 / Lx_nm, 0.0], [0.0, 1.0 / Ly_nm]], dtype=float)
        g1_cov_ads_nm = scale_matrix @ g1_cov_ads_px @ scale_matrix.T if g1_cov_ads_px is not None else None
        g2_cov_ads_nm = scale_matrix @ g2_cov_ads_px @ scale_matrix.T if g2_cov_ads_px is not None else None

        g1_cov_ads_nm = augment_covariance_with_calibration(
            g1_ads_px,
            g1_cov_ads_nm,
            Lx_nm,
            Ly_nm,
            sigma_Lx_nm,
            sigma_Ly_nm,
        )
        g2_cov_ads_nm = augment_covariance_with_calibration(
            g2_ads_px,
            g2_cov_ads_nm,
            Lx_nm,
            Ly_nm,
            sigma_Lx_nm,
            sigma_Ly_nm,
        )
        g1_cov_ads_nm = apply_k_resolution_floor_to_covariance(g1_cov_ads_nm, Lx_nm, Ly_nm)
        g2_cov_ads_nm = apply_k_resolution_floor_to_covariance(g2_cov_ads_nm, Lx_nm, Ly_nm)

        metrics_uncertainty = None
        if g1_cov_ads_nm is not None and g2_cov_ads_nm is not None:
            combined_covariance = np.zeros((4, 4), dtype=float)
            combined_covariance[:2, :2] = g1_cov_ads_nm
            combined_covariance[2:, 2:] = g2_cov_ads_nm
            try:
                metrics_uncertainty = compute_real_space_metric_uncertainty(
                    g1_ads_nm_inv,
                    g2_ads_nm_inv,
                    combined_covariance,
                )
            except ValueError as exc:  # pragma: no cover - defensive
                logger.warning("Unable to propagate adsorbate lattice parameter uncertainties: %s", exc)

        results = {
            "a1_nm": a1_ads_mag_nm, "a2_nm": a2_ads_mag_nm, "alpha_deg": alpha_ads_deg,
            "a1_vec_nm": a1_ads_vec_nm, "a2_vec_nm": a2_ads_vec_nm,
            "g1_vec_px_ideal_sys": g1_ads_px,
            "g2_vec_px_ideal_sys": g2_ads_px,
            "g1_vec_nm_inv": g1_ads_nm_inv, 
            "g2_vec_nm_inv": g2_ads_nm_inv,
            "source_corrected_spots_ideal_px": corrected_spots_ideal_px_abs
        }
        if g1_cov_ads_px is not None:
            results["g1_vec_cov_px"] = g1_cov_ads_px
        if g2_cov_ads_px is not None:
            results["g2_vec_cov_px"] = g2_cov_ads_px
        if g1_cov_ads_nm is not None:
            results["g1_vec_cov_nm_inv"] = g1_cov_ads_nm
        if g2_cov_ads_nm is not None:
            results["g2_vec_cov_nm_inv"] = g2_cov_ads_nm
        if metrics_uncertainty is not None:
            metric_cov = np.array(metrics_uncertainty.covariance, dtype=float)
            results["real_space_metric_covariance"] = metric_cov
            diag_entries = np.clip(np.diag(metric_cov), 0.0, None)
            if diag_entries.size >= 3:
                results["a1_nm_sigma"] = float(np.sqrt(diag_entries[0]))
                results["a2_nm_sigma"] = float(np.sqrt(diag_entries[1]))
                results["alpha_deg_sigma"] = float(np.sqrt(diag_entries[2]))
        results["pixel_calibration_sigma_nm"] = (float(sigma_Lx_nm), float(sigma_Ly_nm))
        self._ensure_metric_sigma_fields(results)
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
            data_type="STM"
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
        else:
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
        if method in [
            REFINEMENT_DIRECT_CLICK,
            REFINEMENT_MAX_PIXEL,
            REFINEMENT_GAUSSIAN_FIT,
            REFINEMENT_PARABOLA_3X3,
            REFINEMENT_LOCAL_DFT,
        ]:
            if self.spot_refinement_method != method:
                self.spot_refinement_method = method
                logger.info(f"Spot refinement method set to: {self.spot_refinement_method}")
                self.spot_selection_parameters_changed.emit()
        else:
            logger.warning(f"Attempted to set invalid spot refinement method: {method}")

    def set_refinement_roi_size(self, size: int):
        """Sets the ROI size for spot refinement."""
        if isinstance(size, int) and 3 <= size <= 31 and size % 2 != 0:
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
        self.spot_service.clear_last_adsorbate_spot()

    def reselect_current_adsorbate_set(self):
        """Clears all spots from the current adsorbate set."""
        self.spot_service.clear_current_adsorbate_set()

    def set_expected_adsorbate_lattice_type(self, set_index: int, lattice_type: str):
        """Sets the expected lattice type for a given adsorbate set."""
        self.spot_service.set_expected_adsorbate_lattice_type(set_index, lattice_type)

    def set_pixel_calibration_uncertainty(self, sigma_x_nm: float, sigma_y_nm: float) -> None:
        """Update the stored pixel calibration uncertainties (nm)."""
        try:
            sx = max(float(sigma_x_nm), 0.0)
        except (TypeError, ValueError):
            sx = 0.0
        try:
            sy = max(float(sigma_y_nm), 0.0)
        except (TypeError, ValueError):
            sy = 0.0
        self.pixel_calibration_sigma_nm = (sx, sy)
        logger.info("AppController: Pixel calibration uncertainty set to (%.4f, %.4f) nm", sx, sy)

    def clear_all_adsorbate_sets(self):
        """Clears all adsorbate sets and resets to one empty set."""
        self.spot_service.clear_all_adsorbate_sets()

    def update_adsorbate_set_results(
        self,
        set_index: int,
        raw_spots: List[Tuple[float, float]],
        corrected_spots_ideal_system: List[Tuple[float, float]],
        raw_covariances: Optional[List[Optional[np.ndarray]]] = None,
        corrected_covariances: Optional[List[Optional[np.ndarray]]] = None,
    ):
        """
        Updates the raw and corrected spots for a given adsorbate set.
        """
        self.spot_service.update_adsorbate_set_results(
            set_index=set_index,
            raw_spots=raw_spots,
            corrected_spots_ideal_system=corrected_spots_ideal_system,
            raw_covariances=raw_covariances,
            corrected_covariances=corrected_covariances,
        )

    def update_superstructure_periodicity_results(self, results: Optional[Dict[str, Any]]):
        """Updates and stores the superstructure periodicity analysis results."""
        self.superstructure_periodicity_results = results
        logger.info(f"AppController: Updated superstructure periodicity results: {results}")
        self.superstructure_periodicity_results_updated.emit(self.superstructure_periodicity_results)

    # ------------------------------------------------------------------ Real-space reporting/export helpers
    def get_real_space_summary_text(self) -> str:
        """Return a human-readable summary of available real-space parameters."""
        return build_real_space_summary(
            self.substrate_real_space_results,
            self.adsorbate_real_space_results,
            transform_analysis=self.substrate_transform_analysis_m2i,
        )

    def get_real_space_report_json(self) -> Dict[str, Any]:
        """Return a JSON-serialisable structure describing real-space results."""
        return build_real_space_json(
            self.substrate_real_space_results,
            self.adsorbate_real_space_results,
            transform_analysis=self.substrate_transform_analysis_m2i,
        )

    def get_real_space_report_records(self) -> List[Dict[str, Any]]:
        """Return flattened records suitable for CSV export."""
        return build_real_space_records(
            self.substrate_real_space_results,
            self.adsorbate_real_space_results,
            transform_analysis=self.substrate_transform_analysis_m2i,
        )

    def copy_real_space_summary_to_clipboard(self) -> bool:
        """
        Copy the current real-space summary to the system clipboard.

        Returns:
            bool: True when data was copied, False if no results or clipboard unavailable.
        """
        has_any_results = (
            bool(self.substrate_real_space_results)
            or any(bool(result) for result in self.adsorbate_real_space_results.values())
            or bool(self.substrate_transform_analysis_m2i)
        )
        if not has_any_results:
            logger.info("AppController: No real-space data available to copy to clipboard.")
            return False

        app = QApplication.instance()
        if app is None:
            logger.warning("AppController: Cannot access clipboard (no QApplication instance).")
            return False

        summary = self.get_real_space_summary_text()
        app.clipboard().setText(summary)
        logger.debug("AppController: Real-space summary copied to clipboard.")
        return True

    def export_real_space_report_to_json(self, file_path: str) -> None:
        """Write the real-space results (including uncertainties) to a JSON file."""
        data = self.get_real_space_report_json()
        has_any_results = bool(data.get("substrate")) or any(
            bool(entry) for entry in (data.get("adsorbate") or {}).values()
        )
        if not has_any_results:
            raise ValueError("No real-space results are available to export.")

        with open(file_path, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)
        logger.info("AppController: Real-space report exported to JSON at %s", file_path)

    def export_real_space_report_to_csv(self, file_path: str) -> None:
        """Write the real-space results (including uncertainties) to a CSV file."""
        records = self.get_real_space_report_records()
        if not records:
            raise ValueError("No real-space results are available to export.")

        fieldnames: List[str] = sorted({key for record in records for key in record.keys()})
        with open(file_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)
        logger.info("AppController: Real-space report exported to CSV at %s", file_path)

    def _on_history_active_node_changed(self, event: "ActiveNodeChangedEvent") -> None:
        node = getattr(event, "node", None)
        if node and getattr(node, "data_type", "") == "FFT" and getattr(node, "image_data", None) is not None:
            self.current_fft_data_shape = node.image_data.shape
        else:
            self.current_fft_data_shape = None

    def get_active_history_node(self) -> Optional[HistoryNode]:
        return self.history_service.get_current_node()

    def can_load_metadata(self, history_node: Optional[HistoryNode]) -> bool:
        return self.analysis_executor.can_load_metadata(history_node)

    def can_calculate_fft(self, history_node: Optional[HistoryNode]) -> bool:
        return self.analysis_executor.can_calculate_fft(history_node)

    def can_select_spots(self, history_node: Optional[HistoryNode]) -> bool:
        return self.analysis_executor.can_select_spots(history_node)

    def can_analyze_superstructure(self, history_node: Optional[HistoryNode]) -> bool:
        return self.analysis_executor.can_analyze_superstructure(history_node)

    def can_visualize_real_space(self, history_node: Optional[HistoryNode]) -> bool:
        return self.analysis_executor.can_visualize_real_space(history_node)

    def can_open_real_space_reconstruction(self, history_node: Optional[HistoryNode]) -> bool:
        return self.analysis_executor.can_open_real_space_reconstruction(history_node)

    def can_open_stm_transform(self, history_node: Optional[HistoryNode]) -> bool:
        return self.analysis_executor.can_open_stm_transform(history_node)

    def evaluate_fft_panel_state(
        self,
        history_node: Optional[HistoryNode],
        lattice_analysis_enabled: bool,
    ) -> FFTPanelState:
        """Delegate FFT-panel enablement calculations to SpotSetService."""
        panel_flags = self.spot_service.evaluate_fft_panel_state(
            history_node=history_node,
            lattice_analysis_enabled=lattice_analysis_enabled,
            analysis_functions_available=LATTICE_ANALYSIS_FUNCTIONS_AVAILABLE,
        )
        return FFTPanelState(**panel_flags)

    def _has_valid_substrate_definition(self) -> bool:
        """Return True when the substrate definition contains enough data for analysis."""
        if self.substrate_lattice_type == LATTICE_TYPE_CUSTOM:
            return isinstance(self.custom_lattice_info, dict)
        return bool(
            self.substrate_lattice_type
            and self.substrate_a_surf
            and self.substrate_a_surf > 0
        )

    def _get_fft_data_shape(self, history_node: Optional[HistoryNode]) -> Optional[Tuple[int, int]]:
        """Expose the FFT data shape, falling back to cached values when nodes lack arrays."""
        if history_node and history_node.data_type == "FFT" and getattr(history_node, "image_data", None) is not None:
            return history_node.image_data.shape
        return self.current_fft_data_shape

    def _resolve_pixel_calibration_uncertainty(self, root_node: Optional[HistoryNode]) -> Tuple[float, float]:
        """Return the (sigma_x, sigma_y) calibration uncertainties in nanometres."""
        default_x, default_y = self.pixel_calibration_sigma_nm
        params = getattr(root_node, "parameters", {}) if root_node else {}

        def _coerce(value, default):
            if value is None:
                return float(default)
            try:
                return max(float(value), 0.0)
            except (TypeError, ValueError):
                return float(default)

        sigma_x = _coerce(params.get("size_nm_x_sigma"), default_x)
        sigma_y = _coerce(params.get("size_nm_y_sigma"), default_y)
        return (sigma_x, sigma_y)

    def _can_calculate_substrate_real_space(self, history_node: HistoryNode) -> bool:
        """Check whether current data is sufficient to compute substrate real-space metrics."""
        return self.analysis_executor.can_calculate_substrate_real_space(history_node)

    def _can_calculate_adsorbate_real_space(self, history_node: HistoryNode) -> bool:
        """Check whether current data is sufficient to compute adsorbate real-space metrics."""
        return self.analysis_executor.can_calculate_adsorbate_real_space(history_node)

    def add_new_node_to_history(self, new_node: HistoryNode):
        """
        Add a prepared history node and make it the current entry.
        """
        if self.history_manager:
            self.history_service.add_node_and_select(new_node)
            logger.info("AppController: Added new node '%s' to history.", new_node.operation_name)

    def add_new_adsorbate_set(self):
        """Adds a new, empty adsorbate spot set and sets it as current."""
        self.spot_service.add_new_adsorbate_set()


    def set_current_adsorbate_set_by_index(self, index: int):
        """Sets the current adsorbate set based on the index."""
        self.spot_service.set_current_adsorbate_set_by_index(index)


    def clear_all_spot_data(self):
        """Clears all spot data."""
        self.spot_service.clear_all_spot_data()

    def export_session_state(self) -> "ControllerState":
        from .session_state import ControllerState

        return ControllerState.from_controller(self)

    def load_session_state(self, state: "ControllerState") -> None:
        from .session_state import ControllerState

        if not isinstance(state, ControllerState):
            raise TypeError("Expected ControllerState when loading session data.")
        state.apply_to(self)









