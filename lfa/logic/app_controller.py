# lfa/logic/app_controller.py
"""
Central controller for the LFA application.
Manages application state and coordinates operations between UI and backend modules.
"""
import logging
import os
import numpy as np

from typing import Optional, List, Tuple, Dict, Any

from PyQt6.QtCore import QObject, pyqtSignal

from ..core.data_models import STMImage # Potrzebne do type hinting i tworzenia obiektu
from ..io.factory import load_stm_file  # Funkcja do ładowania pliku
from ..core.history import HistoryNode   # Do tworzenia węzła historii

logger = logging.getLogger(__name__)

class AppController(QObject):
    """
    Manages the core application state and logic, acting as a bridge
    between the GUI (MainWindow) and the backend processing/analysis modules.
    """

    file_loaded_successfully = pyqtSignal(str)
    file_loading_failed = pyqtSignal(str)   

    def __init__(self, history_manager, parent: Optional[QObject] = None):
        """
        Initializes the AppController.

        Args:
            history_manager (HistoryManager): Instance of the history manager.
            parent (Optional[QObject]): Parent object for Qt memory management.
        """
        super().__init__(parent)
        
        # Referencja do HistoryManager - kluczowa dla wielu operacji
        # Zakładamy, że HistoryManager jest już skonfigurowany z QListWidget
        self.history_manager = history_manager # Typowanie można dodać: HistoryManager

        # --- Atrybuty Przeniesione z MainWindow (Zarządzanie Stanem Aplikacji) ---
        self.original_file_path: Optional[str] = None

        # Dane dotyczące wyboru pików
        self.substrate_spots: List[Tuple[float, float]] = []
        self.adsorbate_spot_sets: List[List[Tuple[float, float]]] = [[]] # Zawsze zaczynamy z jednym pustym zestawem
        self.current_adsorbate_set_index: int = 0

        # Tryby i parametry wyboru/uściślania pików
        self.spot_selection_mode: str = "Substrate"  # Domyślnie "Substrate" lub "Adsorbate"
        self.spot_refinement_method: str = "Direct Click" # Domyślnie, inne opcje: "Max Pixel", "2D Gaussian Fit"
        self.refinement_roi_size: int = 5 # Domyślny rozmiar (np. średnica) obszaru ROI do uściślania

        # Ustawienia dotyczące idealnej sieci i substratu
        self.custom_lattice_info: Optional[Dict[str, Any]] = None # Dla definicji własnej sieci
        self.last_selected_substrate: str = "None" # Ostatnio wybrany substrat z ComboBoxa (lub nazwa custom)
        
        # Dodatkowe stany, które mogą być zarządzane centralnie
        self.show_ideal_lattice: bool = True # Czy pokazywać idealną sieć na FFT
        self.show_substrate_spots_markers: bool = True # Widoczność markerów substratu
        self.show_adsorbate_spots_markers: bool = True # Widoczność markerów adsorbatu

        logger.info("AppController initialized.")

    def get_current_image_data_for_processing(self) -> Optional[Any]: # Any to tymczasowo np.ndarray
        """Pobiera dane obrazu z bieżącego węzła historii do przetwarzania."""
        current_node = self.history_manager.get_current_node()
        if current_node and current_node.image_data is not None:
            return current_node.image_data.copy() # Zwróć kopię, aby uniknąć modyfikacji oryginału
        logger.warning("AppController: No current image data available for processing.")
        return None

    def get_current_node_info_for_dialogs(self) -> Optional[Tuple[str, str, Any]]: # Any to np.ndarray
        """
        Zwraca informacje o bieżącym węźle potrzebne do otwarcia dialogów.
        Returns: Tuple (node_id, node_data_type, image_data_copy) or None.
        """
        current_node = self.history_manager.get_current_node()
        if current_node and current_node.image_data is not None:
            return current_node.node_id, current_node.data_type, current_node.image_data.copy()
        return None
    
    def load_file(self, file_path: str):
        """
        Loads an STM file, creates an initial history node, and updates the application state.
        Emits signals indicating success or failure.
        """
        logger.info(f"AppController: Attempting to load file: {file_path}")
        try:
            stm_image_obj: Optional[STMImage] = load_stm_file(file_path)

            if stm_image_obj and stm_image_obj.data is not None:
                self.original_file_path = file_path # Ustaw ścieżkę w kontrolerze

                self.history_manager.clear_history() # Wyczyść poprzednią historię

                # Przygotuj parametry dla węzła "Original"
                # Te parametry są istotne dla MetadataWidget i potencjalnie dla innych operacji
                root_params = {
                    "filename": os.path.basename(file_path),
                    "pixels_x": stm_image_obj.pixels_x,
                    "pixels_y": stm_image_obj.pixels_y,
                    "size_nm_x": stm_image_obj.size_nm_x,
                    "size_nm_y": stm_image_obj.size_nm_y,
                    "bias_v": stm_image_obj.bias_v,
                    "setpoint_a": stm_image_obj.setpoint_a,
                    "scan_angle_deg": stm_image_obj.scan_angle_deg,
                    # Można dodać więcej standardowych pól z STMImage, jeśli są potrzebne
                }

                root_node = HistoryNode(
                    operation_name="Original", # Nazwa operacji dla korzenia
                    image_data=stm_image_obj.data.copy(), # Zawsze pracuj na kopiach danych
                    parameters=root_params, # Przechowaj kluczowe metadane jako parametry
                    data_type="STM" # Typ danych
                )
                
                self.history_manager.add_node(root_node)
                self.history_manager.set_current_node_by_id(root_node.node_id) # Ustaw jako bieżący
                                                                                # To wyemituje HistoryManager.current_node_changed

                logger.info(f"AppController: File '{os.path.basename(file_path)}' loaded successfully.")
                self.file_loaded_successfully.emit(os.path.basename(file_path))
            else:
                # Jeśli load_stm_file zwróci None lub stm_image_obj.data jest None
                logger.error(f"AppController: Failed to load valid data from file: {file_path}")
                self.history_manager.clear_history() # Wyczyść historię w razie niepowodzenia
                self.original_file_path = None
                self.file_loading_failed.emit(f"Could not load valid data from file: {file_path}")
        
        except FileNotFoundError: # pragma: no cover
            logger.error(f"AppController: File not found: {file_path}")
            self.file_loading_failed.emit(f"File not found: {file_path}")
        except ValueError as ve: # pragma: no cover
            # Np. błędy parsowania nagłówka zgłaszane przez czytniki
            logger.error(f"AppController: Value error while loading file {file_path}: {ve}")
            self.file_loading_failed.emit(f"Format error in file {file_path}: {ve}")
        except Exception as e: # pragma: no cover
            # Inne nieoczekiwane błędy
            logger.exception(f"AppController: An unexpected error occurred while loading file {file_path}: {e}")
            self.file_loading_failed.emit(f"Unexpected error loading file: {e}")


    def add_operation_to_history(self,
                                 parent_node_id: str,
                                 op_name: str,
                                 params: Dict[str, Any],
                                 processed_data: np.ndarray,
                                 data_type: str, # "STM" lub "FFT"
                                 source_roi_slice: Optional[Tuple[slice, slice]] = None):
        """
        Tworzy nowy węzeł historii dla wykonanej operacji i dodaje go do menedżera.

        Args:
            parent_node_id (str): ID węzła rodzica.
            op_name (str): Nazwa wykonanej operacji (np. "Gaussian Blur", "FFT").
            params (Dict[str, Any]): Słownik parametrów użytych do operacji.
            processed_data (np.ndarray): Wynikowe dane obrazu (np.ndarray).
            data_type (str): Typ danych w `processed_data` ("STM" lub "FFT").
            source_roi_slice (Optional[Tuple[slice, slice]]): Jeśli operacja była na ROI,
                                                              przekaż wycinek ROI.
        """
        if processed_data is None:
            logger.warning(f"AppController: No processed data provided for operation '{op_name}'. Node not added.")
            return

        # Sprawdzenie, czy dane faktycznie się zmieniły (opcjonalne, ale może być przydatne)
        parent_node = self.history_manager.get_node_by_id(parent_node_id)
        if parent_node and parent_node.image_data is not None:
            if np.array_equal(processed_data, parent_node.image_data) and \
               params == parent_node.parameters.get(op_name, {}): # Proste porównanie parametrów
                logger.info(f"AppController: Data for '{op_name}' has not changed. Node not added.")
                return

        new_node = HistoryNode(
            parent_id=parent_node_id,
            operation_name=op_name,
            parameters=params,
            image_data=processed_data, # Zakładamy, że to już jest kopia, jeśli trzeba
            data_type=data_type,
            source_roi_slice=source_roi_slice
        )

        self.history_manager.add_node(new_node)
        self.history_manager.set_current_node_by_id(new_node.node_id)
        logger.info(f"AppController: Added '{op_name}' node (ID: {new_node.node_id}) to history.")
        # Sygnał current_node_changed z HistoryManager powinien wystarczyć do aktualizacji UI.

    def apply_gaussian_blur(self, parent_node_id: str, parent_data_type: str,
                            processed_data: np.ndarray, params: Dict[str, Any],
                            source_roi_slice: Optional[Tuple[slice, slice]] = None):
        self.add_operation_to_history(parent_node_id, "Gaussian Blur", params, processed_data, parent_data_type, source_roi_slice)

    def apply_gaussian_sharpening(self, parent_node_id: str, parent_data_type: str,
                                  processed_data: np.ndarray, params: Dict[str, Any],
                                  source_roi_slice: Optional[Tuple[slice, slice]] = None):
        self.add_operation_to_history(parent_node_id, "Gaussian Sharpening", params, processed_data, parent_data_type, source_roi_slice)

    def apply_plane_leveling(self, parent_node_id: str, parent_data_type: str,
                             processed_data: np.ndarray, params: Dict[str, Any],
                             source_roi_slice: Optional[Tuple[slice, slice]] = None):
        # Params mogą zawierać 'mode' i 'points'
        self.add_operation_to_history(parent_node_id, "Plane Leveling", params, processed_data, parent_data_type, source_roi_slice)

    def apply_median_filter(self, parent_node_id: str, parent_data_type: str,
                            processed_data: np.ndarray, params: Dict[str, Any],
                            source_roi_slice: Optional[Tuple[slice, slice]] = None):
        self.add_operation_to_history(parent_node_id, "Median Filter", params, processed_data, parent_data_type, source_roi_slice)

    def apply_nlmeans_denoising(self, parent_node_id: str, parent_data_type: str,
                                processed_data: np.ndarray, params: Dict[str, Any],
                                source_roi_slice: Optional[Tuple[slice, slice]] = None):
        self.add_operation_to_history(parent_node_id, "NL-Means", params, processed_data, parent_data_type, source_roi_slice)

    def apply_bm3d_denoising(self, parent_node_id: str, parent_data_type: str,
                             processed_data: np.ndarray, params: Dict[str, Any],
                             source_roi_slice: Optional[Tuple[slice, slice]] = None):
        self.add_operation_to_history(parent_node_id, "BM3D", params, processed_data, parent_data_type, source_roi_slice)

    def calculate_fft_operation(self, parent_node_id: str, # parent_data_type będzie zawsze "STM" dla FFT
                                processed_fft_data: np.ndarray, # To są już przeskalowane dane magnitudy
                                params: Dict[str, Any], # Zawiera window, scaling_mode, apply_roi_only
                                source_roi_slice: Optional[Tuple[slice, slice]] = None):
        # Dla FFT, data_type nowego węzła to "FFT"
        self.add_operation_to_history(parent_node_id, "FFT", params, processed_fft_data, "FFT", source_roi_slice)

