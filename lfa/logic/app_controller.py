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
from ..gui.dialogs.substrate_spot_dialog import PREDEFINED_SUBSTRATE_NONE, PREDEFINED_SUBSTRATE_CUSTOM

logger = logging.getLogger(__name__)

SPOT_SELECTION_SUBSTRATE = "Substrate"
SPOT_SELECTION_ADSORBATE = "Adsorbate"

REFINEMENT_DIRECT_CLICK = "Direct Click"
REFINEMENT_MAX_PIXEL = "Max Pixel"
REFINEMENT_GAUSSIAN_FIT = "2D Gaussian Fit"

MAX_SUBSTRATE_SPOTS = 8 # Maksymalna liczba pików substratu

class AppController(QObject):
    """
    Manages the core application state and logic, acting as a bridge
    between the GUI (MainWindow) and the backend processing/analysis modules.
    """

    file_loaded_successfully = pyqtSignal(str)
    file_loading_failed = pyqtSignal(str)   

    # Sygnał emitowany po zmianie list pików (substratu lub któregokolwiek zestawu adsorbatu)
    spot_lists_updated = pyqtSignal()
    # Sygnał emitowany po zmianie trybu wyboru pików (Substrate/Adsorbate) lub metody uściślania
    spot_selection_parameters_changed = pyqtSignal()
    # Sygnał emitowany po zmianie bieżącego zestawu adsorbatu (np. dodanie nowego, zmiana indeksu)
    adsorbate_sets_structure_changed = pyqtSignal()
    substrate_transform_results_updated = pyqtSignal()

    substrate_definition_changed = pyqtSignal()

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

        self.current_substrate_a_surf: Optional[float] = None # Przechowuje a_surf dla bieżącego substratu
        self.current_substrate_type: Optional[str] = None   # Przechowuje typ 'hexagonal'/'square'
        self.current_substrate_name: str = PREDEFINED_SUBSTRATE_NONE

        # Atrybuty dla oryginalnych kliknięć użytkownika dla substratu
        self.user_selected_substrate_spots: List[Tuple[float, float]] = [] 

        # Atrybuty dla definicji sieci substratu i wyników transformacji
        self.substrate_lattice_type: Optional[str] = None
        self.substrate_a_surf: Optional[float] = None
        self.substrate_definition_name: str = PREDEFINED_SUBSTRATE_NONE # Importuj stałe jeśli trzeba

        self.substrate_F_m2i: Optional[np.ndarray] = None # F: Measured -> Ideal
        self.substrate_t_m2i: Optional[np.ndarray] = None # t: Measured -> Ideal
        self.substrate_transform_analysis_m2i: Optional[Dict[str, Any]] = None
        
        # Piki do wyświetlenia w MainWindow: idealne piki przetransformowane tak, by pasowały do zmierzonych
        self.displayable_fitted_substrate_spots_on_fft: List[Tuple[float, float]] = []

        self.show_fitted_substrate_spots: bool = True

        logger.info("AppController initialized.")

    def set_show_fitted_substrate_spots(self, visible: bool):
        if self.show_fitted_substrate_spots != visible:
            self.show_fitted_substrate_spots = visible
            logger.debug(f"AppController: Show fitted substrate spots set to {visible}")
            # Emituj sygnał, który spowoduje odświeżenie widoku w MainWindow
            # Może to być istniejący substrate_transform_results_updated lub nowy, np. view_parameters_changed
            self.substrate_transform_results_updated.emit() # Ten sygnał już powoduje display_image_data()

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

                self.clear_all_spot_data()
                self.current_substrate_a_surf = None # Reset przy ładowaniu nowego pliku
                self.current_substrate_type = None
                self.current_substrate_name = PREDEFINED_SUBSTRATE_NONE
                self.substrate_definition_changed.emit() # Poinformuj UI o resecie

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

                self.clear_all_spot_data() # Resetuje też user_selected_substrate_spots
                self.substrate_lattice_type = None
                self.substrate_a_surf = None
                self.substrate_definition_name = PREDEFINED_SUBSTRATE_NONE
                self.substrate_F_m2i = None
                self.substrate_t_m2i = None
                self.substrate_transform_analysis_m2i = None
                self.displayable_fitted_substrate_spots_on_fft.clear()
                self.substrate_definition_changed.emit() # Aby zresetować UI
                self.substrate_transform_results_updated.emit() # Aby zresetować UI transformacji
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

    def update_substrate_analysis_results(self, results: Dict[str, Any]):
        """
        Aktualizuje stan substratu na podstawie wyników z SubstrateSpotSelectionDialog.
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

        # Aktualizacja globalnych definicji, jeśli trzeba (np. dla last_selected_substrate)
        if new_def_name == PREDEFINED_SUBSTRATE_CUSTOM:
            self.custom_lattice_info = {"type": new_lattice_type, "a_surf": new_a_surf, "name": "Custom (Dialog)"}
            self.last_selected_substrate = PREDEFINED_SUBSTRATE_CUSTOM
        elif new_def_name != PREDEFINED_SUBSTRATE_NONE:
            self.custom_lattice_info = None
            self.last_selected_substrate = new_def_name
        else:
            self.custom_lattice_info = None
            self.last_selected_substrate = PREDEFINED_SUBSTRATE_NONE

        logger.info(f"AppController: Substrate analysis results updated. Spots: {len(self.user_selected_substrate_spots)}. "
                    f"Transform F: {'Set' if self.substrate_F_m2i is not None else 'None'}. "
                    f"Displayable fitted spots: {len(self.displayable_fitted_substrate_spots_on_fft)}")

        if spots_changed:
            self.spot_lists_updated.emit() # Informuje o zmianie oryginalnych kliknięć (może niepotrzebne jeśli nie są już rysowane)
        if def_changed:
            self.substrate_definition_changed.emit()
        if transform_changed or spots_changed or def_changed: # Jeśli cokolwiek się zmieniło, co wpływa na transformację lub jej wyświetlanie
            self.substrate_transform_results_updated.emit()

    # def set_substrate_definition_and_spots(self,
    #                                        spots: List[Tuple[float, float]],
    #                                        lattice_type: Optional[str],
    #                                        a_surf: Optional[float],
    #                                        substrate_definition_name: str): # Nazwa z ComboBoxa dialogu
    #     """
    #     Ustawia piki substratu oraz definicję sieci (typ, a_surf, nazwa)
    #     używaną do ich wyboru i potencjalnej analizy.
    #     """
    #     logger.info(f"AppController: Updating substrate spots ({len(spots)}) and definition: "
    #                 f"Type={lattice_type}, a_surf={a_surf}, DefName='{substrate_definition_name}'")

    #     # Aktualizacja pików
    #     if self.substrate_spots != spots:
    #         self.substrate_spots = list(spots) # Zawsze przechowuj kopię
    #         self.spot_lists_updated.emit()

    #     # Aktualizacja definicji sieci substratu
    #     definition_changed = False
    #     if self.current_substrate_type != lattice_type:
    #         self.current_substrate_type = lattice_type
    #         definition_changed = True
        
    #     if self.current_substrate_a_surf != a_surf:
    #         self.current_substrate_a_surf = a_surf
    #         definition_changed = True
            
    #     if self.current_substrate_name != substrate_definition_name:
    #         self.current_substrate_name = substrate_definition_name
    #         # Jeśli nazwa to "<Custom a_surf...>", to a_surf jest kluczowe.
    #         # Jeśli to predefiniowana nazwa, a_surf jest z KNOWN_LATTICES.
    #         # Jeśli to "None (Define a_surf below)", to a_surf może być None lub zdefiniowane.
    #         if substrate_definition_name == PREDEFINED_SUBSTRATE_CUSTOM:
    #             self.custom_lattice_info = { # Zaktualizuj lub utwórz custom_lattice_info
    #                 "name": substrate_definition_name, # Lub bardziej unikalna nazwa
    #                 "type": lattice_type,
    #                 "a_surf": a_surf,
    #                 "source": "User Defined in Dialog"
    #             }
    #             self.last_selected_substrate = substrate_definition_name # Ustaw jako "aktywny"
    #         elif substrate_definition_name != PREDEFINED_SUBSTRATE_NONE:
    #             self.custom_lattice_info = None # Wyczyść custom, jeśli wybrano predefiniowaną
    #             self.last_selected_substrate = substrate_definition_name
    #         else: # PREDEFINED_SUBSTRATE_NONE
    #             self.custom_lattice_info = None
    #             self.last_selected_substrate = PREDEFINED_SUBSTRATE_NONE

    #         definition_changed = True

    #     if definition_changed:
    #         self.substrate_definition_changed.emit()
    #         # spot_selection_parameters_changed może też być odpowiedni, jeśli typ sieci wpływa na parametry
    #         self.spot_selection_parameters_changed.emit()


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

    def set_spot_selection_mode(self, mode: str):
        """Ustawia tryb wyboru pików (Substrate/Adsorbate)."""
        if mode in [SPOT_SELECTION_SUBSTRATE, SPOT_SELECTION_ADSORBATE]:
            if self.spot_selection_mode != mode:
                self.spot_selection_mode = mode
                logger.info(f"Spot selection mode set to: {self.spot_selection_mode}")
                self.spot_selection_parameters_changed.emit()
        else:
            logger.warning(f"Attempted to set invalid spot selection mode: {mode}")

    def set_spot_refinement_method(self, method: str):
        """Ustawia metodę uściślania pików."""
        if method in [REFINEMENT_DIRECT_CLICK, REFINEMENT_MAX_PIXEL, REFINEMENT_GAUSSIAN_FIT]:
            if self.spot_refinement_method != method:
                self.spot_refinement_method = method
                logger.info(f"Spot refinement method set to: {self.spot_refinement_method}")
                self.spot_selection_parameters_changed.emit()
        else:
            logger.warning(f"Attempted to set invalid spot refinement method: {method}")

    def set_refinement_roi_size(self, size: int):
        """Ustawia rozmiar ROI do uściślania pików."""
        if isinstance(size, int) and 3 <= size <= 21 and size % 2 != 0: # Przykładowa walidacja
            if self.refinement_roi_size != size:
                self.refinement_roi_size = size
                logger.info(f"Refinement ROI size set to: {self.refinement_roi_size}")
                self.spot_selection_parameters_changed.emit()
        else:
            logger.warning(f"Attempted to set invalid refinement ROI size: {size}")

    # def add_spot(self, point_kx_ky: Tuple[float, float]):
    #     """Dodaje pik do odpowiedniej listy (substrat lub bieżący zestaw adsorbatu)."""
    #     logger.debug(f"Attempting to add spot {point_kx_ky} in mode {self.spot_selection_mode}")
    #     added = False
    #     if self.spot_selection_mode == SPOT_SELECTION_SUBSTRATE:
    #         if len(self.substrate_spots) < MAX_SUBSTRATE_SPOTS:
    #             if point_kx_ky not in self.substrate_spots:
    #                 self.substrate_spots.append(point_kx_ky)
    #                 logger.info(f"Added substrate spot: {point_kx_ky}. Count: {len(self.substrate_spots)}")
    #                 added = True
    #             else: logger.debug(f"Point {point_kx_ky} already in substrate spots.") # pragma: no cover
    #         else: logger.warning(f"Max substrate spots ({MAX_SUBSTRATE_SPOTS}) reached.") # pragma: no cover
        
    #     elif self.spot_selection_mode == SPOT_SELECTION_ADSORBATE:
    #         if 0 <= self.current_adsorbate_set_index < len(self.adsorbate_spot_sets):
    #             current_set = self.adsorbate_spot_sets[self.current_adsorbate_set_index]
    #             if point_kx_ky not in current_set:
    #                 current_set.append(point_kx_ky)
    #                 logger.info(f"Added adsorbate spot: {point_kx_ky} to set {self.current_adsorbate_set_index}. Set count: {len(current_set)}")
    #                 added = True
    #             else: logger.debug(f"Point {point_kx_ky} already in current adsorbate set.") # pragma: no cover
    #         else: logger.error(f"Invalid current adsorbate set index: {self.current_adsorbate_set_index}") # pragma: no cover
        
    #     if added:
    #         self.spot_lists_updated.emit()

    # def clear_substrate_spots(self):
    #     """Czyści listę pików substratu."""
    #     if self.substrate_spots:
    #         self.substrate_spots.clear()
    #         logger.info("Substrate spots cleared.")
    #         self.spot_lists_updated.emit()

    def clear_last_adsorbate_spot(self):
        """Usuwa ostatni dodany pik z bieżącego zestawu adsorbatu."""
        if self.spot_selection_mode == SPOT_SELECTION_ADSORBATE and \
           0 <= self.current_adsorbate_set_index < len(self.adsorbate_spot_sets):
            current_set = self.adsorbate_spot_sets[self.current_adsorbate_set_index]
            if current_set:
                removed_point = current_set.pop()
                logger.info(f"Removed last adsorbate spot {removed_point} from set {self.current_adsorbate_set_index}.")
                self.spot_lists_updated.emit()
            else: logger.debug("No adsorbate spots in current set to clear.") # pragma: no cover
        else: logger.debug("Not in adsorbate mode or invalid set index for clear_last_adsorbate_spot.") # pragma: no cover

    def reselect_current_adsorbate_set(self):
        """Czyści wszystkie punkty z bieżącego zestawu adsorbatu."""
        if self.spot_selection_mode == SPOT_SELECTION_ADSORBATE and \
            0 <= self.current_adsorbate_set_index < len(self.adsorbate_spot_sets):
            if self.adsorbate_spot_sets[self.current_adsorbate_set_index]:
                self.adsorbate_spot_sets[self.current_adsorbate_set_index].clear()
                logger.info(f"Cleared all spots from adsorbate set {self.current_adsorbate_set_index}.")
                self.spot_lists_updated.emit()
        else: logger.debug("Not in adsorbate mode or invalid set index for reselect_current_adsorbate_set.") # pragma: no cover


    def clear_all_adsorbate_sets(self):
        """Czyści wszystkie zestawy pików adsorbatu i resetuje do jednego pustego zestawu."""
        if self.adsorbate_spot_sets != [[]] or self.current_adsorbate_set_index != 0 : # Jeśli faktycznie jest co czyścić
            self.adsorbate_spot_sets = [[]]
            self.current_adsorbate_set_index = 0
            logger.info("All adsorbate spot sets cleared. Reset to one empty set.")
            self.spot_lists_updated.emit() # Ogólna aktualizacja list
            self.adsorbate_sets_structure_changed.emit() # Sygnał o zmianie struktury zestawów (np. dla ComboBoxa)
        else:
            logger.debug("No adsorbate sets to clear or already in default state.")


    def add_new_adsorbate_set(self):
        """Dodaje nowy, pusty zestaw pików adsorbatu i ustawia go jako bieżący."""
        self.adsorbate_spot_sets.append([])
        self.current_adsorbate_set_index = len(self.adsorbate_spot_sets) - 1
        logger.info(f"Added new adsorbate set. Index: {self.current_adsorbate_set_index}")
        self.spot_lists_updated.emit() # Aktualizacja ogólna (np. dla _update_action_states)
        self.adsorbate_sets_structure_changed.emit() # Sygnał dla GUI o zmianie liczby zestawów

    def set_current_adsorbate_set_by_index(self, index: int):
        """Ustawia bieżący zestaw adsorbatu na podstawie indeksu."""
        if 0 <= index < len(self.adsorbate_spot_sets):
            if self.current_adsorbate_set_index != index:
                self.current_adsorbate_set_index = index
                logger.info(f"Current adsorbate set changed to index: {index}")
                self.spot_selection_parameters_changed.emit() # Zmiana parametrów wyboru
        else:
            logger.warning(f"Attempted to set invalid adsorbate set index: {index}") # pragma: no cover

    def clear_all_spot_data(self):
        changed = False
        if self.substrate_spots:
            self.substrate_spots.clear()
            changed = True
        if self.adsorbate_spot_sets != [[]] or self.current_adsorbate_set_index != 0:
            self.adsorbate_spot_sets = [[]]
            self.current_adsorbate_set_index = 0
            changed = True
        
        self.user_selected_substrate_spots.clear()
        # Resetuj też wyniki transformacji, jeśli są powiązane
        self.substrate_F_m2i = None
        self.substrate_t_m2i = None
        self.substrate_transform_analysis_m2i = None
        self.displayable_fitted_substrate_spots_on_fft.clear()

        if hasattr(self, 'spot_lists_updated'): self.spot_lists_updated.emit()
        if hasattr(self, 'adsorbate_sets_structure_changed'): self.adsorbate_sets_structure_changed.emit()
        if hasattr(self, 'substrate_transform_results_updated'): self.substrate_transform_results_updated.emit()
        logger.debug("All spot data and substrate transform results cleared.")
        
        if changed:
            logger.debug("All spot data cleared by clear_all_spot_data.")
            self.spot_lists_updated.emit() # Ogólna aktualizacja, jeśli coś się zmieniło
            self.adsorbate_sets_structure_changed.emit() # Aby zresetować combo box
        else:
            logger.debug("No spot data to clear or already in default state.")
            
    # def clear_all_spot_data(self): # Metoda pomocnicza wywoływana np. przy ładowaniu nowego pliku
    #     """Resetuje wszystkie dane dotyczące pików."""
    #     self.substrate_spots.clear()
    #     self.adsorbate_spot_sets = [[]]
    #     self.current_adsorbate_set_index = 0
    #     # Nie emitujemy tutaj sygnałów indywidualnie, bo load_file i tak spowoduje odświeżenie UI
    #     logger.debug("All spot data cleared.")

