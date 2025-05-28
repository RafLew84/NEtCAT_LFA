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

try:
    from ..analysis.lattice import (
        get_real_space_lattice_parameters, # Główna nowa funkcja
        LATTICE_TYPE_HEXAGONAL, LATTICE_TYPE_SQUARE, # Stałe, jeśli potrzebne do logiki
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

    # Sygnał emitowany po zmianie list pików (substratu lub któregokolwiek zestawu adsorbatu)
    spot_lists_updated = pyqtSignal()
    adsorbate_set_updated = pyqtSignal(int)
    # Sygnał emitowany po zmianie trybu wyboru pików (Substrate/Adsorbate) lub metody uściślania
    spot_selection_parameters_changed = pyqtSignal()
    # Sygnał emitowany po zmianie bieżącego zestawu adsorbatu (np. dodanie nowego, zmiana indeksu)
    adsorbate_sets_structure_changed = pyqtSignal()
    substrate_transform_results_updated = pyqtSignal()

    substrate_definition_changed = pyqtSignal()

    substrate_real_space_params_updated = pyqtSignal(dict)
    adsorbate_real_space_params_updated = pyqtSignal(int, dict)

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
        self.corrected_adsorbate_spot_sets: List[List[Tuple[float, float]]] = [[]]
        self.current_adsorbate_set_index: int = 0

        # Tryby i parametry wyboru/uściślania pików
        self.spot_selection_mode: str = "Substrate"  # Domyślnie "Substrate" lub "Adsorbate"
        self.spot_refinement_method: str = "Direct Click" # Domyślnie, inne opcje: "Max Pixel", "2D Gaussian Fit"
        self.refinement_roi_size: int = 5 # Domyślny rozmiar (np. średnica) obszaru ROI do uściślania
        self.reference_ideal_substrate_spots_px: List[Tuple[float, float]] = []

        # Ustawienia dotyczące idealnej sieci i substratu
        self.custom_lattice_info: Optional[Dict[str, Any]] = None # Dla definicji własnej sieci
        self.last_selected_substrate: str = "None" # Ostatnio wybrany substrat z ComboBoxa (lub nazwa custom)
        
        # Dodatkowe stany, które mogą być zarządzane centralnie
        self.show_ideal_lattice: bool = True # Czy pokazywać idealną sieć na FFT
        # self.show_substrate_spots_markers: bool = True # Widoczność markerów substratu
        # self.show_adsorbate_spots_markers: bool = True # Widoczność markerów adsorbatu

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

        self.current_fft_data_shape: Optional[Tuple[int, int]] = None
        self.substrate_real_space_results: Optional[Dict[str, Any]] = None
        self.adsorbate_real_space_results: Dict[int, Dict[str, Any]] = {} 

        logger.info("AppController initialized.")

    # def set_show_fitted_substrate_spots(self, visible: bool):
    #     if self.show_fitted_substrate_spots != visible:
    #         self.show_fitted_substrate_spots = visible
    #         logger.debug(f"AppController: Show fitted substrate spots set to {visible}")
    #         # Emituj sygnał, który spowoduje odświeżenie widoku w MainWindow
    #         # Może to być istniejący substrate_transform_results_updated lub nowy, np. view_parameters_changed
    #         self.substrate_transform_results_updated.emit() # Ten sygnał już powoduje display_image_data()

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
                self.reference_ideal_substrate_spots_px.clear()
                self.current_substrate_a_surf = None # Reset przy ładowaniu nowego pliku
                self.current_substrate_type = None
                self.current_substrate_name = PREDEFINED_SUBSTRATE_NONE
                self.substrate_definition_changed.emit() # Poinformuj UI o resecie

                self.history_manager.clear_history() # Wyczyść poprzednią historię
                self.corrected_adsorbate_spot_sets = [[]]
                if hasattr(self, 'adsorbate_set_updated'): self.adsorbate_set_updated.emit(0)
                if hasattr(self, 'adsorbate_real_space_params_updated'): self.adsorbate_real_space_params_updated.emit(0, {})

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
                self.substrate_real_space_results = None
                self.current_fft_data_shape = None
                self.displayable_fitted_substrate_spots_on_fft.clear()
                self.adsorbate_real_space_results.clear()
                self.substrate_definition_changed.emit() # Aby zresetować UI
                self.substrate_transform_results_updated.emit() # Aby zresetować UI transformacji
                self.substrate_real_space_params_updated.emit({})
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

        new_ideal_ref_spots = results.get("ideal_substrate_spots_px_for_reference", [])
        self.reference_ideal_substrate_spots_px = list(new_ideal_ref_spots)
        logger.info(f"AppController: Updated reference ideal substrate spots count: {len(self.reference_ideal_substrate_spots_px)}")

        logger.info(f"AppController: Substrate analysis results updated. Spots: {len(self.user_selected_substrate_spots)}. "
                    f"Transform F: {'Set' if self.substrate_F_m2i is not None else 'None'}. "
                    f"Displayable fitted spots: {len(self.displayable_fitted_substrate_spots_on_fft)}")
        
        self.substrate_real_space_results = None
        self.substrate_real_space_params_updated.emit({})

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

        # 1. Walidacja danych wejściowych
        # Potrzebujemy typu sieci i a_surf, które były użyte do wygenerowania idealnej siatki,
        # która następnie została dopasowana.
        if not (self.substrate_lattice_type and self.substrate_a_surf and self.substrate_a_surf > 0):
            logger.warning("Substrate definition (type/a_surf) used for fitting not set. Cannot calculate real space params.")
            self.substrate_real_space_params_updated.emit({"error": "Substrate definition for fit missing."})
            return
        
        # Używamy displayable_fitted_substrate_spots_on_fft, które są idealnymi punktami
        # przetransformowanymi tak, aby pasowały do kliknięć użytkownika.
        # Te punkty reprezentują wierzchołki "zdeformowanej" idealnej siatki na obrazie FFT.
        if not self.displayable_fitted_substrate_spots_on_fft:
            logger.warning("Fitted substrate spots (in FFT px) are not available.")
            self.substrate_real_space_params_updated.emit({"error": "Fitted substrate spots missing."})
            return

        if not self.current_fft_data_shape:
            logger.warning("Current FFT data shape not available. Cannot calculate real space params.")
            self.substrate_real_space_params_updated.emit({"error": "FFT shape missing."})
            return

        # 2. Pobierz Lx_nm, Ly_nm
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

        # 3. Przygotuj wektory g* względem centrum FFT
        # self.displayable_fitted_substrate_spots_on_fft to absolutne pozycje pikselowe na FFT.
        fft_rows_ky, fft_cols_kx = self.current_fft_data_shape
        center_kx_px = fft_cols_kx / 2.0
        center_ky_px = fft_rows_ky / 2.0

        # Te "fitted_g_vectors" reprezentują wektory sieci odwrotnej ZDEFORMOWANEJ siatki substratu
        fitted_g_vectors_relative_px = [
            (kx_abs - center_kx_px, ky_abs - center_ky_px)
            for kx_abs, ky_abs in self.displayable_fitted_substrate_spots_on_fft
        ]
        
        # Sprawdź, czy mamy odpowiednią liczbę pików dla danego typu sieci
        # Liczba displayable_fitted_substrate_spots_on_fft powinna odpowiadać liczbie
        # idealnych punktów, które zostały dopasowane (czyli 6 dla hex, 4 dla square,
        # jeśli `ideal_points_that_were_matched` w dialogu miało tyle elementów).
        expected_spot_count = 0
        if self.substrate_lattice_type == LATTICE_TYPE_HEXAGONAL: expected_spot_count = 6
        elif self.substrate_lattice_type == LATTICE_TYPE_SQUARE: expected_spot_count = 4
        
        if len(fitted_g_vectors_relative_px) != expected_spot_count and expected_spot_count > 0:
            logger.warning(f"Incorrect number of fitted spots ({len(fitted_g_vectors_relative_px)}) "
                           f"for {self.substrate_lattice_type} (expected {expected_spot_count}). Cannot calculate real space params.")
            self.substrate_real_space_params_updated.emit({"error": f"Need {expected_spot_count} fitted spots."})
            return
        if expected_spot_count == 0 and len(fitted_g_vectors_relative_px) < 2 : # Jeśli typ nieznany, potrzebujemy co najmniej 2 wektorów
             logger.warning(f"Not enough fitted spots to define basis for unknown lattice type.")
             self.substrate_real_space_params_updated.emit({"error": "Need >=2 fitted spots."})
             return


        # 4. Wywołaj funkcję z lattice.py
        # Przekazujemy `fitted_g_vectors_relative_px` jako podstawę do obliczeń
        # oraz `self.substrate_lattice_type` (który był użyty do wygenerowania idealnej siatki, do której dopasowywaliśmy)
        print(f"displayable_fitted_substrate_spots_on_fft: {self.displayable_fitted_substrate_spots_on_fft}")
        print(f"fitted_g_vectors_relative_px: {fitted_g_vectors_relative_px}")
        results = get_real_space_lattice_parameters(
            selected_g_vectors_relative_px=fitted_g_vectors_relative_px,
            lattice_type=self.substrate_lattice_type, # Typ sieci, do której dopasowano
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



    # def calculate_and_store_substrate_real_params(self):
    #     """
    #     Calculates real-space lattice parameters for the currently defined substrate
    #     and stores them. Emits a signal upon completion.
    #     """
    #     if not LATTICE_ANALYSIS_FUNCTIONS_AVAILABLE: # pragma: no cover
    #         logger.error("Lattice analysis functions (get_real_space_lattice_parameters) not available.")
    #         self.substrate_real_space_params_updated.emit({"error": "Lattice functions missing."})
    #         return

    #     logger.info("AppController: Attempting to calculate substrate real space parameters.")

    #     # 1. Walidacja danych wejściowych
    #     if not (self.substrate_lattice_type and self.substrate_a_surf and self.substrate_a_surf > 0):
    #         logger.warning("Substrate definition (type/a_surf) not set. Cannot calculate real space params.")
    #         self.substrate_real_space_params_updated.emit({"error": "Substrate definition incomplete."})
    #         return
        
    #     if not self.reference_ideal_substrate_spots_px:
    #         logger.warning("Reference ideal substrate spots (in FFT px) are not available.")
    #         # Spróbuj je wygenerować, jeśli ich nie ma, ale definicja substratu jest
    #         # To jest bardziej skomplikowane, bo get_substrate_ideal_spots_for_real_space_calc
    #         # potrzebuje Lx, Ly i kształtu FFT - załóżmy, że reference_ideal_substrate_spots_px
    #         # są ustawiane po dialogu substratu.
    #         # Lepsze byłoby, gdyby metoda `update_substrate_analysis_results` zawsze je ustawiała.
    #         self.substrate_real_space_params_updated.emit({"error": "Reference ideal spots missing."})
    #         return

    #     if not self.current_fft_data_shape: # pragma: no cover
    #         logger.warning("Current FFT data shape not available. Cannot calculate real space params.")
    #         self.substrate_real_space_params_updated.emit({"error": "FFT shape missing."})
    #         return

    #     # 2. Pobierz Lx_nm, Ly_nm
    #     # Zakładamy, że jest aktywny jakiś węzeł FFT, aby znaleźć jego korzeń "Original"
    #     current_node = self.history_manager.get_current_node()
    #     if not current_node: # pragma: no cover
    #          logger.warning("No current history node available for Lx/Ly."); self.substrate_real_space_params_updated.emit({"error": "No active node."}); return
        
    #     root_node = self.history_manager.get_root_node_for_node(current_node.node_id)
    #     if not (root_node and root_node.operation_name == "Original" and root_node.parameters): # pragma: no cover
    #         logger.warning("Could not get Original node parameters for Lx/Ly."); self.substrate_real_space_params_updated.emit({"error": "Original node params missing."}); return
        
    #     Lx_nm = root_node.parameters.get("size_nm_x")
    #     Ly_nm = root_node.parameters.get("size_nm_y")

    #     if not (Lx_nm and Ly_nm and Lx_nm > 0 and Ly_nm > 0): # pragma: no cover
    #         logger.warning("Invalid Lx/Ly from Original node."); self.substrate_real_space_params_updated.emit({"error": "Invalid Lx/Ly."}); return

    #     # 3. Przygotuj wektory g* względem centrum FFT
    #     # self.reference_ideal_substrate_spots_px to absolutne pozycje pikselowe na FFT
    #     fft_rows_ky, fft_cols_kx = self.current_fft_data_shape
    #     center_kx_px = fft_cols_kx / 2.0
    #     center_ky_px = fft_rows_ky / 2.0

    #     selected_g_vectors_relative_px = [
    #         (kx_abs - center_kx_px, ky_abs - center_ky_px)
    #         for kx_abs, ky_abs in self.reference_ideal_substrate_spots_px
    #     ]
        
    #     # Sprawdź, czy mamy odpowiednią liczbę pików dla danego typu sieci
    #     expected_spot_count = 0
    #     if self.substrate_lattice_type == LATTICE_TYPE_HEXAGONAL: expected_spot_count = 6
    #     elif self.substrate_lattice_type == LATTICE_TYPE_SQUARE: expected_spot_count = 4
        
    #     if len(selected_g_vectors_relative_px) != expected_spot_count and expected_spot_count > 0:
    #         logger.warning(f"Incorrect number of reference spots ({len(selected_g_vectors_relative_px)}) "
    #                        f"for {self.substrate_lattice_type} (expected {expected_spot_count}). Cannot calculate.")
    #         self.substrate_real_space_params_updated.emit({"error": f"Need {expected_spot_count} ref. spots."})
    #         return


    #     # 4. Wywołaj funkcję z lattice.py
    #     results = get_real_space_lattice_parameters(
    #         selected_g_vectors_relative_px=selected_g_vectors_relative_px,
    #         lattice_type=self.substrate_lattice_type,
    #         Lx_nm=Lx_nm,
    #         Ly_nm=Ly_nm,
    #         fft_shape_cols_kx=fft_cols_kx,
    #         fft_shape_rows_ky=fft_rows_ky
    #     )

    #     if results:
    #         self.substrate_real_space_results = results
    #         logger.info(f"Successfully calculated substrate real space parameters: {results}")
    #         self.substrate_real_space_params_updated.emit(results)
    #     else: # pragma: no cover
    #         self.substrate_real_space_results = None
    #         logger.warning("Failed to calculate substrate real space parameters.")
    #         self.substrate_real_space_params_updated.emit({"error": "Calculation failed in lattice module."})

    
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

        # 1. Walidacja danych wejściowych
        if not (0 <= set_index < len(self.corrected_adsorbate_spot_sets)): # pragma: no cover
            logger.warning(f"Invalid set_index {set_index} for adsorbate real space params.")
            self.adsorbate_real_space_params_updated.emit(set_index, {"error": "Invalid set index."})
            return
        
        corrected_spots_ideal_px = self.corrected_adsorbate_spot_sets[set_index]
        if not corrected_spots_ideal_px or len(corrected_spots_ideal_px) < 3: # Wymaga co najmniej 3 pików do zdefiniowania 2 wektorów
            logger.warning(f"Not enough corrected adsorbate spots (need >=3, got {len(corrected_spots_ideal_px)}) for set {set_index}.")
            self.adsorbate_real_space_params_updated.emit(set_index, {"error": "Need >= 3 corrected spots."})
            return

        if not self.current_fft_data_shape: # pragma: no cover
            logger.warning("Current FFT data shape not available."); self.adsorbate_real_space_params_updated.emit(set_index, {"error": "FFT shape missing."}); return

        root_node = self.history_manager.get_root_node_for_node(self.history_manager.get_current_node().node_id) # type: ignore
        if not (root_node and root_node.parameters): logger.warning("Cannot get Lx, Ly for adsorbate."); self.adsorbate_real_space_params_updated.emit(set_index, {"error": "Original node params missing."}); return # pragma: no cover
        Lx_nm = root_node.parameters.get("size_nm_x"); Ly_nm = root_node.parameters.get("size_nm_y")
        if not (Lx_nm and Ly_nm and Lx_nm > 0 and Ly_nm > 0): logger.warning("Invalid Lx/Ly."); self.adsorbate_real_space_params_updated.emit(set_index, {"error": "Invalid Lx/Ly."}); return # pragma: no cover

        # 2. Przygotuj wektory g* względem centrum "idealnego" FFT
        # corrected_spots_ideal_px są już w "idealnym" systemie pikseli FFT, wycentrowanym.
        # Funkcja select_reciprocal_lattice_basis_vectors oczekuje wektorów od centrum.
        # Musimy wybrać, który z corrected_spots_ideal_px jest centrum (0,0) lub jak wybrać wektory.
        
        # Założenie: użytkownik wybrał piki tak, że pierwszy może być traktowany jako centrum (0,0)
        # lub trzeba zaimplementować bardziej zaawansowany wybór.
        # Na razie proste: jeśli są 3, pierwszy to centrum, dwa pozostałe definiują g1*, g2*.
        
        # Przekształć corrected_spots_ideal_px na wektory względem (domniemanego) centrum,
        # jeśli nie są już wektorami od (0,0) idealnego systemu.
        # `corrected_spots_ideal_px` są POZYCJAMI w idealnym systemie FFT.
        
        fft_rows_ky, fft_cols_kx = self.current_fft_data_shape
        center_kx_ideal_px = fft_cols_kx / 2.0
        center_ky_ideal_px = fft_rows_ky / 2.0

        # Jeśli `corrected_spots_ideal_px` to absolutne pozycje w idealnym systemie,
        # potrzebujemy ich jako wektorów od centrum tego idealnego systemu.
        # Jeśli jeden z nich to (0,0) tego systemu, to jest proste.
        # Załóżmy, że użytkownik wybiera piki, które *już* są wektorami g*
        # (lub jeden z nich jest (0,0) a pozostałe to g1*, g2*).
        # Dla prostoty, załóżmy, że użytkownik wskazuje piki, które są końcami wektorów g*
        # zaczepionych w (0,0) idealnej przestrzeni odwrotnej.
        # Jeśli są 3 piki: (0,0), g1*, g2*. Wtedy g1* i g2* to bezpośrednio wybrane piki.
        # To wymaga od użytkownika, aby wybrał pik (0,0) w dialogu adsorbatu.
        #
        # Bardziej ogólnie: jeśli mamy N>=3 skorygowanych pików, musimy wybrać 2 wektory bazowe.
        # TODO: Implement a robust way to select basis g-vectors for adsorbate from N spots.
        # For now, let's assume a simple case for demonstration if 3 spots are given:
        # (0,0)-like, g1, g2. Or just take first two non-collinear if more are given.
        
        if len(corrected_spots_ideal_px) < 2 : # Potrzebujemy co najmniej 2 wektorów g*
            logger.warning(f"Need at least 2 (non-origin) corrected adsorbate spots to define basis for set {set_index}.")
            self.adsorbate_real_space_params_updated.emit(set_index, {"error": "Need >= 2 non-origin spots."})
            return

        # Załóżmy, że corrected_spots_ideal_px to już są wektory g* (w pikselach idealnego systemu, od (0,0))
        # Jeśli nie, trzeba je przeliczyć. Np. jeśli pierwszy to (0,0):
        # g_vectors_adsorbate_px = []
        # origin_like_spot = corrected_spots_ideal_px[0] # Załóżmy, że pierwszy to (0,0) lub blisko
        # for spot_px in corrected_spots_ideal_px[1:]:
        #    g_vectors_adsorbate_px.append( (spot_px[0] - origin_like_spot[0], spot_px[1] - origin_like_spot[1]) )
        # if len(g_vectors_adsorbate_px) < 2: ... return
        # g1_ads_px, g2_ads_px = g_vectors_adsorbate_px[0], g_vectors_adsorbate_px[1] # Uproszczenie

        # Bardziej robustne: użyj funkcji podobnej do select_reciprocal_lattice_basis_vectors,
        # ale dla nieznanego typu sieci, próbującej znaleźć dwa najkrótsze, liniowo niezależne wektory.
        # Na razie, dla testu, weźmy pierwsze dwa z listy `corrected_spots_ideal_px`
        # zakładając, że są to już wektory g* względem centrum idealnego systemu.
        # To duże uproszczenie!
        if len(corrected_spots_ideal_px) >= 2:
            g1_ads_px = (corrected_spots_ideal_px[0][0] - center_kx_ideal_px, corrected_spots_ideal_px[0][1] - center_ky_ideal_px)
            g2_ads_px = (corrected_spots_ideal_px[1][0] - center_kx_ideal_px, corrected_spots_ideal_px[1][1] - center_ky_ideal_px)
            # Sprawdź, czy nie są współliniowe z centrum lub ze sobą
            if np.linalg.norm(g1_ads_px) < 1e-3 or np.linalg.norm(g2_ads_px) < 1e-3:
                logger.warning(f"Adsorbate g-vectors too short for set {set_index}."); self.adsorbate_real_space_params_updated.emit(set_index, {}); return
            
            # Sprawdzenie współliniowości g1 i g2
            cross_prod_z = g1_ads_px[0]*g2_ads_px[1] - g1_ads_px[1]*g2_ads_px[0]
            if abs(cross_prod_z) < 1e-3: # Zbyt mały iloczyn wektorowy (współliniowe)
                 logger.warning(f"Adsorbate g-vectors are collinear for set {set_index}."); self.adsorbate_real_space_params_updated.emit(set_index, {}); return
        else:
            logger.warning(f"Not enough corrected adsorbate spots to form basis for set {set_index}.")
            self.adsorbate_real_space_params_updated.emit(set_index, {"error": "Need >= 2 distinct spots."})
            return
            
        # 3. Konwersja na nm^-1
        g1_ads_nm_inv = convert_g_vector_px_to_nm_inv(g1_ads_px, Lx_nm, Ly_nm, fft_cols_kx, fft_rows_ky)
        g2_ads_nm_inv = convert_g_vector_px_to_nm_inv(g2_ads_px, Lx_nm, Ly_nm, fft_cols_kx, fft_rows_ky)

        if g1_ads_nm_inv is None or g2_ads_nm_inv is None: # pragma: no cover
            self.adsorbate_real_space_params_updated.emit(set_index, {"error": "g-vector conversion failed."}); return

        # 4. Obliczenie wektorów sieci rzeczywistej a1, a2 (w nm)
        real_space_vecs_ads = calculate_real_space_vectors_from_g(g1_ads_nm_inv, g2_ads_nm_inv)
        if real_space_vecs_ads is None: # pragma: no cover
            self.adsorbate_real_space_params_updated.emit(set_index, {"error": "Real space vector calc failed."}); return
        a1_ads_vec_nm, a2_ads_vec_nm = real_space_vecs_ads

        # 5. Parametry
        a1_ads_mag_nm = np.linalg.norm(a1_ads_vec_nm)
        a2_ads_mag_nm = np.linalg.norm(a2_ads_vec_nm)
        dot_product_ads = np.dot(a1_ads_vec_nm, a2_ads_vec_nm)
        if a1_ads_mag_nm < 1e-9 or a2_ads_mag_nm < 1e-9: # pragma: no cover
             self.adsorbate_real_space_params_updated.emit(set_index, {"error": "Real vectors too short."}); return
        cos_alpha_ads = np.clip(dot_product_ads / (a1_ads_mag_nm * a2_ads_mag_nm), -1.0, 1.0)
        alpha_ads_deg = np.degrees(np.arccos(cos_alpha_ads))

        results = {
            "a1_nm": a1_ads_mag_nm, "a2_nm": a2_ads_mag_nm, "alpha_deg": alpha_ads_deg,
            "a1_vec_nm": a1_ads_vec_nm, "a2_vec_nm": a2_ads_vec_nm,
            "g1_vec_px_ideal_sys": g1_ads_px, "g2_vec_px_ideal_sys": g2_ads_px, # W pikselach idealnego systemu
            "g1_vec_nm_inv": g1_ads_nm_inv, "g2_vec_nm_inv": g2_ads_nm_inv
        }
        logger.info(f"Adsorbate set {set_index} real space params: a1={a1_ads_mag_nm:.3f}nm, a2={a2_ads_mag_nm:.3f}nm, alpha={alpha_ads_deg:.2f}deg")
        self.adsorbate_real_space_results[set_index] = results
        self.adsorbate_real_space_params_updated.emit(set_index, results)


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
        if processed_fft_data is not None:
            self.current_fft_data_shape = processed_fft_data.shape
            logger.info(f"AppController: Stored current FFT data shape: {self.current_fft_data_shape}")
        else: # pragma: no cover
            self.current_fft_data_shape = None
            logger.warning("AppController: FFT data is None, cannot store shape.")
            
        self.add_operation_to_history(parent_node_id, "FFT", params, processed_fft_data, "FFT", source_roi_slice)
        # Po nowym FFT, poprzednie obliczenia parametrów rzeczywistych mogą być nieaktualne
        self.substrate_real_space_results = None
        self.substrate_real_space_params_updated.emit({})
        self.adsorbate_real_space_results.clear()
        if hasattr(self, 'adsorbate_real_space_params_updated'): self.adsorbate_real_space_params_updated.emit(self.current_adsorbate_set_index, {})

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
            self.corrected_adsorbate_spot_sets = [[]]
            self.current_adsorbate_set_index = 0
            logger.info("All adsorbate spot sets cleared. Reset to one empty set.")
            self.adsorbate_sets_structure_changed.emit() # Dla ComboBoxa
            self.adsorbate_set_updated.emit(0)
        else:
            logger.debug("No adsorbate sets to clear or already in default state.")

    def update_adsorbate_set_results(self, 
                                     set_index: int, 
                                     raw_spots: List[Tuple[float, float]], 
                                     corrected_spots_ideal_system: List[Tuple[float, float]]):
        """
        Aktualizuje surowe i skorygowane piki dla danego zestawu adsorbatu.
        """
        if not (0 <= set_index < len(self.adsorbate_spot_sets)):
            logger.error(f"AppController: Invalid set_index {set_index} for updating adsorbate spots.")
            return

        # Upewnij się, że lista skorygowanych ma odpowiedni rozmiar
        while len(self.corrected_adsorbate_spot_sets) <= set_index:
            self.corrected_adsorbate_spot_sets.append([])

        raw_changed = self.adsorbate_spot_sets[set_index] != raw_spots
        corrected_changed = self.corrected_adsorbate_spot_sets[set_index] != corrected_spots_ideal_system

        if raw_changed:
            self.adsorbate_spot_sets[set_index] = list(raw_spots)
            logger.info(f"AppController: Updated raw adsorbate spots for set {set_index}. Count: {len(raw_spots)}")
        
        if corrected_changed:
            self.corrected_adsorbate_spot_sets[set_index] = list(corrected_spots_ideal_system)
            logger.info(f"AppController: Updated corrected adsorbate spots (ideal sys) for set {set_index}. Count: {len(corrected_spots_ideal_system)}")

        if raw_changed or corrected_changed:
            self.adsorbate_set_updated.emit(set_index) # Emituj z indeksem zmienionego zestawu
            # spot_lists_updated może być nadal używany do ogólnych aktualizacji, np. tekstowego podglądu
            if raw_changed and hasattr(self, 'spot_lists_updated'):
                 self.spot_lists_updated.emit()
    
        if set_index in self.adsorbate_real_space_results:
            del self.adsorbate_real_space_results[set_index]
            self.adsorbate_real_space_params_updated.emit(set_index, {})


    def add_new_adsorbate_set(self):
        """Dodaje nowy, pusty zestaw pików adsorbatu i ustawia go jako bieżący."""
        self.adsorbate_spot_sets.append([])
        self.corrected_adsorbate_spot_sets.append([])
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
            self.corrected_adsorbate_spot_sets = [[]]
            self.current_adsorbate_set_index = 0
            changed = True
        
        self.user_selected_substrate_spots.clear()
        self.adsorbate_real_space_results.clear()
        # Resetuj też wyniki transformacji, jeśli są powiązane
        self.substrate_F_m2i = None
        self.substrate_t_m2i = None
        self.substrate_transform_analysis_m2i = None
        self.displayable_fitted_substrate_spots_on_fft.clear()
        self.substrate_real_space_results = None

        if hasattr(self, 'substrate_real_space_params_updated'): self.substrate_real_space_params_updated.emit({})
        if hasattr(self, 'spot_lists_updated'): self.spot_lists_updated.emit()
        if hasattr(self, 'adsorbate_sets_structure_changed'): self.adsorbate_sets_structure_changed.emit()
        if hasattr(self, 'substrate_transform_results_updated'): self.substrate_transform_results_updated.emit()
        if hasattr(self, 'adsorbate_real_space_params_updated'): self.adsorbate_real_space_params_updated.emit(0, {})
        if changed and hasattr(self, 'adsorbate_set_updated'): self.adsorbate_set_updated.emit(0)
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

