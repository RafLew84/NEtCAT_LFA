# lfa/logic/app_controller.py
"""
Central controller for the LFA application.
Manages application state and coordinates operations between UI and backend modules.
"""
import logging
from typing import Optional, List, Tuple, Dict, Any

from PyQt6.QtCore import QObject, pyqtSignal

logger = logging.getLogger(__name__)

class AppController(QObject):
    """
    Manages the core application state and logic, acting as a bridge
    between the GUI (MainWindow) and the backend processing/analysis modules.
    """

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