# lfa/gui/ui_setup/menu_action_manager.py
import logging
from PyQt6.QtWidgets import QMainWindow, QMenuBar # QMainWindow jest potrzebne do type hinting i dostępu do slotów
from PyQt6.QtGui import QAction

# Ważne: Jeśli AppController ma być używany do bezpośredniego łączenia akcji,
# trzeba go tu zaimportować. Na razie akcje łączą się ze slotami MainWindow.
# from ...logic.app_controller import AppController # Przykład importu AppController

logger = logging.getLogger(__name__)

class MenuActionManager:
    def __init__(self, main_window: QMainWindow): # Na razie bez app_controller, chyba że sloty będą w nim
        """
        Manages the creation of menus and actions for the main window.

        Args:
            main_window (QMainWindow): The main window instance, used for context
                                       (e.g., parent for QAction) and connecting slots.
        """
        self.main_window = main_window
        # Jeśli sloty są w AppController, przekazalibyśmy go i użyli:
        # self.app_controller = main_window.app_controller
        
        self.menu_bar: QMenuBar = self.main_window.menuBar()

        self.file_actions: dict[str, QAction] = {}
        self.preprocessing_actions: dict[str, QAction] = {}
        self.analysis_actions: dict[str, QAction] = {}
        self.view_actions: dict[str, QAction] = {} # Dla akcji z menu View (np. przełączanie doków)
        self.help_actions: dict[str, QAction] = {}
        
        self._create_all_menus()
        logger.debug("MenuActionManager initialized and menus created.")

    def _create_action(self, text: str, status_tip: str, triggered_slot, 
                       shortcut: str = "", enabled: bool = True, 
                       is_checkable: bool = False, checked: bool = False) -> QAction:
        """Helper method to create a QAction."""
        action = QAction(text, self.main_window) # self.main_window jako rodzic
        action.setStatusTip(status_tip)
        if triggered_slot: # Slot może być None, jeśli akcja ma inny cel (np. tylko submenu)
            action.triggered.connect(triggered_slot)
        if shortcut:
            action.setShortcut(shortcut)
        action.setEnabled(enabled)
        if is_checkable:
            action.setCheckable(True)
            action.setChecked(checked)
        return action

    def _create_file_menu(self):
        file_menu = self.menu_bar.addMenu("&File")
        
        self.file_actions["open"] = self._create_action(
            text="&Open...", 
            status_tip="Open an STM data file",
            triggered_slot=self.main_window.open_file_dialog, # Slot w MainWindow
            shortcut="Ctrl+O"
        )
        file_menu.addAction(self.file_actions["open"])
        
        file_menu.addSeparator()
        
        self.file_actions["exit"] = self._create_action(
            text="&Exit", 
            status_tip="Exit the application",
            triggered_slot=self.main_window.close, # Slot wbudowany w QMainWindow
            shortcut="Ctrl+Q"
        )
        file_menu.addAction(self.file_actions["exit"])

    def _create_preprocessing_menu(self):
        preprocessing_menu = self.menu_bar.addMenu("&Preprocessing")

        # Definicje akcji (nazwa klucza, tekst, tooltip, slot, initial_enabled)
        actions_definitions = [
            ("gaussian_blur", "&Gaussian Blur...", "Apply Gaussian blur filter", self.main_window.open_gaussian_blur_dialog, False),
            ("gaussian_sharpen", "Gaussian &Sharpening...", "Apply Gaussian Sharpening (Unsharp Mask)", self.main_window.open_gaussian_sharpening_dialog, False),
            ("plane_level", "&Plane Leveling...", "Level image by subtracting a fitted plane", self.main_window.open_plane_leveling_dialog, False),
            ("median_filter", "&Median Filter...", "Apply median filter for noise reduction", self.main_window.open_median_filter_dialog, False),
            ("nlmeans", "&NL-Means Denoising...", "Apply Non-Local Means denoising (skimage)", self.main_window.open_nlmeans_dialog, False),
            ("bm3d", "&BM3D Denoising...", "Apply BM3D denoising (Computationally intensive)", self.main_window.open_bm3d_dialog, False),
        ]

        for name, text, tip, slot, enabled in actions_definitions:
            action = self._create_action(text, tip, slot, enabled=enabled)
            self.preprocessing_actions[name] = action
            preprocessing_menu.addAction(action)
            # Ustawienie atrybutu w MainWindow, aby _update_action_states działało
            setattr(self.main_window, f"{name}_action", action)


    def _create_analysis_menu(self):
        analysis_menu = self.menu_bar.addMenu("&Analysis")
        
        self.analysis_actions["fft"] = self._create_action(
            text="Calculate &FFT...", 
            status_tip="Calculate Fast Fourier Transform",
            triggered_slot=self.main_window.open_fft_dialog,
            enabled=False # Początkowo wyłączone
        )
        analysis_menu.addAction(self.analysis_actions["fft"])
        # Ustawienie atrybutu w MainWindow
        self.main_window.fft_action = self.analysis_actions["fft"]

        analysis_menu.addSeparator() # Dodajemy separator

        sel_subs_action = self._create_action(
            text="Select &Substrate Spots...",
            status_tip="Open dialog to select/edit substrate spots",
            triggered_slot=self.main_window.open_substrate_spot_selection_dialog, # Nowy slot w MainWindow
            enabled=False # Początkowo wyłączone, włączane gdy jest obraz FFT
        )
        analysis_menu.addAction(sel_subs_action)
        self.main_window.select_substrate_spots_action = sel_subs_action # Ustaw atrybut

        sel_ads_action = self._create_action(
            text="Select &Adsorbate Spots...",
            status_tip="Open dialog to select/edit adsorbate spots for the current set",
            triggered_slot=self.main_window.open_adsorbate_spot_selection_dialog, # Nowy slot w MainWindow
            enabled=False # Początkowo wyłączone
        )
        analysis_menu.addAction(sel_ads_action)
        self.main_window.select_adsorbate_spots_action = sel_ads_action # Ustaw atrybut

        analysis_menu.addSeparator() # Separator przed nową grupą opcji
        calc_dist_action = self._create_action(
            text="Calculate Spot &Distances (Real Space)...",
            status_tip="Open dialog to select spots and calculate their real space distances from center",
            triggered_slot=self.main_window.open_spot_distance_dialog, # Nowy slot w MainWindow
            enabled=False # Początkowo wyłączone
        )
        analysis_menu.addAction(calc_dist_action)
        setattr(self.main_window, "calculate_spot_distances_action", calc_dist_action) # Ustaw atrybut

        analysis_menu.addSeparator()
        vis_action = self._create_action(
            text="Visualize Real Space...",
            status_tip="Open dialog for real space and FFT visualization",
            triggered_slot=self.main_window.open_real_space_fft_visualizer, # Nowy slot
            enabled=False # Początkowo wyłączone
        )
        analysis_menu.addAction(vis_action)
        self.main_window.visualize_real_space_action = vis_action # Ustaw atrybut


    def _create_view_menu(self):
        self.main_window.view_menu = self.menu_bar.addMenu("&View")

    def _create_help_menu(self):
        help_menu = self.menu_bar.addMenu("&Help")
        
        self.help_actions["about"] = self._create_action(
            text="&About LFA...", 
            status_tip="Show information about LFA",
            triggered_slot=self.main_window.show_about_dialog
        )
        help_menu.addAction(self.help_actions["about"])

    def _create_all_menus(self):
        """Tworzy wszystkie główne menu."""
        self._create_file_menu()
        self._create_preprocessing_menu()
        self._create_analysis_menu()
        self._create_view_menu() # Kluczowe, aby self.main_window.view_menu było dostępne dla doków
        self._create_help_menu()

    def get_action(self, menu_key: str, action_key: str) -> QAction | None:
        """
        Pobiera QAction na podstawie klucza menu i klucza akcji.
        Przydatne, jeśli MainWindow._update_action_states potrzebuje dostępu
        do akcji bez bezpośredniego przechowywania ich jako atrybutów w MainWindow.
        """
        menu_map = {
            "file": self.file_actions,
            "preprocessing": self.preprocessing_actions,
            "analysis": self.analysis_actions,
            "view": self.view_actions,
            "help": self.help_actions,
        }
        return menu_map.get(menu_key, {}).get(action_key)