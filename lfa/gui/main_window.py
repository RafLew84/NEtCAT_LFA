# lfa/gui/main_window.py
"""
Defines the main window for the Lattice Fourier Analyzer (LFA) application.
"""

import logging
import os
import numpy as np
from typing import Optional, Dict, Any, Tuple, List, Union
import time

from PyQt6.QtWidgets import (
    QMainWindow, QVBoxLayout, QWidget, QFileDialog, QMessageBox, QApplication, 
    QDialog, QHBoxLayout, QSplitter, QListWidget, QListWidgetItem, QDockWidget,
    QComboBox, QToolBar, QToolButton, QLabel, QLineEdit, QPushButton, QTextEdit, QCheckBox,
    QGroupBox, QFormLayout, QRadioButton, QSpinBox
)
from PyQt6.QtGui import QAction, QIcon
from PyQt6.QtCore import Qt, pyqtSlot, QPointF

# Import pyqtgraph after checking for installation
try:
    import pyqtgraph as pg
except ImportError:
    logging.error("PyQtGraph not found. Please install it: pip install pyqtgraph")
    pg = None

# Import LFA core components
from ..core.data_models import STMImage
from ..io.factory import load_stm_file
from ..core.history import HistoryNode
from .widgets.metadata_widget import MetadataWidget
from ..logic.history_manager import HistoryManager
from .panels.fft_analysis_panel import FFTAnalysisPanel
from .visualization_manager import VisualizationManager

try:
    from .preprocessing_dialogs import (GaussianBlurDialog, PlaneLevelingDialog, 
    MedianFilterDialog, NLMeansDialog, BM3DDialog, GaussianSharpeningDialog)
    from .fft_dialog import FFTDialog
    DIALOG_CLASSES_EXIST = True
except ImportError:
    GaussianBlurDialog = None
    PlaneLevelingDialog = None
    MedianFilterDialog = None
    NLMeansDialog = None
    BM3DDialog = None
    GaussianSharpeningDialog = None
    FFTDialog = None
    DIALOG_CLASSES_EXIST = False
    logging.warning("Could not import preprocessing dialogs. Preprocessing options may be unavailable.")

try:
    from ..analysis.peak_fitting import find_max_pixel_in_roi, fit_2d_gaussian_in_roi
    PEAK_FITTING_AVAILABLE = True
except ImportError:
    logging.error("Could not import peak fitting functions.")
    # Dummy functions if module is missing
    def find_max_pixel_in_roi(data, center, radius): return center
    def fit_2d_gaussian_in_roi(data, center, radius): return None
    PEAK_FITTING_AVAILABLE = False

logger = logging.getLogger(__name__)


try:
    from ..analysis.lattice import get_reciprocal_points, KNOWN_LATTICES
    from .custom_lattice_dialog import CustomLatticeDialog
    LATTICE_ANALYSIS_AVAILABLE = True
except ImportError:
    logging.error("Could not import lattice analysis functions.")
    KNOWN_LATTICES = {"None": {}} # Placeholder
    def get_reciprocal_points(name, max_hk=2): return None
    CustomLatticeDialog = None
    LATTICE_ANALYSIS_AVAILABLE = False

class MainWindow(QMainWindow):
    """
    The main application window inheriting from QMainWindow.
    Provides menu bar, status bar, and central widget area.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Lattice Fourier Analyzer (LFA)")
        self.resize(1250, 800)

        self._init_core_attributes()
        self._setup_main_layout()
        self._create_menus()
        self._create_status_bar()
        self._create_history_dock()
        self._create_metadata_dock()
        self._create_fft_analysis_dock() # Ta metoda będzie refaktoryzowana w kolejnych krokach

        # === TUTAJ TWORZYMY VISUALIZATION MANAGER ===
        # Musi być po utworzeniu self.image_view i self.history_manager
        if pg and self.image_view and self.history_manager and VisualizationManager:
            self.visualization_manager = VisualizationManager(
                image_view=self.image_view,
                history_manager=self.history_manager,
                # parent=self # Opcjonalnie, jeśli chcesz, aby MainWindow było rodzicem QObject
            )
            logger.info("VisualizationManager created and initialized.")
        else: # pragma: no cover
            self.visualization_manager = None
            logger.error("Could not create VisualizationManager due to missing dependencies (pg, ImageView, HistoryManager, or VisualizationManager class).")
        # ==========================================


        self._connect_signals() # Dedykowana metoda do podłączania sygnałów

        self._update_action_states() # Aktualizacja stanu akcji na starcie
        logger.info("Main window initialized.")

    def _init_core_attributes(self):
        """Initializes non-widget core attributes of the MainWindow."""
        self.original_file_path: Optional[str] = None
        self.history_manager: Optional[HistoryManager] = None # Zostanie zainicjalizowany w _create_history_dock

        # Spot Selection Attributes (później mogą trafić do kontrolera)
        self.substrate_spots: List[Tuple[float, float]] = []
        self.adsorbate_spot_sets: List[List[Tuple[float, float]]] = [[]]
        self.current_adsorbate_set_index: int = 0
        self.spot_selection_mode: str = "Substrate"
        self._points_for_current_adsorbate_set: List[Tuple[float, float]] = []
        self.spot_refinement_method: str = "Direct Click"
        self.refinement_roi_size: int = 5
        self.custom_lattice_info: Optional[Dict[str, Any]] = None # Dla CustomLatticeDialog
        self.last_selected_substrate: str = "None" # Do zapamiętania ostatniego wyboru

        # Visual Markers (później mogą trafić do VisualizationManager)
        # self.ideal_lattice_overlay_item: Optional['pg.ScatterPlotItem'] = None
        # self.substrate_spot_markers: Optional['pg.ScatterPlotItem'] = None
        # self.adsorbate_spot_set_markers: List['pg.ScatterPlotItem'] = []
        self._fft_mouse_click_connection = None
        # Atrybut dla QTextEdit wyświetlającego koordynaty (jeśli jest w głównym oknie)
        # self.selected_spots_display: Optional[QTextEdit] = None
        # self.current_selection_label: Optional[QLabel] = None

    def _setup_main_layout(self):
        """Sets up the central widget, main layout, and image view."""
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        self.splitter = QSplitter(Qt.Orientation.Horizontal) # self.splitter zamiast splitter

        # Image View Widget (Right Panel)
        image_view_container = QWidget()
        image_view_layout = QVBoxLayout(image_view_container)
        image_view_layout.setContentsMargins(0, 0, 0, 0)

        if pg:
            pg.setConfigOption('background', 'w')
            pg.setConfigOption('foreground', 'k')
            self.image_view = pg.ImageView(self) # self.image_view
            image_view_layout.addWidget(self.image_view)
        else:
            self.image_view = None
            logger.error("Cannot create ImageView because PyQtGraph is not available.")

        # Lewy panel (history_list_widget) zostanie dodany w _create_history_dock
        # self.splitter.addWidget(QWidget()) # Placeholder for history dock content area
        self.splitter.addWidget(image_view_container)
        self.splitter.setSizes([250, 950]) # Ustawienie rozmiarów
        main_layout.addWidget(self.splitter)
    
    def _create_status_bar(self):
        """Creates the status bar."""
        self.statusBar().showMessage("Ready - Load an image using File -> Open")

    def _create_history_dock(self):
        """Creates and configures the history list dock widget."""
        self.history_dock = QDockWidget("History", self) # self.history_dock
        self.history_list_widget = QListWidget()
        self.history_dock.setWidget(self.history_list_widget)
        self.history_dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.history_dock)

        # Utwórz instancję HistoryManager
        self.history_manager = HistoryManager(self.history_list_widget, self)

        # Dodaj akcję przełączania widoczności doku historii do menu "View"
        if hasattr(self, 'view_menu'): # Sprawdź, czy view_menu już istnieje
            toggle_history_action = self.history_dock.toggleViewAction()
            toggle_history_action.setText("History Panel")
            self.view_menu.addAction(toggle_history_action)
        else:
            logger.warning("view_menu not found when creating history_dock toggle action.")

    def _create_metadata_dock(self):
        """Creates and configures the metadata dock widget."""
        self.metadata_dock = QDockWidget("Metadata", self)
        self.metadata_widget = MetadataWidget(self)
        self.metadata_dock.setWidget(self.metadata_widget)
        self.metadata_dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.metadata_dock)

        if not hasattr(self, 'view_menu'):
            self.view_menu = self.menuBar().addMenu("&View")

        toggle_metadata_action = self.metadata_dock.toggleViewAction()
        toggle_metadata_action.setText("Metadata Panel")
        toggle_metadata_action.setStatusTip("Show/hide the metadata panel")
        self.view_menu.addAction(toggle_metadata_action)

    def _create_fft_analysis_dock(self):
        """
        Creates the FFT Analysis Tools dock widget and sets its content
        to an instance of FFTAnalysisPanel.
        """
        self.fft_analysis_dock = QDockWidget("FFT Analysis Tools", self)
        # Utwórz instancję nowego panelu
        self.fft_analysis_panel_widget = FFTAnalysisPanel(self) # Przekaż self jako parent
        self.fft_analysis_dock.setWidget(self.fft_analysis_panel_widget)

        self.fft_analysis_dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.fft_analysis_dock)
        self.fft_analysis_dock.setVisible(False) # Ukryj na starcie

        # Dodaj akcję przełączania widoczności do menu "View"
        if not hasattr(self, 'view_menu'):
            self.view_menu = self.menuBar().addMenu("&View")
        toggle_fft_tools_action = self.fft_analysis_dock.toggleViewAction()
        toggle_fft_tools_action.setText("FFT Analysis Tools Panel") # Zaktualizuj tekst
        self.view_menu.addAction(toggle_fft_tools_action)

    def _connect_signals(self):
        """Connects signals to slots for the main window components."""
        if self.history_manager: # Sprawdź, czy history_manager jest zainicjalizowany
            self.history_list_widget.currentItemChanged.connect(self.on_history_selection_changed)
            self.history_manager.current_node_changed.connect(self._on_current_history_node_changed)

        if hasattr(self, 'fft_analysis_panel_widget'):
            self.fft_analysis_panel_widget.substrate_changed.connect(self._handle_substrate_changed)
            self.fft_analysis_panel_widget.custom_lattice_define_requested.connect(self._handle_custom_lattice_request)
            self.fft_analysis_panel_widget.show_ideal_lattice_changed.connect(self._handle_show_ideal_lattice_changed)

            self.fft_analysis_panel_widget.spot_selection_mode_changed.connect(self._handle_spot_selection_mode_changed_from_panel)
            self.fft_analysis_panel_widget.current_adsorbate_set_changed.connect(self._handle_current_adsorbate_set_changed_from_panel)
            self.fft_analysis_panel_widget.add_new_adsorbate_set_requested.connect(self._handle_add_new_adsorbate_set_request)
            self.fft_analysis_panel_widget.reselect_current_adsorbate_set_triggered.connect(self._on_reselect_adsorbate_set_clicked) # Użyj istniejącego slotu jeśli pasuje
            self.fft_analysis_panel_widget.clear_all_adsorbate_sets_triggered.connect(self._on_clear_all_adsorbate_sets_clicked) # Użyj istniejącego slotu
            self.fft_analysis_panel_widget.clear_last_adsorbate_point_triggered.connect(self._on_clear_last_adsorbate_point_clicked) # Użyj istniejącego slotu
            self.fft_analysis_panel_widget.clear_substrate_spots_triggered.connect(self._on_clear_substrate_spots_clicked) # Użyj istniejącego slotu
            self.fft_analysis_panel_widget.substrate_spots_visibility_changed.connect(self._handle_substrate_spots_visibility_changed)
            self.fft_analysis_panel_widget.adsorbate_spots_visibility_changed.connect(self._handle_adsorbate_spots_visibility_changed)
            self.fft_analysis_panel_widget.refinement_method_changed.connect(self._handle_refinement_method_changed_from_panel)
            self.fft_analysis_panel_widget.refinement_area_size_changed.connect(self._handle_refinement_area_size_changed_from_panel)

        if hasattr(self, 'visualization_manager') and self.visualization_manager:
            self.visualization_manager.fft_view_clicked.connect(self._on_fft_view_clicked_from_visualizer)



    def _clear_all_spot_markers_from_view(self, view_box: Optional[pg.ViewBox]):
        """Helper to remove all known spot markers from the view."""
        if not view_box:
            logger.debug("_clear_all_spot_markers_from_view: No ViewBox provided.")
            return
        # Substrate markers
        # if hasattr(self, 'substrate_spot_markers') and self.substrate_spot_markers:
        #     try: view_box.removeItem(self.substrate_spot_markers)
        #     except RuntimeError: pass
        #     self.substrate_spot_markers = None
        # Adsorbate set markers
        # if hasattr(self, 'adsorbate_spot_set_markers'):
        #     for marker_set in self.adsorbate_spot_set_markers:
        #         if marker_set:
        #             try: view_box.removeItem(marker_set)
        #             except RuntimeError: pass
        #     self.adsorbate_spot_set_markers = []
        # Current adsorbate preview markers
        if hasattr(self, 'current_adsorbate_preview_markers') and self.current_adsorbate_preview_markers:
            try: view_box.removeItem(self.current_adsorbate_preview_markers)
            except RuntimeError: pass
            self.current_adsorbate_preview_markers = None
        logger.debug("Cleared all user-selected spot markers from view.")

    def _update_selected_spots_display(self):
        logger.debug("_update_selected_spots_display called (Phase B.2.3 - currently a placeholder).")
        if hasattr(self, 'current_selection_label'):
             self.current_selection_label.setText(f"Mode: {self.spot_selection_mode}")
        if hasattr(self, 'selected_spots_display'):
             self.selected_spots_display.setPlainText("Coordinates will appear here in Phase B.2.3.")

    def _update_selected_spots_display(self):
        """Updates the QTextEdit with current spot coordinates."""
        if not hasattr(self, 'selected_spots_display'): return # Jeśli dock nie jest jeszcze stworzony

        text_output = []
        current_selection_status = ""

        if self.spot_selection_mode == "Substrate":
            current_selection_status = "Selecting: Substrate Spots"
            text_output.append("Substrate Spots:")
            if self.substrate_spots:
                for i, (kx, ky) in enumerate(self.substrate_spots):
                    text_output.append(f"  S{i+1}: (kx={kx}, ky={ky})")
            else:
                text_output.append("  None selected.")
        elif self.spot_selection_mode == "Adsorbate":
            set_idx = self.current_adsorbate_set_index
            set_name = self.adsorbate_set_combo.itemText(set_idx) if hasattr(self, 'adsorbate_set_combo') and set_idx < self.adsorbate_set_combo.count() else f"Set {set_idx + 1}"
            current_selection_status = f"Selecting: Adsorbate {set_name}"
            text_output.append(f"Adsorbate {set_name}:")

            # Wyświetl punkty z aktualnie edytowanego/wybranego zestawu
            if 0 <= set_idx < len(self.adsorbate_spot_sets):
                current_points_to_display = self.adsorbate_spot_sets[set_idx]
                if current_points_to_display:
                    for i, (kx, ky) in enumerate(current_points_to_display):
                        text_output.append(f"  A{i+1}: (kx={kx}, ky={ky})")
                else:
                    text_output.append("  No spots selected for this set.")
            else:
                text_output.append("  Invalid adsorbate set selected.")

        if hasattr(self, 'current_selection_label') and self.current_selection_label:
            self.current_selection_label.setText(current_selection_status)
        self.selected_spots_display.setPlainText("\n".join(text_output))

    def _create_menus(self):
        """Creates the main menu bar and its actions."""
        menu_bar = self.menuBar()

        # --- File Menu ---
        file_menu = menu_bar.addMenu("&File")

        open_action = QAction("&Open...", self)
        open_action.setStatusTip("Open an STM data file")
        open_action.triggered.connect(self.open_file_dialog)
        open_action.setShortcut("Ctrl+O")
        file_menu.addAction(open_action)

        file_menu.addSeparator()

        exit_action = QAction("&Exit", self)
        exit_action.setStatusTip("Exit the application")
        exit_action.triggered.connect(self.close)
        exit_action.setShortcut("Ctrl+Q")
        file_menu.addAction(exit_action)

        # --- Preprocessing Menu ---
        preprocessing_menu = menu_bar.addMenu("&Preprocessing")

        self.gaussian_blur_action = QAction("&Gaussian Blur...", self)
        self.gaussian_blur_action.setStatusTip("Apply Gaussian blur filter")
        self.gaussian_blur_action.triggered.connect(self.open_gaussian_blur_dialog)
        self.gaussian_blur_action.setEnabled(False)
        preprocessing_menu.addAction(self.gaussian_blur_action)

        self.gaussian_sharpen_action = QAction("Gaussian &Sharpening...", self)
        self.gaussian_sharpen_action.setStatusTip("Apply Gaussian Sharpening (Unsharp Mask)")
        self.gaussian_sharpen_action.triggered.connect(self.open_gaussian_sharpening_dialog)
        preprocessing_menu.addAction(self.gaussian_sharpen_action)

        self.plane_level_action = QAction("&Plane Leveling...", self)
        self.plane_level_action.setStatusTip("Level image by subtracting a fitted plane")
        self.plane_level_action.triggered.connect(self.open_plane_leveling_dialog) 
        preprocessing_menu.addAction(self.plane_level_action)

        self.median_filter_action = QAction("&Median Filter...", self)
        self.median_filter_action.setStatusTip("Apply median filter for noise reduction")
        self.median_filter_action.triggered.connect(self.open_median_filter_dialog) 
        preprocessing_menu.addAction(self.median_filter_action)

        self.nlmeans_action = QAction("&NL-Means Denoising...", self)
        self.nlmeans_action.setStatusTip("Apply Non-Local Means denoising (skimage)")
        self.nlmeans_action.triggered.connect(self.open_nlmeans_dialog) 
        preprocessing_menu.addAction(self.nlmeans_action)

        self.bm3d_action = QAction("&BM3D Denoising...", self)
        self.bm3d_action.setStatusTip("Apply BM3D denoising (Computationally intensive)")
        self.bm3d_action.triggered.connect(self.open_bm3d_dialog) 
        preprocessing_menu.addAction(self.bm3d_action)

        # --- Analysis Menu ---
        analysis_menu = menu_bar.addMenu("&Analysis")
        self.fft_action = QAction("Calculate &FFT...", self)
        self.fft_action.setStatusTip("Calculate Fast Fourier Transform")
        self.fft_action.triggered.connect(self.open_fft_dialog)
        analysis_menu.addAction(self.fft_action)

        # --- Help Menu ---
        help_menu = menu_bar.addMenu("&Help")

        about_action = QAction("&About LFA...", self)
        about_action.setStatusTip("Show information about LFA")
        about_action.triggered.connect(self.show_about_dialog)
        help_menu.addAction(about_action)

        logger.debug("Menu bar created.")
    
    def _update_action_states(self):
        """Enables/disables actions and UI elements based on the current application state."""
        current_node = self.history_manager.get_current_node()
        has_node = current_node is not None
        is_stm_data = False
        is_fft_data = False

        if has_node:
            current_node_data_type = current_node.data_type
            is_stm_data = (current_node_data_type == "STM")
            is_fft_data = (current_node_data_type == "FFT")

        # --- Akcje Preprocessingu i Analizy (Menu) ---
        # Dostępne jeśli jakikolwiek węzeł jest aktywny (można by bardziej szczegółowo)
        # Jeśli operacja ma sens tylko dla STM lub tylko dla FFT, można to tu zawęzić.
        preprocessing_possible = has_node # Ogólnie, większość preprocessingu na STM
        fft_calculation_possible = is_stm_data # FFT typowo z danych STM

        if hasattr(self, 'gaussian_blur_action'):
            self.gaussian_blur_action.setEnabled(preprocessing_possible)
        if hasattr(self, 'plane_level_action'):
            self.plane_level_action.setEnabled(preprocessing_possible)
        if hasattr(self, 'median_filter_action'):
            self.median_filter_action.setEnabled(preprocessing_possible)
        if hasattr(self, 'nlmeans_action'):
            self.nlmeans_action.setEnabled(preprocessing_possible)
        if hasattr(self, 'bm3d_action'):
            self.bm3d_action.setEnabled(preprocessing_possible)
        if hasattr(self, 'gaussian_sharpen_action'):
            self.gaussian_sharpen_action.setEnabled(preprocessing_possible)
        
        if hasattr(self, 'fft_action'):
            self.fft_action.setEnabled(fft_calculation_possible)

        # --- Widoczność Paneli Dokowalnych ---
        if hasattr(self, 'fft_analysis_dock'):
            self.fft_analysis_dock.setVisible(is_fft_data)

        # --- Stan Kontrolek w FFTAnalysisPanel ---
        if hasattr(self, 'fft_analysis_panel_widget') and is_fft_data:
            can_clear_substrate = self.spot_selection_mode == "Substrate" and bool(self.substrate_spots)
            self.fft_analysis_panel_widget.set_clear_substrate_spots_button_enabled(can_clear_substrate)

            # Logika dla przycisku "Clear Last Adsorbate Point"
            can_clear_last_adsorbate = False
            if self.spot_selection_mode == "Adsorbate" and \
               0 <= self.current_adsorbate_set_index < len(self.adsorbate_spot_sets):
                if self.adsorbate_spot_sets[self.current_adsorbate_set_index]: # Czy są punkty w bieżącym zestawie
                    can_clear_last_adsorbate = True
            self.fft_analysis_panel_widget.set_clear_last_adsorbate_point_button_enabled(can_clear_last_adsorbate)

            is_adsorbate_mode_active = (self.spot_selection_mode == "Adsorbate")
            self.fft_analysis_panel_widget.set_reselect_adsorbate_set_button_enabled(is_adsorbate_mode_active)
            can_clear_all_adsorbate = is_adsorbate_mode_active and any(s for s in self.adsorbate_spot_sets)
            self.fft_analysis_panel_widget.set_clear_all_adsorbate_sets_button_enabled(can_clear_all_adsorbate)

        elif hasattr(self, 'fft_analysis_panel_widget'): # Jeśli nie jest FFT, ale panel istnieje
            # Wyłącz wszystkie przyciski akcji w panelu, jeśli nie jest wyświetlany obraz FFT
            self.fft_analysis_panel_widget.set_clear_substrate_spots_button_enabled(False)
            self.fft_analysis_panel_widget.set_clear_last_adsorbate_point_button_enabled(False)
            self.fft_analysis_panel_widget.set_reselect_adsorbate_set_button_enabled(False)
            self.fft_analysis_panel_widget.set_clear_all_adsorbate_sets_button_enabled(False)
            # Można też wyłączyć inne kontrolki, jeśli panel nie jest aktywny

        logger.debug(f"_update_action_states: Preprocessing possible: {preprocessing_possible}, FFT Calc possible: {fft_calculation_possible}, Is FFT data: {is_fft_data}")



    @pyqtSlot(str)
    def _handle_substrate_changed(self, substrate_name: str):
        """Obsługuje zmianę wybranego substratu z FFTAnalysisPanel."""
        logger.debug(f"MainWindow: Substrate changed to '{substrate_name}' via panel signal.")
        self.last_selected_substrate = substrate_name # Zaktualizuj stan w MainWindow
        self.custom_lattice_info = None # Wyczyść info o custom, jeśli wybrano predefiniowaną
        self.display_image_data() # display_image_data będzie musiało użyć self.last_selected_substrate

    @pyqtSlot(str)
    def _handle_refinement_method_changed_from_panel(self, method: str):
        """Obsługuje zmianę metody uściślania pików z panelu."""
        logger.debug(f"MainWindow: Refinement method changed to '{method}' via panel signal.")
        self.spot_refinement_method = method # Aktualizuj stan w MainWindow (lub przekaż do SpotSelectionController)

    @pyqtSlot(int)
    def _handle_refinement_area_size_changed_from_panel(self, area_size: int):
        """Obsługuje zmianę rozmiaru obszaru uściślania z panelu."""
        logger.debug(f"MainWindow: Refinement area size changed to {area_size} via panel signal.")
        self.refinement_roi_size = area_size # Aktualizuj stan w MainWindow

    @pyqtSlot()
    def _handle_custom_lattice_request(self):
        """Obsługuje żądanie zdefiniowania własnej sieci z FFTAnalysisPanel."""
        logger.debug("MainWindow: Custom lattice definition requested via panel signal.")
        if not CustomLatticeDialog:
            QMessageBox.critical(self, "Error", "CustomLatticeDialog is not available.")
            return

        dialog = CustomLatticeDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.custom_lattice_info = dialog.get_lattice_definition()
            if self.custom_lattice_info and self.fft_analysis_panel_widget:
                # Ustaw nazwę nowej sieci w combo boxie w panelu
                new_name = self.custom_lattice_info.get("name", "Custom")
                self.fft_analysis_panel_widget.set_substrate_combo_text(new_name) # Użyj nowej metody publicznej
                self.last_selected_substrate = new_name # lub specjalny identyfikator dla custom
                logger.info(f"Custom lattice '{new_name}' defined and selected.")
                self.display_image_data() # Odśwież widok
            else:
                if self.fft_analysis_panel_widget:
                    self.fft_analysis_panel_widget.set_substrate_combo_text(self.last_selected_substrate)
        else: # Dialog anulowany
            logger.debug("Custom lattice definition dialog was cancelled.")
            if self.fft_analysis_panel_widget:
                self.fft_analysis_panel_widget.set_substrate_combo_text(self.last_selected_substrate)
    
    @pyqtSlot(str)
    def _handle_current_adsorbate_set_changed_from_panel(self, set_name: str):
        found_idx = -1
        if hasattr(self, 'fft_analysis_panel_widget'): # Sprawdzenie dla bezpieczeństwa
            combo = self.fft_analysis_panel_widget.adsorbate_set_combo
            for i in range(combo.count()):
                if combo.itemText(i) == set_name and set_name != "<Add New Set...>":
                    found_idx = i
                    break
        
        if found_idx != -1:
            self.current_adsorbate_set_index = found_idx
            logger.info(f"MainWindow: Switched to adsorbate set '{set_name}' (Index: {self.current_adsorbate_set_index}) via panel signal.")
            self._points_for_current_adsorbate_set = [] # Wyczyść tymczasowe
            if self.current_adsorbate_set_index < len(self.adsorbate_spot_sets):
                self._points_for_current_adsorbate_set = list(self.adsorbate_spot_sets[self.current_adsorbate_set_index])
        else:
             logger.warning(f"MainWindow: Could not map adsorbate set name '{set_name}' to an index.")

        self._update_selected_spots_display()
        self.request_spot_markers_update()
        self._update_action_states()


    @pyqtSlot(bool)
    def _handle_show_ideal_lattice_changed(self, is_visible: bool):
        """Obsługuje zmianę widoczności idealnej siatki z FFTAnalysisPanel."""
        logger.debug(f"MainWindow: Show ideal lattice changed to {is_visible} via panel signal.")
        self.display_image_data()

    @pyqtSlot(bool)
    def _handle_substrate_spots_visibility_changed(self, is_visible: bool):
        logger.debug(f"MainWindow: Substrate spots visibility changed to {is_visible} via panel.")
        if hasattr(self, 'visualization_manager') and self.visualization_manager:
            # Pobierz aktualne flagi dla adsorbatu, aby ich nie nadpisać
            show_ads = False
            if hasattr(self, 'fft_analysis_panel_widget') and self.fft_analysis_panel_widget:
                show_ads = self.fft_analysis_panel_widget.is_adsorbate_spots_visible()
            
            self.visualization_manager.redraw_spot_markers(
                self.substrate_spots, is_visible,
                self.adsorbate_spot_sets, show_ads
            )

    @pyqtSlot(bool)
    def _handle_adsorbate_spots_visibility_changed(self, is_visible: bool):
        logger.debug(f"MainWindow: Adsorbate spots visibility changed to {is_visible} via panel.")
        if hasattr(self, 'visualization_manager') and self.visualization_manager:
            # Pobierz aktualne flagi dla substratu
            show_sub = False
            if hasattr(self, 'fft_analysis_panel_widget') and self.fft_analysis_panel_widget:
                show_sub = self.fft_analysis_panel_widget.is_substrate_spots_visible()

            self.visualization_manager.redraw_spot_markers(
                self.substrate_spots, show_sub,
                self.adsorbate_spot_sets, is_visible
            )
    
    def request_spot_markers_update(self): # Nowa metoda pomocnicza
        """Requests an update of only the spot markers."""
        if hasattr(self, 'visualization_manager') and self.visualization_manager and \
           hasattr(self, 'fft_analysis_panel_widget') and self.fft_analysis_panel_widget:
            
            current_node = self.history_manager.get_current_node()
            if current_node and current_node.data_type == "FFT": # Rysuj markery tylko dla FFT
                show_sub = self.fft_analysis_panel_widget.is_substrate_spots_visible()
                show_ads = self.fft_analysis_panel_widget.is_adsorbate_spots_visible()
                self.visualization_manager.redraw_spot_markers(
                    self.substrate_spots, show_sub,
                    self.adsorbate_spot_sets, show_ads
                )
            else: # Jeśli nie FFT, upewnij się, że markery są wyczyszczone
                 self.visualization_manager._clear_spot_markers_only()

    @pyqtSlot()
    def _handle_add_new_adsorbate_set_request(self):
        logger.info("MainWindow: Add new adsorbate set requested via panel signal.")
        new_set_name = f"Set {len(self.adsorbate_spot_sets) + 1}"
        self.adsorbate_spot_sets.append([]) # Dodaj nowy pusty zestaw do danych MainWindow
        self.current_adsorbate_set_index = len(self.adsorbate_spot_sets) - 1 # Ustaw jako bieżący

        # Zaktualizuj QComboBox w panelu
        if hasattr(self, 'fft_analysis_panel_widget'):
            set_names_for_combo = [f"Set {i+1}" for i in range(len(self.adsorbate_spot_sets))]
            self.fft_analysis_panel_widget.update_adsorbate_set_combo(set_names_for_combo, new_set_name)
        
        self._points_for_current_adsorbate_set = [] # Nowy zestaw jest pusty
        self._update_selected_spots_display()
        self.request_spot_markers_update()
        self._update_action_states()

    @pyqtSlot(object)
    def _on_current_history_node_changed(self, current_node: Optional[HistoryNode]):
        logger.debug(f"MainWindow: Slot _on_current_history_node_changed received node: {current_node.node_id if current_node else 'None'}")
        
        self.display_image_data() # Ta metoda zbierze wszystko i wywoła visualization_manager

        if hasattr(self, 'metadata_widget') and self.metadata_widget and hasattr(self, 'history_manager'):
            # Przekazujemy teraz obiekt HistoryManager, a nie słownik
            self.metadata_widget.update_metadata(current_node, self.history_manager) 

    @pyqtSlot()
    def open_file_dialog(self):
        logger.debug("Open file dialog triggered.")
        file_filter = "STM Files (*.stp *.s94);;All Files (*)"
        start_dir = ""
        current_active_node = self.history_manager.get_current_node()
        if current_active_node:
            try:
                if hasattr(self, 'original_file_path') and self.original_file_path:
                    start_dir = os.path.dirname(self.original_file_path)
            except Exception:
                pass
        if not start_dir:
            start_dir = os.path.expanduser("~")

        file_path, _ = QFileDialog.getOpenFileName(self, "Open STM File", start_dir, file_filter)

        if file_path:
            logger.info(f"File selected: {file_path}")
            self.statusBar().showMessage(f"Loading file: {os.path.basename(file_path)}...")
            QApplication.processEvents()

            stm_image_obj = load_stm_file(file_path)

            if stm_image_obj and stm_image_obj.data is not None:
                self.original_file_path = file_path
                self.history_manager.clear_history()

                root_params = {
                    "filename": os.path.basename(file_path),
                    "pixels_x": stm_image_obj.pixels_x,
                    "pixels_y": stm_image_obj.pixels_y,
                    "size_nm_x": stm_image_obj.size_nm_x,
                    "size_nm_y": stm_image_obj.size_nm_y,
                    "bias_v": stm_image_obj.bias_v,
                    "setpoint_a": stm_image_obj.setpoint_a,
                    "scan_angle_deg": stm_image_obj.scan_angle_deg,
                    # Dodaj inne potrzebne standardowe pola
                }

                 # root_params['raw_header'] = stm_image_obj.raw_header

                root_node = HistoryNode(
                    operation_name="Original",
                    image_data=stm_image_obj.data.copy(),
                    parameters=root_params,
                    data_type="STM"
                )
                root_item = self.history_manager.add_node(root_node)
                self.history_manager.set_current_node_by_id(root_node.node_id)

                logger.info("File loaded successfully and history initialized.")
                self.statusBar().showMessage(f"Loaded: {os.path.basename(file_path)}", 5000)
                self.setWindowTitle(f"LFA - {os.path.basename(file_path)}")
            else:
                self.history_manager.clear_history()
                self.statusBar().showMessage("Failed to load file.", 5000)
                QMessageBox.warning(self, "Loading Error", f"Could not load file: {file_path}")
                self.setWindowTitle("Lattice Fourier Analyzer (LFA)")
        else:
            logger.debug("File dialog cancelled.")
            self.statusBar().showMessage("File open cancelled.", 3000)

    # --- Sloty dla Kontrolek FFT Analysis ---

    @pyqtSlot(bool)
    def _on_spot_type_changed(self, is_substrate_selected):
        """Handles change between Substrate and Adsorbate spot selection."""
        if is_substrate_selected:
            self.spot_selection_mode = "Substrate"
            self.adsorbate_set_panel.setVisible(False)
            self.substrate_set_panel.setVisible(True)
            logger.debug("Spot selection mode: Substrate")
        else: # Adsorbate selected
            self.spot_selection_mode = "Adsorbate"
            self.adsorbate_set_panel.setVisible(True)
            self.substrate_set_panel.setVisible(False)
            # Zresetuj/ustaw odpowiednio _points_for_current_adsorbate_set
            self._points_for_current_adsorbate_set = [] # Wyczyść przy zmianie na adsorbat
            if self.current_adsorbate_set_index < len(self.adsorbate_spot_sets):
                 # Jeśli przełączamy z powrotem, załaduj punkty z bieżącego zestawu
                 self._points_for_current_adsorbate_set = list(self.adsorbate_spot_sets[self.current_adsorbate_set_index])
            logger.debug(f"Spot selection mode: Adsorbate, Set Index: {self.current_adsorbate_set_index}")
        self._update_selected_spots_display() # Zaktualizuj wyświetlanie
        self.request_spot_markers_update()

    @pyqtSlot(str)
    def _on_adsorbate_set_combo_changed(self, text: str):
        """Handles selection or addition of an adsorbate set."""
        if text == "<Add New Set...>":
            new_set_name = f"Set {len(self.adsorbate_spot_sets) + 1}"
            self.adsorbate_spot_sets.append([]) # Dodaj nowy pusty zestaw
            self.adsorbate_set_combo.blockSignals(True)
            self.adsorbate_set_combo.insertItem(self.adsorbate_set_combo.count() - 1, new_set_name)
            self.adsorbate_set_combo.setCurrentText(new_set_name)
            self.adsorbate_set_combo.blockSignals(False)
            self.current_adsorbate_set_index = self.adsorbate_set_combo.currentIndex()
            logger.info(f"Added new adsorbate set: {new_set_name}")
        else:
            self.current_adsorbate_set_index = self.adsorbate_set_combo.currentIndex()
            logger.info(f"Switched to adsorbate set: {text} (Index: {self.current_adsorbate_set_index})")

        # Wyczyść tymczasowe punkty i załaduj punkty z nowo wybranego setu (jeśli istnieją)
        self._points_for_current_adsorbate_set = []
        if self.current_adsorbate_set_index < len(self.adsorbate_spot_sets):
            self._points_for_current_adsorbate_set = list(self.adsorbate_spot_sets[self.current_adsorbate_set_index])

        self._update_selected_spots_display()
        self.request_spot_markers_update()
        self._update_action_states()


    @pyqtSlot()
    def _on_reselect_adsorbate_set_clicked(self):
        """Clears points for the current adsorbate set to allow re-selection."""
        if self.current_adsorbate_set_index >= 0 and self.current_adsorbate_set_index < len(self.adsorbate_spot_sets):
            logger.info(f"Reselecting points for adsorbate set {self.current_adsorbate_set_index + 1}")
            self.adsorbate_spot_sets[self.current_adsorbate_set_index] = [] # Wyczyść zapisane punkty
            self._points_for_current_adsorbate_set = [] # Wyczyść punkty tymczasowe
            # Można dodać aktualizację etykiety informującej o wybieraniu punktów
            self._update_selected_spots_display()
            self.request_spot_markers_update()
            self._update_action_states()
        else:
            logger.warning("No valid adsorbate set selected to reselect.")

    @pyqtSlot()
    def _on_clear_all_adsorbate_sets_clicked(self):
        """Clears all adsorbate spot sets and resets the combo box."""
        logger.info("Clearing all adsorbate spot sets.")
        self.adsorbate_spot_sets = [[]] # Zostaw jeden pusty zestaw
        self._points_for_current_adsorbate_set = []
        self.current_adsorbate_set_index = 0
        
        if hasattr(self, 'fft_analysis_panel_widget'):
            self.fft_analysis_panel_widget.adsorbate_set_combo.blockSignals(True)
            self.fft_analysis_panel_widget.adsorbate_set_combo.clear()
            self.fft_analysis_panel_widget.adsorbate_set_combo.addItem("Set 1")
            self.fft_analysis_panel_widget.adsorbate_set_combo.addItem("<Add New Set...>")
            self.fft_analysis_panel_widget.adsorbate_set_combo.setCurrentIndex(0)
            self.fft_analysis_panel_widget.adsorbate_set_combo.blockSignals(False)

        self._update_selected_spots_display()
        self.request_spot_markers_update()
        self._update_action_states()

    @pyqtSlot(str)
    def _handle_spot_selection_mode_changed_from_panel(self, mode: str):
        logger.debug(f"MainWindow: Spot selection mode changed to '{mode}' via panel.")
        self.spot_selection_mode = mode # Aktualizuj stan w MainWindow (lub przekaż do SpotSelectionController)
        # Logika związana z przełączeniem trybu (np. resetowanie tymczasowych list pików)
        # powinna być teraz w SpotSelectionController, jeśli istnieje, lub tutaj, jeśli jeszcze nie.
        if mode == "Adsorbate":
            self._points_for_current_adsorbate_set = [] # Wyczyść przy przejściu na adsorbat
            if self.current_adsorbate_set_index < len(self.adsorbate_spot_sets):
                 self._points_for_current_adsorbate_set = list(self.adsorbate_spot_sets[self.current_adsorbate_set_index])
        self._update_selected_spots_display() # Zaktualizuj wyświetlanie tekstowe
        self.request_spot_markers_update()
        self._update_action_states()          # Zaktualizuj stan przycisków (np. "Clear Last Point")


    @pyqtSlot()
    def _on_visibility_checkbox_changed(self):
        """Slot for all visibility checkboxes."""
        logger.debug("Visibility checkbox changed, updating markers and ideal lattice.")
        # Odśwież idealną sieć (jeśli FFT) i markery spotów
        self.display_image_data() # To odświeży idealną sieć, jeśli trzeba
        self.request_spot_markers_update()

    @pyqtSlot(str)
    def on_substrate_combo_changed(self, selected_text: str):
        """Handles selection change in the substrate combobox."""
        logger.debug(f"Substrate selection changed to: {selected_text}")

        if selected_text == self.custom_option_text:
            dialog = CustomLatticeDialog(self)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                self.custom_lattice_info = dialog.get_lattice_definition()
                if self.custom_lattice_info:
                    self.last_selected_substrate = self.custom_option_text # Zapamiętaj wybór
                    logger.info(f"Custom lattice defined and selected: {self.custom_lattice_info.get('name')}")
                else: # Dialog zaakceptowany, ale brak definicji (nie powinno się zdarzyć)
                    self.custom_lattice_info = None
                    self.substrate_combo.setCurrentText(self.last_selected_substrate) # Wróć do poprzedniego
            else: # Custom dialog anulowany
                self.custom_lattice_info = None
                # Wróć do poprzednio wybranej opcji (lub "None")
                self.substrate_combo.setCurrentText(self.last_selected_substrate)
        else:
            # Wybrano predefiniowaną sieć lub "None"
            self.custom_lattice_info = None # Wyczyść definicję własną
            self.last_selected_substrate = selected_text # Zapamiętaj wybór

        # Zawsze odśwież wyświetlanie (co wywoła logikę nakładki)
        self.display_image_data()

    @pyqtSlot()
    def open_fft_dialog(self):
        """
        Opens the dialog for calculating the Fast Fourier Transform (FFT).

        Retrieves data from the current history node, executes the FFTDialog,
        and adds the resulting scaled FFT magnitude to the history if accepted.
        """
        logger.info("--- open_fft_dialog slot entered ---")

        # --- Pre-checks ---
        current_node = self.history_manager.get_current_node() # Pobierz aktualny węzeł
        if current_node is None: # Sprawdź, czy węzeł istnieje
            logger.warning("open_fft_dialog: No current node selected.")
            QMessageBox.warning(self, "No Image", "No data loaded or selected in history.")
            return

        if not FFTDialog:
            logger.error("open_fft_dialog: FFTDialog class is None (import failed?).")
            QMessageBox.critical(self, "Error", "FFTDialog class not available. Check imports and file.")
            return

        if current_node.image_data is None:
            logger.error(f"open_fft_dialog: Image data missing for node {self.history_manager.current_node_id}.")
            QMessageBox.critical(self, "Internal Error", "No image data in the current history node.")
            return
        # ------------------

        # Pass a copy of the current data to the dialog
        dialog_input_data = current_node.image_data.copy()

        logger.info(f"Opening FFT dialog based on node: {current_node.get_display_text()}")
        # Create and execute the dialog
        dialog = FFTDialog(dialog_input_data, parent=self)
        result = dialog.exec() # Show the dialog modally

        # --- Process Dialog Result ---
        if result == QDialog.DialogCode.Accepted:
            # Retrieve results from the dialog methods
            processed_fft_data = dialog.get_processed_data() # Gets the scaled magnitude (float)
            params = dialog.get_fft_parameters() # Gets params like window, scaling mode, roi checkbox
            source_roi = dialog.get_source_roi_slice() # Gets the ROI slice used, or None

            if processed_fft_data is not None:
                logger.info(f"FFT accepted. Source ROI: {source_roi}. Params: {params}. Creating history node.")

                # Create the new history node for the FFT result
                new_node = HistoryNode(
                    parent_id=self.history_manager.current_node_id,
                    operation_name="FFT",
                    parameters=params, # Store window type, scaling mode, roi checkbox state
                    image_data=processed_fft_data, # Store the scaled magnitude (float)
                    data_type="FFT", # Mark data type as FFT
                    source_roi_slice=source_roi # Store the source ROI slice if used
                )

                # Add node to history and update UI
                new_item = self.history_manager.add_node(new_node)
                self.history_manager.set_current_node_by_id(new_node.node_id)
                self.history_list_widget.setCurrentItem(new_item)
                display_name = new_node.get_display_text() # Should include "(FFT)" and "(from ROI)"
                self.statusBar().showMessage(f"{display_name} calculated.", 3000)
            else:
                # This case should ideally be handled by the dialog's accept logic,
                # but good to have a fallback log here.
                logger.warning("FFT Dialog was accepted, but returned no processed data.")
        else:
            # Dialog was cancelled
            logger.info("FFT dialog cancelled.")
            self.statusBar().showMessage("FFT calculation cancelled.", 3000)


    @pyqtSlot()
    def open_gaussian_sharpening_dialog(self):
        """Opens the dialog for applying Gaussian Sharpening."""
        current_node = self.history_manager.get_current_node() # Pobierz aktualny węzeł
        if current_node is None: # Sprawdź, czy węzeł istnieje
            QMessageBox.warning(self, "No Image", "...")
            return
        if not GaussianSharpeningDialog: 
            QMessageBox.critical(self, "Error", "GaussianSharpeningDialog not available.")
            return

        if current_node.image_data is None: 
            QMessageBox.critical(self, "Internal Error", "..."); return
        dialog_input_data = current_node.image_data.copy()

        logger.info(f"Opening Gaussian Sharpening dialog based on node: {current_node.get_display_text()}")
        dialog = GaussianSharpeningDialog(dialog_input_data, parent=self)
        result = dialog.exec()

        if result == QDialog.DialogCode.Accepted:
            processed_data = dialog.get_processed_data()
            params = dialog.get_parameters()
            was_roi_only = dialog.was_roi_applied_only()
            op_name = "Gaussian Sharpening"

            if processed_data is not None:
                logger.info(f"Sharpening accepted. ROI Only: {was_roi_only}. Creating history node.")
                parent_data_type = current_node.data_type
                logger.info(f"Creating history node. Parent type: {parent_data_type}. ROI Only: {was_roi_only}.")

                
                final_roi_slice = None
                if was_roi_only:
                    final_roi_slice = dialog.get_final_roi_slice()
                
                new_node = HistoryNode(
                    parent_id=self.history_manager.current_node_id,
                    operation_name=op_name,
                    parameters=params,
                    image_data=processed_data,
                    data_type=parent_data_type,
                    source_roi_slice=final_roi_slice
                )
                self.history_manager.add_node(new_node)
                self.history_manager.set_current_node_by_id(new_node.node_id)
                display_name = new_node.get_display_text()
                self.statusBar().showMessage(f"{display_name} applied.", 3000)
            else: logger.warning("Dialog accepted, but no processed data returned.")
        else: logger.info("Gaussian Sharpening dialog cancelled."); self.statusBar().showMessage("Sharpening cancelled.", 3000)


    @pyqtSlot()
    def open_bm3d_dialog(self):
        """Opens the dialog for applying BM3D Denoising."""
        current_node = self.history_manager.get_current_node() # Pobierz aktualny węzeł
        if current_node is None: # Sprawdź, czy węzeł istnieje
            QMessageBox.warning(self, "No Image", "...")
            return
        if not BM3DDialog: QMessageBox.critical(self, "Error", "BM3DDialog not available."); return
        try: import bm3d
        except ImportError: 
            QMessageBox.critical(self, "Missing Dependency", "The 'bm3d' package is required for this feature.\nPlease install it (pip install bm3d).")
            return

        if current_node.image_data is None: 
            QMessageBox.critical(self, "Internal Error", "...")
            return
        dialog_input_data = current_node.image_data.copy()

        logger.info(f"Opening BM3D dialog based on node: {current_node.get_display_text()}")
        dialog = BM3DDialog(dialog_input_data, parent=self)
        result = dialog.exec()

        if result == QDialog.DialogCode.Accepted:
            processed_data = dialog.get_processed_data()
            params = dialog.get_parameters()
            was_roi_only = dialog.was_roi_applied_only()
            op_name = "BM3D"

            if processed_data is not None:
                if np.allclose(processed_data, current_node.image_data): 
                    logger.info("Data not modified.")
                    self.statusBar().showMessage("No changes applied.", 3000)
                    return

                logger.info(f"BM3D accepted. ROI Only: {was_roi_only}. Creating history node.")

                parent_data_type = current_node.data_type
                logger.info(f"Creating history node. Parent type: {parent_data_type}. ROI Only: {was_roi_only}.")

                
                final_roi_slice = None
                if was_roi_only:
                    final_roi_slice = dialog.get_final_roi_slice()
                
                new_node = HistoryNode(
                    parent_id=self.history_manager.current_node_id,
                    operation_name=op_name,
                    parameters=params,
                    image_data=processed_data,
                    data_type=parent_data_type,
                    source_roi_slice=final_roi_slice
                )
                self.history_manager.add_node(new_node)
                self.history_manager.set_current_node_by_id(new_node.node_id)
                display_name = new_node.get_display_text()
                self.statusBar().showMessage(f"{display_name} applied.", 3000)
            else: logger.warning("Dialog accepted, but no processed data returned.")
        else: logger.info("BM3D dialog cancelled."); self.statusBar().showMessage("BM3D cancelled.", 3000)


    @pyqtSlot()
    def open_nlmeans_dialog(self):
        """Opens the dialog for applying NL-Means Denoising."""
        current_node = self.history_manager.get_current_node() # Pobierz aktualny węzeł
        if current_node is None: # Sprawdź, czy węzeł istnieje
            QMessageBox.warning(self, "No Image", "...")
            return
        if not NLMeansDialog: 
            QMessageBox.critical(self, "Error", "NLMeansDialog not available.")
            return

        if current_node.image_data is None: QMessageBox.critical(self, "Internal Error", "..."); return
        dialog_input_data = current_node.image_data.copy()

        logger.info(f"Opening NL-Means dialog based on node: {current_node.get_display_text()}")
        dialog = NLMeansDialog(dialog_input_data, parent=self)
        result = dialog.exec()

        if result == QDialog.DialogCode.Accepted:
            processed_data = dialog.get_processed_data()
            params = dialog.get_parameters()
            was_roi_only = dialog.was_roi_applied_only()
            op_name = "NL-Means" 

            if processed_data is not None:
                logger.info(f"NL-Means accepted. ROI Only: {was_roi_only}. Creating history node.")

                parent_data_type = current_node.data_type
                logger.info(f"Creating history node. Parent type: {parent_data_type}. ROI Only: {was_roi_only}.")

                
                final_roi_slice = None
                if was_roi_only:
                    final_roi_slice = dialog.get_final_roi_slice()
                
                new_node = HistoryNode(
                    parent_id=self.history_manager.current_node_id,
                    operation_name=op_name,
                    parameters=params,
                    image_data=processed_data,
                    data_type=parent_data_type,
                    source_roi_slice=final_roi_slice
                )
                self.history_manager.add_node(new_node)
                self.history_manager.set_current_node_by_id(new_node.node_id)
                display_name = new_node.get_display_text()
                self.statusBar().showMessage(f"{display_name} applied.", 3000)
            else: logger.warning("Dialog accepted, but no processed data returned.")
        else: logger.info("NL-Means dialog cancelled."); self.statusBar().showMessage("NL-Means cancelled.", 3000)


    @pyqtSlot()
    def open_median_filter_dialog(self):
        """Opens the dialog for applying Median Filter."""
        current_node = self.history_manager.get_current_node() # Pobierz aktualny węzeł
        if current_node is None: # Sprawdź, czy węzeł istnieje
            QMessageBox.warning(self, "No Image", "...")
            return
        if not MedianFilterDialog: 
            QMessageBox.critical(self, "Error", "MedianFilterDialog not available.")
            return

        if current_node.image_data is None: QMessageBox.critical(self, "Internal Error", "..."); return
        dialog_input_data = current_node.image_data.copy()

        logger.info(f"Opening Median Filter dialog based on node: {current_node.get_display_text()}")
        dialog = MedianFilterDialog(dialog_input_data, parent=self)
        result = dialog.exec()

        if result == QDialog.DialogCode.Accepted:
            processed_data = dialog.get_processed_data()
            params = dialog.get_parameters()
            was_roi_only = dialog.was_roi_applied_only()
            op_name = "Median Filter"

            if processed_data is not None:
                if np.allclose(processed_data, current_node.image_data): logger.info("Data not modified."); self.statusBar().showMessage("No changes applied.", 3000); return

                logger.info(f"Median Filter accepted. ROI Only: {was_roi_only}. Creating history node.")

                parent_data_type = current_node.data_type
                logger.info(f"Creating history node. Parent type: {parent_data_type}. ROI Only: {was_roi_only}.")

                
                final_roi_slice = None
                if was_roi_only:
                    final_roi_slice = dialog.get_final_roi_slice()
                
                new_node = HistoryNode(
                    parent_id=self.history_manager.current_node_id,
                    operation_name=op_name,
                    parameters=params,
                    image_data=processed_data,
                    data_type=parent_data_type, 
                    source_roi_slice=final_roi_slice
                )
                self.history_manager.add_node(new_node)
                self.history_manager.set_current_node_by_id(new_node.node_id)
                display_name = new_node.get_display_text() 
                self.statusBar().showMessage(f"{display_name} applied.", 3000)
            else: logger.warning("Dialog accepted, but no processed data returned.")
        else: logger.info("Median Filter dialog cancelled."); self.statusBar().showMessage("Median Filter cancelled.", 3000)


    @pyqtSlot()
    def open_plane_leveling_dialog(self):
        """Opens the dialog for applying Plane Leveling."""
        current_node = self.history_manager.get_current_node() # Pobierz aktualny węzeł
        if current_node is None: # Sprawdź, czy węzeł istnieje
            QMessageBox.warning(self, "No Image", "...")
            return
        if not PlaneLevelingDialog: 
            QMessageBox.critical(self, "Error", "PlaneLevelingDialog not available.")
            return

        if current_node.image_data is None: 
            QMessageBox.critical(self, "Internal Error", "...")
            return
        dialog_input_data = current_node.image_data.copy()

        logger.info(f"Opening Plane Leveling dialog based on node: {current_node.get_display_text()}")
        dialog = PlaneLevelingDialog(dialog_input_data, parent=self)
        result = dialog.exec()

        if result == QDialog.DialogCode.Accepted:
            processed_data = dialog.get_processed_data()
            params = dialog.get_parameters()
            was_roi_only = dialog.was_roi_applied_only() 
            op_name = "Plane Leveling"

            if processed_data is not None:
                logger.info(f"Plane Leveling accepted. ROI Only: {was_roi_only}. Creating history node.")

                parent_data_type = current_node.data_type
                logger.info(f"Creating history node. Parent type: {parent_data_type}. ROI Only: {was_roi_only}.")

                
                final_roi_slice = None
                if was_roi_only:
                    final_roi_slice = dialog.get_final_roi_slice()
                
                new_node = HistoryNode(
                    parent_id=self.history_manager.current_node_id,
                    operation_name=op_name,
                    parameters=params,
                    image_data=processed_data,
                    data_type=parent_data_type, 
                    source_roi_slice=final_roi_slice
                )
                self.history_manager.add_node(new_node)
                self.history_manager.set_current_node_by_id(new_node.node_id)
                display_name = new_node.get_display_text() 
                self.statusBar().showMessage(f"{display_name} applied.", 3000)
            else: logger.warning("Dialog accepted, but no processed data returned.")
        else: logger.info("Plane Leveling dialog cancelled."); self.statusBar().showMessage("Plane Leveling cancelled.", 3000)


    @pyqtSlot(QListWidgetItem, QListWidgetItem)
    def on_history_selection_changed(self, current_item: QListWidgetItem, previous_item: QListWidgetItem):
        """Slot called when the selection in the history list changes."""
        if current_item:
            node_id = current_item.data(Qt.ItemDataRole.UserRole)
            if self.history_manager and node_id != self.history_manager.current_node_id: # Dodatkowy warunek, aby uniknąć zbędnych wywołań
                logger.info(f"MainWindow: History item selected: {current_item.text()}, delegating to HistoryManager.")
                self.history_manager.set_current_node_by_id(node_id, emit_signal=True)

    @pyqtSlot()
    def show_about_dialog(self):
        """Displays the 'About' information message box."""
        logger.debug("About dialog triggered.")
        about_text = """
        <H2>Lattice Fourier Analyzer (LFA)</H2>
        <p>Version: 0.1.0 (Development)</p>
        <p>A tool for analyzing Scanning Tunneling Microscopy (STM) images,
        focusing on lattice determination using Fourier methods.</p>
        <p>(Further development ongoing)</p>
        <p><b>Note:</b> This is developmental software.</p>
        """
        QMessageBox.about(self, "About LFA", about_text)

    @pyqtSlot()
    def open_gaussian_blur_dialog(self):
        current_node = self.history_manager.get_current_node() # Pobierz aktualny węzeł
        if current_node is None: # Sprawdź, czy węzeł istnieje
            QMessageBox.warning(self, "No Image", "Please load an image first.")
            return
        if not GaussianBlurDialog:
            QMessageBox.critical(self, "Error", "Gaussian Blur functionality is not available.")
            return
        print(f"Gaussian Blur: Current node ID: {current_node.node_id}")
        if current_node.image_data is None:
            QMessageBox.critical(self, "Internal Error", "No image data available.")
            return
        dialog_input_data = current_node.image_data.copy()

        logger.info(f"Opening Gaussian Blur dialog based on node: {current_node.get_display_text()}")
        dialog = GaussianBlurDialog(dialog_input_data, parent=self)
        result = dialog.exec()

        if result == QDialog.DialogCode.Accepted:
            processed_data = dialog.get_processed_data()
            params = dialog.get_parameters()
            was_roi_only = dialog.was_roi_applied_only()
            op_name = "Gaussian Blur"

            if processed_data is not None:
                if np.allclose(processed_data, current_node.image_data):
                    logger.info("Data was not modified. No history node created.")
                    self.statusBar().showMessage("No changes applied.", 3000)
                    return

                logger.info(f"Dialog accepted. Final apply was ROI: {was_roi_only}. Creating history node.")

                parent_data_type = current_node.data_type
                logger.info(f"Creating history node. Parent type: {parent_data_type}. ROI Only: {was_roi_only}.")

                
                final_roi_slice = None
                if was_roi_only:
                    final_roi_slice = dialog.get_final_roi_slice()
                
                new_node = HistoryNode(
                    parent_id=self.history_manager.current_node_id,
                    operation_name=op_name,
                    parameters=params,
                    image_data=processed_data,
                    data_type=parent_data_type, 
                    source_roi_slice=final_roi_slice
                )
                self.history_manager.add_node(new_node)
                self.history_manager.set_current_node_by_id(new_node.node_id)
                display_name = new_node.get_display_text()
                self.statusBar().showMessage(f"{display_name} applied.", 3000)
            else:
                logger.warning("Dialog accepted, but processed data is None.")
        else:
            logger.info("Gaussian Blur dialog cancelled.")
            self.statusBar().showMessage("Gaussian Blur cancelled.", 3000)

    def display_image_data(self):
        """
        Displays the image data from the currently selected history node
        in the main window's ImageView.

        Handles:
        - Displaying STM or FFT data with appropriate orientation and scaling.
        - Showing/hiding the ideal lattice overlay on FFT images based on
          user selection and checkbox state (obtained from FFTAnalysisPanel).
        - Connecting/disconnecting mouse click handlers for FFT view.
        - Triggering an update of selected spot markers.
        """
        # Ensure the main image view widget exists
        if not hasattr(self, 'visualization_manager') or self.visualization_manager is None:
            logger.error("MainWindow: VisualizationManager not available for displaying data.")
            # Można spróbować wyczyścić widok bezpośrednio, jeśli image_view istnieje
            if hasattr(self, 'image_view') and self.image_view: self.image_view.clear()
            return
        
        if not hasattr(self, 'history_manager') or self.history_manager is None:
            logger.error("MainWindow: HistoryManager not available.")
            if hasattr(self, 'image_view') and self.image_view: self.image_view.clear()
            return
        
        current_node = self.history_manager.get_current_node()

        # === Zbieranie informacji dla VisualizationManager.update_view ===
        show_ideal_lattice = False
        selected_substrate = "None"
        panel_custom_text = "" # Domyślna wartość, jeśli panel nie istnieje

        if hasattr(self, 'fft_analysis_panel_widget') and self.fft_analysis_panel_widget is not None:
            show_ideal_lattice = self.fft_analysis_panel_widget.is_show_ideal_lattice_checked()
            selected_substrate = self.fft_analysis_panel_widget.get_current_substrate()
            panel_custom_text = self.fft_analysis_panel_widget.custom_option_text # Pobierz stałą
        
        custom_def = self.custom_lattice_info # Atrybut MainWindow

        show_sub_markers = False
        show_ads_markers = False
        if hasattr(self, 'fft_analysis_panel_widget') and self.fft_analysis_panel_widget is not None:
            show_sub_markers = self.fft_analysis_panel_widget.is_substrate_spots_visible()
            show_ads_markers = self.fft_analysis_panel_widget.is_adsorbate_spots_visible()

        
        # === DELEGOWANIE DO VISUALIZATION MANAGER ===
        self.visualization_manager.update_view(
            current_node,
            show_ideal_lattice,
            selected_substrate, # Przekaż nazwę lub specjalny tekst
            custom_def,         # Przekaż definicję custom, jeśli istnieje
            panel_custom_text,  # Przekaż tekst opcji custom
            self.substrate_spots, # Dane o pikach zarządzane przez MainWindow (lub przyszły kontroler)
            show_sub_markers,
            self.adsorbate_spot_sets, # Dane o pikach zarządzane przez MainWindow (lub przyszły kontroler)
            show_ads_markers
        )
        # =======================================

        if hasattr(self, '_update_action_states'):
             self._update_action_states()
    
    @pyqtSlot(QPointF) # Sygnał z VisualizationManager przekazuje QPointF
    def _on_fft_view_clicked_from_visualizer(self, mapped_data_pos: QPointF):
        """
        Handles processed mouse clicks on the FFT image, received from VisualizationManager.
        mapped_data_pos.x() is kx (original data columns), mapped_data_pos.y() is ky (original data rows).
        """
        logger.debug(f"MainWindow: Received FFT click from Visualizer at data coords (kx, ky): ({mapped_data_pos.x():.2f}, {mapped_data_pos.y():.2f})")

        current_node = self.history_manager.get_current_node()
        if not (current_node and current_node.data_type == "FFT" and current_node.image_data is not None):
            logger.warning("_on_fft_view_clicked_from_visualizer: No valid FFT data node active.")
            return

        kx_from_signal = mapped_data_pos.x()
        ky_from_signal = mapped_data_pos.y()

        kx_int = int(round(kx_from_signal))
        ky_int = int(round(ky_from_signal))

        # Walidacja granic (opcjonalna, jeśli mapToData już to robi, ale dla pewności)
        original_fft_data = current_node.image_data
        fft_data_rows_ky, fft_data_cols_kx = original_fft_data.shape
        if not (0 <= ky_int < fft_data_rows_ky and 0 <= kx_int < fft_data_cols_kx): # pragma: no cover
            logger.debug(f"MainWindow: Click data coords (kx_int={kx_int}, ky_int={ky_int}) "
                         f"is outside original FFT data bounds ({fft_data_cols_kx}, {fft_data_rows_ky}). Ignoring.")
            return
            
        center_yx_for_refinement = (ky_int, kx_int) # (wiersz, kolumna) dla funkcji uściślających

        refined_kx = kx_int # Domyślnie, jeśli nie ma uściślania
        refined_ky = ky_int

        current_refinement_method = self.spot_refinement_method
        current_refinement_radius = self.refinement_roi_size // 2 # Zakładamy, że refinement_roi_size to średnica

        logger.debug(f"Refinement: Method='{current_refinement_method}', Radius for func={current_refinement_radius}, Click (ky,kx)=({ky_int},{kx_int})")

        if current_refinement_method == "Max Pixel":
            if PEAK_FITTING_AVAILABLE:
                refined_ky_temp, refined_kx_temp = find_max_pixel_in_roi(
                    original_fft_data, center_yx_for_refinement, current_refinement_radius
                )
                refined_kx, refined_ky = int(refined_kx_temp), int(refined_ky_temp)
                logger.info(f"Max Pixel refined: (orig_kx={kx_int}, orig_ky={ky_int}) -> (ref_kx={refined_kx}, ref_ky={refined_ky})")
            else: # pragma: no cover
                logger.warning("Peak fitting (Max Pixel) backend not available. Using raw click.")
        elif current_refinement_method == "2D Gaussian Fit":
            if PEAK_FITTING_AVAILABLE: # SCIPY_AVAILABLE jest z peak_fitting
                fit_result = fit_2d_gaussian_in_roi(
                    original_fft_data, center_yx_for_refinement, current_refinement_radius
                )
                if fit_result:
                    refined_ky_float, refined_kx_float = fit_result
                    refined_kx, refined_ky = int(round(refined_kx_float)), int(round(refined_ky_float))
                    logger.info(f"2D Gaussian Fit refined: (orig_kx={kx_int}, orig_ky={ky_int}) -> (ref_kx={refined_kx:.2f}, ref_ky={refined_ky:.2f})")
                else: # pragma: no cover
                    logger.warning("2D Gaussian Fit failed. Using raw click position.")
            else: # pragma: no cover
                logger.warning("Peak fitting (Gaussian) backend. Using raw click.")
        
        final_point_coords_kx_ky = (refined_kx, refined_ky) # (kx, ky)

        if self.spot_selection_mode == "Substrate":
            MAX_SUBSTRATE_SPOTS = 8 # Można przenieść do stałych
            if len(self.substrate_spots) < MAX_SUBSTRATE_SPOTS:
                if final_point_coords_kx_ky not in self.substrate_spots:
                    self.substrate_spots.append(final_point_coords_kx_ky)
                    logger.debug(f"Added to substrate_spots: {final_point_coords_kx_ky}. Count: {len(self.substrate_spots)}")
                else: # pragma: no cover
                    logger.debug(f"Point {final_point_coords_kx_ky} already in substrate_spots.")
            else: # pragma: no cover
                QMessageBox.information(self, "Limit Reached", f"Maximum number of substrate spots ({MAX_SUBSTRATE_SPOTS}) selected.")
        elif self.spot_selection_mode == "Adsorbate":
            if 0 <= self.current_adsorbate_set_index < len(self.adsorbate_spot_sets):
                current_set_list = self.adsorbate_spot_sets[self.current_adsorbate_set_index]
                if final_point_coords_kx_ky not in current_set_list:
                    current_set_list.append(final_point_coords_kx_ky)
                    logger.debug(f"Added to adsorbate_spot_sets[{self.current_adsorbate_set_index}]: {final_point_coords_kx_ky}. Set count: {len(current_set_list)}")
                else: # pragma: no cover
                    logger.debug(f"Point {final_point_coords_kx_ky} already in current adsorbate set.")
            else: # pragma: no cover
                 logger.error(f"Invalid current_adsorbate_set_index: {self.current_adsorbate_set_index}")
        
        if hasattr(self, '_update_selected_spots_display'): self._update_selected_spots_display()
        self.request_spot_markers_update()
        if hasattr(self, '_update_action_states'): self._update_action_states()

    @pyqtSlot()
    def _on_clear_substrate_spots_clicked(self):
        logger.info("MainWindow: Clearing substrate spots.")
        self.substrate_spots = []
        self.request_spot_markers_update() # Zaktualizuj markery
        if hasattr(self, '_update_selected_spots_display'): self._update_selected_spots_display()
        if hasattr(self, '_update_action_states'): self._update_action_states()

    @pyqtSlot()
    def _on_clear_last_adsorbate_point_clicked(self):
        logger.debug("Attempting to clear last adsorbate point.")
        if self.spot_selection_mode == "Adsorbate":
            if 0 <= self.current_adsorbate_set_index < len(self.adsorbate_spot_sets):
                current_set_list = self.adsorbate_spot_sets[self.current_adsorbate_set_index]
                if current_set_list:
                    removed_point = current_set_list.pop()
                    logger.info(f"Removed last adsorbate point: {removed_point} from set {self.current_adsorbate_set_index + 1}")
                    self._update_selected_spots_display()
                    self.request_spot_markers_update()
                    self._update_action_states() # Ważne, aby zaktualizować enabled przycisku
                else:
                    logger.debug("No points in current adsorbate set to remove.")
            else:
                logger.warning(f"Invalid adsorbate set index: {self.current_adsorbate_set_index}")
        else:
            logger.debug("Not in Adsorbate selection mode, cannot clear last adsorbate point.")


    @pyqtSlot()
    def _on_reselect_substrate_spots_clicked(self):
        """Clears current substrate spots to allow re-selection."""
        if self.spot_selection_mode == "Substrate":
            logger.info("Reselecting substrate spots. Clearing current substrate spots.")
            self.substrate_spots = []
            self._update_selected_spots_display()
            self.request_spot_markers_update()
            self._update_action_states() # Zaktualizuj status np. przycisku "Clear"
        else:
            logger.debug("Not in Substrate selection mode. 'Reselect Substrate' ignored.")

    def closeEvent(self, event):
        """Handle the event when the user tries to close the window."""
        logger.info("Close event triggered. Exiting application.")
        event.accept()