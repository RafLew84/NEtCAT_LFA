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
from .widgets.metadata_widget import MetadataWidget
from ..logic.history_manager import HistoryManager
from .panels.fft_analysis_panel import FFTAnalysisPanel
from .visualization_manager import VisualizationManager
from ..logic.app_controller import AppController, LATTICE_ANALYSIS_FUNCTIONS_AVAILABLE
from ..core.history import HistoryNode
from .ui_setup.menu_action_manager import MenuActionManager
from .ui_setup.dock_panel_manager import DockPanelManager
from ..gui.dialogs.substrate_spot_dialog import PREDEFINED_SUBSTRATE_NONE, PREDEFINED_SUBSTRATE_CUSTOM, LATTICE_TYPE_HEXAGONAL, LATTICE_TYPE_SQUARE
from ..logic.app_controller import ADSORBATE_LATTICE_TYPE_UNKNOWN

try:
    from lfa.gui.dialogs.preprocessing_dialogs import (GaussianBlurDialog, PlaneLevelingDialog, 
    MedianFilterDialog, NLMeansDialog, BM3DDialog, GaussianSharpeningDialog)
    from lfa.gui.dialogs.fft_dialog import FFTDialog
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

try:
    from .dialogs.substrate_spot_dialog import SubstrateSpotSelectionDialog
    from .dialogs.adsorbate_spot_dialog import AdsorbateSpotSelectionDialog
    SPOT_SELECTION_DIALOGS_AVAILABLE = True
except ImportError: # pragma: no cover
    logging.warning("Could not import spot selection dialogs. Spot selection from menu will not work.")
    SubstrateSpotSelectionDialog = None
    AdsorbateSpotSelectionDialog = None
    SPOT_SELECTION_DIALOGS_AVAILABLE = False

logger = logging.getLogger(__name__)


try:
    from ..analysis.lattice import get_reciprocal_points, KNOWN_LATTICES
    from lfa.gui.dialogs.custom_lattice_dialog import CustomLatticeDialog
    LATTICE_ANALYSIS_AVAILABLE = True
except ImportError:
    logging.error("Could not import lattice analysis functions.")
    KNOWN_LATTICES = {"None": {}} # Placeholder
    def get_reciprocal_points(name, max_hk=2): return None
    CustomLatticeDialog = None
    LATTICE_ANALYSIS_AVAILABLE = False

try:
    from .dialogs.real_space_visualizer_dialog import RealSpaceFFTVisualizerDialog
    REAL_SPACE_VIS_DIALOG_AVAILABLE = True
except ImportError: # pragma: no cover
    RealSpaceFFTVisualizerDialog = None
    REAL_SPACE_VIS_DIALOG_AVAILABLE = False
    logging.warning("Could not import RealSpaceFFTVisualizerDialog.")

class MainWindow(QMainWindow):
    """
    The main application window inheriting from QMainWindow.
    Provides menu bar, status bar, and central widget area.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Lattice Fourier Analyzer (LFA)")
        self.resize(1250, 800)

        self._setup_main_layout() # Tworzy m.in. self.history_list_widget
        self.metadata_widget = MetadataWidget(self) # Tworzymy widgety zawartości
        self.fft_analysis_panel_widget = FFTAnalysisPanel(self)

        # 2. Managery logiki
        self.history_manager = HistoryManager(self.history_list_widget, self)
        self.app_controller = AppController(history_manager=self.history_manager)
        
        self._init_core_attributes()

        self.menu_manager = MenuActionManager(self) 
        self.real_space_visualizer_dialog_instance: Optional[RealSpaceFFTVisualizerDialog] = None

        self.dock_manager = DockPanelManager(
            main_window=self,
            history_list_widget=self.history_list_widget,
            metadata_widget=self.metadata_widget,
            fft_analysis_panel_widget=self.fft_analysis_panel_widget
        )

        self._create_status_bar()

        if pg and self.image_view and self.history_manager and VisualizationManager:
            self.visualization_manager = VisualizationManager(
                image_view=self.image_view,
                history_manager=self.history_manager,
            )
            logger.info("VisualizationManager created and initialized.")
        else: # pragma: no cover
            self.visualization_manager = None
            logger.error("Could not create VisualizationManager due to missing dependencies.")

        self._connect_signals()
        self._update_action_states()
        logger.info("Main window initialized.")

    def _init_core_attributes(self):
        """Initializes non-widget core attributes of the MainWindow."""

        self._fft_mouse_click_connection = None

    def _setup_main_layout(self):
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.history_list_widget = QListWidget()

        image_view_container = QWidget()
        image_view_layout = QVBoxLayout(image_view_container)
        image_view_layout.setContentsMargins(0, 0, 0, 0)

        if pg:
            pg.setConfigOption('background', 'w')
            pg.setConfigOption('foreground', 'k')
            self.image_view = pg.ImageView(self)
            image_view_layout.addWidget(self.image_view)
        else:
            self.image_view = None
            logger.error("Cannot create ImageView because PyQtGraph is not available.")

        self.splitter.addWidget(image_view_container)
        self.splitter.setSizes([250, 950])
        main_layout.addWidget(self.splitter)
    
    def _create_status_bar(self):
        """Creates the status bar."""
        self.statusBar().showMessage("Ready - Load an image using File -> Open")

    def _connect_signals(self):
        if self.history_manager:
            self.history_list_widget.currentItemChanged.connect(self.on_history_selection_changed)
            self.history_manager.current_node_changed.connect(self._on_current_history_node_changed)

        if self.app_controller:
            self.app_controller.file_loaded_successfully.connect(self._on_file_loaded_successfully)
            self.app_controller.file_loading_failed.connect(self._on_file_loading_failed)

            self.app_controller.spot_lists_updated.connect(self._on_spot_lists_or_params_changed)
            self.app_controller.spot_selection_parameters_changed.connect(self._on_spot_lists_or_params_changed)
            self.app_controller.adsorbate_sets_structure_changed.connect(self._on_adsorbate_sets_structure_changed)

            self.app_controller.substrate_definition_changed.connect(self._on_substrate_definition_changed)

            self.app_controller.adsorbate_set_updated.connect(self._on_adsorbate_set_updated)

            self.app_controller.adsorbate_expected_type_updated.connect(self._on_adsorbate_expected_type_updated)

            self.app_controller.substrate_transform_results_updated.connect(self._on_substrate_transform_results_updated)
            self.app_controller.substrate_definition_changed.connect(self._on_substrate_definition_changed) # Już powinno być
            self.app_controller.spot_lists_updated.connect(self._on_spot_lists_or_params_changed) # Do ogólnych aktualizacji

            self.app_controller.substrate_real_space_params_updated.connect(self._on_substrate_real_space_params_updated)
            self.app_controller.adsorbate_real_space_params_updated.connect(self._on_adsorbate_real_space_params_updated)

        if hasattr(self, 'fft_analysis_panel_widget'):
            self.fft_analysis_panel_widget.substrate_changed.connect(self._handle_substrate_changed)
            self.fft_analysis_panel_widget.custom_lattice_define_requested.connect(self._handle_custom_lattice_request)
            self.fft_analysis_panel_widget.show_ideal_lattice_changed.connect(self._handle_show_ideal_lattice_changed)
            self.fft_analysis_panel_widget.spot_selection_mode_changed.connect(self._handle_spot_selection_mode_changed_from_panel)
            self.fft_analysis_panel_widget.current_adsorbate_set_changed.connect(self._handle_current_adsorbate_set_changed_from_panel)
            self.fft_analysis_panel_widget.add_new_adsorbate_set_requested.connect(self._handle_add_new_adsorbate_set_request)
            self.fft_analysis_panel_widget.reselect_current_adsorbate_set_triggered.connect(self._on_reselect_adsorbate_set_clicked)
            self.fft_analysis_panel_widget.clear_all_adsorbate_sets_triggered.connect(self._on_clear_all_adsorbate_sets_clicked)
            # self.fft_analysis_panel_widget.clear_last_adsorbate_point_triggered.connect(self._on_clear_last_adsorbate_point_clicked)
            self.fft_analysis_panel_widget.substrate_spots_visibility_changed.connect(self._handle_substrate_spots_visibility_changed)
            self.fft_analysis_panel_widget.adsorbate_spots_visibility_changed.connect(self._handle_adsorbate_spots_visibility_changed)
            # self.fft_analysis_panel_widget.fitted_substrate_spots_visibility_changed.connect(self._handle_fitted_substrate_spots_visibility_changed)
            self.fft_analysis_panel_widget.select_edit_substrate_spots_requested.connect(self.open_substrate_spot_selection_dialog)
            self.fft_analysis_panel_widget.select_edit_adsorbate_spots_requested.connect(self.open_adsorbate_spot_selection_dialog)
            self.fft_analysis_panel_widget.calculate_substrate_real_space_params_requested.connect(self._on_calculate_substrate_rs_params_button_clicked)
            self.fft_analysis_panel_widget.calculate_adsorbate_real_space_params_requested.connect(self._on_calculate_adsorbate_rs_params_button_clicked)
            self.fft_analysis_panel_widget.expected_adsorbate_lattice_type_changed.connect(self._handle_expected_adsorbate_type_changed_from_panel)

        if hasattr(self, 'visualization_manager') and self.visualization_manager:
            self.visualization_manager.fft_view_clicked.connect(self._on_fft_view_clicked_from_visualizer)

    def _clear_all_spot_markers_from_view(self, view_box: Optional[pg.ViewBox]):
        """Helper to remove all known spot markers from the view."""
        logger.debug("Cleared all user-selected spot markers from view.")

    def _update_selected_spots_display(self):
        if not hasattr(self, 'fft_analysis_panel_widget') or not hasattr(self.fft_analysis_panel_widget, 'selected_spots_display'):
            return

        text_output = []
        current_selection_status = ""
        spot_selection_mode = self.app_controller.spot_selection_mode
        current_adsorbate_set_idx = self.app_controller.current_adsorbate_set_index
        substrate_spots = self.app_controller.substrate_spots
        adsorbate_spot_sets = self.app_controller.adsorbate_spot_sets

        if spot_selection_mode == "Substrate":
            current_selection_status = "Selecting: Substrate Spots"
            text_output.append("Substrate Spots:")
            if substrate_spots:
                for i, (kx, ky) in enumerate(substrate_spots):
                    text_output.append(f"  S{i+1}: (kx={kx}, ky={ky})")
            else:
                text_output.append("  None selected.")
        elif spot_selection_mode == "Adsorbate":
            set_name = self.fft_analysis_panel_widget.adsorbate_set_combo.itemText(current_adsorbate_set_idx) if current_adsorbate_set_idx < self.fft_analysis_panel_widget.adsorbate_set_combo.count() else f"Set {current_adsorbate_set_idx + 1}"
            current_selection_status = f"Selecting: Adsorbate {set_name}"
            text_output.append(f"Adsorbate {set_name}:")

            if 0 <= current_adsorbate_set_idx < len(adsorbate_spot_sets):
                current_points_to_display = adsorbate_spot_sets[current_adsorbate_set_idx]
                if current_points_to_display:
                    for i, (kx, ky) in enumerate(current_points_to_display):
                        text_output.append(f"  A{i+1}: (kx={kx}, ky={ky})")
                else:
                    text_output.append("  No spots selected for this set.")
            else:
                text_output.append("  Invalid adsorbate set selected.")

        if hasattr(self.fft_analysis_panel_widget, 'current_selection_label') and self.fft_analysis_panel_widget.current_selection_label:
            self.fft_analysis_panel_widget.current_selection_label.setText(current_selection_status)
        self.fft_analysis_panel_widget.selected_spots_display.setPlainText("\n".join(text_output))

    def _helper_open_processing_dialog(self, DialogClass, op_name_in_controller: str, dialog_specific_checks=None):
        """Pomocnicza metoda do otwierania dialogów przetwarzania i obsługi wyników."""
        current_node_info = self.app_controller.get_current_node_info_for_dialogs()
        if not current_node_info:
            QMessageBox.warning(self, "No Image", "No data loaded or selected in history to process.")
            return

        parent_id, parent_data_type, image_data_copy = current_node_info

        if not DialogClass: # pragma: no cover
            QMessageBox.critical(self, "Error", f"{DialogClass.__name__ if DialogClass else 'Dialog'} is not available.")
            return
            
        if dialog_specific_checks: # pragma: no cover
             if not dialog_specific_checks(): return


        dialog = DialogClass(image_data_copy, parent=self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            processed_data = dialog.get_processed_data()
            params = dialog.get_parameters() # Dialog powinien zwracać słownik parametrów
            was_roi_only = dialog.was_roi_applied_only()

            if processed_data is not None:
                controller_method_to_call = getattr(self.app_controller, op_name_in_controller, None)
                if controller_method_to_call and callable(controller_method_to_call):
                    controller_method_to_call(
                        parent_node_id=parent_id,
                        parent_data_type=parent_data_type,
                        processed_data=processed_data,
                        params=params,
                        source_roi_slice=dialog.get_final_roi_slice() if was_roi_only else None
                    )
                    op_display_name = dialog.operation_name if hasattr(dialog, 'operation_name') else op_name_in_controller.replace("apply_", "").replace("_operation","").title()
                    self.statusBar().showMessage(f"{op_display_name} applied.", 3000)
                else: # pragma: no cover
                    logger.error(f"Method {op_name_in_controller} not found in AppController!")
                    self.statusBar().showMessage(f"Error applying {op_name_in_controller}.", 3000)
            else: # pragma: no cover
                logger.warning(f"{DialogClass.__name__} accepted, but no processed data returned.")
                self.statusBar().showMessage("Operation cancelled or no changes made.", 3000)
        else:
            op_display_name = dialog.operation_name if hasattr(dialog, 'operation_name') else op_name_in_controller.replace("apply_", "").replace("_operation","").title()
            logger.info(f"{op_display_name} dialog cancelled.") # pragma: no cover
            self.statusBar().showMessage(f"{op_display_name} cancelled.", 3000) # pragma: no cover
    
    def _update_action_states(self):
        """
        Updates the enabled/disabled state of various actions and UI controls
        based on the current application state retrieved from AppController
        and HistoryManager.
        """
        logger.debug("MainWindow: Updating action states...")

        # --- Domyślne stany ---
        has_active_node = False
        is_stm_data_active = False
        is_fft_data_active = False
        
        # Stany dla przycisków obliczeń parametrów rzeczywistych
        can_calculate_substrate_rs = False
        can_calculate_adsorbate_rs = False
        
        # Stany dla przycisków w panelu FFT związanymi z edycją spotów
        can_edit_substrate_spots = False
        can_edit_adsorbate_spots = False
        
        # Stany dla przycisków czyszczenia w panelu FFT
        can_clear_substrate_from_panel = False # Czy FFTAnalysisPanel ma zarządzać tym? Raczej dialogi.
        can_clear_last_adsorbate_from_panel = False
        can_clear_current_adsorbate_set_from_panel = False
        can_clear_all_adsorbate_sets_from_panel = False


        # --- Pobierz bieżący stan ---
        current_hist_node: Optional[HistoryNode] = None
        if self.history_manager:
            current_hist_node = self.history_manager.get_current_node()

        if current_hist_node:
            has_active_node = True
            if current_hist_node.data_type == "STM":
                is_stm_data_active = True
            elif current_hist_node.data_type == "FFT":
                is_fft_data_active = True

        can_visualize_real_space = False
        if self.app_controller and REAL_SPACE_VIS_DIALOG_AVAILABLE:
            # Warunek: muszą być obliczone parametry rzeczywiste substratu LUB adsorbatu
            # ORAZ musi być aktywny obraz FFT
            current_hist_node = self.history_manager.get_current_node()
            if current_hist_node and current_hist_node.data_type == "FFT":
                if self.app_controller.substrate_real_space_results or \
                   (self.app_controller.current_adsorbate_set_index in self.app_controller.adsorbate_real_space_results):
                    can_visualize_real_space = True

        # --- Logika dla Akcji Menu ---
        # Preprocessing: dostępne, jeśli jest jakikolwiek aktywny węzeł (STM lub FFT)
        preprocessing_actions_enabled = has_active_node
        if hasattr(self, 'gaussian_blur_action'): self.gaussian_blur_action.setEnabled(preprocessing_actions_enabled and DIALOG_CLASSES_EXIST)
        if hasattr(self, 'gaussian_sharpen_action'): self.gaussian_sharpen_action.setEnabled(preprocessing_actions_enabled and DIALOG_CLASSES_EXIST)
        if hasattr(self, 'plane_level_action'): self.plane_level_action.setEnabled(preprocessing_actions_enabled and DIALOG_CLASSES_EXIST)
        if hasattr(self, 'median_filter_action'): self.median_filter_action.setEnabled(preprocessing_actions_enabled and DIALOG_CLASSES_EXIST)
        if hasattr(self, 'nlmeans_action'): self.nlmeans_action.setEnabled(preprocessing_actions_enabled and DIALOG_CLASSES_EXIST)
        if hasattr(self, 'bm3d_action'): self.bm3d_action.setEnabled(preprocessing_actions_enabled and DIALOG_CLASSES_EXIST)
        # Analysis:
        # Obliczanie FFT: tylko jeśli aktywny węzeł to STM
        if hasattr(self, 'fft_action'): self.fft_action.setEnabled(is_stm_data_active and DIALOG_CLASSES_EXIST)
        
        # Wybór spotów substratu/adsorbatu: tylko jeśli aktywny węzeł to FFT i dialogi istnieją
        can_open_spot_selection_dialogs = is_fft_data_active and SPOT_SELECTION_DIALOGS_AVAILABLE
        if hasattr(self, 'select_substrate_spots_action'):
            self.select_substrate_spots_action.setEnabled(can_open_spot_selection_dialogs)
        if hasattr(self, 'select_adsorbate_spots_action'):
            self.select_adsorbate_spots_action.setEnabled(can_open_spot_selection_dialogs)

        # --- Logika dla FFTAnalysisPanel ---
        if hasattr(self, 'fft_analysis_dock'): # Dok panelu FFT
            self.fft_analysis_dock.setVisible(is_fft_data_active)

        if hasattr(self, 'visualize_real_space_action'): # Akcja z MenuActionManager
            self.visualize_real_space_action.setEnabled(can_visualize_real_space)

        if hasattr(self, 'fft_analysis_panel_widget') and self.fft_analysis_panel_widget:
            panel = self.fft_analysis_panel_widget
            
            if is_fft_data_active and self.app_controller:
                # Przyciski "Edit/Select..."
                panel.set_edit_substrate_spots_button_enabled(True) # Zawsze włączone, jeśli jest FFT
                panel.set_edit_adsorbate_spots_button_enabled(True) # Zawsze włączone, jeśli jest FFT

                # Przyciski czyszczenia dla adsorbatu (zależne od stanu w AppController)
                current_ads_idx = self.app_controller.current_adsorbate_set_index
                ads_sets = self.app_controller.adsorbate_spot_sets
                
                can_clear_last_adsorbate_from_panel = (
                    0 <= current_ads_idx < len(ads_sets) and 
                    bool(ads_sets[current_ads_idx])
                )
                # panel.set_clear_last_adsorbate_point_button_enabled(can_clear_last_adsorbate_from_panel)

                can_clear_current_adsorbate_set_from_panel = (
                    0 <= current_ads_idx < len(ads_sets) and
                    bool(ads_sets[current_ads_idx]) # Czy są jakieś spoty w bieżącym zestawie
                )
                panel.set_reselect_adsorbate_set_button_enabled(can_clear_current_adsorbate_set_from_panel) # Ten przycisk czyści bieżący set

                can_clear_all_adsorbate_sets_from_panel = any(s for s in ads_sets if s) # Czy jakikolwiek zestaw ma spoty
                panel.set_clear_all_adsorbate_sets_button_enabled(can_clear_all_adsorbate_sets_from_panel)

                # Przyciski obliczeń parametrów rzeczywistych
                ac = self.app_controller
                # Substrat
                if (ac.substrate_lattice_type and ac.substrate_a_surf and ac.substrate_a_surf > 0 and
                    ac.reference_ideal_substrate_spots_px and ac.current_fft_data_shape and
                    LATTICE_ANALYSIS_FUNCTIONS_AVAILABLE):
                    
                    expected_sub_spot_count = 0
                    if ac.substrate_lattice_type == LATTICE_TYPE_HEXAGONAL: expected_sub_spot_count = 6
                    elif ac.substrate_lattice_type == LATTICE_TYPE_SQUARE: expected_sub_spot_count = 4
                    
                    if len(ac.reference_ideal_substrate_spots_px) == expected_sub_spot_count and expected_sub_spot_count > 0:
                        can_calculate_substrate_rs = True
                
                # Adsorbat
                if (0 <= current_ads_idx < len(ac.corrected_adsorbate_spot_sets) and
                    ac.corrected_adsorbate_spot_sets[current_ads_idx] and # Muszą być skorygowane piki
                    ac.current_fft_data_shape and LATTICE_ANALYSIS_FUNCTIONS_AVAILABLE):
                    
                    # Sprawdzenie, czy można pobrać Lx, Ly (wymagane do konwersji g* na nm^-1)
                    if current_hist_node:
                        root_node = self.history_manager.get_root_node_for_node(current_hist_node.node_id)
                        if root_node and root_node.parameters and \
                           root_node.parameters.get("size_nm_x") and root_node.parameters.get("size_nm_y"):
                            
                            # Ile pików jest wymaganych przez logikę select_adsorbate_basis_g_vectors_px?
                            # Załóżmy, że co najmniej 2 (jako wektory od centrum) lub 3 (z centrum).
                            # select_adsorbate_basis_g_vectors_px samo sprawdzi dokładniej.
                            if len(ac.corrected_adsorbate_spot_sets[current_ads_idx]) >= 2: # Minimalne wymaganie
                                can_calculate_adsorbate_rs = True
                
                panel.set_calculate_substrate_rs_button_enabled(can_calculate_substrate_rs)
                panel.set_calculate_adsorbate_rs_button_enabled(can_calculate_adsorbate_rs)

            else: # Jeśli nie ma danych FFT, wyłącz wszystko w panelu FFT
                panel.set_edit_substrate_spots_button_enabled(False)
                panel.set_edit_adsorbate_spots_button_enabled(False)
                # panel.set_clear_last_adsorbate_point_button_enabled(False)
                panel.set_reselect_adsorbate_set_button_enabled(False)
                panel.set_clear_all_adsorbate_sets_button_enabled(False)
                panel.set_calculate_substrate_rs_button_enabled(False)
                panel.set_calculate_adsorbate_rs_button_enabled(False)
                # Wyzeruj wyświetlane parametry, jeśli panel jest ukrywany
                panel.update_substrate_real_space_display(None)
                panel.update_adsorbate_real_space_display(None)
                panel.update_transform_results_display(None) # Dla informacji o transformacji substratu

        logger.debug(f"Action states updated. HasNode={has_active_node}, IsSTM={is_stm_data_active}, IsFFT={is_fft_data_active}, "
                     f"CanCalcSubRS={can_calculate_substrate_rs}, CanCalcAdsRS={can_calculate_adsorbate_rs}")

    @pyqtSlot()
    def open_real_space_fft_visualizer(self):
        logger.info("MainWindow: Opening Real Space/FFT Visualizer dialog...")
        if not REAL_SPACE_VIS_DIALOG_AVAILABLE: # pragma: no cover
            QMessageBox.critical(self, "Error", "RealSpaceFFTVisualizerDialog is not available.")
            return

        current_fft_node = self.history_manager.get_current_node()
        if not (current_fft_node and current_fft_node.data_type == "FFT"): # pragma: no cover
            QMessageBox.warning(self, "No FFT Data", "Please calculate FFT first to use the visualizer.")
            return

        # Można tworzyć nową instancję za każdym razem lub pokazywać istniejącą (jeśli nie jest modalna)
        # Dla modalnego dialogu, tworzymy nową:
        if self.real_space_visualizer_dialog_instance is not None: # pragma: no cover (jeśli zawsze tworzymy nowy)
            if self.real_space_visualizer_dialog_instance.isVisible():
                 logger.warning("RealSpaceFFTVisualizerDialog is already open.")
                 self.real_space_visualizer_dialog_instance.raise_()
                 self.real_space_visualizer_dialog_instance.activateWindow()
                 return
            else: # Został zamknięty, ale referencja może istnieć, lepiej ją usunąć
                 self.real_space_visualizer_dialog_instance.deleteLater()
                 self.real_space_visualizer_dialog_instance = None


        dialog = RealSpaceFFTVisualizerDialog(
            app_controller=self.app_controller,
            history_manager=self.history_manager,
            current_fft_node_id=current_fft_node.node_id, # Przekaż ID bieżącego węzła FFT
            parent=self
        )
        self.real_space_visualizer_dialog_instance = dialog # Zapisz referencję, jeśli chcemy zarządzać instancją
        dialog.exec() # Modalny dialog
        # Po zamknięciu modalnego dialogu, można wyczyścić referencję, jeśli nie chcemy go reużywać
        self.real_space_visualizer_dialog_instance = None 
        logger.info("RealSpaceFFTVisualizerDialog closed.")

    @pyqtSlot(int, str)
    def _handle_expected_adsorbate_type_changed_from_panel(self, set_index: int, selected_type: str):
        """Obsługuje zmianę spodziewanego typu sieci adsorbatu z FFTAnalysisPanel."""
        logger.debug(f"MainWindow: Expected adsorbate type for set {set_index} changed to '{selected_type}' via panel.")
        self.app_controller.set_expected_adsorbate_lattice_type(set_index, selected_type)
        self._update_action_states()


    @pyqtSlot(int, str)
    def _on_adsorbate_expected_type_updated(self, set_index: int, type_name: str):
        """
        Slot wywoływany, gdy spodziewany typ sieci dla zestawu adsorbatu zmieni się w AppController.
        Aktualizuje ComboBox w FFTAnalysisPanel, jeśli zmiana dotyczy bieżącego zestawu.
        """
        logger.debug(f"MainWindow: AppController reported expected adsorbate type for set {set_index} is now '{type_name}'.")
        if hasattr(self, 'fft_analysis_panel_widget') and self.fft_analysis_panel_widget:
            # Aktualizuj ComboBox tylko jeśli zmiana dotyczy aktualnie wybranego zestawu w panelu
            # (lub jeśli panel powinien zawsze odzwierciedlać stan z AppController)
            current_panel_set_index = self.fft_analysis_panel_widget.adsorbate_set_combo.currentIndex()
            # Sprawdź, czy to nie "<Add New Set...>"
            if current_panel_set_index < (self.fft_analysis_panel_widget.adsorbate_set_combo.count() -1) \
               and current_panel_set_index == set_index :
                self.fft_analysis_panel_widget.set_expected_adsorbate_type(type_name)
        # Aktualizacja stanu przycisków po zmianie typu może być potrzebna
        self._update_action_states()

    @pyqtSlot(dict)
    def _on_substrate_real_space_params_updated(self, params_dict: dict):
        logger.debug(f"MainWindow: Received substrate_real_space_params_updated: {params_dict}")
        if hasattr(self, 'fft_analysis_panel_widget') and self.fft_analysis_panel_widget:
            if "error" in params_dict:
                QMessageBox.warning(self, "Substrate Calculation Error", params_dict["error"])
                self.fft_analysis_panel_widget.update_substrate_real_space_display(None)
            else:
                self.fft_analysis_panel_widget.update_substrate_real_space_display(params_dict)
                self.statusBar().showMessage("Substrate real space parameters calculated.", 3000)

    @pyqtSlot(int, dict) # set_index, params_dict
    def _on_adsorbate_real_space_params_updated(self, set_index: int, params_dict: dict):
        logger.debug(f"MainWindow: Received adsorbate_real_space_params_updated for set {set_index}: {params_dict}")
        if hasattr(self, 'fft_analysis_panel_widget') and self.fft_analysis_panel_widget:
            # Aktualizuj UI tylko jeśli dotyczy to bieżącego zestawu wyświetlanego w panelu
            # (FFTAnalysisPanel może wyświetlać tylko dla app_controller.current_adsorbate_set_index)
            if set_index == self.app_controller.current_adsorbate_set_index:
                if "error" in params_dict:
                    QMessageBox.warning(self, f"Adsorbate Set {set_index+1} Calc Error", params_dict["error"])
                    self.fft_analysis_panel_widget.update_adsorbate_real_space_display(None)
                else:
                    self.fft_analysis_panel_widget.update_adsorbate_real_space_display(params_dict)
                    self.statusBar().showMessage(f"Adsorbate Set {set_index+1} real space parameters calculated.", 3000)
            # else: Można dodać log, że zaktualizowano dla nieaktywnego zestawu

    @pyqtSlot()
    def _on_calculate_substrate_rs_params_button_clicked(self):
        logger.debug("MainWindow: Calculate Substrate Real Space Params button clicked.")
        self.app_controller.calculate_and_store_substrate_real_params()

    @pyqtSlot(int) # Odbiera indeks zestawu adsorbatu z sygnału FFTAnalysisPanel
    def _on_calculate_adsorbate_rs_params_button_clicked(self, set_index: int):
        logger.debug(f"MainWindow: 'Calculate Adsorbate Real Space Params' button clicked for set index {set_index}.")
        if self.app_controller:
            # Upewnij się, że przekazujemy poprawny indeks (np. ten aktywny w AppController)
            # Jeśli FFTAnalysisPanel emituje indeks wybranego tam zestawu, to jest OK.
            # Ale jeśli logika ma zawsze działać na app_controller.current_adsorbate_set_index:
            active_set_index = self.app_controller.current_adsorbate_set_index
            if set_index != active_set_index: # pragma: no cover
                logger.warning(f"Request to calculate adsorbate params for set {set_index}, "
                               f"but current active set in AppController is {active_set_index}. Using active set.")
            self.app_controller.calculate_and_store_adsorbate_real_params(active_set_index)
        # Wyniki zostaną obsłużone przez slot _on_adsorbate_real_space_params_updated


    @pyqtSlot(int)
    def _on_adsorbate_set_updated(self, set_index: int):
        """
        Slot wywoływany, gdy dane zestawu adsorbatu (surowe lub skorygowane) zostaną zaktualizowane.
        Odświeża markery w VisualizationManager i potencjalnie inne UI.
        """
        logger.info(f"MainWindow: Adsorbate set {set_index} updated. Refreshing view.")
        self.display_image_data() # To powinno pobrać aktualne dane i flagi i przerysować
        self._update_action_states() # Zaktualizuj stan przycisków

    @pyqtSlot()
    def _on_substrate_definition_changed(self):
        """
        Slot wywoływany, gdy definicja substratu (typ, a_surf, nazwa) zmieni się w AppController.
        Aktualizuje odpowiednie UI, np. ComboBox w FFTAnalysisPanel i odświeża widok.
        """
        logger.debug("MainWindow: Received substrate_definition_changed from AppController.")
        if hasattr(self, 'fft_analysis_panel_widget') and self.fft_analysis_panel_widget:
            # Ustaw ComboBox w FFTAnalysisPanel na podstawie wartości z AppController
            current_def_name = self.app_controller.current_substrate_name # Używamy nowej nazwy atrybutu
            self.fft_analysis_panel_widget.set_substrate_combo_text(current_def_name)
        self.display_image_data() # Odśwież widok (np. idealną siatkę)
        self._update_action_states()

    @pyqtSlot()
    def _on_substrate_transform_results_updated(self):
        """
        Slot wywoływany, gdy AppController zaktualizuje wyniki transformacji substratu.
        Aktualizuje UI, w tym markery dopasowanych pików i wyświetlane parametry transformacji.
        """
        logger.debug("MainWindow: Received substrate_transform_results_updated signal.")
        
        # 1. Aktualizacja parametrów transformacji w FFTAnalysisPanel
        if hasattr(self, 'fft_analysis_panel_widget') and self.fft_analysis_panel_widget:
            analysis = self.app_controller.substrate_transform_analysis_m2i
            if analysis:
                self.fft_analysis_panel_widget.rotation_angle_label.setText(f"Rotation (M->I): {analysis.get('rotation_angle_deg', 'N/A'):.2f}°")
                s_x = analysis.get('principal_stretches', [np.nan, np.nan])[0]
                s_y = analysis.get('principal_stretches', [np.nan, np.nan])[1]
                self.fft_analysis_panel_widget.scale_factor_label.setText(f"Stretches (M->I): ({s_x:.3f}, {s_y:.3f})")
                self.fft_analysis_panel_widget.rmse_label.setText(f"Fit RMSE (M->I, px): {analysis.get('rmse', 'N/A'):.3f}")
            #     self.fft_analysis_panel_widget.show_fitted_substrate_spots_checkbox.setChecked(
            #         self.app_controller.show_fitted_substrate_spots
            # )
            else:
                self.fft_analysis_panel_widget.rotation_angle_label.setText("Rotation: -")
                self.fft_analysis_panel_widget.scale_factor_label.setText("Stretches: -")
                self.fft_analysis_panel_widget.rmse_label.setText("RMSE: -")

        # 2. Odświeżenie markerów w VisualizationManager
        # Zakładamy, że display_image_data() pobierze nowe displayable_fitted_substrate_spots_on_fft
        # i przekaże je do VisualizationManager.
        self.display_image_data() 
        self._update_action_states() # Na wszelki wypadek

    @pyqtSlot()
    def open_substrate_spot_selection_dialog(self):
        logger.info("MainWindow: Opening substrate spot selection dialog...")
        
        current_node_info = self.app_controller.get_current_node_info_for_dialogs()
        if not (current_node_info and current_node_info[1] == "FFT"): QMessageBox.warning(self, "Incorrect Data Type", "Substrate spots can only be selected on an FFT image."); return
        if not SubstrateSpotSelectionDialog: QMessageBox.critical(self, "Dialog Error", "SubstrateSpotSelectionDialog is not available."); return

        _, _, fft_image_data_copy = current_node_info
        
        dialog = SubstrateSpotSelectionDialog(
            fft_image_data=fft_image_data_copy,
            history_manager=self.history_manager,
            current_fft_node_id=current_node_info[0],
            current_spots=self.app_controller.user_selected_substrate_spots, # Używamy user_selected...
            initial_lattice_type=self.app_controller.substrate_lattice_type if self.app_controller.substrate_lattice_type else LATTICE_TYPE_HEXAGONAL,
            initial_selected_substrate_name=self.app_controller.substrate_definition_name,
            initial_custom_a_surf=self.app_controller.substrate_a_surf if self.app_controller.substrate_definition_name == PREDEFINED_SUBSTRATE_CUSTOM else None,
            default_refinement_method=self.app_controller.spot_refinement_method,
            default_refinement_roi_size=self.app_controller.refinement_roi_size,
            # Przekaż istniejące wyniki transformacji, jeśli dialog ma je wyświetlać/edytować
            initial_transform_F = self.app_controller.substrate_F_m2i,
            initial_transform_t = self.app_controller.substrate_t_m2i,
            initial_fitted_spots = self.app_controller.displayable_fitted_substrate_spots_on_fft,
            parent=self
        )
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            results = dialog.get_dialog_results()
            logger.info(f"Substrate spots dialog accepted. Results: {results}")
            self.app_controller.update_substrate_analysis_results(results)
            # statusBar message może być teraz emitowane przez AppController lub po sygnale
        else:
            logger.info("Substrate spots selection cancelled.")
            self.statusBar().showMessage("Substrate spots selection cancelled.", 3000)


    @pyqtSlot()
    def open_adsorbate_spot_selection_dialog(self):
        logger.info("MainWindow: Opening adsorbate spot selection dialog...")

        # 1. Sprawdź, czy aktywny jest obraz FFT i pobierz dane
        current_node_info = self.app_controller.get_current_node_info_for_dialogs()
        if not (current_node_info and current_node_info[1] == "FFT"):
            QMessageBox.warning(self, "Incorrect Data Type", 
                                "Adsorbate spots can only be selected on an FFT image.")
            logger.warning("Attempted to open adsorbate spot selection on non-FFT data.")
            return
            
        if not AdsorbateSpotSelectionDialog: # pragma: no cover
            QMessageBox.critical(self, "Dialog Error", 
                                 "AdsorbateSpotSelectionDialog is not available. Please check application setup.")
            logger.error("AdsorbateSpotSelectionDialog class is not available.")
            return

        parent_node_id, _, fft_image_data_copy = current_node_info
        
        # 2. Pobierz istniejące piki dla bieżącego zestawu adsorbatu z AppController
        current_set_idx = self.app_controller.current_adsorbate_set_index
        current_adsorbate_spots_for_set = []
        if 0 <= current_set_idx < len(self.app_controller.adsorbate_spot_sets):
            current_adsorbate_spots_for_set = list(self.app_controller.adsorbate_spot_sets[current_set_idx]) # Kopia
        else: # pragma: no cover
            logger.error(f"Invalid current_adsorbate_set_index ({current_set_idx}) for opening dialog.")
            # Można rozważyć ustawienie na 0, jeśli lista nie jest pusta, lub wyświetlenie błędu.
            # Na razie, jeśli indeks jest zły, dialog dostanie pustą listę.
            if not self.app_controller.adsorbate_spot_sets: # Jeśli jakimś cudem lista jest pusta
                self.app_controller.adsorbate_spot_sets.append([]) # Dodaj pierwszy pusty set
                current_set_idx = 0
            else: # Jeśli indeks jest zły, ale lista nie jest pusta, można np. ustawić na 0
                current_set_idx = 0
                current_adsorbate_spots_for_set = list(self.app_controller.adsorbate_spot_sets[current_set_idx])


        logger.debug(f"Passing to AdsorbateSpotSelectionDialog: set_index={current_set_idx}, existing_spots_count={len(current_adsorbate_spots_for_set)}")
        current_fft_node = self.history_manager.get_current_node()
        current_real_fft_node_id = current_fft_node.node_id

        sub_F = self.app_controller.substrate_F_m2i
        sub_t = self.app_controller.substrate_t_m2i
        sub_analysis = self.app_controller.substrate_transform_analysis_m2i

        fitted_sub_spots_px = self.app_controller.displayable_fitted_substrate_spots_on_fft

        ideal_sub_spots_for_ads_dialog = list(self.app_controller.reference_ideal_substrate_spots_px)

        # 3. Utwórz i wyświetl dialog
        dialog = AdsorbateSpotSelectionDialog(
            fft_image_data=fft_image_data_copy,
            history_manager=self.history_manager,
            current_fft_node_id=current_real_fft_node_id,
            current_adsorbate_spots=current_adsorbate_spots_for_set,
            adsorbate_set_index=current_set_idx,
            default_refinement_method=self.app_controller.spot_refinement_method,
            default_refinement_roi_size=self.app_controller.refinement_roi_size,
            substrate_F_m2i=sub_F,
            substrate_t_m2i=sub_t,
            substrate_transform_analysis=sub_analysis,
            ideal_substrate_spots_for_display_px=ideal_sub_spots_for_ads_dialog, # Przekaż obliczone/pobrane
            fitted_substrate_spots_for_display_px=fitted_sub_spots_px,
            parent=self
        )
        
        # 4. Obsłuż wynik dialogu
        if dialog.exec() == QDialog.DialogCode.Accepted:
            results = dialog.get_dialog_results()
            
            raw_spots_from_dialog = results.get("raw_adsorbate_spots", [])
            corrected_spots_from_dialog = results.get("corrected_adsorbate_spots_in_ideal_system", [])
            # Indeks zestawu powinien być ten sam, który przekazaliśmy, ale dialog go zwraca dla pewności
            set_idx_from_dialog = results.get("adsorbate_set_index", current_set_idx) 

            logger.info(f"Adsorbate spots dialog (set {set_idx_from_dialog + 1}) accepted. "
                        f"Raw: {len(raw_spots_from_dialog)}, Corrected: {len(corrected_spots_from_dialog)}")
            
            self.app_controller.update_adsorbate_set_results(
                set_index=set_idx_from_dialog,
                raw_spots=raw_spots_from_dialog,
                corrected_spots_ideal_system=corrected_spots_from_dialog
            )
            self.statusBar().showMessage(f"Adsorbate spots (Set {set_idx_from_dialog + 1}) updated.", 3000)
        else:
            logger.info(f"Adsorbate spots selection for set {current_set_idx + 1} cancelled.")
            self.statusBar().showMessage(f"Adsorbate spots (Set {current_set_idx + 1}) selection cancelled.", 3000)


    @pyqtSlot()
    def _on_spot_lists_or_params_changed(self):
        """
        Slot wywoływany, gdy zmienią się listy pików lub parametry ich wyboru/uściślania.
        Aktualizuje wyświetlanie tekstowe pików, markery na obrazie i stan akcji.
        """
        logger.debug("MainWindow: Received spot_lists_updated or spot_selection_parameters_changed signal.")
        if hasattr(self, '_update_selected_spots_display'):
            self._update_selected_spots_display() # Aktualizuje QTextEdit z listą pików
        
        # self.request_spot_markers_update() # Mówi VisualizationManager, aby przerysował markery
        
        if hasattr(self, '_update_action_states'):
            self._update_action_states() # Aktualizuje dostępność przycisków np. "Clear Last Point"

    @pyqtSlot()
    def _on_adsorbate_sets_structure_changed(self):
        """
        Slot wywoływany, gdy zmieni się struktura zestawów adsorbatu (np. dodano nowy, wyczyszczono wszystkie).
        Aktualizuje ComboBox w FFTAnalysisPanel.
        """
        logger.debug("MainWindow: Received adsorbate_sets_structure_changed signal.")
        if hasattr(self, 'fft_analysis_panel_widget') and self.fft_analysis_panel_widget:
            num_sets = len(self.app_controller.adsorbate_spot_sets)
            set_names_for_combo = [f"Set {i+1}" for i in range(num_sets)]
            
            current_set_text = ""
            if 0 <= self.app_controller.current_adsorbate_set_index < num_sets:
                current_set_text = f"Set {self.app_controller.current_adsorbate_set_index + 1}"
            elif num_sets > 0 : # Jeśli indeks jest niepoprawny, ale są zestawy, wybierz pierwszy
                current_set_text = "Set 1"
            
            self.fft_analysis_panel_widget.update_adsorbate_set_combo(set_names_for_combo, current_set_text)
        
        # Po zmianie struktury zestawów, również zaktualizuj wyświetlanie pików i stan akcji
        self._on_spot_lists_or_params_changed()

    @pyqtSlot(str)
    def _handle_substrate_changed(self, substrate_name: str):
        logger.debug(f"MainWindow: Substrate changed to '{substrate_name}' via panel signal.")
        self.app_controller.last_selected_substrate = substrate_name # Nadal można ustawiać bezpośrednio, jeśli to prosty stan
        self.app_controller.custom_lattice_info = None # Lub metoda self.app_controller.set_selected_substrate(substrate_name)
        self.display_image_data()

    @pyqtSlot(str)
    def _handle_refinement_method_changed_from_panel(self, method: str):
        logger.debug(f"MainWindow: Refinement method changed to '{method}' via panel signal.")
        self.app_controller.set_spot_refinement_method(method)

    @pyqtSlot(int)
    def _handle_refinement_area_size_changed_from_panel(self, area_size: int):
        logger.debug(f"MainWindow: Refinement area size changed to {area_size} via panel signal.")
        self.app_controller.set_refinement_roi_size(area_size)

    @pyqtSlot()
    def _handle_custom_lattice_request(self):
        logger.debug("MainWindow: Custom lattice definition requested via panel signal.")
        if not CustomLatticeDialog: # pragma: no cover
            QMessageBox.critical(self, "Error", "CustomLatticeDialog is not available.")
            return

        dialog = CustomLatticeDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            custom_def = dialog.get_lattice_definition()
            self.app_controller.custom_lattice_info = custom_def
            if self.app_controller.custom_lattice_info and self.fft_analysis_panel_widget:
                new_name = self.app_controller.custom_lattice_info.get("name", "Custom")
                self.fft_analysis_panel_widget.set_substrate_combo_text(new_name)
                self.app_controller.last_selected_substrate = new_name # Ustawiamy też last_selected
                logger.info(f"Custom lattice '{new_name}' defined and selected.")
                self.display_image_data()
            else: # pragma: no cover
                if self.fft_analysis_panel_widget:
                    self.fft_analysis_panel_widget.set_substrate_combo_text(self.app_controller.last_selected_substrate)
        else:
            logger.debug("Custom lattice definition dialog was cancelled.")
            if self.fft_analysis_panel_widget:
                self.fft_analysis_panel_widget.set_substrate_combo_text(self.app_controller.last_selected_substrate)

    
    @pyqtSlot(str)
    def _handle_current_adsorbate_set_changed_from_panel(self, set_name: str):
        found_idx = -1
        if hasattr(self, 'fft_analysis_panel_widget'):
            combo = self.fft_analysis_panel_widget.adsorbate_set_combo
            for i in range(combo.count()):
                if combo.itemText(i) == set_name and set_name != "<Add New Set...>":
                    found_idx = i
                    break

        if self.app_controller.current_adsorbate_set_index >= 0: # Upewnij się, że indeks jest prawidłowy
            expected_type = self.app_controller.adsorbate_expected_lattice_types.get(
                self.app_controller.current_adsorbate_set_index, ADSORBATE_LATTICE_TYPE_UNKNOWN
            )
            if hasattr(self, 'fft_analysis_panel_widget'):
                self.fft_analysis_panel_widget.set_expected_adsorbate_type(expected_type)
        
        if found_idx != -1:
            self.app_controller.set_current_adsorbate_set_by_index(found_idx)
            logger.info(f"MainWindow: Switched to adsorbate set '{set_name}' (Index: {self.app_controller.current_adsorbate_set_index}) via panel signal.")
        else: # pragma: no cover
             logger.warning(f"MainWindow: Could not map adsorbate set name '{set_name}' to an index.")


    @pyqtSlot(bool)
    def _handle_show_ideal_lattice_changed(self, is_visible: bool):
        logger.debug(f"MainWindow: Show ideal lattice changed to {is_visible} via panel signal.")
        self.app_controller.show_ideal_lattice = is_visible
        self.display_image_data()

    @pyqtSlot(bool)
    def _handle_substrate_spots_visibility_changed(self, is_visible: bool):
        logger.debug(f"MainWindow: Substrate spots visibility changed to {is_visible} via panel.")
        self.app_controller.show_substrate_spots_markers = is_visible
        # self.request_spot_markers_update() # Na razie zostawiamy, aby działało


    @pyqtSlot(bool)
    def _handle_adsorbate_spots_visibility_changed(self, is_visible: bool):
        logger.debug(f"MainWindow: Adsorbate spots visibility changed to {is_visible} via panel.")
        self.app_controller.show_adsorbate_spots_markers = is_visible
        # self.request_spot_markers_update() # Na razie zostawiamy
    
    def request_spot_markers_update(self):
        if hasattr(self, 'visualization_manager') and self.visualization_manager and \
           hasattr(self, 'fft_analysis_panel_widget') and self.fft_analysis_panel_widget:
            current_node = self.history_manager.get_current_node()
            if current_node and current_node.data_type == "FFT":
                # show_sub = self.app_controller.show_substrate_spots_markers
                # show_ads = self.app_controller.show_adsorbate_spots_markers
                substrate_spots_data = self.app_controller.substrate_spots
                adsorbate_spot_sets_data = self.app_controller.adsorbate_spot_sets
                self.visualization_manager.redraw_spot_markers(
                    substrate_spots_data, True,
                    adsorbate_spot_sets_data, True
                )
            else:
                 self.visualization_manager._clear_spot_markers_only()

    @pyqtSlot()
    def _handle_add_new_adsorbate_set_request(self):
        logger.info("MainWindow: Add new adsorbate set requested via panel signal.")
        if hasattr(self, 'fft_analysis_panel_widget'):
            num_sets = len(self.app_controller.adsorbate_spot_sets)
            set_names_for_combo = [f"Set {i+1}" for i in range(num_sets)]
            new_set_name = f"Set {num_sets}" # Ostatni dodany
            self.fft_analysis_panel_widget.update_adsorbate_set_combo(set_names_for_combo, new_set_name) # To ustawi currentIndex
            
            # Ustaw expected_type_combo dla nowego (teraz bieżącego) zestawu
            new_set_idx = self.app_controller.current_adsorbate_set_index
            expected_type = self.app_controller.adsorbate_expected_lattice_types.get(new_set_idx, ADSORBATE_LATTICE_TYPE_UNKNOWN)
            self.fft_analysis_panel_widget.set_expected_adsorbate_type(expected_type)
        self.app_controller.add_new_adsorbate_set()

    @pyqtSlot(object) # Sygnał z HistoryManager przekazuje HistoryNode lub None
    def _on_current_history_node_changed(self, current_node: Optional[HistoryNode]):
        logger.debug(f"MainWindow: Slot _on_current_history_node_changed received node: {current_node.node_id if current_node else 'None'}")
        
        # Najpierw zaktualizuj stany akcji i widoczność paneli
        self._update_action_states() # To ustawi widoczność fft_analysis_dock

        # Następnie wyświetl obraz
        self.display_image_data() # To odświeży główny obraz i może wywołać _update_action_states ponownie

        # Zaktualizuj metadane
        if hasattr(self, 'metadata_widget') and self.metadata_widget and hasattr(self, 'history_manager'):
            self.metadata_widget.update_metadata(current_node, self.history_manager)

        # --- NOWA LOGIKA: Zawsze odświeżaj FFTAnalysisPanel, jeśli jest widoczny ---
        # Po tym, jak _update_action_states ustawił jego widoczność
        if hasattr(self, 'fft_analysis_dock') and self.fft_analysis_dock.isVisible():
            if hasattr(self, 'fft_analysis_panel_widget') and self.fft_analysis_panel_widget and self.app_controller:
                logger.debug("MainWindow: FFT Analysis Panel is visible, refreshing its content.")
                # Odśwież informacje o transformacji substratu
                self.fft_analysis_panel_widget.update_transform_results_display(
                    self.app_controller.substrate_transform_analysis_m2i
                )
                # Odśwież parametry rzeczywiste substratu
                self.fft_analysis_panel_widget.update_substrate_real_space_display(
                    self.app_controller.substrate_real_space_results
                )
                # Odśwież parametry rzeczywiste dla bieżącego zestawu adsorbatu
                current_ads_idx = self.app_controller.current_adsorbate_set_index
                ads_params = self.app_controller.adsorbate_real_space_results.get(current_ads_idx)
                self.fft_analysis_panel_widget.update_adsorbate_real_space_display(ads_params)
                
                # Upewnij się, że ComboBox dla zestawów adsorbatu jest zsynchronizowany
                expected_type = self.app_controller.adsorbate_expected_lattice_types.get(
                    current_ads_idx, ADSORBATE_LATTICE_TYPE_UNKNOWN # Zaimportuj lub zdefiniuj stałą
                )
                self.fft_analysis_panel_widget.set_expected_adsorbate_type(expected_type)
                # Oraz sam ComboBox z listą zestawów (choć to powinno być zarządzane przez _on_adsorbate_sets_structure_changed)
                # self.fft_analysis_panel_widget.update_adsorbate_set_combo(...)
        # --- KONIEC NOWEJ LOGIKI ---

    # @pyqtSlot(object)
    # def _on_current_history_node_changed(self, current_node: Optional[HistoryNode]):
    #     logger.debug(f"MainWindow: Slot _on_current_history_node_changed received node: {current_node.node_id if current_node else 'None'}")
        
    #     self.display_image_data() # Ta metoda zbierze wszystko i wywoła visualization_manager

    #     if hasattr(self, 'metadata_widget') and self.metadata_widget and hasattr(self, 'history_manager'):
    #         self.metadata_widget.update_metadata(current_node, self.history_manager) 

    @pyqtSlot()
    def open_file_dialog(self):
        logger.debug("Open file dialog triggered.")
        file_filter = "STM Files (*.stp *.s94);;All Files (*)"
        start_dir = ""
        if self.app_controller.original_file_path:
            try:
                start_dir = os.path.dirname(self.app_controller.original_file_path)
            except Exception:  # pragma: no cover
                pass
        if not start_dir:  # pragma: no cover
            start_dir = os.path.expanduser("~")

        file_path, _ = QFileDialog.getOpenFileName(self, "Open STM File", start_dir, file_filter)

        if file_path:
            logger.info(f"File selected by user: {file_path}")
            self.statusBar().showMessage(f"Loading file: {os.path.basename(file_path)}...")
            QApplication.processEvents() # type: ignore # Daj GUI szansę na odświeżenie
            
            self.app_controller.load_file(file_path)
        else: # pragma: no cover
            logger.debug("File dialog cancelled by user.")
            self.statusBar().showMessage("File open cancelled.", 3000)

    @pyqtSlot(str)
    def _on_file_loaded_successfully(self, filename: str):
        """Slot called when AppController successfully loads a file."""
        logger.info(f"MainWindow: Received file_loaded_successfully signal for '{filename}'.")
        self.statusBar().showMessage(f"Loaded: {filename}", 5000)
        self.setWindowTitle(f"LFA - {filename}")

    @pyqtSlot(str)
    def _on_file_loading_failed(self, error_message: str):
        """Slot called when AppController fails to load a file."""
        logger.warning(f"MainWindow: Received file_loading_failed signal: {error_message}")
        self.statusBar().showMessage(f"Failed to load file: {error_message}", 5000)
        QMessageBox.warning(self, "Loading Error", error_message)
        self.setWindowTitle("Lattice Fourier Analyzer (LFA)")

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
            self._points_for_current_adsorbate_set = [] # Wyczyść przy zmianie na adsorbat
            if self.current_adsorbate_set_index < len(self.adsorbate_spot_sets):
                 self._points_for_current_adsorbate_set = list(self.adsorbate_spot_sets[self.current_adsorbate_set_index])
            logger.debug(f"Spot selection mode: Adsorbate, Set Index: {self.current_adsorbate_set_index}")

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

        self._points_for_current_adsorbate_set = []
        if self.current_adsorbate_set_index < len(self.adsorbate_spot_sets):
            self._points_for_current_adsorbate_set = list(self.adsorbate_spot_sets[self.current_adsorbate_set_index])


    @pyqtSlot()
    def _on_reselect_adsorbate_set_clicked(self):
        """Clears points for the current adsorbate set to allow re-selection."""
        if self.current_adsorbate_set_index >= 0 and self.current_adsorbate_set_index < len(self.adsorbate_spot_sets):
            logger.info(f"Reselecting points for adsorbate set {self.current_adsorbate_set_index + 1}")
            self.adsorbate_spot_sets[self.current_adsorbate_set_index] = [] # Wyczyść zapisane punkty
            self._points_for_current_adsorbate_set = [] # Wyczyść punkty tymczasowe
        else:
            logger.warning("No valid adsorbate set selected to reselect.")

    @pyqtSlot()
    def _on_clear_all_adsorbate_sets_clicked(self):
        logger.info("MainWindow: Clearing all adsorbate spot sets button clicked.")
        self.app_controller.clear_all_adsorbate_sets()
        if hasattr(self, 'fft_analysis_panel_widget'):
            self.fft_analysis_panel_widget.update_adsorbate_set_combo(["Set 1"], "Set 1")

    @pyqtSlot(str)
    def _handle_spot_selection_mode_changed_from_panel(self, mode: str):
        logger.debug(f"MainWindow: Spot selection mode changed to '{mode}' via panel.")
        self.app_controller.set_spot_selection_mode(mode)


    @pyqtSlot()
    def _on_visibility_checkbox_changed(self):
        """Slot for all visibility checkboxes."""
        logger.debug("Visibility checkbox changed, updating markers and ideal lattice.")
        self.display_image_data() # To odświeży idealną sieć, jeśli trzeba
        # self.request_spot_markers_update()

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
                self.substrate_combo.setCurrentText(self.last_selected_substrate)
        else:
            self.custom_lattice_info = None # Wyczyść definicję własną
            self.last_selected_substrate = selected_text # Zapamiętaj wybór

        self.display_image_data()

    def open_fft_dialog(self):
        current_node_info = self.app_controller.get_current_node_info_for_dialogs()
        if not current_node_info:
            QMessageBox.warning(self, "No Image", "No data loaded or selected to calculate FFT.") # pragma: no cover
            return

        parent_id, parent_data_type, image_data_copy = current_node_info

        if parent_data_type != "STM": # pragma: no cover
            QMessageBox.warning(self, "Invalid Data Type", "FFT can only be calculated from STM data (not from an existing FFT).")
            return
        if not FFTDialog: # pragma: no cover
            QMessageBox.critical(self, "Error", "FFTDialog is not available.")
            return

        dialog = FFTDialog(image_data_copy, parent=self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            processed_fft_data = dialog.get_processed_data() # To jest już przeskalowana magnituda
            params = dialog.get_fft_parameters() # Zawiera m.in. scaling_mode, window_type
            was_roi_only = dialog.was_roi_applied_only() # Czy FFT było tylko z ROI

            if processed_fft_data is not None:
                self.app_controller.calculate_fft_operation(
                    parent_node_id=parent_id,
                    processed_fft_data=processed_fft_data,
                    params=params,
                    source_roi_slice=dialog.get_source_roi_slice() if was_roi_only else None
                )
                self.statusBar().showMessage("FFT calculated and scaled.", 3000)
            else: # pragma: no cover
                logger.warning("FFTDialog accepted, but no processed data returned.")
                self.statusBar().showMessage("FFT calculation failed or no data.", 3000)
        else: # pragma: no cover
            logger.info("FFT dialog cancelled.")
            self.statusBar().showMessage("FFT calculation cancelled.", 3000)


    def open_gaussian_sharpening_dialog(self):
        self._helper_open_processing_dialog(GaussianSharpeningDialog, "apply_gaussian_sharpening")

    def open_bm3d_dialog(self):
        def bm3d_checks(): # pragma: no cover
            try: import bm3d; return True
            except ImportError: QMessageBox.critical(self,"Missing Dependency","BM3D package needed."); return False
        self._helper_open_processing_dialog(BM3DDialog, "apply_bm3d_denoising", dialog_specific_checks=bm3d_checks)

    def open_nlmeans_dialog(self):
        self._helper_open_processing_dialog(NLMeansDialog, "apply_nlmeans_denoising")

    def open_median_filter_dialog(self):
        self._helper_open_processing_dialog(MedianFilterDialog, "apply_median_filter")

    def open_plane_leveling_dialog(self):
        self._helper_open_processing_dialog(PlaneLevelingDialog, "apply_plane_leveling")


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

    def open_gaussian_blur_dialog(self):
        self._helper_open_processing_dialog(GaussianBlurDialog, "apply_gaussian_blur")

    def display_image_data(self):
        if not hasattr(self, 'visualization_manager') or self.visualization_manager is None:
            logger.error("MainWindow: VisualizationManager not available for displaying data.")
            if hasattr(self, 'image_view') and self.image_view: self.image_view.clear() # pragma: no cover
            return
        if not hasattr(self, 'history_manager') or self.history_manager is None: # pragma: no cover
            logger.error("MainWindow: HistoryManager not available.")
            if hasattr(self, 'image_view') and self.image_view: self.image_view.clear()
            return

        current_node = self.history_manager.get_current_node()
        show_ideal_lattice = False
        selected_substrate = "None" # Domyślna wartość
        panel_custom_text = ""

        substrate_spots_to_draw = self.app_controller.displayable_fitted_substrate_spots_on_fft
        corrected_adsorbate_sets_ideal_sys = self.app_controller.corrected_adsorbate_spot_sets

        if hasattr(self, 'fft_analysis_panel_widget') and self.fft_analysis_panel_widget is not None:
            show_ideal_lattice = self.fft_analysis_panel_widget.is_show_ideal_lattice_checked() # lub self.app_controller.show_ideal_lattice
            selected_substrate = self.fft_analysis_panel_widget.get_current_substrate() # lub self.app_controller.last_selected_substrate
            panel_custom_text = self.fft_analysis_panel_widget.custom_option_text
        
        self.visualization_manager.update_view(
            current_node,
            # ... (parametry dla idealnej siatki) ...
            show_ideal_lattice, 
            selected_substrate, 
            self.app_controller.custom_lattice_info, 
            panel_custom_text,  
            # --- Przekazanie odpowiednich danych i flagi widoczności ---
            substrate_spots_to_draw,      # Tylko dopasowane piki
            True, # Flaga widoczności dla nich
            # --- Koniec ---
            corrected_adsorbate_sets_ideal_sys,
            True # Ta flaga też powinna być zarządzana przez AppController
        )
        if hasattr(self, '_update_action_states'): self._update_action_states()
    
    @pyqtSlot(QPointF)
    def _on_fft_view_clicked_from_visualizer(self, mapped_data_pos: QPointF):
        logger.debug(f"MainWindow: Received FFT click at data coords (kx, ky): ({mapped_data_pos.x():.2f}, {mapped_data_pos.y():.2f})")
        current_node = self.history_manager.get_current_node()
        if not (current_node and current_node.data_type == "FFT" and current_node.image_data is not None):
            logger.warning("_on_fft_view_clicked_from_visualizer: No valid FFT data node active.") # pragma: no cover
            return

        kx_from_signal, ky_from_signal = mapped_data_pos.x(), mapped_data_pos.y()
        kx_int, ky_int = int(round(kx_from_signal)), int(round(ky_from_signal))
        original_fft_data = current_node.image_data
        fft_data_rows_ky, fft_data_cols_kx = original_fft_data.shape
        if not (0 <= ky_int < fft_data_rows_ky and 0 <= kx_int < fft_data_cols_kx): # pragma: no cover
            logger.debug(f"Click data coords outside original FFT data bounds. Ignoring.")
            return

        center_yx_for_refinement = (ky_int, kx_int)
        refined_kx, refined_ky = kx_int, ky_int

        current_refinement_method = self.app_controller.spot_refinement_method
        current_refinement_radius = self.app_controller.refinement_roi_size // 2

        logger.debug(f"Refinement: Method='{current_refinement_method}', Radius for func={current_refinement_radius}, Click (ky,kx)=({ky_int},{kx_int})")
        if current_refinement_method == "Max Pixel": # REFINEMENT_MAX_PIXEL
            if PEAK_FITTING_AVAILABLE:
                refined_ky_temp, refined_kx_temp = find_max_pixel_in_roi(original_fft_data, center_yx_for_refinement, current_refinement_radius)
                refined_kx, refined_ky = int(refined_kx_temp), int(refined_ky_temp)
                logger.info(f"Max Pixel refined: (orig_kx={kx_int}, orig_ky={ky_int}) -> (ref_kx={refined_kx}, ref_ky={refined_ky})")
        elif current_refinement_method == "2D Gaussian Fit": # REFINEMENT_GAUSSIAN_FIT
            if PEAK_FITTING_AVAILABLE:
                fit_result = fit_2d_gaussian_in_roi(original_fft_data, center_yx_for_refinement, current_refinement_radius)
                if fit_result:
                    refined_ky_float, refined_kx_float = fit_result
                    refined_kx, refined_ky = int(round(refined_kx_float)), int(round(refined_ky_float)) # Używamy zaokrąglonych intów dla spójności
                    logger.info(f"2D Gaussian Fit refined: -> (ref_kx_float={refined_kx_float:.2f}, ref_ky_float={refined_ky_float:.2f}) -> int({refined_kx},{refined_ky})")
                else: # pragma: no cover
                    logger.warning("2D Gaussian Fit failed. Using rounded click position.")
            else:  # pragma: no cover
                 logger.warning("Peak fitting (Gaussian) backend not available. Using rounded click.")
        
        # final_point_coords_kx_ky = (float(refined_kx), float(refined_ky)) # Przechowuj jako float dla potencjalnej przyszłej precyzji

        # self.app_controller.add_spot(final_point_coords_kx_ky)

    @pyqtSlot()
    def _on_clear_substrate_spots_clicked(self):
        logger.info("MainWindow: Clearing substrate spots.")
        self.app_controller.clear_substrate_spots()


    @pyqtSlot()
    def _on_reselect_adsorbate_set_clicked(self):
        logger.debug("MainWindow: Reselect current adsorbate set button clicked.")
        self.app_controller.reselect_current_adsorbate_set()


    def closeEvent(self, event):
        """Handle the event when the user tries to close the window."""
        logger.info("Close event triggered. Exiting application.")
        event.accept()