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
    QDialog, QHBoxLayout, QSplitter, QListWidget, QListWidgetItem, QDockWidget
)
from PyQt6.QtGui import QAction, QIcon
from PyQt6.QtCore import Qt, pyqtSlot, QPointF
from PIL import Image

try:
    import pyqtgraph as pg
except ImportError:
    logging.error("PyQtGraph not found. Please install it: pip install pyqtgraph")
    pg = None

from .widgets.metadata_widget import MetadataWidget
from ..logic.history_manager import HistoryManager
from .panels.fft_analysis_panel import FFTAnalysisPanel
from .visualization_manager import VisualizationManager
from ..logic.app_controller import AppController, LATTICE_ANALYSIS_FUNCTIONS_AVAILABLE
from ..core.history import HistoryNode
from .ui_setup.menu_action_manager import MenuActionManager
from .ui_setup.dock_panel_manager import DockPanelManager
from .ui_setup.history_context_menu import HistoryContextMenu
from .controllers.ui_state_binder import UIStateBinder
from .controllers.overlay_visibility_binder import OverlayVisibilityBinder
from .controllers.dialog_coordinator import DialogCoordinator
from ..core.constants import (
    ADSORBATE_LATTICE_TYPE_UNKNOWN,
    LATTICE_TYPE_CUSTOM,
    LATTICE_TYPE_HEXAGONAL,
    LATTICE_TYPE_SQUARE,
    PREDEFINED_SUBSTRATE_CUSTOM,
    PREDEFINED_SUBSTRATE_NONE,
)
from .utils.display import format_float, format_pair

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
    def find_max_pixel_in_roi(data, center, radius): return center
    def fit_2d_gaussian_in_roi(data, center, radius): return None
    PEAK_FITTING_AVAILABLE = False

try:
    from .dialogs.real_space_reconstruction_dialog import RealSpaceReconstructionDialog
    RECONSTRUCTION_DIALOG_AVAILABLE = True
except ImportError as e:
    RealSpaceReconstructionDialog = None; RECONSTRUCTION_DIALOG_AVAILABLE = False
    logging.warning(f"Could not import RealSpaceReconstructionDialog: {e}")

try:
    from .dialogs.substrate_spot_dialog import SubstrateSpotSelectionDialog
    from .dialogs.adsorbate_spot_dialog import AdsorbateSpotSelectionDialog
    SPOT_SELECTION_DIALOGS_AVAILABLE = True
except ImportError:
    logging.warning("Could not import spot selection dialogs. Spot selection from menu will not work.")
    SubstrateSpotSelectionDialog = None
    AdsorbateSpotSelectionDialog = None
    SPOT_SELECTION_DIALOGS_AVAILABLE = False

try:
    from .dialogs.superstructure_periodicity_dialog import SuperstructurePeriodicityDialog
    SUPERSTRUCTURE_PERIODICITY_DIALOG_AVAILABLE = True
except ImportError:
    SuperstructurePeriodicityDialog = None
    SUPERSTRUCTURE_PERIODICITY_DIALOG_AVAILABLE = False
    logging.warning("Could not import SuperstructurePeriodicityDialog.")

logger = logging.getLogger(__name__)


try:
    from ..analysis.lattice import get_reciprocal_points, KNOWN_LATTICES
    from lfa.gui.dialogs.custom_lattice_dialog import CustomLatticeDialog
    LATTICE_ANALYSIS_AVAILABLE = True
except ImportError:
    logging.error("Could not import lattice analysis functions.")
    KNOWN_LATTICES = {"None": {}} 
    def get_reciprocal_points(name, max_hk=2): return None
    CustomLatticeDialog = None
    LATTICE_ANALYSIS_AVAILABLE = False

try:
    from .dialogs.real_space_visualizer_dialog import RealSpaceFFTVisualizerDialog
    REAL_SPACE_VIS_DIALOG_AVAILABLE = True
except ImportError: 
    RealSpaceFFTVisualizerDialog = None
    REAL_SPACE_VIS_DIALOG_AVAILABLE = False
    logging.warning("Could not import RealSpaceFFTVisualizerDialog.")

try:
    from .dialogs.stm_transform_dialog import StmTransformDialog
    STM_TRANSFORM_DIALOG_AVAILABLE = True
except ImportError as e:
    StmTransformDialog = None; STM_TRANSFORM_DIALOG_AVAILABLE = False
    logging.warning(f"Could not import StmTransformDialog: {e}")

class MainWindow(QMainWindow):
    """
    The main application window inheriting from QMainWindow.
    Provides menu bar, status bar, and central widget area.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Lattice Fourier Analyzer (LFA)")
        self.resize(1250, 800)

        self._setup_main_layout() 
        self.metadata_widget = MetadataWidget(self) # Create content widgets
        self.fft_analysis_panel_widget = FFTAnalysisPanel(self)

        self.history_manager = HistoryManager(self.history_list_widget, self)
        self.app_controller = AppController(history_manager=self.history_manager)
        
        self._init_core_attributes()

        self.menu_manager = MenuActionManager(self) 
        self.history_context_menu = HistoryContextMenu(self)

        self.dock_manager = DockPanelManager(
            main_window=self,
            history_list_widget=self.history_list_widget,
            metadata_widget=self.metadata_widget,
            fft_analysis_panel_widget=self.fft_analysis_panel_widget
        )

        self._create_status_bar()

        self.ui_state_binder = UIStateBinder(
            app_controller=self.app_controller,
            history_manager=self.history_manager,
            fft_analysis_panel=self.fft_analysis_panel_widget,
            fft_analysis_dock=getattr(self, "fft_analysis_dock", None),
            actions={
                "load_metadata": self.menu_manager.file_actions.get("load_metadata"),
                "gaussian_blur": getattr(self, "gaussian_blur_action", None),
                "gaussian_sharpen": getattr(self, "gaussian_sharpen_action", None),
                "plane_level": getattr(self, "plane_level_action", None),
                "median_filter": getattr(self, "median_filter_action", None),
                "nlmeans": getattr(self, "nlmeans_action", None),
                "bm3d": getattr(self, "bm3d_action", None),
                "fft": getattr(self, "fft_action", None),
                "select_substrate_spots": getattr(self, "select_substrate_spots_action", None),
                "select_adsorbate_spots": getattr(self, "select_adsorbate_spots_action", None),
                "superstructure_periodicity": getattr(self, "superstructure_periodicity_action", None),
                "stm_transform": getattr(self, "stm_transform_action", None),
                "visualize_real_space": getattr(self, "visualize_real_space_action", None),
                "real_space_reconstruction": getattr(self, "real_space_reconstruction_action", None),
            },
            availability={
                "preprocessing_dialogs": DIALOG_CLASSES_EXIST,
                "spot_dialogs": SPOT_SELECTION_DIALOGS_AVAILABLE,
                "superstructure_dialog": SUPERSTRUCTURE_PERIODICITY_DIALOG_AVAILABLE,
                "stm_transform_dialog": STM_TRANSFORM_DIALOG_AVAILABLE,
                "lattice_analysis": LATTICE_ANALYSIS_FUNCTIONS_AVAILABLE,
            },
        )

        if pg and self.image_view and self.history_manager and VisualizationManager:
            self.visualization_manager = VisualizationManager(
                image_view=self.image_view,
                history_manager=self.history_manager,
            )
            logger.info("VisualizationManager created and initialized.")
        else:
            self.visualization_manager = None
            logger.error("Could not create VisualizationManager due to missing dependencies.")

        self.overlay_binder = OverlayVisibilityBinder(
            app_controller=self.app_controller,
            fft_analysis_panel=self.fft_analysis_panel_widget,
            visualization_manager=self.visualization_manager,
        )
        self.overlay_binder.apply_panel_state_to_controller()

        self.dialog_coordinator = DialogCoordinator(
            main_window=self,
            app_controller=self.app_controller,
            history_manager=self.history_manager,
            visualization_manager=self.visualization_manager,
            fft_dialog_class=FFTDialog,
            substrate_dialog_class=SubstrateSpotSelectionDialog,
            adsorbate_dialog_class=AdsorbateSpotSelectionDialog,
            superstructure_dialog_class=SuperstructurePeriodicityDialog,
            real_space_visualizer_class=RealSpaceFFTVisualizerDialog,
            real_space_reconstruction_class=RealSpaceReconstructionDialog,
            stm_transform_dialog_class=StmTransformDialog,
        )

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
        self.history_list_widget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)

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
        """Connects signals to slots."""
        if self.history_manager:
            self.history_list_widget.currentItemChanged.connect(self.on_history_selection_changed)
            self.history_manager.current_node_changed.connect(self._on_current_history_node_changed)
            self.history_list_widget.customContextMenuRequested.connect(self.history_context_menu.show_menu)

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
            self.app_controller.substrate_definition_changed.connect(self._on_substrate_definition_changed)
            self.app_controller.spot_lists_updated.connect(self._on_spot_lists_or_params_changed) 

            self.app_controller.substrate_real_space_params_updated.connect(self._on_substrate_real_space_params_updated)
            self.app_controller.adsorbate_real_space_params_updated.connect(self._on_adsorbate_real_space_params_updated)
            self.app_controller.superstructure_periodicity_results_updated.connect(self._on_superstructure_periodicity_results_updated)

        if hasattr(self, 'fft_analysis_panel_widget'):
            self.fft_analysis_panel_widget.substrate_changed.connect(self._handle_substrate_changed)
            self.fft_analysis_panel_widget.custom_lattice_define_requested.connect(self._handle_custom_lattice_request)
            self.fft_analysis_panel_widget.show_ideal_lattice_changed.connect(self._handle_show_ideal_lattice_changed)
            self.fft_analysis_panel_widget.spot_selection_mode_changed.connect(self._handle_spot_selection_mode_changed_from_panel)
            self.fft_analysis_panel_widget.current_adsorbate_set_changed.connect(self._handle_current_adsorbate_set_changed_from_panel)
            self.fft_analysis_panel_widget.add_new_adsorbate_set_requested.connect(self._handle_add_new_adsorbate_set_request)
            self.fft_analysis_panel_widget.reselect_current_adsorbate_set_triggered.connect(self._on_reselect_adsorbate_set_clicked)
            self.fft_analysis_panel_widget.clear_all_adsorbate_sets_triggered.connect(self._on_clear_all_adsorbate_sets_clicked)
            self.fft_analysis_panel_widget.substrate_spots_visibility_changed.connect(self._handle_substrate_spots_visibility_changed)
            self.fft_analysis_panel_widget.adsorbate_spots_visibility_changed.connect(self._handle_adsorbate_spots_visibility_changed)
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
        """Updates the display of selected spots in the FFTAnalysisPanel."""
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

    @pyqtSlot()
    def save_analysis(self):
        if self.app_controller:
            logger.debug("Menu 'Save Analysis' clicked, calling controller.")
            self.app_controller.save_analysis_session()

    @pyqtSlot()
    def load_analysis(self):
        if self.app_controller:
            logger.debug("Menu 'Load Analysis' clicked, calling controller.")
            self.app_controller.load_analysis_session()

    def _helper_open_processing_dialog(self, DialogClass, op_name_in_controller: str, dialog_specific_checks=None):
        """Helper method to open processing dialogs and handle results."""
        current_node_info = self.app_controller.get_current_node_info_for_dialogs()
        if not current_node_info:
            QMessageBox.warning(self, "No Image", "No data loaded or selected in history to process.")
            return

        parent_id, parent_data_type, image_data_copy, source_image_id, source_label = current_node_info

        if not DialogClass:
            QMessageBox.critical(self, "Error", f"{DialogClass.__name__ if DialogClass else 'Dialog'} is not available.")
            return
            
        if dialog_specific_checks:
             if not dialog_specific_checks(): return


        dialog = DialogClass(image_data_copy, parent=self)
        dialog.source_image_id = source_image_id
        dialog.source_image_label = source_label
        if source_label and source_label not in dialog.windowTitle():
            dialog.setWindowTitle(f"{dialog.windowTitle()} [{source_label}]")
        if dialog.exec() == QDialog.DialogCode.Accepted:
            processed_data = dialog.get_processed_data()
            params = dialog.get_parameters()
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
                else:
                    logger.error(f"Method {op_name_in_controller} not found in AppController!")
                    self.statusBar().showMessage(f"Error applying {op_name_in_controller}.", 3000)
            else:
                logger.warning(f"{DialogClass.__name__} accepted, but no processed data returned.")
                self.statusBar().showMessage("Operation cancelled or no changes made.", 3000)
        else:
            op_display_name = dialog.operation_name if hasattr(dialog, 'operation_name') else op_name_in_controller.replace("apply_", "").replace("_operation","").title()
            logger.info(f"{op_display_name} dialog cancelled.")
            self.statusBar().showMessage(f"{op_display_name} cancelled.", 3000)
    
    def _update_action_states(self):
        """Delegate UI state refresh to the UIStateBinder helper."""
        if hasattr(self, "ui_state_binder") and self.ui_state_binder:
            self.ui_state_binder.refresh()
    
    @pyqtSlot()
    def open_stm_transform_dialog(self):
        """Open the STM transform dialog via the coordinator."""
        if hasattr(self, "dialog_coordinator") and self.dialog_coordinator:
            self.dialog_coordinator.open_stm_transform_dialog()

    @pyqtSlot()
    def load_metadata_for_session(self):
        """Intermediate slot that triggers metadata loading logic."""
        if self.app_controller:
            self.app_controller.load_metadata_into_session()

    @pyqtSlot()
    def clear_analysis_session(self):
        """Clears all loaded images and analysis results after user confirmation."""
        response = QMessageBox.question(
            self,
            "Clear Session",
            "Remove all loaded images and analysis results?\nThis action cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if response != QMessageBox.StandardButton.Yes:
            logger.info("MainWindow: Session clear cancelled by user.")
            return

        logger.info("MainWindow: Clearing analysis session at user request.")
        if self.app_controller:
            self.app_controller.reset_session()
        if hasattr(self, "metadata_widget") and self.metadata_widget:
            self.metadata_widget.clear_labels()

        self.display_image_data()
        self._update_action_states()
        self.statusBar().showMessage("Session cleared.", 3000)
        self.setWindowTitle("Lattice Fourier Analyzer (LFA)")

    @pyqtSlot(object) 
    def _on_superstructure_periodicity_results_updated(self, results: Optional[Dict[str, Any]]):
        """Updates the panel with the superstructure periodicity analysis results."""
        if hasattr(self, 'fft_analysis_panel_widget') and self.fft_analysis_panel_widget:
            logger.debug(f"MainWindow: Received superstructure periodicity results update: {results}")
            self.fft_analysis_panel_widget.update_superstructure_periodicity_display(results)

    @pyqtSlot()
    def open_real_space_reconstruction_dialog(self):
        """Open the real space reconstruction dialog via the coordinator."""
        if hasattr(self, "dialog_coordinator") and self.dialog_coordinator:
            self.dialog_coordinator.open_real_space_reconstruction_dialog()

    @pyqtSlot()
    def open_real_space_fft_visualizer(self):
        """Open the real space FFT visualizer via the coordinator."""
        if hasattr(self, "dialog_coordinator") and self.dialog_coordinator:
            self.dialog_coordinator.open_real_space_fft_visualizer()

    @pyqtSlot(int, str)
    def _handle_expected_adsorbate_type_changed_from_panel(self, set_index: int, selected_type: str):
        """Handles the update of expected adsorbate type from FFTAnalysisPanel."""
        logger.debug(f"MainWindow: Expected adsorbate type for set {set_index} changed to '{selected_type}' via panel.")
        self.app_controller.set_expected_adsorbate_lattice_type(set_index, selected_type)
        self._update_action_states()


    @pyqtSlot(int, str)
    def _on_adsorbate_expected_type_updated(self, set_index: int, type_name: str):
        """Handles the update of expected adsorbate type in AppController."""
        logger.debug(f"MainWindow: AppController reported expected adsorbate type for set {set_index} is now '{type_name}'.")
        if hasattr(self, 'fft_analysis_panel_widget') and self.fft_analysis_panel_widget:
            current_panel_set_index = self.fft_analysis_panel_widget.adsorbate_set_combo.currentIndex()
            if current_panel_set_index < (self.fft_analysis_panel_widget.adsorbate_set_combo.count() -1) \
               and current_panel_set_index == set_index :
                self.fft_analysis_panel_widget.set_expected_adsorbate_type(type_name)
        self._update_action_states()

    @pyqtSlot(dict)
    def _on_substrate_real_space_params_updated(self, params_dict: dict):
        """Handles the update of substrate real space parameters."""
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
        """Handles the update of adsorbate real space parameters."""
        logger.debug(f"MainWindow: Received adsorbate_real_space_params_updated for set {set_index}: {params_dict}")
        if hasattr(self, 'fft_analysis_panel_widget') and self.fft_analysis_panel_widget:
            if set_index == self.app_controller.current_adsorbate_set_index:
                if "error" in params_dict:
                    QMessageBox.warning(self, f"Adsorbate Set {set_index+1} Calc Error", params_dict["error"])
                    self.fft_analysis_panel_widget.update_adsorbate_real_space_display(None)
                else:
                    self.fft_analysis_panel_widget.update_adsorbate_real_space_display(params_dict)
                    self.statusBar().showMessage(f"Adsorbate Set {set_index+1} real space parameters calculated.", 3000)
    @pyqtSlot()
    def open_superstructure_periodicity_dialog(self):
        """Open the superstructure periodicity dialog via the coordinator."""
        if hasattr(self, "dialog_coordinator") and self.dialog_coordinator:
            self.dialog_coordinator.open_superstructure_periodicity_dialog()

    @pyqtSlot()
    def _on_calculate_substrate_rs_params_button_clicked(self):
        """Handles the calculation of substrate real space parameters."""
        logger.debug("MainWindow: Calculate Substrate Real Space Params button clicked.")
        self.app_controller.calculate_and_store_substrate_real_params()

    @pyqtSlot(int)
    def _on_calculate_adsorbate_rs_params_button_clicked(self, set_index: int):
        """Handles the calculation of adsorbate real space parameters."""
        logger.debug(f"MainWindow: 'Calculate Adsorbate Real Space Params' button clicked for set index {set_index}.")
        if self.app_controller:
            active_set_index = self.app_controller.current_adsorbate_set_index
            if set_index != active_set_index: # pragma: no cover
                logger.warning(f"Request to calculate adsorbate params for set {set_index}, "
                               f"but current active set in AppController is {active_set_index}. Using active set.")
            self.app_controller.calculate_and_store_adsorbate_real_params(active_set_index)


    @pyqtSlot(int)
    def _on_adsorbate_set_updated(self, set_index: int):
        """Handles the update of adsorbate set in AppController."""
        logger.info(f"MainWindow: Adsorbate set {set_index} updated. Refreshing view.")
        self.display_image_data() 
        self._update_action_states() 

    @pyqtSlot()
    def _on_substrate_definition_changed(self):
        """Handles the update of substrate definition in AppController."""
        logger.debug("MainWindow: Received substrate_definition_changed from AppController.")
        if hasattr(self, 'fft_analysis_panel_widget') and self.fft_analysis_panel_widget:
            current_def_name = self.app_controller.current_substrate_name 
            self.fft_analysis_panel_widget.set_substrate_combo_text(current_def_name)
        self.display_image_data()
        self._update_action_states()

    @pyqtSlot()
    def _on_substrate_transform_results_updated(self):
        """Handles the update of substrate transform results in AppController."""
        logger.debug("MainWindow: Received substrate_transform_results_updated signal.")
        
        if hasattr(self, 'fft_analysis_panel_widget') and self.fft_analysis_panel_widget:
            analysis = self.app_controller.substrate_transform_analysis_m2i
            if analysis:
                rotation_text = format_float(analysis.get("rotation_angle_deg"), precision=2)
                rotation_display = rotation_text if rotation_text == "-" else f"{rotation_text} deg"

                stretch_display = format_pair(analysis.get("principal_stretches"), precision=3)

                rmse_text = format_float(analysis.get("rmse"), precision=3)

                self.fft_analysis_panel_widget.rotation_angle_label.setText(
                    f"Rotation (M->I): {rotation_display}"
                )
                self.fft_analysis_panel_widget.scale_factor_label.setText(
                    f"Stretches (M->I): {stretch_display}"
                )
                self.fft_analysis_panel_widget.rmse_label.setText(
                    f"Fit RMSE (M->I, px): {rmse_text}"
                )
            else:
                self.fft_analysis_panel_widget.rotation_angle_label.setText("Rotation: -")
                self.fft_analysis_panel_widget.scale_factor_label.setText("Stretches: -")
                self.fft_analysis_panel_widget.rmse_label.setText("RMSE: -")
        self.display_image_data() 
        self._update_action_states()

    @pyqtSlot()
    def open_substrate_spot_selection_dialog(self):
        """Open the substrate spot selection dialog via the coordinator."""
        if hasattr(self, "dialog_coordinator") and self.dialog_coordinator:
            self.dialog_coordinator.open_substrate_spot_selection_dialog()


    @pyqtSlot()
    def open_adsorbate_spot_selection_dialog(self):
        """Open the adsorbate spot selection dialog via the coordinator."""
        if hasattr(self, "dialog_coordinator") and self.dialog_coordinator:
            self.dialog_coordinator.open_adsorbate_spot_selection_dialog()


    @pyqtSlot()
    def _on_spot_lists_or_params_changed(self):
        """Handles the update of spot lists or parameters in AppController."""
        logger.debug("MainWindow: Received spot_lists_updated or spot_selection_parameters_changed signal.")
        if hasattr(self, '_update_selected_spots_display'):
            self._update_selected_spots_display()
                
        if hasattr(self, '_update_action_states'):
            self._update_action_states()

    def _on_adsorbate_sets_structure_changed(self):
        """Handles the update of adsorbate sets structure in AppController."""
        logger.debug("MainWindow: Received adsorbate_sets_structure_changed signal.")
        num_sets = len(self.app_controller.adsorbate_spot_sets) if self.app_controller else 0
        if hasattr(self, 'fft_analysis_panel_widget') and self.fft_analysis_panel_widget:
            set_names_for_combo = [f"Set {i+1}" for i in range(num_sets)]
            current_set_text = ""
            if 0 <= self.app_controller.current_adsorbate_set_index < num_sets:
                current_set_text = f"Set {self.app_controller.current_adsorbate_set_index + 1}"
            elif num_sets > 0:
                current_set_text = "Set 1"
            self.fft_analysis_panel_widget.update_adsorbate_set_combo(set_names_for_combo, current_set_text)

        if hasattr(self, "overlay_binder") and self.overlay_binder:
            self.overlay_binder.refresh_all_sets_in_visualization()
            current_index = self.app_controller.current_adsorbate_set_index if self.app_controller else -1
            if current_index >= 0:
                self.overlay_binder.sync_checkboxes_for_set(current_index)
        self._on_spot_lists_or_params_changed()

    @pyqtSlot(str)
    def _handle_substrate_changed(self, substrate_name: str):
        """Handles the update of substrate in AppController."""
        logger.debug(f"MainWindow: Substrate changed to '{substrate_name}' via panel signal.")
        self.app_controller.last_selected_substrate = substrate_name 
        self.app_controller.custom_lattice_info = None
        self.display_image_data()

    @pyqtSlot(str)
    def _handle_refinement_method_changed_from_panel(self, method: str):
        """Handles the update of refinement method in AppController."""
        logger.debug(f"MainWindow: Refinement method changed to '{method}' via panel signal.")
        self.app_controller.set_spot_refinement_method(method)

    @pyqtSlot(int)
    def _handle_refinement_area_size_changed_from_panel(self, area_size: int):
        """Handles the update of refinement area size in AppController."""
        logger.debug(f"MainWindow: Refinement area size changed to {area_size} via panel signal.")
        self.app_controller.set_refinement_roi_size(area_size)

    @pyqtSlot()
    def _handle_custom_lattice_request(self):
        """Handles the request for custom lattice definition in AppController."""
        logger.debug("MainWindow: Custom lattice definition requested via panel signal.")
        if not CustomLatticeDialog:
            QMessageBox.critical(self, "Error", "CustomLatticeDialog is not available.")
            return

        dialog = CustomLatticeDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            custom_def = dialog.get_lattice_definition()
            self.app_controller.custom_lattice_info = custom_def
            if self.app_controller.custom_lattice_info and self.fft_analysis_panel_widget:
                new_name = self.app_controller.custom_lattice_info.get("name", "Custom")
                self.fft_analysis_panel_widget.set_substrate_combo_text(new_name)
                self.app_controller.last_selected_substrate = new_name
                logger.info(f"Custom lattice '{new_name}' defined and selected.")
                self.display_image_data()
            else:
                if self.fft_analysis_panel_widget:
                    self.fft_analysis_panel_widget.set_substrate_combo_text(self.app_controller.last_selected_substrate)
        else:
            logger.debug("Custom lattice definition dialog was cancelled.")
            if self.fft_analysis_panel_widget:
                self.fft_analysis_panel_widget.set_substrate_combo_text(self.app_controller.last_selected_substrate)

    
    @pyqtSlot(str)
    def _handle_current_adsorbate_set_changed_from_panel(self, set_name: str):
        """Handles the update of current adsorbate set in AppController."""
        found_idx = -1
        if hasattr(self, 'fft_analysis_panel_widget'):
            combo = self.fft_analysis_panel_widget.adsorbate_set_combo
            for i in range(combo.count()):
                if combo.itemText(i) == set_name and set_name != "<Add New Set...>":
                    found_idx = i
                    break

        if self.app_controller.current_adsorbate_set_index >= 0:
            expected_type = self.app_controller.adsorbate_expected_lattice_types.get(
                self.app_controller.current_adsorbate_set_index, ADSORBATE_LATTICE_TYPE_UNKNOWN
            )
            if hasattr(self, 'fft_analysis_panel_widget'):
                self.fft_analysis_panel_widget.set_expected_adsorbate_type(expected_type)
        
        if found_idx != -1:
            self.app_controller.set_current_adsorbate_set_by_index(found_idx)
            if hasattr(self, "overlay_binder") and self.overlay_binder:
                self.overlay_binder.handle_current_adsorbate_set_changed(found_idx)
            logger.info(f"MainWindow: Switched to adsorbate set '{set_name}' (Index: {self.app_controller.current_adsorbate_set_index}) via panel signal.")
        else:
             logger.warning(f"MainWindow: Could not map adsorbate set name '{set_name}' to an index.")


    @pyqtSlot(bool)
    def _handle_show_ideal_lattice_changed(self, is_visible: bool):
        """Handles the update of ideal lattice visibility in AppController."""
        logger.debug(f"MainWindow: Show ideal lattice changed to {is_visible} via panel signal.")
        self.app_controller.show_ideal_lattice = is_visible
        self.display_image_data()

    @pyqtSlot(bool)
    def _handle_substrate_spots_visibility_changed(self, is_visible: bool):
        """Handles the update of substrate spots visibility in AppController."""
        logger.debug(f"MainWindow: Substrate spots visibility changed to {is_visible} via panel.")
        self.app_controller.show_substrate_spots_markers = is_visible

    @pyqtSlot(bool)
    def _handle_adsorbate_spots_visibility_changed(self, is_visible: bool):
        """Handles the update of adsorbate spots visibility in AppController."""
        logger.debug(f"MainWindow: Adsorbate spots visibility changed to {is_visible} via panel.")
        self.app_controller.show_adsorbate_spots_markers = is_visible
    
    def request_spot_markers_update(self):
        """Requests the update of spot markers in VisualizationManager."""
        self.display_image_data()
        if hasattr(self, "overlay_binder") and self.overlay_binder:
            self.overlay_binder.refresh_all_sets_in_visualization()

    @pyqtSlot()
    def _handle_add_new_adsorbate_set_request(self):
        """Handles the request for adding a new adsorbate set in AppController."""
        logger.info("MainWindow: Add new adsorbate set requested via panel signal.")
        if hasattr(self, 'fft_analysis_panel_widget'):
            num_sets = len(self.app_controller.adsorbate_spot_sets)
            set_names_for_combo = [f"Set {i+1}" for i in range(num_sets)]
            new_set_name = f"Set {num_sets}"
            self.fft_analysis_panel_widget.update_adsorbate_set_combo(set_names_for_combo, new_set_name)
            
            new_set_idx = self.app_controller.current_adsorbate_set_index
            expected_type = self.app_controller.adsorbate_expected_lattice_types.get(new_set_idx, ADSORBATE_LATTICE_TYPE_UNKNOWN)
            self.fft_analysis_panel_widget.set_expected_adsorbate_type(expected_type)
        self.app_controller.add_new_adsorbate_set()

    @pyqtSlot(object)
    def _on_current_history_node_changed(self, current_node: Optional[HistoryNode]):
        """Handles the update of current history node in AppController."""
        logger.debug(f"MainWindow: Slot _on_current_history_node_changed received node: {current_node.node_id if current_node else 'None'}")

        if self.app_controller:
            if current_node and current_node.data_type == "FFT" and current_node.image_data is not None:
                self.app_controller.current_fft_data_shape = current_node.image_data.shape
                logger.debug(f"Updated app_controller.current_fft_data_shape to {current_node.image_data.shape}")
            else:
                self.app_controller.current_fft_data_shape = None
                logger.debug("Cleared app_controller.current_fft_data_shape (current node is not FFT).")
            
        self._update_action_states()

        self.display_image_data()

        if hasattr(self, 'metadata_widget') and self.metadata_widget and hasattr(self, 'history_manager'):
            self.metadata_widget.update_metadata(current_node, self.history_manager)

        if hasattr(self, 'fft_analysis_dock') and self.fft_analysis_dock.isVisible():
            if hasattr(self, 'fft_analysis_panel_widget') and self.fft_analysis_panel_widget and self.app_controller:
                logger.debug("MainWindow: FFT Analysis Panel is visible, refreshing its content.")
                self.fft_analysis_panel_widget.update_transform_results_display(
                    self.app_controller.substrate_transform_analysis_m2i
                )
                self.fft_analysis_panel_widget.update_substrate_real_space_display(
                    self.app_controller.substrate_real_space_results
                )
                current_ads_idx = self.app_controller.current_adsorbate_set_index
                ads_params = self.app_controller.adsorbate_real_space_results.get(current_ads_idx)
                self.fft_analysis_panel_widget.update_adsorbate_real_space_display(ads_params)
                
                expected_type = self.app_controller.adsorbate_expected_lattice_types.get(
                    current_ads_idx, ADSORBATE_LATTICE_TYPE_UNKNOWN # Imported constant
                )
                self.fft_analysis_panel_widget.set_expected_adsorbate_type(expected_type)
    @pyqtSlot()
    def open_file_dialog(self):
        """Opens the file dialog for loading STM files."""
        logger.debug("Open file dialog triggered.")
        file_filter = "STM Files (*.stp *.s94);;All Files (*)"
        start_dir = ""
        if self.app_controller.original_file_path:
            try:
                start_dir = os.path.dirname(self.app_controller.original_file_path)
            except Exception:
                pass
        if not start_dir: 
            start_dir = os.path.expanduser("~")

        file_path, _ = QFileDialog.getOpenFileName(self, "Open STM File", start_dir, file_filter)

        if file_path:
            logger.info(f"File selected by user: {file_path}")
            self.statusBar().showMessage(f"Loading file: {os.path.basename(file_path)}...")
            QApplication.processEvents()
            
            self.app_controller.load_file(file_path)
        else:
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
        else:
            self.spot_selection_mode = "Adsorbate"
            self.adsorbate_set_panel.setVisible(True)
            self.substrate_set_panel.setVisible(False)
            self._points_for_current_adsorbate_set = []
            if self.current_adsorbate_set_index < len(self.adsorbate_spot_sets):
                 self._points_for_current_adsorbate_set = list(self.adsorbate_spot_sets[self.current_adsorbate_set_index])
            logger.debug(f"Spot selection mode: Adsorbate, Set Index: {self.current_adsorbate_set_index}")

    @pyqtSlot(str)
    def _on_adsorbate_set_combo_changed(self, text: str):
        """Handles selection or addition of an adsorbate set."""
        if text == "<Add New Set...>":
            new_set_name = f"Set {len(self.adsorbate_spot_sets) + 1}"
            self.adsorbate_spot_sets.append([])
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
            self.adsorbate_spot_sets[self.current_adsorbate_set_index] = []
            self._points_for_current_adsorbate_set = []
        else:
            logger.warning("No valid adsorbate set selected to reselect.")

    @pyqtSlot()
    def _on_clear_all_adsorbate_sets_clicked(self):
        """Clears all adsorbate spot sets."""
        logger.info("MainWindow: Clearing all adsorbate spot sets button clicked.")
        self.app_controller.clear_all_adsorbate_sets()
        if hasattr(self, 'fft_analysis_panel_widget'):
            self.fft_analysis_panel_widget.update_adsorbate_set_combo(["Set 1"], "Set 1")

    @pyqtSlot(str)
    def _handle_spot_selection_mode_changed_from_panel(self, mode: str):
        """Handles the change of spot selection mode in AppController."""
        logger.debug(f"MainWindow: Spot selection mode changed to '{mode}' via panel.")
        self.app_controller.set_spot_selection_mode(mode)


    @pyqtSlot()
    def _on_visibility_checkbox_changed(self):
        """Slot for all visibility checkboxes."""
        logger.debug("Visibility checkbox changed, updating markers and ideal lattice.")
        self.display_image_data()

    @pyqtSlot(str)
    def on_substrate_combo_changed(self, selected_text: str):
        """Handles selection change in the substrate combobox."""
        logger.debug(f"Substrate selection changed to: {selected_text}")

        if selected_text == self.custom_option_text:
            dialog = CustomLatticeDialog(self)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                self.custom_lattice_info = dialog.get_lattice_definition()
                if self.custom_lattice_info:
                    self.last_selected_substrate = self.custom_option_text
                    logger.info(f"Custom lattice defined and selected: {self.custom_lattice_info.get('name')}")
                else: 
                    self.custom_lattice_info = None
                    self.substrate_combo.setCurrentText(self.last_selected_substrate)
            else:
                self.custom_lattice_info = None
                self.substrate_combo.setCurrentText(self.last_selected_substrate)
        else:
            self.custom_lattice_info = None
            self.last_selected_substrate = selected_text

        self.display_image_data()

    def open_fft_dialog(self):
        """Open the FFT dialog via the dialog coordinator."""
        if hasattr(self, "dialog_coordinator") and self.dialog_coordinator:
            self.dialog_coordinator.open_fft_dialog()


    def open_gaussian_sharpening_dialog(self):
        """Opens the Gaussian sharpening dialog for applying Gaussian sharpening."""
        self._helper_open_processing_dialog(GaussianSharpeningDialog, "apply_gaussian_sharpening")

    def open_bm3d_dialog(self):
        """Opens the BM3D dialog for applying BM3D denoising."""
        def bm3d_checks():
            try: import bm3d; return True
            except ImportError: QMessageBox.critical(self,"Missing Dependency","BM3D package needed."); return False
        self._helper_open_processing_dialog(BM3DDialog, "apply_bm3d_denoising", dialog_specific_checks=bm3d_checks)

    def open_nlmeans_dialog(self):
        """Opens the NL-Means dialog for applying NL-Means denoising."""
        self._helper_open_processing_dialog(NLMeansDialog, "apply_nlmeans_denoising")

    def open_median_filter_dialog(self):
        """Opens the median filter dialog for applying median filter."""
        self._helper_open_processing_dialog(MedianFilterDialog, "apply_median_filter")

    def open_plane_leveling_dialog(self):
        """Opens the plane leveling dialog for applying plane leveling."""
        self._helper_open_processing_dialog(PlaneLevelingDialog, "apply_plane_leveling")


    @pyqtSlot(QListWidgetItem, QListWidgetItem)
    def on_history_selection_changed(self, current_item: QListWidgetItem, previous_item: QListWidgetItem):
        """Slot called when the selection in the history list changes."""
        if current_item:
            node_id = current_item.data(Qt.ItemDataRole.UserRole)
            if self.history_manager and node_id != self.history_manager.current_node_id: # Additional condition to avoid unnecessary callbacks
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
        """Opens the Gaussian blur dialog for applying Gaussian blur."""
        self._helper_open_processing_dialog(GaussianBlurDialog, "apply_gaussian_blur")

    def display_image_data(self):
        """Displays the image data in the visualization manager."""
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
        selected_substrate = "None"
        panel_custom_text = ""

        substrate_spots_to_draw = self.app_controller.displayable_fitted_substrate_spots_on_fft
        corrected_adsorbate_sets_ideal_sys = self.app_controller.corrected_adsorbate_spot_sets

        show_substrate_markers = True
        show_adsorbate_markers = True
        if self.app_controller:
            show_substrate_markers = self.app_controller.show_substrate_spots_markers
            show_adsorbate_markers = self.app_controller.show_adsorbate_spots_markers

        if hasattr(self, 'fft_analysis_panel_widget') and self.fft_analysis_panel_widget is not None:
            show_ideal_lattice = self.fft_analysis_panel_widget.is_show_ideal_lattice_checked() 
            selected_substrate = self.fft_analysis_panel_widget.get_current_substrate() 
            panel_custom_text = self.fft_analysis_panel_widget.custom_option_text
        
        self.visualization_manager.update_view(
            current_node,
            show_ideal_lattice, 
            selected_substrate, 
            self.app_controller.custom_lattice_info, 
            panel_custom_text,  
            substrate_spots_to_draw, 
            show_substrate_markers,
            corrected_adsorbate_sets_ideal_sys,
            show_adsorbate_markers
        )

        if current_node and current_node.data_type == "FFT":
            substrate_pairs = self.app_controller.substrate_spot_pairs if show_substrate_markers else []
            self.visualization_manager.update_substrate_spot_pairs(substrate_pairs)

            if show_adsorbate_markers:
                for set_index, pair_list in self.app_controller.adsorbate_spot_pairs.items():
                    self.visualization_manager.update_adsorbate_spot_pairs(set_index, pair_list)
            else:
                self.visualization_manager.clear_adsorbate_layers()
        else:
            self.visualization_manager.update_substrate_spot_pairs([])
            self.visualization_manager.clear_adsorbate_layers()
        if hasattr(self, '_update_action_states'): self._update_action_states()
    
    @pyqtSlot(QPointF)
    def _on_fft_view_clicked_from_visualizer(self, mapped_data_pos: QPointF):
        """Handles the click event in the FFT view."""
        logger.debug(f"MainWindow: Received FFT click at data coords (kx, ky): ({mapped_data_pos.x():.2f}, {mapped_data_pos.y():.2f})")
        current_node = self.history_manager.get_current_node()
        if not (current_node and current_node.data_type == "FFT" and current_node.image_data is not None):
            logger.warning("_on_fft_view_clicked_from_visualizer: No valid FFT data node active.")
            return

        kx_from_signal, ky_from_signal = mapped_data_pos.x(), mapped_data_pos.y()
        kx_int, ky_int = int(round(kx_from_signal)), int(round(ky_from_signal))
        original_fft_data = current_node.image_data
        fft_data_rows_ky, fft_data_cols_kx = original_fft_data.shape
        if not (0 <= ky_int < fft_data_rows_ky and 0 <= kx_int < fft_data_cols_kx):
            logger.debug(f"Click data coords outside original FFT data bounds. Ignoring.")
            return

        center_yx_for_refinement = (ky_int, kx_int)
        refined_kx, refined_ky = kx_int, ky_int

        current_refinement_method = self.app_controller.spot_refinement_method
        current_refinement_radius = self.app_controller.refinement_roi_size // 2

        logger.debug(f"Refinement: Method='{current_refinement_method}', Radius for func={current_refinement_radius}, Click (ky,kx)=({ky_int},{kx_int})")
        if current_refinement_method == "Max Pixel":
            if PEAK_FITTING_AVAILABLE:
                refined_ky_temp, refined_kx_temp = find_max_pixel_in_roi(original_fft_data, center_yx_for_refinement, current_refinement_radius)
                refined_kx, refined_ky = int(refined_kx_temp), int(refined_ky_temp)
                logger.info(f"Max Pixel refined: (orig_kx={kx_int}, orig_ky={ky_int}) -> (ref_kx={refined_kx}, ref_ky={refined_ky})")
        elif current_refinement_method == "2D Gaussian Fit":
            if PEAK_FITTING_AVAILABLE:
                fit_result = fit_2d_gaussian_in_roi(original_fft_data, center_yx_for_refinement, current_refinement_radius)
                if fit_result:
                    refined_ky_float, refined_kx_float = fit_result
                    refined_kx, refined_ky = int(round(refined_kx_float)), int(round(refined_ky_float))
                    logger.info(f"2D Gaussian Fit refined: -> (ref_kx_float={refined_kx_float:.2f}, ref_ky_float={refined_ky_float:.2f}) -> int({refined_kx},{refined_ky})")
                else:
                    logger.warning("2D Gaussian Fit failed. Using rounded click position.")
            else:
                 logger.warning("Peak fitting (Gaussian) backend not available. Using rounded click.")

    @pyqtSlot()
    def _on_clear_substrate_spots_clicked(self):
        """Clears the substrate spots."""
        logger.info("MainWindow: Clearing substrate spots.")
        self.app_controller.clear_substrate_spots()


    @pyqtSlot()
    def _on_reselect_adsorbate_set_clicked(self):
        """Reselects the current adsorbate set."""
        logger.debug("MainWindow: Reselect current adsorbate set button clicked.")
        self.app_controller.reselect_current_adsorbate_set()


    def closeEvent(self, event):
        """Handle the event when the user tries to close the window."""
        logger.info("Close event triggered. Exiting application.")
        event.accept()

