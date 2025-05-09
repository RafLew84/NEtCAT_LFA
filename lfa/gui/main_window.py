# lfa/gui/main_window.py
"""
Defines the main window for the Lattice Fourier Analyzer (LFA) application.
"""

import logging
import os
import numpy as np
from typing import Optional, Dict

from PyQt6.QtWidgets import (
    QMainWindow, QVBoxLayout, QWidget, QFileDialog, QMessageBox, QApplication, 
    QDialog, QHBoxLayout, QSplitter, QListWidget, QListWidgetItem, QDockWidget,
    QComboBox, QToolBar, QToolButton, QLabel, QLineEdit, QPushButton, QTextEdit
)
from PyQt6.QtGui import QAction, QIcon
from PyQt6.QtCore import Qt, pyqtSlot

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

try:
    from .preprocessing_dialogs import (GaussianBlurDialog, PlaneLevelingDialog, 
    MedianFilterDialog, NLMeansDialog, BM3DDialog, GaussianSharpeningDialog)
    from .fft_dialog import FFTDialog
except ImportError:
    GaussianBlurDialog = None
    PlaneLevelingDialog = None
    MedianFilterDialog = None
    NLMeansDialog = None
    BM3DDialog = None
    GaussianSharpeningDialog = None
    FFTDialog = None
    logging.warning("Could not import preprocessing dialogs. Preprocessing options may be unavailable.")

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

        # --- History Management ---
        self.history: Dict[str, HistoryNode] = {}
        self.current_node_id: Optional[str] = None
        self.original_file_path: Optional[str] = None

        self.setWindowTitle("Lattice Fourier Analyzer (LFA)")
        self.resize(1000, 700)

        # --- Central Widget and Layout ---
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # --- History List Widget (Left Panel) ---
        self.history_list_widget = QListWidget()
        self.history_list_widget.currentItemChanged.connect(self.on_history_selection_changed)
        self.history_list_widget.setMaximumWidth(250)
        splitter.addWidget(self.history_list_widget)

        # --- Image View Widget (Right Panel) ---
        image_view_container = QWidget()
        image_view_layout = QVBoxLayout(image_view_container)
        image_view_layout.setContentsMargins(0,0,0,0)

        if pg:
            pg.setConfigOption('background', 'w')
            pg.setConfigOption('foreground', 'k')
            self.image_view = pg.ImageView(self)
            image_view_layout.addWidget(self.image_view)
        else:
            self.image_view = None
            logger.error("Cannot create ImageView because PyQtGraph is not available.")
        
        splitter.addWidget(image_view_container)
        splitter.setSizes([250, 950])
        main_layout.addWidget(splitter)

        # --- Lettice ---
        self.lattice_toolbar = QToolBar("Lattice Overlay")
        self.lattice_toolbar.setMovable(False)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, self.lattice_toolbar)
        self.lattice_toolbar.addWidget(QLabel("Substrate Overlay:"))
        self.substrate_combo = QComboBox()
        # Dodaj "None", predefiniowane i opcję "<Custom Define...>"
        self.predefined_substrates = sorted(KNOWN_LATTICES.keys())
        self.substrate_combo.addItem("None")
        self.substrate_combo.addItems(self.predefined_substrates)
        self.custom_option_text = "<Custom Define...>"
        self.substrate_combo.addItem(self.custom_option_text)
        # Podłącz sygnał zmiany wyboru do nowego slotu
        self.substrate_combo.currentTextChanged.connect(self.on_substrate_combo_changed)
        self.lattice_toolbar.addWidget(self.substrate_combo)
        self.lattice_toolbar.setVisible(False) # Pokaż tylko przy FFT
        # ---------------------------------------

        # --- Menu Bar ---
        self.create_menus()

        # --- Status Bar ---
        self.statusBar().showMessage("Ready - Load an image using File -> Open")

        self.metadata_dock = QDockWidget("Metadata", self)
        self.metadata_widget = MetadataWidget(self) 
        self.metadata_dock.setWidget(self.metadata_widget)
        self.metadata_dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.metadata_dock)
        view_menu = self.menuBar().addMenu("&View")
        toggle_metadata_action = self.metadata_dock.toggleViewAction()
        toggle_metadata_action.setText("Metadata Panel")
        toggle_metadata_action.setStatusTip("Show/hide the metadata panel")
        view_menu.addAction(toggle_metadata_action)

        self._update_action_states()

        logger.info("Main window initialized with history panel.")
    
    def _clear_history(self):
        """Clears the history tree and list widget."""
        self.history.clear()
        self.current_node_id = None
        self.history_list_widget.clear()
        if self.image_view:
            self.image_view.clear()
        logger.info("History cleared.")
        self._update_action_states()
    
    def _add_history_node(self, node: HistoryNode):
        """Adds a node to the history dict and the list widget."""
        if not node or not node.node_id:
            return
        self.history[node.node_id] = node
        item = QListWidgetItem(node.get_display_text())
        item.setData(Qt.ItemDataRole.UserRole, node.node_id)
        self.history_list_widget.addItem(item)
        logger.debug(f"Added history node: {node.get_display_text()} (ID: {node.node_id})")
        return item
    
    def _set_current_node(self, node_id: Optional[str]):
        """Sets the current node ID and updates selection in the list."""
        if node_id not in self.history and node_id is not None:
            logger.error(f"Cannot set current node: ID {node_id} not found in history.")
            return

        self.current_node_id = node_id
        print(f"Current history node set to: {node_id}")
        logger.info(f"Current history node set to: {node_id}")

        self.history_list_widget.blockSignals(True)
        found_item = None
        for i in range(self.history_list_widget.count()):
            item = self.history_list_widget.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == node_id:
                item.setSelected(True)
                self.history_list_widget.setCurrentItem(item)
                found_item = item
                break
        if not found_item and self.history_list_widget.count() > 0:
            pass
        self.history_list_widget.blockSignals(False)

        self.display_image_data()
        current_node_obj = self.history.get(self.current_node_id)
        self.metadata_widget.update_metadata(current_node_obj, self.history)
        self._update_action_states()

    def create_menus(self):
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
        """Enables/disables actions based on the current state."""
        has_node = self.current_node_id is not None and self.current_node_id in self.history
        is_stm_data = False
        is_fft_data = False
        if has_node:
             current_node_data_type = self.history[self.current_node_id].data_type
             is_stm_data = (current_node_data_type == "STM")
             is_fft_data = (current_node_data_type == "FFT")
        # self.gaussian_blur_action.setEnabled(has_node)
        # self.plane_level_action.setEnabled(has_node)
        # self.median_filter_action.setEnabled(has_node)
        # self.nlmeans_action.setEnabled(has_node)
        # self.bm3d_action.setEnabled(has_node)
        # self.gaussian_sharpen_action.setEnabled(has_node)

        if hasattr(self, 'gaussian_blur_action'): self.gaussian_blur_action.setEnabled(has_node)
        if hasattr(self, 'plane_level_action'): self.plane_level_action.setEnabled(has_node)
        if hasattr(self, 'median_filter_action'): self.median_filter_action.setEnabled(has_node)
        if hasattr(self, 'nlmeans_action'): self.nlmeans_action.setEnabled(has_node)
        if hasattr(self, 'bm3d_action'): self.bm3d_action.setEnabled(has_node)
        if hasattr(self, 'gaussian_sharpen_action'): self.gaussian_sharpen_action.setEnabled(has_node)

        # FFT action enabled if any image is loaded (można by ograniczyć tylko do STM)
        if hasattr(self, 'fft_action'): self.fft_action.setEnabled(has_node)

    @pyqtSlot()
    def open_file_dialog(self):
        logger.debug("Open file dialog triggered.")
        file_filter = "STM Files (*.stp *.s94);;All Files (*)"
        start_dir = ""
        if self.current_node_id and self.current_node_id in self.history:
            curr = self.history[self.current_node_id]
            while curr.parent_id and curr.parent_id in self.history:
                curr = self.history[curr.parent_id]
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
                self._clear_history()

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
                root_item = self._add_history_node(root_node)
                self._set_current_node(root_node.node_id)
                # self.history_list_widget.setCurrentItem(root_item)

                logger.info("File loaded successfully and history initialized.")
                self.statusBar().showMessage(f"Loaded: {os.path.basename(file_path)}", 5000)
                self.setWindowTitle(f"LFA - {os.path.basename(file_path)}")
            else:
                self._clear_history()
                self.statusBar().showMessage("Failed to load file.", 5000)
                QMessageBox.warning(self, "Loading Error", f"Could not load file: {file_path}")
                self.setWindowTitle("Lattice Fourier Analyzer (LFA)")
        else:
            logger.debug("File dialog cancelled.")
            self.statusBar().showMessage("File open cancelled.", 3000)

    @pyqtSlot(str)
    def on_substrate_combo_changed(self, selected_text: str):
        """Handles selection change in the substrate combobox."""
        logger.debug(f"Substrate selection changed to: {selected_text}")

        if selected_text == self.custom_option_text:
            dialog = CustomLatticeDialog(self)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                self.custom_lattice_info = dialog.get_lattice_definition()
                if self.custom_lattice_info:
                    # Opcjonalnie: dodaj tymczasowo nazwę custom do listy,
                    # ale może być prościej po prostu użyć self.custom_lattice_info
                    # Możemy ustawić tekst na chwilę, ale to tylko wizualne
                    # self.substrate_combo.setItemText(self.substrate_combo.currentIndex(), self.custom_lattice_info.get("name", "Custom"))
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
        if self.current_node_id is None or self.current_node_id not in self.history:
            logger.warning("open_fft_dialog: No current node selected.")
            QMessageBox.warning(self, "No Image", "No data loaded or selected in history.")
            return

        if not FFTDialog:
            logger.error("open_fft_dialog: FFTDialog class is None (import failed?).")
            QMessageBox.critical(self, "Error", "FFTDialog class not available. Check imports and file.")
            return

        current_node = self.history[self.current_node_id]
        if current_node.image_data is None:
            logger.error(f"open_fft_dialog: Image data missing for node {self.current_node_id}.")
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
                    parent_id=self.current_node_id,
                    operation_name="FFT",
                    parameters=params, # Store window type, scaling mode, roi checkbox state
                    image_data=processed_fft_data, # Store the scaled magnitude (float)
                    data_type="FFT", # Mark data type as FFT
                    source_roi_slice=source_roi # Store the source ROI slice if used
                )

                # Add node to history and update UI
                new_item = self._add_history_node(new_node)
                self._set_current_node(new_node.node_id)
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
        if self.current_node_id is None or self.current_node_id not in self.history: QMessageBox.warning(self, "No Image", "..."); return
        if not GaussianSharpeningDialog: QMessageBox.critical(self, "Error", "GaussianSharpeningDialog not available."); return

        current_node = self.history[self.current_node_id]
        if current_node.image_data is None: QMessageBox.critical(self, "Internal Error", "..."); return
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
                    parent_id=self.current_node_id,
                    operation_name=op_name,
                    parameters=params,
                    image_data=processed_data,
                    data_type=parent_data_type,
                    source_roi_slice=final_roi_slice
                )
                new_item = self._add_history_node(new_node)
                self._set_current_node(new_node.node_id)
                self.history_list_widget.setCurrentItem(new_item)
                display_name = new_node.get_display_text()
                self.statusBar().showMessage(f"{display_name} applied.", 3000)
            else: logger.warning("Dialog accepted, but no processed data returned.")
        else: logger.info("Gaussian Sharpening dialog cancelled."); self.statusBar().showMessage("Sharpening cancelled.", 3000)


    @pyqtSlot()
    def open_bm3d_dialog(self):
        """Opens the dialog for applying BM3D Denoising."""
        if self.current_node_id is None or self.current_node_id not in self.history: QMessageBox.warning(self, "No Image", "..."); return
        if not BM3DDialog: QMessageBox.critical(self, "Error", "BM3DDialog not available."); return
        try: import bm3d
        except ImportError: QMessageBox.critical(self, "Missing Dependency", "The 'bm3d' package is required for this feature.\nPlease install it (pip install bm3d)."); return


        current_node = self.history[self.current_node_id]
        if current_node.image_data is None: QMessageBox.critical(self, "Internal Error", "..."); return
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
                    parent_id=self.current_node_id,
                    operation_name=op_name,
                    parameters=params,
                    image_data=processed_data,
                    data_type=parent_data_type,
                    source_roi_slice=final_roi_slice
                )
                new_item = self._add_history_node(new_node)
                self._set_current_node(new_node.node_id)
                self.history_list_widget.setCurrentItem(new_item)
                display_name = new_node.get_display_text()
                self.statusBar().showMessage(f"{display_name} applied.", 3000)
            else: logger.warning("Dialog accepted, but no processed data returned.")
        else: logger.info("BM3D dialog cancelled."); self.statusBar().showMessage("BM3D cancelled.", 3000)


    @pyqtSlot()
    def open_nlmeans_dialog(self):
        """Opens the dialog for applying NL-Means Denoising."""
        if self.current_node_id is None or self.current_node_id not in self.history: QMessageBox.warning(self, "No Image", "..."); return
        if not NLMeansDialog: QMessageBox.critical(self, "Error", "NLMeansDialog not available."); return

        current_node = self.history[self.current_node_id]
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
                    parent_id=self.current_node_id,
                    operation_name=op_name,
                    parameters=params,
                    image_data=processed_data,
                    data_type=parent_data_type,
                    source_roi_slice=final_roi_slice
                )
                new_item = self._add_history_node(new_node)
                self._set_current_node(new_node.node_id)
                self.history_list_widget.setCurrentItem(new_item)
                display_name = new_node.get_display_text()
                self.statusBar().showMessage(f"{display_name} applied.", 3000)
            else: logger.warning("Dialog accepted, but no processed data returned.")
        else: logger.info("NL-Means dialog cancelled."); self.statusBar().showMessage("NL-Means cancelled.", 3000)


    @pyqtSlot()
    def open_median_filter_dialog(self):
        """Opens the dialog for applying Median Filter."""
        if self.current_node_id is None or self.current_node_id not in self.history: QMessageBox.warning(self, "No Image", "..."); return
        if not MedianFilterDialog: QMessageBox.critical(self, "Error", "MedianFilterDialog not available."); return

        current_node = self.history[self.current_node_id]
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
                    parent_id=self.current_node_id,
                    operation_name=op_name,
                    parameters=params,
                    image_data=processed_data,
                    data_type=parent_data_type, 
                    source_roi_slice=final_roi_slice
                )
                new_item = self._add_history_node(new_node)
                self._set_current_node(new_node.node_id)
                self.history_list_widget.setCurrentItem(new_item)
                display_name = new_node.get_display_text() 
                self.statusBar().showMessage(f"{display_name} applied.", 3000)
            else: logger.warning("Dialog accepted, but no processed data returned.")
        else: logger.info("Median Filter dialog cancelled."); self.statusBar().showMessage("Median Filter cancelled.", 3000)


    @pyqtSlot()
    def open_plane_leveling_dialog(self):
        """Opens the dialog for applying Plane Leveling."""
        if self.current_node_id is None or self.current_node_id not in self.history: QMessageBox.warning(self, "No Image", "..."); return
        if not PlaneLevelingDialog: QMessageBox.critical(self, "Error", "PlaneLevelingDialog not available."); return

        current_node = self.history[self.current_node_id]
        if current_node.image_data is None: QMessageBox.critical(self, "Internal Error", "..."); return
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
                    parent_id=self.current_node_id,
                    operation_name=op_name,
                    parameters=params,
                    image_data=processed_data,
                    data_type=parent_data_type, 
                    source_roi_slice=final_roi_slice
                )
                new_item = self._add_history_node(new_node)
                self._set_current_node(new_node.node_id)
                self.history_list_widget.setCurrentItem(new_item)
                display_name = new_node.get_display_text() 
                self.statusBar().showMessage(f"{display_name} applied.", 3000)
            else: logger.warning("Dialog accepted, but no processed data returned.")
        else: logger.info("Plane Leveling dialog cancelled."); self.statusBar().showMessage("Plane Leveling cancelled.", 3000)


    @pyqtSlot(QListWidgetItem, QListWidgetItem)
    def on_history_selection_changed(self, current_item: QListWidgetItem, previous_item: QListWidgetItem):
        """Slot called when the selection in the history list changes."""
        if current_item:
            node_id = current_item.data(Qt.ItemDataRole.UserRole)
            if node_id != self.current_node_id:
                logger.info(f"History item selected: {current_item.text()}")
                self._set_current_node(node_id)

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
        if self.current_node_id is None or self.current_node_id not in self.history:
            QMessageBox.warning(self, "No Image", "Please load an image first.")
            return
        if not GaussianBlurDialog:
            QMessageBox.critical(self, "Error", "Gaussian Blur functionality is not available.")
            return
        print(f"Gaussian Blur: Current node ID: {self.current_node_id}")
        current_node = self.history[self.current_node_id]
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
                    parent_id=self.current_node_id,
                    operation_name=op_name,
                    parameters=params,
                    image_data=processed_data,
                    data_type=parent_data_type, 
                    source_roi_slice=final_roi_slice
                )
                new_item = self._add_history_node(new_node)
                self._set_current_node(new_node.node_id)
                self.history_list_widget.setCurrentItem(new_item)
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
        in the main window's ImageView, applying appropriate orientation
        and level scaling based on data type (STM or FFT).
        """
        if not self.image_view: logger.error("ImageView not available."); return

        if not hasattr(self, 'image_view') or self.image_view is None:
            logger.error("MainWindow's ImageView widget is not available.")
            return
        
        if hasattr(self, 'lattice_overlay_item') and self.lattice_overlay_item is not None:
            try:
                view = self.image_view.getView()
                view.removeItem(self.lattice_overlay_item)
                logger.debug("Removed previous lattice overlay item.")
            except Exception as e:
                logger.warning(f"Could not remove previous lattice overlay item: {e}")
            self.lattice_overlay_item = None
        
        self.lattice_toolbar.setVisible(False)

        if self.current_node_id and self.current_node_id in self.history:
            node_to_display = self.history[self.current_node_id]
            # Data is always float (STM or scaled FFT magnitude)
            display_data = node_to_display.image_data

            if display_data is not None:
                node_info = (f"Node: {self.current_node_id[:8]}... "
                             f"Desc: {node_to_display.get_display_text()} "
                             f"(Type: {node_to_display.data_type}, Shape: {display_data.shape})")
                logger.info(f"Displaying {node_info}")

                try:
                    view_box = self.image_view.getView()
                    image_item = self.image_view.getImageItem()

                    # --- Set Image based on Data Type ---
                    if node_to_display.data_type == "STM":
                        # Invert Y axis for STM (origin bottom-left)
                        view_box.invertY(True)
                        # Transpose STM data for display
                        image_item.setImage(display_data.astype(np.float32).T, autoLevels=True)
                        logger.debug("Set STM image with transpose and Y inversion.")

                    elif node_to_display.data_type == "FFT":
                        # Do NOT invert Y axis for FFT (origin top-left or center)
                        view_box.invertY(False)
                        # Display FFT data (already scaled magnitude)
                        # Apply transpose .T based on previous user feedback for desired orientation
                        image_item.setImage(np.fliplr(display_data.astype(np.float32).T)) # Using .T as requested
                        logger.debug("Set FFT image with transpose, no Y inversion.")

                        # --- Apply Percentile Levels for FFT ---
                        # This helps visualize log/power scaled data better
                        try:
                            finite_data = display_data[np.isfinite(display_data)]
                            if finite_data.size > 0:
                                # Use percentiles to ignore extreme noise/DC for level scaling
                                min_level = np.percentile(finite_data, 1.0)
                                max_level = np.percentile(finite_data, 99.5)
                                logger.debug(f"Setting main FFT view levels (1%, 99.5%): {min_level:.3f} - {max_level:.3f}")
                                image_item.setLevels([min_level, max_level])
                            else:
                                # Fallback if no finite data (unlikely)
                                logger.warning("No finite data in FFT image for level calculation, using autoLevels.")
                                image_item.setAutoLevels()
                        except Exception as e:
                            logger.error(f"Could not set percentile levels for main FFT view: {e}")
                            image_item.setAutoLevels() # Fallback on error
                        # --------------------------------------
                    else:
                        logger.warning(f"Unknown data type '{node_to_display.data_type}', displaying as STM.")
                        view_box.invertY(True)
                        image_item.setImage(display_data.astype(np.float32).T, autoLevels=True)
                    
                    if node_to_display.data_type == "FFT" and LATTICE_ANALYSIS_AVAILABLE:
                        self.lattice_toolbar.setVisible(True)
                        selected_substrate_text = self.substrate_combo.currentText()
                        lattice_info_to_use: Optional[Dict] = None

                        if selected_substrate_text == "None":
                            pass # Nie rób nic, nakładka już usunięta
                        elif selected_substrate_text == self.custom_option_text and self.custom_lattice_info:
                            lattice_info_to_use = self.custom_lattice_info
                            logger.info(f"Using custom lattice: {self.custom_lattice_info.get('name')}")
                        elif selected_substrate_text in KNOWN_LATTICES:
                            lattice_info_to_use = selected_substrate_text # Przekaż nazwę lub KNOWN_LATTICES[selected_substrate_text]
                            logger.info(f"Using predefined lattice: {selected_substrate_text}")

                        if lattice_info_to_use:
                            # ... (kod pobierania Lx, Ly, N_rows, N_cols z root_node - bez zmian) ...
                            root_node = node_to_display; visited = {node_to_display.node_id}
                            while root_node.parent_id and root_node.parent_id in self.history and root_node.parent_id not in visited: visited.add(root_node.parent_id); root_node = self.history[root_node.parent_id]
                            if root_node.operation_name == "Original":
                                orig_params = root_node.parameters; Lx = orig_params.get("size_nm_x"); Ly = orig_params.get("size_nm_y"); N_rows, N_cols = display_data.shape
                                if Lx and Ly and N_cols > 0 and N_rows > 0:
                                    # Przekaż lattice_info_to_use (string lub dict)
                                    ideal_points_g = get_reciprocal_points(lattice_info_to_use, max_hk=2)
                                    # ... (kod obliczania pixel_coords i rysowania ScatterPlotItem - bez zmian) ...
                                    if ideal_points_g:
                                        pixel_coords = []
                                        row_c = N_rows / 2.0; col_c = N_cols / 2.0
                                        for Gx, Gy in ideal_points_g:
                                            col_pixel = Gy * Ly + col_c; row_pixel = Gx * Lx + row_c
                                            pixel_coords.append({'pos': (col_pixel, row_pixel), 'symbol': 'o', 'size': 8, 'pen': pg.mkPen('r'), 'brush': pg.mkBrush(None)})
                                        if pixel_coords:
                                            self.lattice_overlay_item = pg.ScatterPlotItem(pixel_coords)
                                            self.image_view.getView().addItem(self.lattice_overlay_item) # Dodaj do ViewBox
                                            logger.info(f"Displayed overlay for '{selected_substrate_text}'.")
                                else: logger.warning("Overlay: Missing calibration.")
                            else: logger.warning("Overlay: Could not find root node.")
                    # -------------------------------------
                    # ------------------------------------

                    # Adjust view range after setting image
                    view_box.autoRange()

                except Exception as e:
                    logger.exception(f"Error setting image in MainWindow's ImageView: {e}")
                    QMessageBox.critical(self, "Display Error", f"Could not display image data for node {self.current_node_id}.\nError: {e}")
                    self.image_view.clear() # Clear view on error
            else:
                # Data in node is None
                logger.error(f"No image data found for selected history node {self.current_node_id}!")
                self.image_view.clear()
                self.statusBar().showMessage(f"Error: Image data missing for selected state.", 5000)
        else:
            # No node selected or history empty
            logger.debug("No current history node selected. Clearing image view.")
            self.lattice_toolbar.setVisible(False)
            self.image_view.clear()



    def closeEvent(self, event):
        """Handle the event when the user tries to close the window."""
        logger.info("Close event triggered. Exiting application.")
        event.accept()