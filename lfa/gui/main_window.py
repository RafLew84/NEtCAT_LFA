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
    QGroupBox, QFormLayout, QRadioButton
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

        # --- Spot Selection Attributes ---
        self.substrate_spots: List[Tuple[float, float]] = []
        self.adsorbate_spot_sets: List[List[Tuple[float, float]]] = [[]] # Start with one empty set
        self.current_adsorbate_set_index: int = 0
        self.spot_selection_mode: str = "Substrate" # "Substrate" or "Adsorbate"
        # self.spot_refinement_method: str = "Max Pixel" # For later phases
        self._points_for_current_adsorbate_set: List[Tuple[float, float]] = []

        # --- Visual Markers for Spots ---
        self.ideal_lattice_overlay_item: Optional['pg.ScatterPlotItem'] = None
        self.substrate_spot_markers: Optional['pg.ScatterPlotItem'] = None
        self.adsorbate_spot_set_markers: List['pg.ScatterPlotItem'] = []
        self._fft_mouse_click_connection = None

        self.setWindowTitle("Lattice Fourier Analyzer (LFA)")
        self.resize(1250, 800)

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
        # self.lattice_toolbar = QToolBar("Lattice Overlay")
        # self.lattice_toolbar.setMovable(False)
        # self.addToolBar(Qt.ToolBarArea.TopToolBarArea, self.lattice_toolbar)
        # self.lattice_toolbar.addWidget(QLabel("Substrate Overlay:"))
        self.substrate_combo = QComboBox()
        # Dodaj "None", predefiniowane i opcję "<Custom Define...>"
        self.predefined_substrates = sorted(KNOWN_LATTICES.keys())
        self.substrate_combo.addItem("None")
        self.substrate_combo.addItems(self.predefined_substrates)
        self.custom_option_text = "<Custom Define...>"
        self.substrate_combo.addItem(self.custom_option_text)
        # Podłącz sygnał zmiany wyboru do nowego slotu
        self.substrate_combo.currentTextChanged.connect(self.on_substrate_combo_changed)
        # self.lattice_toolbar.addWidget(self.substrate_combo)
        # self.lattice_toolbar.setVisible(False) # Pokaż tylko przy FFT
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

        # --- FFT Analysis Tools Dock Widget ---
        self.fft_analysis_dock = QDockWidget("FFT Analysis Tools", self)
        fft_analysis_widget = QWidget()
        fft_analysis_layout = QVBoxLayout(fft_analysis_widget)

        # --- Lattice Overlay Controls ---
        lattice_group = QGroupBox("Ideal Lattice Overlay")
        lattice_layout = QFormLayout() # QFormLayout dla par etykieta-kontrolka
        self.substrate_combo = QComboBox()
        substrates = ["None"] + sorted(KNOWN_LATTICES.keys())
        self.substrate_combo.addItems(substrates)
        self.custom_option_text = "<Custom Define...>" # Definicja dla spójności
        self.substrate_combo.addItem(self.custom_option_text)
        lattice_layout.addRow("Substrate:", self.substrate_combo)
        self.show_ideal_lattice_checkbox = QCheckBox("Show Ideal Lattice")
        self.show_ideal_lattice_checkbox.setChecked(True)
        self.show_ideal_lattice_checkbox.stateChanged.connect(self.on_ideal_lattice_visibility_changed)
        lattice_layout.addRow(self.show_ideal_lattice_checkbox)
        lattice_group.setLayout(lattice_layout)
        fft_analysis_layout.addWidget(lattice_group)
        # -----------------------------------

        # Zaktualizuj lub dodaj sekcję Action Buttons:
        action_buttons_layout = QHBoxLayout()
        self.clear_substrate_spots_button = QPushButton("Clear Substrate Spots")
        # NOWY Przycisk:
        self.reselect_substrate_spots_button = QPushButton("Reselect Substrate Spots")

        action_buttons_layout.addWidget(self.clear_substrate_spots_button)
        action_buttons_layout.addWidget(self.reselect_substrate_spots_button) # Dodaj nowy przycisk

        # --- Spot Selection Controls ---
        spot_selection_group = QGroupBox("Spot Selection")
        spot_selection_layout = QVBoxLayout()

        # Spot Type
        spot_type_layout = QHBoxLayout()
        self.rb_select_substrate = QRadioButton("Substrate"); self.rb_select_substrate.setChecked(True)
        self.rb_select_adsorbate = QRadioButton("Adsorbate")
        spot_type_layout.addWidget(self.rb_select_substrate); spot_type_layout.addWidget(self.rb_select_adsorbate)
        spot_selection_layout.addLayout(spot_type_layout)

        # Adsorbate Set Management
        self.adsorbate_set_panel = QWidget() # Panel do pokazywania/ukrywania
        adsorbate_set_layout = QFormLayout(self.adsorbate_set_panel) # Użyj QFormLayout
        adsorbate_set_layout.setContentsMargins(0,5,0,5)
        self.adsorbate_set_combo = QComboBox()
        self.adsorbate_set_combo.addItem("Set 1") # Zacznij z jednym zestawem
        self.adsorbate_set_combo.addItem("<Add New Set...>")
        adsorbate_set_layout.addRow("Current Set:", self.adsorbate_set_combo)
        # Przyciski dla zestawów adsorbatu
        adsorbate_buttons_layout = QHBoxLayout()
        self.reselect_adsorbate_set_button = QPushButton("Reselect Set")
        self.clear_all_adsorbate_sets_button = QPushButton("Clear All Sets")
        adsorbate_buttons_layout.addWidget(self.reselect_adsorbate_set_button)
        adsorbate_buttons_layout.addWidget(self.clear_all_adsorbate_sets_button)
        adsorbate_set_layout.addRow(adsorbate_buttons_layout) # Dodaj layout przycisków
        spot_selection_layout.addWidget(self.adsorbate_set_panel)
        self.adsorbate_set_panel.setVisible(False) # Ukryj na starcie

        # Spot Visibility Checkboxes
        self.show_substrate_spots_checkbox = QCheckBox("Show Substrate Spots"); self.show_substrate_spots_checkbox.setChecked(True)
        self.show_adsorbate_spots_checkbox = QCheckBox("Show Adsorbate Spots"); self.show_adsorbate_spots_checkbox.setChecked(True)
        spot_selection_layout.addWidget(self.show_substrate_spots_checkbox)
        spot_selection_layout.addWidget(self.show_adsorbate_spots_checkbox)
        # Dalsze kontrolki (Refinement, Clear Points) dodasz w Fazie B.2/B.3
        spot_selection_group.setLayout(spot_selection_layout)
        fft_analysis_layout.addWidget(spot_selection_group)
        # --------------------------------

        fft_analysis_layout.addStretch()
        self.fft_analysis_dock.setWidget(fft_analysis_widget)
        self.fft_analysis_dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.fft_analysis_dock)
        self.fft_analysis_dock.setVisible(False) # Ukryj na starcie

        # Dodaj akcję do menu View dla FFT Analysis Tools
        toggle_fft_tools_action = self.fft_analysis_dock.toggleViewAction()
        toggle_fft_tools_action.setText("FFT Analysis Tools")
        view_menu.addAction(toggle_fft_tools_action) # Dodaj do istniejącego view_menu
        # ------------------------------------------------

        # --- Połączenia Sygnałów dla Nowych Kontrolek ---
        self.substrate_combo.currentTextChanged.connect(self.on_substrate_combo_changed) # Slot do aktualizacji nakładki
        self.show_ideal_lattice_checkbox.stateChanged.connect(self.on_ideal_lattice_visibility_changed) # Również aktualizuje nakładkę

        self.rb_select_substrate.toggled.connect(self._on_spot_type_changed)
        # self.rb_select_adsorbate.toggled.connect(...) # Już obsłużone przez substrate

        self.adsorbate_set_combo.currentTextChanged.connect(self._on_adsorbate_set_combo_changed)
        self.reselect_adsorbate_set_button.clicked.connect(self._on_reselect_adsorbate_set_clicked)
        self.clear_all_adsorbate_sets_button.clicked.connect(self._on_clear_all_adsorbate_sets_clicked)

        self.show_substrate_spots_checkbox.stateChanged.connect(self._on_selected_spots_visibility_changed)
        self.show_adsorbate_spots_checkbox.stateChanged.connect(self._on_selected_spots_visibility_changed)
        # ---------------------------------------------------

        self._update_action_states()

        logger.info("Main window initialized with history panel.")

    def _clear_all_spot_markers_from_view(self, view_box: Optional[pg.ViewBox]):
        """Helper to remove all known spot markers from the view."""
        if not view_box:
            logger.debug("_clear_all_spot_markers_from_view: No ViewBox provided.")
            return
        # Substrate markers
        if hasattr(self, 'substrate_spot_markers') and self.substrate_spot_markers:
            try: view_box.removeItem(self.substrate_spot_markers)
            except RuntimeError: pass
            self.substrate_spot_markers = None
        # Adsorbate set markers
        if hasattr(self, 'adsorbate_spot_set_markers'):
            for marker_set in self.adsorbate_spot_set_markers:
                if marker_set:
                    try: view_box.removeItem(marker_set)
                    except RuntimeError: pass
            self.adsorbate_spot_set_markers = []
        # Current adsorbate preview markers
        if hasattr(self, 'current_adsorbate_preview_markers') and self.current_adsorbate_preview_markers:
            try: view_box.removeItem(self.current_adsorbate_preview_markers)
            except RuntimeError: pass
            self.current_adsorbate_preview_markers = None
        logger.debug("Cleared all user-selected spot markers from view.")

    def _update_spot_markers(self):
        logger.debug("_update_spot_markers called (Phase B.2.5 - currently a placeholder).")
        # W Fazie B.2.5 tutaj będzie logika rysowania markerów
        # Na razie upewnijmy się, że czyści stare markery, jeśli to nie jest FFT
        if self.current_node_id and self.current_node_id in self.history:
            if self.history[self.current_node_id].data_type != "FFT":
                if hasattr(self, 'image_view') and self.image_view:
                    self._clear_all_spot_markers_from_view(self.image_view.getView())
        elif hasattr(self, 'image_view') and self.image_view: # No current node
             self._clear_all_spot_markers_from_view(self.image_view.getView())

    def _update_selected_spots_display(self):
        logger.debug("_update_selected_spots_display called (Phase B.2.3 - currently a placeholder).")
        if hasattr(self, 'current_selection_label'):
             self.current_selection_label.setText(f"Mode: {self.spot_selection_mode}")
        if hasattr(self, 'selected_spots_display'):
             self.selected_spots_display.setPlainText("Coordinates will appear here in Phase B.2.3.")



    
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
        # self.metadata_widget.update_metadata(current_node_obj, self.history)

        if self.metadata_widget: 
            self.metadata_widget.update_metadata(current_node_obj, self.history)
        self._update_action_states()

    def _update_spot_markers(self):
        """Clears and redraws all selected spot markers based on checkbox visibility."""
        if not hasattr(self, 'image_view') or not self.image_view.getView(): return

        view = self.image_view.getView()
        # Clear previous substrate markers
        if hasattr(self, 'substrate_spot_markers') and self.substrate_spot_markers:
            try: view.removeItem(self.substrate_spot_markers)
            except RuntimeError: pass # Already removed
            self.substrate_spot_markers = None
        # Clear previous adsorbate markers
        if hasattr(self, 'adsorbate_spot_set_markers'):
            for marker_set in self.adsorbate_spot_set_markers:
                try: view.removeItem(marker_set)
                except RuntimeError: pass
            self.adsorbate_spot_set_markers = []

        # Redraw substrate spots if visible
        if hasattr(self, 'show_substrate_spots_checkbox') and self.show_substrate_spots_checkbox.isChecked() and self.substrate_spots:
            self.substrate_spot_markers = pg.ScatterPlotItem(
                pos=np.array(self.substrate_spots), symbol='o', size=10,
                pen=pg.mkPen('g', width=2), brush=pg.mkBrush(None) # Green circles
            )
            view.addItem(self.substrate_spot_markers)
            logger.debug(f"Redrew {len(self.substrate_spots)} substrate spots.")

        # Redraw adsorbate spots if visible
        if hasattr(self, 'show_adsorbate_spots_checkbox') and self.show_adsorbate_spots_checkbox.isChecked() and self.adsorbate_spot_sets:
            adsorbate_colors = ['b', 'c', 'm', (255, 165, 0)] # Blue, Cyan, Magenta, Orange
            for i, spot_set in enumerate(self.adsorbate_spot_sets):
                if spot_set: # Only draw if the set is not empty
                    color = adsorbate_colors[i % len(adsorbate_colors)]
                    markers = pg.ScatterPlotItem(
                        pos=np.array(spot_set), symbol='s', size=10, # Squares
                        pen=pg.mkPen(color, width=2), brush=pg.mkBrush(None)
                    )
                    view.addItem(markers)
                    self.adsorbate_spot_set_markers.append(markers)
            logger.debug(f"Redrew adsorbate spots for {len(self.adsorbate_spot_set_markers)} sets.")

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

        if hasattr(self, 'fft_analysis_dock'): self.fft_analysis_dock.setVisible(is_fft_data)
        # if hasattr(self, 'lattice_toolbar'): self.lattice_toolbar.setVisible(is_fft_data)

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

    # --- Sloty dla Kontrolek FFT Analysis ---
    @pyqtSlot(str)
    def on_substrate_combo_changed(self, selected_text: str): # text jest opcjonalny
        """Slot to refresh lattice overlay when substrate or visibility changes."""
        if self.current_node_id and self.history[self.current_node_id].data_type == "FFT":
            self.display_image_data() # display_image_data teraz obsługuje rysowanie/ukrywanie nakładki
        logger.debug("Substrate overlay settings changed.")
        if selected_text == self.custom_option_text:
            dialog = CustomLatticeDialog(self)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                self.custom_lattice_info = dialog.get_lattice_definition()
                if self.custom_lattice_info:
                    self.last_selected_substrate = self.custom_option_text
                    logger.info(f"Custom lattice defined: {self.custom_lattice_info.get('name')}")
                else:
                    self.custom_lattice_info = None
                    self.substrate_combo.setCurrentText(self.last_selected_substrate)
            else: # Custom dialog anulowany
                self.custom_lattice_info = None
                self.substrate_combo.setCurrentText(self.last_selected_substrate)
        else: # Wybrano predefiniowaną sieć lub "None"
            self.custom_lattice_info = None
            self.last_selected_substrate = selected_text

        self.display_image_data() # Zawsze odśwież po zmianie

    @pyqtSlot(int) # Akceptuje int ze stateChanged
    def on_ideal_lattice_visibility_changed(self, state: int):
        """Slot to refresh ideal lattice overlay when its visibility checkbox changes."""
        logger.debug(f"Ideal lattice visibility changed. New state: {state}")
        # Wystarczy odświeżyć główny widok, display_image_data sprawdzi stan checkboxa
        self.display_image_data()

    @pyqtSlot(bool)
    def _on_spot_type_changed(self, is_substrate_selected):
        """Handles change between Substrate and Adsorbate spot selection."""
        if is_substrate_selected:
            self.spot_selection_mode = "Substrate"
            self.adsorbate_set_panel.setVisible(False)
            logger.debug("Spot selection mode: Substrate")
        else: # Adsorbate selected
            self.spot_selection_mode = "Adsorbate"
            self.adsorbate_set_panel.setVisible(True)
            # Zresetuj/ustaw odpowiednio _points_for_current_adsorbate_set
            self._points_for_current_adsorbate_set = [] # Wyczyść przy zmianie na adsorbat
            if self.current_adsorbate_set_index < len(self.adsorbate_spot_sets):
                 # Jeśli przełączamy z powrotem, załaduj punkty z bieżącego zestawu
                 self._points_for_current_adsorbate_set = list(self.adsorbate_spot_sets[self.current_adsorbate_set_index])
            logger.debug(f"Spot selection mode: Adsorbate, Set Index: {self.current_adsorbate_set_index}")
        self._update_selected_spots_display() # Zaktualizuj wyświetlanie
        self._update_spot_markers() # Zaktualizuj markery

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
        self._update_spot_markers()
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
            self._update_spot_markers()
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

        self.adsorbate_set_combo.blockSignals(True)
        self.adsorbate_set_combo.clear()
        self.adsorbate_set_combo.addItem("Set 1")
        self.adsorbate_set_combo.addItem("<Add New Set...>")
        self.adsorbate_set_combo.setCurrentIndex(0)
        self.adsorbate_set_combo.blockSignals(False)

        self._update_selected_spots_display()
        self._update_spot_markers()
        self._update_action_states()

    @pyqtSlot()
    def _on_visibility_checkbox_changed(self):
        """Slot for all visibility checkboxes."""
        logger.debug("Visibility checkbox changed, updating markers and ideal lattice.")
        # Odśwież idealną sieć (jeśli FFT) i markery spotów
        self.display_image_data() # To odświeży idealną sieć, jeśli trzeba
        self._update_spot_markers() # To odświeży zaznaczone spoty

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

    @pyqtSlot(int) # Akceptuje int ze stateChanged
    def _on_selected_spots_visibility_changed(self, state: int):
        """Slot for selected spots visibility checkboxes."""
        logger.debug(f"Selected spots visibility checkbox changed. New state: {state}")
        # Odśwież tylko markery zaznaczonych spotów
        self._update_spot_markers()


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
        in the main window's ImageView.

        Handles:
        - Displaying STM or FFT data with appropriate orientation and scaling.
        - Showing/hiding the ideal lattice overlay on FFT images based on
          user selection and checkbox state.
        - Triggering an update of selected spot markers.
        - Showing/hiding UI elements relevant to FFT analysis.
        """
        # Ensure the main image view widget exists
        if not hasattr(self, 'image_view') or self.image_view is None:
            logger.error("MainWindow's ImageView widget is not available for displaying data.")
            return

        current_view_box = self.image_view.getView()
        image_item = self.image_view.getImageItem()

        if current_view_box is None or image_item is None:
            logger.error("ImageView's ViewBox or ImageItem is not available.")
            self._update_spot_markers() # Ensure markers are cleared if view is invalid
            self._update_action_states() # Update UI state
            return

        # --- Clear Previous Graphics Items Specific to This Method ---
        # Remove previous ideal lattice overlay if it exists
        if hasattr(self, 'ideal_lattice_overlay_item') and self.ideal_lattice_overlay_item is not None:
            try:
                current_view_box.removeItem(self.ideal_lattice_overlay_item)
                logger.debug("Removed previous ideal_lattice_overlay_item.")
            except RuntimeError:
                logger.debug("Previous ideal_lattice_overlay_item already removed (RuntimeError).")
            except Exception as e:
                logger.warning(f"Could not remove previous ideal_lattice_overlay_item: {e}")
            self.ideal_lattice_overlay_item = None

        scene = getattr(image_item, 'scene', lambda: None)()
        if scene and hasattr(self, '_fft_mouse_click_connection') and self._fft_mouse_click_connection:
            try:
                scene.sigMouseClicked.disconnect(self._fft_mouse_click_connection)
                logger.debug("Disconnected previous FFT mouse click handler.")
            except (TypeError, RuntimeError):
                logger.debug("Could not disconnect FFT mouse click (normal if not previously connected).")
            self._fft_mouse_click_connection = None

        # Note: Selected spot markers are handled by _update_spot_markers()

        # --- Set Default UI State for Non-FFT / No Node ---
        # Toolbars/Docks specific to FFT analysis should be hidden by default
        # Their visibility will be updated by _update_action_states later if an FFT node is active
        # if hasattr(self, 'lattice_toolbar'):
        #     self.lattice_toolbar.setVisible(False)
        # The fft_analysis_dock visibility is managed by _update_action_states and _set_current_node

        # --- Process Current Node ---
        if self.current_node_id and self.current_node_id in self.history:
            node_to_display = self.history[self.current_node_id]
            display_data = node_to_display.image_data # This is float32 (STM or scaled FFT)

            if display_data is not None:
                node_info = (f"Node: {self.current_node_id[:8]}... "
                             f"Desc: {node_to_display.get_display_text()} "
                             f"(Type: {node_to_display.data_type}, Shape: {display_data.shape})")
                logger.info(f"Displaying {node_info}")

                try:
                    view_box = self.image_view.getView()
                    image_item = self.image_view.getImageItem()

                    # --- Set Image Data based on Type ---
                    if node_to_display.data_type == "STM":
                        view_box.invertY(True) # Origin bottom-left for STM
                        image_item.setImage(display_data.astype(np.float32).T, autoLevels=True)
                        logger.debug("Set STM image with transpose and Y inversion.")
                    elif node_to_display.data_type == "FFT":
                        view_box.invertY(False) # Origin top-left/center for FFT
                        # FFT data (scaled magnitude) is displayed with transpose as per previous decision
                        image_item.setImage(display_data.astype(np.float32).T)
                        logger.debug("Set FFT image with transpose, no Y inversion.")

                        # Apply percentile levels for FFT visualization
                        try:
                            finite_data = display_data[np.isfinite(display_data)]
                            if finite_data.size > 0:
                                min_level = np.percentile(finite_data, 1.0)
                                max_level = np.percentile(finite_data, 99.5)
                                logger.debug(f"Setting main FFT view levels (1%, 99.5%): {min_level:.3f} - {max_level:.3f}")
                                image_item.setLevels([min_level, max_level])
                            else:
                                image_item.setAutoLevels() # Fallback
                        except Exception as e:
                            logger.error(f"Could not set percentile levels for FFT view: {e}")
                            image_item.setAutoLevels() # Fallback on error
                        
                                                # --- Connect Mouse Click Handler for FFT images ---
                        scene = getattr(image_item, 'scene', lambda: None)() # Re-fetch scene
                        if scene:
                             # Check if PYQTGRAPH_AVAILABLE and GraphicsSceneMouseEvent type is valid before connecting
                            #  if PYQTGRAPH_AVAILABLE: # GraphicsSceneMouseEvent import check done at top level
                            self._fft_mouse_click_connection = scene.sigMouseClicked.connect(self._on_fft_view_clicked)
                            logger.debug("Connected FFT mouse click handler.")
                            #  else:
                                #  logger.error("Cannot connect FFT mouse click handler: PyQtGraph types not available.")
                        else:
                             logger.error("Cannot connect FFT click: ImageItem scene is None.")
                        # -------------------------------------------------
                    else:
                        # Default/Unknown data type: display as STM
                        logger.warning(f"Unknown data type '{node_to_display.data_type}', displaying as STM.")
                        view_box.invertY(True)
                        image_item.setImage(display_data.astype(np.float32).T, autoLevels=True)
                    # --- End Set Image Data ---


                    # --- Draw Ideal Lattice Overlay for FFT Data ---
                    if node_to_display.data_type == "FFT" and LATTICE_ANALYSIS_AVAILABLE:
                        # self.lattice_toolbar.setVisible(True)
                        # Visibility of toolbar is now handled in _update_action_states
                        # self.lattice_toolbar.setVisible(True)

                        # Check if the "Show Ideal Lattice" checkbox is checked
                        if hasattr(self, 'show_ideal_lattice_checkbox') and \
                           self.show_ideal_lattice_checkbox.isChecked():
                            selected_substrate_text = self.substrate_combo.currentText()
                            lattice_info_to_use: Optional[Union[str, Dict]] = None

                            if selected_substrate_text == self.custom_option_text and \
                               hasattr(self, 'custom_lattice_info') and self.custom_lattice_info:
                                lattice_info_to_use = self.custom_lattice_info
                            elif selected_substrate_text != "None" and selected_substrate_text != self.custom_option_text:
                                lattice_info_to_use = selected_substrate_text # Pass name to get_reciprocal_points

                            if lattice_info_to_use:
                                # Find root node for calibration data
                                root_node = node_to_display
                                visited = {node_to_display.node_id}
                                for _ in range(100): # Safety break for deep histories
                                    if not root_node.parent_id or root_node.parent_id not in self.history or root_node.parent_id in visited:
                                        break
                                    visited.add(root_node.parent_id)
                                    root_node = self.history[root_node.parent_id]

                                if root_node.operation_name == "Original":
                                    orig_params = root_node.parameters
                                    Lx = orig_params.get("size_nm_x")
                                    Ly = orig_params.get("size_nm_y")
                                    # N_rows, N_cols are for the FFT data itself
                                    N_rows_fft, N_cols_fft = display_data.shape

                                    if Lx and Ly and N_cols_fft > 0 and N_rows_fft > 0:
                                        ideal_points_g = get_reciprocal_points(lattice_info_to_use, max_hk=2)
                                        if ideal_points_g:
                                            pixel_coords = []
                                            # Center of the FFT image (N_rows_fft, N_cols_fft)
                                            row_c = N_rows_fft / 2.0
                                            col_c = N_cols_fft / 2.0
                                            for Gx, Gy in ideal_points_g:
                                                # If FFT is displayed as (ky, kx).T -> effectively (kx, ky) on screen
                                                # So, Gx (k-space x) maps to screen X (columns)
                                                # And Gy (k-space y) maps to screen Y (rows)
                                                # Lx, Ly from original image are total lengths for scaling
                                                # N_cols_fft, N_rows_fft are pixel dimensions of FFT array
                                                #
                                                # Original mapping before .T for display:
                                                # col_pixel_raw = Gx * Lx + col_c_orig_fft_shape
                                                # row_pixel_raw = Gy * Ly + row_c_orig_fft_shape
                                                #
                                                # With .T for display: FFT axes are swapped relative to array indices
                                                # display_data.T means:
                                                # - original FFT rows (ky) become display columns (visual X)
                                                # - original FFT columns (kx) become display rows (visual Y)
                                                #
                                                # So, Gx (which corresponds to FFT columns before .T) maps to Y-axis of display.
                                                # And Gy (which corresponds to FFT rows before .T) maps to X-axis of display.
                                                display_col_pixel = Gy * Ly + col_c # Mapped to screen X
                                                display_row_pixel = Gx * Lx + row_c # Mapped to screen Y

                                                pixel_coords.append({'pos': (display_col_pixel, display_row_pixel),
                                                                     'symbol': 'o', 'size': 7,
                                                                     'pen': pg.mkPen('r', width=1.5), 'brush': pg.mkBrush(None)})
                                            if pixel_coords:
                                                self.ideal_lattice_overlay_item = pg.ScatterPlotItem(pixel_coords)
                                                view_box.addItem(self.ideal_lattice_overlay_item)
                                                logger.info(f"Displayed ideal lattice overlay for '{selected_substrate_text}'.")
                                    else:
                                        logger.warning("Could not get ideal reciprocal points.")
                                else:
                                    logger.warning("Cannot display lattice overlay: Missing calibration data (Lx, Ly) or FFT shape is invalid.")
                            else:
                                logger.warning("Cannot display lattice overlay: Could not trace back to original image node.")
                        # else: "Show Ideal Lattice" checkbox is unchecked or no substrate selected
                    # --- End Lattice Overlay Logic ---

                    # Always call this to update selected spot markers based on their visibility checkboxes
                    self._update_spot_markers()

                    # Adjust view range after setting image and all overlays
                    view_box.autoRange()

                except Exception as e:
                    logger.exception(f"Error setting image or overlays in MainWindow: {e}")
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
            self.image_view.clear()
            # Hide FFT-specific UI elements if no node is active
            # if hasattr(self, 'lattice_toolbar'):
            #     self.lattice_toolbar.setVisible(False)
            if hasattr(self, 'fft_analysis_dock'):
                self.fft_analysis_dock.setVisible(False)


        # --- SLOT do obsługi kliknięć na obrazie FFT ---
    @pyqtSlot(object) # Use 'object' for the decorator for robustness
    def _on_fft_view_clicked(self, event): # Python type hint can be more specific if imported
        """
        Handles mouse clicks on the FFT image for spot selection.
        Phase B.2.1: Converts click coordinates to original FFT data coordinates and logs them.
        """
        # if not PYQTGRAPH_AVAILABLE:
        #     logger.warning("Ignoring click: pyqtgraph not available.")
        #     return

        # Check if the event object has the methods we expect (duck typing)
        if not all(hasattr(event, attr) for attr in ['button', 'scenePos', 'accept']):
            logger.warning(f"Click ignored: event object of type {type(event)} missing required attributes.")
            return

        # Only process if current node is FFT
        if not (self.current_node_id and
                self.current_node_id in self.history and
                self.history[self.current_node_id].data_type == "FFT"):
            logger.debug("_on_fft_view_clicked: Not an FFT image currently displayed, ignoring click.")
            return

        # Only process left clicks
        if event.button() != Qt.MouseButton.LeftButton:
            logger.debug(f"_on_fft_view_clicked: Ignored button {event.button()}.")
            return

        # Ensure ImageView and its ImageItem are available
        if not hasattr(self, 'image_view') or self.image_view is None or \
           self.image_view.getImageItem() is None or \
           self.image_view.getImageItem().image is None:
            logger.warning("_on_fft_view_clicked: ImageView, ImageItem or its image data is not available.")
            return

        img_item = self.image_view.getImageItem()

        # Convert click position from scene coordinates to image data coordinates
        # mapToData expects coordinates in the item's local coordinate system
        # event.scenePos() gives coordinates in the scene's coordinate system
        # We need to map from scene to the item first
        pos_in_item_coords = img_item.mapFromScene(event.scenePos())
        pos_data = img_item.mapToData(pos_in_item_coords)

        if pos_data is None:
            logger.debug("_on_fft_view_clicked: Click mapped to None (likely outside image data in item).")
            return

        # Coordinates are now relative to the displayed image data (which is FFT_data.T)
        # Displayed X (pos_data.x()) is original FFT ky
        # Displayed Y (pos_data.y()) is original FFT kx
        kx_original_fft_coord = round(pos_data.x())
        ky_original_fft_coord = round(pos_data.y())

        # Validate coordinates against the *original* FFT data dimensions (before transpose)
        original_fft_data = self.history[self.current_node_id].image_data
        if original_fft_data is None:
            logger.error("_on_fft_view_clicked: Original FFT data is None in history node.")
            return
        # Original shape: (rows_ky, cols_kx)
        fft_data_rows_ky, fft_data_cols_kx = original_fft_data.shape

        if not (0 <= int(kx_original_fft_coord) < fft_data_cols_kx and \
                0 <= int(ky_original_fft_coord) < fft_data_rows_ky):
            logger.debug(f"FFT click original coords (kx={kx_original_fft_coord}, ky={ky_original_fft_coord}) "
                         f"is outside original FFT data bounds ({fft_data_cols_kx}, {fft_data_rows_ky}). "
                         f"Displayed click was (col:{pos_data.x():.1f}, row:{pos_data.y():.1f})")
            return

        final_point_kx_ky = (int(kx_original_fft_coord), int(ky_original_fft_coord))
        logger.info(f"FFT image clicked. Mapped to original FFT data coords (kx, ky): {final_point_kx_ky}. "
                    f"Current selection mode: {self.spot_selection_mode}")

        # --- LOGIKA DODAWANIA PUNKTÓW (Faza B.2.2) ---
        if self.spot_selection_mode == "Substrate":
            # Ustalmy maksymalną liczbę punktów podłoża, np. 8
            MAX_SUBSTRATE_SPOTS = 8
            if len(self.substrate_spots) < MAX_SUBSTRATE_SPOTS:
                if final_point_kx_ky not in self.substrate_spots: # Unikaj duplikatów
                    self.substrate_spots.append(final_point_kx_ky)
                    logger.debug(f"Added to substrate_spots: {final_point_kx_ky}. Count: {len(self.substrate_spots)}")
                else:
                    logger.debug(f"Point {final_point_kx_ky} already in substrate_spots.")
            else:
                QMessageBox.information(self, "Limit Reached", f"Maximum number of substrate spots ({MAX_SUBSTRATE_SPOTS}) selected.")
        elif self.spot_selection_mode == "Adsorbate":
            # Użytkownik wybiera punkty dla self.adsorbate_spot_sets[self.current_adsorbate_set_index]
            # Zbieramy je najpierw w self._points_for_current_adsorbate_set
            # Ustalmy, że chcemy zbierać dowolną liczbę punktów dla adsorbatu
            # (np. minimum 2, aby zdefiniować wektory, ale bez sztywnego górnego limitu)
            MIN_ADSORBATE_SPOTS_FOR_LATTICE = 2 # Minimum do zdefiniowania sieci (np. + początek układu)
                                            # lub 6 jeśli użytkownik chce zaznaczyć heksagon

            current_set_list = self.adsorbate_spot_sets[self.current_adsorbate_set_index]
            if final_point_kx_ky not in current_set_list: # Unikaj duplikatów w zestawie
                current_set_list.append(final_point_kx_ky)
                logger.debug(f"Added to adsorbate_spot_sets[{self.current_adsorbate_set_index}]: {final_point_kx_ky}. Current set count: {len(current_set_list)}")
            else:
                logger.debug(f"Point {final_point_kx_ky} already in current adsorbate set.")
        # ------------------------------------------------

        self._update_selected_spots_display() # Aktualizuj UI (Faza B.2.3)
        self._update_spot_markers()           # Aktualizuj markery (Faza B.2.5)
        self._update_action_states()          # Aktualizuj stan przycisków

        event.accept() # Consume the event


    # --- Placeholder Slots dla Przycisków Czyszczenia (Faza B.2.4) ---
    @pyqtSlot()
    def _on_clear_substrate_spots_clicked(self):
        logger.info("Clearing substrate spots (Phase B.2.4 - Placeholder).")
        self.substrate_spots = []
        self._update_spot_markers()
        self._update_selected_spots_display()

    @pyqtSlot()
    def _on_clear_last_adsorbate_point_clicked(self):
        logger.info("Clearing last adsorbate point (Phase B.2.4 - Placeholder).")
        if self.spot_selection_mode == "Adsorbate" and self._points_for_current_adsorbate_set:
            self._points_for_current_adsorbate_set.pop()
        self._update_spot_markers()
        self._update_selected_spots_display()
        self._update_action_states()



    def closeEvent(self, event):
        """Handle the event when the user tries to close the window."""
        logger.info("Close event triggered. Exiting application.")
        event.accept()