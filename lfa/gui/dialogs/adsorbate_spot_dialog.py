# lfa/gui/dialogs/adsorbate_spot_dialog.py
import logging
from typing import List, Tuple, Optional, Dict, Any
import numpy as np

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QDialogButtonBox,
    QLabel, QListWidget, QAbstractItemView, QWidget, QGroupBox,
    QFormLayout, QRadioButton, QSpinBox, QCheckBox, QMessageBox,
    QGridLayout, QSplitter
)
from PyQt6.QtCore import Qt, pyqtSlot, QPointF
from PyQt6.QtGui import QPen, QVector3D # Dodano QVector3D

try:
    import pyqtgraph as pg
    import pyqtgraph.opengl as gl
    from pyqtgraph.opengl import GLViewWidget, GLSurfacePlotItem
    ImageItem = pg.ImageItem
    RectROI = pg.RectROI
    ScatterPlotItem = pg.ScatterPlotItem
    ViewBox = pg.ViewBox
    GraphicsLayoutWidget = pg.GraphicsLayoutWidget
    PYQTGRAPH_AVAILABLE = True
except ImportError: # pragma: no cover
    pg = None
    gl = None
    GLViewWidget = None
    GLSurfacePlotItem = None
    ImageItem = None
    RectROI = None
    ScatterPlotItem = None
    ViewBox = None
    GraphicsLayoutWidget = None
    PYQTGRAPH_AVAILABLE = False
    logging.error("AdsorbateSpotSelectionDialog: PyQtGraph or pyqtgraph.opengl not found.")

try:
    from scipy.optimize import curve_fit as scipy_curve_fit
    SCIPY_OPTIMIZE_AVAILABLE = True
except ImportError: # pragma: no cover
    logging.error("AdsorbateSpotSelectionDialog: SciPy (for curve_fit) not found.")
    SCIPY_OPTIMIZE_AVAILABLE = False
    def scipy_curve_fit(*args, **kwargs): raise ImportError("scipy.optimize.curve_fit is not available")

try:
    from ...analysis.peak_fitting import find_max_pixel_in_roi, fit_2d_gaussian_in_roi, _gaussian_2d, SCIPY_AVAILABLE
    from ...analysis.lattice import KNOWN_LATTICES, get_reciprocal_points # Może być potrzebne dla ideal_substrate_spots
    from ...core.history import HistoryNode
    from ...logic.history_manager import HistoryManager
    PEAK_FITTING_MODULE_AVAILABLE = True
except ImportError: # pragma: no cover
    PEAK_FITTING_MODULE_AVAILABLE = False; SCIPY_AVAILABLE = False; KNOWN_LATTICES = {}
    logging.error("AdsorbateSpotSelectionDialog: Could not import peak_fitting or lattice modules.")
    def find_max_pixel_in_roi(data, center, radius): return center
    def fit_2d_gaussian_in_roi(data, center, radius): return None
    def _gaussian_2d(*args, **kwargs): 
        raise ImportError("Gaussian 2D function is not available")


logger = logging.getLogger(__name__)

REFINEMENT_DIRECT_CLICK = "Direct Click"
REFINEMENT_MAX_PIXEL = "Max Pixel"
REFINEMENT_GAUSSIAN_FIT = "2D Gaussian Fit"

class AdsorbateSpotSelectionDialog(QDialog):
    def __init__(self,
                 fft_image_data: Optional[np.ndarray],
                 history_manager: HistoryManager, # Do pobrania Lx, Ly dla ideal_substrate_spots
                 current_fft_node_id: str,
                 current_adsorbate_spots: Optional[List[Tuple[float, float]]] = None,
                 adsorbate_set_index: int = 0,
                 default_refinement_method: str = REFINEMENT_DIRECT_CLICK,
                 default_refinement_roi_size: int = 5,
                 # Informacje o transformacji substratu
                 substrate_F_m2i: Optional[np.ndarray] = None,
                 substrate_t_m2i: Optional[np.ndarray] = None,
                 substrate_transform_analysis: Optional[Dict[str, Any]] = None,
                 # Piki referencyjne substratu (już w koordynatach pikselowych FFT)
                 ideal_substrate_spots_for_display_px: Optional[List[Tuple[float, float]]] = None,
                 fitted_substrate_spots_for_display_px: Optional[List[Tuple[float, float]]] = None,
                 parent=None):
        super().__init__(parent)

        if not PYQTGRAPH_AVAILABLE: # pragma: no cover
            err_layout = QVBoxLayout(self)
            err_layout.addWidget(QLabel("Critical Error: PyQtGraph library is not available.\nThis dialog cannot function."))
            self.setWindowTitle("Error")
            return

        self.setWindowTitle(f"Select Adsorbate Spots (Set {adsorbate_set_index + 1})")
        self.setMinimumSize(1200, 750)

        self.fft_data = fft_image_data
        self.history_manager = history_manager # Potrzebne do Lx, Ly
        self.current_fft_node_id = current_fft_node_id # ID węzła FFT
        self.adsorbate_set_index = adsorbate_set_index

        # Piki wybierane w tym dialogu (surowe kliknięcia adsorbatu)
        self.selected_adsorbate_spots_raw: List[Tuple[float, float]] = list(current_adsorbate_spots) if current_adsorbate_spots else []
        self.raw_adsorbate_spot_markers: Optional[ScatterPlotItem] = None # Dla surowych kliknięć adsorbatu

        # Przechowywanie przekazanej transformacji substratu
        self.sub_F_m2i = substrate_F_m2i
        self.sub_t_m2i = substrate_t_m2i
        self.sub_transform_analysis = substrate_transform_analysis
        
        # Przechowywanie przekazanych pików referencyjnych substratu
        self.ideal_substrate_spots_to_display_px = list(ideal_substrate_spots_for_display_px) if ideal_substrate_spots_for_display_px else []
        self.fitted_substrate_spots_to_display_px = list(fitted_substrate_spots_for_display_px) if fitted_substrate_spots_for_display_px else []
        self.ideal_substrate_marker_item: Optional[ScatterPlotItem] = None
        self.fitted_substrate_marker_item: Optional[ScatterPlotItem] = None

        # Skorygowane piki adsorbatu (obliczane w tym dialogu)
        self.corrected_adsorbate_spots_in_ideal_system: List[Tuple[float, float]] = []
        self.corrected_adsorbate_marker_item: Optional[ScatterPlotItem] = None # Dla skorygowanych pików adsorbatu

        # Parametry uściślania
        self.current_refinement_method = default_refinement_method
        self.refinement_roi_size = default_refinement_roi_size

        # Atrybuty dla podglądów (jak w SubstrateSpotSelectionDialog)
        self.last_preview_gauss_fit_popt: Optional[np.ndarray] = None
        self.last_preview_gauss_fit_center_abs: Optional[Tuple[float, float]] = None
        self.last_preview_gauss_roi_state: Optional[Dict] = None
        self.gl_roi_view_widget: Optional[GLViewWidget] = None
        self.gl_roi_surface_plot_item: Optional[GLSurfacePlotItem] = None
        self.gl_gauss_view_widget: Optional[GLViewWidget] = None
        self.gl_gauss_surface_plot_item: Optional[GLSurfacePlotItem] = None
        self.gl_roi_placeholder: Optional[QWidget] = None
        self.gl_gauss_placeholder: Optional[QWidget] = None

        self._init_ui()
        # self._connect_signals()
        # self._update_adsorbate_spots_list_widget() # Zmieniono nazwę
        # self._redraw_all_markers_in_dialog() # Zmieniono nazwę
        # self._update_add_spot_button_state()
        # self._update_correction_button_state()

        if self.current_refinement_method == REFINEMENT_MAX_PIXEL: self.rb_refine_max_pixel.setChecked(True)
        elif self.current_refinement_method == REFINEMENT_GAUSSIAN_FIT: self.rb_refine_gaussian.setChecked(True)
        else: self.rb_refine_direct.setChecked(True)
        self.refinement_roi_size_spinbox.setValue(self.refinement_roi_size)
        
        # self._on_refinement_method_changed()
        # self._display_substrate_transform_info() # Wyświetl info o transformacji substratu

        logger.debug(f"AdsorbateSpotSelectionDialog for set {self.adsorbate_set_index} initialized.")


    def _init_ui(self):
        top_level_layout = QHBoxLayout(self)
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        top_level_layout.addWidget(main_splitter)

        # === LEWY PANEL: Kontrolki ===
        left_controls_widget = QWidget()
        left_controls_layout = QVBoxLayout(left_controls_widget)
        left_controls_widget.setMinimumWidth(300)
        left_controls_widget.setMaximumWidth(350)

        # Grupa: Spot Refinement (jak w Substrate dialog)
        refinement_group = QGroupBox("Adsorbate Spot Refinement") # Zmiana tytułu
        refinement_layout = QFormLayout(refinement_group)
        self.rb_refine_direct = QRadioButton(REFINEMENT_DIRECT_CLICK)
        self.rb_refine_direct.setChecked(True)
        self.rb_refine_max_pixel = QRadioButton(REFINEMENT_MAX_PIXEL)
        self.rb_refine_gaussian = QRadioButton(REFINEMENT_GAUSSIAN_FIT)
        refinement_layout.addRow(self.rb_refine_direct)
        refinement_layout.addRow(self.rb_refine_max_pixel)
        refinement_layout.addRow(self.rb_refine_gaussian)
        self.refinement_roi_size_spinbox = QSpinBox()
        self.refinement_roi_size_spinbox.setMinimum(3)
        self.refinement_roi_size_spinbox.setMaximum(31)
        self.refinement_roi_size_spinbox.setSingleStep(2)
        self.refinement_roi_size_spinbox.setValue(self.refinement_roi_size)
        refinement_layout.addRow("Refinement Area Size (px):", self.refinement_roi_size_spinbox)
        left_controls_layout.addWidget(refinement_group)

        # Przycisk dodawania spotu adsorbatu
        self.add_adsorbate_spot_button = QPushButton("Add Adsorbate Spot from ROI") # Zmiana etykiety
        self.add_adsorbate_spot_button.setEnabled(False)
        left_controls_layout.addWidget(self.add_adsorbate_spot_button)
        
        # --- NOWA SEKCJA: Substrate Transformation Info & Actions ---
        sub_transform_group = QGroupBox("Substrate Transformation (Reference)")
        sub_transform_layout = QFormLayout(sub_transform_group)
        self.sub_transform_info_label_status = QLabel("Status: Not calculated / Not passed.")
        self.sub_transform_info_label_rot = QLabel("Sub. Rotation: -")
        self.sub_transform_info_label_scale = QLabel("Sub. Scale (X,Y): -")
        self.sub_transform_info_label_rmse = QLabel("Sub. RMSE (px): -")
        sub_transform_layout.addRow(self.sub_transform_info_label_status)
        sub_transform_layout.addRow(self.sub_transform_info_label_rot)
        sub_transform_layout.addRow(self.sub_transform_info_label_scale)
        sub_transform_layout.addRow(self.sub_transform_info_label_rmse)
        
        self.apply_correction_button = QPushButton("Apply Substrate Correction to Adsorbate Spots")
        self.apply_correction_button.setEnabled(False) # Włączany, gdy są piki adsorbatu i transformacja substratu
        sub_transform_layout.addRow(self.apply_correction_button)
        left_controls_layout.addWidget(sub_transform_group)
        # --- KONIEC NOWEJ SEKCJI ---

        left_controls_layout.addStretch(1)
        main_splitter.addWidget(left_controls_widget)


        # === CENTRALNY PANEL: Główny obraz FFT ===
        self.fft_plot_widget = GraphicsLayoutWidget()
        self.fft_view_box = self.fft_plot_widget.addViewBox(row=0, col=0, lockAspect=True, invertY=True)
        self.fft_image_item = ImageItem()
        self.fft_view_box.addItem(self.fft_image_item)
        self.fft_view_box.setMenuEnabled(True)
        self.fft_view_box.setMouseMode(ViewBox.PanMode)
        self.fft_view_box.setMouseEnabled(x=True, y=True)
        if self.fft_data is not None: self.fft_image_item.setImage(self.fft_data.T)
        self.selection_roi = RectROI(pos=(0,0), size=(self.refinement_roi_size, self.refinement_roi_size), pen=pg.mkPen('m', width=2), translateSnap=True, scaleSnap=True, movable=True, resizable=True, rotatable=False) # Inny kolor ROI
        self.fft_view_box.addItem(self.selection_roi)
        self.selection_roi.setVisible(False)
        main_splitter.addWidget(self.fft_plot_widget)


        # === PRAWY PANEL: Podglądy, lista spotów, przyciski OK/Anuluj ===
        right_panel_widget = QWidget()
        right_panel_layout = QVBoxLayout(right_panel_widget)
        right_panel_widget.setMinimumWidth(450)
        right_panel_widget.setMaximumWidth(550)

        # Grupa Podglądów (2D ROI, 3D ROI, 2D Gauss, 3D Gauss) - jak w Substrate dialog
        preview_group = QGroupBox("Live Previews")
        preview_grid_layout = QGridLayout(preview_group)

        roi_2d_container = QWidget()
        roi_2d_v_layout = QVBoxLayout(roi_2d_container)
        roi_2d_v_layout.addWidget(QLabel("ROI 2D Preview:"))
        self.enable_2d_roi_preview_checkbox = QCheckBox("Enable")
        self.enable_2d_roi_preview_checkbox.setChecked(True)
        roi_2d_v_layout.addWidget(self.enable_2d_roi_preview_checkbox)
        self.roi_preview_2d_widget = GraphicsLayoutWidget()
        self.roi_preview_2d_widget.setMinimumSize(150,150)
        self.roi_preview_2d_plot = self.roi_preview_2d_widget.addViewBox(lockAspect=True, invertY=True)
        self.roi_preview_2d_image_item = ImageItem()
        self.roi_preview_2d_plot.addItem(self.roi_preview_2d_image_item)
        roi_2d_v_layout.addWidget(self.roi_preview_2d_widget, 1)
        preview_grid_layout.addWidget(roi_2d_container, 0, 0)

        roi_3d_container = QWidget()
        roi_3d_v_layout = QVBoxLayout(roi_3d_container)
        roi_3d_v_layout.addWidget(QLabel("ROI 3D Preview:"))
        self.enable_3d_roi_preview_checkbox = QCheckBox("Enable")
        self.enable_3d_roi_preview_checkbox.setChecked(False)
        roi_3d_v_layout.addWidget(self.enable_3d_roi_preview_checkbox)
        self.gl_roi_view_widget = GLViewWidget()
        self.gl_roi_view_widget.setMinimumSize(150,150)
        self.gl_roi_surface_plot_item = GLSurfacePlotItem(color=(0.5,0.5,1,0.7))
        self.gl_roi_view_widget.addItem(self.gl_roi_surface_plot_item)
        roi_3d_v_layout.addWidget(self.gl_roi_view_widget, 1)
        preview_grid_layout.addWidget(roi_3d_container, 0, 1)

        gauss_2d_container = QWidget()
        gauss_2d_v_layout = QVBoxLayout(gauss_2d_container)
        gauss_2d_v_layout.addWidget(QLabel("Gaussian Fit 2D Preview:"))
        self.enable_gauss_2d_preview_checkbox = QCheckBox("Enable")
        self.enable_gauss_2d_preview_checkbox.setChecked(True)
        gauss_2d_v_layout.addWidget(self.enable_gauss_2d_preview_checkbox)
        self.gaussian_preview_2d_widget = GraphicsLayoutWidget()
        self.gaussian_preview_2d_widget.setMinimumSize(150,150)
        self.gaussian_preview_2d_plot = self.gaussian_preview_2d_widget.addViewBox(lockAspect=True, invertY=True)
        self.gaussian_preview_2d_image_item = ImageItem()
        self.gaussian_preview_2d_plot.addItem(self.gaussian_preview_2d_image_item)
        gauss_2d_v_layout.addWidget(self.gaussian_preview_2d_widget, 1)
        preview_grid_layout.addWidget(gauss_2d_container, 1, 0)

        gauss_3d_container = QWidget()
        gauss_3d_v_layout = QVBoxLayout(gauss_3d_container)
        gauss_3d_v_layout.addWidget(QLabel("Gaussian Fit 3D Preview:"))
        self.enable_gauss_3d_preview_checkbox = QCheckBox("Enable")
        self.enable_gauss_3d_preview_checkbox.setChecked(False)
        gauss_3d_v_layout.addWidget(self.enable_gauss_3d_preview_checkbox)
        self.gl_gauss_view_widget = GLViewWidget()
        self.gl_gauss_view_widget.setMinimumSize(150,150)
        self.gl_gauss_surface_plot_item = GLSurfacePlotItem(color=(1,0.5,0.5,0.7))
        self.gl_gauss_view_widget.addItem(self.gl_gauss_surface_plot_item)
        gauss_3d_v_layout.addWidget(self.gl_gauss_view_widget, 1)
        preview_grid_layout.addWidget(gauss_3d_container, 1, 1)

        preview_grid_layout.setColumnStretch(0,1)
        preview_grid_layout.setColumnStretch(1,1)
        preview_grid_layout.setRowStretch(0,1)
        preview_grid_layout.setRowStretch(1,1)
        self.gauss_2d_container = gauss_2d_container
        self.gauss_3d_container = gauss_3d_container
        self.gauss_2d_container.setVisible(False)
        self.gauss_3d_container.setVisible(False)
        right_panel_layout.addWidget(preview_group)

        # Grupa "Display Options (Reference Spots)"
        display_options_group = QGroupBox("Display Options (Reference Spots)")
        display_options_layout = QVBoxLayout(display_options_group)
        self.show_ideal_substrate_checkbox = QCheckBox("Show Ideal Substrate Spots (Nearest)")
        self.show_ideal_substrate_checkbox.setChecked(False) # Domyślnie mogą być wyłączone, aby nie zaciemniać
        self.show_fitted_substrate_checkbox = QCheckBox("Show Fitted Substrate Spots")
        self.show_fitted_substrate_checkbox.setChecked(False)
        self.show_corrected_adsorbate_checkbox = QCheckBox("Show Corrected Adsorbate Spots (This Set)")
        self.show_corrected_adsorbate_checkbox.setChecked(True) # Domyślnie pokaż skorygowane, jeśli są
        display_options_layout.addWidget(self.show_ideal_substrate_checkbox)
        display_options_layout.addWidget(self.show_fitted_substrate_checkbox)
        display_options_layout.addWidget(self.show_corrected_adsorbate_checkbox)
        right_panel_layout.addWidget(display_options_group)

        # Grupa: Selected Adsorbate Spots
        spots_list_group = QGroupBox(f"Selected Adsorbate Spots (Set {self.adsorbate_set_index + 1})")
        spots_list_layout = QVBoxLayout(spots_list_group)
        self.spots_list_widget = QListWidget()
        self.spots_list_widget.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        spots_list_layout.addWidget(self.spots_list_widget)
        spot_buttons_layout = QHBoxLayout()
        self.remove_spot_button = QPushButton("Remove Selected")
        self.clear_all_spots_button = QPushButton("Clear All (This Set)")
        spot_buttons_layout.addWidget(self.remove_spot_button)
        spot_buttons_layout.addWidget(self.clear_all_spots_button)
        spots_list_layout.addLayout(spot_buttons_layout)
        right_panel_layout.addWidget(spots_list_group)
        
        self.status_label = QLabel("Click on FFT to place ROI, or drag existing ROI.")
        right_panel_layout.addWidget(self.status_label)
        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        right_panel_layout.addWidget(self.button_box)
        
        right_panel_layout.addStretch(1)
        main_splitter.addWidget(right_panel_widget)

        main_splitter.setSizes([320, 500, 380])
        main_splitter.setStretchFactor(1, 1)