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
        self._connect_signals()
        self._update_adsorbate_spots_list_widget() # Zmieniono nazwę
        self._redraw_all_markers_in_dialog() # Zmieniono nazwę
        self._update_add_spot_button_state()
        self._update_correction_button_state()

        if self.current_refinement_method == REFINEMENT_MAX_PIXEL: self.rb_refine_max_pixel.setChecked(True)
        elif self.current_refinement_method == REFINEMENT_GAUSSIAN_FIT: self.rb_refine_gaussian.setChecked(True)
        else: self.rb_refine_direct.setChecked(True)
        self.refinement_roi_size_spinbox.setValue(self.refinement_roi_size)
        
        self._on_refinement_method_changed()
        self._display_substrate_transform_info() # Wyświetl info o transformacji substratu

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

    def _connect_signals(self):
        """Connects all UI element signals to their respective slots."""
        # Przyciski OK/Anuluj
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        # Przyciski zarządzania listą spotów adsorbatu
        self.remove_spot_button.clicked.connect(self._remove_selected_spot)
        self.clear_all_spots_button.clicked.connect(self._clear_all_spots_in_dialog) # Użyj poprawnej nazwy slotu

        # Interakcja z głównym obrazem FFT
        if self.fft_view_box and self.fft_view_box.scene(): # Upewnij się, że scena istnieje
            self.fft_view_box.scene().sigMouseClicked.connect(self._handle_fft_image_click)
        
        # Interakcja z ROI
        self.selection_roi.sigRegionChanged.connect(self._handle_roi_region_changing)
        # self.selection_roi.sigRegionChangeFinished.connect(self._handle_roi_changed_finished) # Jeśli potrzebne

        # Kontrolki metody uściślania
        self.rb_refine_direct.toggled.connect(self._on_refinement_method_changed)
        self.rb_refine_max_pixel.toggled.connect(self._on_refinement_method_changed)
        self.rb_refine_gaussian.toggled.connect(self._on_refinement_method_changed)
        self.refinement_roi_size_spinbox.valueChanged.connect(self._on_refinement_roi_size_changed)

        # Przycisk dodawania piku adsorbatu
        self.add_adsorbate_spot_button.clicked.connect(self._add_current_adsorbate_spot_from_roi) # Użyj poprawnej nazwy slotu

        # Przycisk korekcji adsorbatu
        self.apply_correction_button.clicked.connect(self._on_apply_substrate_correction_clicked) # Pamiętaj o zdefiniowaniu slotu

        # Checkboxy podglądów Live
        self.enable_2d_roi_preview_checkbox.stateChanged.connect(self._update_roi_previews)
        self.enable_3d_roi_preview_checkbox.stateChanged.connect(self._update_roi_previews)
        self.enable_gauss_2d_preview_checkbox.stateChanged.connect(self._update_roi_previews)
        self.enable_gauss_3d_preview_checkbox.stateChanged.connect(self._update_roi_previews)

        # Checkboxy dla wyświetlania spotów referencyjnych substratu i skorygowanych adsorbatu
        self.show_ideal_substrate_checkbox.stateChanged.connect(self._redraw_all_markers_in_dialog)
        self.show_fitted_substrate_checkbox.stateChanged.connect(self._redraw_all_markers_in_dialog)
        self.show_corrected_adsorbate_checkbox.stateChanged.connect(self._redraw_all_markers_in_dialog)

        logger.debug("AdsorbateSpotSelectionDialog signals connected.")
    
    def _clear_last_preview_gauss_fit(self): # Już istnieje
        self.last_preview_gauss_fit_popt = None
        self.last_preview_gauss_fit_center_abs = None
        self.last_preview_gauss_roi_state = None
        logger.debug("Cleared last preview Gaussian fit results for adsorbate dialog.")

    @pyqtSlot(object)
    def _handle_roi_region_changing(self, roi_item: Optional[pg.ROI] = None): # Już istnieje (logika do weryfikacji)
        if roi_item is None: roi_item = self.selection_roi
        if not isinstance(roi_item, RectROI): return
        if roi_item.isVisible():
            roi_pos = roi_item.pos()
            roi_size = roi_item.size()
            current_roi_w = int(round(roi_size.x()))
            if current_roi_w != self.refinement_roi_size_spinbox.value() and \
               self.refinement_roi_size_spinbox.minimum() <= current_roi_w <= self.refinement_roi_size_spinbox.maximum() and \
               current_roi_w % 2 != 0 :
                self.refinement_roi_size_spinbox.blockSignals(True)
                self.refinement_roi_size_spinbox.setValue(current_roi_w)
                self.refinement_roi_size_spinbox.blockSignals(False)
            self._clear_last_preview_gauss_fit()
            self._update_roi_previews()
            logger.debug(f"Adsorbate ROI region changing: Pos ({roi_pos.x():.1f}, {roi_pos.y():.1f}), Size ({roi_size.x():.1f}, {roi_size.y():.1f})")
            
    def _update_roi_previews(self): # Już istnieje (logika do weryfikacji/uzupełnienia)
        logger.debug("Adsorbate dialog: Updating ROI previews...")
        # TODO: Implement or verify the full logic here, similar to SubstrateSpotSelectionDialog
        # Pamiętaj o używaniu self.selected_adsorbate_spots_raw
        pass

    def _update_3d_surface_plot(self, surface_item: GLSurfacePlotItem, data_2d: Optional[np.ndarray]): # Już istnieje
        if data_2d is None or data_2d.size == 0 or data_2d.ndim != 2:
            self._clear_3d_surface(surface_item) # Użyj metody czyszczącej
            return

        h, w = data_2d.shape
        x = np.linspace(-w/2, w/2, w)
        y = np.linspace(-h/2, h/2, h)
        
        # Kolory można ustawić na podstawie wysokości Z
        # Prosty gradient od niebieskiego do czerwonego
        colors = np.empty((w,h,4), dtype=np.float32)
        z_norm = (data_2d - data_2d.min()) / (data_2d.max() - data_2d.min() + 1e-9) # Normalizacja 0-1
        colors[..., 0] = z_norm.T # R
        colors[..., 1] = 0       # G
        colors[..., 2] = 1 - z_norm.T # B
        colors[..., 3] = 0.7     # Alpha

        surface_item.setData(x=x, y=y, z=data_2d.T, colors=colors) # Transpozycja Z dla setData
        # surface_item.opts['distance'] = 40 # Dostosuj odległość kamery
        # surface_item.opts['elevation'] = 30
        # surface_item.opts['azimuth'] = -90
        
    def _clear_3d_surface(self, surface_item: Optional[GLSurfacePlotItem]): # Już istnieje
        if surface_item:
            try:
                # Ustaw minimalne prawidłowe dane
                x = np.array([0, 1], dtype=np.float32)
                y = np.array([0, 1], dtype=np.float32)
                z = np.zeros((2, 2), dtype=np.float32)
                colors = np.zeros((2, 2, 4), dtype=np.float32)
                
                surface_item.setData(x=x, y=y, z=z, colors=colors)
                surface_item.meshDataChanged()
            except Exception as e:
                logger.error(f"Error clearing 3D surface: {e}")

    @pyqtSlot()
    def _on_refinement_method_changed(self): # Już istnieje (logika do weryfikacji)
        is_gaussian_mode = self.rb_refine_gaussian.isChecked()
        if hasattr(self, 'gauss_2d_container'): self.gauss_2d_container.setVisible(is_gaussian_mode)
        if hasattr(self, 'gauss_3d_container'): self.gauss_3d_container.setVisible(is_gaussian_mode)
        if self.rb_refine_direct.isChecked(): 
            self.current_refinement_method = REFINEMENT_DIRECT_CLICK
            self.refinement_roi_size_spinbox.setEnabled(False)
            self.selection_roi.setVisible(False)
            self.add_adsorbate_spot_button.setEnabled(False)
            self.status_label.setText("Click directly on FFT to add adsorbate spot.")
        else: 
            self.current_refinement_method = REFINEMENT_MAX_PIXEL if self.rb_refine_max_pixel.isChecked() else REFINEMENT_GAUSSIAN_FIT; self.refinement_roi_size_spinbox.setEnabled(True); self.add_adsorbate_spot_button.setEnabled(self.selection_roi.isVisible()); self.status_label.setText("Click on FFT to place ROI, or drag ROI. Then Add Spot.")
        if not is_gaussian_mode: self._clear_last_preview_gauss_fit()
        self._update_roi_previews()
        logger.debug(f"Adsorbate refinement method: {self.current_refinement_method}")


    @pyqtSlot(int)
    def _on_refinement_roi_size_changed(self, value: int): # Już istnieje (logika do weryfikacji)
        self.refinement_roi_size = value
        self._clear_last_preview_gauss_fit()
        if self.selection_roi.isVisible():
            current_pos = self.selection_roi.pos()
            old_size = self.selection_roi.size()
            center_x = current_pos.x() + old_size.x()/2
            center_y = current_pos.y() + old_size.y()/2
            new_pos_x = center_x - value/2
            new_pos_y = center_y - value/2
            self.selection_roi.setPos((new_pos_x, new_pos_y), update=False)
            self.selection_roi.setSize((value,value), update=False)
            self._handle_roi_region_changing()
        logger.debug(f"Adsorbate refinement ROI size: {self.refinement_roi_size}")

    @pyqtSlot()
    def _add_current_adsorbate_spot_from_roi(self): # Zmieniono nazwę slotu
        logger.debug("Attempting to add adsorbate spot from ROI...")
        # TODO: Implementacja logiki dodawania piku adsorbatu, podobnie jak w Substrate,
        # ale bez sprawdzania limitu (chyba że chcesz np. min 3 piki)
        # Aktualizuj self.selected_adsorbate_spots_raw
        # Wywołaj self._update_adsorbate_spots_list_widget()
        # Wywołaj self._redraw_all_markers_in_dialog()
        # Wywołaj self._update_correction_button_state()
        pass

    def _update_adsorbate_spots_list_widget(self): # Zmieniono nazwę metody
        self.spots_list_widget.clear()
        for i, (kx, ky) in enumerate(self.selected_adsorbate_spots_raw):
            self.spots_list_widget.addItem(f"A{i+1} (Set {self.adsorbate_set_index + 1}): ({kx:.2f}, {ky:.2f})")
        # self._update_add_spot_button_state() # Jeśli jest logika limitu
        self._update_correction_button_state()


    @pyqtSlot()
    def _remove_selected_spot(self): # Już istnieje (logika do weryfikacji)
        current_item = self.spots_list_widget.currentItem()
        if current_item:
            row = self.spots_list_widget.row(current_item)
            if 0 <= row < len(self.selected_adsorbate_spots_raw):
                del self.selected_adsorbate_spots_raw[row]
                self._update_adsorbate_spots_list_widget()
                self._redraw_all_markers_in_dialog()
                logger.debug(f"Removed adsorbate spot at index {row} from set {self.adsorbate_set_index}")

    @pyqtSlot()
    def _clear_all_spots_in_dialog(self): # Już istnieje (logika do weryfikacji)
        self.selected_adsorbate_spots_raw.clear()
        self.corrected_adsorbate_spots_in_ideal_system.clear() # Wyczyść też skorygowane
        self._update_adsorbate_spots_list_widget()
        self._redraw_all_markers_in_dialog()
        logger.debug(f"Cleared all adsorbate spots in dialog for set {self.adsorbate_set_index}.")

    def _redraw_all_markers_in_dialog(self): # Zmieniono nazwę
        logger.debug("Adsorbate dialog: Redrawing all markers...")
        # Usuń wszystkie stare markery
        if self.raw_adsorbate_spot_markers: 
            try: 
                self.fft_view_box.removeItem(self.raw_adsorbate_spot_markers)
                self.raw_adsorbate_spot_markers = None
            except RuntimeError: pass
        if self.ideal_substrate_marker_item: 
            try: 
                self.fft_view_box.removeItem(self.ideal_substrate_marker_item)
                self.ideal_substrate_marker_item = None
            except RuntimeError: pass
        if self.fitted_substrate_marker_item:
            try: 
                self.fft_view_box.removeItem(self.fitted_substrate_marker_item)
                self.fitted_substrate_marker_item = None
            except RuntimeError: pass
        if self.corrected_adsorbate_marker_item: 
            try: 
                self.fft_view_box.removeItem(self.corrected_adsorbate_marker_item)
                self.corrected_adsorbate_marker_item = None
            except RuntimeError: pass
            
        # 1. Rysuj surowe piki adsorbatu (self.selected_adsorbate_spots_raw)
        if self.selected_adsorbate_spots_raw:
            raw_spots_data = [{'pos': spot, 'symbol': 'o', 'size': 10, 'pen': pg.mkPen('b', width=1.5), 'brush': pg.mkBrush(0,0,255,120)} for spot in self.selected_adsorbate_spots_raw]
            self.raw_adsorbate_spot_markers = ScatterPlotItem(spots=raw_spots_data)
            self.fft_view_box.addItem(self.raw_adsorbate_spot_markers)

        # 2. Rysuj idealne piki substratu (jeśli checkbox i dane)
        if self.show_ideal_substrate_checkbox.isChecked() and self.ideal_substrate_spots_to_display_px:
            ideal_sub_data = [{'pos': spot, 'symbol': '+', 'size': 12, 'pen': pg.mkPen('m', width=1.5)} for spot in self.ideal_substrate_spots_to_display_px]
            self.ideal_substrate_marker_item = ScatterPlotItem(spots=ideal_sub_data)
            self.fft_view_box.addItem(self.ideal_substrate_marker_item)

        # 3. Rysuj dopasowane piki substratu (jeśli checkbox i dane)
        if self.show_fitted_substrate_checkbox.isChecked() and self.fitted_substrate_spots_to_display_px:
            fitted_sub_data = [{'pos': spot, 'symbol': 'x', 'size': 12, 'pen': pg.mkPen('c', width=2.0)} for spot in self.fitted_substrate_spots_to_display_px]
            self.fitted_substrate_marker_item = ScatterPlotItem(spots=fitted_sub_data)
            self.fft_view_box.addItem(self.fitted_substrate_marker_item)

        # 4. Rysuj skorygowane piki adsorbatu (jeśli checkbox i dane)
        # Pamiętaj, że corrected_adsorbate_spots_in_ideal_system są w "idealnym" systemie.
        # Do narysowania na oryginalnym FFT, trzeba je przetransformować z powrotem.
        if self.show_corrected_adsorbate_checkbox.isChecked() and self.corrected_adsorbate_spots_in_ideal_system and self.sub_F_m2i is not None and self.sub_t_m2i is not None:
            try:
                F_inv = np.linalg.inv(self.sub_F_m2i)
                # t_prime dla P_measured = P_ideal @ F_inv.T + t_prime
                t_prime = (-self.sub_t_m2i @ F_inv.T).flatten() # type: ignore
                
                from ...analysis.drift_correction import apply_affine_transform # Import lokalny dla bezpieczeństwa
                
                spots_to_draw_on_fft = apply_affine_transform(
                    np.array(self.corrected_adsorbate_spots_in_ideal_system),
                    F_inv,
                    t_prime
                )
                if spots_to_draw_on_fft is not None:
                    corrected_ads_data = [{'pos': tuple(pt), 'symbol': 's', 'size': 10, 'pen': pg.mkPen('r', width=1.5), 'brush': pg.mkBrush(255,0,0,120)} for pt in spots_to_draw_on_fft]
                    self.corrected_adsorbate_marker_item = ScatterPlotItem(spots=corrected_ads_data)
                    self.fft_view_box.addItem(self.corrected_adsorbate_marker_item)
            except Exception as e: # pragma: no cover
                logger.error(f"Error transforming corrected adsorbate spots for display: {e}")


    @pyqtSlot()
    def _handle_fft_image_click(self, event): # Już istnieje (logika do weryfikacji)
        if event.button() == Qt.MouseButton.LeftButton:
            pos_viewbox = self.fft_view_box.mapSceneToView(event.scenePos())
            mapped_pos = self.fft_image_item.mapToData(pos_viewbox)
            if mapped_pos is not None:
                kx, ky = int(round(mapped_pos.x())), int(round(mapped_pos.y()))
                logger.debug(f"Adsorbate Dialog FFT click: (kx, ky) = ({kx}, {ky})")
                roi_size = self.refinement_roi_size_spinbox.value()
                roi_x = kx - roi_size//2
                roi_y = ky - roi_size//2
                if self.fft_data is not None: max_y, max_x = self.fft_data.shape
                roi_x = np.clip(roi_x, 0, max_x - roi_size)
                roi_y = np.clip(roi_y, 0, max_y - roi_size)
                self.selection_roi.setPos((roi_x, roi_y), update=False)
                self.selection_roi.setSize((roi_size,roi_size), update=False)
                self.selection_roi.setVisible(True)
                self.add_adsorbate_spot_button.setEnabled(True)
                self._update_roi_previews()
            event.accept()
        else: event.ignore() # pragma: no cover
            
    def _update_add_spot_button_state(self): # Już istnieje (ew. modyfikacja logiki limitu)
        # Dla adsorbatu może nie być sztywnego limitu, ale przycisk może być włączony, gdy ROI jest widoczne
        self.add_adsorbate_spot_button.setEnabled(self.selection_roi.isVisible())
        # Można dodać logikę, np. "Select at least 3 spots for lattice definition."
        num_selected = len(self.selected_adsorbate_spots_raw)
        if num_selected < 3:
            self.status_label.setText(f"Select at least {3-num_selected} more adsorbate spot(s). Current: {num_selected}.")
        else:
            self.status_label.setText(f"Selected {num_selected} adsorbate spots. Ready to add or correct.")


    def _update_correction_button_state(self): # TODO: Implement
        """Włącza/wyłącza przycisk Apply Substrate Correction."""
        can_correct = bool(self.sub_F_m2i is not None and self.sub_t_m2i is not None and self.selected_adsorbate_spots_raw)
        self.apply_correction_button.setEnabled(can_correct)
        if not (self.sub_F_m2i is not None and self.sub_t_m2i is not None):
            self.sub_transform_info_label_status.setText("Status: Substrate transform not available.")


    def _display_substrate_transform_info(self): # TODO: Implement
        """Wyświetla informacje o przekazanej transformacji substratu."""
        if self.sub_transform_analysis:
            self.sub_transform_info_label_status.setText("Status: Substrate transform applied/available.")
            self.sub_transform_info_label_rot.setText(f"Sub. Rotation (M->I): {self.sub_transform_analysis.get('rotation_angle_deg', 'N/A'):.2f}°")
            s_x = self.sub_transform_analysis.get('principal_stretches', [np.nan, np.nan])[0]
            s_y = self.sub_transform_analysis.get('principal_stretches', [np.nan, np.nan])[1]
            self.sub_transform_info_label_scale.setText(f"Sub. Stretches (M->I): ({s_x:.3f}, {s_y:.3f})")
            self.sub_transform_info_label_rmse.setText(f"Sub. Fit RMSE (M->I, px): {self.sub_transform_analysis.get('rmse', 'N/A'):.3f}")
        else:
            self.sub_transform_info_label_status.setText("Status: Substrate transform not available.")
            self.sub_transform_info_label_rot.setText("Sub. Rotation: -")
            self.sub_transform_info_label_scale.setText("Sub. Scale (X,Y): -")
            self.sub_transform_info_label_rmse.setText("Sub. RMSE (px): -")

    @pyqtSlot()
    def _on_apply_substrate_correction_clicked(self): # TODO: Implement
        logger.info("Apply Substrate Correction button clicked.")
        # Tutaj logika z Kroku 8 (Faza III) - zastosowanie self.sub_F_m2i, self.sub_t_m2i
        # do self.selected_adsorbate_spots_raw i zapisanie w self.corrected_adsorbate_spots_in_ideal_system
        # oraz odświeżenie markerów.
        if self.sub_F_m2i is None or self.sub_t_m2i is None or not self.selected_adsorbate_spots_raw:
            QMessageBox.warning(self, "Cannot Correct", "Substrate transformation is not available or no adsorbate spots selected.")
            return

        try:
            from ...analysis.drift_correction import apply_affine_transform
            raw_spots_np = np.array(self.selected_adsorbate_spots_raw, dtype=float)
            corrected_spots_np = apply_affine_transform(raw_spots_np, self.sub_F_m2i, self.sub_t_m2i)
            if corrected_spots_np is not None:
                self.corrected_adsorbate_spots_in_ideal_system = [tuple(pt) for pt in corrected_spots_np]
                logger.info(f"Applied substrate correction to {len(self.corrected_adsorbate_spots_in_ideal_system)} adsorbate spots.")
                self._redraw_all_markers_in_dialog() # Aby pokazać skorygowane, jeśli checkbox zaznaczony
                self.status_label.setText(f"{len(self.corrected_adsorbate_spots_in_ideal_system)} adsorbate spots corrected.")
            else:
                raise ValueError("apply_affine_transform returned None")
        except Exception as e:
            logger.error(f"Error applying substrate correction to adsorbate spots: {e}")
            QMessageBox.critical(self, "Correction Error", f"Could not apply correction: {e}")


    def get_dialog_results(self) -> Dict[str, Any]: # TODO: Implement
        return {
            "raw_adsorbate_spots": list(self.selected_adsorbate_spots_raw),
            "corrected_adsorbate_spots": list(self.corrected_adsorbate_spots_in_ideal_system),
            "adsorbate_set_index": self.adsorbate_set_index
        }

    def accept(self): # TODO: Implement
        # Można dodać walidację, np. czy wybrano przynajmniej 3 piki adsorbatu
        if len(self.selected_adsorbate_spots_raw) < 1 and not self.corrected_adsorbate_spots_in_ideal_system: # Pozwól zaakceptować, jeśli są skorygowane
            reply = QMessageBox.question(self, "No Spots Selected", 
                                         "No adsorbate spots have been selected or corrected for this set. Continue anyway?",
                                         QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                         QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.No:
                return
        
        logger.info(f"AdsorbateSpotSelectionDialog for set {self.adsorbate_set_index} accepted. "
                    f"Raw spots: {len(self.selected_adsorbate_spots_raw)}, "
                    f"Corrected spots: {len(self.corrected_adsorbate_spots_in_ideal_system)}")
        super().accept()

    def reject(self): # Już istnieje
        logger.info(f"AdsorbateSpotSelectionDialog for set {self.adsorbate_set_index} rejected.")
        super().reject()

    def closeEvent(self, event): # Już istnieje (logika do weryfikacji)
        logger.debug("AdsorbateSpotSelectionDialog closing. Cleaning up GL items.")
        if hasattr(self, 'gl_roi_view_widget') and self.gl_roi_view_widget:
            if hasattr(self, 'gl_roi_surface_plot_item') and self.gl_roi_surface_plot_item: self.gl_roi_view_widget.removeItem(self.gl_roi_surface_plot_item)
            del self.gl_roi_surface_plot_item
        if hasattr(self, 'gl_gauss_view_widget') and self.gl_gauss_view_widget:
            if hasattr(self, 'gl_gauss_surface_plot_item') and self.gl_gauss_surface_plot_item: self.gl_gauss_view_widget.removeItem(self.gl_gauss_surface_plot_item)
            del self.gl_gauss_surface_plot_item
        super().closeEvent(event)