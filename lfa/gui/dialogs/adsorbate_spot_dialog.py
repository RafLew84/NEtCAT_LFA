# lfa/gui/dialogs/adsorbate_spot_dialog.py
import logging
from typing import List, Tuple, Optional, Dict, Any
import numpy as np

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QDialogButtonBox,
    QLabel, QListWidget, QAbstractItemView, QWidget, QGroupBox,
    QFormLayout, QRadioButton, QSpinBox, QCheckBox, QMessageBox,
    QGridLayout, QSplitter # QSplitter jest używany
)
from PyQt6.QtCore import Qt, pyqtSlot, QPointF
from PyQt6.QtGui import QPen, QVector3D

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
    from ...analysis.lattice import KNOWN_LATTICES, get_reciprocal_points
    from ...core.history import HistoryNode
    from ...logic.history_manager import HistoryManager
    PEAK_FITTING_MODULE_AVAILABLE = True
except ImportError: # pragma: no cover
    PEAK_FITTING_MODULE_AVAILABLE = False
    SCIPY_AVAILABLE = False
    KNOWN_LATTICES = {}
    logging.error("AdsorbateSpotSelectionDialog: Could not import peak_fitting or lattice modules.")
    def find_max_pixel_in_roi(data, center, radius): return center
    def fit_2d_gaussian_in_roi(data, center, radius): return None
    def _gaussian_2d(*args, **kwargs): raise ImportError("Gaussian 2D function is not available")

logger = logging.getLogger(__name__)

REFINEMENT_DIRECT_CLICK = "Direct Click"
REFINEMENT_MAX_PIXEL = "Max Pixel"
REFINEMENT_GAUSSIAN_FIT = "2D Gaussian Fit"

class AdsorbateSpotSelectionDialog(QDialog):
    def __init__(self,
                 fft_image_data: Optional[np.ndarray],
                 history_manager: HistoryManager,
                 current_fft_node_id: str,
                 current_adsorbate_spots: Optional[List[Tuple[float, float]]] = None,
                 adsorbate_set_index: int = 0,
                 default_refinement_method: str = REFINEMENT_DIRECT_CLICK,
                 default_refinement_roi_size: int = 5,
                 substrate_F_m2i: Optional[np.ndarray] = None,
                 substrate_t_m2i: Optional[np.ndarray] = None,
                 substrate_transform_analysis: Optional[Dict[str, Any]] = None,
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
        self.history_manager = history_manager
        self.current_fft_node_id = current_fft_node_id
        self.adsorbate_set_index = adsorbate_set_index

        self.selected_adsorbate_spots_raw: List[Tuple[float, float]] = list(current_adsorbate_spots) if current_adsorbate_spots else []
        self.raw_adsorbate_spot_markers: Optional[ScatterPlotItem] = None

        self.sub_F_m2i = substrate_F_m2i
        self.sub_t_m2i = substrate_t_m2i
        self.sub_transform_analysis = substrate_transform_analysis
        
        self.ideal_substrate_spots_to_display_px = list(ideal_substrate_spots_for_display_px) if ideal_substrate_spots_for_display_px else []
        self.fitted_substrate_spots_to_display_px = list(fitted_substrate_spots_for_display_px) if fitted_substrate_spots_for_display_px else []
        self.ideal_substrate_marker_item: Optional[ScatterPlotItem] = None
        self.fitted_substrate_marker_item: Optional[ScatterPlotItem] = None

        self.corrected_adsorbate_spots_in_ideal_system: List[Tuple[float, float]] = []
        self.corrected_adsorbate_marker_item: Optional[ScatterPlotItem] = None

        self.current_refinement_method = default_refinement_method
        self.refinement_roi_size = default_refinement_roi_size

        self.last_preview_gauss_fit_popt: Optional[np.ndarray] = None
        self.last_preview_gauss_fit_center_abs: Optional[Tuple[float, float]] = None
        self.last_preview_gauss_roi_state: Optional[Dict] = None
        self.gl_roi_view_widget: Optional[GLViewWidget] = None
        self.gl_roi_surface_plot_item: Optional[GLSurfacePlotItem] = None # Zmienione z self.gl_roi_surface_item
        self.gl_gauss_view_widget: Optional[GLViewWidget] = None
        self.gl_gauss_surface_plot_item: Optional[GLSurfacePlotItem] = None # Zmienione z self.gl_gauss_surface_item
        self.gl_roi_placeholder: Optional[QWidget] = None
        self.gl_gauss_placeholder: Optional[QWidget] = None

        self._init_ui()
        self._connect_signals() 
        self._update_adsorbate_spots_list_widget()
        self._redraw_all_markers_in_dialog() 
        self._update_add_spot_button_state() 
        self._update_correction_button_state()

        if self.current_refinement_method == REFINEMENT_MAX_PIXEL: self.rb_refine_max_pixel.setChecked(True)
        elif self.current_refinement_method == REFINEMENT_GAUSSIAN_FIT: self.rb_refine_gaussian.setChecked(True)
        else: self.rb_refine_direct.setChecked(True)
        self.refinement_roi_size_spinbox.setValue(self.refinement_roi_size)
        
        self._on_refinement_method_changed()
        self._display_substrate_transform_info()
        self._redraw_all_markers_in_dialog() # Ponowne wywołanie, aby uwzględnić piki referencyjne po _display_substrate_transform_info

        logger.debug(f"AdsorbateSpotSelectionDialog for set {self.adsorbate_set_index} initialized.")

    def _init_ui(self): # Kod UI jak w poprzedniej odpowiedzi
        top_level_layout = QHBoxLayout(self)
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        top_level_layout.addWidget(main_splitter)

        left_controls_widget = QWidget()
        left_controls_layout = QVBoxLayout(left_controls_widget)
        left_controls_widget.setMinimumWidth(300)
        left_controls_widget.setMaximumWidth(350)

        refinement_group = QGroupBox("Adsorbate Spot Refinement")
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
        self.add_adsorbate_spot_button = QPushButton("Add Adsorbate Spot from ROI")
        self.add_adsorbate_spot_button.setEnabled(False)
        left_controls_layout.addWidget(self.add_adsorbate_spot_button)
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
        self.apply_correction_button.setEnabled(False)
        sub_transform_layout.addRow(self.apply_correction_button)
        left_controls_layout.addWidget(sub_transform_group)
        left_controls_layout.addStretch(1)
        main_splitter.addWidget(left_controls_widget)

        self.fft_plot_widget = GraphicsLayoutWidget()
        self.fft_view_box = self.fft_plot_widget.addViewBox(row=0, col=0, lockAspect=True, invertY=True)
        self.fft_image_item = ImageItem()
        self.fft_view_box.addItem(self.fft_image_item)
        self.fft_view_box.setMenuEnabled(True)
        self.fft_view_box.setMouseMode(ViewBox.PanMode)
        self.fft_view_box.setMouseEnabled(x=True, y=True)
        if self.fft_data is not None: self.fft_image_item.setImage(self.fft_data.T)
        self.selection_roi = RectROI(pos=(0,0), size=(self.refinement_roi_size, self.refinement_roi_size), pen=pg.mkPen('m', width=2), translateSnap=True, scaleSnap=True, movable=True, resizable=True, rotatable=False)
        self.fft_view_box.addItem(self.selection_roi)
        self.selection_roi.setVisible(False)
        main_splitter.addWidget(self.fft_plot_widget)

        right_panel_widget = QWidget()
        right_panel_layout = QVBoxLayout(right_panel_widget)
        right_panel_widget.setMinimumWidth(450)
        right_panel_widget.setMaximumWidth(550)
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
        self.roi_preview_2d_widget.setMaximumHeight(200)
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
        self.gl_roi_view_widget.setMaximumHeight(200)
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
        self.gaussian_preview_2d_widget.setMaximumHeight(200)
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
        self.gl_gauss_view_widget.setMaximumHeight(200)
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

        display_options_group = QGroupBox("Display Options (Reference Spots)")
        display_options_layout = QVBoxLayout(display_options_group)
        self.show_ideal_substrate_checkbox = QCheckBox("Show Ideal Substrate Spots (Nearest)")
        self.show_ideal_substrate_checkbox.setChecked(False)
        self.show_fitted_substrate_checkbox = QCheckBox("Show Fitted Substrate Spots")
        self.show_fitted_substrate_checkbox.setChecked(False)
        self.show_corrected_adsorbate_checkbox = QCheckBox("Show Corrected Adsorbate Spots (This Set)")
        self.show_corrected_adsorbate_checkbox.setChecked(True)
        display_options_layout.addWidget(self.show_ideal_substrate_checkbox)
        display_options_layout.addWidget(self.show_fitted_substrate_checkbox)
        display_options_layout.addWidget(self.show_corrected_adsorbate_checkbox)
        right_panel_layout.addWidget(display_options_group)
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
        main_splitter.setStretchFactor(1,1)

    def _connect_signals(self):
        """Connects all UI element signals to their respective slots."""
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        self.remove_spot_button.clicked.connect(self._remove_selected_spot)
        self.clear_all_spots_button.clicked.connect(self._clear_all_spots_in_dialog)

        if self.fft_view_box and self.fft_view_box.scene():
            self.fft_view_box.scene().sigMouseClicked.connect(self._handle_fft_image_click)
        
        self.selection_roi.sigRegionChanged.connect(self._handle_roi_region_changing)

        self.rb_refine_direct.toggled.connect(self._on_refinement_method_changed)
        self.rb_refine_max_pixel.toggled.connect(self._on_refinement_method_changed)
        self.rb_refine_gaussian.toggled.connect(self._on_refinement_method_changed)
        self.refinement_roi_size_spinbox.valueChanged.connect(self._on_refinement_roi_size_changed)

        self.add_adsorbate_spot_button.clicked.connect(self._add_current_adsorbate_spot_from_roi)
        self.apply_correction_button.clicked.connect(self._on_apply_substrate_correction_clicked)

        # Checkboxy podglądów Live
        self.enable_2d_roi_preview_checkbox.stateChanged.connect(self._update_roi_previews)
        self.enable_3d_roi_preview_checkbox.stateChanged.connect(self._update_roi_previews)
        self.enable_gauss_2d_preview_checkbox.stateChanged.connect(self._update_roi_previews)
        self.enable_gauss_3d_preview_checkbox.stateChanged.connect(self._update_roi_previews)

        # Checkboxy dla wyświetlania spotów referencyjnych
        self.show_ideal_substrate_checkbox.stateChanged.connect(self._redraw_all_markers_in_dialog)
        self.show_fitted_substrate_checkbox.stateChanged.connect(self._redraw_all_markers_in_dialog)
        self.show_corrected_adsorbate_checkbox.stateChanged.connect(self._redraw_all_markers_in_dialog)

        logger.debug("AdsorbateSpotSelectionDialog signals connected.")

    def _clear_last_preview_gauss_fit(self):
        self.last_preview_gauss_fit_popt = None
        self.last_preview_gauss_fit_center_abs = None
        self.last_preview_gauss_roi_state = None
        logger.debug("AdsorbateDialog: Cleared last preview Gaussian fit results.")

    @pyqtSlot(object)
    def _handle_roi_region_changing(self, roi_item: Optional[pg.ROI] = None):
        if roi_item is None: roi_item = self.selection_roi
        if not isinstance(roi_item, RectROI): return

        if roi_item.isVisible():
            roi_pos = roi_item.pos()
            roi_size = roi_item.size()
            current_roi_w = int(round(roi_size.x()))
            if current_roi_w != self.refinement_roi_size_spinbox.value() and \
               self.refinement_roi_size_spinbox.minimum() <= current_roi_w <= self.refinement_roi_size_spinbox.maximum() and \
               current_roi_w % 2 != 0:
                self.refinement_roi_size_spinbox.blockSignals(True)
                self.refinement_roi_size_spinbox.setValue(current_roi_w)
                self.refinement_roi_size_spinbox.blockSignals(False)
            
            self._clear_last_preview_gauss_fit()
            self._update_roi_previews()

    def _update_roi_previews(self): # Logika bardzo podobna do SubstrateSpotSelectionDialog
        # ... (Implementacja na podstawie SubstrateSpotSelectionDialog._update_roi_previews)
        # Pamiętaj, aby dostosować logowanie i ewentualnie kolory/shadery dla 3D
        if not self.selection_roi.isVisible() or self.fft_data is None: # type: ignore
            self._clear_last_preview_gauss_fit()# ... (czyszczenie widgetów podglądu) ...
            if hasattr(self, 'roi_preview_2d_image_item'): self.roi_preview_2d_image_item.clear()
            if hasattr(self, 'gaussian_preview_2d_image_item'): self.gaussian_preview_2d_image_item.clear()
            if hasattr(self, 'gl_roi_surface_plot_item') and self.gl_roi_surface_plot_item: self._clear_3d_surface(self.gl_roi_surface_plot_item)
            if hasattr(self, 'gl_gauss_surface_plot_item') and self.gl_gauss_surface_plot_item: self._clear_3d_surface(self.gl_gauss_surface_plot_item)
            return
        # ... reszta logiki kopiowana i dostosowana ...
        roi_state_for_comparison = self.selection_roi.getState()
        x0_roi, y0_roi = int(round(roi_state_for_comparison['pos'].x())), int(round(roi_state_for_comparison['pos'].y()))
        width_roi, height_roi = int(round(roi_state_for_comparison['size'].x())), int(round(roi_state_for_comparison['size'].y()))
        x1_roi, y1_roi = x0_roi + width_roi, y0_roi + height_roi
        max_ky, max_kx = self.fft_data.shape
        y0_cl = np.clip(y0_roi, 0, max_ky)
        y1_cl = np.clip(y1_roi, 0, max_ky)
        x0_cl = np.clip(x0_roi, 0, max_kx)
        x1_cl = np.clip(x1_roi, 0, max_kx) # type: ignore
        if y1_cl <= y0_cl or x1_cl <= x0_cl : 
            logger.warning("Invalid ROI slice for preview.")
            return # pragma: no cover
        roi_patch = self.fft_data[y0_cl:y1_cl, x0_cl:x1_cl]
        if roi_patch.size > 0:
            if self.enable_2d_roi_preview_checkbox.isChecked() and hasattr(self,'roi_preview_2d_image_item'): 
                self.roi_preview_2d_image_item.setImage(roi_patch.T)
                self.roi_preview_2d_plot.autoRange()
            elif hasattr(self,'roi_preview_2d_image_item'): self.roi_preview_2d_image_item.clear()
            if self.enable_3d_roi_preview_checkbox.isChecked() and hasattr(self,'gl_roi_surface_plot_item') and self.gl_roi_surface_plot_item: 
                self._update_3d_surface_plot(self.gl_roi_surface_plot_item, roi_patch)
            elif hasattr(self,'gl_roi_surface_plot_item') and self.gl_roi_surface_plot_item: 
                self._clear_3d_surface(self.gl_roi_surface_plot_item)
            if self.rb_refine_gaussian.isChecked():
                fitted_gauss_2d_for_preview = None
                if PEAK_FITTING_MODULE_AVAILABLE and SCIPY_OPTIMIZE_AVAILABLE and SCIPY_AVAILABLE:
                    patch_h, patch_w = roi_patch.shape
                    p_y,p_x=np.mgrid[0:patch_h,0:patch_w]
                    p_xy_flat=(p_y.flatten(),p_x.flatten())
                    p_data_flat=roi_patch.flatten()
                    try:
                        p0_gauss = [roi_patch.max()-roi_patch.min(),patch_h/2.,patch_w/2.,patch_w/4.,patch_h/4.,0.,roi_patch.min()]
                        if callable(scipy_curve_fit) and callable(_gaussian_2d):
                            popt_gauss,pcov_gauss=scipy_curve_fit(_gaussian_2d,p_xy_flat,p_data_flat,p0=p0_gauss,maxfev=3000)
                            self.last_preview_gauss_fit_popt=popt_gauss
                            abs_fit_ky=y0_roi+popt_gauss[1]
                            abs_fit_kx=x0_roi+popt_gauss[2]
                            self.last_preview_gauss_fit_center_abs=(abs_fit_kx,abs_fit_ky)
                            self.last_preview_gauss_roi_state=self.selection_roi.getState().copy()
                            logger.info(f"Adsorbate Preview GaussFit OK. Center: {self.last_preview_gauss_fit_center_abs}") # type: ignore
                            fitted_gauss_flat=_gaussian_2d(p_xy_flat,*popt_gauss)
                            fitted_gauss_2d_for_preview=fitted_gauss_flat.reshape(patch_h,patch_w)
                    except Exception as e:
                        logger.warning(f"Adsorbate GaussFit Preview failed: {e}")
                        self._clear_last_preview_gauss_fit()
                        fitted_gauss_2d_for_preview=roi_patch # pragma: no cover
                if self.enable_gauss_2d_preview_checkbox.isChecked() and hasattr(self,'gaussian_preview_2d_image_item'):
                    if fitted_gauss_2d_for_preview is not None: self.gaussian_preview_2d_image_item.setImage(fitted_gauss_2d_for_preview.T)
                    else: self.gaussian_preview_2d_image_item.setImage(roi_patch.T)
                    self.gaussian_preview_2d_plot.autoRange()
                elif hasattr(self,'gaussian_preview_2d_image_item'):self.gaussian_preview_2d_image_item.clear()
                if self.enable_gauss_3d_preview_checkbox.isChecked() and hasattr(self,'gl_gauss_surface_plot_item') and self.gl_gauss_surface_plot_item:
                    if fitted_gauss_2d_for_preview is not None: self._update_3d_surface_plot(self.gl_gauss_surface_plot_item,fitted_gauss_2d_for_preview)
                    else: self._update_3d_surface_plot(self.gl_gauss_surface_plot_item,roi_patch)
                elif hasattr(self,'gl_gauss_surface_plot_item') and self.gl_gauss_surface_plot_item: self._clear_3d_surface(self.gl_gauss_surface_plot_item)
            else: 
                self._clear_last_preview_gauss_fit()
                if hasattr(self,'gaussian_preview_2d_image_item'):self.gaussian_preview_2d_image_item.clear()
                if hasattr(self,'gl_gauss_surface_plot_item') and self.gl_gauss_surface_plot_item: self._clear_3d_surface(self.gl_gauss_surface_plot_item)
        else: # pragma: no cover
            if hasattr(self,'roi_preview_2d_image_item'):self.roi_preview_2d_image_item.clear()
            if hasattr(self,'gaussian_preview_2d_image_item'):self.gaussian_preview_2d_image_item.clear()
            if hasattr(self,'gl_roi_surface_plot_item') and self.gl_roi_surface_plot_item: 
                self._clear_3d_surface(self.gl_roi_surface_plot_item)
            if hasattr(self,'gl_gauss_surface_plot_item') and self.gl_gauss_surface_plot_item: 
                self._clear_3d_surface(self.gl_gauss_surface_plot_item)


    def _update_3d_surface_plot(self, surface_item: GLSurfacePlotItem, data_2d: Optional[np.ndarray]):
        if data_2d is None or data_2d.size == 0 or data_2d.ndim != 2: 
            self._clear_3d_surface(surface_item)
            return
        h, w = data_2d.shape
        if h < 2 or w < 2: 
            self._clear_3d_surface(surface_item)
            return
        x = np.linspace(-w/2., w/2., w)
        y = np.linspace(-h/2., h/2., h)
        z_norm = (data_2d - data_2d.min()) / (data_2d.max() - data_2d.min() + 1e-9)
        colors = np.zeros((h,w,4), dtype=np.float32)
        colors[...,0]=z_norm
        colors[...,2]=1-z_norm
        colors[...,3]=0.7 # R, B, Alpha
        surface_item.setData(x=x,y=y,z=data_2d.T,colors=colors.transpose(1,0,2))

    def _clear_3d_surface(self, surface_item: Optional[GLSurfacePlotItem]):
        if surface_item:
            x=np.array([0,1e-9])
            y=np.array([0,1e-9])
            z=np.array([[0,0],[0,0]],dtype=np.float32)
            colors=np.array([[[0,0,0,0],[0,0,0,0]],[[0,0,0,0],[0,0,0,0]]],dtype=np.float32)
            try: 
                surface_item.setData(x=x,y=y,z=z,colors=colors)
                surface_item.meshDataChanged()
            except Exception as e: 
                logger.error(f"Error clearing 3D surface: {e}") # pragma: no cover

    @pyqtSlot()
    def _on_refinement_method_changed(self):
        is_gaussian_mode = self.rb_refine_gaussian.isChecked()
        if hasattr(self, 'gauss_2d_container'): self.gauss_2d_container.setVisible(is_gaussian_mode)
        if hasattr(self, 'gauss_3d_container'): self.gauss_3d_container.setVisible(is_gaussian_mode)
        if self.rb_refine_direct.isChecked(): 
            self.current_refinement_method=REFINEMENT_DIRECT_CLICK
            self.refinement_roi_size_spinbox.setEnabled(False)
            self.selection_roi.setVisible(False)
            self.add_adsorbate_spot_button.setEnabled(False)
            self.status_label.setText("Click directly on FFT to add adsorbate spot.")
        else: 
            self.current_refinement_method = REFINEMENT_MAX_PIXEL if self.rb_refine_max_pixel.isChecked() else REFINEMENT_GAUSSIAN_FIT
            self.refinement_roi_size_spinbox.setEnabled(True)
            self.add_adsorbate_spot_button.setEnabled(self.selection_roi.isVisible())
            self.status_label.setText("Click on FFT to place ROI, or drag ROI. Then Add Spot.")
        if not is_gaussian_mode: self._clear_last_preview_gauss_fit()
        self._update_roi_previews()
        logger.debug(f"Adsorbate refinement method: {self.current_refinement_method}")

    @pyqtSlot(int)
    def _on_refinement_roi_size_changed(self, value: int):
        self.refinement_roi_size = value
        self._clear_last_preview_gauss_fit()
        if self.selection_roi.isVisible():
            current_pos = self.selection_roi.pos()
            old_size = self.selection_roi.size()
            center_x = current_pos.x()+old_size.x()/2
            center_y = current_pos.y()+old_size.y()/2
            new_pos_x = center_x-value/2
            new_pos_y = center_y-value/2
            self.selection_roi.setPos((new_pos_x, new_pos_y), update=False)
            self.selection_roi.setSize((value,value), update=False)
            self._handle_roi_region_changing()
        logger.debug(f"Adsorbate refinement ROI size: {self.refinement_roi_size}")

    @pyqtSlot()
    def _add_current_adsorbate_spot_from_roi(self):
        if not self.selection_roi.isVisible() or self.fft_data is None: 
            self.status_label.setText("Error: No ROI or FFT data.")
            return # pragma: no cover
        roi_state=self.selection_roi.getState()
        x0,y0=int(round(roi_state['pos'].x())),int(round(roi_state['pos'].y()))
        w,h=int(round(roi_state['size'].x())),int(round(roi_state['size'].y()))
        ckx,cky=x0+w//2,y0+h//2
        ref_kx,ref_ky = float(ckx),float(cky) # Domyślnie środek ROI

        if self.current_refinement_method == REFINEMENT_MAX_PIXEL and PEAK_FITTING_MODULE_AVAILABLE:
            pr = self.refinement_roi_size//2
            max_h,max_w=self.fft_data.shape
            eff_cky=np.clip(cky,pr,max_h-1-pr)
            eff_ckx=np.clip(ckx,pr,max_w-1-pr)
            fky,fkx=find_max_pixel_in_roi(self.fft_data,(eff_cky,eff_ckx),pr)
            ref_kx,ref_ky=float(fkx),float(fky)
        elif self.current_refinement_method == REFINEMENT_GAUSSIAN_FIT and PEAK_FITTING_MODULE_AVAILABLE and SCIPY_AVAILABLE:
            curr_roi_state=self.selection_roi.getState()
            roi_state_match = False
            if self.last_preview_gauss_roi_state and curr_roi_state:
                # Proste porównanie, można dodać tolerancję
                if self.last_preview_gauss_roi_state['pos'] == curr_roi_state['pos'] and \
                   self.last_preview_gauss_roi_state['size'] == curr_roi_state['size']:
                    roi_state_match = True

            if self.last_preview_gauss_fit_center_abs and roi_state_match:
                ref_kx,ref_ky = self.last_preview_gauss_fit_center_abs
                logger.info(f"Using PREVIEW GaussFit for Adsorbate: ({ref_kx:.2f},{ref_ky:.2f})")
            else:
                pr=self.refinement_roi_size//2
                max_h,max_w=self.fft_data.shape
                eff_cky=np.clip(cky,pr,max_h-1-pr)
                eff_ckx=np.clip(ckx,pr,max_w-1-pr)
                fit_res=fit_2d_gaussian_in_roi(self.fft_data,(eff_cky,eff_ckx),pr)
                if fit_res: 
                    _popt, (fky_abs,fkx_abs), _patch = fit_res
                    ref_kx,ref_ky=float(fkx_abs),float(fky_abs)
                    logger.info(f"NEW Adsorbate GaussFit: ({ref_kx:.2f},{ref_ky:.2f})")
                else: 
                    logger.warning("Adsorbate GaussFit FAILED for Add Spot. Using ROI center.") # pragma: no cover
        
        new_spot = (ref_kx, ref_ky)
        if new_spot not in self.selected_adsorbate_spots_raw: 
            self.selected_adsorbate_spots_raw.append(new_spot)
            self._update_adsorbate_spots_list_widget()
            self._redraw_all_markers_in_dialog()
            self.status_label.setText(f"Adsorbate spot {len(self.selected_adsorbate_spots_raw)} added.")
        else: 
            self.status_label.setText(f"Adsorbate spot ({ref_kx:.2f},{ref_ky:.2f}) already selected.") # pragma: no cover
        self._clear_last_preview_gauss_fit()
        self._update_correction_button_state()


    def _update_adsorbate_spots_list_widget(self):
        self.spots_list_widget.clear()
        for i, (kx,ky) in enumerate(self.selected_adsorbate_spots_raw): 
            self.spots_list_widget.addItem(f"A{i+1} (S{self.adsorbate_set_index+1}): ({kx:.2f},{ky:.2f})")
        self._update_add_spot_button_state()
        self._update_correction_button_state()

    @pyqtSlot()
    def _remove_selected_spot(self):
        ci = self.spots_list_widget.currentItem()
        if ci: row=self.spots_list_widget.row(ci)
        if 0 <= row < len(self.selected_adsorbate_spots_raw): 
            del self.selected_adsorbate_spots_raw[row]
            self._update_adsorbate_spots_list_widget()
            self._redraw_all_markers_in_dialog()
            logger.debug(f"Removed adsorbate spot at idx {row}")

    @pyqtSlot()
    def _clear_all_spots_in_dialog(self):
        self.selected_adsorbate_spots_raw.clear()
        self.corrected_adsorbate_spots_in_ideal_system.clear()
        self._update_adsorbate_spots_list_widget()
        self._redraw_all_markers_in_dialog()
        logger.debug("Cleared all adsorbate spots in dialog.")

    def _redraw_all_markers_in_dialog(self):
        logger.debug("AdsorbateDialog: Redrawing all markers...")
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

        if self.selected_adsorbate_spots_raw:
            d=[{'pos':s,'symbol':'o','size':10,'pen':pg.mkPen('b',width=1.5),'brush':pg.mkBrush(0,0,255,120)} for s in self.selected_adsorbate_spots_raw]
            self.raw_adsorbate_spot_markers=ScatterPlotItem(spots=d)
            self.fft_view_box.addItem(self.raw_adsorbate_spot_markers)
        if self.show_ideal_substrate_checkbox.isChecked() and self.ideal_substrate_spots_to_display_px:
            d=[{'pos':s,'symbol':'+','size':12,'pen':pg.mkPen('m',width=1.5)} for s in self.ideal_substrate_spots_to_display_px]
            self.ideal_substrate_marker_item=ScatterPlotItem(spots=d)
            self.fft_view_box.addItem(self.ideal_substrate_marker_item)
        if self.show_fitted_substrate_checkbox.isChecked() and self.fitted_substrate_spots_to_display_px:
            d=[{'pos':s,'symbol':'x','size':12,'pen':pg.mkPen('c',width=2.0)} for s in self.fitted_substrate_spots_to_display_px]
            self.fitted_substrate_marker_item=ScatterPlotItem(spots=d)
            self.fft_view_box.addItem(self.fitted_substrate_marker_item)
        
        if self.show_corrected_adsorbate_checkbox.isChecked() and self.corrected_adsorbate_spots_in_ideal_system and self.sub_F_m2i is not None and self.sub_t_m2i is not None:
            try:
                F_inv=np.linalg.inv(self.sub_F_m2i)
                t_prime=(-self.sub_t_m2i@F_inv.T).flatten() # type: ignore
                from ...analysis.drift_correction import apply_affine_transform
                spots_draw=apply_affine_transform(np.array(self.corrected_adsorbate_spots_in_ideal_system),F_inv,t_prime)
                if self.corrected_adsorbate_spots_in_ideal_system is not None: d=[{'pos':tuple(p),'symbol':'s','size':10,'pen':pg.mkPen('r',width=1.5),'brush':pg.mkBrush(255,0,0,120)} for p in self.corrected_adsorbate_spots_in_ideal_system]
                self.corrected_adsorbate_marker_item=ScatterPlotItem(spots=d)
                self.fft_view_box.addItem(self.corrected_adsorbate_marker_item)
            except Exception as e:logger.error(f"Error transforming corrected adsorbate spots for display: {e}") # pragma: no cover

    def _handle_fft_image_click(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            pos_vb = self.fft_view_box.mapSceneToView(event.scenePos())
            mapped_pos = self.fft_image_item.mapToData(pos_vb)
            if mapped_pos:
                kx,ky=int(round(mapped_pos.x())),int(round(mapped_pos.y()))
                logger.debug(f"Ads Dlg FFT click: (kx,ky)=({kx},{ky})")
                rs=self.refinement_roi_size_spinbox.value()
                rx=kx-rs//2
                ry=ky-rs//2
                if self.fft_data is not None: 
                    my,mx=self.fft_data.shape
                    rx=np.clip(rx,0,mx-rs)
                    ry=np.clip(ry,0,my-rs)
                self.selection_roi.setPos((rx,ry),update=False)
                self.selection_roi.setSize((rs,rs),update=False)
                self.selection_roi.setVisible(True)
                self.add_adsorbate_spot_button.setEnabled(True)
                self._update_roi_previews()
            event.accept()
        else: event.ignore() # pragma: no cover
            
    def _update_add_spot_button_state(self):
        # Prosta logika, można dodać limit, jeśli potrzebny dla adsorbatu
        self.add_adsorbate_spot_button.setEnabled(self.selection_roi.isVisible())
        num_sel = len(self.selected_adsorbate_spots_raw)
        # Można dodać minimalną wymaganą liczbę, np. 3
        min_req = 3 
        if num_sel < min_req : self.status_label.setText(f"Select at least {min_req-num_sel} more adsorbate spot(s). Current: {num_sel}.")
        else: self.status_label.setText(f"Selected {num_sel} adsorbate spots. Ready to add or correct.")


    def _update_correction_button_state(self):
        can_correct = bool(self.sub_F_m2i is not None and self.sub_t_m2i is not None and self.selected_adsorbate_spots_raw)
        self.apply_correction_button.setEnabled(can_correct)
        if not (self.sub_F_m2i is not None and self.sub_t_m2i is not None):
            self.sub_transform_info_label_status.setText("Status: Substrate transform not available to apply.")

    def _display_substrate_transform_info(self):
        if self.sub_transform_analysis:
            self.sub_transform_info_label_status.setText("Status: Substrate transform data available.")
            self.sub_transform_info_label_rot.setText(f"Sub. Rotation (M->I): {self.sub_transform_analysis.get('rotation_angle_deg', 'N/A'):.2f}°")
            s_x,s_y = self.sub_transform_analysis.get('principal_stretches',[np.nan,np.nan])
            self.sub_transform_info_label_scale.setText(f"Sub. Stretches (M->I): ({s_x:.3f}, {s_y:.3f})")
            self.sub_transform_info_label_rmse.setText(f"Sub. Fit RMSE (M->I, px): {self.sub_transform_analysis.get('rmse', 'N/A'):.3f}")
        else:
            self.sub_transform_info_label_status.setText("Status: Substrate transform not passed to dialog.")
            self.sub_transform_info_label_rot.setText("Sub. Rotation: -")
            self.sub_transform_info_label_scale.setText("Sub. Scale (X,Y): -")
            self.sub_transform_info_label_rmse.setText("Sub. RMSE (px): -")
        self._update_correction_button_state() # Zaktualizuj stan przycisku po wyświetleniu info

    @pyqtSlot()
    def _on_apply_substrate_correction_clicked(self):
        logger.info("Apply Substrate Correction button clicked in Adsorbate Dialog.")
        if self.sub_F_m2i is None or self.sub_t_m2i is None or not self.selected_adsorbate_spots_raw:
            QMessageBox.warning(self, "Cannot Correct", "Substrate transformation data is not available or no adsorbate spots are selected to correct.")
            return

        try:
            from ...analysis.drift_correction import apply_affine_transform # Upewnij się, że import jest OK
            raw_spots_np = np.array(self.selected_adsorbate_spots_raw, dtype=float)
            corrected_spots_np = apply_affine_transform(raw_spots_np, self.sub_F_m2i, self.sub_t_m2i)
            print(f"corrected_spots_np: {corrected_spots_np}")
            
            if corrected_spots_np is not None:
                self.corrected_adsorbate_spots_in_ideal_system = [tuple(pt) for pt in corrected_spots_np]
                logger.info(f"Applied substrate correction to {len(self.corrected_adsorbate_spots_in_ideal_system)} adsorbate spots.")
                self._redraw_all_markers_in_dialog() # Aby pokazać skorygowane, jeśli checkbox zaznaczony
                self.status_label.setText(f"{len(self.corrected_adsorbate_spots_in_ideal_system)} adsorbate spots corrected (in ideal system).")
            else: raise ValueError("apply_affine_transform returned None for adsorbate spots.") # pragma: no cover
        except Exception as e: # pragma: no cover
            logger.error(f"Error applying substrate correction to adsorbate spots: {e}")
            QMessageBox.critical(self, "Correction Error", f"Could not apply correction: {e}")

    def get_dialog_results(self) -> Dict[str, Any]:
        return {
            "raw_adsorbate_spots": list(self.selected_adsorbate_spots_raw),
            "corrected_adsorbate_spots_in_ideal_system": list(self.corrected_adsorbate_spots_in_ideal_system),
            "adsorbate_set_index": self.adsorbate_set_index
        }

    def accept(self):
        # Można dodać walidację, np. czy wybrano przynajmniej 3 piki adsorbatu, jeśli to potrzebne do dalszej analizy
        # if len(self.selected_adsorbate_spots_raw) < 3 and not self.corrected_adsorbate_spots_in_ideal_system:
        #     reply = QMessageBox.question(self, "Few Spots Selected", 
        #                                  f"Only {len(self.selected_adsorbate_spots_raw)} raw spots selected for set {self.adsorbate_set_index + 1}. "
        #                                  "At least 3 are often needed for lattice definition. Continue anyway?",
        #                                  QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
        #     if reply == QMessageBox.StandardButton.No:
        #         return
        
        logger.info(f"AdsorbateSpotSelectionDialog for set {self.adsorbate_set_index + 1} accepted. "
                    f"Raw spots: {len(self.selected_adsorbate_spots_raw)}, "
                    f"Corrected spots: {len(self.corrected_adsorbate_spots_in_ideal_system)}")
        super().accept()

    def reject(self):
        logger.info(f"AdsorbateSpotSelectionDialog for set {self.adsorbate_set_index + 1} rejected.")
        super().reject()

    def closeEvent(self, event):
        logger.debug("AdsorbateSpotSelectionDialog closing. Cleaning up GL items.")
        if hasattr(self, 'gl_roi_view_widget') and self.gl_roi_view_widget:
            if hasattr(self, 'gl_roi_surface_plot_item') and self.gl_roi_surface_plot_item: self.gl_roi_view_widget.removeItem(self.gl_roi_surface_plot_item)
            self.gl_roi_view_widget.setParent(None)
            self.gl_roi_view_widget.deleteLater() # Lepsze czyszczenie
        if hasattr(self, 'gl_gauss_view_widget') and self.gl_gauss_view_widget:
            if hasattr(self, 'gl_gauss_surface_plot_item') and self.gl_gauss_surface_plot_item: self.gl_gauss_view_widget.removeItem(self.gl_gauss_surface_plot_item)
            self.gl_gauss_view_widget.setParent(None)
            self.gl_gauss_view_widget.deleteLater()
        super().closeEvent(event)