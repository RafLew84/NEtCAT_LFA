# lfa/gui/dialogs/substrate_spot_dialog.py
import logging
import math
from typing import List, Tuple, Optional, Dict, Any
import numpy as np

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QDialogButtonBox,
    QLabel, QListWidget, QAbstractItemView, QWidget, QSplitter, QGroupBox,
    QFormLayout, QRadioButton, QSpinBox, QComboBox, QCheckBox, QMessageBox,
    QGridLayout, QDoubleSpinBox, QApplication, QDockWidget
)
from PyQt6.QtCore import Qt, pyqtSlot, QPointF

from ..utils.display import format_float, format_pair
from .scenes import SubstrateSpotScene, MarkerSpec
from .presenters.substrate_spot_presenter import (
    SubstrateSpotPresenter,
    SubstrateSpotState,
    TransformComputationError,
)

try:
    import pyqtgraph as pg
    ImageView = pg.ImageView
    PlotItem = pg.PlotItem
    ImageItem = pg.ImageItem
    RectROI = pg.RectROI
    ScatterPlotItem = pg.ScatterPlotItem
    ViewBox = pg.ViewBox
    GraphicsLayoutWidget = pg.GraphicsLayoutWidget
    PYQTGRAPH_AVAILABLE = True
except ImportError:
    ImageView = None
    PlotItem = None
    ImageItem = None
    RectROI = None
    ScatterPlotItem = None
    ViewBox = None
    GraphicsLayoutWidget = None
    PYQTGRAPH_AVAILABLE = False
    logging.error("SubstrateSpotSelectionDialog: PyQtGraph or pyqtgraph.opengl not found.")

try:
    from scipy.optimize import curve_fit as scipy_curve_fit 
    SCIPY_OPTIMIZE_AVAILABLE = True
except ImportError:
    logging.error("SubstrateSpotSelectionDialog: SciPy (for curve_fit) not found.")
    SCIPY_OPTIMIZE_AVAILABLE = False
    def scipy_curve_fit(*args, **kwargs): raise ImportError("scipy.optimize.curve_fit is not available")


try:
    from ...analysis.peak_fitting import (
        find_max_pixel_in_roi,
        fit_2d_gaussian_in_roi,
        _gaussian_2d,
        SCIPY_AVAILABLE,
    )
    from ...analysis.lattice import (
        KNOWN_LATTICES,
        get_reciprocal_points,
        get_nearest_reciprocal_points,
    )
    from ...core.constants import (
        LATTICE_TYPE_CUSTOM,
        LATTICE_TYPE_HEXAGONAL,
        LATTICE_TYPE_SQUARE,
        PREDEFINED_SUBSTRATE_CUSTOM,
        PREDEFINED_SUBSTRATE_FROM_SELECTION,
        PREDEFINED_SUBSTRATE_NONE,
        REFINEMENT_DIRECT_CLICK,
        REFINEMENT_GAUSSIAN_FIT,
        REFINEMENT_MAX_PIXEL,
    )
    from ...core.history import HistoryNode
    from ...logic.history_manager import HistoryManager
    PEAK_FITTING_MODULE_AVAILABLE = True
except ImportError: # pragma: no cover
    PEAK_FITTING_MODULE_AVAILABLE = False
    SCIPY_AVAILABLE = False
    KNOWN_LATTICES = {}
    logging.error("SubstrateSpotSelectionDialog: Could not import peak_fitting or lattice modules.")
    def find_max_pixel_in_roi(data, center, radius): return center
    def fit_2d_gaussian_in_roi(data, center, radius): return None
    def _gaussian_2d(*args, **kwargs): raise ImportError("Gaussian 2D function is not available")

logger = logging.getLogger(__name__)

class SubstrateSpotSelectionDialog(QDialog):
    def __init__(self,
                 fft_image_data: Optional[np.ndarray],
                 history_manager: HistoryManager,
                 current_fft_node_id: str,
                 current_spots: Optional[List[Tuple[float, float]]] = None,
                 default_refinement_method: str = REFINEMENT_DIRECT_CLICK,
                 default_refinement_roi_size: int = 5,
                 initial_lattice_type: str = LATTICE_TYPE_HEXAGONAL,
                 initial_selected_substrate_name: Optional[str] = None, 
                 initial_custom_a_surf: Optional[float] = None,
                 initial_custom_definition: Optional[Dict[str, Any]] = None,
                 initial_transform_F: Optional[np.ndarray] = None, 
                 initial_transform_t: Optional[np.ndarray] = None,
                 initial_fitted_spots: Optional[List[Tuple[float, float]]] = None,
                 parent=None):
        super().__init__(parent)
        
        if not PYQTGRAPH_AVAILABLE: # pragma: no cover
            err_layout = QVBoxLayout(self)
            err_layout.addWidget(QLabel("Critical Error: PyQtGraph library is not available.\nThis dialog cannot function."))
            self.setWindowTitle("Error")
            return

        self.setWindowTitle("Select Substrate Spots")
        self.setMinimumSize(1200, 750)
        current_flags=self.windowFlags()
        self.setWindowFlags(current_flags | Qt.WindowType.WindowMinimizeButtonHint | Qt.WindowType.WindowMaximizeButtonHint)


        self.fft_data = fft_image_data
        self.history_manager = history_manager
        self.current_fft_node_id = current_fft_node_id

        self.selected_spots: List[Tuple[float, float]] = list(current_spots) if current_spots else []

        self.current_refinement_method = default_refinement_method
        self.refinement_roi_size = default_refinement_roi_size

        self.current_lattice_type: str = initial_lattice_type
        self.current_a_surf: Optional[float] = None
        self.current_custom_definition: Optional[Dict[str, Any]] = None
        self.limits_per_lattice = {
            LATTICE_TYPE_HEXAGONAL: 6,
            LATTICE_TYPE_SQUARE: 4,
            LATTICE_TYPE_CUSTOM: 0
        }
        self._initial_selected_substrate_name = initial_selected_substrate_name
        self._initial_custom_a_surf = initial_custom_a_surf
        self._initial_custom_definition = dict(initial_custom_definition) if initial_custom_definition else None
        if self._initial_custom_definition:
            self.current_custom_definition = dict(self._initial_custom_definition)
            if self._initial_custom_definition.get("a_length_nm"):
                self.current_a_surf = self._initial_custom_definition.get("a_length_nm")

        self.last_preview_gauss_fit_popt: Optional[np.ndarray] = None
        self.last_preview_gauss_fit_center_abs: Optional[Tuple[float, float]] = None
        self.last_preview_gauss_roi_state: Optional[Dict] = None

        self.substrate_transformation_matrix_F: Optional[np.ndarray] = initial_transform_F
        self.substrate_translation_vector_t: Optional[np.ndarray] = initial_transform_t
        self.substrate_transform_analysis: Optional[Dict[str, Any]] = None
        self.fitted_substrate_spots_px: List[Tuple[float, float]] = list(initial_fitted_spots) if initial_fitted_spots else []
        self.calculated_ideal_substrate_spots_px: List[Tuple[float, float]] = []

        presenter_state = SubstrateSpotState(
            selected_spots=self.selected_spots,
            lattice_type=self.current_lattice_type,
            selected_definition=self._initial_selected_substrate_name,
            custom_definition=self.current_custom_definition,
            custom_a_surf=self.current_a_surf,
            transform_matrix_F=self.substrate_transformation_matrix_F,
            transform_translation_t=self.substrate_translation_vector_t,
            transform_analysis=self.substrate_transform_analysis,
            fitted_spots_px=self.fitted_substrate_spots_px,
            ideal_spots_px_for_reference=self.calculated_ideal_substrate_spots_px,
        )
        self.presenter = SubstrateSpotPresenter(
            history_manager=self.history_manager,
            fft_node_id=self.current_fft_node_id,
            fft_data=self.fft_data,
            state=presenter_state,
        )
        self.scene = SubstrateSpotScene()

        self._init_ui()
        self._connect_signals()
        self._update_spots_list_widget()
        self._redraw_all_spot_markers()
        self._update_add_spot_button_state()

        if self.current_refinement_method == REFINEMENT_MAX_PIXEL: 
            self.rb_refine_max_pixel.setChecked(True)
        elif self.current_refinement_method == REFINEMENT_GAUSSIAN_FIT: 
            self.rb_refine_gaussian.setChecked(True)
        else: 
            self.rb_refine_direct.setChecked(True)
        self.refinement_roi_size_spinbox.setValue(self.refinement_roi_size)
        
        self._on_refinement_method_changed()
        self._on_lattice_type_changed()

        logger.debug("SubstrateSpotSelectionDialog initialized.")

    def _init_ui(self):
        top_level_layout = QHBoxLayout(self)
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        top_level_layout.addWidget(main_splitter)

        left_controls_widget = QWidget()
        left_controls_layout = QVBoxLayout(left_controls_widget)
        left_controls_widget.setMinimumWidth(300)
        left_controls_widget.setMaximumWidth(350)

        substrate_def_group = QGroupBox("Substrate Definition & Overlay")
        substrate_def_layout = QFormLayout(substrate_def_group)
        self.lattice_type_combo = QComboBox()
        self.lattice_type_combo.addItems([
            LATTICE_TYPE_HEXAGONAL.capitalize(),
            LATTICE_TYPE_SQUARE.capitalize(),
            LATTICE_TYPE_CUSTOM.capitalize()
        ])
        substrate_def_layout.addRow("Lattice Type:", self.lattice_type_combo)
        self.substrate_definition_combo = QComboBox()
        substrate_def_layout.addRow("Predefined/Custom:", self.substrate_definition_combo)
        self.custom_a_surf_spinbox = QDoubleSpinBox()
        self.custom_a_surf_spinbox.setDecimals(4)
        self.custom_a_surf_spinbox.setRange(0.001, 100.0)
        self.custom_a_surf_spinbox.setSingleStep(0.001)
        self.custom_a_surf_spinbox.setValue(0.3)
        self.custom_a_surf_label = QLabel("Custom 'a_surf' (nm):")

        substrate_def_layout.addRow(self.custom_a_surf_label, self.custom_a_surf_spinbox)
        self.custom_a_surf_label.setVisible(False)
        self.custom_a_surf_spinbox.setVisible(False)

        self.custom_vec_a_label = QLabel("Vector |a| (nm):")
        self.custom_vec_a_spinbox = QDoubleSpinBox()
        self.custom_vec_a_spinbox.setDecimals(4)
        self.custom_vec_a_spinbox.setRange(0.0001, 50.0)
        self.custom_vec_a_spinbox.setSingleStep(0.001)
        self.custom_vec_a_spinbox.setValue(0.300)
        substrate_def_layout.addRow(self.custom_vec_a_label, self.custom_vec_a_spinbox)

        self.custom_vec_b_label = QLabel("Vector |b| (nm):")
        self.custom_vec_b_spinbox = QDoubleSpinBox()
        self.custom_vec_b_spinbox.setDecimals(4)
        self.custom_vec_b_spinbox.setRange(0.0001, 50.0)
        self.custom_vec_b_spinbox.setSingleStep(0.001)
        self.custom_vec_b_spinbox.setValue(0.300)
        substrate_def_layout.addRow(self.custom_vec_b_label, self.custom_vec_b_spinbox)

        self.custom_vec_angle_label = QLabel("Angle γ (deg):")
        self.custom_vec_angle_spinbox = QDoubleSpinBox()
        self.custom_vec_angle_spinbox.setDecimals(2)
        self.custom_vec_angle_spinbox.setRange(1.0, 179.0)
        self.custom_vec_angle_spinbox.setSingleStep(0.1)
        self.custom_vec_angle_spinbox.setValue(60.0)
        substrate_def_layout.addRow(self.custom_vec_angle_label, self.custom_vec_angle_spinbox)

        for widget in (self.custom_vec_a_label, self.custom_vec_a_spinbox,
                       self.custom_vec_b_label, self.custom_vec_b_spinbox,
                       self.custom_vec_angle_label, self.custom_vec_angle_spinbox):
            widget.setVisible(False)

        self.show_ideal_lattice_checkbox = QCheckBox("Show Ideal Lattice Overlay")
        self.show_ideal_lattice_checkbox.setChecked(True)
        substrate_def_layout.addRow(self.show_ideal_lattice_checkbox)
        left_controls_layout.addWidget(substrate_def_group)

        refinement_group = QGroupBox("Spot Refinement")
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

        self.add_spot_button = QPushButton("Add/Update Spot from ROI")
        self.add_spot_button.setEnabled(False)
        left_controls_layout.addWidget(self.add_spot_button)
        
        transform_group = QGroupBox("Lattice Transformation")
        transform_layout = QFormLayout(transform_group)
        self.transform_status_label = QLabel("Select spots and define lattice.")
        self.calculate_transform_button = QPushButton("Calculate Transformation")
        self.calculate_transform_button.setEnabled(False)
        self.rotation_angle_label = QLabel("Rotation: -")
        self.scale_factor_label = QLabel("Scale (X,Y): -")
        self.rmse_label = QLabel("RMSE (px): -")

        transform_layout.addRow(self.transform_status_label)
        transform_layout.addRow(self.calculate_transform_button)
        transform_layout.addRow(self.rotation_angle_label)
        transform_layout.addRow(self.scale_factor_label)
        transform_layout.addRow(self.rmse_label)
        left_controls_layout.addWidget(transform_group)

        left_controls_layout.addStretch(1)
        main_splitter.addWidget(left_controls_widget)
        self.fft_plot_widget = self.scene.widget()
        self.fft_view_box = self.scene.view_box
        self.fft_image_item = self.scene.image_item
        self.fft_histogram = self.scene.histogram
        self.selection_roi = self.scene.roi()

        if self.fft_data is not None:
            self.scene.set_image(self.fft_data)

        self.selection_roi.setSize((self.refinement_roi_size, self.refinement_roi_size))
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
        self.roi_preview_2d_widget.setMinimumHeight(150)
        self.roi_preview_2d_widget.setMaximumHeight(200)
        self.roi_preview_2d_plot = self.roi_preview_2d_widget.addViewBox(lockAspect=True, invertY=True)
        self.roi_preview_2d_image_item = ImageItem()
        self.roi_preview_2d_plot.addItem(self.roi_preview_2d_image_item)
        roi_2d_v_layout.addWidget(self.roi_preview_2d_widget, 1)
        preview_grid_layout.addWidget(roi_2d_container, 0, 0)

        gauss_2d_container = QWidget()
        gauss_2d_v_layout = QVBoxLayout(gauss_2d_container)
        gauss_2d_v_layout.addWidget(QLabel("Gaussian Fit 2D Preview:"))
        self.enable_gauss_2d_preview_checkbox = QCheckBox("Enable")
        self.enable_gauss_2d_preview_checkbox.setChecked(True)
        gauss_2d_v_layout.addWidget(self.enable_gauss_2d_preview_checkbox)

        self.gaussian_preview_2d_widget = GraphicsLayoutWidget()
        self.gaussian_preview_2d_widget.setMinimumHeight(150)
        self.gaussian_preview_2d_widget.setMaximumHeight(200)
        self.gaussian_preview_2d_plot = self.gaussian_preview_2d_widget.addViewBox(lockAspect=True, invertY=True)
        self.gaussian_preview_2d_image_item = ImageItem()
        self.gaussian_preview_2d_plot.addItem(self.gaussian_preview_2d_image_item)
        gauss_2d_v_layout.addWidget(self.gaussian_preview_2d_widget, 1)
        preview_grid_layout.addWidget(gauss_2d_container, 1, 0)
        
        preview_grid_layout.setColumnStretch(0,1)
        preview_grid_layout.setColumnStretch(1,1)
        preview_grid_layout.setRowStretch(0,1)
        preview_grid_layout.setRowStretch(1,1)
        
        self.gauss_2d_container = gauss_2d_container
        self.gauss_2d_container.setVisible(False)
        right_panel_layout.addWidget(preview_group)

        spots_list_group = QGroupBox("Selected Spots Management")
        spots_list_layout = QVBoxLayout(spots_list_group)
        self.spots_list_widget = QListWidget()
        self.spots_list_widget.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        spots_list_layout.addWidget(self.spots_list_widget)
        spot_buttons_layout = QHBoxLayout()
        self.remove_spot_button = QPushButton("Remove Selected")
        self.clear_all_spots_button = QPushButton("Clear All")
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
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        self.remove_spot_button.clicked.connect(self._remove_selected_spot)
        self.clear_all_spots_button.clicked.connect(self._clear_all_spots_in_dialog)

        self.fft_view_box.scene().sigMouseClicked.connect(self._handle_fft_image_click)
        self.selection_roi.sigRegionChanged.connect(self._handle_roi_region_changing)

        self.rb_refine_direct.toggled.connect(self._on_refinement_method_changed)
        self.rb_refine_max_pixel.toggled.connect(self._on_refinement_method_changed)
        self.rb_refine_gaussian.toggled.connect(self._on_refinement_method_changed)
        self.refinement_roi_size_spinbox.valueChanged.connect(self._on_refinement_roi_size_changed)

        self.lattice_type_combo.currentTextChanged.connect(self._on_lattice_type_changed)
        self.substrate_definition_combo.currentTextChanged.connect(self._on_substrate_definition_combo_changed)
        self.custom_a_surf_spinbox.valueChanged.connect(self._on_custom_a_surf_changed)
        self.custom_vec_a_spinbox.valueChanged.connect(self._on_custom_vectors_changed)
        self.custom_vec_b_spinbox.valueChanged.connect(self._on_custom_vectors_changed)
        self.custom_vec_angle_spinbox.valueChanged.connect(self._on_custom_vectors_changed)
        self.show_ideal_lattice_checkbox.stateChanged.connect(self._redraw_ideal_lattice_overlay)

        self.add_spot_button.clicked.connect(self._add_current_roi_spot)
        
        self.lattice_type_combo.currentTextChanged.connect(self._on_lattice_type_changed)
        self.show_ideal_lattice_checkbox.stateChanged.connect(self._redraw_ideal_lattice_overlay)

        self.enable_2d_roi_preview_checkbox.stateChanged.connect(self._update_roi_previews)
        self.enable_gauss_2d_preview_checkbox.stateChanged.connect(self._update_roi_previews)

        if hasattr(self, 'calculate_transform_button'): 
            self.calculate_transform_button.clicked.connect(self._on_calculate_transform_clicked)
    def _update_spots_list_widget(self):
        self.spots_list_widget.clear()
        for i, (kx, ky) in enumerate(self.selected_spots):
            point_text = format_pair((kx, ky), precision=2)
            self.spots_list_widget.addItem(f"S{i+1}: {point_text}")
        self._update_add_spot_button_state()

    def _populate_substrate_definition_combo(self):
        """Populates the substrate_definition_combo based on the selected lattice_type_combo."""
        self.substrate_definition_combo.blockSignals(True)
        self.substrate_definition_combo.clear()
        
        selected_type = self.lattice_type_combo.currentText().lower()
        
        self.substrate_definition_combo.addItem(PREDEFINED_SUBSTRATE_NONE)

        if selected_type == LATTICE_TYPE_CUSTOM:
            self.substrate_definition_combo.addItem(PREDEFINED_SUBSTRATE_CUSTOM)
        else:
            if KNOWN_LATTICES:
                for name, details in KNOWN_LATTICES.items():
                    if details.get("type") == selected_type:
                        self.substrate_definition_combo.addItem(name)
            self.substrate_definition_combo.addItem(PREDEFINED_SUBSTRATE_CUSTOM)

        initial_set = False
        if selected_type == LATTICE_TYPE_CUSTOM:
            if self._initial_custom_definition:
                self.substrate_definition_combo.setCurrentText(PREDEFINED_SUBSTRATE_CUSTOM)
                self.custom_vec_a_spinbox.blockSignals(True)
                self.custom_vec_b_spinbox.blockSignals(True)
                self.custom_vec_angle_spinbox.blockSignals(True)
                self.custom_vec_a_spinbox.setValue(self._initial_custom_definition.get("a_length_nm", 0.300))
                self.custom_vec_b_spinbox.setValue(self._initial_custom_definition.get("b_length_nm", 0.300))
                self.custom_vec_angle_spinbox.setValue(self._initial_custom_definition.get("gamma_deg", 60.0))
                self.custom_vec_a_spinbox.blockSignals(False)
                self.custom_vec_b_spinbox.blockSignals(False)
                self.custom_vec_angle_spinbox.blockSignals(False)
                self._update_custom_definition_from_inputs()
                initial_set = True
        elif self._initial_selected_substrate_name:
            if self._initial_selected_substrate_name == PREDEFINED_SUBSTRATE_CUSTOM and self._initial_custom_a_surf is not None:
                self.substrate_definition_combo.setCurrentText(PREDEFINED_SUBSTRATE_CUSTOM)
                self.custom_a_surf_spinbox.setValue(self._initial_custom_a_surf)
                initial_set = True
            else:
                idx = self.substrate_definition_combo.findText(self._initial_selected_substrate_name)
                if idx != -1:
                    self.substrate_definition_combo.setCurrentIndex(idx)
                    initial_set = True

        if not initial_set:
            self.substrate_definition_combo.setCurrentText(PREDEFINED_SUBSTRATE_NONE)

        self.substrate_definition_combo.blockSignals(False)
        self._on_substrate_definition_combo_changed(self.substrate_definition_combo.currentText())


    @pyqtSlot()
    def _remove_selected_spot(self):
        current_item = self.spots_list_widget.currentItem()
        if current_item:
            row = self.spots_list_widget.row(current_item)
            if 0 <= row < len(self.selected_spots):
                del self.selected_spots[row]
                if self.current_lattice_type == LATTICE_TYPE_CUSTOM and \
                   self.substrate_definition_combo.currentText() == PREDEFINED_SUBSTRATE_CUSTOM:
                    self._update_custom_definition_from_inputs()
                self._update_spots_list_widget()
                self._redraw_all_spot_markers()
                logger.debug(f"Removed spot at index {row}")
            self._update_transform_button_state()

    @pyqtSlot()
    def _clear_all_spots_in_dialog(self):
        self.selected_spots.clear()
        if self.current_lattice_type == LATTICE_TYPE_CUSTOM and \
           self.substrate_definition_combo.currentText() == PREDEFINED_SUBSTRATE_CUSTOM:
            self._update_custom_definition_from_inputs()
        self._update_spots_list_widget()
        self._redraw_all_spot_markers()
        self._update_transform_button_state()
        logger.debug("Cleared all substrate spots in dialog.")

    def _redraw_all_spot_markers(self):
        """Update scene markers for selected and fitted spots."""
        logger.debug("Redrawing substrate spot markers by updating scene overlays.")

        selected_specs = [
            MarkerSpec(pos=tuple(spot), symbol="o", size=10, pen=(0, 255, 0), brush=(50, 205, 50, 120))
            for spot in self.selected_spots
        ]
        self.scene.show_selected_spots(selected_specs)

        fitted_specs = [
            MarkerSpec(pos=tuple(spot), symbol="x", size=12, pen=(0, 255, 255))
            for spot in self.fitted_substrate_spots_px
        ]
        self.scene.show_fitted_spots(fitted_specs)

    @pyqtSlot(float)
    def _on_custom_a_surf_changed(self, value: float):
        """Handles change in the custom a_surf spinbox."""
        if self.substrate_definition_combo.currentText() == PREDEFINED_SUBSTRATE_CUSTOM:
            self.current_a_surf = value
            self.presenter.state.custom_a_surf = self.current_a_surf
            logger.debug(f"Dialog: Custom a_surf changed to: {self.current_a_surf}")
            self._update_transform_button_state()
            self._redraw_ideal_lattice_overlay()

    def _update_custom_definition_from_inputs(self):
        """Refreshes the cached manual lattice definition from spinbox values."""
        if self.current_lattice_type != LATTICE_TYPE_CUSTOM:
            self.current_custom_definition = None
            self.presenter.state.custom_definition = None
            return
        if self.substrate_definition_combo.currentText() != PREDEFINED_SUBSTRATE_CUSTOM:
            self.current_custom_definition = None
            self.presenter.state.custom_definition = None
            return

        a_length = self.custom_vec_a_spinbox.value()
        b_length = self.custom_vec_b_spinbox.value()
        gamma_deg = self.custom_vec_angle_spinbox.value()

        if a_length <= 0 or b_length <= 0:
            self.current_custom_definition = None
            self.presenter.state.custom_definition = None
            return

        gamma_rad = math.radians(gamma_deg)
        if abs(math.sin(gamma_rad)) < 1e-6:
            self.current_custom_definition = None
            self.presenter.state.custom_definition = None
            return

        a_vec = (float(a_length), 0.0)
        b_vec = (
            float(b_length * math.cos(gamma_rad)),
            float(b_length * math.sin(gamma_rad))
        )

        self.current_custom_definition = {
            "name": "Manual Definition",
            "type": LATTICE_TYPE_CUSTOM,
            "a_length_nm": float(a_length),
            "b_length_nm": float(b_length),
            "gamma_deg": float(gamma_deg),
            "a_vec_nm": a_vec,
            "b_vec_nm": b_vec,
            "preferred_point_count": max(len(self.selected_spots), 6) if self.selected_spots else 6
        }
        self.current_a_surf = float(a_length)
        self.presenter.state.custom_definition = dict(self.current_custom_definition)
        self.presenter.state.custom_a_surf = self.current_a_surf

    @pyqtSlot(float)
    def _on_custom_vectors_changed(self, _value: float):
        if self.current_lattice_type == LATTICE_TYPE_CUSTOM and \
           self.substrate_definition_combo.currentText() == PREDEFINED_SUBSTRATE_CUSTOM:
            self._update_custom_definition_from_inputs()
            self._update_transform_button_state()
            self._redraw_ideal_lattice_overlay()

    def _update_lattice_definition_inputs_visibility(self):
        is_custom_type = self.current_lattice_type == LATTICE_TYPE_CUSTOM
        custom_selected = (self.substrate_definition_combo.currentText() == PREDEFINED_SUBSTRATE_CUSTOM)

        show_a_surf = (not is_custom_type) and custom_selected
        self.custom_a_surf_label.setVisible(show_a_surf)
        self.custom_a_surf_spinbox.setVisible(show_a_surf)

        show_vectors = is_custom_type and custom_selected
        for widget in (
            self.custom_vec_a_label, self.custom_vec_a_spinbox,
            self.custom_vec_b_label, self.custom_vec_b_spinbox,
            self.custom_vec_angle_label, self.custom_vec_angle_spinbox
        ):
            widget.setVisible(show_vectors)

    def _build_lattice_info_dict(self, preferred_point_count: Optional[int] = None) -> Optional[Dict[str, Any]]:
        self.presenter.state.lattice_type = self.current_lattice_type
        self.presenter.state.custom_definition = self.current_custom_definition
        self.presenter.state.custom_a_surf = self.current_a_surf
        return self.presenter.build_lattice_info_dict(preferred_point_count=preferred_point_count)

    def _update_transform_button_state(self):
        """Enable/disable the Calculate Transformation button and update the status label."""
        if not hasattr(self, 'calculate_transform_button'):
            return

        can_transform = False
        required_spots = 0
        status_message = "Define lattice and select spots for transformation."

        if self.current_lattice_type == LATTICE_TYPE_CUSTOM:
            required_spots = 3
            if not self.current_custom_definition:
                status_message = "Define custom lattice vectors first."
            elif len(self.selected_spots) >= required_spots:
                can_transform = True
                status_message = f"Ready to transform for {len(self.selected_spots)} spots (Custom)."
            else:
                status_message = f"Select {required_spots - len(self.selected_spots)} more spot(s) for Custom."
        elif self.current_lattice_type and self.current_a_surf is not None and self.current_a_surf > 0:
            if self.current_lattice_type == LATTICE_TYPE_HEXAGONAL:
                required_spots = 6
                if len(self.selected_spots) == required_spots:
                    can_transform = True
                    status_message = f"Ready to transform for {required_spots} spots (Hexagonal)."
                elif len(self.selected_spots) > required_spots:
                    status_message = f"Too many spots selected for Hexagonal (max {required_spots}). Remove {len(self.selected_spots) - required_spots}."
                else:
                    status_message = f"Select {required_spots - len(self.selected_spots)} more spot(s) for Hexagonal."
            elif self.current_lattice_type == LATTICE_TYPE_SQUARE:
                required_spots = 4
                if len(self.selected_spots) == required_spots:
                    can_transform = True
                    status_message = f"Ready to transform for {required_spots} spots (Square)."
                elif len(self.selected_spots) > required_spots:
                    status_message = f"Too many spots selected for Square (max {required_spots}). Remove {len(self.selected_spots) - required_spots}."
                else:
                    status_message = f"Select {required_spots - len(self.selected_spots)} more spot(s) for Square."
            else: # pragma: no cover 
                status_message = "Unknown lattice type selected for spot limit."
        else:
            status_message = "Define lattice parameters first."
        
        self.calculate_transform_button.setEnabled(can_transform)
        self.transform_status_label.setText(status_message)

        if not can_transform:
            if self.substrate_transformation_matrix_F is not None or \
               self.substrate_translation_vector_t is not None or \
               self.fitted_substrate_spots_px:
                
                logger.debug("Conditions for transformation no longer met, clearing previous transform results.")
                self.substrate_transformation_matrix_F = None
                self.substrate_translation_vector_t = None
                self.substrate_transform_analysis = None
                self.fitted_substrate_spots_px.clear()
                self.calculated_ideal_substrate_spots_px.clear()
                self.presenter.state.transform_matrix_F = None
                self.presenter.state.transform_translation_t = None
                self.presenter.state.transform_analysis = None
                self.presenter.state.fitted_spots_px = self.fitted_substrate_spots_px
                self.presenter.state.ideal_spots_px_for_reference = self.calculated_ideal_substrate_spots_px
                self.rotation_angle_label.setText("Rotation: -")
                self.scale_factor_label.setText("Scale (X,Y): -")
                self.rmse_label.setText("RMSE (px): -")
                self._redraw_all_spot_markers()
                self.scene.show_pair_lines([])

    def _handle_fft_image_click(self, event):
        """Handle mouse clicks on the main FFT image."""
        if event.button() == Qt.MouseButton.LeftButton:
            mapped_pos = self.scene.map_scene_to_data(event.scenePos())

            if mapped_pos is not None:
                kx, ky = int(round(mapped_pos.x())), int(round(mapped_pos.y()))
                logger.debug(f"Dialog FFT click: mapped to data (kx, ky) = ({kx}, {ky})")
                roi_size = self.refinement_roi_size_spinbox.value()
                roi_x = kx - roi_size // 2
                roi_y = ky - roi_size // 2
                
                if self.fft_data is not None:
                    max_y, max_x = self.fft_data.shape 
                    roi_x = np.clip(roi_x, 0, max_x - roi_size)
                    roi_y = np.clip(roi_y, 0, max_y - roi_size)

                self.selection_roi.setPos((roi_x, roi_y), update=False) 
                self.selection_roi.setSize((roi_size, roi_size), update=False) 
                self.selection_roi.setVisible(True)
                self.add_spot_button.setEnabled(True)

                self._update_roi_previews()
            event.accept()
        else:
            event.ignore() # pragma: no cover

    def _handle_roi_changed_finished(self):
        """Handle ROI change completion (move/resize)."""
        if self.selection_roi.isVisible():
            self.add_spot_button.setEnabled(True)
            roi_pos = self.selection_roi.pos()
            roi_size = self.selection_roi.size()
            logger.debug(f"ROI changed/moved: Pos ({roi_pos.x():.1f}, {roi_pos.y():.1f}), Size ({roi_size.x():.1f}, {roi_size.y():.1f})")
            
            current_roi_w = int(round(roi_size.x()))
            if current_roi_w != self.refinement_roi_size_spinbox.value() and current_roi_w >= self.refinement_roi_size_spinbox.minimum() and current_roi_w <= self.refinement_roi_size_spinbox.maximum() and current_roi_w % 2 != 0 :
                self.refinement_roi_size_spinbox.blockSignals(True)
                self.refinement_roi_size_spinbox.setValue(current_roi_w)
                self.refinement_roi_size_spinbox.blockSignals(False)

            self._update_roi_previews()

    def _clear_last_preview_gauss_fit(self):
        """Clears the last Gaussian fit preview data."""
        self.last_preview_gauss_fit_popt = None
        self.last_preview_gauss_fit_center_abs = None
        self.last_preview_gauss_roi_state = None

    @pyqtSlot(object) 
    def _handle_roi_region_changing(self, roi_item: Optional[RectROI] = None):
        """Handle ROI changes (move/resize) with live updates."""
        if roi_item is None:
            roi_item = self.selection_roi

        if roi_item.isVisible():
            roi_pos = roi_item.pos()
            roi_size = roi_item.size()
            logger.debug(f"ROI region changing: Pos ({roi_pos.x():.1f}, {roi_pos.y():.1f}), Size ({roi_size.x():.1f}, {roi_size.y():.1f})")
            
            current_roi_w = int(round(roi_size.x())) 
            if current_roi_w != self.refinement_roi_size_spinbox.value() and \
               self.refinement_roi_size_spinbox.minimum() <= current_roi_w <= self.refinement_roi_size_spinbox.maximum() and \
               current_roi_w % 2 != 0 :
                self.refinement_roi_size_spinbox.blockSignals(True)
                self.refinement_roi_size_spinbox.setValue(current_roi_w)
                self.refinement_roi_size_spinbox.blockSignals(False)
            
            self._clear_last_preview_gauss_fit()
            self._update_roi_previews() 


    def _update_roi_previews(self):
        """Updates ROI previews including 2D and 3D views, and Gaussian fits if enabled."""
        if not self.selection_roi.isVisible() or self.fft_data is None:
            self._clear_last_preview_gauss_fit()
            if hasattr(self, 'roi_preview_2d_image_item'): self.roi_preview_2d_image_item.clear()
            if hasattr(self, 'gaussian_preview_2d_image_item'): self.gaussian_preview_2d_image_item.clear()
            return

        roi_state_for_comparison = self.selection_roi.getState() # type: ignore
        x0_roi, y0_roi = int(round(roi_state_for_comparison['pos'].x())), int(round(roi_state_for_comparison['pos'].y()))

        roi_state = self.selection_roi.getState() # type: ignore
        x0_roi, y0_roi = int(round(roi_state['pos'].x())), int(round(roi_state['pos'].y()))
        width_roi, height_roi = int(round(roi_state['size'].x())), int(round(roi_state['size'].y()))
        x1_roi, y1_roi = x0_roi + width_roi, y0_roi + height_roi
        
        max_ky, max_kx = self.fft_data.shape
        y0_cl = np.clip(y0_roi, 0, max_ky)
        y1_cl = np.clip(y1_roi, 0, max_ky)
        x0_cl = np.clip(x0_roi, 0, max_kx)
        x1_cl = np.clip(x1_roi, 0, max_kx)

        if y1_cl <= y0_cl or x1_cl <= x0_cl :
             logger.warning("Invalid ROI slice for preview.")
             return

        roi_patch = self.fft_data[y0_cl:y1_cl, x0_cl:x1_cl]

        if roi_patch.size > 0:
            if self.enable_2d_roi_preview_checkbox.isChecked() and hasattr(self, 'roi_preview_2d_image_item'):
                self.roi_preview_2d_image_item.setImage(roi_patch.T)
                self.roi_preview_2d_plot.autoRange()
            elif hasattr(self, 'roi_preview_2d_image_item'): 
                self.roi_preview_2d_image_item.clear()

            if self.rb_refine_gaussian.isChecked():
                fitted_gauss_params = None
                fitted_gauss_2d_for_preview = None
                if PEAK_FITTING_MODULE_AVAILABLE and SCIPY_AVAILABLE:
                    patch_h, patch_w = roi_patch.shape
                    p_y, p_x = np.mgrid[0:patch_h, 0:patch_w]
                    p_xy_flat = (p_y.flatten(), p_x.flatten())
                    p_data_flat = roi_patch.flatten()
                    try:
                        p0_gauss = [roi_patch.max() - roi_patch.min(), patch_h/2.0, patch_w/2.0, patch_w/4.0, patch_h/4.0, 0.0, roi_patch.min()]
                        if callable(scipy_curve_fit) and callable(_gaussian_2d): 
                            popt_gauss, pcov_gauss = scipy_curve_fit(_gaussian_2d, p_xy_flat, p_data_flat, p0=p0_gauss)
                            self.last_preview_gauss_fit_popt = popt_gauss
                            abs_fit_ky = y0_roi + popt_gauss[1]
                            abs_fit_kx = x0_roi + popt_gauss[2]
                            self.last_preview_gauss_fit_center_abs = (abs_fit_kx, abs_fit_ky)
                            self.last_preview_gauss_roi_state = roi_state_for_comparison.copy() 
                            logger.info(f"Preview Gaussian fit successful. Stored center: {self.last_preview_gauss_fit_center_abs}")

                            fitted_gauss_flat = _gaussian_2d(p_xy_flat, *popt_gauss)
                            fitted_gauss_2d_for_preview = fitted_gauss_flat.reshape(patch_h, patch_w)
                            fitted_gauss_params = popt_gauss 
                        else:
                            self._clear_last_preview_gauss_fit()
                    except Exception as e_fit:
                        logger.warning(f"Gaussian fit for preview failed: {e_fit}")
                        fitted_gauss_2d_for_preview = roi_patch 

                if self.enable_gauss_2d_preview_checkbox.isChecked() and hasattr(self, 'gaussian_preview_2d_image_item'):
                    if fitted_gauss_2d_for_preview is not None:
                        self.gaussian_preview_2d_image_item.setImage(fitted_gauss_2d_for_preview.T)
                        self.gaussian_preview_2d_plot.autoRange()
                    else:
                        self.gaussian_preview_2d_image_item.setImage(roi_patch.T) 
                        self.gaussian_preview_2d_plot.autoRange()
                elif hasattr(self, 'gaussian_preview_2d_image_item'): self.gaussian_preview_2d_image_item.clear()

            else: 
                self._clear_last_preview_gauss_fit()
                if hasattr(self, 'gaussian_preview_2d_image_item'): self.gaussian_preview_2d_image_item.clear()
        else:
            if hasattr(self, 'roi_preview_2d_image_item'): self.roi_preview_2d_image_item.clear()
            if hasattr(self, 'gaussian_preview_2d_image_item'): self.gaussian_preview_2d_image_item.clear()

    @pyqtSlot()
    def _on_refinement_method_changed(self):
        if not self.rb_refine_gaussian.isChecked():
            self._clear_last_preview_gauss_fit() 
        
        is_gaussian_mode = self.rb_refine_gaussian.isChecked()
        self.gaussian_preview_2d_widget.setVisible(is_gaussian_mode)
        
        if self.rb_refine_direct.isChecked():
            self.current_refinement_method = REFINEMENT_DIRECT_CLICK
            self.refinement_roi_size_spinbox.setEnabled(False)
            self.selection_roi.setVisible(False)
            self.add_spot_button.setEnabled(False)
            self.status_label.setText("Click directly on FFT image to add spot.")
        else: 
            self.current_refinement_method = REFINEMENT_MAX_PIXEL if self.rb_refine_max_pixel.isChecked() else REFINEMENT_GAUSSIAN_FIT
            self.refinement_roi_size_spinbox.setEnabled(True)
            self.add_spot_button.setEnabled(self.selection_roi.isVisible())
            self.status_label.setText("Drag ROI to desired spot, then click 'Add/Update Spot'.")
            if self.current_refinement_method == REFINEMENT_GAUSSIAN_FIT:
                self.gauss_2d_container.setVisible(True)
            else:
                self.gauss_2d_container.setVisible(False)
        
        self._update_roi_previews()
        logger.debug(f"Refinement method changed to: {self.current_refinement_method}")


    @pyqtSlot(int)
    def _on_refinement_roi_size_changed(self, value: int):
        self.refinement_roi_size = value
        self._clear_last_preview_gauss_fit()
        if self.selection_roi.isVisible():
            current_pos = self.selection_roi.pos()
            old_size = self.selection_roi.size()
            center_x = current_pos.x() + old_size.x() / 2
            center_y = current_pos.y() + old_size.y() / 2
            new_pos_x = center_x - value / 2
            new_pos_y = center_y - value / 2
            self.selection_roi.setPos((new_pos_x, new_pos_y), update=False)
            self.selection_roi.setSize((value, value), update=False)
            self._handle_roi_changed_finished()
            
        logger.debug(f"Refinement ROI size changed to: {self.refinement_roi_size}")

    @pyqtSlot()
    def _on_calculate_transform_clicked(self):
        """Handles the calculation of transformation between measured and ideal spots."""
        logger.info("Calculate Transformation button clicked.")
        self.transform_status_label.setText("Calculating transformation...")
        QApplication.processEvents()

        try:
            result = self.presenter.calculate_transform(
                preferred_point_count=max(len(self.selected_spots), 6)
            )
        except TransformComputationError as exc:
            if exc.severity == "warning":
                QMessageBox.warning(self, "Transform Error", exc.user_message)
            else:
                QMessageBox.critical(self, "Transform Error", exc.user_message)
            self.transform_status_label.setText(exc.status_message)
            return
        except Exception as exc:  # pragma: no cover
            logger.exception("Unexpected error during transformation calculation: ")
            QMessageBox.critical(self, "Error", f"An error occurred: {exc}")
            self.transform_status_label.setText("Error.")
            return

        self.substrate_transformation_matrix_F = self.presenter.state.transform_matrix_F
        self.substrate_translation_vector_t = self.presenter.state.transform_translation_t
        self.substrate_transform_analysis = self.presenter.state.transform_analysis
        self.fitted_substrate_spots_px = list(self.presenter.state.fitted_spots_px)
        self.presenter.state.fitted_spots_px = self.fitted_substrate_spots_px
        self.calculated_ideal_substrate_spots_px = list(self.presenter.state.ideal_spots_px_for_reference)
        self.presenter.state.ideal_spots_px_for_reference = self.calculated_ideal_substrate_spots_px

        pairs = [
            (tuple(measured), tuple(ideal))
            for measured, ideal in zip(result.measured_spots_px, result.ideal_spots_px)
        ]
        self.scene.show_pair_lines(pairs)

        analysis = result.analysis
        rotation_text = format_float(analysis.get("rotation_angle_deg"), precision=2)
        rotation_display = rotation_text if rotation_text == "-" else f"{rotation_text} deg"
        self.rotation_angle_label.setText(f"Rotation: {rotation_display}")

        stretch_display = format_pair(analysis.get("principal_stretches"), precision=3)
        self.scale_factor_label.setText(f"Stretches: {stretch_display}")

        rmse_text = format_float(analysis.get("rmse"), precision=3)
        self.rmse_label.setText(f"RMSE (px): {rmse_text}")
        self.transform_status_label.setText("Transformation calculated.")

        self._redraw_all_spot_markers() 

    @pyqtSlot()
    def _add_current_roi_spot(self):
        """Adds a spot based on the current ROI and selected refinement method."""
        if not self.selection_roi.isVisible() or self.fft_data is None:
            self.status_label.setText("Error: No ROI selected or no FFT data.")
            return

        max_spots = self.limits_per_lattice.get(self.current_lattice_type, 6)
        if max_spots > 0 and len(self.selected_spots) >= max_spots:
            QMessageBox.warning(self, "Limit Reached",
                                f"Maximum number of spots ({max_spots}) for {self.current_lattice_type} lattice already selected.")
            return

        roi_state = self.selection_roi.getState()
        x0_roi, y0_roi = int(round(roi_state['pos'].x())), int(round(roi_state['pos'].y()))
        width_roi, height_roi = int(round(roi_state['size'].x())), int(round(roi_state['size'].y()))
        
        center_kx = x0_roi + width_roi // 2
        center_ky = y0_roi + height_roi // 2
        
        refined_kx, refined_ky = float(center_kx), float(center_ky)

        if self.current_refinement_method == REFINEMENT_MAX_PIXEL and PEAK_FITTING_MODULE_AVAILABLE:
            patch_radius = self.refinement_roi_size // 2 
            max_h, max_w = self.fft_data.shape
            eff_center_ky = np.clip(center_ky, patch_radius, max_h - 1 - patch_radius)
            eff_center_kx = np.clip(center_kx, patch_radius, max_w - 1 - patch_radius)

            fit_ky, fit_kx = find_max_pixel_in_roi(self.fft_data, (eff_center_ky, eff_center_kx), patch_radius)
            refined_kx, refined_ky = float(fit_kx) + 0.5, float(fit_ky) + 0.5
            logger.info(f"Spot refined by Max Pixel: ({refined_kx:.2f}, {refined_ky:.2f})")
        elif self.current_refinement_method == REFINEMENT_GAUSSIAN_FIT and PEAK_FITTING_MODULE_AVAILABLE and SCIPY_AVAILABLE:
            current_selection_roi_state = self.selection_roi.getState() # type: ignore
            roi_state_matches_preview = False
            if self.last_preview_gauss_roi_state and current_selection_roi_state:
                preview_pos = self.last_preview_gauss_roi_state.get('pos')
                current_pos = current_selection_roi_state.get('pos')
                preview_size = self.last_preview_gauss_roi_state.get('size')
                current_size = current_selection_roi_state.get('size')
                if preview_pos and current_pos and preview_size and current_size:
                    if preview_pos == current_pos and preview_size == current_size:
                        roi_state_matches_preview = True
            
            if self.last_preview_gauss_fit_center_abs is not None and roi_state_matches_preview:
                refined_kx, refined_ky = self.last_preview_gauss_fit_center_abs
                logger.info(f"Using PREVIEW Gaussian fit result for Add Spot: ({refined_kx:.2f}, {refined_ky:.2f})")
            else: 
                logger.info("Performing NEW Gaussian fit for Add Spot (preview data not used or ROI changed).")
                patch_radius = self.refinement_roi_size // 2
                max_h, max_w = self.fft_data.shape # type: ignore
                eff_center_ky = np.clip(center_ky, patch_radius, max_h - 1 - patch_radius)
                eff_center_kx = np.clip(center_kx, patch_radius, max_w - 1 - patch_radius)
                
                fit_output = fit_2d_gaussian_in_roi(self.fft_data, (eff_center_ky, eff_center_kx), patch_radius)
                if fit_output:
                    _popt, (fit_ky_abs, fit_kx_abs), _patch = fit_output
                    refined_kx, refined_ky = float(fit_kx_abs), float(fit_ky_abs)
                    logger.info(f"Spot refined by NEW 2D Gaussian Fit: ({refined_kx:.2f}, {refined_ky:.2f})")
                else:
                    logger.warning("2D Gaussian fit failed for Add Spot. Using ROI center.")

        new_spot = (refined_kx, refined_ky)
        if new_spot not in self.selected_spots:
            self.selected_spots.append(new_spot)
            if self.current_lattice_type == LATTICE_TYPE_CUSTOM and \
               self.substrate_definition_combo.currentText() == PREDEFINED_SUBSTRATE_CUSTOM:
                self._update_custom_definition_from_inputs()
            self._update_spots_list_widget()
            self._redraw_all_spot_markers()
            self._update_transform_button_state()
            point_text = format_pair((refined_kx, refined_ky), precision=2)
            self.status_label.setText(f"Spot {len(self.selected_spots)} added: {point_text}.")
        else:
            point_text = format_pair((refined_kx, refined_ky), precision=2)
            self.status_label.setText(f"Spot {point_text} already selected.")

    @pyqtSlot(str)
    def _on_lattice_type_changed(self, text: Optional[str] = None):
        """Handles change in the general lattice type (Hexagonal/Square)."""
        selected_type_text = self.lattice_type_combo.currentText().lower()
        if LATTICE_TYPE_HEXAGONAL in selected_type_text:
            self.current_lattice_type = LATTICE_TYPE_HEXAGONAL
        elif LATTICE_TYPE_SQUARE in selected_type_text:
            self.current_lattice_type = LATTICE_TYPE_SQUARE
        elif LATTICE_TYPE_CUSTOM in selected_type_text:
            self.current_lattice_type = LATTICE_TYPE_CUSTOM
        else:
            self.current_lattice_type = None
        
        self.presenter.state.lattice_type = self.current_lattice_type
        logger.debug(f"Dialog: General lattice type selected: {self.current_lattice_type}")
        self._populate_substrate_definition_combo()
        if self.current_lattice_type == LATTICE_TYPE_CUSTOM:
            self._update_custom_definition_from_inputs()
        self._update_lattice_definition_inputs_visibility()
        self._update_transform_button_state()

    @pyqtSlot(str)
    def _on_substrate_definition_combo_changed(self, text: str):
        """Handles change in the specific substrate definition or custom option."""
        self.current_a_surf = None
        if self.current_lattice_type != LATTICE_TYPE_CUSTOM:
            self.current_custom_definition = None
            self.presenter.state.custom_definition = None

        is_custom = (text == PREDEFINED_SUBSTRATE_CUSTOM)
        is_none_selection = (text == PREDEFINED_SUBSTRATE_NONE)
        self.presenter.state.selected_definition = text

        if self.current_lattice_type == LATTICE_TYPE_CUSTOM:
            if is_custom:
                self._update_custom_definition_from_inputs()
                logger.debug("Dialog: Using manual custom lattice definition.")
            else:
                logger.debug("Dialog: Custom lattice definition disabled via combo selection.")
                self.current_custom_definition = None
                self.presenter.state.custom_definition = None
        else:
            if is_custom:
                self.current_a_surf = self.custom_a_surf_spinbox.value()
                self.presenter.state.custom_a_surf = self.current_a_surf
                logger.debug(f"Dialog: Switched to custom a_surf definition. Current a_surf: {self.current_a_surf}")
            elif is_none_selection:
                logger.debug("Dialog: Substrate definition set to None. No a_surf.")
                self.presenter.state.custom_a_surf = None
            elif KNOWN_LATTICES and text in KNOWN_LATTICES:
                self.current_a_surf = KNOWN_LATTICES[text].get("a_surf")
                self.presenter.state.custom_a_surf = self.current_a_surf
                known_type = KNOWN_LATTICES[text].get("type")
                if known_type != self.current_lattice_type:
                    logger.warning(f"Mismatch between combo lattice type ({self.current_lattice_type}) and known lattice type for '{text}' ({known_type}). Using type from combo.")
                logger.debug(f"Dialog: Selected predefined substrate '{text}'. a_surf: {self.current_a_surf}, type: {self.current_lattice_type}")
            else:
                self.presenter.state.custom_a_surf = None

        self._update_lattice_definition_inputs_visibility()
        self._update_add_spot_button_state()
        self._update_transform_button_state()
        self._redraw_ideal_lattice_overlay()

    def _update_add_spot_button_state(self):
        if self.current_lattice_type:
            limit = self.limits_per_lattice.get(self.current_lattice_type, 0)
            unlimited = limit <= 0
            can_add_more = unlimited or len(self.selected_spots) < limit
            self.add_spot_button.setEnabled(self.selection_roi.isVisible() and can_add_more) # type: ignore

            if unlimited:
                if not self.selection_roi.isVisible():
                    self.status_label.setText("Click on FFT to place ROI for custom lattice.")
                else:
                    self.status_label.setText("Adjust ROI and click 'Add Spot'.")
            else:
                if not can_add_more and limit > 0:
                    self.status_label.setText(f"Max {limit} spots for {self.current_lattice_type} lattice reached.")
                elif not self.selection_roi.isVisible() and len(self.selected_spots) < limit:
                    self.status_label.setText(f"Click on FFT to place ROI. {limit - len(self.selected_spots)} spots remaining.")
                elif self.selection_roi.isVisible() and can_add_more:
                    self.status_label.setText(f"Adjust ROI and click 'Add Spot'. {limit - len(self.selected_spots)} spots remaining.")
        else:
            self.add_spot_button.setEnabled(False)
            self.status_label.setText("Select a lattice type first.")

    def _redraw_ideal_lattice_overlay(self):
        """Redraws the ideal lattice overlay based on current lattice type and parameters."""
        if not self.show_ideal_lattice_checkbox.isChecked() or self.fft_data is None:
            self.scene.show_ideal_overlay([])
            return

        lattice_info_for_calc = self._build_lattice_info_dict()
        if not lattice_info_for_calc:
            logger.debug("Cannot draw ideal lattice: definition not available.")
            self.scene.show_ideal_overlay([])
            return

        root_node = self.history_manager.get_root_node_for_node(self.current_fft_node_id)
        if not (root_node and root_node.operation_name == "Original" and root_node.parameters):
            logger.warning("Could not trace to Original node or missing parameters for lattice overlay.")
            self.scene.show_ideal_overlay([])
            return
        
        Lx_nm = root_node.parameters.get("size_nm_x")
        Ly_nm = root_node.parameters.get("size_nm_y")
        fft_data_rows_ky, fft_data_cols_kx = self.fft_data.shape 

        if not (Lx_nm and Ly_nm and Lx_nm > 0 and Ly_nm > 0 and fft_data_cols_kx > 0 and fft_data_rows_ky > 0):
            logger.warning("Missing calibration data (Lx, Ly) or invalid FFT shape for lattice overlay.")
            self.scene.show_ideal_overlay([])
            return

        ideal_points_g_nm_inv = get_nearest_reciprocal_points(lattice_info_for_calc)

        if not ideal_points_g_nm_inv:
            logger.warning("Could not get ideal reciprocal points for overlay.") # pragma: no cover
            self.scene.show_ideal_overlay([])
            return

        overlay_specs = []
        center_display_kx = fft_data_cols_kx / 2.0
        center_display_ky = fft_data_rows_ky / 2.0
        self.calculated_ideal_substrate_spots_px.clear()

        for Gx_nm_inv, Gy_nm_inv in ideal_points_g_nm_inv:
            display_y_px = center_display_kx + (Gx_nm_inv * Lx_nm)
            display_x_px = center_display_ky + (Gy_nm_inv * Ly_nm)
            current_pixel_coord = (display_x_px, display_y_px)
            self.calculated_ideal_substrate_spots_px.append(current_pixel_coord)
            overlay_specs.append(
                MarkerSpec(pos=current_pixel_coord, symbol="+", size=10, pen=(255, 0, 0))
            )
            logger.debug("Ideal lattice point mapped to pixels: %s", current_pixel_coord)
        self.presenter.state.ideal_spots_px_for_reference = self.calculated_ideal_substrate_spots_px
        if overlay_specs:
            self.scene.show_ideal_overlay(overlay_specs)
            logger.info("Displayed ideal lattice overlay for current definition.")
        else:
            self.scene.show_ideal_overlay([])


    def get_selected_spots(self) -> List[Tuple[float, float]]:
        return list(self.selected_spots)

    def get_dialog_results(self) -> Dict[str, Any]:
        self.presenter.state.selected_definition = self.substrate_definition_combo.currentText()
        self.presenter.state.lattice_type = self.current_lattice_type
        self.presenter.state.custom_a_surf = self.current_a_surf
        self.presenter.state.custom_definition = self.current_custom_definition
        self.presenter.state.transform_matrix_F = self.substrate_transformation_matrix_F
        self.presenter.state.transform_translation_t = self.substrate_translation_vector_t
        self.presenter.state.transform_analysis = self.substrate_transform_analysis
        self.presenter.state.fitted_spots_px = self.fitted_substrate_spots_px
        self.presenter.state.ideal_spots_px_for_reference = self.calculated_ideal_substrate_spots_px
        return self.presenter.build_results_dict()

    def accept(self):
        if self.current_lattice_type:
            limit = self.limits_per_lattice.get(self.current_lattice_type, 0)
            if len(self.selected_spots) != limit and limit > 0 :
                QMessageBox.warning(self, "Spot Count Error",
                                    f"Please select exactly {limit} spots for a "
                                    f"{self.current_lattice_type} lattice. "
                                    f"Currently selected: {len(self.selected_spots)}.")
                return 
                    
        logger.info(f"SubstrateSpotSelectionDialog accepted with {len(self.selected_spots)} spots for {self.current_lattice_type} lattice.")
        super().accept()

def closeEvent(self, event):
    """Handle dialog close event to clean up OpenGL resources."""
    logger.debug("SubstrateSpotSelectionDialog closing. Cleaning up GL items.")
    super().closeEvent(event)
