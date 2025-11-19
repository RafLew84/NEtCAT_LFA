# lfa/gui/dialogs/superstructure_periodicity_dialog.py
import logging
import math
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PyQt6.QtCore import Qt, QTimer, pyqtSlot
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ...analysis.drift_correction import apply_affine_transform
from ...analysis.lattice import (
    calculate_d_spacing_from_ideal_spot,
    calculate_superstructure_periodicity_parameters,
    convert_g_vector_px_to_nm_inv,
)
from ..utils.display import (
    format_float,
    format_pair_with_sigma,
    format_ratio,
    format_value_with_sigma,
)

try:
    import pyqtgraph as pg
    from pyqtgraph import GraphicsLayoutWidget, ImageItem, RectROI, ScatterPlotItem, ViewBox
    PYQTGRAPH_AVAILABLE = True
except ImportError:
    pg = None
    GraphicsLayoutWidget = None
    ImageItem = None
    ViewBox = None
    RectROI = None
    ScatterPlotItem = None
    PYQTGRAPH_AVAILABLE = False
    logging.error("SuperstructurePeriodicityDialog: PyQtGraph not found.")

try:
    from ...logic.app_controller import AppController
except ImportError:
    AppController = None

try:
    from ...analysis.lattice import KNOWN_LATTICES
    from ...analysis.peak_fitting import (
        SCIPY_AVAILABLE,
        _gaussian_2d,
        find_max_pixel_in_roi,
        fit_2d_gaussian_in_roi_with_all_data,
    )
    from ...logic.history_manager import HistoryManager
    PEAK_FITTING_MODULE_AVAILABLE = True
except ImportError:
    PEAK_FITTING_MODULE_AVAILABLE = False
    SCIPY_AVAILABLE = False
    KNOWN_LATTICES = {}
    logging.error("AdsorbateSpotSelectionDialog: Could not import peak_fitting or lattice modules.")
    def find_max_pixel_in_roi(data, center, radius): return center
    def _gaussian_2d(*args, **kwargs): raise ImportError("Gaussian 2D function is not available")

try:
    from scipy.optimize import curve_fit as scipy_curve_fit
    SCIPY_OPTIMIZE_AVAILABLE = True
except ImportError:
    logging.error("AdsorbateSpotSelectionDialog: SciPy (for curve_fit) not found.")
    SCIPY_OPTIMIZE_AVAILABLE = False
    def scipy_curve_fit(*args, **kwargs): raise ImportError("scipy.optimize.curve_fit is not available")

logger = logging.getLogger(__name__)

class SuperstructurePeriodicityDialog(QDialog):
    """
    Dialog for analyzing superstructure periodicity by selecting a main peak
    and one or more satellite peaks on an FFT image.
    (Szkielet - implementacja w kolejnych krokach)
    """
    def __init__(self,
                 fft_image_data: Optional[np.ndarray],
                 history_manager: HistoryManager,
                 current_fft_node_id: str,
                 substrate_F_m2i: Optional[np.ndarray],
                 substrate_t_m2i: Optional[np.ndarray],
                 substrate_transform_analysis: Optional[Dict[str, Any]],
                 default_refinement_roi_size: int = 5,
                 parent=None):
        super().__init__(parent)

        self.fft_data = fft_image_data
        self.history_manager = history_manager
        self.current_fft_node_id = current_fft_node_id
        self.refinement_roi_size = default_refinement_roi_size
        self.sub_F_m2i = substrate_F_m2i
        self.sub_t_m2i = substrate_t_m2i
        self.sub_transform_analysis = substrate_transform_analysis

        if not (self.history_manager and self.current_fft_node_id):
            QMessageBox.critical(self, "Initialization Error", "History context was not provided to the dialog.")
            QTimer.singleShot(0, self.reject)
            return

        fft_node = self.history_manager.get_node_by_id(self.current_fft_node_id)
        is_power_scale = False
        if fft_node and fft_node.parameters:
            if fft_node.parameters.get("scaling_mode") == "power":
                is_power_scale = True
        
        if not is_power_scale:
            logger.warning("SuperstructurePeriodicityDialog cannot be used: "
                           "Active FFT was not calculated with 'Power' scaling.")
            
            QMessageBox.warning(
                self, 
                "Incorrect FFT Scaling", 
                "Superstructure periodicity analysis requires the FFT to be calculated "
                "with the **'Power'** scaling mode (|F|²).\n\n"
                "Please go back, recalculate the FFT with the correct setting, and try again."
            )
            
            QTimer.singleShot(0, self.reject)
            return

        if not PYQTGRAPH_AVAILABLE: # pragma: no cover
            QVBoxLayout(self).addWidget(QLabel("Critical Error: PyQtGraph is required..."))
            self.setWindowTitle("Error")
            return

        self.setWindowTitle("Superstructure Periodicity Analysis")
        self.setMinimumSize(1200, 700)
        current_flags=self.windowFlags()
        self.setWindowFlags(current_flags | Qt.WindowType.WindowMinimizeButtonHint | Qt.WindowType.WindowMaximizeButtonHint)


        self._selection_mode: Optional[str] = None
        self.main_peak_raw_refined_px: Optional[Tuple[float, float]] = None
        self.main_peak_raw_sigma_px: Optional[Tuple[float, float]] = None
        self.satellite_peak_raw_refined_px: Optional[Tuple[float, float]] = None
        self.satellite_peak_raw_sigma_px: Optional[Tuple[float, float]] = None
        self.main_peak_raw_marker: Optional[ScatterPlotItem] = None
        self.satellite_raw_marker: Optional[ScatterPlotItem] = None
        self.main_peak_corrected_marker: Optional[ScatterPlotItem] = None
        self.satellite_corrected_marker: Optional[ScatterPlotItem] = None
        self.main_peak_corrected_ideal_px: Optional[Tuple[float, float]] = None
        self.main_peak_corrected_sigma_px: Optional[Tuple[float, float]] = None
        self.satellite_peak_corrected_ideal_px: Optional[Tuple[float, float]] = None
        self.satellite_peak_corrected_sigma_px: Optional[Tuple[float, float]] = None
        self.main_peak_amplitude: Optional[float] = None
        self.main_peak_amplitude_sigma: Optional[float] = None
        self.satellite_peak_amplitude: Optional[float] = None
        self.satellite_peak_amplitude_sigma: Optional[float] = None
        self.main_peak_intensity: Optional[float] = None
        self.main_peak_intensity_sigma: Optional[float] = None
        self.satellite_peak_intensity: Optional[float] = None
        self.satellite_peak_intensity_sigma: Optional[float] = None
        self.main_peak_max_value: Optional[float] = None
        self.main_peak_max_value_sigma: Optional[float] = None
        self.satellite_peak_max_value: Optional[float] = None
        self.satellite_peak_max_value_sigma: Optional[float] = None
        self.basic_main_periodicity_nm: Optional[float] = None
        self.basic_satellite_periodicity_nm: Optional[float] = None
        self._final_results: Optional[Dict[str, Any]] = None

        self._init_ui()
        self._connect_signals()
        
        self.refinement_roi_size_spinbox.setValue(self.refinement_roi_size)
        self._display_substrate_transform_info() 
        self._update_all_ui_elements() 

        logger.debug("SuperstructurePeriodicityDialog initialized.")

    def _init_ui(self):
        top_level_layout = QHBoxLayout(self)
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        top_level_layout.addWidget(main_splitter)

        left_controls_widget = QWidget()
        left_controls_layout = QVBoxLayout(left_controls_widget)
        left_controls_widget.setMinimumWidth(300)
        left_controls_widget.setMaximumWidth(380)

        refinement_group = QGroupBox("Spot Selection")
        refinement_layout = QFormLayout(refinement_group)
        refinement_layout.addRow(QLabel("Refinement Method: 2D Gaussian Fit"))
        self.refinement_roi_size_spinbox = QSpinBox()
        self.refinement_roi_size_spinbox.setMinimum(3)
        self.refinement_roi_size_spinbox.setMaximum(31)
        self.refinement_roi_size_spinbox.setSingleStep(2)
        refinement_layout.addRow("Refinement Area Size (px):", self.refinement_roi_size_spinbox)
        
        self.add_main_spot_button = QPushButton("Add/Update Main Spot from ROI")
        self.add_main_spot_button.setEnabled(True)
        self.add_satellite_spot_button = QPushButton("Add Satellite Spot from ROI")
        self.add_satellite_spot_button.setEnabled(True)
        
        refinement_layout.addRow(self.add_main_spot_button)
        refinement_layout.addRow(self.add_satellite_spot_button)
        left_controls_layout.addWidget(refinement_group)
        
        sub_transform_group = QGroupBox("Substrate Transformation Info (Applied)")
        sub_transform_layout = QFormLayout(sub_transform_group)
        self.dist_sub_transform_info_label_status = QLabel("Status: -")
        self.dist_sub_transform_info_label_rot = QLabel("Sub. Rotation: -")
        self.dist_sub_transform_info_label_scale = QLabel("Sub. Scale (X,Y): -")
        self.dist_sub_transform_info_label_rmse = QLabel("Sub. RMSE (px): -")
        sub_transform_layout.addRow("Status:", self.dist_sub_transform_info_label_status)
        sub_transform_layout.addRow("Rotation:", self.dist_sub_transform_info_label_rot)
        sub_transform_layout.addRow("Scale:", self.dist_sub_transform_info_label_scale)
        sub_transform_layout.addRow("RMSE:", self.dist_sub_transform_info_label_rmse)
        self.apply_substrate_transform_checkbox = QCheckBox("Apply Substrate Transformation")
        self.apply_substrate_transform_checkbox.setChecked(True) 
        sub_transform_layout.addRow("Apply Transform:", self.apply_substrate_transform_checkbox)
        left_controls_layout.addWidget(sub_transform_group)
        
        left_controls_layout.addStretch(1)
        main_splitter.addWidget(left_controls_widget)

        self.fft_plot_widget = GraphicsLayoutWidget()
        self.fft_view_box = self.fft_plot_widget.addViewBox(row=0, col=0, lockAspect=True, invertY=True)
        self.fft_image_item = ImageItem()
        self.fft_view_box.addItem(self.fft_image_item)
        self.fft_histogram = pg.HistogramLUTItem()
        self.fft_histogram.setImageItem(self.fft_image_item)
        self.fft_plot_widget.addItem(self.fft_histogram, row=0, col=1)
        self.fft_view_box.setMenuEnabled(True)
        self.fft_view_box.setMouseMode(ViewBox.PanMode)
        self.fft_view_box.setMouseEnabled(x=True,y=True)
        if self.fft_data is not None: self.fft_image_item.setImage(self.fft_data.T)
        self.selection_roi = RectROI(pos=(0,0), size=(self.refinement_roi_size, self.refinement_roi_size), pen=pg.mkPen('cyan', width=2), movable=True, resizable=True)
        self.fft_view_box.addItem(self.selection_roi)
        self.selection_roi.setVisible(False)
        main_splitter.addWidget(self.fft_plot_widget)

        right_panel_widget = QWidget()
        right_panel_layout = QVBoxLayout(right_panel_widget)
        right_panel_widget.setMinimumWidth(400)
        right_panel_widget.setMaximumWidth(500)

        preview_group = QGroupBox("Live Previews (Gaussian Fit)")
        preview_grid_layout = QGridLayout(preview_group)
        roi_2d_container = QWidget()
        roi_2d_v_layout = QVBoxLayout(roi_2d_container)
        roi_2d_h_layout = QHBoxLayout()
        roi_2d_h_layout.addWidget(QLabel("ROI 2D Preview:"))
        self.enable_2d_roi_preview_checkbox = QCheckBox("Enable")
        self.enable_2d_roi_preview_checkbox.setChecked(True)
        roi_2d_h_layout.addWidget(self.enable_2d_roi_preview_checkbox)
        roi_2d_h_layout.addStretch()
        roi_2d_v_layout.addLayout(roi_2d_h_layout)
        self.roi_preview_2d_widget = GraphicsLayoutWidget()
        self.roi_preview_2d_widget.setMinimumHeight(150)
        self.roi_preview_2d_widget.setMaximumHeight(200)
        self.roi_preview_2d_plot = self.roi_preview_2d_widget.addViewBox(lockAspect=True, invertY=True)
        self.roi_preview_2d_image_item = ImageItem()
        self.roi_preview_2d_plot.addItem(self.roi_preview_2d_image_item)
        roi_2d_v_layout.addWidget(self.roi_preview_2d_widget, 1)
        preview_grid_layout.addWidget(roi_2d_container, 0, 0)
        self.gauss_2d_container = QWidget()
        gauss_2d_v_layout = QVBoxLayout(self.gauss_2d_container)
        gauss_2d_h_layout = QHBoxLayout()
        gauss_2d_h_layout.addWidget(QLabel("Gaussian Fit 2D Preview:"))
        self.enable_gauss_2d_preview_checkbox = QCheckBox("Enable")
        self.enable_gauss_2d_preview_checkbox.setChecked(True)
        gauss_2d_h_layout.addWidget(self.enable_gauss_2d_preview_checkbox)
        gauss_2d_h_layout.addStretch()
        gauss_2d_v_layout.addLayout(gauss_2d_h_layout)
        self.gaussian_preview_2d_widget = GraphicsLayoutWidget()
        self.gaussian_preview_2d_widget.setMinimumHeight(150)
        self.gaussian_preview_2d_widget.setMaximumHeight(200)
        self.gaussian_preview_2d_plot = self.gaussian_preview_2d_widget.addViewBox(lockAspect=True, invertY=True)
        self.gaussian_preview_2d_image_item = ImageItem()
        self.gaussian_preview_2d_plot.addItem(self.gaussian_preview_2d_image_item)
        gauss_2d_v_layout.addWidget(self.gaussian_preview_2d_widget, 1)
        preview_grid_layout.addWidget(self.gauss_2d_container, 0, 1)
        right_panel_layout.addWidget(preview_group)

        selected_spots_group = QGroupBox("Selected Peaks Information")
        selected_spots_layout = QFormLayout(selected_spots_group)
        self.main_peak_info_label = QLabel("Not Selected")
        self.main_peak_info_label.setWordWrap(True)
        self.satellite_peak_info_label = QLabel("Not Selected")
        self.satellite_peak_info_label.setWordWrap(True)
        selected_spots_layout.addRow("Main Peak:", self.main_peak_info_label)
        selected_spots_layout.addRow("Satellite Peak:", self.satellite_peak_info_label)
        right_panel_layout.addWidget(selected_spots_group)
        
        results_group = QGroupBox("Calculated Results")
        results_layout = QFormLayout(results_group)
        self.calculate_distance_button = QPushButton("Calculate Superstructure Periodicity Parameters")
        self.calculate_distance_button.setEnabled(False)
        results_layout.addRow(self.calculate_distance_button)
        self.distance_fft_label = QLabel("-")
        self.distance_real_space_label = QLabel("-")
        self.intensity_ratio_label = QLabel("-")
        self.amplitude_ratio_label = QLabel("-")
        self.max_value_label = QLabel("-")
        results_layout.addRow("Distance in k-space (Δg*):", self.distance_fft_label)
        results_layout.addRow("Real Space Periodicity (P):", self.distance_real_space_label)
        results_layout.addRow("Intensity Ratio (Sat/Main):", self.intensity_ratio_label)
        results_layout.addRow("Amplitude Ratio (Sat/Main):", self.amplitude_ratio_label)
        results_layout.addRow("Max Value Ratio (Set/Main):", self.max_value_label)
        right_panel_layout.addWidget(results_group)

        self.status_label = QLabel("Click on FFT to select a spot.")
        right_panel_layout.addWidget(self.status_label)
        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        right_panel_layout.addWidget(self.button_box)
        right_panel_layout.addStretch(1)
        main_splitter.addWidget(right_panel_widget)
        
        main_splitter.setSizes([350,550,300])
        main_splitter.setStretchFactor(1,1)


    def _connect_signals(self):
        """Connects signals from UI to slots."""
        self.button_box.clicked.connect(self.accept)

        if self.fft_view_box and self.fft_view_box.scene():
            self.fft_view_box.scene().sigMouseClicked.connect(self._handle_fft_image_click)
        
        self.selection_roi.sigRegionChanged.connect(self._handle_roi_region_changing)
        self.refinement_roi_size_spinbox.valueChanged.connect(self._on_refinement_roi_size_changed)

        self.add_main_spot_button.clicked.connect(self._on_add_main_spot_clicked)
        self.add_satellite_spot_button.clicked.connect(self._on_add_satellite_peak_clicked)

        self.enable_2d_roi_preview_checkbox.stateChanged.connect(self._update_roi_previews)
        self.enable_gauss_2d_preview_checkbox.stateChanged.connect(self._update_roi_previews)
        
        self.calculate_distance_button.clicked.connect(self._on_calculate_distance_clicked)

        logger.debug("SpotDistanceDialog signals connected.")
    
    @pyqtSlot()
    def _on_calculate_distance_clicked(self):
        """
        Calculates and displays the distance, periodicity and intensity/amplitude ratios between the corrected main peak and the corrected satellite peak.
        """
        logger.debug("Calculate Distance button clicked.")
        
        if not (self.main_peak_corrected_ideal_px and self.satellite_peak_corrected_ideal_px and
                self.main_peak_intensity is not None and self.satellite_peak_intensity is not None and
                self.main_peak_amplitude is not None and self.satellite_peak_amplitude is not None and
                self.main_peak_max_value is not None and self.satellite_peak_max_value is not None):
            QMessageBox.warning(self, "Incomplete Data", "Both Main and Satellite peaks must be selected and successfully processed to perform calculation.")
            return

        main_peak_data = {
            'corrected': self.main_peak_corrected_ideal_px,
            'intensity': self.main_peak_intensity,
            'intensity_sigma': self.main_peak_intensity_sigma,
            'amplitude': self.main_peak_amplitude,
            'amplitude_sigma': self.main_peak_amplitude_sigma,
            'max_value': self.main_peak_max_value,
            'max_value_sigma': self.main_peak_max_value_sigma,
        }
        
        satellite_peak_data = {
            'corrected': self.satellite_peak_corrected_ideal_px,
            'intensity': self.satellite_peak_intensity,
            'intensity_sigma': self.satellite_peak_intensity_sigma,
            'amplitude': self.satellite_peak_amplitude,
            'amplitude_sigma': self.satellite_peak_amplitude_sigma,
            'max_value': self.satellite_peak_max_value,
            'max_value_sigma': self.satellite_peak_max_value_sigma
        }

        if self.fft_data is None or self.history_manager is None:
            QMessageBox.critical(self, "Error", "Internal error: FFT data or History Manager not available.")
            return
        root_node = self.history_manager.get_root_node_for_node(self.current_fft_node_id)
        if not (root_node and root_node.parameters):
            QMessageBox.critical(self, "Error", "Could not retrieve calibration data (Lx, Ly) from original image.")
            return
        Lx_nm = root_node.parameters.get("size_nm_x")
        Ly_nm = root_node.parameters.get("size_nm_y")
        if not (Lx_nm and Ly_nm and Lx_nm > 0 and Ly_nm > 0):
             QMessageBox.critical(self, "Error", "Invalid calibration data (Lx, Ly).")
             return
        
        results = calculate_superstructure_periodicity_parameters(
            main_peak_data=main_peak_data,
            satellite_peak_data=satellite_peak_data,
            fft_shape=self.fft_data.shape,
            lx_nm=Lx_nm,
            ly_nm=Ly_nm
        )
            

        if results:
            # enrich with positional context for downstream overlays/reports
            if self.main_peak_corrected_ideal_px and self.satellite_peak_corrected_ideal_px:
                delta_px = (
                    float(self.satellite_peak_corrected_ideal_px[0] - self.main_peak_corrected_ideal_px[0]),
                    float(self.satellite_peak_corrected_ideal_px[1] - self.main_peak_corrected_ideal_px[1]),
                )
                results["delta_g_px"] = delta_px
                delta_nm_inv_vec = convert_g_vector_px_to_nm_inv(
                    delta_px,
                    Lx_nm,
                    Ly_nm,
                    self.fft_data.shape[1],
                    self.fft_data.shape[0],
                )
                if delta_nm_inv_vec is not None:
                    results["delta_g_nm_inv_vec"] = tuple(float(v) for v in delta_nm_inv_vec)
                dist_sigma_px, dist_sigma_nm_inv = self._propagate_superstructure_uncertainties(
                    delta_px=delta_px,
                    main_sigma=self.main_peak_corrected_sigma_px,
                    satellite_sigma=self.satellite_peak_corrected_sigma_px,
                    lx_nm=Lx_nm,
                    ly_nm=Ly_nm,
                )
                if dist_sigma_px is not None:
                    results["dist_px_sigma"] = dist_sigma_px
                if dist_sigma_nm_inv is not None:
                    results["dist_nm_inv_sigma"] = dist_sigma_nm_inv
                    dist_nm_value = results.get("dist_nm_inv")
                    if dist_nm_value and dist_nm_value > 1e-12:
                        results["periodicity_nm_sigma"] = dist_sigma_nm_inv / (dist_nm_value ** 2)

            def _flt_pair(pair):
                if not pair:
                    return None
                return (float(pair[0]), float(pair[1]))

            results["main_peak_raw_px"] = _flt_pair(self.main_peak_raw_refined_px)
            results["satellite_peak_raw_px"] = _flt_pair(self.satellite_peak_raw_refined_px)
            results["main_peak_corrected_px"] = _flt_pair(self.main_peak_corrected_ideal_px)
            results["satellite_peak_corrected_px"] = _flt_pair(self.satellite_peak_corrected_ideal_px)
            results["main_peak_raw_sigma_px"] = _flt_pair(self.main_peak_raw_sigma_px)
            results["satellite_peak_raw_sigma_px"] = _flt_pair(self.satellite_peak_raw_sigma_px)
            results["main_peak_corrected_sigma_px"] = _flt_pair(self.main_peak_corrected_sigma_px)
            results["satellite_peak_corrected_sigma_px"] = _flt_pair(self.satellite_peak_corrected_sigma_px)

            results["intensity_ratio_sigma"] = self._ratio_sigma(
                self.satellite_peak_intensity,
                self.satellite_peak_intensity_sigma,
                self.main_peak_intensity,
                self.main_peak_intensity_sigma,
            )
            results["amplitude_ratio_sigma"] = self._ratio_sigma(
                self.satellite_peak_amplitude,
                self.satellite_peak_amplitude_sigma,
                self.main_peak_amplitude,
                self.main_peak_amplitude_sigma,
            )
            results["max_value_ratio_sigma"] = self._ratio_sigma(
                self.satellite_peak_max_value,
                self.satellite_peak_max_value_sigma,
                self.main_peak_max_value,
                self.main_peak_max_value_sigma,
            )

            self._final_results = results

            dist_px_text = format_value_with_sigma(
                results.get("dist_px"),
                results.get("dist_px_sigma"),
                "px",
                value_precision=2,
                sigma_precision=2,
            )
            dist_nm_inv_text = format_value_with_sigma(
                results.get("dist_nm_inv"),
                results.get("dist_nm_inv_sigma"),
                "nm⁻¹",
                value_precision=4,
                sigma_precision=4,
            )
            self.distance_fft_label.setText(f"{dist_px_text} | {dist_nm_inv_text}")

            periodicity_text = format_value_with_sigma(
                results.get("periodicity_nm"),
                results.get("periodicity_nm_sigma"),
                "nm",
                value_precision=3,
                sigma_precision=3,
            )
            self.distance_real_space_label.setText(periodicity_text)

            intensity_text = format_ratio(
                results.get('intensity_ratio'),
                precision=3,
                sigma=results.get('intensity_ratio_sigma'),
                sigma_precision=3,
            )
            amplitude_text = format_ratio(
                results.get('amplitude_ratio'),
                precision=3,
                sigma=results.get('amplitude_ratio_sigma'),
                sigma_precision=3,
            )
            max_value_text = format_ratio(
                results.get('max_value_ratio'),
                precision=3,
                sigma=results.get('max_value_ratio_sigma'),
                sigma_precision=3,
            )
            self.intensity_ratio_label.setText(intensity_text)
            self.amplitude_ratio_label.setText(amplitude_text)
            self.max_value_label.setText(max_value_text)
            self.status_label.setText("Calculation successful.")
        else:
            QMessageBox.critical(self, "Calculation Error", "Could not calculate superstructure periodicity parameters.")
            self._final_results = None
            self.distance_fft_label.setText("Error"); self.distance_real_space_label.setText("Error")
            self.intensity_ratio_label.setText("Error"); self.amplitude_ratio_label.setText("Error"); self.max_value_label.setText("Error")

    def _on_add_satellite_peak_clicked(self, event):
        if not self.selection_roi.isVisible(): 
            QMessageBox.warning(self,"No ROI","Please place ROI on the main peak first.") 
            return
        results = self._refine_and_process_spot()
        if results:
            raw = results["raw"]
            corr = results["corrected"]
            self.satellite_peak_raw_refined_px = raw
            self.satellite_peak_corrected_ideal_px = corr
            self.satellite_peak_intensity = results["intensity"]
            self.satellite_peak_intensity_sigma = results.get("intensity_sigma")
            self.satellite_peak_amplitude = results["amplitude"]
            self.satellite_peak_amplitude_sigma = results.get("amplitude_sigma")
            self.satellite_peak_max_value = results["max_value"]
            self.satellite_peak_max_value_sigma = results.get("max_value_sigma")
            self.basic_satellite_periodicity_nm = results["d_spacing_nm"]
            self.satellite_peak_raw_sigma_px = results.get("raw_sigma")
            self.satellite_peak_corrected_sigma_px = results.get("corrected_sigma")
            logger.info(
                "Satelite peak selected/updated: Raw=%s, Corrected=%s, Intensity=%.2e, Amplitude=%.2e",
                raw,
                corr,
                self.satellite_peak_intensity,
                self.satellite_peak_amplitude,
            )
            self._update_all_ui_elements()
        self.selection_roi.setVisible(False)
        self._update_buttons_state()

    @pyqtSlot(object)
    def _handle_fft_image_click(self, event):
        """Handles click on the main FFT image in this dialog."""
        if not (self.fft_data is not None and self.fft_image_item and self.selection_roi):
            return

        if event.button() == Qt.MouseButton.LeftButton:
            pos_viewbox = self.fft_view_box.mapSceneToView(event.scenePos())
            mapped_pos = self.fft_image_item.mapToData(pos_viewbox)

            if mapped_pos is not None:
                kx, ky = mapped_pos.x(), mapped_pos.y()
                logger.debug(f"SpotDistanceDialog FFT click: data (kx, ky) = ({kx:.1f}, {ky:.1f})")

                roi_size = self.refinement_roi_size_spinbox.value()
                roi_x = kx - roi_size // 2
                roi_y = ky - roi_size // 2
                
                max_h, max_w = self.fft_data.shape
                roi_x = np.clip(roi_x, 0, max_w - roi_size)
                roi_y = np.clip(roi_y, 0, max_h - roi_size)

                self.selection_roi.setPos((roi_x, roi_y), update=False)
                self.selection_roi.setSize((roi_size, roi_size), update=False)
                self.selection_roi.setVisible(True)
                
                self._update_buttons_state()
                self._update_roi_previews()
                event.accept()
        else:
            event.ignore()

    @pyqtSlot(object)
    def _handle_roi_region_changing(self, roi_item: Optional[pg.ROI] = None):
        """Triggered when the ROI changes; updates live previews."""
        if roi_item is None: roi_item = self.selection_roi
        if not isinstance(roi_item, RectROI) or not roi_item.isVisible(): return

        current_roi_w = int(round(roi_item.size().x()))
        if current_roi_w != self.refinement_roi_size_spinbox.value() and \
           self.refinement_roi_size_spinbox.minimum() <= current_roi_w <= self.refinement_roi_size_spinbox.maximum() and \
           current_roi_w % 2 != 0:
            self.refinement_roi_size_spinbox.blockSignals(True)
            self.refinement_roi_size_spinbox.setValue(current_roi_w)
            self.refinement_roi_size_spinbox.blockSignals(False)
        
        self._clear_last_preview_gauss_fit()
        self._update_roi_previews()

    @pyqtSlot(int)
    def _on_refinement_roi_size_changed(self, value: int):
        """Slot called when the ROI size value in the spinbox changes."""
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
            self._handle_roi_region_changing()

    def _update_roi_previews(self):
        """Updates 2D ROI and Gaussian fit previews."""
        if not self.selection_roi.isVisible() or self.fft_data is None:
            if hasattr(self, 'roi_preview_2d_image_item'): self.roi_preview_2d_image_item.clear()
            if hasattr(self, 'gaussian_preview_2d_image_item'): self.gaussian_preview_2d_image_item.clear()
            self._clear_last_preview_gauss_fit()
            return

        roi_state = self.selection_roi.getState()
        x0r, y0r = int(round(roi_state['pos'].x())), int(round(roi_state['pos'].y()))
        wr, hr = int(round(roi_state['size'].x())), int(round(roi_state['size'].y()))
        
        mky, mkx = self.fft_data.shape
        y0c=np.clip(y0r,0,mky)
        y1c=np.clip(y0r+hr,0,mky)
        x0c=np.clip(x0r,0,mkx)
        x1c=np.clip(x0r+wr,0,mkx)
        
        if y1c <= y0c or x1c <= x0c:
            self.roi_preview_2d_image_item.clear()
            self.gaussian_preview_2d_image_item.clear()
            return
            
        roi_patch = self.fft_data[y0c:y1c, x0c:x1c]

        if roi_patch.size > 0:
            if self.enable_2d_roi_preview_checkbox.isChecked():
                self.roi_preview_2d_image_item.setImage(roi_patch.T)
                self.roi_preview_2d_plot.autoRange()
            else:
                self.roi_preview_2d_image_item.clear()

            if self.gauss_2d_container.isVisible():
                fitted_gauss_2d = None
                if self.enable_gauss_2d_preview_checkbox.isChecked():
                    if PEAK_FITTING_MODULE_AVAILABLE and SCIPY_OPTIMIZE_AVAILABLE and callable(_gaussian_2d) and callable(scipy_curve_fit):
                        ph, pw = roi_patch.shape
                        py_g, px_g = np.mgrid[0:ph,0:pw]
                        pxy_flat_g = (py_g.flatten(), px_g.flatten())
                        pdata_flat_g = roi_patch.flatten()
                        try:
                            p0g = [roi_patch.max()-roi_patch.min(), ph/2., pw/2., pw/4., ph/4., 0., roi_patch.min()]
                            popt_g, _ = scipy_curve_fit(_gaussian_2d, pxy_flat_g, pdata_flat_g, p0=p0g, maxfev=2000)
                            
                            self.last_preview_gauss_fit_popt = popt_g
                            afk_g, afky_g = x0r + popt_g[2], y0r + popt_g[1]
                            self.last_preview_gauss_fit_center_abs = (afk_g, afky_g)
                            self.last_preview_gauss_roi_state = roi_state.copy()
                            
                            fitted_gauss_flat = _gaussian_2d(pxy_flat_g, *popt_g)
                            fitted_gauss_2d = fitted_gauss_flat.reshape(ph, pw)
                        except Exception as e:
                            logger.warning(f"DistDlg Preview GaussFit Fail: {e}")
                            self._clear_last_preview_gauss_fit()
                            fitted_gauss_2d = roi_patch
                    
                    if fitted_gauss_2d is not None:
                        self.gaussian_preview_2d_image_item.setImage(fitted_gauss_2d.T)
                    else:
                        self.gaussian_preview_2d_image_item.setImage(roi_patch.T)
                    self.gaussian_preview_2d_plot.autoRange()
                else:
                    self.gaussian_preview_2d_image_item.clear()
        else:
            self.roi_preview_2d_image_item.clear()
            self.gaussian_preview_2d_image_item.clear()

    def _display_substrate_transform_info(self):
        """
        Fills the UI labels with information about the substrate transformation,
        which was passed to the dialog during its creation.
        """
        if self.sub_transform_analysis:
            self.dist_sub_transform_info_label_status.setText("Status: Available")
            
            rot_display = format_value_with_sigma(
                self.sub_transform_analysis.get('rotation_angle_deg'),
                self.sub_transform_analysis.get('rotation_angle_deg_sigma'),
                'deg',
                value_precision=2,
                sigma_precision=2,
            )
            self.dist_sub_transform_info_label_rot.setText(rot_display)

            stretch_pair = format_pair_with_sigma(
                self.sub_transform_analysis.get('principal_stretches'),
                self.sub_transform_analysis.get('principal_stretches_sigma'),
                precision=3,
                sigma_precision=3,
            )
            self.dist_sub_transform_info_label_scale.setText(stretch_pair)

            rmse_text = format_float(self.sub_transform_analysis.get('rmse'), precision=3)
            rmse_display = rmse_text if rmse_text == '-' else f"{rmse_text} px"
            self.dist_sub_transform_info_label_rmse.setText(rmse_display)
            
            logger.info("Displayed available substrate transformation info.")
        else:
            self.dist_sub_transform_info_label_status.setText("Status: Not Calculated / Not Available")
            self.dist_sub_transform_info_label_rot.setText("-")
            self.dist_sub_transform_info_label_scale.setText("-")
            self.dist_sub_transform_info_label_rmse.setText("-")
            logger.warning("Substrate transformation info not passed to dialog.")

    def _update_all_ui_elements(self):
        """Updates all UI elements."""
        self._update_spot_info_display()
        self._redraw_all_markers_on_fft()
        self._update_buttons_state()
        self._auto_calculate_results()
    
    def _auto_calculate_results(self): pass

    def _update_spot_info_display(self):
        """
        Updates the text fields, displaying information about the selected peaks.
        """
        if self.main_peak_raw_refined_px and self.main_peak_corrected_ideal_px and self.main_peak_intensity is not None:
            corr = self.main_peak_corrected_ideal_px
            intensity = self.main_peak_intensity
            amplitude = self.main_peak_amplitude
            max_value = self.main_peak_max_value
            d_spacing_nm = self.basic_main_periodicity_nm
            corr_text = format_pair_with_sigma(
                corr,
                self.main_peak_corrected_sigma_px,
                precision=2,
                sigma_precision=3,
            )
            d_spacing_text = format_value_with_sigma(
                d_spacing_nm,
                None,
                "nm",
                value_precision=2,
                sigma_precision=2,
            )
            self.main_peak_info_label.setText(
                f"Corrected: {corr_text} px | I: {intensity:.2e} \n"
                f"A: {amplitude:.2e} | Max: {max_value:.2e} \n d_spacing: {d_spacing_text}"
            )
        else:
            self.main_peak_info_label.setText("Not Selected")
        
        if self.satellite_peak_raw_refined_px and self.satellite_peak_corrected_ideal_px and self.satellite_peak_intensity is not None:
            corr = self.satellite_peak_corrected_ideal_px
            intensity = self.satellite_peak_intensity
            amplitude = self.satellite_peak_amplitude
            max_value = self.satellite_peak_max_value
            d_spacing_nm = self.basic_satellite_periodicity_nm
            corr_text = format_pair_with_sigma(
                corr,
                self.satellite_peak_corrected_sigma_px,
                precision=2,
                sigma_precision=3,
            )
            d_spacing_text = format_value_with_sigma(
                d_spacing_nm,
                None,
                "nm",
                value_precision=2,
                sigma_precision=2,
            )
            self.satellite_peak_info_label.setText(
                f"Corrected: {corr_text} px | I: {intensity:.2e} \n"
                f"A: {amplitude:.2e} | Max: {max_value:.2e} \n d_spacing: {d_spacing_text}"
            )
        else:
            self.satellite_peak_info_label.setText("Not Selected")

    def _redraw_all_markers_on_fft(self):
        """
        Draws markers for the main peak and satellite peak (raw/refined and corrected positions) on the FFT image.
        """
        if self.main_peak_raw_marker: 
            self.fft_view_box.removeItem(self.main_peak_raw_marker)
            self.main_peak_raw_marker=None
        if self.main_peak_corrected_marker: 
            self.fft_view_box.removeItem(self.main_peak_corrected_marker)
            self.main_peak_corrected_marker=None
        if self.satellite_raw_marker: 
            self.fft_view_box.removeItem(self.satellite_raw_marker)
            self.satellite_raw_marker=None
        if self.satellite_corrected_marker: 
            self.fft_view_box.removeItem(self.satellite_corrected_marker)
            self.satellite_corrected_marker=None

        if self.main_peak_raw_refined_px:
            self.main_peak_raw_marker = pg.ScatterPlotItem(
                spots=[{'pos': self.main_peak_raw_refined_px, 'symbol': 'o', 'size': 14, 'pen': pg.mkPen('y', width=2), 'brush': pg.mkBrush(255, 255, 0, 120)}]
            )
            self.fft_view_box.addItem(self.main_peak_raw_marker)
        
        if self.satellite_peak_raw_refined_px:
            self.satellite_raw_marker = pg.ScatterPlotItem(
                spots=[{'pos': self.satellite_peak_raw_refined_px, 'symbol': 'o', 'size': 10, 'pen': pg.mkPen('orange', width=1.5), 'brush': pg.mkBrush(255, 165, 0, 100)}]
            )
            self.fft_view_box.addItem(self.satellite_raw_marker)

        if self.main_peak_corrected_ideal_px:
            self.main_peak_corrected_marker = pg.ScatterPlotItem(
                spots=[{'pos': tuple(self.main_peak_corrected_ideal_px), 'symbol': 'x', 'size': 14, 'pen': pg.mkPen('c', width=2)}]
            )
            self.fft_view_box.addItem(self.main_peak_corrected_marker)
                
        if self.satellite_peak_corrected_ideal_px:
                self.satellite_corrected_marker = pg.ScatterPlotItem(
                    spots=[{'pos': tuple(self.satellite_peak_corrected_ideal_px), 'symbol': 'x', 'size': 10, 'pen': pg.mkPen('cyan', width=1.5)}]
                )
                self.fft_view_box.addItem(self.satellite_corrected_marker)
        
        if self.main_peak_corrected_ideal_px and self.satellite_peak_corrected_ideal_px:
            self.calculate_distance_button.setEnabled(True)
        else:
            self.calculate_distance_button.setEnabled(False)

    def _clear_last_preview_gauss_fit(self):
        """
        Resets the stored results from the last live Gaussian preview fit.

        This is called when the ROI changes or the selection mode is altered,
        invalidating the previous preview calculation.
        """
        self.last_preview_gauss_fit_popt = None
        self.last_preview_gauss_fit_center_abs = None
        self.last_preview_gauss_roi_state = None
        logger.debug("Cleared last preview Gaussian fit results.")
    
    def _refine_and_process_spot(self) -> Optional[Dict[str, Any]]:
        if not self.selection_roi.isVisible() or self.fft_data is None:
            return None

        roi_state = self.selection_roi.getState()
        x0r, y0r = int(round(roi_state["pos"].x())), int(round(roi_state["pos"].y()))
        wr, hr = int(round(roi_state["size"].x())), int(round(roi_state["size"].y()))
        ckx_roi, cky_roi = x0r + wr // 2, y0r + hr // 2

        if not (
            PEAK_FITTING_MODULE_AVAILABLE
            and fit_2d_gaussian_in_roi_with_all_data
            and callable(fit_2d_gaussian_in_roi_with_all_data)
        ):
            return None

        pr = self.refinement_roi_size_spinbox.value() // 2
        mh, mw = self.fft_data.shape
        eff_cky, eff_ckx = np.clip(cky_roi, pr, mh - 1 - pr), np.clip(ckx_roi, pr, mw - 1 - pr)

        fit_res = fit_2d_gaussian_in_roi_with_all_data(self.fft_data, (eff_cky, eff_ckx), pr)
        if not fit_res:
            logger.warning("Gaussian fit failed.")
            return None

        refined_kx_fft = float(fit_res.center[1])
        refined_ky_fft = float(fit_res.center[0])
        raw_refined_spot = (refined_kx_fft, refined_ky_fft)

        amplitude = 0.0
        amplitude_sigma: Optional[float] = None
        sigma_y = 0.0
        sigma_x = 0.0
        intensity = 0.0
        intensity_sigma: Optional[float] = None
        if fit_res.popt is not None and len(fit_res.popt) >= 5:
            amplitude = float(fit_res.popt[0])
            sigma_y = float(fit_res.popt[3])
            sigma_x = float(fit_res.popt[4])
            intensity = 2 * math.pi * abs(amplitude) * abs(sigma_x) * abs(sigma_y)
            if fit_res.pcov is not None and fit_res.pcov.shape[0] >= 5:
                var_amp = float(fit_res.pcov[0, 0])
                if math.isfinite(var_amp) and var_amp >= 0.0:
                    amplitude_sigma = math.sqrt(var_amp)
                amp_sigma_indices = np.ix_([0, 3, 4], [0, 3, 4])
                amp_cov = np.asarray(fit_res.pcov[amp_sigma_indices], dtype=float)
                if amp_cov.shape == (3, 3) and np.all(np.isfinite(amp_cov)):
                    intensity_sigma = self._compute_intensity_sigma(
                        amplitude,
                        sigma_x,
                        sigma_y,
                        amp_cov,
                    )

        roi_patch_used = fit_res.roi_patch
        max_value = float(np.max(roi_patch_used)) if roi_patch_used.size > 0 else 0.0
        max_value_sigma = None
        if hasattr(fit_res, "noise_sigma") and fit_res.noise_sigma is not None:
            try:
                nv = float(fit_res.noise_sigma)
            except (TypeError, ValueError):
                nv = None
            else:
                if math.isfinite(nv) and nv >= 0.0:
                    max_value_sigma = nv

        raw_sigma: Optional[Tuple[float, float]] = None
        if fit_res.center_std:
            std_y, std_x = fit_res.center_std
            try:
                raw_sigma = (abs(float(std_x)), abs(float(std_y)))
            except (TypeError, ValueError):
                raw_sigma = None

        raw_cov = None
        if raw_sigma:
            raw_cov = np.array(
                [[raw_sigma[0] ** 2, 0.0], [0.0, raw_sigma[1] ** 2]],
                dtype=float,
            )

        corrected_spot: Optional[Tuple[float, float]] = raw_refined_spot
        corrected_cov = raw_cov.copy() if raw_cov is not None else None
        if self.apply_substrate_transform_checkbox.isChecked():
            corrected_spot = None
            try:
                if (
                    self.sub_F_m2i is not None
                    and self.sub_t_m2i is not None
                    and apply_affine_transform
                ):
                    corrected_array = apply_affine_transform(
                        np.array([raw_refined_spot]),
                        self.sub_F_m2i,
                        self.sub_t_m2i,
                    )
                    if corrected_array is not None:
                        corrected_spot = tuple(corrected_array[0])
                        if corrected_cov is not None:
                            corrected_cov = self.sub_F_m2i @ corrected_cov @ self.sub_F_m2i.T
            except Exception as exc:
                logger.error(f"Error correcting spot {raw_refined_spot}: {exc}")
        if corrected_spot is None:
            logger.warning(f"Could not correct spot {raw_refined_spot}.")
            return None

        d_spacing_nm = None
        if self.fft_data is not None and self.history_manager:
            root_node = self.history_manager.get_root_node_for_node(self.current_fft_node_id)
            if root_node and root_node.parameters:
                lx = root_node.parameters.get("size_nm_x")
                ly = root_node.parameters.get("size_nm_y")
                if lx and ly:
                    d_spacing_nm = calculate_d_spacing_from_ideal_spot(
                        spot_corrected_ideal_px=corrected_spot,
                        fft_shape=self.fft_data.shape,
                        lx_nm=lx,
                        ly_nm=ly,
                    )

        corrected_sigma = self._sigma_from_covariance(corrected_cov)

        return {
            "raw": raw_refined_spot,
            "corrected": corrected_spot,
            "intensity": float(intensity),
            "intensity_sigma": intensity_sigma,
            "amplitude": float(amplitude),
            "amplitude_sigma": amplitude_sigma,
            "max_value": float(max_value),
            "max_value_sigma": max_value_sigma,
            "d_spacing_nm": d_spacing_nm,
            "raw_sigma": raw_sigma,
            "corrected_sigma": corrected_sigma,
        }

    @staticmethod
    def _sigma_from_covariance(covariance: Optional[np.ndarray]) -> Optional[Tuple[float, float]]:
        if covariance is None or getattr(covariance, "shape", None) != (2, 2):
            return None
        try:
            sigma_x = math.sqrt(max(float(covariance[0, 0]), 0.0))
            sigma_y = math.sqrt(max(float(covariance[1, 1]), 0.0))
        except (TypeError, ValueError):
            return None
        return (sigma_x, sigma_y)

    @staticmethod
    def _compute_intensity_sigma(
        amplitude: float,
        sigma_x: float,
        sigma_y: float,
        covariance: Optional[np.ndarray],
    ) -> Optional[float]:
        if covariance is None:
            return None
        try:
            amp = float(amplitude)
            sx = float(sigma_x)
            sy = float(sigma_y)
        except (TypeError, ValueError):
            return None
        abs_amp = abs(amp)
        abs_sx = abs(sx)
        abs_sy = abs(sy)
        prefactor = 2.0 * math.pi
        intensity = prefactor * abs_amp * abs_sx * abs_sy
        if not math.isfinite(intensity):
            return None
        cov = np.asarray(covariance, dtype=float)
        if cov.shape != (3, 3) or not np.all(np.isfinite(cov)):
            return None

        def _component_derivative(component_value: float, other_product: float) -> float:
            if abs(component_value) < 1e-12:
                return 0.0
            return prefactor * other_product * (1.0 if component_value >= 0.0 else -1.0)

        d_da = _component_derivative(amp, abs_sx * abs_sy)
        d_dsigma_y = _component_derivative(sy, abs_amp * abs_sx)
        d_dsigma_x = _component_derivative(sx, abs_amp * abs_sy)
        gradient = np.array([d_da, d_dsigma_y, d_dsigma_x], dtype=float)
        variance = float(gradient @ cov @ gradient.T)
        if not math.isfinite(variance) or variance < 0.0:
            return None
        return math.sqrt(variance)

    def _propagate_superstructure_uncertainties(
        self,
        *,
        delta_px: Tuple[float, float],
        main_sigma: Optional[Tuple[float, float]],
        satellite_sigma: Optional[Tuple[float, float]],
        lx_nm: float,
        ly_nm: float,
    ) -> Tuple[Optional[float], Optional[float]]:
        var_dx_px = self._combined_component_variance(main_sigma, satellite_sigma, index=0)
        var_dy_px = self._combined_component_variance(main_sigma, satellite_sigma, index=1)
        dist_sigma_px = self._magnitude_sigma(delta_px, var_dx_px, var_dy_px)

        dist_sigma_nm_inv = None
        if lx_nm and ly_nm and (var_dx_px is not None or var_dy_px is not None):
            delta_nm = (
                delta_px[0] / lx_nm if lx_nm else 0.0,
                delta_px[1] / ly_nm if ly_nm else 0.0,
            )
            var_dx_nm = var_dx_px / (lx_nm**2) if var_dx_px is not None else None
            var_dy_nm = var_dy_px / (ly_nm**2) if var_dy_px is not None else None
            dist_sigma_nm_inv = self._magnitude_sigma(delta_nm, var_dx_nm, var_dy_nm)

        return dist_sigma_px, dist_sigma_nm_inv

    @staticmethod
    def _combined_component_variance(
        first_sigma: Optional[Tuple[float, float]],
        second_sigma: Optional[Tuple[float, float]],
        *,
        index: int,
    ) -> Optional[float]:
        variance = 0.0
        has_value = False
        for sigma_pair in (first_sigma, second_sigma):
            if sigma_pair is None or len(sigma_pair) <= index:
                continue
            component = sigma_pair[index]
            if component is None:
                continue
            try:
                component_value = float(component)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(component_value):
                continue
            variance += max(component_value, 0.0) ** 2
            has_value = True
        return variance if has_value else None

    @staticmethod
    def _magnitude_sigma(
        vector: Tuple[float, float],
        var_x: Optional[float],
        var_y: Optional[float],
    ) -> Optional[float]:
        if vector is None:
            return None
        dx, dy = vector
        norm = math.hypot(dx, dy)
        if norm <= 1e-9:
            return None
        variance = 0.0
        has_component = False
        if var_x is not None and var_x >= 0.0:
            variance += (dx / norm) ** 2 * var_x
            has_component = True
        if var_y is not None and var_y >= 0.0:
            variance += (dy / norm) ** 2 * var_y
            has_component = True
        if not has_component:
            return None
        return math.sqrt(variance) if variance > 0.0 else 0.0

    @staticmethod
    def _ratio_sigma(
        numerator: Optional[float],
        numerator_sigma: Optional[float],
        denominator: Optional[float],
        denominator_sigma: Optional[float],
    ) -> Optional[float]:
        try:
            num = float(numerator)
            den = float(denominator)
        except (TypeError, ValueError):
            return None
        if abs(den) < 1e-12:
            return None

        variance = 0.0
        has_component = False
        if numerator_sigma is not None:
            try:
                sigma_num = float(numerator_sigma)
            except (TypeError, ValueError):
                sigma_num = None
            else:
                if math.isfinite(sigma_num) and sigma_num >= 0.0:
                    variance += (sigma_num / den) ** 2
                    has_component = True
        if denominator_sigma is not None:
            try:
                sigma_den = float(denominator_sigma)
            except (TypeError, ValueError):
                sigma_den = None
            else:
                if math.isfinite(sigma_den) and sigma_den >= 0.0:
                    variance += ((num * sigma_den) / (den ** 2)) ** 2
                    has_component = True

        if not has_component or variance < 0.0 or not math.isfinite(variance):
            return None
        return math.sqrt(variance)

    @pyqtSlot()
    def _on_add_main_spot_clicked(self):
        if not self.selection_roi.isVisible(): 
            QMessageBox.warning(self,"No ROI","Please place ROI on the main peak first.")
            return
        
        results = self._refine_and_process_spot()
        if results:
            raw = results["raw"]
            corr = results["corrected"]
            self.main_peak_raw_refined_px = raw
            self.main_peak_corrected_ideal_px = corr
            self.main_peak_intensity = results["intensity"]
            self.main_peak_intensity_sigma = results.get("intensity_sigma")
            self.main_peak_amplitude = results["amplitude"]
            self.main_peak_amplitude_sigma = results.get("amplitude_sigma")
            self.main_peak_max_value = results["max_value"]
            self.main_peak_max_value_sigma = results.get("max_value_sigma")
            self.basic_main_periodicity_nm = results["d_spacing_nm"]
            self.main_peak_raw_sigma_px = results.get("raw_sigma")
            self.main_peak_corrected_sigma_px = results.get("corrected_sigma")
            logger.info(
                "Main peak selected/updated: Raw=%s, Corrected=%s, Intensity=%.2e, Amplitude=%.2e",
                raw,
                corr,
                self.main_peak_intensity,
                self.main_peak_amplitude,
            )
            self._update_all_ui_elements()
        self.selection_roi.setVisible(False)
        self._update_buttons_state()

    def _update_buttons_state(self):
        roi_is_visible = self.selection_roi.isVisible()
        main_peak_exists = self.main_peak_raw_refined_px is not None
        
        self.add_main_spot_button.setEnabled(roi_is_visible)
        self.add_satellite_spot_button.setEnabled(roi_is_visible and main_peak_exists)

    def accept(self):
        """Ensure final results are saved before accepting."""
        if not self._final_results:
             reply = QMessageBox.question(self, "No Results", 
                                          "No calculations have been performed. Close anyway?",
                                          QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                          QMessageBox.StandardButton.No)
             if reply == QMessageBox.StandardButton.No:
                 return 

        logger.info("SuperstructurePeriodicityDialog accepted.")
        super().accept()
    
    def get_analysis_results(self) -> Optional[Dict[str, Any]]:
        """Returns a dictionary with the final calculated parameters."""
        return self._final_results


    
