# lfa/gui/dialogs/domain_walls_analysis_dialog.py
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
from PyQt6.QtGui import QPen

try:
    import pyqtgraph as pg
    from pyqtgraph import GraphicsLayoutWidget, ImageItem, ViewBox, RectROI, ScatterPlotItem
    PYQTGRAPH_AVAILABLE = True
except ImportError: # pragma: no cover
    pg = None; GraphicsLayoutWidget = None; ImageItem = None; ViewBox = None; RectROI = None; ScatterPlotItem = None; PYQTGRAPH_AVAILABLE = False
    logging.error("DomainWallsAnalysisDialog: PyQtGraph not found.")

# Importy z projektu (dostosuj ścieżki, jeśli są inne)
try:
    from ...logic.app_controller import AppController
    from ...logic.history_manager import HistoryManager
    from ...analysis.peak_fitting import find_max_pixel_in_roi, fit_2d_gaussian_in_roi, _gaussian_2d, SCIPY_AVAILABLE
    from ...analysis.lattice import convert_g_vector_px_to_nm_inv
    from ...core.history import HistoryNode
    from ...analysis.drift_correction import apply_affine_transform
    PEAK_FITTING_MODULE_AVAILABLE = True
except ImportError as e: # pragma: no cover
    AppController=None
    HistoryManager=None
    HistoryNode=None
    apply_affine_transform=None
    find_max_pixel_in_roi=None
    fit_2d_gaussian_in_roi=None
    _gaussian_2d=None; SCIPY_AVAILABLE=False
    convert_g_vector_px_to_nm_inv=None
    logging.error(f"DomainWallsAnalysisDialog: Error importing project modules: {e}")
    if find_max_pixel_in_roi is None: 
        def find_max_pixel_in_roi(data, center, radius): return center
    if fit_2d_gaussian_in_roi is None: 
        def fit_2d_gaussian_in_roi(data, center, radius): return None
    if _gaussian_2d is None: 
        def _gaussian_2d(*args, **kwargs): raise ImportError("Gaussian 2D function is not available")

try:
    from scipy.optimize import curve_fit as scipy_curve_fit
    SCIPY_OPTIMIZE_AVAILABLE = True
except ImportError: # pragma: no cover
    SCIPY_OPTIMIZE_AVAILABLE = False; logging.error("DomainWallsAnalysisDialog: SciPy not found.")
    def scipy_curve_fit(*args, **kwargs): raise ImportError("scipy.optimize.curve_fit is not available")

logger = logging.getLogger(__name__)

# W tym dialogu jest tylko jedna metoda uściślania, ale stałe mogą się przydać
REFINEMENT_GAUSSIAN_FIT = "2D Gaussian Fit"

class DomainWallsAnalysisDialog(QDialog):
    def __init__(self,
                 fft_image_data: Optional[np.ndarray],
                 history_manager: HistoryManager,
                 current_fft_node_id: str,
                 substrate_F_m2i: Optional[np.ndarray] = None,
                 substrate_t_m2i: Optional[np.ndarray] = None,
                 substrate_transform_analysis: Optional[Dict[str, Any]] = None,
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

        if not PYQTGRAPH_AVAILABLE: # pragma: no cover
            QVBoxLayout(self).addWidget(QLabel("Critical Error: PyQtGraph is required for this dialog."))
            self.setWindowTitle("Error"); return

        self.setWindowTitle("Domain Wall Analysis")
        self.setMinimumSize(1200, 700)

        # Listy do przechowywania danych
        self.selected_spots_raw_refined_fft_px: List[Tuple[float, float]] = [] # Max 2
        self.corrected_spots_ideal_system_px: List[Optional[Tuple[float, float]]] = [] # Max 2

        # Markery na obrazie FFT
        self.raw_refined_spot_markers: Optional[ScatterPlotItem] = None
        self.corrected_spot_display_markers: Optional[ScatterPlotItem] = None

        # Atrybuty dla podglądu Gaussa
        self.last_preview_gauss_fit_popt: Optional[np.ndarray] = None
        self.last_preview_gauss_fit_center_abs: Optional[Tuple[float, float]] = None
        self.last_preview_gauss_roi_state: Optional[Dict] = None
        
        self._init_ui()
        self._connect_signals()
        self._update_list_widget()
        self._redraw_all_markers_on_fft()
        self._update_buttons_state()
        self._display_substrate_transform_info()

        logger.debug("DomainWallsAnalysisDialog initialized.")

    def _init_ui(self):
        top_level_layout = QHBoxLayout(self)
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        top_level_layout.addWidget(main_splitter)

        # === LEWY PANEL: Kontrolki ===
        left_controls_widget = QWidget()
        left_controls_layout = QVBoxLayout(left_controls_widget)
        left_controls_widget.setMinimumWidth(300); left_controls_widget.setMaximumWidth(380)

        refinement_group = QGroupBox("Spot Selection (Max 2 spots)")
        refinement_layout = QFormLayout(refinement_group)
        refinement_layout.addRow(QLabel("Refinement Method: 2D Gaussian Fit"))
        self.refinement_roi_size_spinbox = QSpinBox()
        self.refinement_roi_size_spinbox.setMinimum(3); self.refinement_roi_size_spinbox.setMaximum(31)
        self.refinement_roi_size_spinbox.setSingleStep(2); self.refinement_roi_size_spinbox.setValue(self.refinement_roi_size)
        refinement_layout.addRow("Refinement Area Size (px):", self.refinement_roi_size_spinbox)
        self.add_spot_button = QPushButton("Refine & Add Selected Spot")
        self.add_spot_button.setEnabled(False) # Aktywowany po umieszczeniu ROI
        refinement_layout.addRow(self.add_spot_button)
        left_controls_layout.addWidget(refinement_group)
        
        sub_transform_group = QGroupBox("Substrate Transformation Info (Applied)")
        sub_transform_layout = QFormLayout(sub_transform_group)
        self.sub_transform_info_label_status = QLabel("Status: -")
        self.sub_transform_info_label_rot = QLabel("Sub. Rotation: -")
        self.sub_transform_info_label_scale = QLabel("Sub. Scale (X,Y): -")
        self.sub_transform_info_label_rmse = QLabel("Sub. RMSE (px): -")
        sub_transform_layout.addRow(self.sub_transform_info_label_status)
        sub_transform_layout.addRow(self.sub_transform_info_label_rot)
        sub_transform_layout.addRow(self.sub_transform_info_label_scale)
        sub_transform_layout.addRow(self.sub_transform_info_label_rmse)
        left_controls_layout.addWidget(sub_transform_group)
        
        left_controls_layout.addStretch(1)
        main_splitter.addWidget(left_controls_widget)

        # === CENTRALNY PANEL: Główny obraz FFT ===
        self.fft_plot_widget = GraphicsLayoutWidget()
        self.fft_view_box = self.fft_plot_widget.addViewBox(row=0, col=0, lockAspect=True, invertY=True)
        self.fft_image_item = ImageItem()
        self.fft_view_box.addItem(self.fft_image_item)
        self.fft_view_box.setMenuEnabled(True); self.fft_view_box.setMouseMode(ViewBox.PanMode)
        if self.fft_data is not None: self.fft_image_item.setImage(self.fft_data.T)
        self.selection_roi = RectROI(pos=(0,0), size=(self.refinement_roi_size, self.refinement_roi_size), pen=pg.mkPen('cyan', width=2), movable=True, resizable=True)
        self.fft_view_box.addItem(self.selection_roi); self.selection_roi.setVisible(False)
        main_splitter.addWidget(self.fft_plot_widget)

        # === PRAWY PANEL: Podglądy i Wyniki ===
        right_panel_widget = QWidget(); right_panel_layout = QVBoxLayout(right_panel_widget)
        right_panel_widget.setMinimumWidth(400); right_panel_widget.setMaximumWidth(500)

        preview_group = QGroupBox("Live Previews (Gaussian Fit)")
        preview_grid_layout = QGridLayout(preview_group)
        # 2D ROI Preview
        roi_2d_container = QWidget(); roi_2d_v_layout = QVBoxLayout(roi_2d_container)
        roi_2d_h_layout = QHBoxLayout(); roi_2d_h_layout.addWidget(QLabel("ROI 2D Preview:")); self.enable_2d_roi_preview_checkbox = QCheckBox("Enable"); self.enable_2d_roi_preview_checkbox.setChecked(True); roi_2d_h_layout.addWidget(self.enable_2d_roi_preview_checkbox); roi_2d_h_layout.addStretch()
        roi_2d_v_layout.addLayout(roi_2d_h_layout); self.roi_preview_2d_widget = GraphicsLayoutWidget(); self.roi_preview_2d_widget.setMinimumHeight(150); self.roi_preview_2d_widget.setMaximumHeight(200); self.roi_preview_2d_plot = self.roi_preview_2d_widget.addViewBox(lockAspect=True, invertY=True); self.roi_preview_2d_image_item = ImageItem(); self.roi_preview_2d_plot.addItem(self.roi_preview_2d_image_item); roi_2d_v_layout.addWidget(self.roi_preview_2d_widget, 1); preview_grid_layout.addWidget(roi_2d_container, 0, 0)
        # 2D Gaussian Fit Preview
        gauss_2d_container = QWidget(); gauss_2d_v_layout = QVBoxLayout(gauss_2d_container)
        gauss_2d_h_layout = QHBoxLayout(); gauss_2d_h_layout.addWidget(QLabel("Gaussian Fit 2D Preview:")); self.enable_gauss_2d_preview_checkbox = QCheckBox("Enable"); self.enable_gauss_2d_preview_checkbox.setChecked(True); gauss_2d_h_layout.addWidget(self.enable_gauss_2d_preview_checkbox); gauss_2d_h_layout.addStretch()
        gauss_2d_v_layout.addLayout(gauss_2d_h_layout); self.gaussian_preview_2d_widget = GraphicsLayoutWidget(); self.gaussian_preview_2d_widget.setMinimumHeight(150); self.gaussian_preview_2d_widget.setMaximumHeight(200); self.gaussian_preview_2d_plot = self.gaussian_preview_2d_widget.addViewBox(lockAspect=True, invertY=True); self.gaussian_preview_2d_image_item = ImageItem(); self.gaussian_preview_2d_plot.addItem(self.gaussian_preview_2d_image_item); gauss_2d_v_layout.addWidget(self.gaussian_preview_2d_widget, 1); preview_grid_layout.addWidget(gauss_2d_container, 0, 1)
        right_panel_layout.addWidget(preview_group)

        spots_dist_group = QGroupBox("Selected Spots (Max 2)")
        spots_dist_layout = QVBoxLayout(spots_dist_group)
        self.spots_list_widget = QListWidget(); self.spots_list_widget.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection) # Zmieniono na NoSelection
        spots_dist_layout.addWidget(self.spots_list_widget)
        spot_buttons_layout = QHBoxLayout(); self.remove_last_spot_button = QPushButton("Remove Last Spot"); self.clear_all_spots_button = QPushButton("Clear All Spots"); spot_buttons_layout.addWidget(self.remove_last_spot_button); spot_buttons_layout.addWidget(self.clear_all_spots_button); spots_dist_layout.addLayout(spot_buttons_layout)
        right_panel_layout.addWidget(spots_dist_group)
        
        results_group = QGroupBox("Results"); results_layout = QFormLayout(results_group)
        self.calculate_distance_button = QPushButton("Calculate Distance Between Corrected Spots"); self.calculate_distance_button.setEnabled(False)
        results_layout.addRow(self.calculate_distance_button)
        self.distance_fft_label = QLabel("Δg* (px): - | (nm⁻¹): -"); self.distance_real_space_label = QLabel("Periodicity P (nm): -")
        results_layout.addRow("Distance in k-space:", self.distance_fft_label); results_layout.addRow("Real Space Periodicity:", self.distance_real_space_label)
        right_panel_layout.addWidget(results_group)

        self.status_label = QLabel("Click on FFT to select spots for analysis."); right_panel_layout.addWidget(self.status_label)
        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close); right_panel_layout.addWidget(self.button_box)
        right_panel_layout.addStretch(1); main_splitter.addWidget(right_panel_widget)
        main_splitter.setSizes([320,550,330]); main_splitter.setStretchFactor(1,1)

    def _connect_signals(self):
        # TODO (Krok 3)
        pass

    def _update_spot_distance_list_widget(self):
        # TODO (Krok 3)
        pass

    def _redraw_all_markers_on_fft(self):
        # TODO (Krok 3)
        pass
        
    def _update_add_spot_button_state(self):
        # TODO (Krok 3)
        pass

    def _display_substrate_transform_info(self):
        # TODO (Krok 3)
        pass

    def _update_list_widget(self):
        # TODO (Krok 3)
        pass

    def _update_buttons_state(self):
        # TODO (Krok 3)
        pass
    
    # ... inne szkielety metod do implementacji w kolejnych krokach ...