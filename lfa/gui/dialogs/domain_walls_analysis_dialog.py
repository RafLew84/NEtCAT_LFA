# lfa/gui/dialogs/domain_walls_analysis_dialog.py
import logging
from typing import List, Tuple, Optional, Dict, Any
import numpy as np

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QDialogButtonBox,
    QLabel, QListWidget, QAbstractItemView, QWidget, QGroupBox,
    QFormLayout, QSpinBox, QCheckBox, QMessageBox,
    QGridLayout, QSplitter, QLineEdit
)

from PyQt6.QtCore import Qt, pyqtSlot, QTimer 

try:
    import pyqtgraph as pg
    from pyqtgraph import GraphicsLayoutWidget, ImageItem, ViewBox, RectROI, ScatterPlotItem
    PYQTGRAPH_AVAILABLE = True
except ImportError: # pragma: no cover
    pg = None; GraphicsLayoutWidget = None; ImageItem = None; ViewBox = None; RectROI = None; ScatterPlotItem = None; PYQTGRAPH_AVAILABLE = False
    logging.error("DomainWallsAnalysisDialog: PyQtGraph not found.")

# Importy z projektu (tylko dla type hinting w konstruktorze)
try:
    from ...logic.app_controller import AppController
    from ...logic.history_manager import HistoryManager
except ImportError: # pragma: no cover
    AppController = None; HistoryManager = None

logger = logging.getLogger(__name__)

class DomainWallsAnalysisDialog(QDialog):
    """
    Dialog for analyzing domain wall structures by selecting a main peak
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

        # Przechowywanie danych i referencji
        self.fft_data = fft_image_data
        self.history_manager = history_manager
        self.current_fft_node_id = current_fft_node_id
        self.refinement_roi_size = default_refinement_roi_size
        self.sub_F_m2i = substrate_F_m2i
        self.sub_t_m2i = substrate_t_m2i
        self.sub_transform_analysis = substrate_transform_analysis

                # Sprawdź, czy przekazano poprawne referencje
        if not (self.history_manager and self.current_fft_node_id):
            QMessageBox.critical(self, "Initialization Error", "History context was not provided to the dialog.")
            QTimer.singleShot(0, self.reject)
            return

        # Pobierz węzeł FFT i sprawdź jego parametry
        fft_node = self.history_manager.get_node_by_id(self.current_fft_node_id)
        print("=======================================================")
        print(fft_node.parameters['scaling_mode'])
        is_power_scale = False
        if fft_node and fft_node.parameters:
            if fft_node.parameters.get("scaling_mode") == "power":
                is_power_scale = True
        
        if not is_power_scale:
            logger.warning("DomainWallsAnalysisDialog cannot be used: "
                           "Active FFT was not calculated with 'Power' scaling.")
            
            QMessageBox.warning(
                self, 
                "Incorrect FFT Scaling", 
                "Domain wall intensity analysis requires the FFT to be calculated "
                "with the **'Power'** scaling mode (|F|²).\n\n"
                "Please go back, recalculate the FFT with the correct setting, and try again."
            )
            
            # Zaplanuj zamknięcie dialogu natychmiast po tym, jak pętla zdarzeń go przetworzy.
            # To jest bezpieczny sposób na zamknięcie dialogu z wnętrza jego konstruktora.
            QTimer.singleShot(0, self.reject)
            return # Zakończ inicjalizację, aby nie tworzyć reszty UI niepotrzebnie.

        if not PYQTGRAPH_AVAILABLE: # pragma: no cover
            QVBoxLayout(self).addWidget(QLabel("Critical Error: PyQtGraph is required...")); self.setWindowTitle("Error"); return

        self.setWindowTitle("Domain Wall Analysis")
        self.setMinimumSize(1200, 700)

        # Inicjalizacja atrybutów dla danych i UI
        self._selection_mode: Optional[str] = None
        self.main_peak_raw_refined_px: Optional[Tuple[float, float]] = None
        self.satellite_peaks_raw_refined_px: List[Tuple[float, float]] = []
        self.main_peak_raw_marker: Optional[ScatterPlotItem] = None
        self.satellite_raw_markers: Optional[ScatterPlotItem] = None
        self.main_peak_corrected_marker: Optional[ScatterPlotItem] = None
        self.satellite_corrected_markers: Optional[ScatterPlotItem] = None

        self._init_ui()
        self._connect_signals()
        
        self.refinement_roi_size_spinbox.setValue(self.refinement_roi_size)
        self._display_substrate_transform_info() # Placeholder
        self._update_all_ui_elements() # Placeholder

        logger.debug("DomainWallsAnalysisDialog initialized.")

    def _init_ui(self):
        top_level_layout = QHBoxLayout(self)
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        top_level_layout.addWidget(main_splitter)

        # === LEWY PANEL: Kontrolki ===
        left_controls_widget = QWidget()
        left_controls_layout = QVBoxLayout(left_controls_widget)
        left_controls_widget.setMinimumWidth(300); left_controls_widget.setMaximumWidth(380)

        # Grupa: Spot Selection
        spot_selection_group = QGroupBox("Spot Selection (Max 2 spots)")
        spot_selection_layout = QFormLayout(spot_selection_group)
        spot_selection_layout.addRow(QLabel("Refinement Method: 2D Gaussian Fit"))
        self.refinement_roi_size_spinbox = QSpinBox()
        self.refinement_roi_size_spinbox.setMinimum(3); self.refinement_roi_size_spinbox.setMaximum(31); self.refinement_roi_size_spinbox.setSingleStep(2)
        spot_selection_layout.addRow("Refinement Area Size (px):", self.refinement_roi_size_spinbox)
        self.add_spot_button = QPushButton("Refine & Add Selected Spot")
        self.add_spot_button.setEnabled(False)
        spot_selection_layout.addRow(self.add_spot_button)
        left_controls_layout.addWidget(spot_selection_group)

        # Grupa: Substrate Transformation Info
        sub_transform_group = QGroupBox("Substrate Transformation (Reference)")
        sub_transform_layout = QFormLayout(sub_transform_group)
        self.dist_sub_transform_info_label_status = QLabel("Status: -")
        self.dist_sub_transform_info_label_rot = QLabel("Sub. Rotation: -")
        self.dist_sub_transform_info_label_scale = QLabel("Sub. Scale (X,Y): -")
        self.dist_sub_transform_info_label_rmse = QLabel("Sub. RMSE (px): -")
        sub_transform_layout.addRow("Status:", self.dist_sub_transform_info_label_status)
        sub_transform_layout.addRow("Rotation:", self.dist_sub_transform_info_label_rot)
        sub_transform_layout.addRow("Scale:", self.dist_sub_transform_info_label_scale)
        sub_transform_layout.addRow("RMSE:", self.dist_sub_transform_info_label_rmse)
        left_controls_layout.addWidget(sub_transform_group)
        
        left_controls_layout.addStretch(1)
        main_splitter.addWidget(left_controls_widget)

        # === CENTRALNY PANEL: Główny obraz FFT ===
        self.fft_plot_widget = GraphicsLayoutWidget()
        self.fft_view_box = self.fft_plot_widget.addViewBox(row=0, col=0, lockAspect=True, invertY=True)
        self.fft_image_item = ImageItem()
        self.fft_view_box.addItem(self.fft_image_item)
        self.fft_view_box.setMenuEnabled(True); self.fft_view_box.setMouseMode(ViewBox.PanMode); self.fft_view_box.setMouseEnabled(x=True,y=True)
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
        roi_2d_container = QWidget(); roi_2d_v_layout = QVBoxLayout(roi_2d_container); roi_2d_h_layout = QHBoxLayout(); roi_2d_h_layout.addWidget(QLabel("ROI 2D Preview:")); self.enable_2d_roi_preview_checkbox = QCheckBox("Enable"); self.enable_2d_roi_preview_checkbox.setChecked(True); roi_2d_h_layout.addWidget(self.enable_2d_roi_preview_checkbox); roi_2d_h_layout.addStretch(); roi_2d_v_layout.addLayout(roi_2d_h_layout); self.roi_preview_2d_widget = GraphicsLayoutWidget(); self.roi_preview_2d_widget.setMinimumHeight(150); self.roi_preview_2d_widget.setMaximumHeight(200); self.roi_preview_2d_plot = self.roi_preview_2d_widget.addViewBox(lockAspect=True, invertY=True); self.roi_preview_2d_image_item = ImageItem(); self.roi_preview_2d_plot.addItem(self.roi_preview_2d_image_item); roi_2d_v_layout.addWidget(self.roi_preview_2d_widget, 1); preview_grid_layout.addWidget(roi_2d_container, 0, 0)
        # 2D Gaussian Fit Preview
        self.gauss_2d_container = QWidget(); gauss_2d_v_layout = QVBoxLayout(self.gauss_2d_container); gauss_2d_h_layout = QHBoxLayout(); gauss_2d_h_layout.addWidget(QLabel("Gaussian Fit 2D Preview:")); self.enable_gauss_2d_preview_checkbox = QCheckBox("Enable"); self.enable_gauss_2d_preview_checkbox.setChecked(True); gauss_2d_h_layout.addWidget(self.enable_gauss_2d_preview_checkbox); gauss_2d_h_layout.addStretch(); gauss_2d_v_layout.addLayout(gauss_2d_h_layout); self.gaussian_preview_2d_widget = GraphicsLayoutWidget(); self.gaussian_preview_2d_widget.setMinimumHeight(150); self.gaussian_preview_2d_widget.setMaximumHeight(200); self.gaussian_preview_2d_plot = self.gaussian_preview_2d_widget.addViewBox(lockAspect=True, invertY=True); self.gaussian_preview_2d_image_item = ImageItem(); self.gaussian_preview_2d_plot.addItem(self.gaussian_preview_2d_image_item); gauss_2d_v_layout.addWidget(self.gaussian_preview_2d_widget, 1); preview_grid_layout.addWidget(self.gauss_2d_container, 0, 1)
        right_panel_layout.addWidget(preview_group)

        spots_dist_group = QGroupBox("Selected Spots (Raw & Corrected)")
        spots_dist_layout = QVBoxLayout(spots_dist_group)
        self.spots_list_widget = QListWidget(); self.spots_list_widget.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        spots_dist_layout.addWidget(self.spots_list_widget)
        spot_buttons_layout = QHBoxLayout(); self.remove_spot_button = QPushButton("Remove Selected Spot"); self.clear_all_spots_button = QPushButton("Clear All Spots"); spot_buttons_layout.addWidget(self.remove_spot_button); spot_buttons_layout.addWidget(self.clear_all_spots_button); spots_dist_layout.addLayout(spot_buttons_layout)
        right_panel_layout.addWidget(spots_dist_group)
        
        results_group = QGroupBox("Calculated Results"); results_layout = QFormLayout(results_group)
        self.calculate_distance_button = QPushButton("Calculate Distance")
        self.calculate_distance_button.setEnabled(False)
        results_layout.addRow(self.calculate_distance_button)
        self.distance_fft_label = QLabel("-")
        self.distance_real_space_label = QLabel("-")
        self.intensity_ratio_label = QLabel("-")
        results_layout.addRow("Distance in k-space (Δg*):", self.distance_fft_label)
        results_layout.addRow("Real Space Periodicity (P):", self.distance_real_space_label)
        results_layout.addRow("Intensity Ratio (Sat/Main):", self.intensity_ratio_label) # <<< NOWY ELEMENT
        right_panel_layout.addWidget(results_group)

        self.status_label = QLabel("Click on FFT to select a spot."); right_panel_layout.addWidget(self.status_label)
        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close); right_panel_layout.addWidget(self.button_box)
        right_panel_layout.addStretch(1); main_splitter.addWidget(right_panel_widget)
        
        main_splitter.setSizes([350,550,300]); main_splitter.setStretchFactor(1,1)


    def _connect_signals(self):
        """Metoda do podłączenia sygnałów. Zostanie zaimplementowana w kolejnym kroku."""
        pass # TODO

    def _update_all_ui_elements(self):
        """Metoda do aktualizacji całego UI. Zostanie zaimplementowana w kolejnym kroku."""
        pass # TODO

    def _display_substrate_transform_info(self):
        """Metoda do wyświetlania informacji o transformacji. Zostanie zaimplementowana w kolejnym kroku."""
        pass # TODO