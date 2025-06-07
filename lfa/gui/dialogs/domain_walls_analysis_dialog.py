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

from ...analysis.drift_correction import apply_affine_transform

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
except ImportError: # pragma: no cover
    AppController = None

try:
    from ...analysis.peak_fitting import find_max_pixel_in_roi, fit_2d_gaussian_in_roi_with_all_data, _gaussian_2d, SCIPY_AVAILABLE
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
    # def fit_2d_gaussian_in_roi(data, center, radius): return None
    def _gaussian_2d(*args, **kwargs): raise ImportError("Gaussian 2D function is not available")

try:
    from scipy.optimize import curve_fit as scipy_curve_fit
    SCIPY_OPTIMIZE_AVAILABLE = True
except ImportError: # pragma: no cover
    logging.error("AdsorbateSpotSelectionDialog: SciPy (for curve_fit) not found.")
    SCIPY_OPTIMIZE_AVAILABLE = False
    def scipy_curve_fit(*args, **kwargs): raise ImportError("scipy.optimize.curve_fit is not available")

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
        self.satellite_peak_raw_refined_px: Optional[Tuple[float, float]] = []
        self.main_peak_raw_marker: Optional[ScatterPlotItem] = None
        self.satellite_raw_marker: Optional[ScatterPlotItem] = None
        self.main_peak_corrected_marker: Optional[ScatterPlotItem] = None
        self.satellite_corrected_marker: Optional[ScatterPlotItem] = None
        self.main_peak_corrected_ideal_px: Optional[Tuple[float, float]] = None
        self.satellite_peak_corrected_ideal_px: Optional[Tuple[float, float]] = None
        self.main_peak_amplitude: Optional[float] = None
        self.satellite_peak_amplitude: Optional[float] = None
        self.main_peak_intensity: Optional[float] = None
        self.satellite_peak_intensity: Optional[float] = None
        self.main_peak_max_value: Optional[float] = None
        self.satellite_peak_max_value: Optional[float] = None

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

        refinement_group = QGroupBox("Spot Selection") # Zmieniono nazwę grupy
        refinement_layout = QFormLayout(refinement_group)
        refinement_layout.addRow(QLabel("Refinement Method: 2D Gaussian Fit"))
        self.refinement_roi_size_spinbox = QSpinBox()
        self.refinement_roi_size_spinbox.setMinimum(3); self.refinement_roi_size_spinbox.setMaximum(31); self.refinement_roi_size_spinbox.setSingleStep(2)
        refinement_layout.addRow("Refinement Area Size (px):", self.refinement_roi_size_spinbox)
        
        # Nowe, oddzielne przyciski
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

        selected_spots_group = QGroupBox("Selected Peaks Information")
        selected_spots_layout = QFormLayout(selected_spots_group)
        self.main_peak_info_label = QLabel("Not Selected"); self.main_peak_info_label.setWordWrap(True)
        self.satellite_peak_info_label = QLabel("Not Selected"); self.satellite_peak_info_label.setWordWrap(True)
        self.clear_all_peaks_button = QPushButton("Clear Both Peaks") # Jeden przycisk do czyszczenia
        selected_spots_layout.addRow("Main Peak:", self.main_peak_info_label)
        selected_spots_layout.addRow("Satellite Peak:", self.satellite_peak_info_label)
        selected_spots_layout.addRow(self.clear_all_peaks_button)
        right_panel_layout.addWidget(selected_spots_group)
        
        results_group = QGroupBox("Calculated Results"); results_layout = QFormLayout(results_group)
        self.calculate_distance_button = QPushButton("Calculate Distance")
        self.calculate_distance_button.setEnabled(False)
        results_layout.addRow(self.calculate_distance_button)
        self.distance_fft_label = QLabel("-")
        self.distance_real_space_label = QLabel("-")
        self.intensity_ratio_label = QLabel("-")
        self.amplitude_ratio_label = QLabel("-")
        self.max_value_label = QLabel("-")
        results_layout.addRow("Distance in k-space (Δg*):", self.distance_fft_label)
        results_layout.addRow("Real Space Periodicity (P):", self.distance_real_space_label)
        results_layout.addRow("Intensity Ratio (Sat/Main):", self.intensity_ratio_label) # <<< NOWY ELEMENT
        results_layout.addRow("Amplitude Ratio (Sat/Main):", self.amplitude_ratio_label)
        results_layout.addRow("Max Value Ratio (Set/Main):", self.max_value_label)
        right_panel_layout.addWidget(results_group)

        self.status_label = QLabel("Click on FFT to select a spot."); right_panel_layout.addWidget(self.status_label)
        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close); right_panel_layout.addWidget(self.button_box)
        right_panel_layout.addStretch(1); main_splitter.addWidget(right_panel_widget)
        
        main_splitter.setSizes([350,550,300]); main_splitter.setStretchFactor(1,1)


    def _connect_signals(self):
        """Podłącza sygnały z UI do slotów."""
        self.button_box.clicked.connect(self.accept)

        if self.fft_view_box and self.fft_view_box.scene():
            self.fft_view_box.scene().sigMouseClicked.connect(self._handle_fft_image_click)
        
        self.selection_roi.sigRegionChanged.connect(self._handle_roi_region_changing)
        self.refinement_roi_size_spinbox.valueChanged.connect(self._on_refinement_roi_size_changed)

        # Użyj poprawnej nazwy przycisku z _init_ui
        self.add_main_spot_button.clicked.connect(self._on_add_main_spot_clicked)
        self.add_satellite_spot_button.clicked.connect(self._on_add_satellite_peak_clicked)

        # Checkboxy podglądów
        self.enable_2d_roi_preview_checkbox.stateChanged.connect(self._update_roi_previews)
        self.enable_gauss_2d_preview_checkbox.stateChanged.connect(self._update_roi_previews)
        
        # Przycisk obliczeń (zostanie zaimplementowany później)
        # self.calculate_distance_button.clicked.connect(self._on_calculate_distance_clicked)

        logger.debug("SpotDistanceDialog signals connected.")

    def _on_add_satellite_peak_clicked(self, event):
        pass

    @pyqtSlot(object)
    def _handle_fft_image_click(self, event):
        """Obsługuje kliknięcie na głównym obrazie FFT w tym dialogu."""
        if not (self.fft_data is not None and self.fft_image_item and self.selection_roi):
            return

        if event.button() == Qt.MouseButton.LeftButton:
            pos_viewbox = self.fft_view_box.mapSceneToView(event.scenePos())
            mapped_pos = self.fft_image_item.mapToData(pos_viewbox)

            if mapped_pos is not None:
                kx, ky = mapped_pos.x(), mapped_pos.y()
                logger.debug(f"SpotDistanceDialog FFT click: data (kx, ky) = ({kx:.1f}, {ky:.1f})")

                # Zawsze umieszczaj ROI po kliknięciu, niezależnie od trybu uściślania
                roi_size = self.refinement_roi_size_spinbox.value()
                roi_x = kx - roi_size // 2
                roi_y = ky - roi_size // 2
                
                max_h, max_w = self.fft_data.shape
                roi_x = np.clip(roi_x, 0, max_w - roi_size)
                roi_y = np.clip(roi_y, 0, max_h - roi_size)

                self.selection_roi.setPos((roi_x, roi_y), update=False)
                self.selection_roi.setSize((roi_size, roi_size), update=False)
                self.selection_roi.setVisible(True)
                
                # Uaktywnij przyciski po umieszczeniu ROI
                self._update_buttons_state()
                self._update_roi_previews()
                event.accept()
        else:
            event.ignore()

    @pyqtSlot(object)
    def _handle_roi_region_changing(self, roi_item: Optional[pg.ROI] = None):
        """Slot wywoływany przy zmianie ROI, aktualizuje podglądy na żywo."""
        if roi_item is None: roi_item = self.selection_roi
        if not isinstance(roi_item, RectROI) or not roi_item.isVisible(): return

        # Zaktualizuj spinbox, jeśli użytkownik ręcznie zmienił rozmiar ROI
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
        """Slot wywoływany przy zmianie wartości w spinboxie rozmiaru ROI."""
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
        """Aktualizuje podglądy 2D ROI i dopasowania Gaussa."""
        if not self.selection_roi.isVisible() or self.fft_data is None:
            if hasattr(self, 'roi_preview_2d_image_item'): self.roi_preview_2d_image_item.clear()
            if hasattr(self, 'gaussian_preview_2d_image_item'): self.gaussian_preview_2d_image_item.clear()
            self._clear_last_preview_gauss_fit()
            return

        roi_state = self.selection_roi.getState()
        x0r, y0r = int(round(roi_state['pos'].x())), int(round(roi_state['pos'].y()))
        wr, hr = int(round(roi_state['size'].x())), int(round(roi_state['size'].y()))
        
        mky, mkx = self.fft_data.shape
        y0c=np.clip(y0r,0,mky); y1c=np.clip(y0r+hr,0,mky)
        x0c=np.clip(x0r,0,mkx); x1c=np.clip(x0r+wr,0,mkx)
        
        if y1c <= y0c or x1c <= x0c:
            self.roi_preview_2d_image_item.clear()
            self.gaussian_preview_2d_image_item.clear()
            return
            
        roi_patch = self.fft_data[y0c:y1c, x0c:x1c]

        if roi_patch.size > 0:
            # Podgląd 2D ROI
            if self.enable_2d_roi_preview_checkbox.isChecked():
                self.roi_preview_2d_image_item.setImage(roi_patch.T)
                self.roi_preview_2d_plot.autoRange()
            else:
                self.roi_preview_2d_image_item.clear()

            # Podgląd Dopasowania Gaussa 2D
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
                            
                            # Zapisz wyniki podglądu do późniejszego użycia
                            self.last_preview_gauss_fit_popt = popt_g
                            afk_g, afky_g = x0r + popt_g[2], y0r + popt_g[1]
                            self.last_preview_gauss_fit_center_abs = (afk_g, afky_g)
                            self.last_preview_gauss_roi_state = roi_state.copy()
                            
                            fitted_gauss_flat = _gaussian_2d(pxy_flat_g, *popt_g)
                            fitted_gauss_2d = fitted_gauss_flat.reshape(ph, pw)
                        except Exception as e:
                            logger.warning(f"DistDlg Preview GaussFit Fail: {e}")
                            self._clear_last_preview_gauss_fit()
                            fitted_gauss_2d = roi_patch # Pokaż oryginał w razie błędu
                    
                    if fitted_gauss_2d is not None:
                        self.gaussian_preview_2d_image_item.setImage(fitted_gauss_2d.T)
                    else:
                        self.gaussian_preview_2d_image_item.setImage(roi_patch.T) # Fallback
                    self.gaussian_preview_2d_plot.autoRange()
                else: # Checkbox odznaczony
                    self.gaussian_preview_2d_image_item.clear()
        else: # roi_patch.size == 0
            self.roi_preview_2d_image_item.clear()
            self.gaussian_preview_2d_image_item.clear()

    def _display_substrate_transform_info(self):
        """
        Wypełnia etykiety w UI informacjami o transformacji substratu,
        które zostały przekazane do dialogu podczas jego tworzenia.
        """
        if self.sub_transform_analysis:
            # Jeśli dane analizy transformacji są dostępne
            self.dist_sub_transform_info_label_status.setText("Status: Available")
            
            # Pobierz i sformatuj kąt rotacji
            rot_angle = self.sub_transform_analysis.get('rotation_angle_deg', 'N/A')
            rot_text = f"{rot_angle:.2f}°" if isinstance(rot_angle, (int, float)) else "N/A"
            self.dist_sub_transform_info_label_rot.setText(rot_text)

            # Pobierz i sformatuj współczynniki rozciągania
            stretches = self.sub_transform_analysis.get('principal_stretches', [np.nan, np.nan])
            if stretches is not None and len(stretches) == 2:
                scale_text = f"({stretches[0]:.3f}, {stretches[1]:.3f})"
            else:
                scale_text = "N/A"
            self.dist_sub_transform_info_label_scale.setText(scale_text)

            # Pobierz i sformatuj RMSE
            rmse = self.sub_transform_analysis.get('rmse', 'N/A')
            rmse_text = f"{rmse:.3f} px" if isinstance(rmse, (int, float)) else "N/A"
            self.dist_sub_transform_info_label_rmse.setText(rmse_text)
            
            logger.info("Displayed available substrate transformation info.")
        else:
            # Jeśli dane analizy transformacji nie zostały przekazane
            self.dist_sub_transform_info_label_status.setText("Status: Not Calculated / Not Available")
            self.dist_sub_transform_info_label_rot.setText("-")
            self.dist_sub_transform_info_label_scale.setText("-")
            self.dist_sub_transform_info_label_rmse.setText("-")
            logger.warning("Substrate transformation info not passed to dialog.")

    def _update_all_ui_elements(self):
        self._update_spot_info_display()
        self._redraw_all_markers_on_fft()
        self._update_buttons_state()
        self._auto_calculate_results()
    
    def _auto_calculate_results(self): pass

    def _update_spot_info_display(self):
        """
        Aktualizuje pola tekstowe, wyświetlając informacje o wybranych pikach.
        """
        # Aktualizacja wyświetlacza piku głównego
        if self.main_peak_raw_refined_px and self.main_peak_corrected_ideal_px and self.main_peak_intensity is not None:
            raw = self.main_peak_raw_refined_px
            corr = self.main_peak_corrected_ideal_px
            intensity = self.main_peak_intensity
            amplitude = self.main_peak_amplitude
            max_value = self.main_peak_max_value
            self.main_peak_info_label.setText(f"Corrected: ({corr[0]:.1f}, {corr[1]:.1f}) px | I: {intensity:.2e} \n A: {amplitude:.2e} | Max: {max_value:.2e}")
        else:
            self.main_peak_info_label.setText("Not Selected")

        # if self.main_peak_data:
        #     # Używamy skorygowanych współrzędnych i obliczonych wartości
        #     corr = self.main_peak_data.get('corrected')
        #     intensity = self.main_peak_data.get('intensity')
        #     amplitude = self.main_peak_data.get('amplitude')
            
        #     corr_text = f"Corr: ({corr[0]:.1f}, {corr[1]:.1f}) px" if corr else "Corr: Error"
        #     intensity_text = f"I: {intensity:.2e}" if intensity is not None else "I: -"
        #     amplitude_text = f"A: {amplitude:.2e}" if amplitude is not None else "A: -"
            
        #     self.main_peak_info_label.setText(f"{corr_text} | {intensity_text} | {amplitude_text}")
        # else:
        #     self.main_peak_info_label.setText("Not Selected")
        
        # # Aktualizacja wyświetlacza piku satelitarnego
        # if self.satellite_peak_data:
        #     corr = self.satellite_peak_data.get('corrected')
        #     intensity = self.satellite_peak_data.get('intensity')
        #     amplitude = self.satellite_peak_data.get('amplitude')

        #     corr_text = f"Corr: ({corr[0]:.1f}, {corr[1]:.1f}) px" if corr else "Corr: Error"
        #     intensity_text = f"I: {intensity:.2e}" if intensity is not None else "I: -"
        #     amplitude_text = f"A: {amplitude:.2e}" if amplitude is not None else "A: -"
            
        #     self.satellite_peak_info_label.setText(f"{corr_text} | {intensity_text} | {amplitude_text}")
        # else:
        #     self.satellite_peak_info_label.setText("Not Selected")

    def _redraw_all_markers_on_fft(self):
        """
        Rysuje markery dla piku głównego i satelitarnego (surowe/uściślone
        oraz ich skorygowane pozycje) na obrazie FFT.
        """
        # Usuń wszystkie stare markery
        if self.main_peak_raw_marker: 
            self.fft_view_box.removeItem(self.main_peak_raw_marker)
            self.main_peak_raw_marker=None
        if self.main_peak_corrected_marker: 
            self.fft_view_box.removeItem(self.main_peak_corrected_marker)
            self.main_peak_corrected_marker=None
        # if self.satellite_raw_marker: self.fft_view_box.removeItem(self.satellite_raw_marker); self.satellite_raw_marker=None
        # if self.satellite_corrected_marker: self.fft_view_box.removeItem(self.satellite_corrected_marker); self.satellite_corrected_marker=None

        # --- Rysuj surowe (uściślone) piki ---
        # Pik główny (np. duży żółty okrąg)
        if self.main_peak_raw_refined_px:
            self.main_peak_raw_marker = pg.ScatterPlotItem(
                spots=[{'pos': self.main_peak_raw_refined_px, 'symbol': 'o', 'size': 14, 'pen': pg.mkPen('y', width=2), 'brush': pg.mkBrush(255, 255, 0, 120)}]
            )
            self.fft_view_box.addItem(self.main_peak_raw_marker)
        
        # Pik satelitarny (np. mniejszy pomarańczowy okrąg)
        # if self.satellite_peak_data and self.satellite_peak_data.get('raw'):
        #     self.satellite_raw_marker = pg.ScatterPlotItem(
        #         spots=[{'pos': self.satellite_peak_data['raw'], 'symbol': 'o', 'size': 10, 'pen': pg.mkPen('orange', width=1.5), 'brush': pg.mkBrush(255, 165, 0, 100)}]
        #     )
        #     self.fft_view_box.addItem(self.satellite_raw_marker)

        # --- Rysuj skorygowane piki (przetransformowane z powrotem do przestrzeni obrazu FFT) ---
        spots_to_transform = []
        if self.main_peak_corrected_ideal_px:
            spots_to_transform.append(self.main_peak_corrected_ideal_px)
        # if self.satellite_peak_data and self.satellite_peak_data.get('corrected'):
        #     spots_to_transform.append(self.satellite_peak_data['corrected'])

        if spots_to_transform and self.sub_F_m2i is not None and apply_affine_transform:
            try:
                F_inv = np.linalg.inv(self.sub_F_m2i)
                t_m2i = self.sub_t_m2i or np.zeros(2)
                t_prime_for_display = (-t_m2i @ F_inv.T).flatten()
                
                transformed_back = apply_affine_transform(np.array(spots_to_transform), F_inv, t_prime_for_display)
                
                if transformed_back is not None:
                    # Skorygowany pik główny (np. duży cyjanowy kwadrat)
                    if self.main_peak_corrected_ideal_px:
                        self.main_peak_corrected_marker = pg.ScatterPlotItem(
                            spots=[{'pos': tuple(transformed_back[0]), 'symbol': 's', 'size': 14, 'pen': pg.mkPen('c', width=2)}]
                        )
                        self.fft_view_box.addItem(self.main_peak_corrected_marker)
                    
                    # # Skorygowany pik satelitarny (np. mniejszy cyjanowy kwadrat)
                    # if self.satellite_peak_data and self.satellite_peak_data.get('corrected'):
                    #     # Indeks będzie 1, jeśli jest pik główny, w przeciwnym razie 0
                    #     start_index = 1 if self.main_peak_data and self.main_peak_data.get('corrected') else 0
                    #     if len(transformed_back) > start_index:
                    #         self.satellite_corrected_marker = pg.ScatterPlotItem(
                    #             spots=[{'pos': tuple(transformed_back[start_index]), 'symbol': 's', 'size': 10, 'pen': pg.mkPen('cyan', width=1.5)}]
                    #         )
                    #         self.fft_view_box.addItem(self.satellite_corrected_marker)
            except Exception as e:
                logger.error(f"Error drawing corrected markers: {e}")

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
    
    def _refine_and_process_spot(self) -> Optional[Tuple[Tuple[float, float], Tuple[float, float], float]]:
        if not self.selection_roi.isVisible() or self.fft_data is None: return None
        
        roi_state=self.selection_roi.getState(); x0r,y0r=int(round(roi_state['pos'].x())),int(round(roi_state['pos'].y())); wr,hr=int(round(roi_state['size'].x())),int(round(roi_state['size'].y())); ckx_roi,cky_roi=x0r+wr//2,y0r+hr//2
        if not (PEAK_FITTING_MODULE_AVAILABLE and fit_2d_gaussian_in_roi_with_all_data and callable(fit_2d_gaussian_in_roi_with_all_data)): return None
        pr=self.refinement_roi_size_spinbox.value()//2; mh,mw=self.fft_data.shape; eff_cky,eff_ckx=np.clip(cky_roi,pr,mh-1-pr),np.clip(ckx_roi,pr,mw-1-pr)
        fit_res=fit_2d_gaussian_in_roi_with_all_data(self.fft_data, (eff_cky,eff_ckx), pr)
        if not fit_res: logger.warning("Gaussian fit failed."); return None
        popt_fit,(fky_abs,fkx_abs),roi_patch_used = fit_res; refined_kx_fft,refined_ky_fft=float(fkx_abs),float(fky_abs);
        
        raw_refined_spot = (refined_kx_fft, refined_ky_fft)
        
        intensity = 0.0
        amplitude,_,_,sigma_y,sigma_x,_,_ = popt_fit
        intensity = 2*np.pi*abs(amplitude)*abs(sigma_x)*abs(sigma_y)
        max_value = np.max(roi_patch_used) if roi_patch_used.size > 0 else 0.0
        
        corrected_spot = None
        if self.sub_F_m2i is not None and self.sub_t_m2i is not None and apply_affine_transform:
            try:
                corrected_array = apply_affine_transform(np.array([raw_refined_spot]), self.sub_F_m2i, self.sub_t_m2i)
                if corrected_array is not None:
                    corrected_spot = tuple(corrected_array[0])
            except Exception as e:
                logger.error(f"Error correcting spot {raw_refined_spot}: {e}")
        
        if corrected_spot is None: logger.warning(f"Could not correct spot {raw_refined_spot}."); return None
        print(f"raw_refined_spot: {raw_refined_spot}")
        print(f"corrected_spot: {corrected_spot}")
        print(f"intensity: {intensity}")
        print(f"amplitude: {amplitude}")
        return raw_refined_spot, corrected_spot, intensity, amplitude, max_value

    @pyqtSlot()
    def _on_add_main_spot_clicked(self):
        if not self.selection_roi.isVisible(): QMessageBox.warning(self,"No ROI","Please place ROI on the main peak first."); return
        results = self._refine_and_process_spot()
        if results:
            raw,corr,intensity,amplitude, max_value = results
            self.main_peak_raw_refined_px=raw
            self.main_peak_corrected_ideal_px=corr
            self.main_peak_intensity=intensity
            self.main_peak_amplitude=amplitude
            self.main_peak_max_value=max_value
            logger.info(f"Main peak selected/updated: Raw={raw}, Corrected={corr}, Intensity={intensity:.2e}, Amplitude={amplitude:.2e}")
            self._update_all_ui_elements()
        self.selection_roi.setVisible(False); self._update_buttons_state()

    def _update_buttons_state(self):
        roi_is_visible = self.selection_roi.isVisible()
        main_peak_exists = self.main_peak_raw_refined_px is not None
        
        self.add_main_spot_button.setEnabled(roi_is_visible)
        self.add_satellite_spot_button.setEnabled(roi_is_visible and main_peak_exists)
        
        # self.clear_main_peak_button.setEnabled(main_peak_exists)
        # self.clear_satellites_button.setEnabled(bool(self.satellite_peaks_raw_refined_px))


    