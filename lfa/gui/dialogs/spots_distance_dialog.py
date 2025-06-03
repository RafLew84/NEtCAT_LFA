# lfa/gui/dialogs/spot_distance_dialog.py
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
    from pyqtgraph.opengl import GLViewWidget, GLSurfacePlotItem
    PYQTGRAPH_AVAILABLE = True
except ImportError: # pragma: no cover
    pg = None; GraphicsLayoutWidget = None; ImageItem = None; ViewBox = None; RectROI = None; ScatterPlotItem = None; PYQTGRAPH_AVAILABLE = False
    logging.error("SpotDistanceDialog: PyQtGraph not found.")

# Importy z projektu (dostosuj ścieżki, jeśli są inne)
try:
    from ...logic.app_controller import AppController # Dla type hinting
except ImportError as e: # pragma: no cover
    AppController = None
    logging.error(f"SpotDistanceDialog: Error importing project modules: {e}")

try:
    from ...analysis.peak_fitting import find_max_pixel_in_roi, fit_2d_gaussian_in_roi, _gaussian_2d, SCIPY_AVAILABLE
    # Zakładamy, że convert_g_vector_px_to_nm_inv i calculate_real_space_vectors_from_g są w lattice.py
    from ...analysis.lattice import convert_g_vector_px_to_nm_inv, calculate_real_space_vectors_from_g 
    from ...core.history import HistoryNode # Potrzebne do pobrania Lx, Ly
    from ...logic.history_manager import HistoryManager
    PEAK_FITTING_MODULE_AVAILABLE = True
except ImportError as e: # pragma: no cover
    PEAK_FITTING_MODULE_AVAILABLE = False; SCIPY_AVAILABLE = False; convert_g_vector_px_to_nm_inv = None; calculate_real_space_vectors_from_g = None; HistoryNode = None; HistoryManager = None; logging.error(f"SpotDistanceDialog: Error importing project modules: {e}")
    def find_max_pixel_in_roi(data, center, radius): return center
    def fit_2d_gaussian_in_roi(data, center, radius): return None # type: ignore
    def _gaussian_2d(*args, **kwargs): raise ImportError("Gaussian 2D function is not available")


try:
    from scipy.optimize import curve_fit as scipy_curve_fit
    SCIPY_OPTIMIZE_AVAILABLE = True
except ImportError: # pragma: no cover
    SCIPY_OPTIMIZE_AVAILABLE = False; logging.error("SpotDistanceDialog: SciPy not found.")
    def scipy_curve_fit(*args, **kwargs): raise ImportError("scipy.optimize.curve_fit is not available")



logger = logging.getLogger(__name__)

REFINEMENT_DIRECT_CLICK = "Direct Click"
REFINEMENT_MAX_PIXEL = "Max Pixel"
REFINEMENT_GAUSSIAN_FIT = "2D Gaussian Fit"

class SpotDistanceDialog(QDialog):
    def __init__(self,
                 fft_image_data: Optional[np.ndarray],
                 history_manager: HistoryManager,
                 current_fft_node_id: str,
                 default_refinement_method: str = REFINEMENT_DIRECT_CLICK,
                 default_refinement_roi_size: int = 5,
                 # Informacje o transformacji substratu (z AppController)
                 substrate_F_m2i: Optional[np.ndarray] = None,
                 substrate_t_m2i: Optional[np.ndarray] = None,
                 substrate_transform_analysis: Optional[Dict[str, Any]] = None,
                 parent=None):
        super().__init__(parent)

        self.fft_data = fft_image_data
        self.history_manager = history_manager
        self.current_fft_node_id = current_fft_node_id
        self.current_refinement_method = default_refinement_method
        self.refinement_roi_size = default_refinement_roi_size
        self.sub_F_m2i = substrate_F_m2i
        self.sub_t_m2i = substrate_t_m2i
        self.sub_transform_analysis = substrate_transform_analysis

        if not PYQTGRAPH_AVAILABLE: # pragma: no cover
            QVBoxLayout(self).addWidget(QLabel("Critical Error: PyQtGraph is required for this dialog."))
            self.setWindowTitle("Error"); return

        self.setWindowTitle("Calculate Spot Distances in Real Space")
        self.setMinimumSize(1200, 700)

        # Listy do przechowywania danych
        self.selected_spots_fft_px: List[Tuple[float, float]] = [] # Surowe (uściślone) piki w FFT
        self.calculated_real_space_distances_nm: List[Optional[float]] = [] # Obliczone odległości

        # Atrybuty dla UI podglądów (jak w innych dialogach)
        self.last_preview_gauss_fit_popt: Optional[np.ndarray] = None
        self.last_preview_gauss_fit_center_abs: Optional[Tuple[float, float]] = None
        self.last_preview_gauss_roi_state: Optional[Dict] = None
        self.spot_markers: Optional[ScatterPlotItem] = None
        # ... (ewentualne placeholdery dla GL widgetów, jeśli będą)

        self._init_ui()
        self._connect_signals()
        self._update_spot_distance_list_widget() # Wstępna aktualizacja listy

        if self.current_refinement_method == REFINEMENT_MAX_PIXEL: self.rb_refine_max_pixel.setChecked(True)
        elif self.current_refinement_method == REFINEMENT_GAUSSIAN_FIT: self.rb_refine_gaussian.setChecked(True)
        else: self.rb_refine_direct.setChecked(True)
        self.refinement_roi_size_spinbox.setValue(self.refinement_roi_size)
        self._on_refinement_method_changed() # Ustaw widoczność kontrolek
        self._display_substrate_transform_info() # Wyświetl info o transformacji

        logger.debug("SpotDistanceDialog initialized.")

    def _init_ui(self):
        top_level_layout = QHBoxLayout(self)
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        top_level_layout.addWidget(main_splitter)

        # === LEWY PANEL: Kontrolki ===
        left_controls_widget = QWidget()
        left_controls_layout = QVBoxLayout(left_controls_widget)
        left_controls_widget.setMinimumWidth(300); left_controls_widget.setMaximumWidth(350)

        refinement_group = QGroupBox("Spot Refinement")
        refinement_layout = QFormLayout(refinement_group)
        self.rb_refine_direct = QRadioButton(REFINEMENT_DIRECT_CLICK); self.rb_refine_direct.setChecked(True)
        self.rb_refine_max_pixel = QRadioButton(REFINEMENT_MAX_PIXEL); self.rb_refine_gaussian = QRadioButton(REFINEMENT_GAUSSIAN_FIT)
        refinement_layout.addRow(self.rb_refine_direct); refinement_layout.addRow(self.rb_refine_max_pixel); refinement_layout.addRow(self.rb_refine_gaussian)
        self.refinement_roi_size_spinbox = QSpinBox()
        self.refinement_roi_size_spinbox.setMinimum(3); self.refinement_roi_size_spinbox.setMaximum(31); self.refinement_roi_size_spinbox.setSingleStep(2); self.refinement_roi_size_spinbox.setValue(self.refinement_roi_size)
        refinement_layout.addRow("Refinement Area Size (px):", self.refinement_roi_size_spinbox)
        left_controls_layout.addWidget(refinement_group)

        self.add_spot_to_list_button = QPushButton("Add Spot to List from ROI")
        self.add_spot_to_list_button.setEnabled(False)
        left_controls_layout.addWidget(self.add_spot_to_list_button)
        
        sub_transform_group = QGroupBox("Substrate Transformation Info (Applied)")
        sub_transform_layout = QFormLayout(sub_transform_group)
        self.dist_sub_transform_info_label_status = QLabel("Status: -")
        self.dist_sub_transform_info_label_rot = QLabel("Sub. Rotation: -")
        self.dist_sub_transform_info_label_scale = QLabel("Sub. Scale (X,Y): -")
        self.dist_sub_transform_info_label_rmse = QLabel("Sub. RMSE (px): -")
        sub_transform_layout.addRow(self.dist_sub_transform_info_label_status)
        sub_transform_layout.addRow(self.dist_sub_transform_info_label_rot)
        sub_transform_layout.addRow(self.dist_sub_transform_info_label_scale)
        sub_transform_layout.addRow(self.dist_sub_transform_info_label_rmse)
        left_controls_layout.addWidget(sub_transform_group)
        
        left_controls_layout.addStretch(1)
        main_splitter.addWidget(left_controls_widget)

        # === CENTRALNY PANEL: Główny obraz FFT ===
        self.fft_plot_widget = GraphicsLayoutWidget() # Inna nazwa, aby uniknąć konfliktu, jeśli dialog jest dzieckiem innego
        self.fft_view_box = self.fft_plot_widget.addViewBox(row=0, col=0, lockAspect=True, invertY=True)
        self.fft_image_item = ImageItem()
        self.fft_view_box.addItem(self.fft_image_item)
        self.fft_view_box.setMenuEnabled(True); self.fft_view_box.setMouseMode(ViewBox.PanMode)
        if self.fft_data is not None: self.fft_image_item.setImage(self.fft_data.T)
        self.selection_roi = RectROI(pos=(0,0), size=(self.refinement_roi_size, self.refinement_roi_size), pen=pg.mkPen('orange', width=2), translateSnap=True, scaleSnap=True, movable=True, resizable=True, rotatable=False)
        self.fft_view_box.addItem(self.selection_roi); self.selection_roi.setVisible(False)
        main_splitter.addWidget(self.fft_plot_widget)

        # === PRAWY PANEL: Podglądy i Wyniki ===
        right_panel_widget = QWidget(); right_panel_layout = QVBoxLayout(right_panel_widget)
        right_panel_widget.setMinimumWidth(450); right_panel_widget.setMaximumWidth(550)

        preview_group = QGroupBox("Live Previews (Spot Refinement)")
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

        preview_grid_layout.setColumnStretch(0,1)
        preview_grid_layout.setColumnStretch(1,1)
        preview_grid_layout.setRowStretch(0,1)
        preview_grid_layout.setRowStretch(1,1)
        self.gauss_2d_container = gauss_2d_container
        self.gauss_2d_container.setVisible(False)
        right_panel_layout.addWidget(preview_group)

        spots_group = QGroupBox("Selected Spots & Real Space Distances")
        spots_layout = QVBoxLayout(spots_group)
        self.spots_list_widget = QListWidget()
        self.spots_list_widget.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        spots_layout.addWidget(self.spots_list_widget)
        spot_buttons_layout = QHBoxLayout()
        self.remove_spot_button = QPushButton("Remove Selected Spot")
        self.clear_all_spots_button = QPushButton("Clear All Spots")
        spot_buttons_layout.addWidget(self.remove_spot_button)
        spot_buttons_layout.addWidget(self.clear_all_spots_button)
        spots_layout.addLayout(spot_buttons_layout)
        right_panel_layout.addWidget(spots_group)
        
        self.status_label = QLabel("Click on FFT to select spots for distance calculation.")
        right_panel_layout.addWidget(self.status_label)
        
        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close) # Tylko przycisk Close
        right_panel_layout.addWidget(self.button_box)
        
        right_panel_layout.addStretch(1)
        main_splitter.addWidget(right_panel_widget)

        main_splitter.setSizes([320, 500, 380])
        main_splitter.setStretchFactor(1, 1) # Środkowy panel (obraz FFT) może się bardziej rozciągać

    # Poniżej szkielety metod, które będą potrzebne:
    def _connect_signals(self):
        self.button_box.clicked.connect(self.accept) # Przycisk Close zamyka dialog

        # Interakcja z głównym obrazem FFT
        if self.fft_view_box and self.fft_view_box.scene():
            self.fft_view_box.scene().sigMouseClicked.connect(self._handle_fft_image_click)
        self.selection_roi.sigRegionChanged.connect(self._handle_roi_region_changing)

        # Kontrolki metody uściślania
        self.rb_refine_direct.toggled.connect(self._on_refinement_method_changed)
        self.rb_refine_max_pixel.toggled.connect(self._on_refinement_method_changed)
        self.rb_refine_gaussian.toggled.connect(self._on_refinement_method_changed)
        self.refinement_roi_size_spinbox.valueChanged.connect(self._on_refinement_roi_size_changed)

        # Przycisk dodawania piku
        self.add_spot_to_list_button.clicked.connect(self._add_spot_to_list)

        # Checkboxy podglądów Live
        self.enable_2d_roi_preview_checkbox.stateChanged.connect(self._update_roi_previews)
        self.enable_gauss_2d_preview_checkbox.stateChanged.connect(self._update_roi_previews)
        
        # Przyciski zarządzania listą
        self.remove_spot_button.clicked.connect(self._remove_selected_spot_from_list)
        self.clear_all_spots_button.clicked.connect(self._clear_all_listed_spots)
        logger.debug("SpotDistanceDialog signals connected.")

    def _clear_last_preview_gauss_fit(self): # Może być współdzielona lub skopiowana
        self.last_preview_gauss_fit_popt = None
        self.last_preview_gauss_fit_center_abs = None
        self.last_preview_gauss_roi_state = None

    def _update_spot_distance_list_widget(self): # TODO
        self.spots_list_widget.clear()
        if not self.selected_spots_fft_px:
            self.spots_list_widget.addItem("No spots selected yet.")
            return
        for i, spot_fft_px in enumerate(self.selected_spots_fft_px):
            distance_nm_str = "-"
            if i < len(self.calculated_real_space_distances_nm) and self.calculated_real_space_distances_nm[i] is not None:
                distance_nm_str = f"{self.calculated_real_space_distances_nm[i]:.3f} nm" # type: ignore
            
            self.spots_list_widget.addItem(
                f"Spot {i+1}: FFT (kx={spot_fft_px[0]:.1f}, ky={spot_fft_px[1]:.1f} px) | Real Dist: {distance_nm_str}"
            )
        logger.debug("Updated spot distance list widget.")

    def _display_substrate_transform_info(self): # TODO
        if self.sub_transform_analysis:
            self.dist_sub_transform_info_label_status.setText("Status: Substrate transform available.")
            self.dist_sub_transform_info_label_rot.setText(f"{self.sub_transform_analysis.get('rotation_angle_deg', 'N/A'):.2f}°")
            s_x,s_y = self.sub_transform_analysis.get('principal_stretches',[np.nan,np.nan])
            self.dist_sub_transform_info_label_scale.setText(f"({s_x:.3f}, {s_y:.3f})")
            self.dist_sub_transform_info_label_rmse.setText(f"{self.sub_transform_analysis.get('rmse', 'N/A'):.3f} px")
        else:
            self.dist_sub_transform_info_label_status.setText("Status: Substrate transform NOT passed.")
            self.dist_sub_transform_info_label_rot.setText("-"); self.dist_sub_transform_info_label_scale.setText("-"); self.dist_sub_transform_info_label_rmse.setText("-")

    @pyqtSlot(object)
    def _handle_roi_region_changing(self, roi_item: Optional[pg.ROI] = None):
        if roi_item is None: roi_item = self.selection_roi
        if not isinstance(roi_item, RectROI): return

        if roi_item.isVisible():
            roi_pos = roi_item.pos(); roi_size = roi_item.size()
            current_roi_w = int(round(roi_size.x()))
            if current_roi_w != self.refinement_roi_size_spinbox.value() and \
               self.refinement_roi_size_spinbox.minimum() <= current_roi_w <= self.refinement_roi_size_spinbox.maximum() and \
               current_roi_w % 2 != 0 :
                self.refinement_roi_size_spinbox.blockSignals(True)
                self.refinement_roi_size_spinbox.setValue(current_roi_w)
                self.refinement_roi_size_spinbox.blockSignals(False)
            
            self._clear_last_preview_gauss_fit()
            self._update_roi_previews()

    @pyqtSlot(object)
    def _handle_fft_image_click(self, event):
        """Obsługuje kliknięcie na głównym obrazie FFT w tym dialogu."""
        # if not self.fft_data or not self.fft_image_item or not self.selection_roi : 
        #     return # pragma: no cover

        if event.button() == Qt.MouseButton.LeftButton:
            pos_viewbox = self.fft_view_box.mapSceneToView(event.scenePos())
            mapped_pos = self.fft_image_item.mapToData(pos_viewbox)

            if mapped_pos is not None:
                kx, ky = int(round(mapped_pos.x())), int(round(mapped_pos.y()))
                logger.debug(f"SpotDistanceDialog FFT click: mapped to data (kx, ky) = ({kx}, {ky})")

                if self.current_refinement_method == REFINEMENT_DIRECT_CLICK:
                    # Dla Direct Click, od razu dodaj ten punkt (bez pokazywania ROI do dodania)
                    self._add_refined_spot_to_list((float(kx), float(ky)))
                else:
                    # Dla Max Pixel i Gaussian Fit, umieść/zaktualizuj ROI
                    roi_size = self.refinement_roi_size_spinbox.value()
                    roi_x = kx - roi_size // 2
                    roi_y = ky - roi_size // 2
                    
                    max_h, max_w = self.fft_data.shape # Użyj self.fft_data
                    roi_x = np.clip(roi_x, 0, max_w - roi_size)
                    roi_y = np.clip(roi_y, 0, max_h - roi_size)

                    self.selection_roi.setPos((roi_x, roi_y), update=False)
                    self.selection_roi.setSize((roi_size, roi_size), update=False)
                    self.selection_roi.setVisible(True)
                    self.add_spot_to_list_button.setEnabled(True)
                    self._update_roi_previews() # Zaktualizuj podglądy
            event.accept()
        else: # pragma: no cover
            event.ignore()

    def _update_roi_previews(self): 
        if not self.selection_roi.isVisible() or self.fft_data is None:
            self._clear_last_preview_gauss_fit()
            if hasattr(self, 'roi_preview_2d_image_item'): self.roi_preview_2d_image_item.clear()
            if hasattr(self, 'gaussian_preview_2d_image_item'): self.gaussian_preview_2d_image_item.clear()
            return

        roi_state = self.selection_roi.getState()
        x0_roi, y0_roi = int(round(roi_state['pos'].x())), int(round(roi_state['pos'].y()))
        width_roi, height_roi = int(round(roi_state['size'].x())), int(round(roi_state['size'].y()))

        max_ky, max_kx = self.fft_data.shape; y0_cl = np.clip(y0_roi, 0, max_ky)
        y1_cl = np.clip(y0_roi + height_roi, 0, max_ky)
        x0_cl = np.clip(x0_roi, 0, max_kx)
        x1_cl = np.clip(x0_roi + width_roi, 0, max_kx)
        if y1_cl <= y0_cl or x1_cl <= x0_cl : 
            logger.warning("Invalid ROI slice for dist preview.")
            return
        roi_patch = self.fft_data[y0_cl:y1_cl, x0_cl:x1_cl]
        if roi_patch.size > 0:
            if self.enable_2d_roi_preview_checkbox.isChecked(): self.roi_preview_2d_image_item.setImage(roi_patch.T); self.roi_preview_2d_plot.autoRange()
            else: self.roi_preview_2d_image_item.clear()
            if self.rb_refine_gaussian.isChecked():
                fitted_gauss_2d = None
                if PEAK_FITTING_MODULE_AVAILABLE and SCIPY_OPTIMIZE_AVAILABLE and SCIPY_AVAILABLE:
                    ph, pw = roi_patch.shape; py_g, px_g = np.mgrid[0:ph,0:pw]; pxy_flat_g = (py_g.flatten(),px_g.flatten()); pdata_flat_g=roi_patch.flatten()
                    try:
                        p0g=[roi_patch.max()-roi_patch.min(),ph/2.,pw/2.,pw/4.,ph/4.,0.,roi_patch.min()]
                        if callable(scipy_curve_fit) and callable(_gaussian_2d):
                            popt_g,pcov_g = scipy_curve_fit(_gaussian_2d,pxy_flat_g,pdata_flat_g,p0=p0g,maxfev=3000)
                            self.last_preview_gauss_fit_popt=popt_g; afk_g=x0_roi+popt_g[2]; afky_g=y0_roi+popt_g[1]; self.last_preview_gauss_fit_center_abs=(afk_g,afky_g); self.last_preview_gauss_roi_state=roi_state.copy(); logger.info(f"DistDlg Preview GaussFit OK. Center:{self.last_preview_gauss_fit_center_abs}")
                            fitted_gauss_flat=_gaussian_2d(pxy_flat_g,*popt_g);fitted_gauss_2d=fitted_gauss_flat.reshape(ph,pw)
                    except Exception as e_gf_prev: logger.warning(f"DistDlg GaussFit Preview Fail: {e_gf_prev}"); self._clear_last_preview_gauss_fit(); fitted_gauss_2d=roi_patch
                if self.enable_gauss_2d_preview_checkbox.isChecked():
                    if fitted_gauss_2d is not None: self.gaussian_preview_2d_image_item.setImage(fitted_gauss_2d.T)
                    else: self.gaussian_preview_2d_image_item.setImage(roi_patch.T)
                    self.gaussian_preview_2d_plot.autoRange()
                else: self.gaussian_preview_2d_image_item.clear()
            else: self._clear_last_preview_gauss_fit();self.gaussian_preview_2d_image_item.clear()

    @pyqtSlot()
    def _on_refinement_method_changed(self): 
        is_gaussian_mode = self.rb_refine_gaussian.isChecked()
        print(f"is_gaussian_mode: {is_gaussian_mode}")
        if hasattr(self, 'gauss_2d_container'): self.gauss_2d_container.setVisible(is_gaussian_mode)
        if self.rb_refine_direct.isChecked(): 
            self.current_refinement_method=REFINEMENT_DIRECT_CLICK
            self.refinement_roi_size_spinbox.setEnabled(False)
            self.selection_roi.setVisible(False)
            self.add_spot_to_list_button.setEnabled(False)
            self.status_label.setText("Click on FFT to select spot for distance.")
        else: 
            self.current_refinement_method=REFINEMENT_MAX_PIXEL if self.rb_refine_max_pixel.isChecked() else REFINEMENT_GAUSSIAN_FIT
            self.refinement_roi_size_spinbox.setEnabled(True)
            self.add_spot_to_list_button.setEnabled(self.selection_roi.isVisible())
            self.status_label.setText("Place ROI, then click 'Add Spot to List'.")
        if not is_gaussian_mode: 
            self._clear_last_preview_gauss_fit()
        self._update_roi_previews()
        logger.debug(f"SpotDistanceDialog Refinement method: {self.current_refinement_method}")


    
    
    
    @pyqtSlot(int)
    def _on_refinement_roi_size_changed(self, value): 
        self.refinement_roi_size = value; self._clear_last_preview_gauss_fit()
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
        logger.debug(f"SpotDistanceDialog Refinement ROI size: {self.refinement_roi_size}")
    
    
    
    
    @pyqtSlot()
    def _add_spot_to_list(self):
        if not self.selection_roi.isVisible() or self.fft_data is None: 
            self.status_label.setText("No ROI selected.")
            return
        
        roi_state=self.selection_roi.getState()
        x0r,y0r=int(round(roi_state['pos'].x())),int(round(roi_state['pos'].y()))
        wr,hr=int(round(roi_state['size'].x())),int(round(roi_state['size'].y()))
        ckx_roi,cky_roi=x0r+wr//2,y0r+hr//2
        refined_kx_fft, refined_ky_fft = float(ckx_roi), float(cky_roi)

        if self.current_refinement_method == REFINEMENT_MAX_PIXEL and PEAK_FITTING_MODULE_AVAILABLE:
            pr=self.refinement_roi_size//2
            mh,mw=self.fft_data.shape
            eff_cky=np.clip(cky_roi,pr,mh-1-pr)
            eff_ckx=np.clip(ckx_roi,pr,mw-1-pr)
            fky,fkx=find_max_pixel_in_roi(self.fft_data,(eff_cky,eff_ckx),pr)
            refined_kx_fft,refined_ky_fft=float(fkx),float(fky)
        elif self.current_refinement_method == REFINEMENT_GAUSSIAN_FIT and PEAK_FITTING_MODULE_AVAILABLE and SCIPY_AVAILABLE:
            curr_roi_state=self.selection_roi.getState()
            roi_match = False
            if self.last_preview_gauss_roi_state and curr_roi_state and self.last_preview_gauss_roi_state['pos']==curr_roi_state['pos'] and self.last_preview_gauss_roi_state['size']==curr_roi_state['size']: 
                roi_match=True
            if self.last_preview_gauss_fit_center_abs and roi_match: 
                refined_kx_fft,refined_ky_fft=self.last_preview_gauss_fit_center_abs
                logger.info(f"DistDlg: Using PREVIEW GaussFit: ({refined_kx_fft:.2f},{refined_ky_fft:.2f})")
            else:
                pr=self.refinement_roi_size//2
                mh,mw=self.fft_data.shape
                eff_cky=np.clip(cky_roi,pr,mh-1-pr)
                eff_ckx=np.clip(ckx_roi,pr,mw-1-pr)
                fit_res=fit_2d_gaussian_in_roi(self.fft_data,(eff_cky,eff_ckx),pr)
                if fit_res:
                    _popt,(fky_abs,fkx_abs),_patch=fit_res
                    refined_kx_fft,refined_ky_fft=float(fkx_abs),float(fky_abs)
                    logger.info(f"DistDlg: NEW GaussFit: ({refined_kx_fft:.2f},{refined_ky_fft:.2f})")
                else:logger.warning("DistDlg: GaussFit FAILED for Add Spot. Using ROI center.")
        
        new_spot_fft_px = (refined_kx_fft, refined_ky_fft)
        if new_spot_fft_px not in self.selected_spots_fft_px:
            self.selected_spots_fft_px.append(new_spot_fft_px)
            # Oblicz odległość w przestrzeni rzeczywistej
            real_distance = self._calculate_real_space_distance_for_spot(new_spot_fft_px)
            self.calculated_real_space_distances_nm.append(real_distance)
            self._update_spot_distance_list_widget()
            self._redraw_spot_markers() # Narysuj nowy spot
            self.status_label.setText(f"Spot {len(self.selected_spots_fft_px)} added. Dist: {f'{real_distance:.3f} nm' if real_distance is not None else 'Error'}")
        else: self.status_label.setText("Spot already selected.") # pragma: no cover
        self._clear_last_preview_gauss_fit()
        # self._update_add_spot_button_state()

    def _update_spot_distance_list_widget(self):
        self.spots_list_widget.clear()
        if not self.selected_spots_fft_px:
            self.spots_list_widget.addItem("No spots selected yet.")
            return
        for i, spot_fft_px in enumerate(self.selected_spots_fft_px):
            distance_nm_str = "-"
            # Upewnij się, że lista calculated_real_space_distances_nm ma odpowiednią długość
            if i < len(self.calculated_real_space_distances_nm) and \
               self.calculated_real_space_distances_nm[i] is not None:
                distance_nm_str = f"{self.calculated_real_space_distances_nm[i]:.3f} nm"
            
            self.spots_list_widget.addItem(
                f"Spot {i+1}: FFT (kx={spot_fft_px[0]:.1f}, ky={spot_fft_px[1]:.1f} px) | Real Dist: {distance_nm_str}"
            )
    
    @pyqtSlot()
    def _remove_selected_spot_from_list(self):
        current_row = self.spots_list_widget.currentRow()
        if 0 <= current_row < len(self.selected_spots_fft_px):
            del self.selected_spots_fft_px[current_row]
            if current_row < len(self.calculated_real_space_distances_nm): # Usuń też odpowiadającą odległość
                del self.calculated_real_space_distances_nm[current_row]
            self._update_spot_distance_list_widget()
            self._redraw_spot_markers()
            logger.debug(f"Removed spot at index {current_row} from distance list.")

    @pyqtSlot()
    def _clear_all_listed_spots(self):
        self.selected_spots_fft_px.clear()
        self.calculated_real_space_distances_nm.clear()
        self._update_spot_distance_list_widget()
        self._redraw_spot_markers()
        logger.debug("Cleared all spots from distance list.")

    def _calculate_real_space_distance_for_spot(self, spot_fft_px_abs: Tuple[float, float]) -> Optional[float]:
        """Oblicza odległość spotu od centrum w przestrzeni rzeczywistej (nm)."""
        if self.fft_data is None or self.sub_F_m2i is None or self.sub_t_m2i is None or self.history_manager is None: # pragma: no cover
            logger.warning("Cannot calculate real space distance: missing data (FFT, transform, or history_manager).")
            return None
        
        # 1. Skoryguj pozycję spotu (kx_raw, ky_raw) używając transformacji substratu
        # spot_fft_px_abs to (kx_abs, ky_abs) na obrazie FFT
        spot_np = np.array([spot_fft_px_abs], dtype=float) # Musi być 2D array
        try:
            from ...analysis.drift_correction import apply_affine_transform # Upewnij się o imporcie
            spot_corrected_ideal_px_array = apply_affine_transform(spot_np, self.sub_F_m2i, self.sub_t_m2i)
            if spot_corrected_ideal_px_array is None: raise ValueError("Affine transform failed.")
            spot_corrected_ideal_px = tuple(spot_corrected_ideal_px_array[0]) # (kx_corr_ideal, ky_corr_ideal)
        except Exception as e: # pragma: no cover
            logger.error(f"Error applying substrate transform to spot {spot_fft_px_abs}: {e}")
            return None

        # 2. Konwersja na wektor g* (względem centrum idealnego FFT)
        fft_rows_ky, fft_cols_kx = self.fft_data.shape
        center_kx_ideal_px = fft_cols_kx / 2.0
        center_ky_ideal_px = fft_rows_ky / 2.0
        g_vector_ideal_px = (spot_corrected_ideal_px[0] - center_kx_ideal_px, 
                             spot_corrected_ideal_px[1] - center_ky_ideal_px)

        # 3. Konwersja wektora g* na nm^-1
        root_node = self.history_manager.get_root_node_for_node(self.current_fft_node_id)
        if not (root_node and root_node.operation_name == "Original" and root_node.parameters): # pragma: no cover
            logger.warning("Could not get Original node for Lx/Ly for distance calc."); return None
        Lx_nm = root_node.parameters.get("size_nm_x"); Ly_nm = root_node.parameters.get("size_nm_y")
        if not (Lx_nm and Ly_nm and Lx_nm > 0 and Ly_nm > 0): 
            logger.warning("Invalid Lx/Ly for distance calc.")
            return None # pragma: no cover

        if convert_g_vector_px_to_nm_inv is None: 
            logger.error("convert_g_vector_px_to_nm_inv not available")
            return None # pragma: no cover
        g_vector_nm_inv = convert_g_vector_px_to_nm_inv(g_vector_ideal_px, Lx_nm, Ly_nm, fft_cols_kx, fft_rows_ky)
        if g_vector_nm_inv is None: 
            logger.warning("g-vector to nm^-1 conversion failed.")
            return None # pragma: no cover

        # 4. Oblicz magnitudę |g*| w nm^-1
        g_mag_nm_inv = np.linalg.norm(g_vector_nm_inv)
        if g_mag_nm_inv < 1e-9: # pragma: no cover (spot w centrum)
            return 0.0 
        
        # 5. Oblicz odległość w przestrzeni rzeczywistej d = 1 / |g*| (bez 2pi)
        real_distance_nm = 1.0 / g_mag_nm_inv
        return real_distance_nm
    
    def _redraw_spot_markers(self):
        """Rysuje tylko wybrane spoty (surowe/uściślone) na obrazie FFT tego dialogu."""
        if self.spot_markers:
            try: self.fft_view_box.removeItem(self.spot_markers)
            except RuntimeError: pass
            self.spot_markers = None
        
        if self.selected_spots_fft_px:
            spots_data = [{'pos': spot, 'symbol': 'o', 'size': 10, 
                           'pen': pg.mkPen('orange', width=1.5), 
                           'brush': pg.mkBrush(255,165,0,100)} 
                          for spot in self.selected_spots_fft_px]
            self.spot_markers = ScatterPlotItem(spots=spots_data)
            self.fft_view_box.addItem(self.spot_markers)
    
    # Metody _update_3d_surface_plot i _clear_3d_surface mogą być skopiowane z SubstrateSpotDialog

    def accept(self): super().accept()
    def reject(self): super().reject()
    def closeEvent(self, event):
        # TODO: Czyszczenie widgetów GL, jeśli są tworzone dynamicznie
        super().closeEvent(event)