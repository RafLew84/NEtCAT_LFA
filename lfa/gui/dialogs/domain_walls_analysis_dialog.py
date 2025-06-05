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
    pg = None
    GraphicsLayoutWidget = None
    ImageItem = None
    ViewBox = None
    RectROI = None
    ScatterPlotItem = None
    PYQTGRAPH_AVAILABLE = False
    logging.error("DomainWallsAnalysisDialog: PyQtGraph not found.")

try:
    from scipy.optimize import curve_fit as scipy_curve_fit
    SCIPY_OPTIMIZE_AVAILABLE = True
except ImportError: # pragma: no cover
    SCIPY_OPTIMIZE_AVAILABLE = False
    logging.error("DomainWallsAnalysisDialog: SciPy (for curve_fit) not found.")
    def scipy_curve_fit(*args, **kwargs): raise ImportError("scipy.optimize.curve_fit is not available")

try:
    from ...analysis.peak_fitting import find_max_pixel_in_roi, fit_2d_gaussian_in_roi, _gaussian_2d, SCIPY_AVAILABLE
    from ...analysis.lattice import convert_g_vector_px_to_nm_inv
    from ...core.history import HistoryNode
    from ...logic.history_manager import HistoryManager
    from ...analysis.drift_correction import apply_affine_transform
    PEAK_FITTING_MODULE_AVAILABLE = True
except ImportError as e: # pragma: no cover
    PEAK_FITTING_MODULE_AVAILABLE = False
    SCIPY_AVAILABLE = False
    convert_g_vector_px_to_nm_inv = None
    HistoryNode = None
    HistoryManager = None
    apply_affine_transform = None
    logging.error(f"DomainWallsAnalysisDialog: Error importing project modules: {e}")
    if fit_2d_gaussian_in_roi is None: 
        def fit_2d_gaussian_in_roi(data, center, radius): return None
    if _gaussian_2d is None: 
        def _gaussian_2d(*args, **kwargs): raise ImportError("Gaussian 2D function is not available")

logger = logging.getLogger(__name__)

# W tym dialogu używamy tylko Gaussa, ale zostawmy stałe dla spójności
REFINEMENT_GAUSSIAN_FIT = "2D Gaussian Fit"

class DomainWallsAnalysisDialog(QDialog):
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

        if not PYQTGRAPH_AVAILABLE:
            QVBoxLayout(self).addWidget(QLabel("Critical Error: PyQtGraph is required..."))
            self.setWindowTitle("Error")
            return

        self.setWindowTitle("Analyze Domain Wall Distances")
        self.setMinimumSize(1200, 650)

        self.selected_spots_raw_refined_fft_px: List[Tuple[float, float]] = []
        self.corrected_spots_ideal_px: List[Optional[Tuple[float, float]]] = []
        self.calculated_real_space_distances_nm: List[Optional[float]] = []
        
        self.raw_refined_spot_markers: Optional[ScatterPlotItem] = None
        self.corrected_spot_display_markers: Optional[ScatterPlotItem] = None

        self.last_preview_gauss_fit_popt: Optional[np.ndarray] = None
        self.last_preview_gauss_fit_center_abs: Optional[Tuple[float, float]] = None
        self.last_preview_gauss_roi_state: Optional[Dict] = None

        self._init_ui()
        self._connect_signals()
        self._update_spot_distance_list_widget()
        self._redraw_all_markers_on_fft()
        self._update_buttons_state()
        self._display_substrate_transform_info()
        self.refinement_roi_size_spinbox.setValue(self.refinement_roi_size)
        logger.debug("DomainWallsAnalysisDialog initialized.")

    def _init_ui(self):
        top_level_layout = QHBoxLayout(self)
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        top_level_layout.addWidget(main_splitter)

        # Lewy panel
        left_controls_widget = QWidget()
        left_controls_layout = QVBoxLayout(left_controls_widget)
        left_controls_widget.setMinimumWidth(300)
        left_controls_widget.setMaximumWidth(380)

        refinement_group = QGroupBox("Spot Selection (Max 2 spots)")
        refinement_layout = QFormLayout(refinement_group)
        refinement_layout.addRow(QLabel("Refinement Method: 2D Gaussian Fit"))
        self.refinement_roi_size_spinbox = QSpinBox()
        self.refinement_roi_size_spinbox.setMinimum(3)
        self.refinement_roi_size_spinbox.setMaximum(31)
        self.refinement_roi_size_spinbox.setSingleStep(2)
        self.refinement_roi_size_spinbox.setValue(self.refinement_roi_size)

        refinement_layout.addRow("Refinement Area Size (px):", self.refinement_roi_size_spinbox)
        self.add_spot_button = QPushButton("Refine & Add Selected Spot")
        self.add_spot_button.setEnabled(False)

        refinement_layout.addRow(self.add_spot_button)
        left_controls_layout.addWidget(refinement_group)
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

        # Środkowy panel
        self.fft_plot_widget = GraphicsLayoutWidget()
        self.fft_view_box = self.fft_plot_widget.addViewBox(row=0, col=0, lockAspect=True, invertY=True)
        self.fft_image_item = ImageItem()
        self.fft_view_box.addItem(self.fft_image_item)
        self.fft_view_box.setMenuEnabled(True)
        self.fft_view_box.setMouseMode(ViewBox.PanMode)
        self.fft_view_box.setMouseEnabled(x=True,y=True)
        if self.fft_data is not None: self.fft_image_item.setImage(self.fft_data.T)
        self.selection_roi = RectROI(pos=(0,0), size=(self.refinement_roi_size, self.refinement_roi_size), pen=pg.mkPen('orange', width=2), movable=True, resizable=True)
        self.fft_view_box.addItem(self.selection_roi)
        self.selection_roi.setVisible(False)
        main_splitter.addWidget(self.fft_plot_widget)

        # Prawy panel
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
        self.roi_preview_2d_widget.setMinimumSize(150,150)
        self.roi_preview_2d_widget.setMaximumHeight(200)
        self.roi_preview_2d_plot = self.roi_preview_2d_widget.addViewBox(lockAspect=True, invertY=True)
        self.roi_preview_2d_image_item = ImageItem()
        self.roi_preview_2d_plot.addItem(self.roi_preview_2d_image_item)
        roi_2d_v_layout.addWidget(self.roi_preview_2d_widget, 1)

        preview_grid_layout.addWidget(roi_2d_container, 0, 0)
        gauss_2d_container = QWidget()
        gauss_2d_v_layout = QVBoxLayout(gauss_2d_container)
        gauss_2d_h_layout = QHBoxLayout()
        gauss_2d_h_layout.addWidget(QLabel("Gaussian Fit 2D Preview:"))

        self.enable_gauss_2d_preview_checkbox = QCheckBox("Enable")
        self.enable_gauss_2d_preview_checkbox.setChecked(True)

        gauss_2d_h_layout.addWidget(self.enable_gauss_2d_preview_checkbox)
        gauss_2d_h_layout.addStretch()
        gauss_2d_v_layout.addLayout(gauss_2d_h_layout)
        self.gaussian_preview_2d_widget = GraphicsLayoutWidget()
        self.gaussian_preview_2d_widget.setMinimumSize(150,150)
        self.gaussian_preview_2d_widget.setMaximumHeight(200)
        self.gaussian_preview_2d_plot = self.gaussian_preview_2d_widget.addViewBox(lockAspect=True, invertY=True)
        self.gaussian_preview_2d_image_item = ImageItem()
        self.gaussian_preview_2d_plot.addItem(self.gaussian_preview_2d_image_item)

        gauss_2d_v_layout.addWidget(self.gaussian_preview_2d_widget, 1)
        preview_grid_layout.addWidget(gauss_2d_container, 0, 1)
        right_panel_layout.addWidget(preview_group)
        spots_dist_group = QGroupBox("Selected Spots & Calculated Distances")
        spots_dist_layout = QVBoxLayout(spots_dist_group)

        self.spots_list_widget = QListWidget()
        self.spots_list_widget.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)

        spots_dist_layout.addWidget(self.spots_list_widget)
        spot_buttons_layout = QHBoxLayout()

        self.remove_spot_button = QPushButton("Remove Last Spot")
        self.clear_all_spots_button = QPushButton("Clear All Spots")

        spot_buttons_layout.addWidget(self.remove_spot_button)
        spot_buttons_layout.addWidget(self.clear_all_spots_button)
        spots_dist_layout.addLayout(spot_buttons_layout)
        right_panel_layout.addWidget(spots_dist_group)

        results_group = QGroupBox("Results")
        results_layout = QFormLayout(results_group)
        self.calculate_distance_button = QPushButton("Calculate Distance Between Corrected Spots")
        self.calculate_distance_button.setEnabled(False)
        results_layout.addRow(self.calculate_distance_button)
        self.distance_fft_label = QLabel("Δg* (px): - | (nm⁻¹): -")
        self.distance_real_space_label = QLabel("Periodicity P (nm): -")
        results_layout.addRow("Distance in k-space:", self.distance_fft_label)
        results_layout.addRow("Real Space Periodicity:", self.distance_real_space_label)
        right_panel_layout.addWidget(results_group)
        self.status_label = QLabel("Click on FFT to select spots for analysis.")
        right_panel_layout.addWidget(self.status_label)
        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        right_panel_layout.addWidget(self.button_box)
        right_panel_layout.addStretch(1)
        main_splitter.addWidget(right_panel_widget)
        main_splitter.setSizes([320,550,330])
        main_splitter.setStretchFactor(1,1)

    def _connect_signals(self):
        self.button_box.clicked.connect(self.accept)
        if self.fft_view_box and self.fft_view_box.scene():
            self.fft_view_box.scene().sigMouseClicked.connect(self._handle_fft_image_click)
        self.selection_roi.sigRegionChanged.connect(self._handle_roi_region_changing)
        self.refinement_roi_size_spinbox.valueChanged.connect(self._on_refinement_roi_size_changed)
        self.add_spot_button.clicked.connect(self._add_spot_to_list)
        self.enable_2d_roi_preview_checkbox.stateChanged.connect(self._update_roi_previews)
        self.enable_gauss_2d_preview_checkbox.stateChanged.connect(self._update_roi_previews)
        self.remove_spot_button.clicked.connect(self._remove_last_spot) # Zmieniono na last
        self.clear_all_spots_button.clicked.connect(self._clear_all_listed_spots)
        if hasattr(self, 'calculate_distance_button'):
            self.calculate_distance_button.clicked.connect(self._on_calculate_distance_clicked)
        logger.debug("DomainWallsAnalysisDialog signals connected.")

    def _clear_last_preview_gauss_fit(self):
        self.last_preview_gauss_fit_popt = None
        self.last_preview_gauss_fit_center_abs = None
        self.last_preview_gauss_roi_state = None

    @pyqtSlot(object)
    def _handle_roi_region_changing(self, roi_item: Optional[pg.ROI] = None):
        if roi_item is None: roi_item = self.selection_roi
        if not isinstance(roi_item, RectROI) or not roi_item.isVisible(): return
        self._clear_last_preview_gauss_fit()
        self._update_roi_previews()

    def _update_roi_previews(self):
        if not self.selection_roi.isVisible() or self.fft_data is None:
            self.roi_preview_2d_image_item.clear()
            self.gaussian_preview_2d_image_item.clear()
            return
        
        roi_state=self.selection_roi.getState()
        x0r,y0r=int(round(roi_state['pos'].x())),int(round(roi_state['pos'].y()))
        wr,hr=int(round(roi_state['size'].x())),int(round(roi_state['size'].y()))
        mky,mkx=self.fft_data.shape
        y0c=np.clip(y0r,0,mky)
        y1c=np.clip(y0r+hr,0,mky)
        x0c=np.clip(x0r,0,mkx)
        x1c=np.clip(x0r+wr,0,mkx)

        if y1c<=y0c or x1c<=x0c: 
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
            
            fitted_gauss_2d = None
            if self.enable_gauss_2d_preview_checkbox.isChecked():
                if PEAK_FITTING_MODULE_AVAILABLE and SCIPY_OPTIMIZE_AVAILABLE and callable(_gaussian_2d) and callable(scipy_curve_fit):
                    ph,pw=roi_patch.shape
                    py_g,px_g=np.mgrid[0:ph,0:pw]
                    pxy_flat_g=(py_g.flatten(),px_g.flatten())
                    pdata_flat_g=roi_patch.flatten()
                    try:
                        p0g=[roi_patch.max()-roi_patch.min(),ph/2.,pw/2.,pw/4.,ph/4.,0.,roi_patch.min()]
                        popt_g,_=scipy_curve_fit(_gaussian_2d,pxy_flat_g,pdata_flat_g,p0=p0g,maxfev=2000)
                        self.last_preview_gauss_fit_popt=popt_g
                        afk_g=x0r+popt_g[2]
                        afky_g=y0r+popt_g[1]
                        self.last_preview_gauss_fit_center_abs=(afk_g,afky_g)
                        self.last_preview_gauss_roi_state=roi_state.copy()
                        fitted_gauss_flat=_gaussian_2d(pxy_flat_g,*popt_g)
                        fitted_gauss_2d=fitted_gauss_flat.reshape(ph,pw)
                    except Exception: 
                        self._clear_last_preview_gauss_fit()
                        fitted_gauss_2d=roi_patch
                self.gaussian_preview_2d_image_item.setImage(fitted_gauss_2d.T if fitted_gauss_2d is not None else roi_patch.T)
                self.gaussian_preview_2d_plot.autoRange()
            else: 
                self.gaussian_preview_2d_image_item.clear()
        else: 
            self.roi_preview_2d_image_item.clear()
            self.gaussian_preview_2d_image_item.clear()

    @pyqtSlot()
    def _on_refinement_method_changed(self): # Uproszczone, bo zawsze jest Gauss
        pass # Na razie nic nie robi, bo nie ma wyboru metody

    @pyqtSlot(int)
    def _on_refinement_roi_size_changed(self, value: int):
        self.refinement_roi_size = value
        self._clear_last_preview_gauss_fit()
        if self.selection_roi.isVisible():
            current_pos = self.selection_roi.pos()
            old_size = self.selection_roi.size()
            cx=current_pos.x()+old_size.x()/2
            cy=current_pos.y()+old_size.y()/2
            nx=cx-value/2
            ny=cy-value/2
            self.selection_roi.setPos((nx,ny),update=False)
            self.selection_roi.setSize((value,value),update=False)
            self._handle_roi_region_changing()

    @pyqtSlot(object)
    def _handle_fft_image_click(self, event):
        # if not self.fft_data or not self.fft_image_item or not self.selection_roi : return
        if event.button() == Qt.MouseButton.LeftButton:
            pos_vb = self.fft_view_box.mapSceneToView(event.scenePos())
            mapped_pos = self.fft_image_item.mapToData(pos_vb)
            if mapped_pos:
                kx_abs, ky_abs = mapped_pos.x(), mapped_pos.y()
                roi_s = self.refinement_roi_size_spinbox.value()

                rx_c = kx_abs-roi_s//2
                ry_c = ky_abs-roi_s//2
                mh,mw=self.fft_data.shape
                rx_c=np.clip(rx_c,0,mw-roi_s)
                ry_c=np.clip(ry_c,0,mh-roi_s)

                self.selection_roi.setPos((rx_c, ry_c),update=False)
                self.selection_roi.setSize((roi_s,roi_s),update=False)
                self.selection_roi.setVisible(True)
                self.add_spot_button.setEnabled(True)

                self._update_roi_previews()
            event.accept()
        else: event.ignore()

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

        if PEAK_FITTING_MODULE_AVAILABLE and SCIPY_AVAILABLE and callable(fit_2d_gaussian_in_roi):
            # Użyj zapisanego dopasowania z podglądu, jeśli ROI się nie zmieniło
            curr_roi_state=self.selection_roi.getState()
            roi_match = False
            if self.last_preview_gauss_roi_state and curr_roi_state and self.last_preview_gauss_roi_state['pos']==curr_roi_state['pos'] and self.last_preview_gauss_roi_state['size']==curr_roi_state['size']: 
                roi_match=True
            if self.last_preview_gauss_fit_center_abs and roi_match:
                refined_kx_fft,refined_ky_fft=self.last_preview_gauss_fit_center_abs
                logger.info(f"Using PREVIEW GaussFit: ({refined_kx_fft:.2f},{refined_ky_fft:.2f})")
            else: # Wykonaj nowy, pełny fit
                logger.info("Performing NEW Gaussian fit for Add Spot.")
                pr=self.refinement_roi_size//2
                mh,mw=self.fft_data.shape
                eff_cky=np.clip(cky_roi,pr,mh-1-pr)
                eff_ckx=np.clip(ckx_roi,pr,mw-1-pr)
                fit_res=fit_2d_gaussian_in_roi(self.fft_data,(eff_cky,eff_ckx),pr)
                if fit_res:
                    _popt,(fky_abs,fkx_abs),_patch=fit_res
                    refined_kx_fft,refined_ky_fft=float(fkx_abs),float(fky_abs)
                    logger.info(f"NEW GaussFit: ({refined_kx_fft:.2f},{refined_ky_fft:.2f})")
                else:logger.warning("GaussFit FAILED for Add Spot. Using ROI center.")
        
        self._add_refined_spot_to_list((refined_kx_fft, refined_ky_fft))
        self._clear_last_preview_gauss_fit()

    def _add_refined_spot_to_list(self, refined_spot_fft_px: Tuple[float,float]):
        if len(self.selected_spots_raw_refined_fft_px) >= 2:
            QMessageBox.information(self, "Limit Reached", "A maximum of 2 spots can be selected for distance calculation.")
            return

        if refined_spot_fft_px in self.selected_spots_raw_refined_fft_px:
            self.status_label.setText(f"Spot ({refined_spot_fft_px[0]:.1f},{refined_spot_fft_px[1]:.1f}) already selected.")
            return

        self.selected_spots_raw_refined_fft_px.append(refined_spot_fft_px)
        
        corrected_spot_ideal_system_px = None
        if self.sub_F_m2i is not None and self.sub_t_m2i is not None and apply_affine_transform:
            try:
                spot_np = np.array([refined_spot_fft_px], dtype=float)
                corrected_array = apply_affine_transform(spot_np, self.sub_F_m2i, self.sub_t_m2i)
                if corrected_array is not None: corrected_spot_ideal_system_px = tuple(corrected_array[0])
            except Exception as e: logger.error(f"Error correcting spot {refined_spot_fft_px}: {e}")
        self.corrected_spots_ideal_px.append(corrected_spot_ideal_system_px)

        print("==============================================================")
        print("==============================================================")
        print(f"refined_spot_fft_px: {refined_spot_fft_px}")
        print(f"corrected_spot_ideal_system_px: {corrected_spot_ideal_system_px}")
        print("==============================================================")
        print("==============================================================")
        
        self._update_spot_distance_list_widget()
        self._redraw_all_markers_on_fft()
        self._update_buttons_state()
        self.status_label.setText(f"Spot {len(self.selected_spots_raw_refined_fft_px)} added.")

    def _update_spot_distance_list_widget(self):
        self.spots_list_widget.clear()
        if not self.selected_spots_raw_refined_fft_px:
            self.spots_list_widget.addItem("No spots selected.")
            return
        for i, raw_spot in enumerate(self.selected_spots_raw_refined_fft_px):
            corrected_text = "N/A"
            if i < len(self.corrected_spots_ideal_px) and self.corrected_spots_ideal_px[i] is not None:
                corr_kx, corr_ky = self.corrected_spots_ideal_px[i] # type: ignore
                corrected_text = f"Corr.FFT({corr_kx:.1f},{corr_ky:.1f}px)"
            self.spots_list_widget.addItem(f"Spot {i+1}: RawFFT({raw_spot[0]:.1f},{raw_spot[1]:.1f}px) | {corrected_text}")

    @pyqtSlot()
    def _remove_last_spot(self): # Zmieniono na "last"
        if self.selected_spots_raw_refined_fft_px:
            self.selected_spots_raw_refined_fft_px.pop()
            self.corrected_spots_ideal_px.pop()
            self._update_spot_distance_list_widget()
            self._redraw_all_markers_on_fft()
            self._update_buttons_state()
            logger.debug("Removed last spot from distance list.")

    @pyqtSlot()
    def _clear_all_listed_spots(self):
        self.selected_spots_raw_refined_fft_px.clear()
        self.corrected_spots_ideal_px.clear()
        self._update_spot_distance_list_widget()
        self._redraw_all_markers_on_fft()
        self._update_buttons_state()
        logger.debug("Cleared all spots from distance list.")

    def _redraw_all_markers_on_fft(self): # Ujednolicona nazwa
        if self.raw_refined_spot_markers: 
            try: 
                self.fft_view_box.removeItem(self.raw_refined_spot_markers)
                self.raw_refined_spot_markers = None
            except RuntimeError: 
                pass
        if self.corrected_spot_display_markers: 
            try: 
                self.fft_view_box.removeItem(self.corrected_spot_display_markers)
                self.corrected_spot_display_markers = None
            except RuntimeError: 
                pass
        
        if self.selected_spots_raw_refined_fft_px:
            raw_data = [{'pos':s,'symbol':'o','size':10,'pen':pg.mkPen('orange',width=1.5),'brush':pg.mkBrush(255,165,0,100)} for s in self.selected_spots_raw_refined_fft_px]
            self.raw_refined_spot_markers = ScatterPlotItem(spots=raw_data)
            self.fft_view_box.addItem(self.raw_refined_spot_markers)

        if self.corrected_spots_ideal_px and self.sub_F_m2i is not None and self.sub_t_m2i is not None and apply_affine_transform:
            display_spots = []
            valid_corrected = [s for s in self.corrected_spots_ideal_px if s is not None]
            # if valid_corrected:
            #     try:
            #         F_inv = np.linalg.inv(self.sub_F_m2i)
            #         t_prime = (-self.sub_t_m2i @ F_inv.T).flatten() # type: ignore
            #         transformed = apply_affine_transform(np.array(valid_corrected, dtype=float), F_inv, t_prime)
            #         if transformed is not None: display_spots = [tuple(pt) for pt in transformed]
            #     except Exception as e: logger.error(f"Error transforming corrected spots for display: {e}")
            if valid_corrected:
                corr_data = [{'pos':s,'symbol':'s','size':10,'pen':pg.mkPen('cyan',width=1.5),'brush':pg.mkBrush(0,255,255,100)} for s in valid_corrected]
                self.corrected_spot_display_markers = ScatterPlotItem(spots=corr_data)
                self.fft_view_box.addItem(self.corrected_spot_display_markers)

    def _update_buttons_state(self): # Ujednolicona nazwa
        self.add_spot_button.setEnabled(self.selection_roi.isVisible() and len(self.selected_spots_raw_refined_fft_px) < 2)
        can_calc_dist = (len(self.selected_spots_raw_refined_fft_px) == 2 and all(s is not None for s in self.corrected_spots_ideal_px))
        self.calculate_distance_button.setEnabled(can_calc_dist)
        if not can_calc_dist and len(self.selected_spots_raw_refined_fft_px) == 2:
            self.distance_fft_label.setText("px: - | nm⁻¹: - (Correction failed?)")
        elif not can_calc_dist:
            self.distance_fft_label.setText("px: - | nm⁻¹: -")
            self.distance_real_space_label.setText("nm: -")
            

    def _display_substrate_transform_info(self):
        if self.sub_transform_analysis:
            self.dist_sub_transform_info_label_status.setText("Status: Sub. transform AVAILABLE.")
            self.dist_sub_transform_info_label_rot.setText(f"{self.sub_transform_analysis.get('rotation_angle_deg','N/A'):.2f}°")
            sx,sy=self.sub_transform_analysis.get('principal_stretches',[np.nan,np.nan])
            self.dist_sub_transform_info_label_scale.setText(f"({sx:.3f},{sy:.3f})")
            self.dist_sub_transform_info_label_rmse.setText(f"{self.sub_transform_analysis.get('rmse','N/A'):.3f} px")
        else:
            self.dist_sub_transform_info_label_status.setText("Status: Sub. transform NOT passed.")
            self.dist_sub_transform_info_label_rot.setText("-")
            self.dist_sub_transform_info_label_scale.setText("-")
            self.dist_sub_transform_info_label_rmse.setText("-")

    @pyqtSlot()
    def _on_calculate_distance_clicked(self):
        logger.debug("Calculate Distance button clicked.")
        if not (len(self.corrected_spots_ideal_px) == 2 and self.corrected_spots_ideal_px[0] is not None and self.corrected_spots_ideal_px[1] is not None):
            QMessageBox.warning(self, "Not Enough Data", "Two spots must be selected and successfully corrected to calculate distance.")
            return

        spot1_corr, spot2_corr = self.corrected_spots_ideal_px[0], self.corrected_spots_ideal_px[1]
        
        if self.fft_data is None: 
            self.status_label.setText("Error: No FFT data.")
            return
        fft_rows_ky, fft_cols_kx = self.fft_data.shape
        center_kx_ideal_px = fft_cols_kx / 2.0
        center_ky_ideal_px = fft_rows_ky / 2.0
        g1_vec_ideal_px = (spot1_corr[0] - center_kx_ideal_px, spot1_corr[1] - center_ky_ideal_px)
        g2_vec_ideal_px = (spot2_corr[0] - center_kx_ideal_px, spot2_corr[1] - center_ky_ideal_px)
        delta_g_vec_ideal_px = (g2_vec_ideal_px[0] - g1_vec_ideal_px[0], g2_vec_ideal_px[1] - g1_vec_ideal_px[1])
        dist_fft_px = np.linalg.norm(delta_g_vec_ideal_px)

        root_node = self.history_manager.get_root_node_for_node(self.current_fft_node_id)
        if not (root_node and root_node.parameters): 
            self.status_label.setText("Error: Calibration missing.")
            return
        Lx_nm = root_node.parameters.get("size_nm_x")
        Ly_nm = root_node.parameters.get("size_nm_y")
        if not (Lx_nm and Ly_nm and Lx_nm > 0 and Ly_nm > 0): 
            self.status_label.setText("Error: Invalid Lx/Ly.")
            return
        
        if convert_g_vector_px_to_nm_inv is None: 
            self.status_label.setText("Error: Conversion func missing.")
            return
        delta_g1_vec_nm_inv = convert_g_vector_px_to_nm_inv(g1_vec_ideal_px, Lx_nm, Ly_nm, fft_cols_kx, fft_rows_ky)
        delta_g2_vec_nm_inv = convert_g_vector_px_to_nm_inv(g2_vec_ideal_px, Lx_nm, Ly_nm, fft_cols_kx, fft_rows_ky)
        
        delta_g_vec_nm_inv = (np.abs(np.array(delta_g1_vec_nm_inv)) + np.abs(np.array(delta_g2_vec_nm_inv))) / 2.0

        if delta_g_vec_nm_inv is None:
            self.distance_fft_label.setText(f"px: {dist_fft_px:.2f} | nm⁻¹: Error")
            self.distance_real_space_label.setText("nm: Error")
            return
            
        dist_nm_inv = np.linalg.norm(delta_g_vec_nm_inv)
        self.distance_fft_label.setText(f"px: {dist_fft_px:.2f} | nm⁻¹: {dist_nm_inv:.4f}")

        if dist_nm_inv > 1e-9: self.distance_real_space_label.setText(f"nm: {(1.0 / dist_nm_inv):.3f}")
        else: self.distance_real_space_label.setText("nm: Inf (zero k-space dist)")
        self.status_label.setText("Distance calculated for corrected spots.")

    def get_dialog_results(self) -> Dict[str, Any]:
        return {"selected_spots_fft_px_raw_refined": self.selected_spots_fft_px_raw_refined,
                "corrected_spots_ideal_px": self.corrected_spots_ideal_px,
                "calculated_real_space_distances_nm": self.calculated_real_space_distances_nm}

    def accept(self): 
        logger.info("SpotDistanceDialog closed.")
        super().accept()
    def reject(self): 
        logger.info("SpotDistanceDialog rejected/closed.")
        super().reject()