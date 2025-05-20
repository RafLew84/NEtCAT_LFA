# lfa/gui/dialogs/substrate_spot_dialog.py
import logging
from typing import List, Tuple, Optional, Dict, Any
import numpy as np

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QDialogButtonBox,
    QLabel, QListWidget, QAbstractItemView, QWidget, QSplitter, QGroupBox,
    QFormLayout, QRadioButton, QSpinBox, QComboBox, QCheckBox, QMessageBox,
    QGridLayout
)
from PyQt6.QtCore import Qt, pyqtSlot, QPointF, QRectF
from PyQt6.QtGui import QPen

try:
    import pyqtgraph as pg
    import pyqtgraph.opengl as gl # Dla podglądu 3D
    from pyqtgraph.opengl import GLViewWidget, GLSurfacePlotItem
    ImageView = pg.ImageView
    PlotItem = pg.PlotItem
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
    from scipy.optimize import curve_fit as scipy_curve_fit # Użyj aliasu, aby uniknąć konfliktu nazw, jeśli pg miałby swoją
    SCIPY_OPTIMIZE_AVAILABLE = True
except ImportError: # pragma: no cover
    logging.error("SubstrateSpotSelectionDialog: SciPy (for curve_fit) not found.")
    SCIPY_OPTIMIZE_AVAILABLE = False
    def scipy_curve_fit(*args, **kwargs): raise ImportError("scipy.optimize.curve_fit is not available")


# Importuj funkcje uściślania (dostosuj ścieżkę, jeśli trzeba)
try:
    from ...analysis.peak_fitting import find_max_pixel_in_roi, fit_2d_gaussian_in_roi, _gaussian_2d, SCIPY_AVAILABLE
    from ...analysis.lattice import KNOWN_LATTICES, get_reciprocal_points # Dla idealnej sieci
    from ...core.history import HistoryNode # Dla parametrów Lx, Ly
    from ...logic.history_manager import HistoryManager # Aby znaleźć root_node
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

REFINEMENT_DIRECT_CLICK = "Direct Click"
REFINEMENT_MAX_PIXEL = "Max Pixel"
REFINEMENT_GAUSSIAN_FIT = "2D Gaussian Fit"

LATTICE_TYPE_HEXAGONAL = "hexagonal"
LATTICE_TYPE_SQUARE = "square"

class SubstrateSpotSelectionDialog(QDialog):
    def __init__(self,
                 fft_image_data: Optional[np.ndarray],
                 history_manager: HistoryManager,
                 current_fft_node_id: str,
                 current_spots: Optional[List[Tuple[float, float]]] = None,
                 default_refinement_method: str = REFINEMENT_DIRECT_CLICK,
                 default_refinement_roi_size: int = 5,
                 parent=None):
        super().__init__(parent)
        
        if not PYQTGRAPH_AVAILABLE: # pragma: no cover
            # Proste okno dialogowe z informacją o błędzie, jeśli pyqtgraph nie jest dostępne
            err_layout = QVBoxLayout(self)
            err_layout.addWidget(QLabel("Critical Error: PyQtGraph library is not available.\nThis dialog cannot function."))
            self.setWindowTitle("Error")
            return

        self.setWindowTitle("Select Substrate Spots")
        self.setMinimumSize(1200, 750) # Zwiększony rozmiar

        self.fft_data = fft_image_data
        self.history_manager = history_manager
        self.current_fft_node_id = current_fft_node_id

        self.selected_spots: List[Tuple[float, float]] = list(current_spots) if current_spots else []
        self.spot_markers_on_image: Optional[ScatterPlotItem] = None # Teraz jeden item dla wszystkich markerów

        self.current_refinement_method = default_refinement_method
        self.refinement_roi_size = default_refinement_roi_size

        self.current_lattice_type: Optional[str] = LATTICE_TYPE_HEXAGONAL # Domyślny
        self.limits_per_lattice = {LATTICE_TYPE_HEXAGONAL: 6, LATTICE_TYPE_SQUARE: 4}
        self.ideal_lattice_overlay_item: Optional[ScatterPlotItem] = None

        self.last_preview_gauss_fit_popt: Optional[np.ndarray] = None
        self.last_preview_gauss_fit_center_abs: Optional[Tuple[float, float]] = None # (kx, ky)
        self.last_preview_gauss_roi_state: Optional[Dict] = None # Stan selection_roi dla którego wykonano fit

        self._init_ui()
        self._connect_signals()
        self._update_spots_list_widget()
        self._redraw_all_spot_markers()
        self._update_add_spot_button_state()

        if self.current_refinement_method == REFINEMENT_MAX_PIXEL: self.rb_refine_max_pixel.setChecked(True)
        elif self.current_refinement_method == REFINEMENT_GAUSSIAN_FIT: self.rb_refine_gaussian.setChecked(True)
        else: self.rb_refine_direct.setChecked(True)
        self.refinement_roi_size_spinbox.setValue(self.refinement_roi_size)
        
        self._on_refinement_method_changed() # Aby ustawić początkową widoczność kontrolek
        self._on_lattice_type_changed() # Aby ustawić początkowy typ i narysować siatkę jeśli checkbox zaznaczony

        logger.debug("SubstrateSpotSelectionDialog initialized.")

    def __del__(self):
        """Additional cleanup for garbage collection."""
        if hasattr(self, 'gl_roi_surface_plot_item') and self.gl_roi_surface_plot_item:
            self.gl_roi_surface_plot_item.deleteLater()
        if hasattr(self, 'gl_gauss_surface_plot_item') and self.gl_gauss_surface_plot_item:
            self.gl_gauss_surface_plot_item.deleteLater()

    def _init_ui(self):
        main_layout = QHBoxLayout(self)

        # --- Lewy Panel: Główny obraz FFT i kontrolki ---
        left_panel_widget = QWidget()
        left_panel_layout = QVBoxLayout(left_panel_widget)

        self.fft_plot_widget = GraphicsLayoutWidget()
        self.fft_view_box = self.fft_plot_widget.addViewBox(row=0, col=0, lockAspect=True, invertY=True)
        self.fft_image_item = ImageItem()
        self.fft_view_box.addItem(self.fft_image_item)
        self.fft_view_box.setMenuEnabled(True) # Włącz domyślne menu dla zoom/pan
        self.fft_view_box.setMouseMode(ViewBox.PanMode) # Domyślnie tryb przesuwania
        self.fft_view_box.setMouseEnabled(x=True, y=True) # Włącz przesuwanie/zoomowanie myszką

        if self.fft_data is not None:
            self.fft_image_item.setImage(self.fft_data.T)

        self.selection_roi = RectROI(pos=(0,0), size=(self.refinement_roi_size, self.refinement_roi_size),
                                     pen=pg.mkPen('y', width=2), translateSnap=True, scaleSnap=True,
                                     movable=True, resizable=True, rotatable=False)
        self.fft_view_box.addItem(self.selection_roi)
        self.selection_roi.setVisible(False)
        left_panel_layout.addWidget(self.fft_plot_widget, stretch=1)

        # Kontrolki (przeniesione na dół lewego panelu dla lepszego układu)
        controls_group = QGroupBox("Controls")
        controls_group_layout = QVBoxLayout(controls_group)

        refinement_group = QGroupBox("Spot Refinement")
        refinement_layout = QFormLayout(refinement_group)
        self.rb_refine_direct = QRadioButton(REFINEMENT_DIRECT_CLICK); self.rb_refine_direct.setChecked(True)
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
        self.add_spot_button = QPushButton("Add/Update Spot from ROI"); self.add_spot_button.setEnabled(False)
        refinement_layout.addRow(self.add_spot_button)
        controls_group_layout.addWidget(refinement_group)
        
        lattice_type_group = QGroupBox("Lattice Type & Overlay")
        lattice_type_layout = QFormLayout(lattice_type_group)
        self.lattice_type_combo = QComboBox(); self.lattice_type_combo.addItems([LATTICE_TYPE_HEXAGONAL.capitalize(), LATTICE_TYPE_SQUARE.capitalize()])
        lattice_type_layout.addRow("Substrate Type:", self.lattice_type_combo)
        self.show_ideal_lattice_checkbox = QCheckBox("Show Ideal Lattice Overlay"); self.show_ideal_lattice_checkbox.setChecked(True)
        lattice_type_layout.addRow(self.show_ideal_lattice_checkbox)
        controls_group_layout.addWidget(lattice_type_group)
        
        left_panel_layout.addWidget(self.fft_plot_widget, stretch=1) # Główny obraz
        left_panel_layout.addWidget(controls_group) # Kontrolki pod spodem
        main_layout.addWidget(left_panel_widget, stretch=2)

        right_panel_widget = QWidget()
        right_panel_layout = QVBoxLayout(right_panel_widget)
        right_panel_widget.setMinimumWidth(450) # Utrzymaj minimalną szerokość

        preview_group = QGroupBox("Live Previews")
        preview_grid_layout = QGridLayout(preview_group) # Zamiast QVBoxLayout

        # --- Wiersz 1: Podglądy ROI ---
        # 2D ROI Preview
        roi_2d_container = QWidget()
        roi_2d_v_layout = QVBoxLayout(roi_2d_container)
        roi_2d_v_layout.addWidget(QLabel("ROI 2D Preview:"))
        self.enable_2d_roi_preview_checkbox = QCheckBox("Enable")
        self.enable_2d_roi_preview_checkbox.setChecked(True)
        roi_2d_v_layout.addWidget(self.enable_2d_roi_preview_checkbox)
        self.roi_preview_2d_widget = GraphicsLayoutWidget()
        self.roi_preview_2d_widget.setMinimumSize(150, 150)  # Ujednolicone rozmiary
        self.roi_preview_2d_plot = self.roi_preview_2d_widget.addViewBox(lockAspect=True, invertY=True)
        self.roi_preview_2d_image_item = ImageItem()
        self.roi_preview_2d_plot.addItem(self.roi_preview_2d_image_item)
        roi_2d_v_layout.addWidget(self.roi_preview_2d_widget, 1)  # Dodany stretch factor
        preview_grid_layout.addWidget(roi_2d_container, 0, 0)

        print("3D ROI Preview")
        # 3D ROI Preview
        roi_3d_container = QWidget()
        roi_3d_v_layout = QVBoxLayout(roi_3d_container)
        roi_3d_v_layout.addWidget(QLabel("ROI 3D Preview:"))
        self.enable_3d_roi_preview_checkbox = QCheckBox("Enable")
        self.enable_3d_roi_preview_checkbox.setChecked(False)
        roi_3d_v_layout.addWidget(self.enable_3d_roi_preview_checkbox)
        self.gl_roi_view_widget = GLViewWidget()
        self.gl_roi_view_widget.setMinimumSize(150, 150)  # Ujednolicone rozmiary
        self.gl_roi_surface_plot_item = GLSurfacePlotItem(color=(0.5, 0.5, 1, 0.7))
        self.gl_roi_view_widget.addItem(self.gl_roi_surface_plot_item)
        roi_3d_v_layout.addWidget(self.gl_roi_view_widget, 1)  # Dodany stretch factor
        preview_grid_layout.addWidget(roi_3d_container, 0, 1)

        # --- Wiersz 2: Podglądy Gaussian Fit ---
        # 2D Gaussian Fit Preview
        gauss_2d_container = QWidget()
        gauss_2d_v_layout = QVBoxLayout(gauss_2d_container)
        gauss_2d_v_layout.addWidget(QLabel("Gaussian Fit 2D Preview:"))
        self.enable_gauss_2d_preview_checkbox = QCheckBox("Enable")
        self.enable_gauss_2d_preview_checkbox.setChecked(True)
        gauss_2d_v_layout.addWidget(self.enable_gauss_2d_preview_checkbox)
        self.gaussian_preview_2d_widget = GraphicsLayoutWidget()
        self.gaussian_preview_2d_widget.setMinimumSize(150, 150)  # Ujednolicone rozmiary
        self.gaussian_preview_2d_plot = self.gaussian_preview_2d_widget.addViewBox(lockAspect=True, invertY=True)
        self.gaussian_preview_2d_image_item = ImageItem()
        self.gaussian_preview_2d_plot.addItem(self.gaussian_preview_2d_image_item)
        gauss_2d_v_layout.addWidget(self.gaussian_preview_2d_widget, 1)  # Dodany stretch factor
        preview_grid_layout.addWidget(gauss_2d_container, 1, 0)

        # 3D Gaussian Fit Preview
        gauss_3d_container = QWidget()
        gauss_3d_v_layout = QVBoxLayout(gauss_3d_container)
        gauss_3d_v_layout.addWidget(QLabel("Gaussian Fit 3D Preview:"))
        self.enable_gauss_3d_preview_checkbox = QCheckBox("Enable")
        self.enable_gauss_3d_preview_checkbox.setChecked(False)
        gauss_3d_v_layout.addWidget(self.enable_gauss_3d_preview_checkbox)
        self.gl_gauss_view_widget = GLViewWidget()
        self.gl_gauss_view_widget.setMinimumSize(150, 150)  # Ujednolicone rozmiary
        self.gl_gauss_surface_plot_item = GLSurfacePlotItem(color=(0.5, 0.5, 1, 0.7))
        self.gl_gauss_view_widget.addItem(self.gl_gauss_surface_plot_item)
        gauss_3d_v_layout.addWidget(self.gl_gauss_view_widget, 1)  # Dodany stretch factor
        preview_grid_layout.addWidget(gauss_3d_container, 1, 1)

        # Ustawienia wyrównania dla grid layout
        preview_grid_layout.setColumnStretch(0, 1)  # Wyrównanie kolumn
        preview_grid_layout.setColumnStretch(1, 1)
        preview_grid_layout.setRowStretch(0, 1)    # Wyrównanie wierszy
        preview_grid_layout.setRowStretch(1, 1)

        self.gauss_2d_container = gauss_2d_container
        self.gauss_3d_container = gauss_3d_container
        self.gauss_2d_container.setVisible(False)
        self.gauss_3d_container.setVisible(False)

        right_panel_layout.addWidget(preview_group)


        spots_list_group = QGroupBox("Selected Spots Management")
        spots_list_layout = QVBoxLayout(spots_list_group)
        self.spots_list_widget = QListWidget(); self.spots_list_widget.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        spots_list_layout.addWidget(self.spots_list_widget)
        spot_buttons_layout = QHBoxLayout()
        self.remove_spot_button = QPushButton("Remove Selected"); self.clear_all_spots_button = QPushButton("Clear All")
        spot_buttons_layout.addWidget(self.remove_spot_button); spot_buttons_layout.addWidget(self.clear_all_spots_button)
        spots_list_layout.addLayout(spot_buttons_layout)
        right_panel_layout.addWidget(spots_list_group)
        
        self.status_label = QLabel("Click on FFT to place ROI, or drag existing ROI."); right_panel_layout.addWidget(self.status_label)
        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel); right_panel_layout.addWidget(self.button_box)
        
        right_panel_layout.addStretch(1)
        main_layout.addWidget(right_panel_widget, stretch=1)

    def _clear_3d_surface(self, surface_item: Optional[GLSurfacePlotItem]):
        """Reset surface plot to minimal valid state."""
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

    def _connect_signals(self):
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        self.remove_spot_button.clicked.connect(self._remove_selected_spot)
        self.clear_all_spots_button.clicked.connect(self._clear_all_spots_in_dialog) # Zmieniona nazwa slotu

        # Podłączenie kliknięcia na obrazie FFT
        # Sygnał sigMouseClicked jest emitowany przez scenę ViewBoxa
        self.fft_view_box.scene().sigMouseClicked.connect(self._handle_fft_image_click)
        self.selection_roi.sigRegionChanged.connect(self._handle_roi_region_changing)
        # self.selection_roi.sigRegionChangeFinished.connect(self._handle_roi_changed_finished) # Po zakończeniu zmiany ROI

        # Zmiana metody uściślania
        self.rb_refine_direct.toggled.connect(self._on_refinement_method_changed)
        self.rb_refine_max_pixel.toggled.connect(self._on_refinement_method_changed)
        self.rb_refine_gaussian.toggled.connect(self._on_refinement_method_changed)
        self.refinement_roi_size_spinbox.valueChanged.connect(self._on_refinement_roi_size_changed)

        # Przycisk dodawania spotu
        self.add_spot_button.clicked.connect(self._add_current_roi_spot)
        
        # Zmiana typu sieci
        self.lattice_type_combo.currentTextChanged.connect(self._on_lattice_type_changed)
        self.show_ideal_lattice_checkbox.stateChanged.connect(self._redraw_ideal_lattice_overlay)

        self.enable_2d_roi_preview_checkbox.stateChanged.connect(self._update_roi_previews)
        self.enable_3d_roi_preview_checkbox.stateChanged.connect(self._update_roi_previews)
        self.enable_gauss_2d_preview_checkbox.stateChanged.connect(self._update_roi_previews) # Zmieniono nazwę checkboxa
        self.enable_gauss_3d_preview_checkbox.stateChanged.connect(self._update_roi_previews) # Zmieniono nazwę checkboxa


    def _update_spots_list_widget(self):
        self.spots_list_widget.clear()
        for i, (kx, ky) in enumerate(self.selected_spots):
            self.spots_list_widget.addItem(f"S{i+1}: ({kx:.2f}, {ky:.2f})")
        self._update_add_spot_button_state()

    @pyqtSlot()
    def _remove_selected_spot(self):
        current_item = self.spots_list_widget.currentItem()
        if current_item:
            row = self.spots_list_widget.row(current_item)
            if 0 <= row < len(self.selected_spots):
                del self.selected_spots[row]
                self._update_spots_list_widget()
                self._redraw_all_spot_markers() # Przerysuj markery
                logger.debug(f"Removed spot at index {row}")

    @pyqtSlot()
    def _clear_all_spots_in_dialog(self): # Zmieniona nazwa, aby uniknąć konfliktu
        self.selected_spots.clear()
        self._update_spots_list_widget()
        self._redraw_all_spot_markers() # Przerysuj markery
        logger.debug("Cleared all substrate spots in dialog.")

    def _redraw_all_spot_markers(self):
        """Usuwa stary marker (jeśli istnieje) i rysuje nowe na podstawie self.selected_spots."""
        # Usuń stary ScatterPlotItem, jeśli istnieje
        if self.spot_markers_on_image is not None:
            try:
                self.fft_view_box.removeItem(self.spot_markers_on_image)
            except RuntimeError: # pragma: no cover
                pass # Może już został usunięty lub scena jest nieprawidłowa
            self.spot_markers_on_image = None # Zresetuj referencję

        if not self.selected_spots: # Jeśli nie ma spotów do narysowania, zakończ
            return

        # Dane dla ScatterPlotItem - (kx, ky) są poprawne dla pyqtgraph,
        # jeśli ViewBox ma standardową orientację (X rośnie w prawo, Y w górę),
        # a dane obrazu (fft_data.T) są ustawione tak, że oś 0 danych to X, a oś 1 to Y.
        # Nasz fft_view_box ma invertY=True, a obraz jest .T, więc:
        # Oryginalne dane FFT: (indeks_wiersza_ky, indeks_kolumny_kx)
        # fft_image_item.setImage(self.fft_data.T) -> ImageItem ma dane (indeks_kolumny_kx, indeks_wiersza_ky)
        # ScatterPlotItem(pos=(x_coord, y_coord))
        # x_coord na wykresie odpowiada pierwszej osi ImageItem (kx)
        # y_coord na wykresie odpowiada drugiej osi ImageItem (ky)
        # Więc przekazujemy (kx, ky) bezpośrednio.
        spots_to_draw_final = [{'pos': spot, 
                                'symbol': 'o', 
                                'size': 10, 
                                'pen': pg.mkPen('g', width=1.5),
                                'brush': pg.mkBrush(50,205,50,120)} # Półprzezroczysty zielony
                               for spot in self.selected_spots]

        if spots_to_draw_final: # Dodaj tylko jeśli są spoty
            new_scatter_item = ScatterPlotItem() # Stwórz nowy item
            new_scatter_item.setData(spots=spots_to_draw_final) # Ustaw dane
            self.fft_view_box.addItem(new_scatter_item)
            self.spot_markers_on_image = new_scatter_item # Przypisz nowy item
            logger.debug(f"Redrawn {len(self.selected_spots)} substrate spot markers.")
        else: # pragma: no cover
            logger.debug("No substrate spots to draw.")

    def _handle_fft_image_click(self, event):
        """Obsługuje kliknięcie na głównym obrazie FFT."""
        if event.button() == Qt.MouseButton.LeftButton:
            # mapSceneToView konwertuje współrzędne sceny (kliknięcia) na współrzędne ViewBoxa
            pos_viewbox = self.fft_view_box.mapSceneToView(event.scenePos())
            # mapToData konwertuje współrzędne itemu (które są takie same jak ViewBoxa, jeśli item wypełnia ViewBox)
            # na współrzędne danych obrazu, który został ustawiony przez setImage()
            # Jeśli fft_image_item.setImage(data.T), to mapToData zwróci (indeks_kolumny_oryg_FFT, indeks_wiersza_oryg_FFT)
            # czyli (kx, ky)
            mapped_pos = self.fft_image_item.mapToData(pos_viewbox)

            if mapped_pos is not None:
                kx, ky = int(round(mapped_pos.x())), int(round(mapped_pos.y()))
                logger.debug(f"Dialog FFT click: mapped to data (kx, ky) = ({kx}, {ky})")

                # Ustaw ROI wokół klikniętego punktu
                roi_size = self.refinement_roi_size_spinbox.value()
                roi_x = kx - roi_size // 2
                roi_y = ky - roi_size // 2
                
                # Walidacja granic ROI względem danych obrazu FFT
                if self.fft_data is not None:
                    max_y, max_x = self.fft_data.shape # Oryginalne dane FFT (przed transpozycją)
                    # roi_x (kx) musi być w [0, max_x - roi_size]
                    # roi_y (ky) musi być w [0, max_y - roi_size]
                    roi_x = np.clip(roi_x, 0, max_x - roi_size)
                    roi_y = np.clip(roi_y, 0, max_y - roi_size)

                self.selection_roi.setPos((roi_x, roi_y), update=False) # Ustaw pozycję ROI
                self.selection_roi.setSize((roi_size, roi_size), update=False) # Ustaw rozmiar ROI
                self.selection_roi.setVisible(True)
                self.add_spot_button.setEnabled(True) # Włącz przycisk dodawania

                self._update_roi_previews() # Zaktualizuj podglądy ROI
            event.accept()
        else:
            event.ignore() # pragma: no cover

    def _handle_roi_changed_finished(self):
        """Obsługuje zakończenie zmiany ROI (przesunięcie lub zmiana rozmiaru)."""
        if self.selection_roi.isVisible():
            self.add_spot_button.setEnabled(True)
            roi_pos = self.selection_roi.pos()
            roi_size = self.selection_roi.size()
            logger.debug(f"ROI changed/moved: Pos ({roi_pos.x():.1f}, {roi_pos.y():.1f}), Size ({roi_size.x():.1f}, {roi_size.y():.1f})")
            
            # Zaktualizuj rozmiar w spinboxie, jeśli ROI było ręcznie skalowane
            current_roi_w = int(round(roi_size.x()))
            if current_roi_w != self.refinement_roi_size_spinbox.value() and current_roi_w >= self.refinement_roi_size_spinbox.minimum() and current_roi_w <= self.refinement_roi_size_spinbox.maximum() and current_roi_w % 2 != 0 :
                self.refinement_roi_size_spinbox.blockSignals(True)
                self.refinement_roi_size_spinbox.setValue(current_roi_w)
                self.refinement_roi_size_spinbox.blockSignals(False)

            self._update_roi_previews()

    def _clear_last_preview_gauss_fit(self):
        """Helper to invalidate stored preview Gaussian fit results."""
        self.last_preview_gauss_fit_popt = None
        self.last_preview_gauss_fit_center_abs = None
        self.last_preview_gauss_roi_state = None
        logger.debug("Cleared last preview Gaussian fit results.")

    @pyqtSlot(object) # Zmieniono z QRectF na object, bo sigRegionChanged emituje sam obiekt ROI
    def _handle_roi_region_changing(self, roi_item: Optional[RectROI] = None): # roi_item jest opcjonalny
        """Obsługuje zmianę ROI (przesunięcie lub zmiana rozmiaru) - live update."""
        if roi_item is None: # Jeśli wywołane bez argumentu, użyj self.selection_roi
            roi_item = self.selection_roi

        if roi_item.isVisible(): # type: ignore
            # self.add_spot_button.setEnabled(True) # Przycisk "Add Spot" jest teraz włączany po kliknięciu
            roi_pos = roi_item.pos() # type: ignore
            roi_size = roi_item.size() # type: ignore
            logger.debug(f"ROI region changing: Pos ({roi_pos.x():.1f}, {roi_pos.y():.1f}), Size ({roi_size.x():.1f}, {roi_size.y():.1f})")
            
            current_roi_w = int(round(roi_size.x())) # type: ignore
            if current_roi_w != self.refinement_roi_size_spinbox.value() and \
               self.refinement_roi_size_spinbox.minimum() <= current_roi_w <= self.refinement_roi_size_spinbox.maximum() and \
               current_roi_w % 2 != 0 :
                self.refinement_roi_size_spinbox.blockSignals(True)
                self.refinement_roi_size_spinbox.setValue(current_roi_w)
                self.refinement_roi_size_spinbox.blockSignals(False)
            
            self._clear_last_preview_gauss_fit() # Wyczyść poprzedni fit
            self._update_roi_previews() # Live update podglądów


    def _update_roi_previews(self):
        """Aktualizuje podglądy 2D ROI, dopasowania Gaussa i 3D."""
        if not self.selection_roi.isVisible() or self.fft_data is None: # type: ignore
            self._clear_last_preview_gauss_fit()
            if hasattr(self, 'roi_preview_2d_image_item'): self.roi_preview_2d_image_item.clear()
            if hasattr(self, 'gaussian_preview_2d_image_item'): self.gaussian_preview_2d_image_item.clear()
            if hasattr(self, 'gl_roi_surface_item'): self.gl_roi_surface_item.setData(z=np.array([[0,0],[0,0]])) # Wyczyść podgląd 3D
            # if hasattr(self, 'gl_gauss_surface_item'): self.gl_gauss_surface_item.setData(z=np.array([[0,0],[0,0]]))
            if hasattr(self, 'gl_roi_surface_plot_item'): self._clear_3d_surface(self.gl_roi_surface_plot_item)
            if hasattr(self, 'gl_gauss_surface_plot_item'): self._clear_3d_surface(self.gl_gauss_surface_plot_item)
            return

        roi_state_for_comparison = self.selection_roi.getState() # type: ignore
        x0_roi, y0_roi = int(round(roi_state_for_comparison['pos'].x())), int(round(roi_state_for_comparison['pos'].y()))

        roi_state = self.selection_roi.getState() # type: ignore
        x0_roi, y0_roi = int(round(roi_state['pos'].x())), int(round(roi_state['pos'].y()))
        width_roi, height_roi = int(round(roi_state['size'].x())), int(round(roi_state['size'].y()))
        x1_roi, y1_roi = x0_roi + width_roi, y0_roi + height_roi
        
        max_ky, max_kx = self.fft_data.shape
        y0_cl = np.clip(y0_roi, 0, max_ky); y1_cl = np.clip(y1_roi, 0, max_ky)
        x0_cl = np.clip(x0_roi, 0, max_kx); x1_cl = np.clip(x1_roi, 0, max_kx)

        if y1_cl <= y0_cl or x1_cl <= x0_cl : # pragma: no cover
             logger.warning("Invalid ROI slice for preview."); return

        roi_patch = self.fft_data[y0_cl:y1_cl, x0_cl:x1_cl]

        if roi_patch.size > 0:
            # Podgląd 2D ROI
            if self.enable_2d_roi_preview_checkbox.isChecked() and hasattr(self, 'roi_preview_2d_image_item'):
                self.roi_preview_2d_image_item.setImage(roi_patch.T); self.roi_preview_2d_plot.autoRange()
            elif hasattr(self, 'roi_preview_2d_image_item'): self.roi_preview_2d_image_item.clear()

            # Podgląd 3D ROI
            if self.enable_3d_roi_preview_checkbox.isChecked() and hasattr(self, 'gl_roi_surface_plot_item') and self.gl_roi_surface_plot_item:
                self._update_3d_surface_plot(self.gl_roi_surface_plot_item, roi_patch)
            elif hasattr(self, 'gl_roi_surface_plot_item') and self.gl_roi_surface_plot_item: 
                # self.gl_roi_surface_item.setData(z=np.array([[0,0],[0,0]])) # Wyczyść
                self._clear_3d_surface(self.gl_roi_surface_plot_item)

            # Podglądy Gaussa
            if self.rb_refine_gaussian.isChecked():
                fitted_gauss_params = None
                fitted_gauss_2d_for_preview = None
                if PEAK_FITTING_MODULE_AVAILABLE and SCIPY_AVAILABLE:
                    patch_h, patch_w = roi_patch.shape
                    # Użyj środka ROI patch jako initial guess dla y0, x0 w _gaussian_2d
                    # Ale fit_2d_gaussian_in_roi oczekuje współrzędnych absolutnych i promienia
                    # Dla podglądu, wykonujemy fit na samym patchu
                    p_y, p_x = np.mgrid[0:patch_h, 0:patch_w]
                    p_xy_flat = (p_y.flatten(), p_x.flatten())
                    p_data_flat = roi_patch.flatten()
                    try:
                        p0_gauss = [roi_patch.max() - roi_patch.min(), patch_h/2.0, patch_w/2.0, patch_w/4.0, patch_h/4.0, 0.0, roi_patch.min()]
                        if callable(scipy_curve_fit) and callable(_gaussian_2d): # Sprawdź, czy funkcje są dostępne
                            popt_gauss, pcov_gauss = scipy_curve_fit(_gaussian_2d, p_xy_flat, p_data_flat, p0=p0_gauss)
                            self.last_preview_gauss_fit_popt = popt_gauss
                            # Oblicz absolutne centrum dopasowania
                            # popt_gauss[1] to y0_patch, popt_gauss[2] to x0_patch (wzgl. roi_patch)
                            abs_fit_ky = y0_roi + popt_gauss[1] # y0_roi to górny lewy róg roi_patch
                            abs_fit_kx = x0_roi + popt_gauss[2] # x0_roi to górny lewy róg roi_patch
                            self.last_preview_gauss_fit_center_abs = (abs_fit_kx, abs_fit_ky)
                            self.last_preview_gauss_roi_state = roi_state_for_comparison.copy() # Zapisz stan ROI dla tego fita
                            logger.info(f"Preview Gaussian fit successful. Stored center: {self.last_preview_gauss_fit_center_abs}")

                            fitted_gauss_flat = _gaussian_2d(p_xy_flat, *popt_gauss)
                            fitted_gauss_2d_for_preview = fitted_gauss_flat.reshape(patch_h, patch_w)
                            fitted_gauss_params = popt_gauss # Zapisz parametry dla 3D
                        else: # pragma: no cover
                            self._clear_last_preview_gauss_fit()
                    except Exception as e_fit: # pragma: no cover
                        logger.warning(f"Gaussian fit for preview failed: {e_fit}")
                        fitted_gauss_2d_for_preview = roi_patch # Pokaż oryginał jeśli fit się nie uda

                # Podgląd 2D Gaussa
                if self.enable_gauss_2d_preview_checkbox.isChecked() and hasattr(self, 'gaussian_preview_2d_image_item'):
                    if fitted_gauss_2d_for_preview is not None:
                        self.gaussian_preview_2d_image_item.setImage(fitted_gauss_2d_for_preview.T)
                        self.gaussian_preview_2d_plot.autoRange()
                    else:
                        self.gaussian_preview_2d_image_item.setImage(roi_patch.T) # Fallback na oryginalny patch
                        self.gaussian_preview_2d_plot.autoRange()
                elif hasattr(self, 'gaussian_preview_2d_image_item'): self.gaussian_preview_2d_image_item.clear()

                # Podgląd 3D Gaussa
                if self.enable_gauss_3d_preview_checkbox.isChecked() and hasattr(self, 'gl_gauss_surface_plot_item') and self.gl_gauss_surface_plot_item:
                    if fitted_gauss_2d_for_preview is not None: # Użyj tych samych danych co dla 2D Gaussa
                        self._update_3d_surface_plot(self.gl_gauss_surface_plot_item, fitted_gauss_2d_for_preview)
                    else: # Jeśli fit się nie udał, można wyczyścić lub pokazać oryginalny patch
                         self._update_3d_surface_plot(self.gl_gauss_surface_plot_item, roi_patch)
                elif hasattr(self, 'gl_gauss_surface_plot_item') and self.gl_gauss_surface_plot_item: 
                    # self.gl_gauss_surface_item.setData(z=np.array([[0,0],[0,0]]))
                    self._clear_3d_surface(self.gl_gauss_surface_plot_item)

            else: # Jeśli nie jest wybrany tryb Gaussa
                self._clear_last_preview_gauss_fit()
                if hasattr(self, 'gaussian_preview_2d_image_item'): self.gaussian_preview_2d_image_item.clear()
                if hasattr(self, 'gl_gauss_surface_item') and self.gl_gauss_surface_item: self.gl_gauss_surface_item.setData(z=np.array([[0,0],[0,0]]))
        else: # pragma: no cover
            # ... (czyszczenie wszystkich podglądów, jeśli roi_patch.size == 0) ...
            if hasattr(self, 'roi_preview_2d_image_item'): self.roi_preview_2d_image_item.clear()
            if hasattr(self, 'gaussian_preview_2d_image_item'): self.gaussian_preview_2d_image_item.clear()
            if hasattr(self, 'gl_roi_surface_item') and self.gl_roi_surface_plot_item: self.gl_roi_surface_plot_item.setData(z=np.array([[0,0],[0,0]]))
            if hasattr(self, 'gl_gauss_surface_item') and self.gl_gauss_surface_plot_item: self.gl_gauss_surface_plot_item.setData(z=np.array([[0,0],[0,0]]))



    def _update_3d_surface_plot(self, surface_item: GLSurfacePlotItem, data_2d: np.ndarray):
        """Aktualizuje GLSurfacePlotItem danymi 2D."""
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

    @pyqtSlot()
    def _on_refinement_method_changed(self):
        if not self.rb_refine_gaussian.isChecked():
            self._clear_last_preview_gauss_fit() # Wyczyść, jeśli zmieniono z Gaussa
        
        is_gaussian_mode = self.rb_refine_gaussian.isChecked()
        self.gaussian_preview_2d_widget.setVisible(is_gaussian_mode)
        # self.enable_3d_gauss_preview_checkbox.setVisible(is_gaussian_mode) # Jeśli jest podgląd 3D Gaussa
        
        if self.rb_refine_direct.isChecked():
            self.current_refinement_method = REFINEMENT_DIRECT_CLICK
            self.refinement_roi_size_spinbox.setEnabled(False)
            self.selection_roi.setVisible(False) # Direct click nie używa ROI na obrazie
            self.add_spot_button.setEnabled(False) # Direct click - kliknięcie na obrazie dodaje od razu
            self.status_label.setText("Click directly on FFT image to add spot.")
        else: # Max Pixel lub Gaussian Fit
            self.current_refinement_method = REFINEMENT_MAX_PIXEL if self.rb_refine_max_pixel.isChecked() else REFINEMENT_GAUSSIAN_FIT
            self.refinement_roi_size_spinbox.setEnabled(True)
            # self.selection_roi.setVisible(True) # Pokaż ROI, jeśli jeszcze nie jest
            self.add_spot_button.setEnabled(self.selection_roi.isVisible())
            # self.add_spot_button.setEnabled(True) # Przycisk "Add Spot" staje się głównym sposobem dodawania
            self.status_label.setText("Drag ROI to desired spot, then click 'Add/Update Spot'.")
            if self.current_refinement_method == REFINEMENT_GAUSSIAN_FIT:
                self.gauss_2d_container.setVisible(True)
                self.gauss_3d_container.setVisible(True)
            else:
                self.gauss_2d_container.setVisible(False)
                self.gauss_3d_container.setVisible(False)
        
        self._update_roi_previews() # Zaktualizuj podglądy po zmianie trybu (np. aby pokazać/ukryć podgląd Gaussa)
        logger.debug(f"Refinement method changed to: {self.current_refinement_method}")


    @pyqtSlot(int)
    def _on_refinement_roi_size_changed(self, value: int):
        self.refinement_roi_size = value
        # Zaktualizuj rozmiar selection_roi, jeśli jest widoczne
        self._clear_last_preview_gauss_fit()
        if self.selection_roi.isVisible():
            current_pos = self.selection_roi.pos()
            # Wycentruj nowy rozmiar ROI wokół poprzedniego środka ROI
            old_size = self.selection_roi.size()
            center_x = current_pos.x() + old_size.x() / 2
            center_y = current_pos.y() + old_size.y() / 2
            new_pos_x = center_x - value / 2
            new_pos_y = center_y - value / 2
            self.selection_roi.setPos((new_pos_x, new_pos_y), update=False)
            self.selection_roi.setSize((value, value), update=False) # Aktualizuj rozmiar ROI na obrazie
            self._handle_roi_changed_finished() # Wywołaj aktualizację podglądów
            
        logger.debug(f"Refinement ROI size changed to: {self.refinement_roi_size}")

    @pyqtSlot()
    def _add_current_roi_spot(self):
        """Dodaje spot na podstawie aktualnego ROI i wybranej metody uściślania."""
        if not self.selection_roi.isVisible() or self.fft_data is None: # pragma: no cover
            self.status_label.setText("Error: No ROI selected or no FFT data.")
            return

        # Sprawdź limit spotów
        max_spots = self.limits_per_lattice.get(self.current_lattice_type, 6) # Domyślnie 6
        if len(self.selected_spots) >= max_spots:
            QMessageBox.warning(self, "Limit Reached",
                                f"Maximum number of spots ({max_spots}) for {self.current_lattice_type} lattice already selected.")
            return

        roi_state = self.selection_roi.getState()
        x0_roi, y0_roi = int(round(roi_state['pos'].x())), int(round(roi_state['pos'].y()))
        width_roi, height_roi = int(round(roi_state['size'].x())), int(round(roi_state['size'].y()))
        
        # Środek ROI (jako punkt startowy dla uściślania)
        center_kx = x0_roi + width_roi // 2
        center_ky = y0_roi + height_roi // 2
        
        refined_kx, refined_ky = float(center_kx), float(center_ky) # Domyślnie środek ROI

        # Uściślanie
        if self.current_refinement_method == REFINEMENT_MAX_PIXEL and PEAK_FITTING_MODULE_AVAILABLE:
            # find_max_pixel_in_roi oczekuje (center_ky, center_kx) i patch_radius
            # ROI dla find_max_pixel_in_roi jest definiowane przez jego promień, a nie przez selection_roi
            patch_radius = self.refinement_roi_size // 2 
            # Upewnij się, że centrum jest w granicach obrazu
            max_h, max_w = self.fft_data.shape
            eff_center_ky = np.clip(center_ky, patch_radius, max_h - 1 - patch_radius)
            eff_center_kx = np.clip(center_kx, patch_radius, max_w - 1 - patch_radius)

            fit_ky, fit_kx = find_max_pixel_in_roi(self.fft_data, (eff_center_ky, eff_center_kx), patch_radius)
            refined_kx, refined_ky = float(fit_kx), float(fit_ky)
            logger.info(f"Spot refined by Max Pixel: ({refined_kx:.2f}, {refined_ky:.2f})")
        elif self.current_refinement_method == REFINEMENT_GAUSSIAN_FIT and PEAK_FITTING_MODULE_AVAILABLE and SCIPY_AVAILABLE:
            # --- ZMIANA: Użyj zapisanego wyniku z podglądu, jeśli dostępny i pasuje ---
            current_selection_roi_state = self.selection_roi.getState() # type: ignore
            # Proste porównanie stanu ROI (można zrobić bardziej zaawansowane, np. z tolerancją)
            # Porównujemy słowniki stanów (pozycja i rozmiar)
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
            else: # Wykonaj nowy, pełny fit
                logger.info("Performing NEW Gaussian fit for Add Spot (preview data not used or ROI changed).")
                patch_radius = self.refinement_roi_size // 2
                max_h, max_w = self.fft_data.shape # type: ignore
                eff_center_ky = np.clip(center_ky, patch_radius, max_h - 1 - patch_radius)
                eff_center_kx = np.clip(center_kx, patch_radius, max_w - 1 - patch_radius)
                
                fit_output = fit_2d_gaussian_in_roi(self.fft_data, (eff_center_ky, eff_center_kx), patch_radius)
                if fit_output:
                    _popt, (fit_ky_abs, fit_kx_abs), _patch = fit_output # Zmieniono na podstawie sugestii o zwracaniu popt i patcha
                    refined_kx, refined_ky = float(fit_kx_abs), float(fit_ky_abs)
                    logger.info(f"Spot refined by NEW 2D Gaussian Fit: ({refined_kx:.2f}, {refined_ky:.2f})")
                else: # pragma: no cover
                    logger.warning("2D Gaussian fit failed for Add Spot. Using ROI center.")
            # --- KONIEC ZMIANY ---

        new_spot = (refined_kx, refined_ky)
        if new_spot not in self.selected_spots:
            self.selected_spots.append(new_spot)
            self._update_spots_list_widget()
            self._redraw_all_spot_markers()
            self.status_label.setText(f"Spot {len(self.selected_spots)} added: ({refined_kx:.2f}, {refined_ky:.2f}).")
        else: # pragma: no cover
            self.status_label.setText(f"Spot ({refined_kx:.2f}, {refined_ky:.2f}) already selected.")

    @pyqtSlot(str)
    def _on_lattice_type_changed(self, text: Optional[str] = None):
        """Obsługuje zmianę wybranego typu sieci."""
        selected_type_text = self.lattice_type_combo.currentText().lower()
        if LATTICE_TYPE_HEXAGONAL in selected_type_text:
            self.current_lattice_type = LATTICE_TYPE_HEXAGONAL
        elif LATTICE_TYPE_SQUARE in selected_type_text:
            self.current_lattice_type = LATTICE_TYPE_SQUARE
        else: # pragma: no cover
            self.current_lattice_type = None 
        logger.debug(f"Dialog: Lattice type set to {self.current_lattice_type}")
        self._update_add_spot_button_state()
        self._redraw_ideal_lattice_overlay()

    def _update_add_spot_button_state(self):
        """Aktualizuje stan przycisku 'Add Spot' na podstawie limitu."""
        if self.current_lattice_type:
            limit = self.limits_per_lattice.get(self.current_lattice_type, 0)
            can_add_more = len(self.selected_spots) < limit
            self.add_spot_button.setEnabled(self.selection_roi.isVisible() and can_add_more)
            if not can_add_more:
                self.status_label.setText(f"Max {limit} spots for {self.current_lattice_type} lattice reached.")
            elif not self.selection_roi.isVisible() and len(self.selected_spots) < limit:
                 self.status_label.setText(f"Click on FFT or drag ROI. {limit - len(self.selected_spots)} spots remaining.")
        else: # pragma: no cover
            self.add_spot_button.setEnabled(self.selection_roi.isVisible()) # Jeśli typ nieznany, pozwól dodawać bez limitu (lub inny default)

    def _redraw_ideal_lattice_overlay(self):
        # TODO: Implementacja rysowania idealnej sieci (analogicznie do VisualizationManager)
        # Będzie potrzebować Lx, Ly z oryginalnego obrazu STM.
        # Można je pobrać przez self.history_manager i self.current_fft_node_id
        # oraz stałej sieci `a_surf` dla wybranego typu (to może być problematyczne bez
        # pełnej informacji o kalibracji lub predefiniowanych wartości `a_surf` dla typów).
        # Na razie pomijamy implementację tej funkcji.
        logger.debug("Redraw ideal lattice overlay requested (not yet implemented in dialog).")
        pass


    def get_selected_spots(self) -> List[Tuple[float, float]]:
        return list(self.selected_spots)

    def accept(self):
        if self.current_lattice_type:
            limit = self.limits_per_lattice.get(self.current_lattice_type, 0)
            if not (0 < len(self.selected_spots) <= limit) and limit > 0 : # Pozwól na 0 jeśli limit 0 (np. błąd)
                QMessageBox.warning(self, "Spot Count Error",
                                    f"Please select between 1 and {limit} spots for a "
                                    f"{self.current_lattice_type} lattice. "
                                    f"Currently selected: {len(self.selected_spots)}.")
                return # Nie zamykaj dialogu
        logger.info(f"SubstrateSpotSelectionDialog accepted with {len(self.selected_spots)} spots for {self.current_lattice_type} lattice.")
        super().accept()

    def reject(self):
        logger.info("SubstrateSpotSelectionDialog rejected.")
        super().reject()

def closeEvent(self, event):
    """Handle dialog close event to clean up OpenGL resources."""
    logger.debug("SubstrateSpotSelectionDialog closing. Cleaning up GL items.")
    
    # Czyszczenie i usuwanie GLSurfacePlotItem
    if hasattr(self, 'gl_roi_surface_plot_item') and self.gl_roi_surface_plot_item:
        if self.gl_roi_view_widget:
            self.gl_roi_view_widget.removeItem(self.gl_roi_surface_plot_item)
        self.gl_roi_surface_plot_item = None
    
    if hasattr(self, 'gl_gauss_surface_plot_item') and self.gl_gauss_surface_plot_item:
        if self.gl_gauss_view_widget:
            self.gl_gauss_view_widget.removeItem(self.gl_gauss_surface_plot_item)
        self.gl_gauss_surface_plot_item = None
    
    # Usuwanie widgetów OpenGL
    if hasattr(self, 'gl_roi_view_widget'):
        self.gl_roi_view_widget.deleteLater()
        self.gl_roi_view_widget = None
    
    if hasattr(self, 'gl_gauss_view_widget'):
        self.gl_gauss_view_widget.deleteLater()
        self.gl_gauss_view_widget = None
    
    super().closeEvent(event)